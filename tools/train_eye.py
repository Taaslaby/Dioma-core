#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Консоль Windows часто в cp1251: юникод-стрелки ломают print — включаем utf-8.
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
# ---------------------------------------------------------------------------
# train_eye.py — «Глаза»: ANN-энкодер «сырая сетчатка → аналитические сигналы»
#
# Пайплайн (этап 0):
#   1. examples/eye_data.rs          → tools/eye_data.json  (рекордер ретины)
#   2. ЭТОТ скрипт                   → tools/eye.dbw        (запечённые веса)
#                                   → tools/eye_golden.json (золотые выходы)
#                                   → tools/eye_report.txt  (метрики)
#   3. examples/eye_check.rs         → паритет Python↔Rust + качество hold-out
#
# Модель: полносвязный MLP 48 → 24 (tanh) → 6 (линейный). Обучение — ручной
# numpy-forward + backprop, MSE, Adam, батчи 128, 40 эпох, сид 123
# (детерминированный прогон). Выходы — ровно аналитические сигналы глаза:
#   vis_l, vis_r, food_bearing_sin, food_bearing_cos, food_dist_norm, wall_ahead
#
# Экспорт — СТРОГО формат DBW1 из tools/export_dbw.py: тензоры
# layer0.w [24,48] / layer0.b [24] / layer1.w [6,24] / layer1.b [6] — ровно
# то, что ждёт BakedMlp (crates/dioma-engine/src/engine/baked.rs).
#
# Метрики (train и hold-out, per-output): MAE + Пирсон r. Считаются ДВАЖДЫ:
# по ВСЕМ сэмплам и по «бодрым» (ретина ненулевая). Почему дважды: случайный
# мозг спит ~75% тиков (усталость) — ретина и цели там нулевые («глаза
# закрыты»), сеть обязана выучить «закрытые глаза → нет сигнала»; но честная
# мера геометрии — на бодрых сэмплах. Оба числа уходят в отчёт.
#
# numpy обязателен (проверен: python -c "import numpy" — установлен).
# ---------------------------------------------------------------------------
import argparse
import json
import math
import os
import sys

import numpy as np

# ── Константы модели и пайплайна ───────────────────────────────────────────
IN_DIM = 48       # ретина: 16 лучей × [ягодность, стенность, соседность]
HIDDEN = 24       # скрытый слой (tanh)
OUT_DIM = 6       # vis_l, vis_r, bearing_sin, bearing_cos, dist_norm, wall_ahead
EPOCHS = 40       # эпох (спека)
SEED = 123        # сид инициализации (детерминированный прогон)
SEED0 = 2000      # базовый сид рекордера (совпадает с --seed0 рекордера)
TRAIN_N = 6       # особи seed0+0..5 — обучение
HOLDOUT_N = 2     # особи seed0+6..7 — hold-out
LR = 0.01         # Adam learning rate
BATCH = 128       # размер батча
GOLDEN_N = 5      # золотых сэмплов паритета (seed 123)

OUT_NAMES = ["vis_l", "vis_r", "food_bearing_sin", "food_bearing_cos", "food_dist_norm", "wall_ahead"]

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_dbw import export_dbw  # тот же DBW1-писатель, что и весь пайплайн


# ── Инициализация (Xavier-ish, детерминированная: RandomState(SEED)) ───────
def init_params(rng):
    def mat(rows, cols, k):
        return rng.uniform(-k, k, size=(rows, cols)).astype(np.float64)

    k0 = 1.0 / math.sqrt(IN_DIM)
    k1 = 1.0 / math.sqrt(HIDDEN)
    return {
        "w0": mat(HIDDEN, IN_DIM, k0),
        "b0": np.zeros(HIDDEN),
        "w1": mat(OUT_DIM, HIDDEN, k1),
        "b1": np.zeros(OUT_DIM),
    }


# ── Forward: ТОЧНО как BakedMlp::forward (baked.rs) ────────────────────────
#   h = tanh(x·W0ᵀ + b0)    y = h·W1ᵀ + b1   (выход ЛИНЕЙНЫЙ — как в BakedMlp)
def forward(p, x):
    h = np.tanh(x @ p["w0"].T + p["b0"])
    y = h @ p["w1"].T + p["b1"]
    return y, h


# ── Adam (как в whisperer) ─────────────────────────────────────────────────
def adam_step(p, grads, m, v, tstep):
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in p:
        m[k] = b1 * m[k] + (1.0 - b1) * grads[k]
        v[k] = b2 * v[k] + (1.0 - b2) * grads[k] ** 2
        mh = m[k] / (1.0 - b1 ** tstep)
        vh = v[k] / (1.0 - b2 ** tstep)
        p[k] -= LR * mh / (np.sqrt(vh) + eps)


def train_batch(p, x, t, m, v, tstep):
    """Один батч: forward + backprop (MSE), шаг Adam. Возвращает loss."""
    b = x.shape[0]
    y, h = forward(p, x)
    loss = float(np.mean((y - t) ** 2))
    dy = 2.0 * (y - t) / (b * OUT_DIM)  # d(mean((y-t)²))/dy
    dz = (dy @ p["w1"]) * (1.0 - h ** 2)  # через tanh': 1 − h²
    grads = {
        "w1": dy.T @ h,
        "b1": dy.sum(axis=0),
        "w0": dz.T @ x,
        "b0": dz.sum(axis=0),
    }
    adam_step(p, grads, m, v, tstep)
    return loss


# ── Метрики ────────────────────────────────────────────────────────────────
def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(p, creatures, awake_only=False):
    """Per-output метрики, пул по сэмплам. awake_only — только сэмплы с
    ненулевой ретиной («глаза открыты»). Возвращает (строки, метрики)."""
    preds = [[] for _ in range(OUT_DIM)]
    trues = [[] for _ in range(OUT_DIM)]
    n = 0
    for cre in creatures:
        for sample in cre["samples"]:
            x = np.asarray(sample[0], dtype=np.float64)
            if awake_only and not np.any(x):
                continue
            t = np.asarray(sample[1], dtype=np.float64)
            y, _ = forward(p, x[None, :])
            for k in range(OUT_DIM):
                preds[k].append(float(y[0, k]))
                trues[k].append(float(t[k]))
            n += 1
    rows = []
    metrics = {"n": n}
    for k in range(OUT_DIM):
        pv = np.asarray(preds[k])
        tv = np.asarray(trues[k])
        mae = float(np.mean(np.abs(pv - tv)))
        r = pearson(pv, tv)
        metrics[OUT_NAMES[k]] = {"mae": mae, "r": r}
        rows.append(f"  {OUT_NAMES[k]:<16}  MAE={mae:.4f}  r={r:+.4f}")
    return rows, metrics


# ── Экспорт в DBW1 (тензоры BakedMlp) ──────────────────────────────────────
def model_tensors(p):
    def flat(a):
        return [float(v) for v in np.asarray(a, dtype=np.float64).reshape(-1)]

    return [
        ("layer0.w", (HIDDEN, IN_DIM), flat(p["w0"])),
        ("layer0.b", (HIDDEN,), flat(p["b0"])),
        ("layer1.w", (OUT_DIM, HIDDEN), flat(p["w1"])),
        ("layer1.b", (OUT_DIM,), flat(p["b1"])),
    ]


def dump_golden(p, creatures, seed0):
    """5 случайных (seed 123) сэмплов из датасета: вход + python-выходы f64 —
    эталон для eye_check.rs (паритет python↔rust). Берутся ТОЛЬКО сэмплы с
    ненулевой ретиной («глаза открыты»): нулевой вход гонял бы через сеть
    только смещения и не тестировал бы нелинейный режим forward."""
    rng = np.random.RandomState(123)
    samples = []
    attempts = 0
    while len(samples) < GOLDEN_N:
        attempts += 1
        ci = int(rng.randint(len(creatures)))
        cre = creatures[ci]
        si = int(rng.randint(len(cre["samples"])))
        x = np.asarray(cre["samples"][si][0], dtype=np.float64)
        if not np.any(x):
            continue
        y, _ = forward(p, x[None, :])
        samples.append({
            "seed": cre["seed"],
            "start": si,
            "input": [float(v) for v in x],
            "output": [float(v) for v in y[0]],
        })
    golden = {
        "version": "eye_golden_v1",
        "in_dim": IN_DIM,
        "out_dim": OUT_DIM,
        "note": f"5 случайных сэмплов (seed 123, только ненулевая ретина; {attempts} попыток) из tools/eye_data.json; input — ретина, output — python-выход MLP (f64)",
        "seed0": seed0,
        "samples": samples,
    }
    path = os.path.join(HERE, "eye_golden.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=1)
    print(f"золото паритета: {len(samples)} сэмплов -> {path}")
    return path


# ── CLI / main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        prog="train_eye.py",
        description="Глаза: ANN-энкодер ретины (48 -> 24 tanh -> 6 линейный), "
                    "дистилляция аналитических сигналов.",
        epilog="Пример: python tools/train_eye.py --data tools/eye_data.json --epochs 40",
    )
    ap.add_argument("--data", default=os.path.join(HERE, "eye_data.json"),
                    help="датасет рекордера (по умолчанию tools/eye_data.json)")
    ap.add_argument("--out", default=os.path.join(HERE, "eye.dbw"),
                    help="выходной DBW1 (по умолчанию tools/eye.dbw)")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"число эпох (по умолчанию {EPOCHS})")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"сид инициализации (по умолчанию {SEED})")
    ap.add_argument("--seed0", type=int, default=SEED0,
                    help=f"базовый сид рекордера (по умолчанию {SEED0})")
    args = ap.parse_args()

    print(f"загрузка датасета: {args.data}")
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == "eye_data_v1", data.get("version")
    assert data["in_dim"] == IN_DIM and data["out_dim"] == OUT_DIM, (data["in_dim"], data["out_dim"])
    creatures = data["creatures"]
    assert [c["seed"] for c in creatures] == [args.seed0 + i for i in range(len(creatures))], (
        "сиды датасета должны идти подряд с базового сида"
    )
    train = [c for c in creatures if c["seed"] - args.seed0 < TRAIN_N]
    holdout = [c for c in creatures if c["seed"] - args.seed0 >= TRAIN_N]
    print(
        f"особей: всего {len(creatures)} | train {len(train)} "
        f"({[c['seed'] for c in train]}) | hold-out {len(holdout)} "
        f"({[c['seed'] for c in holdout]})"
    )

    rng = np.random.RandomState(args.seed)
    p = init_params(rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(kv) for k, kv in p.items()}

    xs_all = np.asarray([r[0] for c in train for r in c["samples"]], dtype=np.float64)
    ts_all = np.asarray([r[1] for c in train for r in c["samples"]], dtype=np.float64)
    n_all = len(xs_all)
    idx = np.arange(n_all)
    shuf = np.random.RandomState(args.seed + 1)  # отдельный поток — порядок батчей

    print(
        f"MLP {IN_DIM}->{HIDDEN}(tanh)->{OUT_DIM}(линейный), MSE, Adam lr={LR}, "
        f"батч {BATCH}, сид {args.seed}, эпох {args.epochs}; сэмплов train {n_all}"
    )
    tstep = 0
    for epoch in range(args.epochs):
        shuf.shuffle(idx)
        ep_loss = 0.0
        n_b = 0
        for s in range(0, n_all, BATCH):
            b = idx[s:s + BATCH]
            tstep += 1
            ep_loss += train_batch(p, xs_all[b], ts_all[b], m, v, tstep)
            n_b += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  эпоха {epoch:3d}  loss = {ep_loss / n_b:.6f}  (батчей {n_b})")

    # ── Метрики: train + hold-out, ВСЕ сэмплы и только «бодрые» ──
    print("\nметрики per-output (пул по сэмплам):")
    lines_all = []
    agg = {}
    for split_name, cre_list in (("train", train), ("holdout", holdout)):
        for awake_name, awake_only in (("все сэмплы", False), ("бодрые", True)):
            rows, mm = evaluate(p, cre_list, awake_only=awake_only)
            lines_all.append(f"  [{split_name} / {awake_name}]  n={mm['n']}")
            lines_all.extend(rows)
            if split_name == "holdout" and not awake_only:
                agg = mm
    for line in lines_all:
        print(line)

    # ── Экспорт DBW1 + золото ──
    tensors = model_tensors(p)
    nbytes = export_dbw(tensors, args.out)
    n_params = sum(t[2].__len__() for t in tensors)
    print(f"\nэкспортировано тензоров: {len(tensors)} (layer0.w/b, layer1.w/b) -> {args.out} ({nbytes} байт, {n_params} весов)")
    golden_path = dump_golden(p, creatures, args.seed0)

    # ── Отчёт ──
    report_path = os.path.join(HERE, "eye_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ГЛАЗА (eye): прототип ANN-энкодера ретины — дистилляция аналитических сигналов\n")
        f.write("=" * 78 + "\n")
        f.write(f"данные: {args.data} (верс. {data['version']}, ретина {data['retina_rays']} лучей, "
                f"in {data['in_dim']}, out {data['out_dim']})\n")
        f.write(f"модель: MLP {IN_DIM}->{HIDDEN}(tanh)->{OUT_DIM}(линейный), MSE, Adam lr={LR}, "
                f"батч {BATCH}, эпох {args.epochs}, сид {args.seed}\n")
        f.write(f"входы: {', '.join(data['in_names'])}\n")
        f.write(f"выходы: {', '.join(OUT_NAMES)}\n")
        f.write(f"разбиение: train = {[c['seed'] for c in train]}"
                f", hold-out = {[c['seed'] for c in holdout]}\n")
        f.write(f"статистика рекордера: {json.dumps(data.get('stats', {}), ensure_ascii=False)}\n\n")
        f.write("МЕТРИКИ (python, пул по сэмплам):\n")
        for line in lines_all:
            f.write(line + "\n")
        f.write("\n")
        f.write("ОЖИДАНИЕ ТЗ: r(vis_l/vis_r/food_*) > 0.9 на hold-out -> факт в таблице выше "
                f"(hold-out / все сэмплы: r(vis_l)={agg.get('vis_l', {}).get('r', 0):+.4f}, "
                f"r(vis_r)={agg.get('vis_r', {}).get('r', 0):+.4f}, "
                f"r(bearing_sin)={agg.get('food_bearing_sin', {}).get('r', 0):+.4f}, "
                f"r(bearing_cos)={agg.get('food_bearing_cos', {}).get('r', 0):+.4f})\n")
        f.write(f"экспорт: {args.out} ({nbytes} байт, {n_params} весов, тензоров {len(tensors)})\n")
        f.write(f"золото паритета: {golden_path} (сэмплов {GOLDEN_N})\n")
    print(f"\nотчёт -> {report_path}")


if __name__ == "__main__":
    main()
