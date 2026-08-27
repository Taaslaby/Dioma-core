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
# train_whisperer.py — «Шептун»: GRU, который учится ПРЕДСКАЗЫВАТЬ действия
# спайковых мозгов существ по их сенсорам (мировая модель мозга).
#
# Пайплайн (этап 0):
#   1. examples/whisperer_data.rs  → tools/whisperer_data.json  (рекордер)
#   2. ЭТОТ скрипт                 → tools/whisperer.dbw        (запечённые веса)
#                                 → tools/whisperer_golden.json (золотые выходы)
#                                 → tools/whisperer_report.txt (метрики)
#   3. examples/whisperer_check.rs → паритет Python↔Rust + качество на hold-out
#
# Модель: GRU(12 → 32) + линейный readout(32 → 3). Выходы: steer = tanh,
# drive = sigmoid, eat = sigmoid. Обучение: ручной numpy-forward + BPTT
# (truncated T=16, скрытое состояние переносится между окнами внутри особи),
# MSE, Adam. Сид фиксирован (np.random.seed(123)) — прогон детерминирован.
#
# Экспорт — СТРОГО формат DBW1 из tools/export_dbw.py (тот же код): тензоры
# gru.wz/wr/wh [32,12], gru.uz/ur/uh [32,32], gru.bz/br/bh [32] — ровно то,
# что ждёт BakedGru (crates/dioma-engine/src/engine/baked.rs), плюс readout
# layer0.w [3,32] / layer0.b [3] — ровно то, что ждёт BakedMlp.
#
# Метрики (train и hold-out): MAE(steer), MAE(drive), точность eat,
# Пирсон r(steer), r(drive). Цель «понял мозги»: hold-out r(steer) > 0.5
# (реальное значение честно печатается в отчёт).
#
# numpy обязателен (проверен: python -c "import numpy" — установлен).
# Если numpy нет: python -m pip install numpy (в этом окружении уже стоит).
# ---------------------------------------------------------------------------
import argparse
import json
import math
import os
import sys

import numpy as np

# ── Константы модели и пайплайна ───────────────────────────────────────────
IN_DIM = 12       # 12 сенсорных входов (порядок — input_names из датасета)
HIDDEN = 32       # скрытое состояние GRU
OUT_DIM = 3       # steer (tanh), drive (sigmoid), eat (sigmoid)
T = 16            # длина окна BPTT (truncation)
EPOCHS = 30       # проходов по train (20..40 по ТЗ)
SEED = 123        # фиксированный сид — детерминированный прогон
SEED0 = 1000      # базовый сид рекордера (совпадает с --seed0 рекордера)
TRAIN_N = 6       # особи seed0+0..5 — обучение
HOLDOUT_N = 2     # особи seed0+6..7 — hold-out
LR = 0.01         # Adam learning rate
CLIP = 5.0        # глобальная норма градиента (клэмп)

# Окна паритета: 5 фиксированных окон (seed, start) — python-золото сверяет
# с Rust (whisperer_check.rs) через tools/whisperer_golden.json.
GOLDEN_WINDOWS = [
    (SEED0 + 6, 0),
    (SEED0 + 6, 500),
    (SEED0 + 6, 1000),
    (SEED0 + 7, 0),
    (SEED0 + 7, 1200),
]

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_dbw import export_dbw  # тот же DBW1-писатель, что и весь пайплайн


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ── Инициализация (Xavier-ish, детерминированная: np.random.seed(SEED)) ────
def init_params(rng):
    def mat(rows, cols, k):
        return rng.uniform(-k, k, size=(rows, cols)).astype(np.float64)

    k_in = 1.0 / math.sqrt(IN_DIM)
    k_h = 1.0 / math.sqrt(HIDDEN)
    p = {
        "wz": mat(HIDDEN, IN_DIM, k_in),
        "wr": mat(HIDDEN, IN_DIM, k_in),
        "wh": mat(HIDDEN, IN_DIM, k_in),
        "uz": mat(HIDDEN, HIDDEN, k_h),
        "ur": mat(HIDDEN, HIDDEN, k_h),
        "uh": mat(HIDDEN, HIDDEN, k_h),
        "bz": np.zeros(HIDDEN),
        "br": np.zeros(HIDDEN),
        "bh": np.zeros(HIDDEN),
        "wo": mat(OUT_DIM, HIDDEN, k_h),
        "bo": np.zeros(OUT_DIM),
    }
    return p


# ── GRU-шаг: ТОЧНО как BakedGru::step (baked.rs), потом readout ────────────
#   z = σ(x·Wzᵀ + h·Uzᵀ + bz)    r = σ(x·Wrᵀ + h·Urᵀ + br)
#   n = tanh(x·Whᵀ + (r⊙h)·Uhᵀ + bh)      h' = (1−z)⊙n + z⊙h
#   y = h'·Woᵀ + bo;  pred = [tanh(y0), σ(y1), σ(y2)]
def gru_step(p, x, h):
    z = sigmoid(x @ p["wz"].T + h @ p["uz"].T + p["bz"])
    r = sigmoid(x @ p["wr"].T + h @ p["ur"].T + p["br"])
    n = np.tanh(x @ p["wh"].T + (r * h) @ p["uh"].T + p["bh"])
    h2 = (1.0 - z) * n + z * h
    y = h2 @ p["wo"].T + p["bo"]
    pred = np.array([np.tanh(y[0]), sigmoid(y[1]), sigmoid(y[2])])
    return h2, z, r, n, pred


def forward_hidden(p, x, h):
    return gru_step(p, x, h)[0]


def forward_readout(p, h):
    y = h @ p["wo"].T + p["bo"]
    return np.array([np.tanh(y[0]), sigmoid(y[1]), sigmoid(y[2])])


# ── BPTT одного окна: градиенты по всем параметрам, возврат h-переноса ────
def train_window(p, xs, targets, h_in):
    """xs: (T, 12), targets: (T, 3). Возвращает (loss, grads, h_out)."""
    Tn = xs.shape[0]
    h_seq = np.zeros((Tn + 1, HIDDEN))
    h_seq[0] = h_in
    zs = np.zeros((Tn, HIDDEN))
    rs = np.zeros((Tn, HIDDEN))
    ns = np.zeros((Tn, HIDDEN))
    preds = np.zeros((Tn, OUT_DIM))
    for t in range(Tn):
        h_seq[t + 1], zs[t], rs[t], ns[t], preds[t] = gru_step(p, xs[t], h_seq[t])

    loss = float(np.mean((preds - targets) ** 2))

    grads = {k: np.zeros_like(v) for k, v in p.items()}
    scale = 2.0 / (Tn * OUT_DIM)  # d(mean(²))/d(pred)
    dh_next = np.zeros(HIDDEN)
    for t in range(Tn - 1, -1, -1):
        # ── readout: pred → y → h_after ──
        dy = (preds[t] - targets[t]) * scale
        d_act = np.array([
            dy[0] * (1.0 - preds[t][0] ** 2),                    # tanh'
            dy[1] * preds[t][1] * (1.0 - preds[t][1]),           # σ'
            dy[2] * preds[t][2] * (1.0 - preds[t][2]),
        ])
        grads["wo"] += np.outer(d_act, h_seq[t + 1])
        grads["bo"] += d_act
        g = dh_next + d_act @ p["wo"]  # dL/dh_after

        # ── h' = (1−z)⊙n + z⊙h ──
        h_t = h_seq[t]
        z, r, n = zs[t], rs[t], ns[t]
        dz = g * (n - h_t)
        dn = g * (1.0 - z)
        dh_t = g * z

        # ── n = tanh(x·Whᵀ + (r⊙h)·Uhᵀ + bh) ──
        dpre_n = dn * (1.0 - n ** 2)
        rh = r * h_t
        grads["wh"] += np.outer(dpre_n, xs[t])
        grads["uh"] += np.outer(dpre_n, rh)
        grads["bh"] += dpre_n
        t_uh = dpre_n @ p["uh"]            # [hidden]
        dh_t += r * t_uh                   # ∂/∂h через (r⊙h)@Uhᵀ
        dr = h_t * t_uh                    # ∂/∂r через (r⊙h)@Uhᵀ
        dpre_r = dr * r * (1.0 - r)
        grads["wr"] += np.outer(dpre_r, xs[t])
        grads["ur"] += np.outer(dpre_r, h_t)
        grads["br"] += dpre_r
        dh_t += dpre_r @ p["ur"]

        # ── z = σ(x·Wzᵀ + h·Uzᵀ + bz) ──
        dpre_z = dz * z * (1.0 - z)
        grads["wz"] += np.outer(dpre_z, xs[t])
        grads["uz"] += np.outer(dpre_z, h_t)
        grads["bz"] += dpre_z
        dh_t += dpre_z @ p["uz"]

        dh_next = dh_t
    return loss, grads, h_seq[Tn]


# ── Adam ────────────────────────────────────────────────────────────────────
def adam_step(p, grads, m, v, tstep):
    # Клэмп глобальной нормы градиента.
    norm = math.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
    if norm > CLIP:
        for k in grads:
            grads[k] *= CLIP / norm
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in p:
        m[k] = b1 * m[k] + (1.0 - b1) * grads[k]
        v[k] = b2 * v[k] + (1.0 - b2) * grads[k] ** 2
        mh = m[k] / (1.0 - b1 ** tstep)
        vh = v[k] / (1.0 - b2 ** tstep)
        p[k] -= LR * mh / (np.sqrt(vh) + eps)


# ── Метрики ────────────────────────────────────────────────────────────────
def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(p, creatures, names):
    """Состоятельный прогон: входы кормятся подряд (h переносится), сравниваем
    с записанными действиями. Возвращает строки отчёта + диктовку метрик."""
    report = []
    metrics = {}
    for name, cre in zip(names, creatures):
        h = np.zeros(HIDDEN)
        pred_steer, true_steer = [], []
        pred_drive, true_drive = [], []
        pred_eat, true_eat = [], []
        for row in cre["samples"]:
            x = np.asarray(row[:IN_DIM], dtype=np.float64)
            t = np.asarray(row[IN_DIM:IN_DIM + OUT_DIM], dtype=np.float64)
            h = forward_hidden(p, x, h)
            pred = forward_readout(p, h)
            pred_steer.append(pred[0])
            pred_drive.append(pred[1])
            pred_eat.append(pred[2] > 0.5)
            true_steer.append(t[0])
            true_drive.append(t[1])
            true_eat.append(t[2] > 0.5)
        n = len(true_steer)
        mae_s = float(np.mean(np.abs(np.asarray(pred_steer) - np.asarray(true_steer))))
        mae_d = float(np.mean(np.abs(np.asarray(pred_drive) - np.asarray(true_drive))))
        acc_e = float(np.mean(np.asarray(pred_eat) == np.asarray(true_eat)))
        r_s = pearson(pred_steer, true_steer)
        r_d = pearson(pred_drive, true_drive)
        metrics[name] = {
            "mae_steer": mae_s, "mae_drive": mae_d, "eat_acc": acc_e,
            "r_steer": r_s, "r_drive": r_d, "n": n,
        }
        report.append(
            f"  {name}: n={n:5d}  MAE(steer)={mae_s:.4f}  MAE(drive)={mae_d:.4f}  "
            f"eat_acc={acc_e:.3f}  r(steer)={r_s:+.4f}  r(drive)={r_d:+.4f}"
        )
    return report, metrics


# ── Экспорт в DBW1 (тензоры BakedGru + BakedMlp) ───────────────────────────
def model_tensors(p):
    def flat(m):
        return [float(v) for v in np.asarray(m, dtype=np.float64).reshape(-1)]

    return [
        ("gru.wz", (HIDDEN, IN_DIM), flat(p["wz"])),
        ("gru.wr", (HIDDEN, IN_DIM), flat(p["wr"])),
        ("gru.wh", (HIDDEN, IN_DIM), flat(p["wh"])),
        ("gru.uz", (HIDDEN, HIDDEN), flat(p["uz"])),
        ("gru.ur", (HIDDEN, HIDDEN), flat(p["ur"])),
        ("gru.uh", (HIDDEN, HIDDEN), flat(p["uh"])),
        ("gru.bz", (HIDDEN,), flat(p["bz"])),
        ("gru.br", (HIDDEN,), flat(p["br"])),
        ("gru.bh", (HIDDEN,), flat(p["bh"])),
        ("layer0.w", (OUT_DIM, HIDDEN), flat(p["wo"])),
        ("layer0.b", (OUT_DIM,), flat(p["bo"])),
    ]


def dump_golden(p, by_seed):
    """Золотые выходы для 5 фиксированных окон — эталон для whisperer_check.rs.
    Счёт строго как в Rust: h=0, 16 шагов, выход после КАЖДОГО шага."""
    windows = []
    for seed, start in GOLDEN_WINDOWS:
        rows = by_seed[seed]["samples"][start:start + 16]
        h = np.zeros(HIDDEN)
        outputs = []
        for row in rows:
            x = np.asarray(row[:IN_DIM], dtype=np.float64)
            h = forward_hidden(p, x, h)
            outputs.append([float(v) for v in forward_readout(p, h)])
        windows.append({"seed": seed, "start": start, "outputs": outputs})
    golden = {
        "version": "whisperer_golden_v1",
        "in_dim": IN_DIM,
        "hidden": HIDDEN,
        "window_len": 16,
        "windows": windows,
    }
    path = os.path.join(HERE, "whisperer_golden.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=1)
    print(f"золото паритета: {len(windows)} окон -> {path}")
    return path


# ── CLI / main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        prog="train_whisperer.py",
        description="Шептун: GRU предсказывает действия спайковых мозгов по сенсорам.",
        epilog="Пример: python tools/train_whisperer.py --data tools/whisperer_data.json --epochs 30",
    )
    ap.add_argument("--data", default=os.path.join(HERE, "whisperer_data.json"),
                    help="датасет рекордера (по умолчанию tools/whisperer_data.json)")
    ap.add_argument("--out", default=os.path.join(HERE, "whisperer.dbw"),
                    help="выходной DBW1 (по умолчанию tools/whisperer.dbw)")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"число эпох (по умолчанию {EPOCHS})")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"сид детерминизма (по умолчанию {SEED})")
    args = ap.parse_args()

    print(f"загрузка датасета: {args.data}")
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == "whisperer_data_v1", data.get("version")
    creatures = data["creatures"]
    assert [c["seed"] for c in creatures] == [SEED0 + i for i in range(len(creatures))], (
        "сиды датасета должны идти подряд с базового сида"
    )
    by_seed = {c["seed"]: c for c in creatures}
    train = [c for c in creatures if c["seed"] - SEED0 < TRAIN_N]
    holdout = [c for c in creatures if c["seed"] - SEED0 >= TRAIN_N]
    print(
        f"особей: всего {len(creatures)} | train {len(train)} "
        f"({[c['seed'] for c in train]}) | hold-out {len(holdout)} "
        f"({[c['seed'] for c in holdout]})"
    )

    np.random.seed(args.seed)
    p = init_params(np.random.RandomState(args.seed))
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(kv) for k, kv in p.items()}

    train_xs = [np.asarray([r[:IN_DIM] for r in c["samples"]], dtype=np.float64) for c in train]
    train_ts = [np.asarray([r[IN_DIM:] for r in c["samples"]], dtype=np.float64) for c in train]

    print(
        f"GRU {IN_DIM}→{HIDDEN}→{OUT_DIM} (steer=tanh, drive=σ, eat=σ), "
        f"BPTT T={T}, Adam lr={LR}, сид {args.seed}, эпох {args.epochs}"
    )
    tstep = 0
    for epoch in range(args.epochs):
        ep_loss = 0.0
        n_win = 0
        for ci in range(TRAIN_N):
            h = np.zeros(HIDDEN)
            xs, ts = train_xs[ci], train_ts[ci]
            for w in range(0, len(xs) - T + 1, T):
                loss, grads, h = train_window(p, xs[w:w + T], ts[w:w + T], h)
                tstep += 1
                adam_step(p, grads, m, v, tstep)
                ep_loss += loss
                n_win += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  эпоха {epoch:3d}  loss = {ep_loss / n_win:.6f}  (окон {n_win})")

    # ── Метрики (train + hold-out) ──
    print("\nметрики (состоятельный прогон: входы подряд, h переносится):")
    lines_all = []
    for split_name, cre_list in (("train", train), ("holdout", holdout)):
        _, m_split = evaluate(p, cre_list, [f"{split_name}:c{c['seed']}" for c in cre_list])
        for name, mm in m_split.items():
            lines_all.append(
                f"  {name}: n={mm['n']:5d}  MAE(steer)={mm['mae_steer']:.4f}  "
                f"MAE(drive)={mm['mae_drive']:.4f}  eat_acc={mm['eat_acc']:.3f}  "
                f"r(steer)={mm['r_steer']:+.4f}  r(drive)={mm['r_drive']:+.4f}"
            )
        agg = {
            "n": sum(mm["n"] for mm in m_split.values()),
            "mae_steer": float(np.mean([mm["mae_steer"] for mm in m_split.values()])),
            "mae_drive": float(np.mean([mm["mae_drive"] for mm in m_split.values()])),
            "eat_acc": float(np.mean([mm["eat_acc"] for mm in m_split.values()])),
            "r_steer": float(np.mean([mm["r_steer"] for mm in m_split.values()])),
            "r_drive": float(np.mean([mm["r_drive"] for mm in m_split.values()])),
        }
        lines_all.append(
            f"  {split_name} СВОДКА: n={agg['n']:5d}  MAE(steer)={agg['mae_steer']:.4f}  "
            f"MAE(drive)={agg['mae_drive']:.4f}  eat_acc={agg['eat_acc']:.3f}  "
            f"r(steer)={agg['r_steer']:+.4f}  r(drive)={agg['r_drive']:+.4f}"
        )
    for line in lines_all:
        print(line)
    holdout_summary = {k: v for k, v in agg.items()}

    # ── Экспорт DBW1 + золото ──
    tensors = model_tensors(p)
    nbytes = export_dbw(tensors, args.out)
    print(f"\nэкспортировано тензоров: {len(tensors)} (gru.* + layer0.*) -> {args.out} ({nbytes} байт)")
    golden_path = dump_golden(p, by_seed)

    # ── Отчёт ──
    report_path = os.path.join(HERE, "whisperer_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ШЕПТУН (whisperer): прототип — GRU-мировая модель спайкового мозга\n")
        f.write("=" * 78 + "\n")
        f.write(f"данные: {args.data} (верс. {data['version']})\n")
        f.write(f"модель: GRU {IN_DIM}->{HIDDEN}->{OUT_DIM}, BPTT T={T}, Adam lr={LR}, "
                f"эпох {args.epochs}, сид {args.seed}\n")
        f.write(f"входы: {', '.join(data['input_names'])}\n")
        f.write(f"разбиение: train = {[c['seed'] for c in train]}"
                f", hold-out = {[c['seed'] for c in holdout]}\n\n")
        f.write("МЕТРИКИ (python, состоятельный прогон):\n")
        for line in lines_all:
            f.write(line + "\n")
        f.write("\n")
        f.write(f"цель ТЗ: hold-out r(steer) > 0.5 -> фактически {holdout_summary['r_steer']:+.4f}\n")
        f.write(f"экспорт: {args.out} ({nbytes} байт, тензоров {len(tensors)})\n")
        f.write(f"золото паритета: {golden_path} (окон {len(GOLDEN_WINDOWS)})\n")
    print(f"\nотчёт -> {report_path}")


if __name__ == "__main__":
    main()
