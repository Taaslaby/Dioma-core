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
# train_mnist_eye.py — «Глаз-классификатор»: видит пиксельную картинку и
# ГОВОРИТ, что видит (10 классов MNIST).
#
# Второй орган пайплайна «Python учит ↔ Rust играет» (первый — train_eye.py,
# дистилляция геометрии ретины; здесь — КЛАССИФИКАЦИЯ):
#
#   1. ЭТОТ скрипт: качает MNIST (кэш tools/data/mnist), учит MLP
#      784 → 64 (tanh) → 10 (логиты), экспортирует:
#        tools/mnist.dbw         — запечённые веса (формат DBW1, BakedMlp)
#        tools/mnist_golden.json — золото паритета (5 сэмплов, f64)
#        tools/mnist_holdout.json— hold-out (1000 сбалансированных, 100/класс)
#        tools/mnist_report.txt  — метрики (accuracy, матрица ошибок)
#   2. examples/mnist_probe.rs — паритет Python↔Rust + точность + «вижу N».
#
# Модель — ровно BakedMlp (baked.rs): h = tanh(x·W0ᵀ+b0), y = h·W1ᵀ+b1.
# Обучение: torch (CPU, детерминированные сиды), CE по логитам с softmax —
# в инференсе argmax(logits) == argmax(softmax), поэтому запечённые логиты
# корректны и для классификации, и для «уверенности».
#
# torch в этом репо уже используется (--torch-gru, Шептун). numpy тоже.
# ---------------------------------------------------------------------------

import argparse
import gzip
import json
import os
import struct
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_dbw import export_dbw  # тот же DBW1-писатель, что и весь пайплайн

# ── Константы ───────────────────────────────────────────────────────────────
IN_DIM = 784      # 28×28 пикселей, [0,1]
HIDDEN = 64       # скрытый слой (tanh)
OUT_DIM = 10      # классы 0..9 (логиты)
EPOCHS = 8
SEED = 123        # сид инициализации/шаффла (детерминированный прогон)
BATCH = 128
LR = 1e-3
GOLDEN_N = 5      # золотых сэмплов паритета (seed 123, из hold-out/test)
HOLDOUT_PER_CLASS = 100  # hold-out: 1000 сэмплов, сбалансировано

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # зеркало torchvision
DATA_DIR = os.path.join(HERE, "data", "mnist")
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

# ── Загрузка/разбор MNIST (idx-формат, кэш) ────────────────────────────────

def ensure_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            continue
        url = MNIST_URL + name
        print(f"  качаю {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "dioma-mnist-eye/1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = r.read()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)
    print(f"  MNIST в кэше: {DATA_DIR}")


def parse_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        buf = f.read(n * rows * cols)
    x = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return x.astype(np.float64) / 255.0, rows, cols


def parse_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        y = np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int64)
    return y


# ── Модель: ТОЧНО forward BakedMlp ──────────────────────────────────────────

class Eye(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(IN_DIM, HIDDEN)
        self.fc1 = nn.Linear(HIDDEN, OUT_DIM)

    def forward(self, x):
        h = torch.tanh(self.fc0(x))
        return self.fc1(h)  # логиты (linear — как BakedMlp)


def numpy_forward(w0, b0, w1, b1, x):
    """f64 forward, побитово повторяющий BakedMlp::forward (baked.rs)."""
    h = np.tanh(x @ w0.T + b0)
    return h @ w1.T + b1


# ── Экспорт ─────────────────────────────────────────────────────────────────

def model_tensors(model):
    def flat(t):
        return [float(v) for v in t.detach().numpy().reshape(-1)]

    # nn.Linear хранит weight [out, in] — ровно как layer0.w [hidden, in] у BakedMlp.
    return [
        ("layer0.w", (HIDDEN, IN_DIM), flat(model.fc0.weight)),
        ("layer0.b", (HIDDEN,), flat(model.fc0.bias)),
        ("layer1.w", (OUT_DIM, HIDDEN), flat(model.fc1.weight)),
        ("layer1.b", (OUT_DIM,), flat(model.fc1.bias)),
    ]


def confusion(y_true, y_pred):
    m = np.zeros((OUT_DIM, OUT_DIM), dtype=np.int64)  # [истина][предсказание]
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        prog="train_mnist_eye.py",
        description="Глаз-классификатор: MLP 784→64(tanh)→10 на MNIST, "
                    "экспорт DBW1 + золото + hold-out.",
        epilog="Пример: python tools/train_mnist_eye.py --epochs 8",
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=os.path.join(HERE, "mnist.dbw"))
    args = ap.parse_args()

    print("MNIST: подготовка данных")
    ensure_mnist()
    xtr, rows, cols = parse_idx_images(os.path.join(DATA_DIR, FILES[0]))
    ytr = parse_idx_labels(os.path.join(DATA_DIR, FILES[1]))
    xte, _, _ = parse_idx_images(os.path.join(DATA_DIR, FILES[2]))
    yte = parse_idx_labels(os.path.join(DATA_DIR, FILES[3]))
    assert xtr.shape == (60000, 784) and xte.shape == (10000, 784), (xtr.shape, xte.shape)
    print(f"  train {xtr.shape[0]}, hold-out(test) {xte.shape[0]}, {rows}×{cols}, x ∈ [0,1]")

    torch.manual_seed(args.seed)
    model = Eye()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xs = torch.tensor(xtr, dtype=torch.float32)
    ys = torch.tensor(ytr, dtype=torch.long)
    n = xs.shape[0]

    print(f"\nобучение: MLP {IN_DIM}->{HIDDEN}(tanh)->{OUT_DIM}(логиты), CE, Adam lr={LR}, "
          f"батч {BATCH}, эпох {args.epochs}, сид {args.seed}, CPU")
    g = torch.Generator().manual_seed(args.seed + 1)  # отдельный поток шаффла
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        nb = 0
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            logits = model(xs[b])
            loss = torch.nn.functional.cross_entropy(logits, ys[b])
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                acc = float((model(xs).argmax(1) == ys).float().mean())
            print(f"  эпоха {epoch}  loss = {total / nb:.4f}  train acc = {acc:.4f}")

    # ── Тест (hold-out): accuracy + матрица ошибок ──
    model.eval()
    with torch.no_grad():
        logits_te = model(torch.tensor(xte, dtype=torch.float32)).numpy()
    pred_te = logits_te.argmax(1)
    acc_te = float((pred_te == yte).mean())
    cm = confusion(yte, pred_te)
    per_class = [float(cm[k, k] / max(cm[k].sum(), 1)) for k in range(OUT_DIM)]

    print(f"\nHOLD-OUT (test 10k): accuracy = {acc_te:.4f}")
    print("матрица ошибок [истина][предсказание]:")
    header = "      " + "".join(f"{d:>6}" for d in range(OUT_DIM))
    print(header)
    for k in range(OUT_DIM):
        print(f"  {k} |" + "".join(f"{cm[k, d]:>6}" for d in range(OUT_DIM)) + f"   ({per_class[k]:.3f})")

    # ── Экспорт DBW1 ──
    tensors = model_tensors(model)
    nbytes = export_dbw(tensors, args.out)
    n_params = sum(len(t[2]) for t in tensors)
    print(f"\nэкспортировано: {args.out} ({nbytes} байт, {n_params} весов, тензоров {len(tensors)})")

    # ── Золото паритета: 5 сэмплов (seed 123) из test, f64-выходы python ──
    w0 = np.asarray(tensors[0][2], dtype=np.float64).reshape(HIDDEN, IN_DIM)
    b0 = np.asarray(tensors[1][2], dtype=np.float64)
    w1 = np.asarray(tensors[2][2], dtype=np.float64).reshape(OUT_DIM, HIDDEN)
    b1 = np.asarray(tensors[3][2], dtype=np.float64)

    rng = np.random.RandomState(GOLDEN_N * 24 + 3)  # фиксированный выбор золота
    gold_idx = []
    while len(gold_idx) < GOLDEN_N:
        i = int(rng.randint(len(xte)))
        if i not in gold_idx:
            gold_idx.append(i)
    samples = []
    for i in gold_idx:
        x = xte[i]
        y = numpy_forward(w0, b0, w1, b1, x[None, :])
        samples.append({
            "label": int(yte[i]),
            "index": i,
            "input": [float(v) for v in x],
            "output": [float(v) for v in y[0]],
        })
    golden = {
        "version": "mnist_golden_v1",
        "in_dim": IN_DIM,
        "out_dim": OUT_DIM,
        "note": "5 сэмплов test MNIST (seed 528): input — пиксели [0,1], "
                "output — python-выход MLP f64 (torch-веса, приведённые к f32-значениям); "
                "паритет с Rust f32 — допуск 1e-4",
        "samples": samples,
    }
    golden_path = os.path.join(HERE, "mnist_golden.json")
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=1)
    print(f"золото паритета: {len(samples)} сэмплов -> {golden_path}")

    # ── Hold-out для Rust: 1000 сбалансированных (100/класс, seed 42) ──
    rng_h = np.random.RandomState(42)
    hold_idx = []
    for k in range(OUT_DIM):
        pool = np.flatnonzero(yte == k)
        take = rng_h.choice(pool, size=HOLDOUT_PER_CLASS, replace=False)
        hold_idx.extend(int(i) for i in take)
    hold = {
        "version": "mnist_holdout_v1",
        "in_dim": IN_DIM,
        "out_dim": OUT_DIM,
        "per_class": HOLDOUT_PER_CLASS,
        "note": "hold-out для mnist_probe.rs: сбалансированная выборка test MNIST "
                "(вход округлён до 5 знаков — паритет меряется по золоту, не здесь)",
        "samples": [
            {"label": int(yte[i]), "index": int(i),
             "input": [round(float(v), 5) for v in xte[i]]}
            for i in hold_idx
        ],
    }
    hold_path = os.path.join(HERE, "mnist_holdout.json")
    with open(hold_path, "w", encoding="utf-8") as f:
        json.dump(hold, f)
    print(f"hold-out: {len(hold_idx)} сэмплов ({HOLDOUT_PER_CLASS}/класс) -> {hold_path} "
          f"({os.path.getsize(hold_path) // 1024} КиБ)")

    # ── Отчёт ──
    report_path = os.path.join(HERE, "mnist_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ГЛАЗ-КЛАССИФИКАТОР (mnist_eye): пиксели → «вижу цифру»\n")
        f.write("=" * 78 + "\n")
        f.write(f"данные: MNIST (torch-зеркало ossci S3), train 60k / test 10k, {rows}×{cols}, [0,1]\n")
        f.write(f"модель: MLP {IN_DIM}->{HIDDEN}(tanh)->{OUT_DIM}(логиты), CE, Adam lr={LR}, "
                f"батч {BATCH}, эпох {args.epochs}, сид {args.seed}, torch CPU\n")
        f.write(f"веса: {n_params} (layer0.w/b {HIDDEN}×{IN_DIM}+{HIDDEN}, "
                f"layer1.w/b {OUT_DIM}×{HIDDEN}+{OUT_DIM})\n\n")
        f.write(f"HOLD-OUT accuracy: {acc_te:.4f}  (10 000 test)\n")
        f.write("per-class recall: " + " ".join(f"{k}:{per_class[k]:.3f}" for k in range(OUT_DIM)) + "\n\n")
        f.write("матрица ошибок [истина][предсказание]:\n")
        f.write(header + "\n")
        for k in range(OUT_DIM):
            f.write(f"  {k} |" + "".join(f"{cm[k, d]:>6}" for d in range(OUT_DIM)) + f"   ({per_class[k]:.3f})\n")
        f.write(f"\nэкспорт: {args.out} ({nbytes} байт)\n")
        f.write(f"золото: {golden_path}; hold-out для rust: {hold_path}\n")
        f.write("ожидание: accuracy ≥ 0.95 на hold-out (MLP 64 hidden — консервативно; "
                "rust-пробник меряет то же на 1000 сбалансированных)\n")
    print(f"отчёт -> {report_path}")


if __name__ == "__main__":
    main()
