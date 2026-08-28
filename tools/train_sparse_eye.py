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
# train_sparse_eye.py — «КОНЦЕПТ-ГЛАЗ»: пиксели → РАЗРЕЖЕННЫЙ распределённый
# код, как в реальной коре (популяция, активны единицы), а не плоский лейбл.
#
# Чем отличается от train_mnist_eye.py (тот — классификатор «вижу цифру N»,
# один горячий логит): здесь глаз — РАЗРЕЖЕННЫЙ АВТОЭНКОДЕР 784 → 128 → 784,
# где 128-мерный скрытый слой — КОНЦЕПТЫ. Штраф L1 заставляет код быть
# разреженным: на входе каждая цифра возбуждает СВОЮ популяцию из ~5-10%
# нейронов — как спайки V1. Одинаковые цифры → почти одинаковые популяции,
# разные → непересекающиеся.
#
#   h = tanh(x·W0ᵀ + b0)   — 128-мерный разреженный код (КОНЦЕПТ)
#   x̂ = h·W1ᵀ + b1         — реконструкция (доказывает, что код несёт образ)
#
# Это ровно два слоя BakedMlp (baked.rs): layer0 = энкодер, layer1 = декодер.
# В игре: код (128 чисел, ~10 активных) инжектится током в зрительные регионы
# спайкового мозга — активные единицы играют роль спайковых источников.
#
# Экспорт: tools/sparse_eye.dbw + sparse_eye_golden.json + sparse_eye_report.txt.
# Hold-out переиспользует tools/mnist_holdout.json (те же 1000 сэмплов).
# ---------------------------------------------------------------------------

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_dbw import export_dbw
from train_mnist_eye import ensure_mnist, parse_idx_images, parse_idx_labels, DATA_DIR, FILES

# ── Константы ───────────────────────────────────────────────────────────────
IN_DIM = 784      # пиксели 28×28, [0,1]
CODE_DIM = 128    # КОНЦЕПТ-СЛОЙ (разреженная популяция)
HIDDEN = CODE_DIM
EPOCHS = 12
SEED = 123
BATCH = 128
LR = 1e-3
L1_LAMBDA = 0.10      # сила разреживания кода (проверено: ~5-10% активных)
ACTIVE_TH = 0.05      # порог «нейрон активен» (|h| > th)
GOLDEN_N = 5
SPARSITY_GOAL = "5-15% активных нейронов на код"

# ── Модель: автоэнкодер 784 → 128(tanh) → 784(linear) ───────────────────────

class ConceptEye(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(IN_DIM, CODE_DIM)   # энкодер: пиксели → концепты
        self.fc1 = nn.Linear(CODE_DIM, IN_DIM)   # декодер: концепты → пиксели

    def encode(self, x):
        return torch.tanh(self.fc0(x))           # код = ровно BakedMlp h

    def forward(self, x):
        return self.fc1(self.encode(x))          # реконструкция


def numpy_forward(w0, b0, w1, b1, x):
    """f64-копия BakedMlp::forward: возвращает (код h, реконструкция y)."""
    h = np.tanh(x @ w0.T + b0)
    return h, h @ w1.T + b1


# ── Метрики концептов ───────────────────────────────────────────────────────

def sparsity_stats(codes):
    """Доля активных нейронов (|h| > ACTIVE_TH) и их среднее число на код."""
    act = np.abs(codes) > ACTIVE_TH
    frac = act.mean()
    per = act.sum(axis=1)
    return frac, float(per.mean()), per


def class_prototypes(codes, labels, k):
    """Средний код класса (прототип) и матрица косинусной близости 10×10."""
    prot = np.zeros((k, codes.shape[1]))
    for c in range(k):
        prot[c] = codes[labels == c].mean(axis=0)
    n = np.linalg.norm(prot, axis=1, keepdims=True) + 1e-9
    prot_u = prot / n
    return prot, prot_u @ prot_u.T  # (10,10) косинус прототипов


def nearest_centroid_acc(codes, labels, k):
    """Честная мера разделения: КАЖДЫЙ сэмпл классифицируется по ближайшему
    прототипу своего кода (косинус). НЕ путать с самопроверкой прототипов
    (та всегда 1.0) — здесь меряется реальная читаемость кода без лейблов."""
    prot, _ = class_prototypes(codes, labels, k)
    prot_n = prot / (np.linalg.norm(prot, axis=1, keepdims=True) + 1e-9)
    c = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
    sim = c @ prot_n.T          # (n, 10): косинус сэмпл ↔ прототип
    pred = sim.argmax(axis=1)
    return float((pred == labels).mean())


# ── Экспорт / золото / отчёт ────────────────────────────────────────────────

def model_tensors(model):
    def flat(t):
        return [float(v) for v in t.detach().numpy().reshape(-1)]

    return [
        ("layer0.w", (CODE_DIM, IN_DIM), flat(model.fc0.weight)),
        ("layer0.b", (CODE_DIM,), flat(model.fc0.bias)),
        ("layer1.w", (IN_DIM, CODE_DIM), flat(model.fc1.weight)),
        ("layer1.b", (IN_DIM,), flat(model.fc1.bias)),
    ]


def main():
    ap = argparse.ArgumentParser(
        prog="train_sparse_eye.py",
        description="Концепт-глаз: разреженный автоэнкодер 784→128→784, "
                    "код — распределённая популяция как в коре.",
        epilog="Пример: python tools/train_sparse_eye.py --epochs 12",
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--l1", type=float, default=L1_LAMBDA,
                    help="сила разреживания кода (по умолчанию 0.10)")
    ap.add_argument("--out", default=os.path.join(HERE, "sparse_eye.dbw"))
    args = ap.parse_args()

    print("MNIST: подготовка данных (кэш общий с train_mnist_eye.py)")
    ensure_mnist()
    xtr, _, _ = parse_idx_images(os.path.join(DATA_DIR, FILES[0]))
    ytr = parse_idx_labels(os.path.join(DATA_DIR, FILES[1]))
    xte, _, _ = parse_idx_images(os.path.join(DATA_DIR, FILES[2]))
    yte = parse_idx_labels(os.path.join(DATA_DIR, FILES[3]))
    print(f"  train {xtr.shape[0]}, test {xte.shape[0]}")

    torch.manual_seed(args.seed)
    model = ConceptEye()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xs = torch.tensor(xtr, dtype=torch.float32)
    ys = torch.tensor(ytr, dtype=torch.long)
    n = xs.shape[0]

    print(f"\nобучение: автоэнкодер {IN_DIM}→{CODE_DIM}(tanh, L1={args.l1})→{IN_DIM}, "
          f"MSE+λ·|h|₁, Adam lr={LR}, батч {BATCH}, эпох {args.epochs}, сид {args.seed}, CPU")
    g = torch.Generator().manual_seed(args.seed + 1)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, generator=g)
        total, nb = 0.0, 0
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            h = model.encode(xs[b])
            recon = model(xs[b])
            loss = torch.nn.functional.mse_loss(recon, xs[b]) + args.l1 * h.abs().mean()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        if epoch % 3 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                h = model.encode(xs[:4096])
                frac = (h.abs() > ACTIVE_TH).float().mean().item()
            print(f"  эпоха {epoch}  loss = {total / nb:.4f}  sparsity(train) = {frac:.3f}")

    # ── Тест: код + реконструкция + качество концептов ──
    model.eval()
    with torch.no_grad():
        codes = model.encode(torch.tensor(xte, dtype=torch.float32)).numpy()
        recon = model(torch.tensor(xte, dtype=torch.float32)).numpy()
    recon_mse = float(np.mean((recon - xte) ** 2))
    frac, mean_active, per = sparsity_stats(codes)
    prot, sim = class_prototypes(codes, yte, 10)
    cent_acc = nearest_centroid_acc(codes, yte, 10)
    within = np.mean([sim[c, c] for c in range(10)])
    between = np.mean([sim[a, b] for a in range(10) for b in range(10) if a != b])

    print(f"\nКОНЦЕПТ-КАЧЕСТВО (test 10k):")
    print(f"  разреженность: активных {frac:.3f} ({mean_active:.1f}/{CODE_DIM} на код; цель {SPARSITY_GOAL})")
    print(f"  реконструкция: MSE = {recon_mse:.4f} (пиксели [0,1])")
    print(f"  разделение:    nearest-centroid accuracy = {cent_acc:.4f} "
          f"(сэмпл по ближайшему прототипу — код несёт класс БЕЗ лейбла)")
    print(f"  прототипы:     косинус внутри класса {within:.3f} vs между классами {between:.3f}")
    print("  матрица близости прототипов [класс][класс]:")
    header = "      " + "".join(f"{d:>7}" for d in range(10))
    print(header)
    for k in range(10):
        print(f"  {k} |" + "".join(f"{sim[k, d]:>7.3f}" for d in range(10)))

    # ── Экспорт DBW1 ──
    tensors = model_tensors(model)
    nbytes = export_dbw(tensors, args.out)
    n_params = sum(len(t[2]) for t in tensors)
    print(f"\nэкспортировано: {args.out} ({nbytes} байт, {n_params} весов, тензоров {len(tensors)})")

    # ── Золото паритета: 5 сэмплов, код + реконструкция (f64) ──
    w0 = np.asarray(tensors[0][2], dtype=np.float64).reshape(CODE_DIM, IN_DIM)
    b0 = np.asarray(tensors[1][2], dtype=np.float64)
    w1 = np.asarray(tensors[2][2], dtype=np.float64).reshape(IN_DIM, CODE_DIM)
    b1 = np.asarray(tensors[3][2], dtype=np.float64)
    rng = np.random.RandomState(GOLDEN_N * 24 + 3)
    gold_idx = []
    while len(gold_idx) < GOLDEN_N:
        i = int(rng.randint(len(xte)))
        if i not in gold_idx:
            gold_idx.append(i)
    samples = []
    for i in gold_idx:
        h, y = numpy_forward(w0, b0, w1, b1, xte[i][None, :])
        samples.append({
            "label": int(yte[i]),
            "index": int(i),
            "input": [float(v) for v in xte[i]],
            "code": [float(v) for v in h[0]],
            "recon": [float(v) for v in y[0]],
        })
    golden = {
        "version": "sparse_eye_golden_v1",
        "in_dim": IN_DIM,
        "code_dim": CODE_DIM,
        "out_dim": IN_DIM,
        "note": "5 сэмплов test MNIST: code — разреженный концепт (128), "
                "recon — реконструкция; паритет с Rust f32 — допуск 1e-4",
        "samples": samples,
    }
    golden_path = os.path.join(HERE, "sparse_eye_golden.json")
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=1)
    print(f"золото паритета: {len(samples)} сэмплов -> {golden_path}")

    # ── Отчёт ──
    report_path = os.path.join(HERE, "sparse_eye_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("КОНЦЕПТ-ГЛАЗ (sparse_eye): пиксели → разреженная популяция концептов\n")
        f.write("=" * 78 + "\n")
        f.write(f"модель: автоэнкодер {IN_DIM}→{CODE_DIM}(tanh)→{IN_DIM}, "
                f"MSE + {args.l1}·|h|₁, Adam lr={LR}, батч {BATCH}, эпох {args.epochs}, сид {args.seed}\n")
        f.write(f"разреженность: активных {frac:.3f} ({mean_active:.1f}/{CODE_DIM} на код; цель {SPARSITY_GOAL})\n")
        f.write(f"реконструкция: MSE = {recon_mse:.4f} (пиксели [0,1])\n")
        f.write(f"nearest-centroid accuracy (сэмпл → ближайший прототип, без обучения): {cent_acc:.4f}\n")
        f.write(f"косинус прототипов: внутри {within:.3f} vs между {between:.3f}\n\n")
        f.write("матрица близости прототипов [класс][класс]:\n")
        f.write(header + "\n")
        for k in range(10):
            f.write(f"  {k} |" + "".join(f"{sim[k, d]:>7.3f}" for d in range(10)) + "\n")
        f.write(f"\nэкспорт: {args.out} ({nbytes} байт, {n_params} весов)\n")
        f.write(f"золото: {golden_path}\n")
        f.write("в игре: код инжектится током в зрительные регионы спайкового мозга —\n")
        f.write("активные единицы кода = спайковые источники (популяционный код, как в коре)\n")
    print(f"отчёт -> {report_path}")


if __name__ == "__main__":
    main()
