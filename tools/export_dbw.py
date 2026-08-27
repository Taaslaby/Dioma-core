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
# export_dbw.py — пайплайн «Python учит ↔ Rust играет»
#
# Обучает маленькую нейросеть ПОЛНОСТЬЮ НА PYTHON и экспортирует веса в
# бинарный формат DBW1 (Dioma Baked Weights, v1), который игра загружает
# через crates/dioma-engine/src/engine/baked.rs (BakedTensors::from_bytes →
# BakedMlp / BakedGru). В игре НИКОГДА не исполняется код обучения — только
# forward по запечённым весам, поэтому симуляция детерминирована.
#
#   python tools/export_dbw.py --out %TEMP%\\xor.dbw
#   python tools/export_dbw.py --torch-gru --out %TEMP%\\gru.dbw   # если есть torch
#   python tools/export_dbw.py --selftest                          # самотест формата
#
# Спецификация DBW1 (единственный источник — шапка модуля engine/baked.rs):
#
#   магия      : байты "DBW1"            (4)
#   версия     : u16 LE                  (=1)
#   n_tensors  : u32 LE
#   на тензор  :
#     имя      : u8 len + utf8
#     dtype    : u8                      (0 = f32 LE; другие reserved → Err)
#     ndim     : u8
#     dims     : ndim × u32 LE
#     данные   : prod(dims) × f32 LE
#
# Реализация строго побайтовая: struct.pack('<H'), '<I', '<f' — little-endian,
# f32 (округляет python-float до 32 бит, как Rust-сторона). Детерминизм:
# random.seed(seed) фиксирует инициализацию и порядок; numpy (если найден)
# ускоряет арифметику, но НЕ генерацию случайных чисел — инициализация
# идентична в обоих режимах.
# ---------------------------------------------------------------------------
import argparse
import math
import os
import random
import struct
import sys

try:
    import numpy as _np  # опционально: ускорение арифметики
except ImportError:
    _np = None

# ── Формат DBW1 ──────────────────────────────────────────────────────────────

DBW_MAGIC = b"DBW1"
DBW_VERSION = 1
DBW_DTYPE_F32 = 0


def pack_tensor(name, dims, data):
    """Упаковать один тензор по спеке DBW1. data — плоский список чисел
    (float64/float32); в файл идут f32 LE (округление до 32 бит)."""
    nb = name.encode("utf-8")
    if len(nb) > 255:
        raise ValueError(f"имя тензора «{name}» длиннее 255 байт")
    if len(dims) > 255:
        raise ValueError(f"тензор «{name}»: ndim {len(dims)} > 255")
    prod = 1
    for d in dims:
        prod *= d
    if prod != len(data):
        raise ValueError(f"тензор «{name}»: произведение dims {prod} != длина данных {len(data)}")
    out = bytearray()
    out.append(len(nb))
    out += nb
    out.append(DBW_DTYPE_F32)
    out.append(len(dims))
    for d in dims:
        out += struct.pack("<I", d)
    out += struct.pack("<" + "f" * len(data), *data)
    return bytes(out)


def export_dbw(tensors, out_path):
    """tensors — список (name, dims, flat_data); пишет файл DBW1 целиком."""
    blob = bytearray(DBW_MAGIC)
    blob += struct.pack("<H", DBW_VERSION)
    blob += struct.pack("<I", len(tensors))
    for name, dims, data in tensors:
        blob += pack_tensor(name, dims, data)
    with open(out_path, "wb") as f:
        f.write(bytes(blob))
    return len(blob)


# ── Самотест формата (без Rust): сверка с золотым байтовым префиксом из
#    теста golden_prefix_bytes в engine/baked.rs ──────────────────────────────

def selftest():
    """Проверяет побайтовую укладку DBW1 на фиксированной структуре."""
    tensors = [
        ("a", (2,), [1.0, 2.0]),
        ("b", (1, 2), [0.5, -0.25]),
    ]
    blob = bytearray(DBW_MAGIC)
    blob += struct.pack("<H", DBW_VERSION)
    blob += struct.pack("<I", len(tensors))
    for name, dims, data in tensors:
        blob += pack_tensor(name, dims, data)
    # Золотой префикс — ровно тот, что assert_eq'ит Rust-тест.
    golden = bytes([
        0x44, 0x42, 0x57, 0x31,  # "DBW1"
        0x01, 0x00,              # версия 1
        0x02, 0x00, 0x00, 0x00,  # 2 тензора
        0x01, 0x61,              # "a"
        0x00,                    # dtype f32
        0x01,                    # ndim 1
        0x02, 0x00, 0x00, 0x00,  # dims [2]
        0x00, 0x00, 0x80, 0x3F,  # 1.0
        0x00, 0x00, 0x00, 0x40,  # 2.0
        0x01, 0x62,              # "b"
        0x00,                    # dtype f32
        0x02,                    # ndim 2
        0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,  # dims [1,2]
        0x00, 0x00, 0x00, 0x3F,  # 0.5
        0x00, 0x00, 0x80, 0xBE,  # -0.25
    ])
    assert bytes(blob) == golden, f"байты разошлись:\n  {bytes(blob).hex()}\n  {golden.hex()}"
    # Обратная проверка: файл разбирается тем же кодом (самодостаточный
    # roundtrip на уровне структуры заголовка).
    assert blob[0:4] == DBW_MAGIC
    (ver,) = struct.unpack_from("<H", blob, 4)
    assert ver == DBW_VERSION
    (n,) = struct.unpack_from("<I", blob, 6)
    assert n == 2
    print(f"selftest OK: {len(blob)} байт совпадают с золотым префиксом Rust-теста")


# ── Микро-арифметика: numpy, если найден, иначе чистые списки ───────────────

def matmul(w, x):
    """w — список строк (выходы), x — вектор. Возвращает вектор y = W·x."""
    if _np is not None:
        return (_np.asarray(w, dtype=_np.float64) @ _np.asarray(x, dtype=_np.float64)).tolist()
    return [sum(wr[i] * x[i] for i in range(len(x))) for wr in w]


def vadd(a, b):
    if _np is not None:
        return (_np.asarray(a, dtype=_np.float64) + _np.asarray(b, dtype=_np.float64)).tolist()
    return [x + y for x, y in zip(a, b)]


def vsub(a, b):
    if _np is not None:
        return (_np.asarray(a, dtype=_np.float64) - _np.asarray(b, dtype=_np.float64)).tolist()
    return [x - y for x, y in zip(a, b)]


def vmul(a, b):
    if _np is not None:
        return (_np.asarray(a, dtype=_np.float64) * _np.asarray(b, dtype=_np.float64)).tolist()
    return [x * y for x, y in zip(a, b)]


def vtanh(a):
    if _np is not None:
        return _np.tanh(_np.asarray(a, dtype=_np.float64)).tolist()
    return [math.tanh(x) for x in a]


def vscale(s, a):
    if _np is not None:
        return (_np.asarray(a, dtype=_np.float64) * s).tolist()
    return [s * x for x in a]


# ── Обучение XOR: 2 → 8 (tanh) → 1 (линейный), полный батч, ручной backprop ─

XOR_IN = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_TGT = [[0.0], [1.0], [1.0], [0.0]]


def init_mlp(in_dim, hidden, out_dim, rng):
    """Случайная инициализация (детерминированная: rng из random.seed(seed)).
    Веса ~ N(0, 0.5) через Box–Muller от random.random()."""
    def gauss():
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    w1 = [[0.5 * gauss() for _ in range(in_dim)] for _ in range(hidden)]
    b1 = [0.5 * gauss() for _ in range(hidden)]
    w2 = [[0.5 * gauss() for _ in range(hidden)] for _ in range(out_dim)]
    b2 = [0.5 * gauss() for _ in range(out_dim)]
    return w1, b1, w2, b2


def forward_mlp(w1, b1, w2, b2, x):
    """h = tanh(W1·x + b1); y = W2·h + b2 (выход линейный)."""
    h = vtanh(vadd(matmul(w1, x), b1))
    y = vadd(matmul(w2, h), b2)
    return h, y


def train_xor(seed, epochs, hidden=8, verbose=True):
    """Полный батч по 4 точкам XOR, MSE, SGD. Возвращает (w1, b1, w2, b2, потери)."""
    rng = random.Random(seed)
    if _np is not None:
        _np.random.seed(seed)  # не влияет на инициализацию (rng — random.Random)
    w1, b1, w2, b2 = init_mlp(2, hidden, 1, rng)
    xs = XOR_IN
    ts = XOR_TGT
    n = len(xs)
    lr = 0.3  # подобрано: 0.5+ расходится на XOR (полный батч), 0.3 сходится в 0
    losses = []
    for epoch in range(epochs):
        # forward по всем точкам
        hs, ys = [], []
        for x in xs:
            h, y = forward_mlp(w1, b1, w2, b2, x)
            hs.append(h)
            ys.append(y)
        # loss = mean((y - t)^2)
        loss = sum((ys[i][0] - ts[i][0]) ** 2 for i in range(n)) / n
        if not math.isfinite(loss):
            raise RuntimeError(
                f"loss разошёлся на эпохе {epoch} ({loss}) — сеть нестабильна; "
                f"это дефект конфигурации, а не формата. Проверьте lr/инициализацию."
            )
        losses.append(loss)
        if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
            print(f"  эпоха {epoch:6d}  loss = {loss:.6f}")
        # обратное распространение (полный батч)
        d_w2 = [[0.0] * hidden]
        d_b2 = [0.0]
        d_w1 = [[0.0] * 2 for _ in range(hidden)]
        d_b1 = [0.0] * hidden
        for i in range(n):
            dy = [2.0 * (ys[i][0] - ts[i][0]) / n]  # dL/dy
            # слой 2 (линейный)
            for j in range(hidden):
                d_w2[0][j] += dy[0] * hs[i][j]
            d_b2[0] += dy[0]
            # скрытый слой (tanh)
            for j in range(hidden):
                dz = dy[0] * w2[0][j] * (1.0 - hs[i][j] * hs[i][j])
                d_b1[j] += dz
                for k in range(2):
                    d_w1[j][k] += dz * xs[i][k]
        # SGD-шаг
        for j in range(hidden):
            for k in range(2):
                w1[j][k] -= lr * d_w1[j][k]
            b1[j] -= lr * d_b1[j]
        for k in range(hidden):
            w2[0][k] -= lr * d_w2[0][k]
        b2[0] -= lr * d_b2[0]
    return w1, b1, w2, b2, losses


def run_xor(w1, b1, w2, b2, x):
    """Инференс (чистый python) — для печати результатов обучения."""
    _, y = forward_mlp(w1, b1, w2, b2, x)
    return y[0]


def mlp_tensors(w1, b1, w2, b2):
    """Экспорт в имена Rust-раннера: layer0.w [out,in], layer0.b [out], ..."""
    return [
        ("layer0.w", (len(w1), len(w1[0])), [v for row in w1 for v in row]),
        ("layer0.b", (len(b1),), list(b1)),
        ("layer1.w", (len(w2), len(w2[0])), [v for row in w2 for v in row]),
        ("layer1.b", (len(b2),), list(b2)),
    ]


# ── Режим --torch-gru (нужен установленный torch; в этом репо не обязателен) ─

def train_gru_torch(seed, steps=200, hidden=16):
    """GRUCell(1→hidden) + Linear(hidden→1) на задаче «синус t → синус t+1»,
    200 шагов Adam, детерминированно по torch.manual_seed. Возвращает тензоры
    в именах Rust-раннера gru.wz/... (см. BakedGru в engine/baked.rs).

    Конвертация из PyTorch GRUCell: порядок ворот у PyTorch — r, z, n:
      weight_ih = [W_ir; W_iz; W_in], weight_hh = [U_ir; U_iz; U_in]
    В наши имена: gru.wr/wz/wh = срезы weight_ih, gru.ur/uz/uh = срезы
    weight_hh, смещения складываются (bias_ih + bias_hh) как в уравнении
    z = σ(x·Wzᵀ + h·Uzᵀ + bz)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    cell = nn.GRUCell(1, hidden)
    readout = nn.Linear(hidden, 1)
    opt = torch.optim.Adam(list(cell.parameters()) + list(readout.parameters()), lr=0.02)

    t = torch.linspace(0.0, 12.0, steps + 1)
    s = torch.sin(t).view(-1, 1)  # (steps+1, 1)

    h = torch.zeros(1, hidden)
    for i in range(steps):
        x = s[i:i + 1]
        target = s[i + 1:i + 2]
        h = cell(x, h)
        y = readout(h)
        loss = torch.nn.functional.mse_loss(y, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        h = h.detach()

    def rows(mat, hdim):
        return [float(v) for v in mat.detach().numpy().reshape(-1)]

    ih = cell.weight_ih.detach().numpy()  # (3h, in)
    hh = cell.weight_hh.detach().numpy()  # (3h, h)
    bih = cell.bias_ih.detach().numpy()
    bhh = cell.bias_hh.detach().numpy()
    tensors = [
        ("gru.wr", (hidden, 1), rows(ih[0:hidden], hidden)),
        ("gru.wz", (hidden, 1), rows(ih[hidden:2 * hidden], hidden)),
        ("gru.wh", (hidden, 1), rows(ih[2 * hidden:3 * hidden], hidden)),
        ("gru.ur", (hidden, hidden), rows(hh[0:hidden], hidden)),
        ("gru.uz", (hidden, hidden), rows(hh[hidden:2 * hidden], hidden)),
        ("gru.uh", (hidden, hidden), rows(hh[2 * hidden:3 * hidden], hidden)),
        ("gru.br", (hidden,), [float(v) for v in (bih[0:hidden] + bhh[0:hidden])]),
        ("gru.bz", (hidden,), [float(v) for v in (bih[hidden:2 * hidden] + bhh[hidden:2 * hidden])]),
        ("gru.bh", (hidden,), [float(v) for v in (bih[2 * hidden:3 * hidden] + bhh[2 * hidden:3 * hidden])]),
    ]
    return tensors, float(loss.item())


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        prog="export_dbw.py",
        description="Пайплайн «Python учит ↔ Rust играет»: обучить маленькую сеть "
                    "и запечь веса в формат DBW1 (см. crates/dioma-engine/src/engine/baked.rs).",
        epilog="Пример: python tools/export_dbw.py --out %TEMP%\\xor.dbw --epochs 5000",
    )
    ap.add_argument("--out", default="xor.dbw",
                    help="путь к выходному файлу .dbw (по умолчанию xor.dbw)")
    ap.add_argument("--seed", type=int, default=7,
                    help="сид детерминизма (random.seed + numpy.seed; по умолчанию 7)")
    ap.add_argument("--epochs", type=int, default=5000,
                    help="число эпох обучения XOR (по умолчанию 5000)")
    ap.add_argument("--torch-gru", action="store_true",
                    help="вместо XOR: GRUCell на синусоиде (нужен установленный torch), "
                         "экспорт тензоров gru.*")
    ap.add_argument("--selftest", action="store_true",
                    help="прогнать встроенный самотест формата DBW1 и выйти")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.torch_gru:
        try:
            import torch  # noqa: F401  (проверка наличия)
        except ImportError:
            print("--torch-gru требует установленный torch; в этом окружении его нет.", file=sys.stderr)
            sys.exit(2)
        tensors, loss = train_gru_torch(args.seed)
        n = export_dbw(tensors, args.out)
        print(f"torch GRU: финальный loss = {loss:.6f}")
        print(f"экспортировано тензоров: {len(tensors)} (gru.wz/wr/wh, gru.uz/ur/uh, gru.bz/br/bh)")
        print(f"записано {n} байт -> {args.out}")
        return

    # Обычный путь: XOR 2 → 8 (tanh) → 1.
    print(f"обучение XOR 2→8→1, сид {args.seed}, эпох {args.epochs} "
          f"({'numpy' if _np is not None else 'чистый python'})")
    w1, b1, w2, b2, losses = train_xor(args.seed, args.epochs)
    final = losses[-1]
    print(f"\nвыходы сети на 4 точках XOR (ожидание ≈0 / ≈1):")
    for x, t in zip(XOR_IN, XOR_TGT):
        y = run_xor(w1, b1, w2, b2, x)
        print(f"  x={x}  ->  {y:+.4f}   (цель {t[0]})")
    ok = all(abs(run_xor(w1, b1, w2, b2, x) - t[0]) < 0.5 for x, t in zip(XOR_IN, XOR_TGT))
    if not ok:
        print(f"ВНИМАНИЕ: сеть не разделила XOR (финальный loss {final:.4f}) — "
              f"увеличьте --epochs или измените сид.", file=sys.stderr)
    n = export_dbw(mlp_tensors(w1, b1, w2, b2), args.out)
    print(f"\nэкспортировано 4 тензора (layer0.w/b, layer1.w/b) -> {args.out} ({n} байт)")


if __name__ == "__main__":
    main()
