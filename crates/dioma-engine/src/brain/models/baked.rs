//! # Запечённые веса: формат DBW1 и инференс-раннеры
//!
//! Пайплайн «Python учит ↔ Rust играет»: нейросетевые модули (ANN-энкодер
//! зрения, GRU-память) тренируются ОФЛАЙН на Python (`tools/export_dbw.py`)
//! и запекаются в бинарный файл весов DBW1; игра грузит его через
//! [`BakedTensors::from_bytes`] и исполняет чистым Rust-инференсом
//! ([`BakedMlp`], [`BakedGru`]). Никакого бэкпропа в игре — только forward,
//! поэтому симуляция детерминирована и повторяема между запусками.
//!
//! Детерминизм: формат LE-байтовый, f32, порядок тензоров = порядок файла;
//! чтение не зависит от платформы. Русские тексты ошибок — для честных
//! сообщений игроку/разработчику.
//!
//! ## Спецификация формата DBW1 (Dioma Baked Weights, v1)
//!
//! ```text
//! магия      : байты "DBW1"            (4)
//! версия     : u16 LE                  (=1)
//! n_tensors  : u32 LE
//! на тензор  :
//!   имя      : u8 len + utf8           (len ≤ 255)
//!   dtype    : u8                      (0 = f32 LE; другие пока reserved → Err)
//!   ndim     : u8                      (0..255; практически 1..4)
//!   dims     : ndim × u32 LE           (произведение ≥ 0; пустой тензор
//!                                        допустим ТОЛЬКО ndim=1, dims=[0];
//!                                        прочие нулевые произведения → Err)
//!   данные   : prod(dims) × f32 LE
//! ```
//!
//! Ограничения и соглашения:
//!
//! - версия ≠ 1 → `Err` (будущие версии — отдельный магия/версия);
//! - dtype ≠ 0 → `Err` (f16/i8 и прочее зарезервированы);
//! - имена — уникальные utf8 (иначе `get` неоднозначен) → `Err`;
//! - длина имени u8, то есть ≤ 255 байт; имена вида `layer0.w`, `gru.wz`
//!   укладываются с запасом;
//! - пустой тензор: нулевое произведение dims допустимо только как
//!   `ndim=1, dims=[0]` (один скалярный срез нулевой длины). Формы вида
//!   `[0, 5]`, `[3, 0, 2]` — `Err` (неоднозначный размер, запрещён).
//! - произведение dims считается в u64 с проверкой переполнения; файл
//!   не обязан быть самосогласованным по размеру — длина проверяется
//!   по факту доступных байт.
//!
//! Инференс-раннеры ниже — обычные циклы без nalgebra/matrix-зависимостей:
//! слой MLP — это `out = W·x + b` (строки W = выходы), GRU — стандартные
//! ворота (см. доки каждого раннера).

/// Магия формата: байты «DBW1» (Dioma Baked Weights, v1).
pub const DBW_MAGIC: &[u8; 4] = b"DBW1";

/// Версия формата (u16 LE в файле).
pub const DBW_VERSION: u16 = 1;

/// dtype f32 (LE) — единственный поддерживаемый.
pub const DBW_DTYPE_F32: u8 = 0;

/// Контейнер тензоров DBW1: упорядоченный список имён + сырые f32-данные.
///
/// Хранит имена, размерности и данные в порядке файла; `get` достаёт срез
/// данных по имени. Создаётся из файла ([`from_bytes`]) или собирается
/// вручную ([`push`]) — например, для тестов и будущих инструментов.
#[derive(Debug)]
pub struct BakedTensors {
    /// Имена тензоров в порядке файла (уникальны — проверено при вставке).
    names: Vec<String>,
    /// Размерности (порядок = `names`).
    dims: Vec<Vec<u32>>,
    /// Плоские f32-данные (порядок = `names`).
    data: Vec<Vec<f32>>,
}

impl Default for BakedTensors {
    fn default() -> Self {
        Self::new()
    }
}

impl BakedTensors {
    /// Пустой контейнер (0 тензоров).
    pub fn new() -> Self {
        BakedTensors {
            names: Vec::new(),
            dims: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Добавить тензор. Проверки: имя уникально и ≤ 255 байт, dims не пусты,
    /// произведение dims (u64, с проверкой переполнения) равно длине данных,
    /// пустой тензор — только `ndim=1, dims=[0]`.
    pub fn push(&mut self, name: &str, dims: &[u32], data: Vec<f32>) -> Result<(), String> {
        if name.is_empty() {
            return Err("имя тензора пустое".to_string());
        }
        if name.len() > 255 {
            return Err(format!("имя тензора «{name}» длиннее 255 байт"));
        }
        if self.names.iter().any(|n| n == name) {
            return Err(format!("тензор «{name}» уже есть — имена обязаны быть уникальными"));
        }
        if dims.is_empty() {
            return Err(format!("тензор «{name}»: пустой список размерностей недопустим"));
        }
        let prod = match tensor_len(dims) {
            Some(p) => p,
            None => return Err(format!("тензор «{name}»: переполнение произведения размерностей")),
        };
        if prod == 0 {
            let ok = dims.len() == 1 && dims[0] == 0;
            if !ok {
                return Err(format!(
                    "тензор «{name}»: пустой тензор допустим только как ndim=1, dims=[0], \
                     а не {dims:?}"
                ));
            }
        }
        if prod != data.len() as u64 {
            return Err(format!(
                "тензор «{name}»: произведение размерностей {prod} не совпадает с длиной \
                 данных {}",
                data.len()
            ));
        }
        self.names.push(name.to_string());
        self.dims.push(dims.to_vec());
        self.data.push(data);
        Ok(())
    }

    /// Имена тензоров в порядке файла.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Размерности тензора по имени.
    pub fn dims(&self, name: &str) -> Option<&[u32]> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.dims[i].as_slice())
    }

    /// Сырые f32-данные тензора (плоский срез, row-major).
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.data[i].as_slice())
    }

    /// Сериализация в DBW1 (детерминированная, LE, побайтовая по спеке).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.names.len() * 8 + self.data.iter().map(Vec::len).sum::<usize>() * 4);
        out.extend_from_slice(DBW_MAGIC);
        out.extend_from_slice(&DBW_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.names.len() as u32).to_le_bytes());
        for i in 0..self.names.len() {
            let name = &self.names[i];
            out.push(name.len() as u8);
            out.extend_from_slice(name.as_bytes());
            out.push(DBW_DTYPE_F32);
            out.push(self.dims[i].len() as u8);
            for &d in &self.dims[i] {
                out.extend_from_slice(&d.to_le_bytes());
            }
            for &x in &self.data[i] {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// Разобрать DBW1. Проверки: магия, версия, dtype, уникальность имён,
    /// произведение dims == длине данных, пустые тензоры только ndim=1
    /// dims=[0], целостность длины (truncate → Err). Ошибки — на русском.
    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        let mut cur = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let end = cur.checked_add(n).ok_or("переполнение смещения при разборе")?;
            let s = b.get(cur..end).ok_or("файл обрезан: не хватает байт")?;
            cur = end;
            Ok(s)
        };

        let magic = take(4)?;
        if magic != DBW_MAGIC {
            return Err(format!(
                "неверная магия формата: {magic:02X?} (ожидалось {DBW_MAGIC:?})"
            ));
        }
        let ver = u16::from_le_bytes(take(2)?.try_into().unwrap());
        if ver != DBW_VERSION {
            return Err(format!("неподдерживаемая версия DBW: {ver} (ожидалась {DBW_VERSION})"));
        }
        let n_tensors = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;

        let mut ts = BakedTensors::new();
        for ti in 0..n_tensors {
            let name_len = take(1)?[0] as usize;
            let name = std::str::from_utf8(take(name_len)?)
                .map_err(|_| format!("тензор #{ti}: имя не является корректным utf8"))?
                .to_string();
            let dtype = take(1)?[0];
            if dtype != DBW_DTYPE_F32 {
                return Err(format!(
                    "тензор «{name}»: неподдерживаемый dtype {dtype} (поддерживается только 0 = f32)"
                ));
            }
            let ndim = take(1)?[0] as usize;
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                dims.push(u32::from_le_bytes(take(4)?.try_into().unwrap()));
            }
            let prod = tensor_len(&dims)
                .ok_or_else(|| format!("тензор «{name}»: переполнение произведения размерностей"))?;
            if prod == 0 && !(ndim == 1 && dims[0] == 0) {
                return Err(format!(
                    "тензор «{name}»: пустой тензор допустим только как ndim=1, dims=[0], \
                     а не {dims:?}"
                ));
            }
            let n = usize::try_from(prod)
                .map_err(|_| format!("тензор «{name}»: размер данных не помещается в usize"))?;
            let n_bytes = n
                .checked_mul(4)
                .ok_or_else(|| format!("тензор «{name}»: переполнение длины данных"))?;
            let raw = take(n_bytes)?;
            let mut data = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(4) {
                data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            ts.push(&name, &dims, data)
                .map_err(|e| format!("тензор #{ti}: {e}"))?;
        }
        Ok(ts)
    }
}

/// Произведение dims в u64 (`None` при переполнении u32-произведения).
fn tensor_len(dims: &[u32]) -> Option<u64> {
    let mut p: u64 = 1;
    for &d in dims {
        p = p.checked_mul(d as u64)?;
    }
    Some(p)
}

/// Достать тензор по имени: данные + размерность; ошибка — на русском.
fn need_tensor<'a>(t: &'a BakedTensors, name: &str) -> Result<(&'a [u32], &'a [f32]), String> {
    let data = t
        .get(name)
        .ok_or_else(|| format!("в запечённых весах нет тензора «{name}»"))?;
    let dims = t.dims(name).unwrap();
    Ok((dims, data))
}

// ─── Полносвязная сеть ──────────────────────────────────────────────────────

/// Полносвязная сеть (MLP) над запечёнными весами.
///
/// Слои берутся из тензоров `"layer{i}.w"` [out, in] и `"layer{i}.b"` [out],
/// начиная с `layer0` и подряд, пока тензоры существуют. Активации: скрытые
/// слои — `tanh`, выходной — линейный. Конструктор валидирует совместность
/// размерностей: у каждого слоя `w` [out, in], `b` [out], и выход предыдущего
/// слоя равен входу следующего; без слоёв — ошибка.
///
/// ## Формула forward
///
/// ```text
/// a_0 = x                                   (вход, len = in_dim)
/// z_k = W_k · a_{k-1} + b_k                 (W_k: [out_k, in_k])
/// a_k = tanh(z_k)                    k < L-1 (скрытые)
/// y   = z_{L-1}                      k = L-1 (выход линейный)
/// ```
///
/// Стоимость одного forward — 2·Σ out_k·in_k флопов (умножения + сложения),
/// без аллокаций внутри цикла по слоям (переиспользуется один буфер).
pub struct BakedMlp {
    /// weights[i][o][in] — строки W_i = выходы слоя i.
    weights: Vec<Vec<Vec<f32>>>,
    /// biases[i][o].
    biases: Vec<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
}

impl BakedMlp {
    /// Собрать сеть из тензоров `layer{i}.w`/`layer{i}.b` (см. доки структуры).
    pub fn from_tensors(t: &BakedTensors) -> Result<Self, String> {
        let mut weights: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut biases: Vec<Vec<f32>> = Vec::new();
        let mut prev_out: Option<usize> = None;
        let mut in_dim = 0usize;
        for i in 0.. {
            let (wd, w) = match t.get(&format!("layer{i}.w")) {
                Some(w) => (t.dims(&format!("layer{i}.w")).unwrap(), w),
                None => break,
            };
            let (bd, b) = need_tensor(t, &format!("layer{i}.b"))?;
            if wd.len() != 2 {
                return Err(format!("слой {i}: «layer{i}.w» обязан быть 2-мерным [out, in], а не {wd:?}"));
            }
            let (out, inn) = (wd[0] as usize, wd[1] as usize);
            if out == 0 || inn == 0 {
                return Err(format!("слой {i}: «layer{i}.w» с нулевым размером ({wd:?}) недопустим"));
            }
            if bd.len() != 1 || bd[0] as usize != out {
                return Err(format!(
                    "слой {i}: «layer{i}.b» обязан быть [{out}], а не {bd:?}"
                ));
            }
            if w.len() != out * inn {
                return Err(format!(
                    "слой {i}: «layer{i}.w» длина {} не равна {out}×{inn}",
                    w.len()
                ));
            }
            if b.len() != out {
                return Err(format!("слой {i}: «layer{i}.b» длина {} не равна {out}", b.len()));
            }
            if let Some(po) = prev_out {
                if po != inn {
                    return Err(format!(
                        "слой {i}: вход {inn} не совпадает с выходом предыдущего слоя {po}"
                    ));
                }
            } else {
                in_dim = inn;
            }
            // Развернуть плоский [out·in] в строки [out][in] (row-major).
            let mut rows = Vec::with_capacity(out);
            for r in 0..out {
                rows.push(w[r * inn..(r + 1) * inn].to_vec());
            }
            weights.push(rows);
            biases.push(b.to_vec());
            prev_out = Some(out);
        }
        if weights.is_empty() {
            return Err("в запечённых весах нет ни одного слоя (нужны «layer0.w»/«layer0.b»)".to_string());
        }
        let out_dim = prev_out.unwrap();
        Ok(BakedMlp {
            weights,
            biases,
            in_dim,
            out_dim,
        })
    }

    /// Размер входа (ожидается `x.len() == in_dim`).
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Размер выхода (длина результата `forward`).
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Forward: `x` длины `in_dim` → вектор длины `out_dim` (см. доки).
    ///
    /// Паника при `x.len() != in_dim` — это программная ошибка вызова,
    /// а не данные файла (конструктор уже всё проверил).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim, "длина входа {} != in_dim {}", x.len(), self.in_dim);
        let mut cur: Vec<f32> = x.to_vec();
        let n_layers = self.weights.len();
        for (li, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let out = w.len();
            let inn = cur.len();
            let mut next = vec![0.0f32; out];
            for o in 0..out {
                let mut acc = b[o];
                let row = &w[o];
                for i in 0..inn {
                    acc += row[i] * cur[i];
                }
                next[o] = if li + 1 == n_layers { acc } else { acc.tanh() };
            }
            cur = next;
        }
        cur
    }

    /// Forward для автоэнкодеров: `(КОД, выход)`. Код — активация
    /// предпоследнего слоя (tanh), выход — последний (линейный). Для
    /// однослойной сети код = вход (тривиально).
    ///
    /// Используется концепт-глазом (`examples/sparse_eye_probe.rs`, тренер
    /// `tools/train_sparse_eye.py`): 128-мерный разреженный код инжектится
    /// током в зрительные регионы спайкового мозга — это интерфейс между
    /// запечённым глазом и живой динамикой, а не пробниковая хитрость.
    pub fn forward_parts(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(x.len(), self.in_dim, "длина входа {} != in_dim {}", x.len(), self.in_dim);
        let mut cur: Vec<f32> = x.to_vec();
        let n_layers = self.weights.len();
        for (li, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let out = w.len();
            let inn = cur.len();
            let mut next = vec![0.0f32; out];
            for o in 0..out {
                let mut acc = b[o];
                let row = &w[o];
                for i in 0..inn {
                    acc += row[i] * cur[i];
                }
                next[o] = if li + 1 == n_layers { acc } else { acc.tanh() };
            }
            if li + 1 == n_layers {
                // cur — активация предпоследнего слоя (tanh), next — выход.
                return (cur, next);
            }
            cur = next;
        }
        (cur.clone(), cur) // недостижимо: from_tensors требует ≥ 1 слоя
    }
}

// ─── GRU-блок ───────────────────────────────────────────────────────────────

/// Классический GRU-блок над запечёнными весами.
///
/// Тензоры: `gru.wz`/`gru.wr`/`gru.wh` [hidden, in] (входные проекции),
/// `gru.uz`/`gru.ur`/`gru.uh` [hidden, hidden] (рекуррентные), смещения
/// `gru.bz`/`gru.br`/`gru.bh` [hidden]. Конструктор валидирует совместность:
/// все проекции одного размера, смещения [hidden], входы совпадают.
///
/// ## Уравнения (стандартный GRU, как GRUCell в PyTorch)
///
/// ```text
/// z = σ(x·Wzᵀ + h·Uzᵀ + bz)     — ворота обновления [hidden]
/// r = σ(x·Wrᵀ + h·Urᵀ + br)     — ворота сброса   [hidden]
/// n = tanh(x·Whᵀ + (r ⊙ h)·Uhᵀ + bh)  — кандидат [hidden]
/// h' = (1 − z) ⊙ n + z ⊙ h      — новое скрытое состояние
/// ```
///
/// где σ — логистическая сигмоида, ⊙ — покомпонентное умножение, `Wᵀ`
/// означает «строки W = выходы» (тензор [hidden, in] умножен справа на x).
///
/// `step` мутирует скрытое состояние на месте (один временной шаг);
/// `hidden` отдаёт текущее состояние; `reset` обнуляет его. Скрытое
/// состояние инициализируется нулями — детерминированный старт.
pub struct BakedGru {
    wz: Vec<Vec<f32>>,
    wr: Vec<Vec<f32>>,
    wh: Vec<Vec<f32>>,
    uz: Vec<Vec<f32>>,
    ur: Vec<Vec<f32>>,
    uh: Vec<Vec<f32>>,
    bz: Vec<f32>,
    br: Vec<f32>,
    bh: Vec<f32>,
    h: Vec<f32>,
    in_dim: usize,
}

impl BakedGru {
    /// Собрать блок из тензоров `gru.*` (см. доки структуры).
    pub fn from_tensors(t: &BakedTensors) -> Result<Self, String> {
        let (wzd, wz) = need_tensor(t, "gru.wz")?;
        let (wrd, wr) = need_tensor(t, "gru.wr")?;
        let (whd, wh) = need_tensor(t, "gru.wh")?;
        let (uzd, uz) = need_tensor(t, "gru.uz")?;
        let (urd, ur) = need_tensor(t, "gru.ur")?;
        let (uhd, uh) = need_tensor(t, "gru.uh")?;
        let mat = |d: &[u32], what: &str| -> Result<(usize, usize), String> {
            if d.len() != 2 {
                return Err(format!("«{what}» обязан быть 2-мерным [hidden, …], а не {d:?}"));
            }
            Ok((d[0] as usize, d[1] as usize))
        };
        let (h1, i1) = mat(wzd, "gru.wz")?;
        let (h2, i2) = mat(wrd, "gru.wr")?;
        let (h3, i3) = mat(whd, "gru.wh")?;
        let (h4, i4) = mat(uzd, "gru.uz")?;
        let (h5, i5) = mat(urd, "gru.ur")?;
        let (h6, i6) = mat(uhd, "gru.uh")?;
        let h = h1;
        let inn = i1;
        if h2 != h || h3 != h || h4 != h || h5 != h || h6 != h {
            return Err(format!(
                "разные размеры hidden в проекциях GRU: {h1}/{h2}/{h3}/{h4}/{h5}/{h6}"
            ));
        }
        if i2 != inn || i3 != inn {
            return Err(format!("входные размеры GRU не совпадают: {i1}/{i2}/{i3}"));
        }
        if i4 != h || i5 != h || i6 != h {
            return Err(format!(
                "рекуррентные проекции GRU обязаны быть [hidden, hidden], а не {i4}/{i5}/{i6}"
            ));
        }
        // Развернуть плоский [n·cols] в строки [n][cols] (row-major, строки = выходы).
        let rows = |name: &str, d: &[f32], n: usize, cols: usize| -> Result<Vec<Vec<f32>>, String> {
            if d.len() != n * cols {
                return Err(format!("«{name}»: длина {} не равна {n}×{cols}", d.len()));
            }
            let mut v = Vec::with_capacity(n);
            for r in 0..n {
                v.push(d[r * cols..(r + 1) * cols].to_vec());
            }
            Ok(v)
        };
        let bias = |name: &str, n: usize| -> Result<Vec<f32>, String> {
            let (d, v) = need_tensor(t, name)?;
            if d.len() != 1 || d[0] as usize != n {
                return Err(format!("«{name}» обязан быть [{n}], а не {d:?}"));
            }
            if v.len() != n {
                return Err(format!("«{name}»: длина {} не равна {n}", v.len()));
            }
            Ok(v.to_vec())
        };
        Ok(BakedGru {
            wz: rows("gru.wz", wz, h, inn)?,
            wr: rows("gru.wr", wr, h, inn)?,
            wh: rows("gru.wh", wh, h, inn)?,
            uz: rows("gru.uz", uz, h, h)?,
            ur: rows("gru.ur", ur, h, h)?,
            uh: rows("gru.uh", uh, h, h)?,
            bz: bias("gru.bz", h)?,
            br: bias("gru.br", h)?,
            bh: bias("gru.bh", h)?,
            h: vec![0.0; h],
            in_dim: inn,
        })
    }

    /// Размер входа (ожидается `x.len() == in_dim`).
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Размер скрытого состояния (длина `hidden()`).
    pub fn hidden_dim(&self) -> usize {
        self.h.len()
    }

    /// Текущее скрытое состояние (после последнего `step`; нули до первого).
    pub fn hidden(&self) -> &[f32] {
        &self.h
    }

    /// Обнулить скрытое состояние (детерминированный перезапуск).
    pub fn reset(&mut self) {
        self.h.fill(0.0);
    }

    /// Один временной шаг: `h ← (1−z)⊙n + z⊙h` (см. доки структуры).
    /// Паника при `x.len() != in_dim` — программная ошибка вызова.
    ///
    /// Все ворота считаются от СТАРОГО скрытого состояния (h_{t-1}), потом
    /// новое состояние собирается в отдельный буфер и заменяет старое —
    /// внутри шага порядок обновления не влияет на результат.
    pub fn step(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.in_dim, "длина входа {} != in_dim {}", x.len(), self.in_dim);
        let h = self.hidden_dim();
        let inn = self.in_dim;
        let mut z = vec![0.0f32; h];
        let mut r = vec![0.0f32; h];
        let mut g = vec![0.0f32; h]; // преактивация кандидата (без Uh-члена)
        for j in 0..h {
            let mut az = self.bz[j];
            let mut ar = self.br[j];
            let mut an = self.bh[j];
            for i in 0..inn {
                az += self.wz[j][i] * x[i];
                ar += self.wr[j][i] * x[i];
                an += self.wh[j][i] * x[i];
            }
            for k in 0..h {
                let hk = self.h[k];
                az += self.uz[j][k] * hk;
                ar += self.ur[j][k] * hk;
            }
            z[j] = sigmoid(az);
            r[j] = sigmoid(ar);
            g[j] = an;
        }
        let mut next = vec![0.0f32; h];
        for j in 0..h {
            let mut an = g[j];
            for k in 0..h {
                an += self.uh[j][k] * (r[k] * self.h[k]);
            }
            next[j] = (1.0 - z[j]) * an.tanh() + z[j] * self.h[j];
        }
        self.h = next;
    }
}

/// Логистическая сигмоида: `1 / (1 + exp(-x))`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ручной контейнер: 1-слойная сеть для тестов (см. тесты ниже).
    fn mlp_fixture() -> BakedTensors {
        let mut t = BakedTensors::new();
        // layer0: 2 → 1, линейный выход (единственный слой — выход).
        t.push("layer0.w", &[1, 2], vec![0.5, -1.0]).unwrap();
        t.push("layer0.b", &[1], vec![0.25]).unwrap();
        t
    }

    #[test]
    fn roundtrip_bytes_bitwise() {
        let mut t = BakedTensors::new();
        t.push("layer0.w", &[2, 3], vec![1.0, -2.0, 0.5, 0.0, 3.5, -0.125]).unwrap();
        t.push("layer0.b", &[2], vec![0.1, -0.2]).unwrap();
        t.push("gru.wz", &[4, 3], vec![0.0; 12]).unwrap();
        let bytes = t.to_bytes();
        let back = BakedTensors::from_bytes(&bytes).unwrap();
        assert_eq!(back.to_bytes(), bytes, "roundtrip побитово");
        assert_eq!(back.names(), t.names());
        assert_eq!(back.get("layer0.w"), Some(&[1.0, -2.0, 0.5, 0.0, 3.5, -0.125][..]));
        assert_eq!(back.dims("gru.wz"), Some(&[4, 3][..]));
        // from_bytes из to_bytes снова даёт те же байты — идемпотентно.
        assert_eq!(BakedTensors::from_bytes(&back.to_bytes()).unwrap().to_bytes(), bytes);
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let mut bad = b"DBW1".to_vec();
        bad[0] = b'X';
        let err = BakedTensors::from_bytes(&bad).unwrap_err();
        assert!(err.contains("магия"), "ошибка про магию: {err}");
    }

    #[test]
    fn from_bytes_rejects_bad_version() {
        let bytes = b"DBW1\x02\x00\x00\x00\x00\x00".to_vec();
        let err = BakedTensors::from_bytes(&bytes).unwrap_err();
        assert!(err.contains("версия"), "ошибка про версию: {err}");
    }

    #[test]
    fn from_bytes_rejects_truncated() {
        let t = mlp_fixture();
        let bytes = t.to_bytes();
        // Режем в середине данных: все срезы обязаны проверяться по длине.
        for cut in [4usize, 6, 10, bytes.len() - 3, bytes.len() - 1] {
            let err = BakedTensors::from_bytes(&bytes[..cut]).unwrap_err();
            assert!(
                err.contains("обрезан") || err.contains("длина") || err.contains("данных"),
                "cut={cut}: {err}"
            );
        }
    }

    #[test]
    fn from_bytes_rejects_unknown_dtype() {
        let mut bytes = mlp_fixture().to_bytes();
        // dtype первого тензора: магия(4)+версия(2)+n_tensors(4)+len(1)+имя(8)+dtype(1).
        let dtype_pos = 4 + 2 + 4 + 1 + 8;
        bytes[dtype_pos] = 7;
        let err = BakedTensors::from_bytes(&bytes).unwrap_err();
        assert!(err.contains("dtype"), "ошибка про dtype: {err}");
    }

    #[test]
    fn from_bytes_empty_tensor_rules() {
        // ndim=1, dims=[0] — допустимо.
        let mut t = BakedTensors::new();
        t.push("empty", &[0], vec![]).unwrap();
        let back = BakedTensors::from_bytes(&t.to_bytes()).unwrap();
        assert_eq!(back.get("empty"), Some(&[][..]));
        // ndim=2, dims=[3, 0] — Err (и в push, и в from_bytes).
        let err = t.push("bad", &[3, 0], vec![]).unwrap_err();
        assert!(err.contains("ndim=1"), "push: {err}");
        // Кривой файл: собрать байты вручную.
        let mut bytes = b"DBW1".to_vec();
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(3);
        bytes.extend_from_slice(b"bad");
        bytes.push(0); // dtype f32
        bytes.push(2); // ndim=2
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let err = BakedTensors::from_bytes(&bytes).unwrap_err();
        assert!(err.contains("ndim=1"), "from_bytes: {err}");
    }

    #[test]
    fn from_bytes_rejects_huge_dims_overflow() {
        // dims [2^32−1, 2^32−1]: произведение ~2^64 не влезает в длину данных —
        // обязано вернуть Err БЕЗ паники (защита от переполнения n*4).
        let mut bytes = b"DBW1".to_vec();
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.push(3);
        bytes.extend_from_slice(b"big");
        bytes.push(0); // dtype f32
        bytes.push(2); // ndim=2
        bytes.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        bytes.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let err = BakedTensors::from_bytes(&bytes).unwrap_err();
        assert!(err.contains("переполн"), "ошибка про переполнение: {err}");
    }

    #[test]
    fn mlp_forward_hand_computed_1_layer() {
        // Один слой, линейный выход: y = W·x + b.
        //   W = [[0.5, -1.0]], b = [0.25]
        //   x = [1.0, 2.0] → y = 0.5·1 + (−1.0)·2 + 0.25 = −1.25
        let mlp = BakedMlp::from_tensors(&mlp_fixture()).unwrap();
        assert_eq!(mlp.in_dim(), 2);
        assert_eq!(mlp.out_dim(), 1);
        let y = mlp.forward(&[1.0, 2.0]);
        assert_eq!(y.len(), 1);
        assert!((y[0] - (-1.25)).abs() < 1e-6, "y = {y:?}, ожидалось −1.25");
    }

    #[test]
    fn mlp_forward_hidden_tanh_2_layers() {
        // Два слоя: 2 → 2 (tanh) → 1 (линейный).
        //   layer0.w = [[0.5, -1.0], [0.2, 0.3]], b0 = [0.1, -0.2]
        //   layer1.w = [[1.0, -2.0]],            b1 = [0.5]
        // x = [1.0, 2.0]:
        //   z0 = 0.5·1 + (−1)·2 + 0.1 = −1.4  → tanh(−1.4) ≈ −0.885351648
        //   z1 = 0.2·1 + 0.3·2 − 0.2 = 0.6    → tanh(0.6) ≈ 0.537049567
        //   y  = 1·tanh(−1.4) + (−2)·tanh(0.6) + 0.5 ≈ −0.885351648 − 1.074099134 + 0.5
        //      ≈ −1.459450782
        let mut t = BakedTensors::new();
        t.push("layer0.w", &[2, 2], vec![0.5, -1.0, 0.2, 0.3]).unwrap();
        t.push("layer0.b", &[2], vec![0.1, -0.2]).unwrap();
        t.push("layer1.w", &[1, 2], vec![1.0, -2.0]).unwrap();
        t.push("layer1.b", &[1], vec![0.5]).unwrap();
        let mlp = BakedMlp::from_tensors(&t).unwrap();
        assert_eq!(mlp.out_dim(), 1);
        let y = mlp.forward(&[1.0, 2.0]);
        let expected = -0.885_351_648 - 2.0 * 0.537_049_567 + 0.5;
        assert!(
            (y[0] - expected).abs() < 1e-6,
            "y = {}, ожидалось {}",
            y[0],
            expected
        );
    }

    #[test]
    fn mlp_forward_parts_returns_code_and_output() {
        // Тот же двухслойный фикстур, что в forward-тесте: код = tanh-активация
        // предпоследнего слоя, выход = последний (линейный).
        let mut t = BakedTensors::new();
        t.push("layer0.w", &[2, 2], vec![0.5, -1.0, 0.2, 0.3]).unwrap();
        t.push("layer0.b", &[2], vec![0.1, -0.2]).unwrap();
        t.push("layer1.w", &[1, 2], vec![1.0, -2.0]).unwrap();
        t.push("layer1.b", &[1], vec![0.5]).unwrap();
        let mlp = BakedMlp::from_tensors(&t).unwrap();
        let (code, y) = mlp.forward_parts(&[1.0, 2.0]);
        // h = tanh(W0·x + b0)
        let h0 = (0.5f32 * 1.0 + -1.0 * 2.0 + 0.1).tanh();
        let h1 = (0.2f32 * 1.0 + 0.3 * 2.0 + -0.2).tanh();
        assert_eq!(code.len(), 2);
        assert!((code[0] - h0).abs() < 1e-6, "code[0] = {}, ожидалось {}", code[0], h0);
        assert!((code[1] - h1).abs() < 1e-6, "code[1] = {}, ожидалось {}", code[1], h1);
        // y = W1·h + b1 — должен совпасть с обычным forward.
        let expected = 1.0 * h0 + -2.0 * h1 + 0.5;
        assert!(
            (y[0] - expected).abs() < 1e-6,
            "y = {}, ожидалось {}",
            y[0],
            expected
        );
        assert!((y[0] - mlp.forward(&[1.0, 2.0])[0]).abs() < 1e-9, "forward_parts ≠ forward");
    }

    #[test]
    fn mlp_from_tensors_validates_dimensions() {
        // Нет слоёв → Err.
        let t = BakedTensors::new();
        assert!(BakedMlp::from_tensors(&t).is_err());
        // Есть w, нет b → Err.
        let mut t = BakedTensors::new();
        t.push("layer0.w", &[2, 2], vec![1.0; 4]).unwrap();
        assert!(BakedMlp::from_tensors(&t).is_err());
        // b не совпадает с out → Err.
        let mut t = BakedTensors::new();
        t.push("layer0.w", &[2, 2], vec![1.0; 4]).unwrap();
        t.push("layer0.b", &[3], vec![0.0; 3]).unwrap();
        assert!(BakedMlp::from_tensors(&t).is_err());
        // Несогласованные слои: out 2 → in 3 → Err.
        let mut t = BakedTensors::new();
        t.push("layer0.w", &[2, 2], vec![1.0; 4]).unwrap();
        t.push("layer0.b", &[2], vec![0.0; 2]).unwrap();
        t.push("layer1.w", &[1, 3], vec![1.0; 3]).unwrap();
        t.push("layer1.b", &[1], vec![0.0]).unwrap();
        assert!(BakedMlp::from_tensors(&t).is_err());
        // Разрыв в нумерации (нет layer1 → сеть просто заканчивается на layer0,
        // это валидно: инференс не требует сквозной нумерации).
        let mut t = BakedTensors::new();
        t.push("layer0.w", &[1, 2], vec![1.0, 0.0]).unwrap();
        t.push("layer0.b", &[1], vec![0.0]).unwrap();
        t.push("layer2.w", &[1, 1], vec![1.0]).unwrap();
        t.push("layer2.b", &[1], vec![0.0]).unwrap();
        let mlp = BakedMlp::from_tensors(&t).unwrap();
        assert_eq!(mlp.out_dim(), 1);
    }

    #[test]
    fn gru_step_determinism_and_zero_invariant() {
        // Сеть 3 → 4 для детерминизма.
        let mut t = BakedTensors::new();
        t.push("gru.wz", &[4, 3], (0..12).map(|i| (i as f32) * 0.1).collect()).unwrap();
        t.push("gru.wr", &[4, 3], (0..12).map(|i| (i as f32) * 0.05).collect()).unwrap();
        t.push("gru.wh", &[4, 3], (0..12).map(|i| (i as f32) * -0.03).collect()).unwrap();
        t.push("gru.uz", &[4, 4], (0..16).map(|i| (i as f32) * 0.01).collect()).unwrap();
        t.push("gru.ur", &[4, 4], (0..16).map(|i| (i as f32) * 0.02).collect()).unwrap();
        t.push("gru.uh", &[4, 4], (0..16).map(|i| (i as f32) * -0.01).collect()).unwrap();
        t.push("gru.bz", &[4], vec![0.1; 4]).unwrap();
        t.push("gru.br", &[4], vec![0.2; 4]).unwrap();
        t.push("gru.bh", &[4], vec![0.0; 4]).unwrap();
        let mut a = BakedGru::from_tensors(&t).unwrap();
        let mut b = BakedGru::from_tensors(&t).unwrap();
        let xs = [
            vec![1.0, 0.5, -0.25],
            vec![0.0, 1.0, 0.0],
            vec![-1.0, 0.0, 0.75],
        ];
        for x in &xs {
            a.step(x);
            b.step(x);
        }
        assert_eq!(a.hidden(), b.hidden(), "GRU детерминирован: одинаковые шаги → одинаковое состояние");
        // reset возвращает в нули.
        a.reset();
        assert_eq!(a.hidden(), &[0.0; 4][..]);
        // Инвариант: НУЛЕВЫЕ веса и смещения, x=0 → скрытое остаётся нулевым.
        // Все ворота = σ(0) = 0.5, кандидат = tanh(0) = 0,
        // h' = (1−0.5)·0 + 0.5·h = 0.5·h → из нуля остаётся ноль.
        let mut t0 = BakedTensors::new();
        for name in ["gru.wz", "gru.wr", "gru.wh"] {
            t0.push(name, &[3, 2], vec![0.0; 6]).unwrap();
        }
        for name in ["gru.uz", "gru.ur", "gru.uh"] {
            t0.push(name, &[3, 3], vec![0.0; 9]).unwrap();
        }
        for name in ["gru.bz", "gru.br", "gru.bh"] {
            t0.push(name, &[3], vec![0.0; 3]).unwrap();
        }
        let mut g = BakedGru::from_tensors(&t0).unwrap();
        for _ in 0..5 {
            g.step(&[0.0, 0.0]);
        }
        assert_eq!(g.hidden(), &[0.0; 3][..], "нулевые веса и x=0 → скрытое остаётся нулевым");
    }

    #[test]
    fn gru_step_hand_computed_1_unit() {
        // GRU hidden=1, in=1, посчитано руками (f64-арифметика, комментарий):
        //   wz=[0.5], uz=[0.2], bz=[0.1]    wr=[-0.3], ur=[0.1], br=[-0.2]
        //   wh=[0.8], uh=[-0.4], bh=[0.05]
        // x=[1.0], h0=[0.0]:
        //   z = σ(0.5·1 + 0.2·0 + 0.1) = σ(0.6) = 0.645656306
        //   r = σ(−0.3·1 + 0.1·0 − 0.2) = σ(−0.5) = 0.377540669
        //   n = tanh(0.8·1 + (−0.4)·(r·0) + 0.05) = tanh(0.85) = 0.691069471
        //   h1 = (1−z)·n + z·h0 = 0.354343694·0.691069471 ≈ 0.244876
        let mut t = BakedTensors::new();
        t.push("gru.wz", &[1, 1], vec![0.5]).unwrap();
        t.push("gru.wr", &[1, 1], vec![-0.3]).unwrap();
        t.push("gru.wh", &[1, 1], vec![0.8]).unwrap();
        t.push("gru.uz", &[1, 1], vec![0.2]).unwrap();
        t.push("gru.ur", &[1, 1], vec![0.1]).unwrap();
        t.push("gru.uh", &[1, 1], vec![-0.4]).unwrap();
        t.push("gru.bz", &[1], vec![0.1]).unwrap();
        t.push("gru.br", &[1], vec![-0.2]).unwrap();
        t.push("gru.bh", &[1], vec![0.05]).unwrap();
        let mut g = BakedGru::from_tensors(&t).unwrap();
        g.step(&[1.0]);
        assert!(
            (g.hidden()[0] - 0.244_876).abs() < 1e-4,
            "h1 = {}, ожидалось ≈ 0.244876",
            g.hidden()[0]
        );
    }

    #[test]
    fn gru_from_tensors_validates() {
        // Неполный набор → Err.
        let t = BakedTensors::new();
        assert!(BakedGru::from_tensors(&t).is_err());
        // Несогласованные hidden → Err.
        let mut t = BakedTensors::new();
        t.push("gru.wz", &[4, 3], vec![0.0; 12]).unwrap();
        t.push("gru.wr", &[5, 3], vec![0.0; 15]).unwrap();
        t.push("gru.wh", &[4, 3], vec![0.0; 12]).unwrap();
        t.push("gru.uz", &[4, 4], vec![0.0; 16]).unwrap();
        t.push("gru.ur", &[4, 4], vec![0.0; 16]).unwrap();
        t.push("gru.uh", &[4, 4], vec![0.0; 16]).unwrap();
        t.push("gru.bz", &[4], vec![0.0; 4]).unwrap();
        t.push("gru.br", &[4], vec![0.0; 4]).unwrap();
        t.push("gru.bh", &[4], vec![0.0; 4]).unwrap();
        assert!(BakedGru::from_tensors(&t).is_err());
    }

    #[test]
    fn golden_prefix_bytes() {
        // Фиксированная структура: тензоры "a" [2] = [1.0, 2.0] и "b" [1,2] =
        // [0.5, −0.25]. Весь to_bytes детерминирован — сверяем префикс побайтово:
        //
        //   "DBW1"           44 42 57 31
        //   версия 1 (u16 LE) 01 00
        //   n=2 (u32 LE)      02 00 00 00
        //   a: len=1 'a'     01 61
        //   dtype 0          00
        //   ndim 1           01
        //   dims [2]         02 00 00 00
        //   1.0 f32 LE       00 00 80 3F
        //   2.0 f32 LE       00 00 00 40
        //   b: len=1 'b'     01 62
        //   dtype 0          00
        //   ndim 2           02
        //   dims [1,2]       01 00 00 00  02 00 00 00
        //   0.5 f32 LE       00 00 00 3F
        //   −0.25 f32 LE     00 00 80 BE
        let mut t = BakedTensors::new();
        t.push("a", &[2], vec![1.0, 2.0]).unwrap();
        t.push("b", &[1, 2], vec![0.5, -0.25]).unwrap();
        let expected: Vec<u8> = vec![
            0x44, 0x42, 0x57, 0x31, // "DBW1"
            0x01, 0x00, // версия 1
            0x02, 0x00, 0x00, 0x00, // 2 тензора
            0x01, 0x61, // "a"
            0x00, // dtype f32
            0x01, // ndim 1
            0x02, 0x00, 0x00, 0x00, // dims [2]
            0x00, 0x00, 0x80, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x40, // 2.0
            0x01, 0x62, // "b"
            0x00, // dtype f32
            0x02, // ndim 2
            0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, // dims [1,2]
            0x00, 0x00, 0x00, 0x3F, // 0.5
            0x00, 0x00, 0x80, 0xBE, // −0.25
        ];
        assert_eq!(t.to_bytes(), expected, "золотой префикс DBW1");
    }

    #[test]
    fn get_returns_none_for_missing() {
        let t = mlp_fixture();
        assert_eq!(t.get("нет_такого"), None);
        assert_eq!(t.dims("нет_такого"), None);
        assert_eq!(t.get("layer0.w"), Some(&[0.5, -1.0][..]));
    }
}
