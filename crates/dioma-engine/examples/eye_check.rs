//! # eye_check — паритет Python↔Rust + качество «Глаз»
//!
//! Пайплайн «Python учит ↔ Rust играет» в действии на энкодере ретины:
//!
//! 1. **ПАРИТЕТ**: грузит `eye.dbw` ([`BakedTensors::from_bytes`] →
//!    [`BakedMlp`]) и 5 фиксированных сэмплов из `eye_golden.json`
//!    (python-золото, f64; только сэмплы с ненулевой ретиной — «глаза
//!    открыты», нелинейный режим сети). Те же входы гонятся через
//!    Rust-инференс (f32), выходы сравниваются: `max|Δ| ≤ 1e-4` — жёсткий
//!    ассерт. Допуск документирован: f32 против f64 (веса округлены в f32
//!    в DBW1, вход f64→f32) — ошибка ~1e-6..1e-5, 1e-4 даёт запас.
//! 2. **КАЧЕСТВО**: полные hold-out записи (seed0+6, seed0+7) прогоняются
//!    через запечённый MLP ПОФАЙЛОВО (каждый сэмпл независим — сеть без
//!    состояния) и считаются те же метрики, что у тренера: per-output
//!    MAE + Пирсон r. Считаются ДВАЖДЫ: по всем сэмплам и по «бодрым»
//!    (ретина ненулевая) — случайный мозг спит ~75% тиков, там ретина и
//!    цели нулевые («глаза закрыты»), и честная мера геометрии — бодрые.
//!    Смоук-ориентир (МЯГКИЙ, пример не валит): r > 0.9 для vis_l/vis_r —
//!    из ТЗ; факт печатается в отчёт в любом случае.
//!
//! Запуск из КОРНЯ репозитория (пути `tools/…` — как в ТЗ):
//!
//! ```text
//! cargo run --manifest-path crates/dioma-engine/Cargo.toml --release --example eye_check -- \
//!   --data tools/eye_data.json \
//!   --dbw tools/eye.dbw \
//!   --golden tools/eye_golden.json
//! ```
//!
//! (или из `crates/dioma-engine` с путями `../../tools/…`).
//!
//! JSON здесь разбирается МИНИМАЛЬНЫМ ручным парсером (ниже): зависимостей
//! у примера нет (ТЗ: «No new RUST deps»), а формат файлов контролируется
//! нашими же писателями (рекордер `eye_data.rs`, тренер `train_eye.py`).

use dioma_engine::engine::baked::{BakedMlp, BakedTensors};
use std::fs;
use std::path::PathBuf;

// ─── Крошечный JSON-парсер (тот же подход, что в whisperer_check.rs) ───────

#[derive(Debug, Clone)]
#[allow(dead_code)] // Str/Null/Bool — парсер производит, а читаем только часть
enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            J::Num(x) => Some(*x),
            _ => None,
        }
    }
    fn as_arr(&self) -> Option<&Vec<J>> {
        match self {
            J::Arr(a) => Some(a),
            _ => None,
        }
    }
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn new(s: &'a str) -> Self {
        P { b: s.as_bytes(), i: 0 }
    }
    /// Пропуск JSON-пробелов — ТОЛЬКО 4 ASCII-символа. Нельзя смотреть на
    /// байт как на char: байт 0x85 внутри UTF-8 («х») — это U+0085 (NEL),
    /// который Unicode считает пробелом, и ws() съел бы середину строки.
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.b.get(self.i).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        self.ws();
        let c = self.b.get(self.i).copied();
        if c.is_some() {
            self.i += 1;
        }
        c
    }
    fn expect(&mut self, c: u8) -> Result<(), String> {
        match self.bump() {
            Some(x) if x == c => Ok(()),
            other => Err(format!(
                "JSON: ожидался «{}», найден {:?}",
                c as char,
                other.map(|x| x as char)
            )),
        }
    }
    fn word(&mut self, w: &str) -> Result<(), String> {
        for c in w.bytes() {
            self.expect(c)?;
        }
        Ok(())
    }
    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = self
                .bump()
                .ok_or_else(|| "JSON: незакрытая строка".to_string())?;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self
                        .bump()
                        .ok_or_else(|| "JSON: битый escape".to_string())?;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(0x08),
                        b'f' => out.push(0x0C),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let mut cp = 0u32;
                            for _ in 0..4 {
                                let h = self
                                    .bump()
                                    .ok_or_else(|| "JSON: битый \\u".to_string())?;
                                cp = cp * 16
                                    + (h as char).to_digit(16).ok_or_else(|| "JSON: \\u не hex".to_string())?;
                            }
                            let ch = char::from_u32(cp).ok_or_else(|| "JSON: \\u вне юникода".to_string())?;
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        other => return Err(format!("JSON: неизвестный escape \\{}", other as char)),
                    }
                }
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| "JSON: строка не utf8".to_string())
    }
    fn number(&mut self) -> Result<f64, String> {
        let mut tok = String::new();
        while let Some(c) = self.peek() {
            if (c as char).is_ascii_digit() || c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' {
                tok.push(c as char);
                self.i += 1;
            } else {
                break;
            }
        }
        tok.parse::<f64>()
            .map_err(|_| format!("JSON: не число: {tok:?}"))
    }
    fn value(&mut self) -> Result<J, String> {
        match self.peek() {
            Some(b'{') => {
                self.bump();
                let mut kv = Vec::new();
                if self.peek() == Some(b'}') {
                    self.bump();
                    return Ok(J::Obj(kv));
                }
                loop {
                    let k = self.string()?;
                    self.expect(b':')?;
                    let v = self.value()?;
                    kv.push((k, v));
                    match self.bump() {
                        Some(b',') => continue,
                        Some(b'}') => break,
                        other => {
                            return Err(format!("JSON: ожидался , или }}, найден {:?}", other.map(|x| x as char)))
                        }
                    }
                }
                Ok(J::Obj(kv))
            }
            Some(b'[') => {
                self.bump();
                let mut arr = Vec::new();
                if self.peek() == Some(b']') {
                    self.bump();
                    return Ok(J::Arr(arr));
                }
                loop {
                    arr.push(self.value()?);
                    match self.bump() {
                        Some(b',') => continue,
                        Some(b']') => break,
                        other => {
                            return Err(format!("JSON: ожидался , или ], найден {:?}", other.map(|x| x as char)))
                        }
                    }
                }
                Ok(J::Arr(arr))
            }
            Some(b'"') => self.string().map(J::Str),
            Some(b't') => {
                self.word("true")?;
                Ok(J::Bool(true))
            }
            Some(b'f') => {
                self.word("false")?;
                Ok(J::Bool(false))
            }
            Some(b'n') => {
                self.word("null")?;
                Ok(J::Null)
            }
            Some(c) if c.is_ascii_digit() || c == b'-' => self.number().map(J::Num),
            other => Err(format!("JSON: неожиданный символ {:?}", other.map(|x| x as char))),
        }
    }
    fn parse(mut self) -> Result<J, String> {
        let v = self.value()?;
        self.ws();
        if self.i != self.b.len() {
            return Err("JSON: мусор после значения".to_string());
        }
        Ok(v)
    }
}

fn parse_json(s: &str) -> Result<J, String> {
    P::new(s).parse()
}

// ─── Структуры данных ──────────────────────────────────────────────────────

const IN_DIM: usize = 48;
const OUT_DIM: usize = 6;
/// Точность паритета f32↔f64 (см. доки файла).
const PARITY_TOL: f64 = 1e-4;
/// Имена выходов (порядок — как в out_names рекордера/тренера).
const OUT_NAMES: [&str; OUT_DIM] = [
    "vis_l",
    "vis_r",
    "food_bearing_sin",
    "food_bearing_cos",
    "food_dist_norm",
    "wall_ahead",
];

struct CreatureData {
    seed: u64,
    /// Каждый сэмпл: ([48 ретины], [6 целей]).
    samples: Vec<([f32; IN_DIM], [f32; OUT_DIM])>,
}

struct GoldenSample {
    seed: u64,
    start: usize,
    input: [f64; IN_DIM],
    output: [f64; OUT_DIM],
}

fn arr_to_f32<const N: usize>(nums: &[J], what: &str) -> Result<[f32; N], String> {
    if nums.len() != N {
        return Err(format!("{what}: длина {} (ожидалось {N})", nums.len()));
    }
    let mut r = [0.0f32; N];
    for (i, n) in nums.iter().enumerate() {
        r[i] = n.as_f64().ok_or(format!("{what}: не число в [{i}]"))? as f32;
    }
    Ok(r)
}

fn arr_to_f64<const N: usize>(nums: &[J], what: &str) -> Result<[f64; N], String> {
    if nums.len() != N {
        return Err(format!("{what}: длина {} (ожидалось {N})", nums.len()));
    }
    let mut r = [0.0f64; N];
    for (i, n) in nums.iter().enumerate() {
        r[i] = n.as_f64().ok_or(format!("{what}: не число в [{i}]"))?;
    }
    Ok(r)
}

fn load_creatures(data: &J) -> Result<Vec<CreatureData>, String> {
    let mut out = Vec::new();
    for c in data
        .get("creatures")
        .and_then(|c| c.as_arr())
        .ok_or("нет creatures[]")?
    {
        let seed = c.get("seed").and_then(|s| s.as_f64()).ok_or("нет seed")? as u64;
        let samples_j = c.get("samples").and_then(|s| s.as_arr()).ok_or("нет samples[]")?;
        let mut samples = Vec::with_capacity(samples_j.len());
        for row in samples_j {
            // Формат рекордера: [[48 ретины], [6 целей]].
            let parts = row.as_arr().ok_or("сэмпл — не массив из двух частей")?;
            if parts.len() != 2 {
                return Err(format!("сэмпл: {} части (ожидалось 2)", parts.len()));
            }
            let retina = parts[0].as_arr().ok_or("ретина — не массив")?;
            let targets = parts[1].as_arr().ok_or("цели — не массив")?;
            samples.push((
                arr_to_f32::<IN_DIM>(retina, "ретина")?,
                arr_to_f32::<OUT_DIM>(targets, "цели")?,
            ));
        }
        out.push(CreatureData { seed, samples });
    }
    Ok(out)
}

fn load_golden(g: &J) -> Result<Vec<GoldenSample>, String> {
    let mut out = Vec::new();
    for s in g
        .get("samples")
        .and_then(|s| s.as_arr())
        .ok_or("нет samples[]")?
    {
        let seed = s.get("seed").and_then(|s| s.as_f64()).ok_or("нет seed")? as u64;
        let start = s.get("start").and_then(|s| s.as_f64()).ok_or("нет start")? as usize;
        let input = s.get("input").and_then(|s| s.as_arr()).ok_or("нет input[]")?;
        let output = s.get("output").and_then(|s| s.as_arr()).ok_or("нет output[]")?;
        out.push(GoldenSample {
            seed,
            start,
            input: arr_to_f64::<IN_DIM>(input, "input")?,
            output: arr_to_f64::<OUT_DIM>(output, "output")?,
        });
    }
    Ok(out)
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let ma = a.iter().sum::<f32>() / n as f32;
    let mb = b.iter().sum::<f32>() / n as f32;
    let (mut cov, mut va, mut vb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va < 1e-12 || vb < 1e-12 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

fn mae(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().max(1);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / n as f32
}

fn main() {
    // ── Аргументы ──
    let mut data_path = PathBuf::from("tools/eye_data.json");
    let mut dbw_path = PathBuf::from("tools/eye.dbw");
    let mut golden_path = PathBuf::from("tools/eye_golden.json");
    let mut seed0 = 2000u64;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| {
            eprintln!("аргумент {a} требует значение");
            std::process::exit(2);
        });
        match a.as_str() {
            "--data" => data_path = PathBuf::from(val()),
            "--dbw" => dbw_path = PathBuf::from(val()),
            "--golden" => golden_path = PathBuf::from(val()),
            "--seed0" => seed0 = val().parse().expect("--seed0: базовый сид"),
            "--help" | "-h" => {
                println!(
                    "eye_check — паритет Python↔Rust + качество «Глаз»\n\
                     \n\
                     Использование:\n  \
                     cargo run --manifest-path crates/dioma-engine/Cargo.toml --release --example eye_check -- [--data PATH] [--dbw PATH] [--golden PATH] [--seed0 N]\n\
                     \n\
                     Параметры:\n  \
                     --data PATH   датасет рекордера (по умолчанию tools/eye_data.json)\n  \
                     --dbw PATH    запечённые веса (по умолчанию tools/eye.dbw)\n  \
                     --golden PATH python-золото паритета (по умолчанию tools/eye_golden.json)\n  \
                     --seed0 N     базовый сид датасета (по умолчанию 2000)"
                );
                return;
            }
            other => {
                eprintln!("неизвестный аргумент: {other}");
                std::process::exit(2);
            }
        }
    }

    // ── Загрузка ──
    let data_text = fs::read_to_string(&data_path).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {data_path:?}: {e}");
        std::process::exit(2);
    });
    let dbw_bytes = fs::read(&dbw_path).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {dbw_path:?}: {e}");
        std::process::exit(2);
    });
    let golden_text = fs::read_to_string(&golden_path).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {golden_path:?}: {e}");
        std::process::exit(2);
    });

    let data = parse_json(&data_text).unwrap_or_else(|e| {
        eprintln!("битый датасет: {e}");
        std::process::exit(2);
    });
    let golden = parse_json(&golden_text).unwrap_or_else(|e| {
        eprintln!("битое золото: {e}");
        std::process::exit(2);
    });
    let creatures = load_creatures(&data).unwrap_or_else(|e| {
        eprintln!("датасет: {e}");
        std::process::exit(2);
    });
    let gold_samples = load_golden(&golden).unwrap_or_else(|e| {
        eprintln!("золото: {e}");
        std::process::exit(2);
    });

    let tensors = BakedTensors::from_bytes(&dbw_bytes).unwrap_or_else(|e| {
        eprintln!("DBW1: {e}");
        std::process::exit(2);
    });
    let mlp = BakedMlp::from_tensors(&tensors).unwrap_or_else(|e| {
        eprintln!("BakedMlp: {e}");
        std::process::exit(2);
    });
    assert_eq!(mlp.in_dim(), IN_DIM, "ожидался MLP с входом {IN_DIM}");
    assert_eq!(mlp.out_dim(), OUT_DIM, "ожидался MLP с выходом {OUT_DIM}");
    // Скрытый размер — из layer0.w [hidden, in] (формат тренера).
    let hidden = tensors
        .dims("layer0.w")
        .and_then(|d| d.first().copied())
        .unwrap_or(0);
    println!(
        "DBW1: {} тензоров, MLP {IN_DIM}→{hidden}→{OUT_DIM} ({} байт)",
        tensors.names().len(),
        dbw_bytes.len()
    );
    // Статистика живости ретины из заголовка датасета (для отчёта).
    if let Some(stats) = data.get("stats") {
        println!(
            "датасет: сэмплов {:.0}, ретина не-нулевая: ягодность {:.3}, стенность {:.3}, соседность {:.3}",
            stats.get("samples").and_then(|s| s.as_f64()).unwrap_or(0.0),
            stats.get("retina_nonzero_berry").and_then(|s| s.as_f64()).unwrap_or(0.0),
            stats.get("retina_nonzero_wall").and_then(|s| s.as_f64()).unwrap_or(0.0),
            stats.get("retina_nonzero_neighbor").and_then(|s| s.as_f64()).unwrap_or(0.0),
        );
    }

    // ── 1. ПАРИТЕТ: те же сэмплы, что в eye_golden.json ──
    println!("\nПАРИТЕТ python(f64) ↔ rust(f32): {} сэмплов, допуск |Δ| ≤ {PARITY_TOL}", gold_samples.len());
    let mut max_delta = 0.0f64;
    for s in &gold_samples {
        let mut x = [0.0f32; IN_DIM];
        for i in 0..IN_DIM {
            x[i] = s.input[i] as f32;
        }
        let pred = mlp.forward(&x);
        let mut d = 0.0f64;
        for k in 0..OUT_DIM {
            d = d.max((pred[k] as f64 - s.output[k]).abs());
        }
        max_delta = max_delta.max(d);
        println!("  сэмпл seed {} start {}: max|Δ| = {d:.2e}", s.seed, s.start);
    }
    assert!(
        max_delta <= PARITY_TOL,
        "ПАРИТЕТ НАРУШЕН: max|Δ| = {max_delta:.3e} > {PARITY_TOL} — python и rust считают по-разному!"
    );
    println!("  ПАРИТЕТ OK: max|Δ| = {max_delta:.3e} ≤ {PARITY_TOL}");

    // ── 2. КАЧЕСТВО: полные hold-out записи через запечённый MLP ──
    let holdout_seeds: Vec<u64> = (seed0 + 6..seed0 + 8).collect();
    println!("\nКАЧЕСТВО (rust, запечённый MLP, hold-out {holdout_seeds:?}):");

    // Пофайлово по каждой особи — только метрики (для совпадения с тренером
    // считается ПУЛ по сэмплам; пер-особные строки печатаем для наглядности).
    let mut per_creature = Vec::new();
    let mut preds: Vec<Vec<f32>> = vec![Vec::new(); OUT_DIM];
    let mut trues: Vec<Vec<f32>> = vec![Vec::new(); OUT_DIM];
    let mut preds_a: Vec<Vec<f32>> = vec![Vec::new(); OUT_DIM];
    let mut trues_a: Vec<Vec<f32>> = vec![Vec::new(); OUT_DIM];
    let mut n_all = 0usize;
    let mut n_awake = 0usize;
    for seed in &holdout_seeds {
        let cd = creatures
            .iter()
            .find(|c| c.seed == *seed)
            .unwrap_or_else(|| panic!("нет hold-out особи с сидом {seed}"));
        let n = cd.samples.len();
        // Пер-особные per-output метрики (все сэмплы).
        let mut p = vec![Vec::with_capacity(n); OUT_DIM];
        let mut t = vec![Vec::with_capacity(n); OUT_DIM];
        for sample in &cd.samples {
            let pred = mlp.forward(&sample.0);
            for k in 0..OUT_DIM {
                preds[k].push(pred[k]);
                trues[k].push(sample.1[k]);
                p[k].push(pred[k]);
                t[k].push(sample.1[k]);
            }
            n_all += 1;
            if sample.0.iter().any(|&v| v != 0.0) {
                for k in 0..OUT_DIM {
                    preds_a[k].push(pred[k]);
                    trues_a[k].push(sample.1[k]);
                }
                n_awake += 1;
            }
        }
        let mut line = format!("  c{seed}: n={n}");
        for k in 0..OUT_DIM {
            line.push_str(&format!(
                "  {}: MAE={:.4} r={:+.4}",
                OUT_NAMES[k],
                mae(&p[k], &t[k]),
                pearson(&p[k], &t[k])
            ));
        }
        per_creature.push(line);
    }
    for line in &per_creature {
        println!("{line}");
    }

    println!("\n  ПУЛ по hold-out (все сэмплы, n={n_all}) и по бодрым (n={n_awake}):");
    println!("  {:<16}  {:>10}  {:>9}  {:>10}  {:>9}", "выход", "MAE(все)", "r(все)", "MAE(бодрые)", "r(бодрые)");
    let mut expect_ok = true;
    for k in 0..OUT_DIM {
        let ma = mae(&preds[k], &trues[k]);
        let ra = pearson(&preds[k], &trues[k]);
        let mb = mae(&preds_a[k], &trues_a[k]);
        let rb = pearson(&preds_a[k], &trues_a[k]);
        println!(
            "  {:<16}  {:>10.4}  {:>+9.4}  {:>10.4}  {:>+9.4}",
            OUT_NAMES[k], ma, ra, mb, rb
        );
        // Ориентир ТЗ: r > 0.9 для зрительных/еда-выходов (все сэмплы).
        if matches!(OUT_NAMES[k], "vis_l" | "vis_r" | "food_bearing_sin" | "food_bearing_cos") && ra < 0.9 {
            expect_ok = false;
        }
    }
    if expect_ok {
        println!("  ОРИЕНТИР ТЗ (r>0.9 на vis_l/vis_r/food_*) — взят (все сэмплы)");
    } else {
        // Мягкий ориентир: нарушение печатается, но пример не валится —
        // честный факт уходит в отчёт (см. доки файла).
        eprintln!("  ВНИМАНИЕ: не все r(vis/food) > 0.9 на hold-out (все сэмплы) — фактическое значение в таблице; паритет остаётся жёстким ассертом");
    }
}
