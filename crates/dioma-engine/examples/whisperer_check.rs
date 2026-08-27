//! # whisperer_check — паритет Python↔Rust + качество «Шептуна»
//!
//! Пайплайн «Python учит ↔ Rust играет» в действии на РЕАЛЬНЫХ данных:
//!
//! 1. **ПАРИТЕТ**: грузит `whisperer.dbw` ([`BakedTensors::from_bytes`] →
//!    [`BakedGru`] + [`BakedMlp`]) и 5 фиксированных окон из
//!    `whisperer_golden.json` (python-золото, f64). Те же входы гонятся через
//!    Rust-инференс (f32), выходы сравниваются: `max|Δ| ≤ 1e-4` — жёсткий
//!    ассерт. Допуск документирован: f32 против f64, 16 шагов GRU — ошибка
//!    округления ~1e-6..1e-5, 1e-4 даёт запас на порядок.
//! 2. **КАЧЕСТВО**: полные hold-out записи (seed0+6, seed0+7) прогоняются
//!    через запечённый GRU СОСТОЯТЕЛЬНО (входы подряд, скрытое состояние
//!    переносится) и считаются те же метрики, что у тренера: MAE(steer),
//!    MAE(drive), точность eat, Пирсон r(steer)/r(drive). Смоук-пол
//!    `r(steer) > 0.3` — МЯГКИЙ: нарушение печатается в отчёт, но не валит
//!    пример (пол — ориентир, факт — в отчёте).
//!
//! Запуск (из `crates/dioma-engine`):
//!
//! ```text
//! cargo run --release --example whisperer_check -- \
//!   --data ../../tools/whisperer_data.json \
//!   --dbw ../../tools/whisperer.dbw \
//!   --golden ../../tools/whisperer_golden.json
//! ```
//!
//! JSON здесь разбирается МИНИМАЛЬНЫМ ручным парсером (ниже): зависимостей
//! у примера нет (ТЗ: «No new RUST deps»), а формат файлов контролируется
//! нашими же писателями (рекордер `whisperer_data.rs`, тренер
//! `train_whisperer.py`).

use dioma_engine::engine::baked::{BakedGru, BakedMlp, BakedTensors};
use std::fs;
use std::path::PathBuf;

// ─── Крошечный JSON-парсер (только для наших двух файлов) ──────────────────

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
    fn as_str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
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
                            // \uXXXX — из json.dump для не-ASCII; наши ключи ASCII,
                            // поэтому кодируем как UTF-8 посимвольно.
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

const IN_DIM: usize = 12;
const OUT_DIM: usize = 3;
/// Точность паритета f32↔f64 (см. доки файла).
const PARITY_TOL: f64 = 1e-4;
/// Смоук-пол качества: r(steer) на hold-out (мягкий, см. доки файла).
const QUALITY_R_FLOOR: f32 = 0.3;

struct CreatureData {
    seed: u64,
    /// Каждая строка: [12 входов, steer, drive, eat].
    samples: Vec<[f32; IN_DIM + OUT_DIM]>,
}

struct GoldenWindow {
    seed: u64,
    start: usize,
    /// Выходы после каждого из 16 шагов: [steer, drive, eat] (python, f64).
    outputs: Vec<[f64; OUT_DIM]>,
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
            let nums = row.as_arr().ok_or("строка сэмпла — не массив")?;
            if nums.len() != IN_DIM + OUT_DIM {
                return Err(format!("строка длины {} (ожидалось {})", nums.len(), IN_DIM + OUT_DIM));
            }
            let mut r = [0.0f32; IN_DIM + OUT_DIM];
            for (i, n) in nums.iter().enumerate() {
                r[i] = n.as_f64().ok_or("число в сэмпле")? as f32;
            }
            samples.push(r);
        }
        out.push(CreatureData { seed, samples });
    }
    Ok(out)
}

fn load_golden(g: &J) -> Result<Vec<GoldenWindow>, String> {
    let mut out = Vec::new();
    for w in g
        .get("windows")
        .and_then(|w| w.as_arr())
        .ok_or("нет windows[]")?
    {
        let seed = w.get("seed").and_then(|s| s.as_f64()).ok_or("нет seed")? as u64;
        let start = w.get("start").and_then(|s| s.as_f64()).ok_or("нет start")? as usize;
        let outs_j = w.get("outputs").and_then(|o| o.as_arr()).ok_or("нет outputs[]")?;
        let mut outputs = Vec::with_capacity(outs_j.len());
        for step in outs_j {
            let nums = step.as_arr().ok_or("выход шага — не массив")?;
            if nums.len() != OUT_DIM {
                return Err(format!("выход длины {} (ожидалось {})", nums.len(), OUT_DIM));
            }
            let mut o = [0.0f64; OUT_DIM];
            for (i, n) in nums.iter().enumerate() {
                o[i] = n.as_f64().ok_or("число в выходе")?;
            }
            outputs.push(o);
        }
        out.push(GoldenWindow { seed, start, outputs });
    }
    Ok(out)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Полный шаг запечённого шептуна: GRU-ячейка + линейный readout + активации.
fn baked_step(gru: &mut BakedGru, mlp: &BakedMlp, x: &[f32]) -> [f32; OUT_DIM] {
    gru.step(x);
    let y = mlp.forward(gru.hidden());
    [y[0].tanh(), sigmoid(y[1]), sigmoid(y[2])]
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

fn main() {
    // ── Аргументы ──
    let mut data_path = PathBuf::from("tools/whisperer_data.json");
    let mut dbw_path = PathBuf::from("tools/whisperer.dbw");
    let mut golden_path = PathBuf::from("tools/whisperer_golden.json");
    let mut seed0 = 1000u64;
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
                    "whisperer_check — паритет Python↔Rust + качество «Шептуна»\n\
                     \n\
                     Использование:\n  \
                     cargo run --release --example whisperer_check -- [--data PATH] [--dbw PATH] [--golden PATH] [--seed0 N]\n\
                     \n\
                     Параметры:\n  \
                     --data PATH   датасет рекордера (по умолчанию tools/whisperer_data.json)\n  \
                     --dbw PATH    запечённые веса (по умолчанию tools/whisperer.dbw)\n  \
                     --golden PATH python-золото паритета (по умолчанию tools/whisperer_golden.json)\n  \
                     --seed0 N     базовый сид датасета (по умолчанию 1000)"
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
    let gold_windows = load_golden(&golden).unwrap_or_else(|e| {
        eprintln!("золото: {e}");
        std::process::exit(2);
    });

    let tensors = BakedTensors::from_bytes(&dbw_bytes).unwrap_or_else(|e| {
        eprintln!("DBW1: {e}");
        std::process::exit(2);
    });
    let mut gru = BakedGru::from_tensors(&tensors).unwrap_or_else(|e| {
        eprintln!("BakedGru: {e}");
        std::process::exit(2);
    });
    let mlp = BakedMlp::from_tensors(&tensors).unwrap_or_else(|e| {
        eprintln!("BakedMlp: {e}");
        std::process::exit(2);
    });
    assert_eq!(gru.in_dim(), IN_DIM, "ожидался GRU с входом {IN_DIM}");
    assert_eq!(mlp.in_dim(), gru.hidden_dim(), "readout 32→3 не состыкован с GRU");
    assert_eq!(mlp.out_dim(), OUT_DIM, "ожидался readout с выходом {OUT_DIM}");
    println!(
        "DBW1: {} тензоров, GRU {IN_DIM}→{} → MLP {}→{} ({} байт)",
        tensors.names().len(),
        gru.hidden_dim(),
        mlp.in_dim(),
        mlp.out_dim(),
        dbw_bytes.len()
    );

    // ── 1. ПАРИТЕТ: те же окна, что в whisperer_golden.json ──
    println!("\nПАРИТЕТ python(f64) ↔ rust(f32): 5 окон × 16 шагов, допуск |Δ| ≤ {PARITY_TOL}");
    let mut max_delta = 0.0f64;
    for w in &gold_windows {
        let cd = creatures
            .iter()
            .find(|c| c.seed == w.seed)
            .unwrap_or_else(|| panic!("нет особи с сидом {} в датасете", w.seed));
        if w.start + w.outputs.len() > cd.samples.len() {
            panic!("окно {}-{} выходит за {} сэмплов", w.seed, w.start, cd.samples.len());
        }
        gru.reset();
        let mut win_delta = 0.0f64;
        for (k, gold) in w.outputs.iter().enumerate() {
            let row = &cd.samples[w.start + k];
            let pred = baked_step(&mut gru, &mlp, &row[..IN_DIM]);
            for j in 0..OUT_DIM {
                let d = (pred[j] as f64 - gold[j]).abs();
                win_delta = win_delta.max(d);
                max_delta = max_delta.max(d);
            }
        }
        println!("  окно seed {} start {}: max|Δ| = {win_delta:.2e}", w.seed, w.start);
    }
    assert!(
        max_delta <= PARITY_TOL,
        "ПАРИТЕТ НАРУШЕН: max|Δ| = {max_delta:.3e} > {PARITY_TOL} — python и rust считают по-разному!"
    );
    println!("  ПАРИТЕТ OK: max|Δ| = {max_delta:.3e} ≤ {PARITY_TOL}");

    // ── 2. КАЧЕСТВО: полные hold-out записи, состоятельный прогон ──
    let holdout_seeds: Vec<u64> = (seed0 + 6..seed0 + 8).collect();
    println!("\nКАЧЕСТВО (rust, запечённый GRU, hold-out {holdout_seeds:?}):");
    let mut per_r_s = Vec::new();
    let mut per_r_d = Vec::new();
    let mut per_mae_s = Vec::new();
    let mut per_mae_d = Vec::new();
    let mut per_eat = Vec::new();
    for seed in &holdout_seeds {
        let cd = creatures
            .iter()
            .find(|c| c.seed == *seed)
            .unwrap_or_else(|| panic!("нет hold-out особи с сидом {seed}"));
        gru.reset();
        let mut pred_steer = Vec::with_capacity(cd.samples.len());
        let mut true_steer = Vec::with_capacity(cd.samples.len());
        let mut pred_drive = Vec::with_capacity(cd.samples.len());
        let mut true_drive = Vec::with_capacity(cd.samples.len());
        let mut pred_eat = Vec::with_capacity(cd.samples.len());
        let mut true_eat = Vec::with_capacity(cd.samples.len());
        for row in &cd.samples {
            let pred = baked_step(&mut gru, &mlp, &row[..IN_DIM]);
            pred_steer.push(pred[0]);
            true_steer.push(row[IN_DIM]);
            pred_drive.push(pred[1]);
            true_drive.push(row[IN_DIM + 1]);
            pred_eat.push(if pred[2] > 0.5 { 1.0 } else { 0.0 });
            true_eat.push(row[IN_DIM + 2]);
        }
        let n = cd.samples.len();
        let mae_s = pred_steer
            .iter()
            .zip(&true_steer)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n as f32;
        let mae_d = pred_drive
            .iter()
            .zip(&true_drive)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / n as f32;
        let eat_acc = pred_eat
            .iter()
            .zip(&true_eat)
            .filter(|(a, b)| a == b)
            .count() as f32
            / n as f32;
        let r_s = pearson(&pred_steer, &true_steer);
        let r_d = pearson(&pred_drive, &true_drive);
        per_r_s.push(r_s);
        per_r_d.push(r_d);
        per_mae_s.push(mae_s);
        per_mae_d.push(mae_d);
        per_eat.push(eat_acc);
        println!(
            "  c{seed}: n={n:5}  MAE(steer)={mae_s:.4}  MAE(drive)={mae_d:.4}  eat_acc={eat_acc:.3}  r(steer)={r_s:+.4}  r(drive)={r_d:+.4}"
        );
    }
    // Сводка — СРЕДНЕЕ по особям (та же формула, что у тренера в
    // whisperer_report.txt). Почему среднее: корреляция «внутри мозга» —
    // честная мера «модель следит за мозгом»; пул по сэмплам смешивает
    // разные средние курсов особей.
    let mae_s = per_mae_s.iter().sum::<f32>() / per_mae_s.len() as f32;
    let mae_d = per_mae_d.iter().sum::<f32>() / per_mae_d.len() as f32;
    let eat_acc = per_eat.iter().sum::<f32>() / per_eat.len() as f32;
    let r_s = per_r_s.iter().sum::<f32>() / per_r_s.len() as f32;
    let r_d = per_r_d.iter().sum::<f32>() / per_r_d.len() as f32;
    println!(
        "  СВОДКА hold-out (среднее по особям): n=2 особи  MAE(steer)={mae_s:.4}  MAE(drive)={mae_d:.4}  eat_acc={eat_acc:.3}  r(steer)={r_s:+.4}  r(drive)={r_d:+.4}"
    );
    if r_s > QUALITY_R_FLOOR {
        println!("  КАЧЕСТВО OK: r(steer) = {r_s:+.4} > {QUALITY_R_FLOOR} (смоук-пол)");
    } else {
        // Мягкий пол: нарушение печатается, но пример не валится — честный
        // факт уходит в отчёт (см. доки файла).
        eprintln!(
            "  ВНИМАНИЕ: r(steer) = {r_s:+.4} ≤ {QUALITY_R_FLOOR} — мягкий пол не взят; фактическое значение в отчёте"
        );
    }
}
