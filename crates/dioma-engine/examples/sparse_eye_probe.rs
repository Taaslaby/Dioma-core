//! # sparse_eye_probe — «КОНЦЕПТ-ГЛАЗ»: пиксели → разреженная популяция
//!
//! В отличие от mnist_probe (классификатор «вижу цифру N» — один горячий
//! логит), здесь глаз — РАЗРЕЖЕННЫЙ АВТОЭНКОДЕР 784→128→784: 128-мерный код
//! — распределённая популяция, активны ~10% нейронов (как спайки в коре).
//! Одинаковые цифры → одинаковые популяции, разные → непересекающиеся.
//!
//! 1. **ПАРИТЕТ** (жёсткий ассерт): `sparse_eye.dbw` против
//!    `sparse_eye_golden.json` (python f64): и КОД (128), и реконструкция
//!    (784) с допуском `max|Δ| ≤ 1e-4`.
//! 2. **КОНЦЕПТ-КАЧЕСТВО** (hold-out 1000, тот же `mnist_holdout.json`):
//!    разреженность, реконструкция MSE, nearest-centroid accuracy
//!    (код → класс БЕЗ лейбла), матрица близости прототипов 10×10.
//! 3. **СПАЙКИ**: для каждой цифры печатается АКТИВНАЯ ПОПУЛЯЦИЯ
//!    (номера возбуждённых концепт-нейронов) + ASCII-арт оригинала и
//!    реконструкции; Jaccard-пересечение популяций одной цифры (высокое)
//!    против разных (низкое) — код ведёт себя как кора, а не как лейбл.
//!
//! Запуск из КОРНЯ репозитория:
//!
//! ```text
//! python tools/train_sparse_eye.py
//! cargo run --manifest-path crates/dioma-engine/Cargo.toml --release --example sparse_eye_probe
//! ```
//!
//! В игре код инжектится током в зрительные регионы спайкового мозга:
//! активные единицы = спайковые источники. JSON — ручной парсер, как в
//! eye_check/mnist_probe (без зависимостей).

use dioma_engine::brain::baked::{BakedMlp, BakedTensors};
use std::fs;
use std::path::PathBuf;

// ─── Крошечный JSON-парсер (тот же, что в eye_check.rs) ────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
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
                            let ch = char::from_u32(cp)
                                .ok_or_else(|| "JSON: \\u вне юникода".to_string())?;
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

// ─── Данные ────────────────────────────────────────────────────────────────

const IN_DIM: usize = 784;
const CODE_DIM: usize = 128;
const OUT_DIM: usize = 10;
const PARITY_TOL: f64 = 1e-4;
/// Порог «нейрон активен» (|h| > th) — тот же, что в тренере.
const ACTIVE_TH: f32 = 0.05;
/// Мягкий ориентир качества (нарушение НЕ валит пример): unsupervised-код
/// без лейблов, реалистично ~0.85 (см. tools/sparse_eye_report.txt).
const SOFT_CENTROID: f64 = 0.80;

struct Gold {
    label: usize,
    index: usize,
    input: Vec<f64>,
    code: Vec<f64>,
    recon: Vec<f64>,
}

struct Sample {
    label: usize,
    index: usize,
    input: Vec<f32>,
}

fn load_golden(g: &J) -> Result<Vec<Gold>, String> {
    let mut out = Vec::new();
    for s in g
        .get("samples")
        .and_then(|s| s.as_arr())
        .ok_or("нет samples[]")?
    {
        let label = s.get("label").and_then(|v| v.as_f64()).ok_or("нет label")? as usize;
        let index = s.get("index").and_then(|v| v.as_f64()).ok_or("нет index")? as usize;
        let input = s
            .get("input")
            .and_then(|v| v.as_arr())
            .ok_or("нет input[]")?
            .iter()
            .map(|v| v.as_f64().ok_or("input: не число"))
            .collect::<Result<Vec<_>, _>>()?;
        let code = s
            .get("code")
            .and_then(|v| v.as_arr())
            .ok_or("нет code[]")?
            .iter()
            .map(|v| v.as_f64().ok_or("code: не число"))
            .collect::<Result<Vec<_>, _>>()?;
        let recon = s
            .get("recon")
            .and_then(|v| v.as_arr())
            .ok_or("нет recon[]")?
            .iter()
            .map(|v| v.as_f64().ok_or("recon: не число"))
            .collect::<Result<Vec<_>, _>>()?;
        if input.len() != IN_DIM || code.len() != CODE_DIM || recon.len() != IN_DIM {
            return Err(format!(
                "золото: input {} code {} recon {} (ожидалось {IN_DIM}/{CODE_DIM}/{IN_DIM})",
                input.len(),
                code.len(),
                recon.len()
            ));
        }
        out.push(Gold { label, index, input, code, recon });
    }
    Ok(out)
}

fn load_holdout(h: &J) -> Result<Vec<Sample>, String> {
    let mut out = Vec::new();
    for s in h
        .get("samples")
        .and_then(|s| s.as_arr())
        .ok_or("нет samples[]")?
    {
        let label = s.get("label").and_then(|v| v.as_f64()).ok_or("нет label")? as usize;
        let index = s.get("index").and_then(|v| v.as_f64()).ok_or("нет index")? as usize;
        let input_j = s.get("input").and_then(|v| v.as_arr()).ok_or("нет input[]")?;
        if input_j.len() != IN_DIM {
            return Err(format!("hold-out: input {} (ожидалось {IN_DIM})", input_j.len()));
        }
        let mut input = Vec::with_capacity(IN_DIM);
        for v in input_j {
            input.push(v.as_f64().ok_or("input: не число")? as f32);
        }
        out.push(Sample { label, index, input });
    }
    Ok(out)
}

fn ascii_digit(x: &[f32]) -> String {
    const CHARS: [char; 5] = [' ', '░', '▒', '▓', '█'];
    let mut s = String::new();
    for r in 0..28 {
        for c in 0..28 {
            let k = ((x[r * 28 + c] * 5.0) as usize).min(4);
            s.push(CHARS[k]);
            s.push(CHARS[k]);
        }
        s.push('\n');
    }
    s
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut ab = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for i in 0..a.len() {
        ab += a[i] as f64 * b[i] as f64;
        aa += a[i] as f64 * a[i] as f64;
        bb += b[i] as f64 * b[i] as f64;
    }
    if aa < 1e-12 || bb < 1e-12 {
        return 0.0;
    }
    ab / (aa * bb).sqrt()
}

/// Активная популяция кода: номера нейронов с |h| > ACTIVE_TH.
fn active_units(h: &[f32]) -> Vec<usize> {
    (0..h.len()).filter(|&i| h[i].abs() > ACTIVE_TH).collect()
}

fn jaccard(a: &[usize], b: &[usize]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.iter().filter(|x| b.contains(x)).count();
    let union = a.len() + b.len() - inter;
    if union == 0 {
        return 0.0;
    }
    inter as f64 / union as f64
}

fn main() {
    let mut dbw_path = PathBuf::from("tools/sparse_eye.dbw");
    let mut golden_path = PathBuf::from("tools/sparse_eye_golden.json");
    let mut holdout_path = PathBuf::from("tools/mnist_holdout.json");
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| {
            eprintln!("аргумент {a} требует значение");
            std::process::exit(2);
        });
        match a.as_str() {
            "--dbw" => dbw_path = PathBuf::from(val()),
            "--golden" => golden_path = PathBuf::from(val()),
            "--holdout" => holdout_path = PathBuf::from(val()),
            "--help" | "-h" => {
                println!(
                    "sparse_eye_probe — концепт-глаз: паритет + разреженность + спайковая популяция\n\n\
                     Использование:\n  \
                     cargo run --manifest-path crates/dioma-engine/Cargo.toml --release --example sparse_eye_probe -- \\\n    \
                     [--dbw tools/sparse_eye.dbw] [--golden tools/sparse_eye_golden.json] [--holdout tools/mnist_holdout.json]"
                );
                return;
            }
            other => {
                eprintln!("неизвестный аргумент: {other}");
                std::process::exit(2);
            }
        }
    }

    let dbw_bytes = fs::read(&dbw_path).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {dbw_path:?}: {e}\n  (сначала: python tools/train_sparse_eye.py)");
        std::process::exit(2);
    });
    let golden_text = fs::read_to_string(&golden_path).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {golden_path:?}: {e}");
        std::process::exit(2);
    });
    let holdout_text = fs::read_to_string(&holdout_path).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {holdout_path:?}: {e}\n  (сначала: python tools/train_mnist_eye.py)");
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
    assert_eq!(mlp.in_dim(), IN_DIM, "ожидался вход {IN_DIM}");
    assert_eq!(mlp.out_dim(), IN_DIM, "автоэнкодер: выход = вход {IN_DIM}");
    // Код — выход энкодера: первый dim layer0.w [code, in].
    let code_dim = tensors
        .dims("layer0.w")
        .and_then(|d| d.first().copied())
        .unwrap_or(0) as usize;
    assert_eq!(code_dim, CODE_DIM, "ожидался код {CODE_DIM}");
    println!(
        "DBW1: {} тензоров, автоэнкодер {IN_DIM}→{CODE_DIM}(tanh, концепты)→{IN_DIM} ({} байт)",
        tensors.names().len(),
        dbw_bytes.len()
    );

    let golden_j = parse_json(&golden_text).unwrap_or_else(|e| {
        eprintln!("битое золото {golden_path:?}: {e}");
        std::process::exit(2);
    });
    let golden = load_golden(&golden_j).unwrap_or_else(|e| {
        eprintln!("золото: {e}");
        std::process::exit(2);
    });
    let holdout_j = parse_json(&holdout_text).unwrap_or_else(|e| {
        eprintln!("битый hold-out {holdout_path:?}: {e}");
        std::process::exit(2);
    });
    let holdout = load_holdout(&holdout_j).unwrap_or_else(|e| {
        eprintln!("hold-out: {e}");
        std::process::exit(2);
    });
    println!("hold-out: {} сэмплов (сбалансированные классы)", holdout.len());

    // ── 1. ПАРИТЕТ: код + реконструкция ──
    println!(
        "\nПАРИТЕТ python(f64) ↔ rust(f32): {} сэмплов, код {CODE_DIM} + реконструкция, допуск |Δ| ≤ {PARITY_TOL}",
        golden.len()
    );
    let mut max_delta = 0.0f64;
    for g in &golden {
        let x: Vec<f32> = g.input.iter().map(|&v| v as f32).collect();
        let (code, recon) = mlp.forward_parts(&x); // (код 128, реконструкция 784)
        let mut d = 0.0f64;
        for k in 0..CODE_DIM {
            d = d.max((code[k] as f64 - g.code[k]).abs());
        }
        for k in 0..IN_DIM {
            d = d.max((recon[k] as f64 - g.recon[k]).abs());
        }
        max_delta = max_delta.max(d);
        println!("  сэмпл test[{}] (цифра {}): max|Δ| = {d:.2e}", g.index, g.label);
    }
    assert!(
        max_delta <= PARITY_TOL,
        "ПАРИТЕТ НАРУШЕН: max|Δ| = {max_delta:.3e} > {PARITY_TOL}!"
    );
    println!("  ПАРИТЕТ OK: max|Δ| = {max_delta:.3e} ≤ {PARITY_TOL}");

    // ── 2. КОНЦЕПТ-КАЧЕСТВО на hold-out ──
    let mut codes: Vec<Vec<f32>> = Vec::with_capacity(holdout.len());
    let mut recon_mse_sum = 0.0f64;
    let mut active_total = 0usize;
    let mut samples_by_class: Vec<Vec<usize>> = vec![Vec::new(); OUT_DIM];
    for (si, s) in holdout.iter().enumerate() {
        let (code, recon) = mlp.forward_parts(&s.input);
        active_total += active_units(&code).len();
        recon_mse_sum += recon
            .iter()
            .zip(&s.input)
            .map(|(a, b)| ((a - b) as f64) * ((a - b) as f64))
            .sum::<f64>()
            / IN_DIM as f64;
        samples_by_class[s.label].push(si);
        codes.push(code);
    }
    let sparsity = active_total as f64 / (holdout.len() as f64 * CODE_DIM as f64);
    let mean_active = active_total as f64 / holdout.len() as f64;
    let recon_mse = recon_mse_sum / holdout.len() as f64;

    // Прототипы классов и nearest-centroid.
    let mut prot: Vec<Vec<f32>> = vec![vec![0.0f32; CODE_DIM]; OUT_DIM];
    for c in 0..OUT_DIM {
        for &si in &samples_by_class[c] {
            for k in 0..CODE_DIM {
                prot[c][k] += codes[si][k];
            }
        }
        let n = samples_by_class[c].len().max(1) as f32;
        for k in 0..CODE_DIM {
            prot[c][k] /= n;
        }
    }
    let mut correct = 0usize;
    for si in 0..holdout.len() {
        let said = (0..OUT_DIM)
            .max_by(|&a, &b| cosine(&codes[si], &prot[a]).partial_cmp(&cosine(&codes[si], &prot[b])).unwrap())
            .unwrap();
        if said == holdout[si].label {
            correct += 1;
        }
    }
    let cent_acc = correct as f64 / holdout.len() as f64;

    println!("\nКОНЦЕПТ-КАЧЕСТВО (rust, hold-out {}):", holdout.len());
    println!(
        "  разреженность: активных {sparsity:.3} ({mean_active:.1}/{CODE_DIM} на код; цель 5–15%)"
    );
    println!("  реконструкция: MSE = {recon_mse:.4} (пиксели [0,1])");
    println!(
        "  разделение:    nearest-centroid accuracy = {cent_acc:.4} (код → класс БЕЗ лейбла)"
    );
    let mut sim = [[0.0f64; OUT_DIM]; OUT_DIM];
    for a in 0..OUT_DIM {
        for b in 0..OUT_DIM {
            sim[a][b] = cosine(&prot[a], &prot[b]);
        }
    }
    println!("  матрица близости прототипов [класс][класс]:");
    print!("      ");
    for d in 0..OUT_DIM {
        print!("{d:>7}");
    }
    println!();
    for k in 0..OUT_DIM {
        print!("  {k} |");
        for d in 0..OUT_DIM {
            print!("{:>7.3}", sim[k][d]);
        }
        println!();
    }
    if cent_acc >= SOFT_CENTROID {
        println!("  ОРИЕНТИР (nearest-centroid ≥ {SOFT_CENTROID}) — взят");
    } else {
        eprintln!(
            "  ВНИМАНИЕ: nearest-centroid {cent_acc:.4} < {SOFT_CENTROID} (мягкий ориентир; паритет — жёсткий ассерт)"
        );
    }

    // ── 3. СПАЙКИ: активная популяция цифр + пересечения ──
    let show = 5usize.min(holdout.len());
    println!("\nСПАЙКОВАЯ ПОПУЛЯЦИЯ (первые {show} hold-out; |h| > {ACTIVE_TH}):");
    for si in 0..show {
        let s = &holdout[si];
        let (code, recon) = mlp.forward_parts(&s.input);
        let units = active_units(&code);
        println!(
            "— test[{}] цифра {}: активны концепт-нейроны {:?}  ({} / {CODE_DIM})",
            s.index,
            s.label,
            units,
            units.len()
        );
        println!("  оригинал:");
        println!("{}", ascii_digit(&s.input));
        println!("  реконструкция:");
        println!("{}", ascii_digit(&recon));
    }

    // Jaccard: одна цифра (внутри класса) против разных.
    let mut same = Vec::new();
    let mut diff = Vec::new();
    'outer: for a in 0..holdout.len() {
        for b in a + 1..holdout.len() {
            if same.len() >= 8 && diff.len() >= 8 {
                break 'outer;
            }
            let ja = jaccard(&active_units(&codes[a]), &active_units(&codes[b]));
            if holdout[a].label == holdout[b].label && same.len() < 8 {
                same.push(ja);
            } else if holdout[a].label != holdout[b].label && diff.len() < 8 {
                diff.push(ja);
            }
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "\nПЕРЕСЕЧЕНИЕ ПОПУЛЯЦИЙ (Jaccard активных нейронов):\n  одна и та же цифра:   {:.3}  ({same:?})\n  разные цифры:         {:.3}  ({diff:?})\n  → код ведёт себя как КОРА (общие нейроны у похожих образов), а не как лейбл",
        mean(&same),
        mean(&diff)
    );
}
