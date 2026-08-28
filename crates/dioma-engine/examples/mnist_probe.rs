//! # mnist_probe — «глаз-классификатор»: видит пиксельную цифру и ГОВОРИТ, что видит
//!
//! Второй орган пайплайна «Python учит ↔ Rust играет» (первый — eye_check.rs,
//! энкодер геометрии ретины; здесь — КЛАССИФИКАЦИЯ, 10 классов MNIST):
//!
//! 1. **ПАРИТЕТ** (жёсткий ассерт): грузит `mnist.dbw` ([`BakedTensors::from_bytes`]
//!    → [`BakedMlp`]) и 5 фиксированных сэмплов из `mnist_golden.json`
//!    (python-выходы f64); Rust-инференс f32 обязан сойтись с допуском
//!    `max|Δ| ≤ 1e-4` (f32-округление весов/входа — та же история, что у глаза).
//! 2. **ТОЧНОСТЬ**: 1000 сбалансированных test-сэмплов (100/класс) из
//!    `mnist_holdout.json` → accuracy, матрица ошибок 10×10, per-class recall.
//!    Смоук-ориентир (МЯГКИЙ, пример не валит): accuracy ≥ 0.95 (python-факт
//!    на полных 10k — 0.9709, см. tools/mnist_report.txt).
//! 3. **ГОВОРИТ**: softmax логитов → «вижу N (p=0.98)» + ASCII-арт цифры —
//!    орган «речи» существа: не только метрика, но и предъявление.
//!
//! Запуск из КОРНЯ репозитория:
//!
//! ```text
//! python tools/train_mnist_eye.py
//! cargo run --manifest-path crates/dioma-engine/Cargo.toml --release --example mnist_probe
//! ```
//!
//! JSON разбирается МИНИМАЛЬНЫМ ручным парсером — тот же подход, что в
//! eye_check.rs/whisperer_check.rs (у примера нет зависимостей, формат
//! файлов контролируется нашими же писателями).

use dioma_engine::brain::baked::{BakedMlp, BakedTensors};
use std::fs;
use std::path::PathBuf;

// ─── Крошечный JSON-парсер (тот же, что в eye_check.rs) ────────────────────

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
    /// Пропуск JSON-пробелов — ТОЛЬКО 4 ASCII-символа (см. eye_check.rs:
    /// байт нельзя трактовать как char — U+0085 внутри UTF-8 «пробел»).
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

const IN_DIM: usize = 784; // 28×28 пикселей, [0,1]
const OUT_DIM: usize = 10; // классы 0..9
const PARITY_TOL: f64 = 1e-4;
const SOFT_ACC: f64 = 0.95; // мягкий ориентир качества (нарушение НЕ валит пример)

struct Gold {
    label: usize,
    index: usize,
    input: Vec<f64>,
    output: Vec<f64>,
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
        let output = s
            .get("output")
            .and_then(|v| v.as_arr())
            .ok_or("нет output[]")?
            .iter()
            .map(|v| v.as_f64().ok_or("output: не число"))
            .collect::<Result<Vec<_>, _>>()?;
        if input.len() != IN_DIM || output.len() != OUT_DIM {
            return Err(format!("золото: input {}, output {} (ожидалось {IN_DIM}/{OUT_DIM})", input.len(), output.len()));
        }
        out.push(Gold { label, index, input, output });
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

/// ASCII-арт цифры 28×28: яркость → символ. `blank` — чем рисовать нули
/// (true — сплошной фон в две колонки на пиксель).
fn ascii_digit(x: &[f32]) -> String {
    const CHARS: [char; 5] = [' ', '░', '▒', '▓', '█'];
    let mut s = String::new();
    for r in 0..28 {
        for c in 0..28 {
            let v = x[r * 28 + c];
            let k = ((v * 5.0) as usize).min(4);
            s.push(CHARS[k]);
            s.push(CHARS[k]);
        }
        s.push('\n');
        if r == 27 {
            break;
        }
    }
    s
}

/// Softmax логитов (числовой устойчивый): уверенность «речи».
fn softmax(y: &[f32]) -> Vec<f64> {
    let m = y.iter().copied().fold(f64::NEG_INFINITY, |a, v| a.max(v as f64));
    let ex: Vec<f64> = y.iter().map(|&v| ((v as f64) - m).exp()).collect();
    let sum: f64 = ex.iter().sum();
    ex.into_iter().map(|v| v / sum).collect()
}

fn main() {
    let mut dbw_path = PathBuf::from("tools/mnist.dbw");
    let mut golden_path = PathBuf::from("tools/mnist_golden.json");
    let mut holdout_path = PathBuf::from("tools/mnist_holdout.json");
    let mut speak_n = 5usize; // сколько цифр показать с ASCII-артом
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
            "--speak" => speak_n = val().parse().expect("--speak: число"),
            "--help" | "-h" => {
                println!(
                    "mnist_probe — глаз-классификатор: паритет + точность + «вижу N»\n\n\
                     Использование:\n  \
                     cargo run --manifest-path crates/dioma-engine/Cargo.toml --release --example mnist_probe -- \\\n    \
                     [--dbw tools/mnist.dbw] [--golden tools/mnist_golden.json] [--holdout tools/mnist_holdout.json] [--speak N]"
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
        eprintln!("не могу прочитать {dbw_path:?}: {e}\n  (сначала: python tools/train_mnist_eye.py)");
        std::process::exit(2);
    });
    let golden_text = fs::read_to_string(&golden_path).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {golden_path:?}: {e}");
        std::process::exit(2);
    });
    let holdout_text = fs::read_to_string(&holdout_path).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {holdout_path:?}: {e}");
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
    assert_eq!(mlp.in_dim(), IN_DIM, "ожидался MLP с входом {IN_DIM} (28×28)");
    assert_eq!(mlp.out_dim(), OUT_DIM, "ожидался MLP с выходом {OUT_DIM} (классы 0..9)");
    let hidden = tensors
        .dims("layer0.w")
        .and_then(|d| d.first().copied())
        .unwrap_or(0) as usize;
    println!(
        "DBW1: {} тензоров, MLP {IN_DIM}→{hidden}→{OUT_DIM} ({} байт, {} весов)",
        tensors.names().len(),
        dbw_bytes.len(),
        mlp.in_dim() * hidden + hidden + hidden * OUT_DIM + OUT_DIM
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

    // ── 1. ПАРИТЕТ python(f64) ↔ rust(f32) ──
    println!(
        "\nПАРИТЕТ python(f64) ↔ rust(f32): {} сэмплов, допуск |Δ| ≤ {PARITY_TOL}",
        golden.len()
    );
    let mut max_delta = 0.0f64;
    for g in &golden {
        let x: Vec<f32> = g.input.iter().map(|&v| v as f32).collect();
        let pred = mlp.forward(&x);
        let mut d = 0.0f64;
        for k in 0..OUT_DIM {
            d = d.max((pred[k] as f64 - g.output[k]).abs());
        }
        max_delta = max_delta.max(d);
        println!("  сэмпл test[{}] (цифра {}): max|Δ| = {d:.2e}", g.index, g.label);
    }
    assert!(
        max_delta <= PARITY_TOL,
        "ПАРИТЕТ НАРУШЕН: max|Δ| = {max_delta:.3e} > {PARITY_TOL} — python и rust считают по-разному!"
    );
    println!("  ПАРИТЕТ OK: max|Δ| = {max_delta:.3e} ≤ {PARITY_TOL}");

    // ── 2. ТОЧНОСТЬ на hold-out ──
    let mut correct = 0usize;
    let mut cm = vec![0usize; OUT_DIM * OUT_DIM]; // [истина*10 + предсказание]
    let mut per_class_hit = vec![0usize; OUT_DIM];
    let mut per_class_all = vec![0usize; OUT_DIM];
    for s in &holdout {
        let pred = mlp.forward(&s.input);
        let said = pred
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k)
            .unwrap();
        cm[s.label * OUT_DIM + said] += 1;
        per_class_all[s.label] += 1;
        if said == s.label {
            correct += 1;
            per_class_hit[s.label] += 1;
        }
    }
    let acc = correct as f64 / holdout.len() as f64;
    println!("\nТОЧНОСТЬ (hold-out {}): {:.4} ({}/{})", holdout.len(), acc, correct, holdout.len());
    println!("матрица ошибок [строка=истина, столбец=сказано]:");
    print!("      ");
    for d in 0..OUT_DIM {
        print!("{d:>6}");
    }
    println!("   recall");
    for k in 0..OUT_DIM {
        print!("  {k} |");
        for d in 0..OUT_DIM {
            print!("{:>6}", cm[k * OUT_DIM + d]);
        }
        println!("{:>8.3}", per_class_hit[k] as f64 / per_class_all[k] as f64);
    }
    if acc >= SOFT_ACC {
        println!("  ОРИЕНТИР (accuracy ≥ {SOFT_ACC}) — взят");
    } else {
        eprintln!(
            "  ВНИМАНИЕ: accuracy {acc:.4} < {SOFT_ACC} (мягкий ориентир; паритет остаётся жёстким ассертом)"
        );
    }

    // ── 3. ГОВОРИТ: «вижу N» с ASCII-артом ──
    let speak = speak_n.min(holdout.len());
    println!("\nГОВОРИТ (первые {speak} hold-out, softmax логитов):");
    for s in holdout.iter().take(speak) {
        let pred = mlp.forward(&s.input);
        let p = softmax(&pred);
        let said = (0..OUT_DIM).max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap()).unwrap();
        println!("— картинка test[{}] (истинная цифра {}):", s.index, s.label);
        println!("{}", ascii_digit(&s.input));
        let mut top: Vec<(usize, f64)> = p.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let alt = top
            .iter()
            .take(3)
            .map(|(k, v)| format!("{k}:{v:.3}"))
            .collect::<Vec<_>>()
            .join("  ");
        let ok = if said == s.label { "✓" } else { "✗" };
        println!("  -> ВИЖУ {} (p={:.3})   [{} ok]   топ: {}", said, p[said], ok, alt);
    }
}
