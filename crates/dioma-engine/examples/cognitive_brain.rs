//! # cognitive_brain — «КОГНИТИВНАЯ АРХИТЕКТУРА МОЗГА»: замкнутая петля
//!     восприятие → концепты → долговременная память → действие
//!
//! Собирает из реально существующих запечённых моделей целостный когнитивный
//! конвейер существа — без выдуманных заглушек. Три органа, каждый уже
//! обучен Python-пайплайном и проверен на паритет Python↔Rust:
//!
//! 1. **КОНЦЕПТ-ГЛАЗ** (`tools/sparse_eye.dbw`, `train_sparse_eye.py`):
//!    разреженный автоэнкодер 784 → 128 → 784. Код — распределённая
//!    популяция (~14 активных из 128, как спайки в коре), одинаковые образы
//!    делят нейроны, разные — почти не пересекаются (Jaccard 0.30 vs 0.03).
//!    Ретина (пиксели) → концепты.
//!
//! 2. **ДОЛГОВРЕМЕННАЯ ПАМЯТЬ (GRU)**: `tools/whisperer.dbw`,
//!    `train_whisperer.py` — GRU-«Шептун». Скрытое состояние `h` — это
//!    долговременный след: ворота z (держать/забыть) и r (сброс) решают,
//!    сколько старого следа сохранить и сколько нового впитать.
//!    `h = (1−z)⊙tanh(W·[r⊙h, x]) + z⊙h_prev`. Мой вход — концепт-код глаза,
//!    так что *то, что запечатлено глазом*, интегрально уходит в память.
//!
//! 3. **МОТОРНЫЙ READOUT**: та же `whisperer.dbw` даёт MLP
//!    hidden → (steer, drive, eat) — из следа памяти рождается действие.
//!
//! ## Что показывает пример
//!
//! * **ЗАМКНУТУЮ ЛЕНТУ**: один тик существа = пиксели → концепт-код →
//!   GRU.step(код) → hidden → readout → моторный выход. Весь цикл
//!   «вижу → помню → действую» в одном месте.
//! * **ЧТО ДАЁТ ПАМЯТЬ**: честный эксперимент на САМОМ GRU — кормим один и
//!   тот же концепт-стимул, а потом «пустой» тик, и меряем след скрытого
//!   состояния. Если память работает, след не обнуляется мгновенно:
//!   h после отбивочного тика сохраняет проекцию на вход → существует
//!   интегральная компонента, которая держит прошлое.
//!
//! Третий орган — слой спайкового мозга на GLIF2/GLIF4 из зоопарка SiMuse
//! (`pillars::zoo::neurons::glif4`) — встаёт получателем концепт-кода и
//! эффектором от readout. Здесь он объявлен как трейт `SpikingEffector`
//! (standalone-пример не тянет тяжёлый спайк-раннер); точка стыка очевидна.
//!
//! Запуск из КОРНЯ:
//!
//! ```text
//! cargo run --manifest-path crates/dioma-engine/Cargo.toml --release --example cognitive_brain
//! ```

use dioma_engine::brain::baked::{BakedGru, BakedMlp, BakedTensors};
use std::path::PathBuf;

// ─── Размерности запечённых моделей ────────────────────────────────────────
const EYE_IN: usize = 784; // пикселей ретины
const CODE_DIM: usize = 128; // концепт-код
const GRU_IN: usize = 12; // вход GRU-памяти (сенсоры существа)
const MOTOR_DIM: usize = 3; // steer, drive, eat

/// Взять «ретину» из hold-out MNIST (`tools/mnist_holdout.json`): реальные
/// пиксели цифры, на которых концепт-глаз обучен. Берём первый сэмпл и его
/// `input` (784 float). Мини-парсер нацелен ровно на формат этого файла:
/// `{"samples":[{"label":…,"index":…,"input":[784]}]}` — без общих
/// зависимостей.
///
/// Возвращает `(label, input: Vec<f32>)`.
fn load_holdout_retina(text: &str) -> (u8, Vec<f32>) {
    // Токен первого сэмпла: возьмём всё до ключа "samples" забудем, а возьмём
    // фрагмент `{"label":N,..."input":[...]}` первым объектом после samples[.
    let samples = text
        .find("\"samples\"")
        .unwrap_or_else(|| panic!("нет \"samples\" в holdout-файле"));
    let obj_start = text[samples..]
        .find('{')
        .map(|k| samples + k)
        .unwrap_or_else(|| panic!("нет первого сэмпла"));
    let obj = &text[obj_start..];
    // label: ищем ключ "label"
    let label_off = obj.find("\"label\"").expect("нет label");
    let after_colon = obj[label_off..].find(':').unwrap_or(0) + label_off + 1;
    let label_str: String = obj[after_colon..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let label: u8 = label_str.parse().unwrap_or(0);
    // input: ищем ключ "input" и открывающую '['
    let input_off = obj.find("\"input\"").expect("нет input");
    let open = obj[input_off..].find('[').expect("нет [ для input") + input_off;
    let close_rel = obj[open..].find(']').expect("нет ] для input");
    let nums = &obj[open + 1..open + close_rel];
    let v: Vec<f32> = nums
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse::<f32>().expect("input: не число"))
        .collect();
    assert_eq!(v.len(), EYE_IN, "ретина должна быть {EYE_IN}, а не {}", v.len());
    (label, v)
}

/// Активные (возбуждённые) концепт-нейроны: |h| > порог.
fn active_units(code: &[f32], th: f32) -> Vec<usize> {
    code.iter()
        .enumerate()
        .filter(|&(_, &h)| h.abs() > th)
        .map(|(i, _)| i)
        .collect()
}

/// Один тик петли: память читает концепт-код, readout рождает моторный выход.
fn tick(ltm: &mut BakedGru, readout: &BakedMlp, concept_in: &[f32]) -> [f32; MOTOR_DIM] {
    ltm.step(concept_in);
    let y = readout.forward(ltm.hidden());
    [y[0].tanh(), sigmoid(y[1]), sigmoid(y[2])]
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Косинус между скрытым состоянием и сохранённым «следом»: 1 = идентичны.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let na = l2(a).max(1e-9);
    let nb = l2(b).max(1e-9);
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (na * nb)
}

// ─── Слой спайкового мозга из зоопарка SiMuse (точка стыка) ────────────────
// В полноценном существе (dioma-engine `Brain` на pillars-connectome) это
// живой спайковый раннер на GLIF2/GLIF4: активные единицы концепт-кода —
// входной ток i_syn, представление → ток моторных популяций.
#[allow(dead_code)]
trait SpikingEffector {
    /// Активные единицы концепт-кода → входной ток зрительных регионов (i_syn).
    fn inject_concept_code(&mut self, code: &[f32], active: &[usize]);
    /// Моторный readout [steer, drive, eat] → внешний ток моторных пулов.
    fn inject_motor(&mut self, readout: [f32; MOTOR_DIM]);
}

fn main() {
    let mut eye_dbw = PathBuf::from("tools/sparse_eye.dbw");
    let mut whisper_dbw = PathBuf::from("tools/whisperer.dbw");
    let mut holdout_path = PathBuf::from("tools/mnist_holdout.json");
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| {
            eprintln!("аргумент {a} требует значение");
            std::process::exit(2);
        });
        match a.as_str() {
            "--eye" => eye_dbw = PathBuf::from(val()),
            "--whisper" => whisper_dbw = PathBuf::from(val()),
            "--holdout" => holdout_path = PathBuf::from(val()),
            "--help" | "-h" => {
                println!(
                    "cognitive_brain — концепт-глаз + GRU-память + моторный readout\n\n\
                     Использование:\n  \
                     cargo run --release --example cognitive_brain -- [--eye PATH] [--whisper PATH] [--holdout tools/mnist_holdout.json]"
                );
                return;
            }
            o => {
                eprintln!("неизвестный аргумент: {o}");
                std::process::exit(2);
            }
        }
    }

    // ── 1. Загрузка обоих органов ──
    let eye_bytes = std::fs::read(&eye_dbw).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {eye_dbw:?}: {e}\n  (сначала: python tools/train_sparse_eye.py)");
        std::process::exit(2);
    });
    let eye_t = BakedTensors::from_bytes(&eye_bytes).unwrap_or_else(|e| {
        eprintln!("концепт-глаз DBW1: {e}");
        std::process::exit(2);
    });
    let eye = BakedMlp::from_tensors(&eye_t).expect("концепт-глаз BakedMlp");
    assert_eq!(eye.in_dim(), EYE_IN);
    assert_eq!(eye.out_dim(), EYE_IN, "глаз — автоэнкодер: выход = вход");
    let code_dim =
        eye_t.dims("layer0.w").and_then(|d| d.first().copied()).unwrap_or(0) as usize;
    assert_eq!(code_dim, CODE_DIM, "ожидался концепт-код {CODE_DIM}");

    let w_bytes = std::fs::read(&whisper_dbw).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {whisper_dbw:?}: {e}\n  (готовится: python tools/train_whisperer.py --data ../../tools)");
        std::process::exit(2);
    });
    let w_t = BakedTensors::from_bytes(&w_bytes).expect("шептун DBW1");
    let mut ltm = BakedGru::from_tensors(&w_t).expect("шептун BakedGru");
    let readout = BakedMlp::from_tensors(&w_t).expect("шептун readout BakedMlp");
    assert_eq!(ltm.in_dim(), GRU_IN, "ожидался GRU с входом {GRU_IN}");
    assert_eq!(readout.in_dim(), ltm.hidden_dim(), "readout не состыкован с GRU");
    assert_eq!(readout.out_dim(), MOTOR_DIM, "readout с выходом {MOTOR_DIM}");

    println!(
        "КОГНИТИВНЫЙ МОЗГ: концепт-глаз {EYE_IN}→{CODE_DIM} (разреж. код) | GRU-память {GRU_IN}→{}→{} мотор",
        ltm.hidden_dim(),
        MOTOR_DIM
    );

    // ── 2. Замкнутая лента: пиксель → концепт-код → память → readout ──
    // Ретина — настоящий сэмпл MNIST из hold-out (в диапазоне обучающего
    // глаза). Если файла нет — падаем с подсказкой.
    let holdout_text = std::fs::read_to_string(&holdout_path).unwrap_or_else(|e| {
        eprintln!("не могу прочитать {holdout_path:?}: {e}\n  (готовится: python tools/train_mnist_eye.py)");
        std::process::exit(2);
    });
    let (label, retina) = load_holdout_retina(&holdout_text);
    let (code, _recon) = eye.forward_parts(&retina);
    let act = active_units(&code, 0.25);
    println!("\n── ОДИН ТИК ПЕТЛИ ──");
    println!("  ретина = MNIST-цифра {label} (первый сэмпл hold-out)");
    println!("  концепт-код: {} активных / {CODE_DIM} (порог 0.25)", act.len());

    // Вход памяти: MAP активного концепт-кода в визуальные сенсоры существа.
    // (Пространство сенсоров GRU шире одного видения глаза; берём первые
    //  каналы под зрение, остальные — как прочие сенсоры = 0.)
    let mut concept_in = vec![0.0f32; GRU_IN];
    for &a in act.iter().take(GRU_IN) {
        concept_in[a] = code[a].clamp(-1.0, 1.0);
    }
    ltm.reset();
    let out = tick(&mut ltm, &readout, &concept_in);
    println!(
        "  readout → steer={:.3} drive={:.3} eat={:.3}",
        out[0], out[1], out[2]
    );

    // ── 3. Что даёт память: след не обнуляется мгновенно ──
    println!("\n── ЧТО ДАЁТ ПАМЯТЬ (скрытое состояние после пустого тика) ──");
    // A: один тик со стимулом.
    let mut g_a = BakedGru::from_tensors(&w_t).unwrap();
    g_a.step(&concept_in);
    let h_after = g_a.hidden().to_vec();
    // B: стимул, затем ПУСТОЙ тик (нулевой вход). Если след «держится»,
    // h после пустого тика сохраняет проекцию на h_after.
    let mut g_b = BakedGru::from_tensors(&w_t).unwrap();
    g_b.step(&concept_in);
    let blank = vec![0.0f32; GRU_IN];
    g_b.step(&blank);
    let h_after_blank = g_b.hidden().to_vec();

    let sim = cosine(&h_after, &h_after_blank);
    let norm_after = l2(&h_after);
    let norm_blank = l2(&h_after_blank);
    println!("  скрытая размерность: {}", ltm.hidden_dim());
    println!("  ‖h‖ после 1 стимула          = {norm_after:.4}");
    println!("  ‖h‖ после стимул + пустой тик = {norm_blank:.4}");
    println!("  cosine(h_ст, h_ст+blank)      = {sim:.4}");
    if sim > 0.9 {
        println!(
            "  → след держится через шумовой тик (cosine≈1) — память хранит прошлое, а не только текущий кадр"
        );
    } else if sim > 0.5 {
        println!(
            "  → след заметно держится (cosine={sim:.2}): интегральная компонента времени на месте"
        );
    } else {
        println!(
            "  → след угас быстро (cosine={sim:.2}): короткая контекстная память этого GRU"
        );
    }

    // ── 4. Итог ──
    println!("\n── ИТОГ ──");
    println!("  Замкнутая когнитивная петля собрана и тикает: пиксель → концепты ({})", act.len());
    println!("  → долговременная память GRU → readout (steer/drive/eat).");
    println!("  Слой спайкового мозга GLIF2/4 SiMuse встаёт как `SpikingEffector`");
    println!("  (активные единицы кода = i_syn зрительных регионов; readout = моторного тока).");

    if act.is_empty() {
        eprintln!("\n  ВНИМАНИЕ: код пуст (порог 0.25) — проверьте sparse_eye.dbw.");
        std::process::exit(1);
    }
}