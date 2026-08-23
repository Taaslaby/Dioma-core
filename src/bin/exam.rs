//! -------------------------------------------------------------------------
//! Filename:    src/bin/exam.rs
//! Class:       SemantikaEngine
//! Purpose:     exam — настоящая сила агентов, а не «пожрал»
//! Description: батарея T1-T4 (поиск/эффективность/экономика/помехи/
//!              характер) для нескольких конфигураций существ + сводный
//!              балл силы.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::exam::{run_exam, ExamConfig};

fn main() {
    let configs = [
        ExamConfig { name: "органный + разум", seed: 0x0000_AE11, with_mind: true, sensory_noise: 0.0, episodes: 6 },
        ExamConfig { name: "органный без разума", seed: 0x0000_AE11, with_mind: false, sensory_noise: 0.0, episodes: 6 },
        ExamConfig { name: "органный + разум (другой сид)", seed: 0x00C0_FFEE, with_mind: true, sensory_noise: 0.0, episodes: 6 },
    ];

    let mut results = Vec::new();
    for cfg in &configs {
        let r = run_exam(cfg);
        r.print(cfg.name);
        println!("  БАЛЛ СИЛЫ: {:.1}/100\n", r.score());
        results.push((cfg.name, r.score()));
    }

    println!("════ РЕЙТИНГ СИЛЫ ════");
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (i, (name, score)) in results.iter().enumerate() {
        println!("{}. {} — {:.1}", i + 1, name, score);
    }
}
