//! -------------------------------------------------------------------------
//! Filename:    src/bin/sardana_personalities.rs
//! Class:       SemantikaEngine
//! Purpose:     разные сардаана-мозги = разные ЛИЧНОСТИ?
//! Description: генерируем мозги 3 видов через sarDNA, сравниваем их
//!              спонтанную активность. Если отпечатки различаются —
//!              каждое существо имеет уникальный характер.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use taas_format::TaasReader;

fn main() {
    println!("╔═══════════════════════════════════════╗");
    println!("║  САРДААНА — ЛИЧНОСТИ МОЗГОВ            ║");
    println!("╚═══════════════════════════════════════╝");

    // ── Генерируем 3 мозга от разных видов ──
    // Используем готовые .taas файлы.
    let brains = vec![
        ("SarDNA_Pro", "D:/Проекты/asm_engines/sardaanaCM/data/generated/SarDNA_Pro_best.taas"),
        ("Diff_snn", "D:/Проекты/asm_engines/sardaanaCM/data/generated/Diff_snn_best.taas"),
    ];

    for (name, path) in &brains {
        println!("\n══ {} ══", name);

        let r = match TaasReader::from_file(path) {
            Ok(r) => r,
            Err(e) => { println!("  ошибка загрузки: {e}"); continue; }
        };

        let rp: Vec<i32> = r.get_i32("row_ptr").unwrap();
        let ci: Vec<i32> = r.get_i32("col_indices").unwrap();
        let vals_raw: Vec<f32> = r.get_f32("values").unwrap();
        let nt: Vec<f32> = r.get_f32("nt_types").unwrap();

        let n = rp.len().saturating_sub(1);
        let n_syn = ci.len();

        // Масштабируем.
        let w_max = vals_raw.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if w_max > 0.0 { 2.0 / w_max } else { 1.0 };
        let vals: Vec<f32> = vals_raw.iter().map(|&v| v * scale).collect();

        // Строим.
        let conn = ConnectomeData::new(rp.clone(), ci.clone(), vals.clone(), nt.clone(), vec![]);
        let mut runner = BrainRunner::new(conn, BrainConfig::lif_alpha(0.05)).unwrap();

        // ── Записываем активность 2000 шагов (100 мс) ──
        let zero = vec![0.5f32; n]; // минимальный тоник — чистая личность
        let mut rates = vec![0i64; n];
        let mut total_per_step = Vec::new();

        for _ in 0..2000 {
            let buf = runner.run(1, &zero);
            for (i, &s) in buf.iter().enumerate() {
                rates[i] += s as i64;
            }
            let step_total: i64 = buf.iter().map(|&s| s as i64).sum();
            total_per_step.push(step_total);
        }

        // Метрики личности.
        let total: f64 = rates.iter().sum::<i64>() as f64;
        let active_neurons = rates.iter().filter(|&&r| r > 0).count();
        let mean_rate = total / n as f64;
        
        // Топ-5 самых активных нейронов.
        let mut ranked: Vec<(usize, i64)> = rates.iter().enumerate()
            .map(|(i, &v)| (i, v)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));

        // Коэффициент вариации (насколько неровная активность).
        let mean_steps = total / 2000.0;
        let variance = total_per_step.iter()
            .map(|&x| (x as f64 - mean_steps).powi(2))
            .sum::<f64>() / 2000.0;
        let cv = variance.sqrt() / mean_steps.max(0.01);

        println!("  нейронов={n} синапсов={n_syn}");
        println!("  активных нейронов: {}/{} ({:.0}%)",
            active_neurons, n, active_neurons as f64 / n as f64 * 100.0);
        println!("  средняя частота: {:.1} сп/тик", mean_rate / 2000.0);
        println!("  CV (хаотичность): {:.2}", cv);

        print!("  топ-5 хабов: ");
        for (idx, act) in ranked.iter().take(5) {
            print!("#{idx}({act}) ");
        }
        println!();

        // Отпечаток личности: первые 20 значений rates как хеш.
        let fingerprint: String = rates.iter().take(20)
            .map(|&v| if v > 0 { format!("{:x}", v.min(15) as u8) } else { ".".to_string() })
            .collect();
        println!("  отпечаток: {fingerprint}");
    }

    println!("\nЕсли отпечатки РАЗНЫЕ — у каждого мозга свой характер ✓");
}
