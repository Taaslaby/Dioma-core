//! -------------------------------------------------------------------------
//! Filename:    src/bin/sardana_probe.rs
//! Class:       SemantikaEngine
//! Purpose:     что УМЕЕТ 10K сардаана-мозг? Не просто спайки — вычисления
//! Description: 1) Pattern separation: разные стимулы → разные ответы?
//!              2) Каскад: как глубоко распространяется сигнал?
//!              3) Хабы: какие нейроны командуют сетью?
//!              4) Oscillation: есть ли ритмы в активности?
//!              5) Noise discrimination: отличает сигнал от шума?
//! author: Atlasov anton
// -------------------------------------------------------------------------
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use taas_format::TaasReader;
use std::collections::HashSet;

fn main() {
    println!("╔══════════════════════════════════════════╗");
    println!("║  SARDAANA 10K — ЧТО ТЫ УМЕЕШЬ, МОЗГ?     ║");
    println!("╚══════════════════════════════════════════╝");

    // ── Загрузка ──
    let path = "D:/Проекты/asm_engines/sardaanaCM/experiments/sardna_ultra/generated/flywire_10k_n10000_t0_85.taas";
    let r = TaasReader::from_file(path).unwrap();
    let rp: Vec<i32> = r.get_i32("row_ptr").unwrap();
    let ci: Vec<i32> = r.get_i32("col_indices").unwrap();
    let vals_raw: Vec<f32> = r.get_f32("values").unwrap();
    let nt: Vec<f32> = r.get_f32("nt_types").unwrap();

    let n = rp.len().saturating_sub(1);
    let w_max = vals_raw.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if w_max > 0.0 { 2.0 / w_max } else { 1.0 };
    let vals: Vec<f32> = vals_raw.iter().map(|&v| v * scale).collect();

    println!("мозг загружен: {n} нейронов · {} синапсов", ci.len());

    // ── Строим runner ──
    let conn = ConnectomeData::new(rp.clone(), ci.clone(), vals.clone(), nt.clone(), vec![]);
    let mut brain = BrainRunner::new(conn, BrainConfig::lif_alpha(0.05)).unwrap();
    let z = vec![0.0f32; n];

    // Разминка.
    for _ in 0..200 { brain.run(1, &z); }

    // ══════════════════════════════════════════════
    // ТЕСТ 1: Pattern Separation
    // Стимулируем разные подмножества → сравниваем ответы.
    // Если паттерны различаются — мозг РАЗЛИЧАЕТ входы.
    // ══════════════════════════════════════════════
    println!("\n── 1. PATTERN SEPARATION ──");
    let mut patterns = Vec::new();
    for pat_id in 0..4 {
        // Сброс.
        let mut r2 = BrainRunner::new(
            ConnectomeData::new(rp.clone(), ci.clone(), vals.clone(), nt.clone(), vec![]),
            BrainConfig::lif_alpha(0.05),
        ).unwrap();
        for _ in 0..100 { r2.run(1, &z); }

        // Стимул: каждые 64-е нейроны для паттерна pat_id.
        let mut stim = z.clone();
        for i in (pat_id * 250)..(pat_id * 250 + 500).min(n) {
            stim[i] = 30.0;
        }

        let mut resp = vec![0i64; n];
        for _ in 0..100 { // 5 мс стимула
            let buf = r2.run(1, &stim);
            for (i, &s) in buf.iter().enumerate() {
                resp[i] += s as i64;
            }
        }
        patterns.push(resp);
    }

    // Попарная корреляция ответов.
    for a in 0..3 {
        for b in (a+1)..4 {
            let dot: f64 = patterns[a].iter().zip(patterns[b].iter())
                .map(|(x, y)| (x * y) as f64).sum();
            let na: f64 = patterns[a].iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = patterns[b].iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let cos = if na * nb > 0.0 { dot / (na * nb) } else { 0.0 };
            println!("  паттерн {} vs {}: cosine={:.3}", a, b, cos);
        }
    }
    let avg_cos = {
        let mut cos_sum = 0.0; let mut cnt = 0;
        for a in 0..4 { for b in (a+1)..4 {
            let dot: f64 = patterns[a].iter().zip(patterns[b].iter())
                .map(|(x, y)| (x * y) as f64).sum();
            let na = patterns[a].iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb = patterns[b].iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            if na * nb > 0.0 { cos_sum += dot / (na * nb); cnt += 1; }
        }}
        cos_sum / cnt.max(1) as f64
    };
    println!("  средняя корреляция: {:.3} ({})", avg_cos,
        if avg_cos < 0.5 {"различает ✓"} else {"слишком похоже"});

    // ══════════════════════════════════════════════
    // ТЕСТ 2: КАСКАД — как далеко распространяется?
    // ══════════════════════════════════════════════
    println!("\n── 2. КАСКАД АКТИАЦИИ ──");
    let mut cascade_runner = BrainRunner::new(
        ConnectomeData::new(rp.clone(), ci.clone(), vals.clone(), nt.clone(), vec![]),
        BrainConfig::lif_alpha(0.05),
    ).unwrap();
    for _ in 0..100 { cascade_runner.run(1, &z); }

    let mut stim2 = z.clone();
    for cell in stim2.iter_mut().take(10) { *cell = 50.0; } // точечный удар

    let mut wavefront: Vec<usize> = Vec::new();
    let mut prev_active: HashSet<usize> = HashSet::new();
    for step in 0..50 {
        let buf = cascade_runner.run(1, &stim2);
        let now_active: HashSet<usize> = buf.iter().enumerate()
            .filter(|(_, &s)| s != 0).map(|(i, _)| i).collect();
        
        // Новые нейроны = активные сейчас но не были раньше.
        let new_count = now_active.difference(&prev_active).count();
        wavefront.push(new_count);
        prev_active.extend(now_active);
    }

    // Волна: сколько НОВЫХ нейронов активируется на каждом шаге.
    let total_wave: usize = wavefront.iter().sum();
    let peak_step = wavefront.iter().enumerate().max_by_key(|(_, v)| **v).map(|(i, _)| i).unwrap_or(0);
    println!("  новых нейронов по шагам (первые 20): {:?}",
        wavefront.iter().take(20).collect::<Vec<_>>());
    println!("  всего вовлечено: {}/{} ({}%)", total_wave / 50, n,
        total_wave / 50 * 100 / n);

    drop(wavefront);

    // ══════════════════════════════════════════════
    // ТЕСТ 3: ХАБЫ — кто командует?
    // ══════════════════════════════════════════════
    println!("\n── 3. ХАБЫ СЕТИ ──");
    // Стимулируем всё, меряем кто стреляет чаще всего.
    let mut full_stim = vec![3.0f32; n];
    let mut neuron_activity = vec![0i64; n];
    for _ in 0..500 {
        let buf = brain.run(1, &full_stim);
        for (i, &s) in buf.iter().enumerate() {
            neuron_activity[i] += s as i64;
        }
    }

    // Топ-10 самых активных.
    let mut ranked: Vec<(usize, i64)> = neuron_activity.iter().enumerate()
        .map(|(i, &v)| (i, v)).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    println!("топ-10 коммандных нейронов:");
    for &(idx, act) in ranked.iter().take(10) {
        let deg = rp[idx + 1] - rp[idx];
        println!("  #{:<5} активность={:<6} out_degree={}", idx, act, deg);
    }

    // Концентрация активности: топ-1% нейронов делает какой % спайков?
    let total: i64 = neuron_activity.iter().sum();
    let top_1pct: i64 = ranked.iter().take(n / 100).map(|(_, v)| *v).sum();
    println!("  топ-1% нейронов генерирует {:.1}% всей активности",
        top_1pct as f64 / total.max(1) as f64 * 100.0);

    // ══════════════════════════════════════════════
    // ТЕСТ 4: OSCILLATION — есть ли ритмы?
    // ══════════════════════════════════════════════
    println!("\n── 4. ОСЦИЛЛЯЦИИ ──");
    let mut rate_series = Vec::new();
    for _ in 0..500 {
        let buf = brain.run(1, &drive_vec(n, 3.0));
        rate_series.push(buf.iter().map(|&s| s as f64).sum::<f64>());
    }
    
    // Простая автокорреляция на лаге 5 и 10.
    let mean = rate_series.iter().sum::<f64>() / rate_series.len() as f64;
    let var = rate_series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rate_series.len() as f64;
    if var > 0.0 {
        for lag in [5usize, 10] {
            let cov: f64 = rate_series.iter().zip(rate_series.iter().skip(lag))
                .map(|(a, b)| (a - mean) * (b - mean)).sum::<f64>()
                / (rate_series.len() - lag) as f64;
            println!("  автокорреляция лаг={}: {:.3}", lag, cov / var);
        }
    }
    let cv = var.sqrt() / mean.max(0.01);
    println!("  коэффициент вариации: {:.2} ({})", cv,
        if cv > 0.3 {"ритмичная активность ✓"} else {"тоник"});
}

fn drive_vec(n: usize, level: f32) -> Vec<f32> {
    vec![level; n]
}
