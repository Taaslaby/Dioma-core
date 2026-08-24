//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_sardana_10k.rs
//! Class:       SemantikaEngine
//! Purpose:     10K нейронов от sarDNA — ГИПЕРПРЫЖОК масштаба
//! Description: грузим 10000-нейронный коннектом сгенерированный
//!              трансформером, запускаем LIF-α, меряем активность.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use taas_format::TaasReader;

fn main() {
    println!("poc-sardana-10k: sarDNA трансформер · 10000 нейронов");

    let path = "D:/Проекты/asm_engines/sardaanaCM/experiments/sardna_ultra/generated/flywire_10k_n10000_t0_85.taas";
    let r = TaasReader::from_file(path).expect("load 10k taas");

    // Смотрим что внутри.
    println!("компоненты:");
    for c in r.list_components() {
        println!("  {} dtype={} shape={:?}", c.name, c.dtype, c.shape);
    }

    let rp: Vec<i64> = r.get_i64("row_ptr").unwrap_or_else(|_| {
        r.get_i32("row_ptr").unwrap().iter().map(|&v| v as i64).collect()
    });
    let ci: Vec<i32> = r.get_i32("col_indices").unwrap();
    let vals: Vec<f32> = r.get_f32("values").unwrap();
    let nt_raw: Vec<f32> = r.get_f32("nt_types").unwrap();

    let n = rp.len().saturating_sub(1);
    let n_syn = ci.len();
    println!("\n10K мозг: {} нейронов · {} синапсов", n, n_syn);

    // Масштабируем веса.
    let w_max = vals.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if w_max > 0.0 { 2.0 / w_max } else { 1.0 };
    let vals_s: Vec<f32> = vals.iter().map(|&v| v * scale).collect();

    // E/I статистика.
    let n_exc = nt_raw.iter().filter(|&&x| x > 0.0).count();
    let n_inh = nt_raw.iter().filter(|&&x| x < 0.0).count();
    println!("E/I: {} / {} ({:.0}% возб.)", n_exc, n_inh, 
        n_exc as f64 / (n_exc + n_inh).max(1) as f64 * 100.0);

    // Строим ConnectomeData.
    let conn = ConnectomeData::new(
        rp.iter().map(|&v| v as i32).collect(),
        ci.clone(),
        vals_s,
        nt_raw,
        vec![],
    );

    // ── Запускаем LIF-α ──
    println!("\nзапускаем LIF-α...");
    let t0 = std::time::Instant::now();
    let mut runner = BrainRunner::new(conn, BrainConfig::lif_alpha(0.05)).unwrap();
    
    let zero = vec![0.0f32; n];
    for _ in 0..100 { runner.run(1, &zero); }

    // Стимул на первые 500 нейронов.
    let mut stim = zero.clone();
    for cell in stim.iter_mut().take(500) { *cell = 30.0; }

    let mut stim_spikes = 0i64;
    let mut stim_active = HashSet::new();
    for _ in 0..200 {
        let buf = runner.run(1, &stim);
        for (i, &s) in buf.iter().enumerate() {
            if s != 0 { stim_spikes += 1; stim_active.insert(i); }
        }
    }
    let stim_time = t0.elapsed().as_secs_f64();

    // Тишина.
    let mut silence_spikes = 0i64;
    for _ in 0..400 {
        let buf = runner.run(1, &zero);
        silence_spikes += buf.iter().map(|&s| s as i64).sum::<i64>();
    }
    let silence_time = t0.elapsed().as_secs_f64() - stim_time;

    println!("\n══ РЕЗУЛЬТАТ ══");
    println!("стимул (10 мс): {stim_spikes} спайков от {} уникальных нейронов", stim_active.len());
    println!("тишина (20 мс): {silence_spikes} спайков");
    println!("время: стимул={stim_time:.2}с тишина+обработка={:.2}с", silence_time);

    if stim_spikes > 100 && stim_active.len() > 200 {
        println!("\n10К МОЗГ ЖИВОЙ И АКТИВНЫЙ ✓✓✓");
        println!("{} из {} нейронов стреляли ({:.0}%)",
            stim_active.len(), n, stim_active.len() as f64 / n as f64 * 100.0);
    } else {
        println!("слабая активность — нужно больше драйва или другая модель");
    }
}

use std::collections::HashSet;
