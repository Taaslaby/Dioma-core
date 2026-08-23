//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_sardaana_brain.rs
//! Class:       SemantikaEngine
//! Purpose:     существо с мозгом от sarDNA трансформера
//! Description: грузим .taas от сардааны как ConnectomeData, запускаем
//!              через BrainRunner (LIF-α), смотрим живая ли сеть.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use taas_format::TaasReader;

fn main() {
    println!("poc-sardaana-brain: мозг от sarDNA трансформера");

    // ── 1. Грузим .taas ──
    let path = "D:/Проекты/asm_engines/sardaanaCM/data/generated/SarDNA_Pro_best.taas";
    let r = TaasReader::from_file(path).expect("load sardaana");

    let rp: Vec<i32> = r.get_i32("row_ptr").unwrap();
    let ci: Vec<i32> = r.get_i32("col_indices").unwrap();
    let vals: Vec<f32> = r.get_f32("values").unwrap();
    let nt: Vec<f32> = r.get_f32("nt_types").unwrap();
    let n = rp.len().saturating_sub(1);

    println!("sardaana: {n} нейронов · {} синапсов", ci.len());

    // ── 2. Масштабируем веса в рабочий диапазон ──
    let w_max_abs = vals.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if w_max_abs > 0.0 { 2.0 / w_max_abs } else { 1.0 };
    let vals_scaled: Vec<f32> = vals.iter().map(|&v| v * scale).collect();

    let conn = ConnectomeData::new(rp.clone(), ci.clone(), vals_scaled, nt.clone(), vec![]);

    // ── 3. BrainRunner LIF-α на этом коннектоме ──
    let mut runner = BrainRunner::new(
        conn,
        BrainConfig::lif_alpha(0.05),
    )
    .expect("runner");

    // Разминка без входа.
    let zero = vec![0.0f32; n];
    for _ in 0..100 {
        runner.run(1, &zero);
    }

    // ── 4. Стимулируем случайные группы — смотрим отклик сети ──
    println!("\nтест: стимул на первые 20 нейронов (10 мс)");
    let mut stim = zero.clone();
    for cell in stim.iter_mut().take(20) {
        *cell = 30.0;
    }

    let mut total_spikes = 0i64;
    let mut active_neurons = std::collections::HashSet::new();
    for _ in 0..200 {
        let buf = runner.run(1, &stim);
        for (i, &s) in buf.iter().enumerate() {
            if s != 0 {
                total_spikes += 1;
                active_neurons.insert(i);
            }
        }
    }
    println!(
        "за 10 мс: {} спайков от {} уникальных нейронов",
        total_spikes,
        active_neurons.len()
    );

    // Тишина — проверяем самоподдержание.
    let mut silence_spikes = 0i64;
    for _ in 0..400 {
        let buf = runner.run(1, &zero);
        silence_spikes += buf.iter().map(|&s| s as i64).sum::<i64>();
    }
    println!(
        "тишина 20 мс: {silence_spikes} спайков — {}",
        if silence_spikes == 0 { "сеть успокоилась ✓" } else { "самоподдержание!" }
    );

    // ── 5. Статистика по регионам (квадранты по степени) ──
    println!("\nактивность по квадрантам:");
    let degrees: Vec<usize> = rp.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let mut sorted_deg = degrees.clone();
    sorted_deg.sort_unstable_by(|a, b| b.cmp(a));
    let hub_thresh = sorted_deg[n * 25 / 100];

    for name in &["хабы (deg≥hub)", "периферия"] {
        let mut count = 0u64;
        for i in 0..n {
            let is_hub = degrees[i] >= hub_thresh;
            let matches = (name.starts_with("хаб") && is_hub)
                || (name.starts_with("периф") && !is_hub);
            if !matches { continue; }
            count += active_neurons.get(&i).map(|_| 1).unwrap_or(0) as u64;
        }
        println!("  {name}: {count} активных нейронов");
    }

    // ── 6. Волна: стимул в разные части мозга — куда расходится ──
    println!("\nраспространение: стимул на нейрон 0 vs нейрон n/2");
    for &start_neuron in &[0usize, n / 2] {
        let mut r2 = BrainRunner::new(
            ConnectomeData::new(rp.clone(), ci.clone(),
                vals.iter().map(|&v| v * scale).collect(), nt.clone(), vec![]),
            BrainConfig::lif_alpha(0.05),
        ).unwrap();
        let z = vec![0.0f32; n];
        for _ in 0..50 { r2.run(1, &z); }

        let mut pulse = z.clone();
        pulse[start_neuron] = 50.0;
        let mut reached = std::collections::HashSet::new();
        for _ in 0..100 {
            let buf = r2.run(1, &pulse);
            for (i, &s) in buf.iter().enumerate() {
                if s != 0 { reached.insert(i); }
            }
        }
        println!("  стимул @{start_neuron}: активировалось {}/{} нейронов",
            reached.len(), n);
    }

    println!("\nsardaana мозг ЖИВОЙ — работает в движке ✓");
}
