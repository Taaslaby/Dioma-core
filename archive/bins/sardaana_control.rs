//! Контрольный эксперимент: sarDNA vs случайный граф (одинаковая плотность).
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};
use taas_format::TaasReader;
use std::collections::HashSet;

#[allow(dead_code)]
struct BenchResult {
    stim_spikes: f64,
    stim_neurons: usize,
    silence_spikes: f64,
    reached_far: usize,
}

fn bench(name: &str, conn: &ConnectomeData, n: usize) -> BenchResult {
    let mut runner = BrainRunner::new(conn.clone(), BrainConfig::lif_alpha(0.05)).unwrap();
    let z = vec![0.0f32; n];
    for _ in 0..100 { runner.run(1, &z); }

    let mut stim = z.clone();
    for cell in stim.iter_mut().take(20) { *cell = 30.0; }
    let mut stim_spikes = 0i64;
    let mut stim_neurons = HashSet::new();
    for _ in 0..200 {
        let buf = runner.run(1, &stim);
        for (i, &s) in buf.iter().enumerate() {
            if s != 0 { stim_spikes += 1; stim_neurons.insert(i); }
        }
    }
    let mut silence_spikes = 0i64;
    for _ in 0..400 {
        let buf = runner.run(1, &z);
        silence_spikes += buf.iter().map(|&s| s as i64).sum::<i64>();
    }

    // Распространение.
    let mut r3 = BrainRunner::new(conn.clone(), BrainConfig::lif_alpha(0.05)).unwrap();
    for _ in 0..50 { r3.run(1, &z); }
    let mut pulse = z.clone();
    pulse[0] = 80.0;
    let mut far = HashSet::new();
    for _ in 0..300 {
        let buf = r3.run(1, &pulse);
        for (i, &s) in buf.iter().enumerate() {
            if s != 0 && i > n / 2 { far.insert(i); }
        }
    }

    println!("  {name}: стимул={stim_spikes} сп · уникальных={} · тишина={silence_spikes} · дальнее={}",
        stim_neurons.len(), far.len());
    BenchResult { stim_spikes: stim_spikes as f64, stim_neurons: stim_neurons.len(), silence_spikes: silence_spikes as f64, reached_far: far.len() }
}

fn main() {
    println!("КОНТРОЛЬ: sarDNA vs случайный граф");

    let path = "D:/Проекты/asm_engines/sardaanaCM/data/generated/SarDNA_Pro_best.taas";
    let r = TaasReader::from_file(path).unwrap();
    let rp: Vec<i32> = r.get_i32("row_ptr").unwrap();
    let ci: Vec<i32> = r.get_i32("col_indices").unwrap();
    let vals: Vec<f32> = r.get_f32("values").unwrap();
    let nt: Vec<f32> = r.get_f32("nt_types").unwrap();
    let n = rp.len().saturating_sub(1);
    let n_syn = ci.len();

    let w_max_abs = vals.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if w_max_abs > 0.0 { 2.0 / w_max_abs } else { 1.0 };
    let vals_s: Vec<f32> = vals.iter().map(|&v| v * scale).collect();

    // Случайный контроль — LCG детерминированный.
    let mut lcg: u64 = 0xDEAD_BEEF;
    let mut lcg_next = || -> f64 {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (lcg >> 33) as f64 / (u32::MAX as f64)
    };
    let mut rand_edges: Vec<(usize, usize)> = Vec::with_capacity(n_syn);
    while rand_edges.len() < n_syn {
        let pre = (lcg_next() * n as f64) as usize % n;
        let post = (lcg_next() * n as f64) as usize % n;
        if pre == post { continue; }
        if !rand_edges.contains(&(pre, post)) { rand_edges.push((pre, post)); }
    }
    rand_edges.sort();
    let mut rp_r = vec![0i32; n + 1];
    let mut ci_r = Vec::new();
    for (neuron, rpe) in rp_r.iter_mut().take(n).enumerate() {
        *rpe = ci_r.len() as i32;
        for &(pre, post) in rand_edges.iter() {
            if pre == neuron { ci_r.push(post as i32); }
        }
    }
    rp_r[n] = ci_r.len() as i32;
    let vals_r: Vec<f32> = (0..ci_r.len()).map(|_| (lcg_next() * 4.0 - 2.0) as f32).collect();

    let conn_sar = ConnectomeData::new(rp.clone(), ci.clone(), vals_s, nt.clone(), vec![]);
    let conn_rand = ConnectomeData::new(rp_r.clone(), ci_r.clone(), vals_r, nt, vec![]);

    // Бенчмарк × 3 прогона.
    let mut sar = Vec::new();
    let mut rnd = Vec::new();
    for trial in 0..3 {
        if trial == 0 { continue; } // пропускаем разминку
        sar.push(bench("sarDNA", &conn_sar, n));
        rnd.push(bench("random", &conn_rand, n));
    }

    let avg = |v: &[BenchResult], f: fn(&BenchResult) -> f64| -> f64 {
        v.iter().map(f).sum::<f64>() / v.len() as f64
    };
    let get_spikes = |b: &BenchResult| b.stim_spikes;
    let get_neurons = |b: &BenchResult| b.stim_neurons as f64;
    let get_silence = |b: &BenchResult| b.silence_spikes;

    println!("\n╔══ ВЕРДИКТ (среднее по {} прогонам) ══", sar.len());
    println!("║ sarDNA : спайков={:.0} нейронов={} тишина={:.0}",
        avg(&sar, get_spikes), avg(&sar, get_neurons), avg(&sar, get_silence));
    println!("║ random : спайков={:.0} нейронов={} тишина={:.0}",
        avg(&rnd, get_spikes), avg(&rnd, get_neurons), avg(&rnd, get_silence));

    let s = avg(&sar, get_spikes);
    let rr = avg(&rnd, get_spikes);
    let verdict = if s > rr * 1.2 { "sarDNA ЗНАЧИМО лучше ✓" }
        else if rr > s * 1.2 { "случайный лучше — сардана мусор ✗" }
        else { "неразличимы" };
    println!("╚══ {verdict}");
}
