//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_cookiecutter.rs
//! Class:       SemantikaEngine
//! Purpose:     CookieCutter генератор коннектомов на чистом Rust
//! Description: GMM сэмплирует состояния нейронов в 4D пространстве,
//!              EdgePredictor (простая функция) предсказывает связи,
//!              top-k фильтр ограничивает fan-out. Результат — CSR.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::rng::XorShift;
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};

const N_NEURONS: usize = 256;
const STATE_DIM: usize = 4;
const N_GMM_COMPONENTS: usize = 8;
const EDGE_THRESHOLD: f32 = 0.3;

// ─── Простая GMM ────────────────────────────────────────────────────────────
struct SimpleGmm {
    means: Vec<[f32; STATE_DIM]>,
    sigmas: [f32; STATE_DIM],
    rng: XorShift,
}

impl SimpleGmm {
    fn new(rng: XorShift) -> Self {
        // Случайные центры кластеров в [-1, 1]^4.
        let mut means = Vec::new();
        for _ in 0..N_GMM_COMPONENTS {
            let mut m = [0.0f32; STATE_DIM];
            for d in 0..STATE_DIM {
                m[d] = (dioma::rng::XorShift::new(42 + d as u64).unit() * 2.0 - 1.0) as f32;
            }
            means.push(m);
        }
        Self { means, sigmas: [0.25; STATE_DIM], rng }
    }

    fn sample(&mut self) -> [f32; STATE_DIM] {
        // Выбираем случайный кластер.
        let ci = (self.rng.unit() * self.means.len() as f64) as usize % self.means.len();
        let mut out = [0.0f32; STATE_DIM];
        for d in 0..STATE_DIM {
            out[d] = self.means[ci][d] + (self.rng.unit() as f32 - 0.5) * self.sigmas[d] * 2.0;
        }
        out
    }

    fn sample_n(&mut self, n: usize) -> Vec<[f32; STATE_DIM]> {
        (0..n).map(|_| self.sample()).collect()
    }
}

// ─── EdgePredictor: простая функция похожести состояний ────────────────────
fn predict_edge(a: &[f32; STATE_DIM], b: &[f32; STATE_DIM]) -> f32 {
    // Cosine similarity → sigmoid.
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb).max(1e-6);
    1.0 / (1.0 + (-cos * 4.0).exp()) // sigmoid(cos × 4)
}

fn main() {
    println!("CookieCutter: GMM → EdgePredictor → CSR");

    let mut rng = XorShift::new(42);
    let mut gmm = SimpleGmm::new(XorShift::new(43));

    // ── 1. Сэмплируем состояния нейронов ──
    let states = gmm.sample_n(N_NEURONS);
    println!("состояния: {} × {}D", N_NEURONS, STATE_DIM);

    // ── 2. Предсказываем рёбра ──
    // Для каждой пары считаем вероятность связи.
    let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..N_NEURONS {
        for j in 0..N_NEURONS {
            if i == j { continue; }
            let p = predict_edge(&states[i], &states[j]);
            if p > EDGE_THRESHOLD {
                candidates.push((i, j, p));
            }
        }
    }
    println!("кандидатов после порога {:.1}: {}", EDGE_THRESHOLD, candidates.len());

    // Top-k на нейрон: сортируем по весу, оставляем max_k.
    let max_k = 12usize;
    let mut by_pre: std::collections::HashMap<usize, Vec<(usize, f32)>> = std::collections::HashMap::new();
    for &(pre, post, w) in &candidates {
        by_pre.entry(pre).or_default().push((post, w));
    }
    let mut all_edges: Vec<(usize, usize, f32)> = Vec::new();
    for (_pre, mut targets) in by_pre {
        targets.sort_by(|a, b| b.1.total_cmp(&a.1));
        targets.truncate(max_k);
        for (post, w) in targets {
            all_edges.push((_pre, post, w));
        }
    }
    println!("после top-k: {} синапсов", all_edges.len());

    // ── 3. Строим CSR и запускаем ──
    all_edges.sort_by_key(|&(a, _, _)| a);
    let mut rp = vec![0i32; N_NEURONS + 1];
    let mut ci = Vec::new();
    let mut vals = Vec::new();
    let mut last = 0usize;
    for &(pre, post, w) in &all_edges {
        while last <= pre {
            rp[last] = ci.len() as i32;
            last += 1;
        }
        ci.push(post as i32);
        vals.push(w * 10.0); // масштаб в рабочий диапазон
    }
    rp[N_NEURONS] = ci.len() as i32;

    let nt = vec![1.0f32; N_NEURONS]; // все возбуждающие
    let conn = ConnectomeData::new(rp.clone(), ci.clone(), vals.clone(), nt, vec![]);

    // ── 4. Запускаем LIF-α сеть ──
    let mut runner = BrainRunner::new(conn, BrainConfig::lif_alpha(0.05)).unwrap();
    let zero = vec![0.0f32; N_NEURONS];
    for _ in 0..100 { runner.run(1, &zero); }

    let mut stim = zero.clone();
    for cell in stim.iter_mut().take(20) { *cell = 5.0; }

    let mut stim_spikes = 0i64;
    let mut stim_active = HashSet::new();
    for _ in 0..200 {
        let buf = runner.run(1, &stim);
        for (i, &s) in buf.iter().enumerate() {
            if s != 0 { stim_spikes += 1; stim_active.insert(i); }
        }
    }

    let mut silence = 0i64;
    for _ in 0..200 {
        let buf = runner.run(1, &zero);
        silence += buf.iter().map(|&s| s as i64).sum::<i64>();
    }

    println!("\nрезультат:");
    println!("  стимул: {stim_spikes} спайков от {} нейронов", stim_active.len());
    println!("  тишина: {silence} спайков");
    if stim_spikes > 50 && stim_active.len() > 10 {
        println!("  ✓ CookieCutter мозг ЖИВОЙ");
    } else {
        println!("  ✗ слишком тихо — нужно больше связей");
    }
}

use std::collections::HashSet;
