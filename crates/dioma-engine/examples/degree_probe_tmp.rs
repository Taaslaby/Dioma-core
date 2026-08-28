// ВРЕМЕННЫЙ зонд (удалить): степень/синапсы по раскладкам на текущем HEAD.
use dioma_engine::brain::growth::grow::{grow, stage_b_layout, stage_c_lite_layout, Layout, Steering};

fn stats(name: &str, l: &Layout) {
    let st = Steering::default();
    let t0 = std::time::Instant::now();
    let grown = grow(l, &st, 42);
    let ms = t0.elapsed().as_millis();
    let n = l.n();
    let syn = grown.edges.len();
    // degree по пре-нейронам для рекуррентной пары CORTEX_E->CORTEX_E (если есть)
    let region_of = l.region_of();
    let mut self_pairs = vec![0usize; n];
    for &(pre, post, _w) in &grown.edges {
        if region_of[pre] == region_of[post] {
            self_pairs[pre] += 1;
        }
    }
    let cortex: Vec<usize> = (0..n).filter(|&i| region_of[i] == 9).collect();
    let deg: usize = cortex.iter().sum::<usize>() / cortex.len().max(1);
    let max_deg = cortex.iter().map(|&i| self_pairs[i]).max().unwrap_or(0);
    println!("{name}: {n} нейронов, {syn} синапсов, рост {ms} мс; CORTEX_E self-degree avg={deg} max={max_deg}");
}

fn main() {
    stats("StageB", &stage_b_layout());
    stats("StageC-lite", &stage_c_lite_layout());
}


