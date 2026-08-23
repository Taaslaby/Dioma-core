//! Проба роста: рёбра/топология по нескольким сидам (диагностика санитизации).
use dioma::grow::{agent_layout, grow, reflex_layout, Steering};

fn main() {
    for seed in [0x0000_AE11u64, 1, 2, 3, 0x00C0_FFEE, 42] {
        let g = grow(&agent_layout(), &Steering::default(), seed);
        let mut own = 0usize;
        let mut cross = 0usize;
        for &(pre, post, _) in &g.edges {
            let (rp, rq) = (g.region_of[pre], g.region_of[post]);
            let to_ml = rq == 3;
            let to_mr = rq == 4;
            if (rp == 0 && to_ml) || (rp == 1 && to_mr) {
                own += 1;
            }
            if (rp == 0 && to_mr) || (rp == 1 && to_ml) {
                cross += 1;
            }
        }
        println!(
            "seed={seed:#x}: edges={} own_S_M={own} cross={cross}",
            g.edges.len()
        );
    }
    let r = grow(&reflex_layout(), &Steering::default(), 0x00C0_FFEE);
    println!("reflex 0xC0FFEE: edges={}", r.edges.len());
}
