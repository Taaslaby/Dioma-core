//! Детерминизм CPU: два свежих CPU-мозга с одинаковым seed и ext.
//! Если CPU сам по себе флуктуирует — GPU здесь ни при чём.
use dioma::creature::Creature;

fn maxdiff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m = 0.0f32;
    let mut mi = 0usize;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
            mi = i;
        }
    }
    (m, mi)
}

fn main() {
    let mut cpu1 = Creature::new(7u64);
    let mut cpu2 = Creature::new(7u64);
    let n = cpu1.brain.n;
    let mut ext = vec![0.12f32; n];
    for e in &mut ext[0..8] {
        *e += 4.0;
    }
    for step in 0..40 {
        let mut s1 = Vec::new();
        let mut s2 = Vec::new();
        cpu1.brain.step_into(&ext, &mut s1);
        cpu2.brain.step_into(&ext, &mut s2);
        let v1 = cpu1.brain.debug_cpu_v();
        let v2 = cpu2.brain.debug_cpu_v();
        let g1 = cpu1.brain.debug_cpu_g();
        let g2 = cpu2.brain.debug_cpu_g();
        let (dv, _) = maxdiff(&v1, &v2);
        let (dg, mi) = maxdiff(&g1, &g2);
        if s1 != s2 || dv > 0.0 || dg > 0.0 {
            println!(
                "шаг {step}: спайки={} maxdv={dv:.3e} maxdg={dg:.3e} @#{mi} (g1={:.9} g2={:.9})",
                if s1 != s2 { "РАЗНЫЕ" } else { "равны" },
                g1[mi],
                g2[mi]
            );
        }
    }
    println!("CPU-детерминизм: прогон завершён");
}