//! Точечная диагностика: первый расходящийся подшаг CPU/GPU.
//! Сравниваем спайки и v на КАЖДОМ подшаге; в первом расходящемся
//! печатаем состояние обеих динамик по первым расходящимся нейронам.
use dioma::creature::{Creature, Sense};

fn region_of(bounds: &[(usize, usize)], i: usize) -> usize {
    for (r, &(a, b)) in bounds.iter().enumerate() {
        if (a..b).contains(&i) {
            return r;
        }
    }
    99
}

fn main() {
    let seed = 7u64;
    let mut cpu = Creature::new(seed);
    let mut gpu = Creature::new_gpu(seed);
    if !gpu.brain.on_gpu() {
        println!("GPU недоступен");
        return;
    }
    let n = cpu.brain.n;
    let mut ext = vec![0.12f32; n];
    for e in &mut ext[0..8] {
        *e += 4.0; // сенсорный стимул слева
    }
    let bounds: Vec<(usize, usize)> = (0..7)
        .map(|r| cpu.brain.region_bounds(r))
        .collect();
    let mut first_mismatch: Option<usize> = None;
    // Три фазы: 12 шагов со стимулом → 8 шагов без → 10 шагов со стимулом.
    let plan: Vec<Vec<f32>> = {
        let mut p = Vec::new();
        for _ in 0..12 {
            p.push(ext.clone());
        }
        let quiet = vec![0.12f32; n];
        for _ in 0..8 {
            p.push(quiet.clone());
        }
        for _ in 0..10 {
            p.push(ext.clone());
        }
        p
    };
    for (step, ext) in plan.iter().enumerate() {
        let mut sc = Vec::new();
        let mut sg = Vec::new();
        cpu.brain.step_into(ext, &mut sc);
        gpu.brain.step_into(ext, &mut sg);
        let vc = cpu.brain.debug_cpu_v();
        let st = gpu.brain.debug_gpu_states();
        let stride = n.div_ceil(64) * 64;
        let mism: Vec<usize> = (0..n).filter(|&i| sc[i] != sg[i]).collect();
        if !mism.is_empty() {
            first_mismatch = Some(step);
            println!("=== ПЕРВОЕ РАСХОЖДЕНИЕ на подшаге {step}: {0} нейронов", mism.len());
            for &i in mism.iter().take(10) {
                let r = region_of(&bounds, i);
                println!(
                    "  нейрон {i} (регион {r}): cpu_spike={} gpu_spike={}",
                    sc[i], sg[i]
                );
                println!(
                    "    cpu v={:.6} | gpu v={:.6} g={:.6} refr={:.3} | diff_v={:.3e}",
                    vc[i],
                    st[0 * stride + i],
                    st[1 * stride + i],
                    st[2 * stride + i],
                    (vc[i] - st[0 * stride + i]).abs()
                );
            }
            // Глобальная сверка v по всем нейронам
            let mut maxdv = 0.0f32;
            let mut maxi = 0usize;
            for i in 0..n {
                let d = (vc[i] - st[0 * stride + i]).abs();
                if d > maxdv {
                    maxdv = d;
                    maxi = i;
                }
            }
            println!(
                "  max |v_cpu - v_gpu| = {maxdv:.3e} у нейрона {maxi} (v_cpu={:.6} v_gpu={:.6})",
                vc[maxi], st[0 * stride + maxi]
            );
            break;
        }
    }
    if first_mismatch.is_none() {
        println!("30 подшагов бит-в-бит (спайки и v)");
    }
    // Дальше — полные тики существа (как gpu_tick)
    for t in 0..12u32 {
        let sense = Sense {
            smell_l: if t % 3 == 0 { 0.6 } else { 0.1 },
            smell_r: if t % 5 == 0 { 0.6 } else { 0.1 },
            wall_l: if t % 7 == 0 { 0.9 } else { 0.0 },
            wall_r: 0.0,
            food_near: t == 11 || t <= 2,
            wall: t % 13 == 0,
            place: 0,
            heading: 0.0,
            see_eat: false,
        };
        let ac = cpu.tick(&sense);
        let ag = gpu.tick(&sense);
        let same = (ac.steer - ag.steer).abs() < 1e-9
            && (ac.drive - ag.drive).abs() < 1e-9
            && ac.eat == ag.eat;
        let wc = cpu.brain.csr_values();
        let wg = gpu.brain.csr_values();
        let wd: f32 = wc
            .iter()
            .zip(&wg)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if wd > 1e-4 {
            for (i, (a, b)) in wc.iter().zip(&wg).enumerate() {
                if (a - b).abs() > 1e-4 {
                    println!(
                        "  вес[{i}]: cpu={a:.6} gpu={b:.6} (δ={:.6})",
                        (a - b).abs()
                    );
                    break;
                }
            }
        }
        println!(
            "t={t} eat(c={},g={}) glu(c={:.0},g={:.0}) same={} wmaxdiff={wd:.6}",
            ac.eat as u8, ag.eat as u8, cpu.glucose(), gpu.glucose(), same
        );
    }
}