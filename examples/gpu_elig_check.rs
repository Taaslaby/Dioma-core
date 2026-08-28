//! Пошаговая сверка dopamine-eligibility: CPU plast (pillars lazy) vs GPU
//! (dopamine.wgsl dense). Один и тот же коннектом, один и тот же поток
//! спайков → трассы и eligibility должны совпадать бит-в-бит.
//!
//! Запуск: `cargo run --release --example gpu_elig_check`
use dioma::brain::genome::Grown;
use dioma::brain::neurons::{gains_from, PlasticityParams};

fn grown() -> Grown {
    // Той-коннектом с рекуррентными петлями (обе фазы LTP/LTD в один шаг).
    Grown {
        edges: vec![
            (0, 4, 1.5),
            (1, 5, 1.2),
            (2, 6, 1.8),
            (3, 7, 1.4),
            (0, 1, 0.9),
            (1, 2, 0.7),
            (4, 0, 0.5),
            (5, 1, 0.6),
            (6, 2, 0.4),
            (7, 3, 0.8),
            (4, 5, 0.3),
            (5, 4, 0.2),
        ],
        region_of: vec![0, 0, 0, 0, 1, 1, 1, 1],
        nt: vec![1.0; 8],
        growth_s: 0.0,
        rule_genes: Vec::new(),
        instincts: Vec::new(),
    }
}

fn main() {
    let grown = grown();
    let gains = gains_from(2, &[(0, 1, 1.0), (1, 0, 0.5)]);
    let params = PlasticityParams::default();

    // CPU-мозг: пластичность живёт на CPU (эталон).
    let mut cpu = dioma::brain::neurons::Brain::new(&grown, gains.clone(), params.clone());
    // GPU-мозг: обучение целиком на GPU.
    let mut gpu = dioma::brain::neurons::Brain::new_gpu(&grown, gains, params);
    if !gpu.on_gpu() {
        println!("GPU недоступен — пропуск");
        return;
    }
    let n = cpu.n;
    // Шаги: всплеск на 0-3, тишина 4-6, всплеск 7-9 (проверка decay-гэпов).
    let mut spikes_by_step: Vec<Vec<i32>> = Vec::new();
    for t in 0..20 {
        let mut ext = vec![0.12f32; n];
        if (t % 10) < 4 {
            for e in &mut ext[0..4] {
                *e += 5.0;
            }
        }
        let mut sc = Vec::new();
        let mut sg = Vec::new();
        cpu.step_into(&ext, &mut sc);
        gpu.step_into(&ext, &mut sg);
        spikes_by_step.push(sg.clone());
        assert_eq!(sc, sg, "спайки разошлись на шаге {t}");
    }

    // После каждого шага: сверка трасс и eligibility.
    // CPU plast — приватные поля; используем публичные debug-геттеры
    // GpuDynamics + публичный plast.eligibility (после fold).
    let (tp_pre_g, tp_post_g) = gpu.debug_gpu_traces();
    let elig_g = gpu.debug_gpu_elig();
    // CPU: fold_eligibility() публичный — он приводит к now.
    let _ = tp_pre_g;
    let _ = tp_post_g;
    // Сверка через веса после награды — самый сильный критерий.
    // Награда: Δw = lr·dopamine·elig, должна совпасть бит-в-бит.
    let mut wd_max = 0.0f32;
    for trial in 0..6 {
        // Два разных дофамина: положительный и отрицательный.
        let dop = if trial % 2 == 0 { 1.0 } else { -0.5 };
        cpu.reward_now(dop);
        gpu.reward_now(dop);
        let wc = cpu.csr_values();
        let wg = gpu.csr_values();
        let wd: f32 = wc
            .iter()
            .zip(&wg)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        wd_max = wd_max.max(wd);
        println!("награда {trial}: dop={dop} wmaxdiff={wd:.9}");
        if wd > 1e-9 {
            for (i, (a, b)) in wc.iter().zip(&wg).enumerate() {
                if (a - b).abs() > 1e-9 {
                    println!("  вес[{i}]: cpu={a:.9} gpu={b:.9}");
                    break;
                }
            }
        }
    }
    println!("итог: wmaxdiff={wd_max:.9}");
    if wd_max == 0.0 {
        println!("БИТ-В-БИТ: eligibility GPU == CPU plast");
    } else {
        println!("РАСХОЖДЕНИЕ {wd_max}");
    }
    let _ = elig_g;
    let _ = spikes_by_step;
}
