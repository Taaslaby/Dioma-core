//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_organs.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-organs — органный мозг: HH спина + HAAP средний + LIF кора
//! Description: A: спинальный рефлекс (латентность боли→мотор, HH-точность);
//!              B: мотивация — голод среднего мозга поднимает ток в кору;
//!              C: кора учится на дофамине (cue→out), контроль без награды.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::brain::{Brain, PlasticityParams};
use dioma::organs::{Axon, OrganBrain, OrganId, WorldInputs};
use dioma::Grown;
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};

// ── Спина: 10 S (боль) → 10 I (inh) → 10 M, проводимостные синапсы ────────
const N_S: usize = 10;
const N_I: usize = 10;
const N_M: usize = 10;
const N_SPINE: usize = N_S + N_I + N_M;

fn spine_conn() -> ConnectomeData {
    let mut rp = vec![0i32; N_SPINE + 1];
    let mut ci = Vec::new();
    let mut vals = Vec::new();
    let push = |pre: usize, posts: std::ops::Range<usize>, w: f32, rp: &mut Vec<i32>, ci: &mut Vec<i32>, vals: &mut Vec<f32>| {
        rp[pre] = ci.len() as i32;
        for p in posts {
            ci.push(p as i32);
            vals.push(w);
        }
    };
    // S→M моносинапс (быстрая дуга), S→I, I→M (полисинаптическая ветвь).
    // Вес = скачок проводимости (мкСм-единицы движка).
    for s in 0..N_S {
        let m = N_S + N_I + (s % N_M);
        push(s, m..m + 1, 0.40, &mut rp, &mut ci, &mut vals);
        push(s, (N_S + s % N_I)..(N_S + s % N_I + 1), 0.25, &mut rp, &mut ci, &mut vals);
    }
    for i in 0..N_I {
        let m = N_S + N_I + (i * 3 % N_M);
        push(N_S + i, m..m + 1, 0.30, &mut rp, &mut ci, &mut vals);
    }
    for n in (0..N_SPINE).rev() {
        rp[n + 1] = ci.len() as i32;
    }
    rp[0] = 0;
    // nt: S exc, I inh, M exc.
    let mut nt = vec![1.0f32; N_SPINE];
    for i in 0..N_I {
        nt[N_S + i] = -1.0;
    }
    let mut conn = ConnectomeData::new(rp, ci, vals, nt, vec![]);
    // Проводимостной путь требует рецепторы по синапсам (nt-знак → AMPA/GABA_A).
    conn.build_receptor_types();
    conn
}

// ── Кора: cueA(12) → out(12), cueB(12) → out(12), LIF+STDP ─────────────────
const N_POOL: usize = 12;

fn cortex_grown() -> Grown {
    let n = 3 * N_POOL;
    let mut edges = Vec::new();
    // Каждый out-нейрон читает 3 пресинапса своего пула (конвергенция),
    // вес ~ рабочий диапазон арены (2.0).
    for j in 0..N_POOL {
        for k in [0usize, 4, 8] {
            edges.push(((j + k) % N_POOL, 2 * N_POOL + j, 2.0)); // A→out
            edges.push((N_POOL + (j + k + 3) % N_POOL, 2 * N_POOL + j, 2.0)); // B→out
        }
    }
    Grown {
        edges,
        region_of: vec![0; N_POOL]
            .into_iter()
            .chain(vec![1; N_POOL])
            .chain(vec![2; N_POOL])
            .collect(),
        nt: vec![1.0; n],
        growth_s: 0.0,
        rule_genes: Vec::new(),
    }
}

fn main() {
    println!("poc-organs: HH спина + HAAP средний мозг + LIF кора");
    let t0 = std::time::Instant::now();

    // ── A. Спинальный рефлекс: латентность боли→мотор на подшагах ──
    let mut spine = BrainRunner::new(spine_conn(), BrainConfig::hh_synaptic(0.05)).unwrap();
    let zero = vec![0.0f32; N_SPINE];
    // Разминка.
    for _ in 0..200 {
        spine.run(1, &zero);
    }
    let pain = {
        let mut e = zero.clone();
        for cell in e.iter_mut().take(N_S) {
            *cell = 15.0;
        }
        e
    };
    let mut latency_ms: Option<f32> = None;
    let mut motor_spikes = 0i64;
    for step in 0..400 {
        let buf = spine.run(1, &pain);
        let m: i64 = buf[N_S + N_I..].iter().map(|&s| s as i64).sum();
        motor_spikes += m;
        if m > 0 && latency_ms.is_none() {
            latency_ms = Some(step as f32 * 0.05);
        }
    }
    println!(
        "A спина HH: латентность боли→мотор = {:?} мс, моторных спайков за 20 мс = {motor_spikes}",
        latency_ms
    );

    // ── Сборка органного мозга ──
    const N_MID: usize = 24;
    // Гейны коры 3×3: A→out и B→out открыты, самосвязей нет.
    let cortex_gains = {
        let mut g = vec![vec![0.0f32; 3]; 3];
        g[0][2] = 1.0;
        g[1][2] = 1.0;
        g
    };
    let mut organs = OrganBrain::new(
        spine_conn(),
        cortex_grown(),
        cortex_gains.clone(),
        PlasticityParams::default(),
        N_MID,
    );
    // Средний мозг → кора: мотивационный тоник (спайки за тик × вес).
    organs.add_axon(Axon {
        from: OrganId::Midbrain,
        pre: (0..N_MID).collect(),
        to: OrganId::Cortex,
        post: (0..3 * N_POOL).collect(),
        weight_per_spike: 0.006,
    });

    // ── B. Мотивация: голод среднего мозга поднимает фон коры ──
    let cortex_rate = |organs: &mut OrganBrain, hungry: bool| -> f64 {
        let mut total = 0i64;
        for _ in 0..40 {
            let mut w = WorldInputs::default();
            if hungry {
                w.midbrain = vec![3.5; N_MID]; // голод-локус (реле среднего мозга)
            }
            let s = organs.tick(&w);
            total += s.of(OrganId::Midbrain).iter().map(|&x| x as i64).sum::<i64>();
        }
        total as f64 / 40.0
    };
    let mid_quiet = cortex_rate(&mut organs, false);
    let mid_hungry = cortex_rate(&mut organs, true);
    println!(
        "B средний мозг HAAP: спайков/тик тихо = {mid_quiet}, при голоде = {mid_hungry} ({:+.0}%)",
        (mid_hungry / mid_quiet.max(1.0) - 1.0) * 100.0
    );

    // ── C. Кора учится: cue A + награда, cue B — нет ──
    let (w_a0, cnt_a) = organs.cortex.mean_path_weight(0, 2);
    let (w_b0, _) = organs.cortex.mean_path_weight(1, 2);
    // Тонический фон коры — как во всех мирах диомы.
    let tonic = vec![0.12f32; 3 * N_POOL];
    let mut cue_a = tonic.clone();
    for cell in cue_a.iter_mut().take(N_POOL) {
        *cell += 3.0;
    }
    let mut cue_b = tonic.clone();
    for cell in cue_b.iter_mut().skip(N_POOL).take(N_POOL) {
        *cell += 3.0;
    }
    // Проверка: cue вообще зажигает out?
    let mut out_fires = 0i64;
    for _ in 0..10 {
        let s = organs.tick(&WorldInputs {
            cortex: cue_a.clone(),
            ..Default::default()
        });
        out_fires += s.of(OrganId::Cortex)[2 * N_POOL..]
            .iter()
            .map(|&x| x as i64)
            .sum::<i64>();
    }
    println!("   прогрев: out спайков/тик при cue A = {}", out_fires as f64 / 10.0);
    for trial in 0..15 {
        // Проба A: cue → награда (дофамин в кору сразу после пробы).
        for _ in 0..20 {
            organs.tick(&WorldInputs {
                cortex: cue_a.clone(),
                ..Default::default()
            });
        }
        organs.cortex.reward_now(1.0);
        // Окно допустимости закрыто: след A не должен дожить до B.
        organs.cortex.clear_eligibility();
        // Проба B: cue без награды; её след тоже гасим до следующей награды.
        for _ in 0..20 {
            organs.tick(&WorldInputs {
                cortex: cue_b.clone(),
                ..Default::default()
            });
        }
        organs.cortex.clear_eligibility();
        if (trial + 1) % 5 == 0 {
            let (wa, _) = organs.cortex.mean_path_weight(0, 2);
            let (wb, _) = organs.cortex.mean_path_weight(1, 2);
            println!("   проба {:+>2}: w(A→out) = {wa:.3}  w(B→out) = {wb:.3}", trial + 1);
        }
    }
    let (w_a1, _) = organs.cortex.mean_path_weight(0, 2);
    let (w_b1, _) = organs.cortex.mean_path_weight(1, 2);
    println!(
        "C кора LIF+STDP: w(A→out) {w_a0:.3}→{w_a1:.3} ({:+.1}%), w(B→out) {w_b0:.3}→{w_b1:.3} ({:+.1}%), путей A→out ×{cnt_a}",
        (w_a1 / w_a0 - 1.0) * 100.0,
        (w_b1 / w_b0 - 1.0) * 100.0,
    );
    let verdict = if (w_a1 - w_b1).abs() < 1e-4 {
        "ничья (обучения нет)".to_string()
    } else if w_a1 > w_b1 {
        "A (награждённая) — верно".to_string()
    } else {
        "B (!) — ошибка".to_string()
    };
    println!("verdict: кора предпочитает путь {verdict}");

    // ── Контроль коры без дофамина ──
    let mut ctrl = Brain::new(&cortex_grown(), cortex_gains, PlasticityParams::default());
    let (c0, _) = ctrl.mean_path_weight(0, 2);
    for _ in 0..15 {
        for _ in 0..20 {
            ctrl.step(&cue_a);
        }
        // Награды нет.
        for _ in 0..20 {
            ctrl.step(&cue_b);
        }
    }
    let (c1, _) = ctrl.mean_path_weight(0, 2);
    println!(
        "контроль (D=0): w(A→out) {c0:.3}→{c1:.3} ({:+.1}%)",
        (c1 / c0 - 1.0) * 100.0
    );

    println!("\nполное время: {:.2} с", t0.elapsed().as_secs_f64());
}
