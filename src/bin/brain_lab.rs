//! Лаборатория мозга: сетка (haap_syn_scale × haap_syn_cap) на арене,
//! выбор победителя по стабильности+скорости, долгий прогон с кривой обучения.
use dioma::arena::{arena_brain_with, grow_agent, Agent, Genome, TONIC_DRIVE};
use dioma::brain::{BrainModel, PlasticityParams};

const SEIZURE_TICK_SPIKES: i64 = 3000; // из 96×500=48k возможных: здорово ≤ ~600
const GRID_EPS: usize = 6;
const LONG_EPS: usize = 30;
const TIMEOUT: usize = 600;

/// Один эпизод (канонический, с жеванием) + пауза сытости после еды.
/// Возвращает (тики до еды, макс спайков за тик, тиков тишины после еды).
fn episode(
    brain: &mut dioma::brain::Brain,
    agent: &mut Agent,
    genome: &Genome,
    _turn_gain: f64,
) -> (usize, i64, usize) {
    let (ticks, max_tick) =
        dioma::arena::run_episode_debug(brain, agent, genome, true);
    // Сытость: после еды вход нулевой — сколько тиков мозг молчит
    // (адаптация u) до первой вспышки.
    let zero = vec![TONIC_DRIVE; brain.n];
    let mut quiet = 0usize;
    for _ in 0..200 {
        let mut any = 0i32;
        for _ in 0..brain.steps_per_tick() {
            let spikes = brain.step(&zero);
            any += spikes.iter().sum::<i32>();
        }
        if any > 0 {
            break;
        }
        quiet += 1;
    }
    (ticks, max_tick, quiet)
}

fn run_config(scale: f32, cap: f32, eps: usize) -> (f64, i64, f64, f32) {
    run_config_full(scale, cap, 1, 0.25, true, eps).0
}

/// Полный прогон: (метрики эпизодов, стартовый w(S→M), финальный w(S→M)).
fn run_config_full(
    scale: f32,
    cap: f32,
    homeo_every: usize,
    homeo_rate: f32,
    dopa_scaled: bool,
    eps: usize,
) -> ((f64, i64, f64, f32), f32, f32) {
    let mut genome = Genome::wildtype(0x0000_AE11);
    genome.brain_model = BrainModel::Haap;
    let grown = grow_agent(&genome);
    let mut brain = arena_brain_with(&grown, &genome, |p: &mut PlasticityParams| {
        p.haap_syn_scale = scale;
        p.haap_syn_cap = cap;
        p.homeo_every_ticks = homeo_every;
        p.homeo_rate = homeo_rate;
        p.haap_dopamine_scaled = dopa_scaled;
    });
    let (w0, _) = brain.mean_path_weight(0, 3);
    let mut agent = Agent::new(genome.seed);
    let mut eaten = 0usize;
    let mut sum_ticks = 0usize;
    let mut worst_tick = 0i64;
    let mut sat_sum = 0usize;
    for _ in 0..eps {
        let (t, mt, q) = episode(&mut brain, &mut agent, &genome, genome.turn_gain);
        if t < TIMEOUT {
            eaten += 1;
            sum_ticks += t;
        }
        worst_tick = worst_tick.max(mt);
        sat_sum += q;
    }
    let (w1, _) = brain.mean_path_weight(0, 3);
    (
        (
            eaten as f64 / eps as f64,
            worst_tick,
            sum_ticks as f64 / eaten.max(1) as f64,
            sat_sum as f32 / eps as f32,
        ),
        w0,
        w1,
    )
}

fn main() {
    println!("=== СЕТКА: syn_scale × syn_cap (арена, {GRID_EPS} эп.) ===");
    println!("scale  cap   eat%  сред.тicks  worstTick  сытость(тиков)");
    let mut winner: Option<(f32, f32)> = None;
    let mut winner_score = f64::INFINITY;
    for &scale in &[0.15f32, 0.20, 0.25, 0.35] {
        for &cap in &[0.30f32, 0.40, 0.60] {
            let (eat, worst, mean_t, sat) = run_config(scale, cap, GRID_EPS);
            let stable = worst < SEIZURE_TICK_SPIKES;
            println!(
                "{scale:.2}  {cap:.2}  {:4.0}%  {:10.1}  {worst:9}  {sat:8.1} {}",
                eat * 100.0,
                mean_t,
                if stable { "" } else { " ← СУДОРОГА" },
            );
            // Победитель: стабилен, ест всё, быстрее всех.
            if stable && eat >= 0.99 {
                let score = mean_t;
                if score < winner_score {
                    winner_score = score;
                    winner = Some((scale, cap));
                }
            }
        }
    }

    let Some((scale, cap)) = winner else {
        println!("\nПобедителя нет — все конфиги либо судорожат, либо не едят.");
        return;
    };

    // ─── Фаза 2: обучение — сетка по тормозу гомеостаза и шагу дофамина ──
    println!("\n=== ОБУЧЕНИЕ: homeo × dopa_scale (scale={scale} cap={cap}, {GRID_EPS} эп.) ===");
    println!("homeo  rate  dopa×s   eat%  тики(ср)  worst  w: старт→финал (Δ%)");
    let mut learn_winner: Option<(usize, f32, bool)> = None;
    let mut learn_score = f64::INFINITY;
    for &(he, hr, ds) in &[
        (1usize, 0.25f32, true),
        (1, 0.25, false),
        (1, 0.08, false),
        (0, 0.25, false),
        (0, 0.08, false),
    ] {
        let ((eat, worst, mean_t, _sat), w0, w1) =
            run_config_full(scale, cap, he, hr, ds, GRID_EPS);
        let dw = if w0 > 1e-6 { (w1 / w0 - 1.0) * 100.0 } else { 0.0 };
        let stable = worst < SEIZURE_TICK_SPIKES;
        println!(
            "{he:>3}  {hr:.2}  {ds:>5}  {:4.0}%  {:8.1}  {worst:5}  {w0:.4}→{w1:.4} ({dw:+.1}%) {}",
            eat * 100.0,
            mean_t,
            if stable { "" } else { " ← СУДОРОГА" },
        );
        // Учимся = веса сдвинулись; стабильно; едим.
        if stable && eat >= 0.99 && dw.abs() > 5.0 {
            let score = mean_t;
            if score < learn_score {
                learn_score = score;
                learn_winner = Some((he, hr, ds));
            }
        }
    }

    // ─── Фаза 3: рефлекс-протокол на HAAP (обучение измеримо средним) ──
    println!("\n=== РЕФЛЕКС HAAP: дофаминовая пластичность S→M ===");
    for &(he, ds) in &[(0usize, true), (1, true), (0, false)] {
        haap_reflex(scale, cap, he, ds);
    }

    // ─── Фаза 4: живость — когнитивная схема (WS пейсмекер, WM плато) ──
    cognitive_aliveness(scale, cap);

    let Some((he, hr, ds)) = learn_winner else {
        println!("\nОбучающегося стабильного конфига в арене не найдено — долгий прогон арены пропущен.");
        return;
    };
    println!(
        "\n=== ФИНАЛИСТ: scale={scale} cap={cap} homeo={he}/{hr} dopa_scaled={ds} — {LONG_EPS} эп. ==="
    );
    let mut genome = Genome::wildtype(0x0000_AE11);
    genome.brain_model = BrainModel::Haap;
    let grown = grow_agent(&genome);
    let mut brain = arena_brain_with(&grown, &genome, |p: &mut PlasticityParams| {
        p.haap_syn_scale = scale;
        p.haap_syn_cap = cap;
        p.homeo_every_ticks = he;
        p.homeo_rate = hr;
        p.haap_dopamine_scaled = ds;
    });
    let (w_start, _) = brain.mean_path_weight(0, 3);
    let mut agent = Agent::new(genome.seed);
    let mut window: Vec<usize> = Vec::new();
    let mut first_half: Vec<usize> = Vec::new();
    let mut second_half: Vec<usize> = Vec::new();
    for ep in 1..=LONG_EPS {
        let (t, mt, _q) = episode(&mut brain, &mut agent, &genome, genome.turn_gain);
        window.push(t);
        if ep <= LONG_EPS / 2 {
            first_half.push(t);
        } else {
            second_half.push(t);
        }
        if ep % 10 == 0 {
            let m: f64 = window.iter().map(|&x| x as f64).sum::<f64>() / window.len() as f64;
            let (w, cnt) = brain.mean_path_weight(0, 3);
            println!(
                "эп{ep:3}: время(ср.10)={m:6.1} тиков  w(S→M)={w:.4} ×{cnt}  worstTick={mt}"
            );
            window.clear();
        }
    }
    let mean = |v: &[usize]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
    let (w_end, _) = brain.mean_path_weight(0, 3);
    let (lo, hi) = brain.values_minmax();
    println!(
        "ИТОГ: первая половина {:.1} → вторая {:.1} тиков; w(S→M) {w_start:.4}→{w_end:.4}; weights=[{lo:.3}; {hi:.3}]",
        mean(&first_half),
        mean(&second_half),
    );

    // ─── Фаза 3: рефлекс-протокол на HAAP (обучение измеримо средним) ──
    println!("\n=== РЕФЛЕКС HAAP: дофаминовая пластичность S→M ===");
    for &(he, ds) in &[(0usize, true), (1, true), (0, false)] {
        haap_reflex(scale, cap, he, ds);
    }
}

/// Фаза 4: проба живости когнитивной схемы на HAAP.
///
/// (а) Тишина: спонтанная активность по регионам без входа — пейсмекер WS
///     обязан гореть сам (у LIF-мозга тут мёртвый ноль).
/// (б) Ignition: кий в S → вспышка WS.
/// (в) Плато памяти: после снятия кия WM держит активность без входа.
/// (г) Стресс: мощный стимул → проверка затухания (не судорога ли).
fn cognitive_aliveness(scale: f32, cap: f32) {
    use dioma::brain::Brain;
    use dioma::delayed::{cognitive_gains, grow_cognitive, SEED};
    use dioma::physiology;

    let grown = grow_cognitive(SEED);
    let cols = physiology::brain_columns(
        &grown.region_of,
        physiology::cognitive_archetype,
        &Default::default(),
    );
    // HAAP-версия гейнов: та же топология, но рекуррентная реверберация
    // внутри workspace ослаблена — сеть иначе синхронизирует пейсмекерный
    // шум в непрерывный фон и контраст ignition пропадает.
    let mut gains = cognitive_gains();
    for (pair, g) in [
        ((5usize, 5usize), 0.04f32),
        ((6, 6), 0.04),
        ((7, 7), 0.06),
        ((8, 8), 0.06),
        ((7, 8), 0.03),
        ((8, 7), 0.03),
        ((7, 5), 0.02),
        ((8, 5), 0.02),
        ((7, 6), 0.02),
        ((8, 6), 0.02),
    ] {
        gains[pair.0][pair.1] = g;
    }
    let mut brain = {
        let p = PlasticityParams {
            haap_syn_scale: scale,
            haap_syn_cap: cap,
            homeo_every_ticks: 1, // стабильность прежде всего
            ..Default::default()
        };
        // Рекуррентности WS→WS/WM→WM — это и есть память, не замораживаем;
        // но пластичность в пробе живости не нужна: дофамин не подаём.
        Brain::new_haap(&grown, gains, p, &cols)
    };
    let n = brain.n;
    let tonic = vec![0.12f32; n];
    let regions = [
        ("S", 0usize),
        ("I", 1),
        ("M", 2),
        ("WS", 3),
        ("WM", 4),
    ];
    let sum_regions = |brain: &Brain, spikes: &[i32], ids: &[usize]| -> i32 {
        ids.iter().map(|&r| brain.region_spikes(spikes, r)).sum()
    };
    // S=0,1 | I=2 | M=3,4 | WS=5,6 | WM=7,8
    let bucket = |brain: &Brain, spikes: &[i32], name: &str| -> i32 {
        match name {
            "S" => sum_regions(brain, spikes, &[0, 1]),
            "I" => sum_regions(brain, spikes, &[2]),
            "M" => sum_regions(brain, spikes, &[3, 4]),
            "WS" => sum_regions(brain, spikes, &[5, 6]),
            _ => sum_regions(brain, spikes, &[7, 8]),
        }
    };

    println!("\n=== ЖИВОСТЬ: когнитивная схема HAAP (n={n}) ===");

    // (а) Тишина: 100 тиков только тоник.
    let mut base = [0i64; 5];
    for _ in 0..100 {
        let spikes = run_tick(&mut brain, &tonic);
        for (k, (_, _)) in regions.iter().enumerate() {
            base[k] += bucket(&brain, &spikes, regions[k].0) as i64;
        }
    }
    println!(
        "тишина 100т (спайков/тик): {}",
        fmt_regions(&regions, &base, 100.0)
    );

    // (б) Ignition: кий — левая половина сенсоров горит 30 тиков.
    let mut cue = tonic.clone();
    for (i, cell) in cue.iter_mut().enumerate().take(n / 9 * 2 / 2) {
        if grown.region_of[i] == 0 {
            *cell += 2.5;
        }
    }
    let mut cue_ws = 0i64;
    for _ in 0..30 {
        let spikes = run_tick(&mut brain, &cue);
        cue_ws += bucket(&brain, &spikes, "WS") as i64;
    }
    println!("ignition: WS при кие = {:.1} сп/тик", cue_ws as f64 / 30.0);

    // (в) Плато памяти: 60 тиков тишины после кия — WM держит?
    let mut wm_plateau = [0i64; 6]; // по декадам тиков
    let mut ws_after = [0i64; 6];
    for t in 0..60 {
        let spikes = run_tick(&mut brain, &tonic);
        wm_plateau[t / 10] += bucket(&brain, &spikes, "WM") as i64;
        ws_after[t / 10] += bucket(&brain, &spikes, "WS") as i64;
    }
    println!(
        "плато WM по декадам (сп/тик): {:?}",
        wm_plateau.map(|x| x as f64 / 10.0)
    );
    println!(
        "WS после кия по декадам:      {:?}",
        ws_after.map(|x| x as f64 / 10.0)
    );

    // (г) Стресс: все сенсоры ×6 на 10 тиков → затухание за 50.
    let mut stress = tonic.clone();
    for (i, cell) in stress.iter_mut().enumerate() {
        if grown.region_of[i] < 2 {
            *cell += 6.0;
        }
    }
    let mut worst = 0i64;
    for _ in 0..10 {
        let spikes = run_tick(&mut brain, &stress);
        let total: i64 = (0..5).map(|k| bucket(&brain, &spikes, regions[k].0) as i64).sum();
        worst = worst.max(total);
    }
    let mut decay = [0i64; 5];
    for t in 0..50 {
        let spikes = run_tick(&mut brain, &tonic);
        let total: i64 = (0..5).map(|k| bucket(&brain, &spikes, regions[k].0) as i64).sum();
        decay[t / 10] += total;
        worst = worst.max(total);
    }
    let st = brain.haap_state_stats();
    println!(
        "стресс: worstTick={worst}, затухание по декадам: {:?}",
        decay.map(|x| x as f64 / 10.0)
    );
    println!("состояние: v/u/h = {st:?}");
}

/// Один мировой тик (steps_per_tick подшагов) — возвращает спайки
/// последнего подшага (bucket-статистика берётся по нему).
fn run_tick(brain: &mut dioma::brain::Brain, ext: &[f32]) -> Vec<i32> {
    let mut last = vec![0i32; brain.n];
    for _ in 0..brain.steps_per_tick() {
        last = brain.step(ext);
    }
    last
}

fn fmt_regions(regions: &[(&str, usize)], sums: &[i64; 5], norm: f64) -> String {
    regions
        .iter()
        .zip(sums.iter())
        .map(|((name, _), s)| format!("{name}={:.1}", *s as f64 / norm))
        .collect::<Vec<_>>()
        .join(" ")
}

/// poc-reflex протокол на HAAP-мозге: A базовый → B покой → C 50 награждённых
/// проб → D финал. Латентность в мс (dt=0.05).
fn haap_reflex(scale: f32, cap: f32, homeo_every: usize, dopa_scaled: bool) {
    use dioma::brain::{gains_from, Brain};
    use dioma::grow::{grow, reflex_layout, Steering};
    use dioma::physiology;

    // nt уже в Grown от layout (grow ставит знак по inhibitory-флагу).
    let grown = grow(&reflex_layout(), &Steering::default(), 0x00C0_FFEE);
    let gains = gains_from(3, &[(0, 2, 0.45), (0, 1, 0.25), (0, 0, 0.12), (1, 2, 0.6)]);
    let cols = physiology::brain_columns(&grown.region_of, physiology::reflex_archetype, &Default::default());
    let mut brain = {
        let p = PlasticityParams {
            haap_syn_scale: scale,
            haap_syn_cap: cap,
            homeo_every_ticks: homeo_every,
            haap_dopamine_scaled: dopa_scaled,
            // S→S рекуррентность — врождённая (урок арены).
            frozen_pairs: vec![(0, 0)],
            ..Default::default()
        };
        Brain::new_haap(&grown, gains, p, &cols)
    };
    let dt = brain.dt_ms();
    const STIM: usize = 400; // 20 мс стимул
    const RESP: usize = 6000; // окно отклика 300 мс
    const REST: usize = 2000; // 100 мс покоя

    let measure = |brain: &mut Brain| -> (f32, f32, i64) {
        let n = brain.n;
        let ext0 = vec![0.0f32; n];
        let mut worst = 0i64;
        for _ in 0..REST {
            brain.step(&ext0);
        }
        let mut ext = ext0.clone();
        let mut latency: Option<usize> = None;
        for t in 0..STIM + RESP {
            let stim = if t < STIM { 3.0 } else { 0.0 };
            for cell in ext.iter_mut().take(40) {
                *cell = stim;
            }
            let spikes = brain.step(&ext);
            let m: i64 = spikes[60..n].iter().map(|&s| s as i64).sum::<i64>();
            worst = worst.max(m);
            if m > 0 && latency.is_none() {
                latency = Some(t);
            }
        }
        let (w, _) = brain.mean_path_weight(0, 2);
        (latency.map(|t| t as f32 * dt).unwrap_or(-1.0), w, worst)
    };

    let tag = format!("homeo={homeo_every} dopa×s={dopa_scaled}");
    let (lat_a, w_a, _) = measure(&mut brain);
    // B: 200 мс спонтанной активности.
    let zero = vec![0.0f32; brain.n];
    for _ in 0..4000 {
        brain.step(&zero);
    }
    // C: 50 награждённых проб.
    for _ in 0..50 {
        let _ = measure(&mut brain);
        for _ in 0..1000 {
            brain.step(&zero);
        }
        brain.reward_now(1.0);
    }
    let (lat_d, w_d, worst) = measure(&mut brain);
    let dw = (w_d - w_a) / w_a.max(1e-9) * 100.0;
    println!(
        "{tag:>28}: латентность {lat_a:6.1}→{lat_d:6.1} мс, w(S→M) {w_a:.4}→{w_d:.4} ({dw:+5.1}%), worstM={worst}, рефлекс {}",
        if lat_d >= 0.0 { "жив" } else { "ПОТЕРЯН" },
    );
}
