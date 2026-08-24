//! -------------------------------------------------------------------------
//! Filename:    tests/smoke.rs
//! Class:       DebugSuite
//! Purpose:     smoke tests for DebugSuite protocols
//! Description: wiring directionality, plasticity, workspace, blood v2 stoichiometry/homeostasis.
//! 
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use dioma::world::arena::{arena_brain, arena_gains, grow_agent, run_episode, Agent, Genome, REG_ML, REG_MR, REG_SL, REG_SR};
use dioma::engine::neurons::{gains_from, Brain, PlasticityParams};
use dioma::world::delayed::{
    ablated_edges, grow_cognitive, run_protocol_n, REG_WSL, SEED as GNW_SEED,
};
use dioma::world::free_life::run_life;
use dioma::engine::grow::{grow, reflex_layout, Steering};


// ─── Рост ───────────────────────────────────────────────────────────────────
#[test]
fn arena_wiring_is_directional() {
    let genome = Genome::wildtype(0x0000_AE11);
    let grown = grow_agent(&genome);
    let (own, cross) = grown.edges.iter().fold((0, 0), |(o, c), &(pre, post, _)| {
        let rp = grown.region_of[pre];
        let rq = grown.region_of[post];
        let to_ml = rq == REG_ML;
        let to_mr = rq == REG_MR;
        (
            o + ((rp == REG_SL && to_ml) || (rp == REG_SR && to_mr)) as usize,
            c + ((rp == REG_SL && to_mr) || (rp == REG_SR && to_ml)) as usize,
        )
    });
    assert!(own >= 5, "мало направленных S→M: {own}");
    assert_eq!(cross, 0, "перекрёстные S→M: {cross}");
}

#[test]
fn cognitive_wiring_is_directional() {
    let grown = grow_cognitive(GNW_SEED);
    let mut own = 0;
    let mut cross = 0;
    for &(pre, post, _) in &grown.edges {
        let rp = grown.region_of[pre];
        let rq = grown.region_of[post];
        if rp == 0 && rq == REG_WSL {
            own += 1;
        }
        if (rp == 0 && rq == REG_WSL + 1) || (rp == 1 && rq == REG_WSL) {
            cross += 1;
        }
    }
    assert!(own >= 10, "мало отчётов S→WS: {own}");
    assert_eq!(cross, 0, "перекрёстные S→WS: {cross}");
}

// ─── Рефлекс + дофамин ──────────────────────────────────────────────────────
#[test]
fn reflex_fires_and_dopamine_strengthens_path() {
    let grown = grow(&reflex_layout(), &Steering::default(), 0x00C0_FFEE);
    let mut nt = vec![1.0f32; 40];
    nt.extend(std::iter::repeat_n(-1.0, 20));
    nt.extend(std::iter::repeat_n(1.0, 40));
    let grown = dioma::Grown { nt, ..grown };
    let gains = gains_from(3, &[(0, 2, 0.45), (0, 1, 0.25), (0, 0, 0.12), (1, 2, 0.6)]);
    let mut brain = Brain::new(&grown, gains, PlasticityParams::default());

    // Базовый рефлекс: стимул в S → отклик M.
    let (lat0, w0) = measure(&mut brain);
    assert!(lat0 > 0.0, "рефлекс не отвечает: {lat0}");

    // Обучение с дофамином.
    for _ in 0..15 {
        let _ = measure(&mut brain);
        let ext = vec![0.0f32; brain.n];
        for _ in 0..100 {
            brain.step(&ext);
        }
        brain.reward_now(1.0);
    }
    let (_, w1) = measure(&mut brain);
    assert!(w1 > w0 * 1.2, "дофамин не усилил S→M: {w0:.4} → {w1:.4}");

    // Рефлекс жив.
    let (lat1, _) = measure(&mut brain);
    assert!(lat1 > 0.0, "рефлекс потерян после обучения");
}

fn measure(brain: &mut Brain) -> (f32, f32) {
    let n = brain.n;
    let (s0, s1) = brain.region_bounds(0);
    let (m0, m1) = brain.region_bounds(2); // только моторы: I — тормозные
    let ext0 = vec![0.0f32; n];
    for _ in 0..100 {
        brain.step(&ext0);
    }
    let mut ext = ext0.clone();
    let mut latency: Option<usize> = None;
    for t in 0..340 {
        let stim_on = t < 40;
        let stim = if stim_on { 3.0 } else { 0.0 };
        for cell in ext[s0..s1].iter_mut() {
            *cell = stim;
        }
        let spikes = brain.step(&ext);
        let m: i32 = (m0..m1).map(|i| spikes[i]).sum();
        if m > 0 && latency.is_none() {
            latency = Some(t);
        }
    }
    let (w, _) = brain.mean_path_weight(0, 2);
    (latency.map(|t| t as f32 * 0.5).unwrap_or(-1.0), w)
}

// ─── Арена ──────────────────────────────────────────────────────────────────
#[test]
fn arena_agent_eats_and_dopamine_moves_weights() {
    let genome = Genome::wildtype(0x0000_AE11);
    let grown = grow_agent(&genome);

    let mut brain = arena_brain(&grown, &genome);
    let (w0, _) = brain.mean_path_weight(REG_SL, REG_ML);
    let mut agent = Agent::new(genome.seed);
    let mut eaten = 0;
    for _ in 0..10 {
        let t = run_episode(&mut brain, &mut agent, &genome, true);
        if t < dioma::world::arena::EPISODE_TIMEOUT_TICKS {
            eaten += 1;
        }
    }
    assert!(eaten >= 3, "агент почти не ест: {eaten}/10");
    let (w1, _) = brain.mean_path_weight(REG_SL, REG_ML);
    assert!(w1 != w0, "дофамин не подействовал: {w0} → {w1}");

    // Контроль без дофамина: веса бит-в-бит неподвижны.
    let mut brain_c = arena_brain(&grown, &genome);
    let mut agent_c = Agent::new(genome.seed);
    for _ in 0..10 {
        run_episode(&mut brain_c, &mut agent_c, &genome, false);
    }
    let (w2, _) = brain_c.mean_path_weight(REG_SL, REG_ML);
    assert_eq!(w2, w0, "без дофамина веса изменились: {w0} → {w2}");
}

// ─── GNW ────────────────────────────────────────────────────────────────────
#[test]
fn workspace_beats_ablation_in_delayed_response() {
    let grown = grow_cognitive(GNW_SEED);
    let full = run_protocol_n(&grown.edges, 10);
    let ablated = run_protocol_n(&ablated_edges(&grown), 10);
    let mean = |s: &[dioma::world::delayed::EpisodeStats]| {
        s.iter().map(|x| x.ticks as f64).sum::<f64>() / s.len() as f64
    };
    let (m_full, m_abl) = (mean(&full), mean(&ablated));
    assert!(m_full < m_abl * 0.9, "workspace не помогает: {m_full} vs {m_abl}");
    let ws: f32 = full.iter().map(|s| s.ws_rate_cue).sum::<f32>() / full.len() as f32;
    assert!(ws > 5.0, "workspace не зажигается: {ws}");
}

// ─── Кровь v2 (порт биохимии Creatures) ─────────────────────────────────────
#[test]
fn stoichiometry_honors_limiting_reagent() {
    use dioma::body::metabolism::{chem, locus, Blood};
    // Прямая проверка лимитирующего реагента: 2A + 1B → 1C.
    let mut b = Blood::creature(false);
    b.reactions = vec![dioma::body::metabolism::Reaction {
        prop_r1: 2.0, r1: chem::GLUCOSE, prop_r2: 1.0, r2: chem::FOOD,
        rate: 1.0, // всё доступное реагирует за тик
        prop_p1: 1.0, p1: chem::HUNGER, prop_p2: 0.0, p2: usize::MAX,
    }];
    b.emitters.clear();
    b.receptors.clear();
    b.pins.clear();
    b.conc[chem::GLUCOSE] = 20.0; // 20/2 = 10 моль «2A»
    b.conc[chem::FOOD] = 15.0; // 15 моль B — в избытке
    b.update();
    // Лимитирует A: среагировало 10 моль → глюкоза −20 (в ноль), еда −10,
    // продукт +10 (минус распад за тик).
    assert!((b.conc[chem::HUNGER] - 10.0 * 0.999).abs() < 0.1, "продукт: {}", b.conc[chem::HUNGER]);
    assert!(b.conc[chem::GLUCOSE] <= 0.0);
    assert!((b.conc[chem::FOOD] - 5.0).abs() < 0.01, "еда: {}", b.conc[chem::FOOD]);
    let _ = locus::N;
}

#[test]
fn hunger_rises_on_deficit_and_falls_when_fed() {
    use dioma::body::metabolism::{chem, locus, Blood};
    let mut b = Blood::creature(false);
    b.emitters.retain(|e| e.chem == chem::HUNGER); // только драйв-эмиттер
    // Голодание: дефицит максимален.
    b.conc[chem::GLUCOSE] = 0.0;
    for _ in 0..300 {
        b.loci[locus::GLUCOSE_DEFICIT] = 150.0;
        b.update();
    }
    assert!(b.conc[chem::HUNGER] > 40.0, "голод не растёт: {}", b.conc[chem::HUNGER]);
    // Накормлен: дефицита нет, голод гаснет.
    b.conc[chem::GLUCOSE] = 200.0;
    b.conc[chem::HUNGER] = 80.0;
    for _ in 0..600 {
        b.loci[locus::GLUCOSE_DEFICIT] = 0.0;
        b.update();
    }
    assert!(b.conc[chem::HUNGER] < 40.0, "голод не гаснет: {}", b.conc[chem::HUNGER]);
}

#[test]
fn brain_activity_costs_energy_and_fatigues() {
    use dioma::body::metabolism::{chem, locus, Blood};
    let mut b = Blood::creature(false);
    b.emitters.retain(|e| e.chem == chem::FATIGUE || e.locus == locus::EVENT_ATE);
    let g0 = b.conc[chem::GLUCOSE];
    // Высокая моторная активность 200 тиков.
    for _ in 0..200 {
        b.loci[locus::RATE_MOTOR] = 60.0;
        b.update();
    }
    assert!(b.conc[chem::GLUCOSE] < g0, "активность не стоит глюкозы");
    assert!(b.conc[chem::FATIGUE] > 5.0, "утомление не копится: {}", b.conc[chem::FATIGUE]);
    // Скорость падает от утомления (рецептор FATIGUE → SPEED reduce).
    let mut b2 = Blood::creature(false);
    b2.conc[chem::FATIGUE] = 200.0;
    b2.update();
    assert!(b2.speed() < 2.0, "утомление не замедляет: {}", b2.speed());
}

#[test]
fn chemistry_agent_eats_less_than_robot_but_survives() {
    let chem = run_life(false, 0x0000_AE11);
    let robot = run_life(true, 0x0000_AE11);
    assert!(chem.eaten < robot.eaten, "химия {} vs робот {}", chem.eaten, robot.eaten);
    assert!(chem.min_glucose > 0.0, "голодает: {}", chem.min_glucose);
    assert!(chem.meals >= 3, "мало циклов питания: {}", chem.meals);
    assert!(chem.peak_fatigue > 10.0, "утомление не проявилось");
    assert!(!chem.dead, "агент умер от голода");
    // Пошаговый API: пара тиков вручную.
    let mut sim = dioma::world::free_life::Sim::new(false, 0x0000_AE11);
    let obs = sim.step_tick();
    assert_eq!(obs.chemicals.len(), dioma::body::metabolism::chem::N);
    assert_eq!(obs.region_spikes.len(), dioma::world::arena::N_REGIONS);
    let _ = &mut sim;
}

// ─── Критические периоды / трейс / геномные гейны ──────────────────────────
#[test]
fn critical_period_thaws_frozen_pair() {
    // Схема рефлекса; пара (S, I) заморожена до world-тика 3.
    // nt уже в Grown от layout (grow ставит знак по inhibitory-флагу).
    let grown = grow(&reflex_layout(), &Steering::default(), 0x00C0_FFEE);
    let gains = gains_from(3, &[(0, 2, 0.45), (0, 1, 0.45), (0, 0, 0.12), (1, 2, 0.6)]);
    let params = PlasticityParams {
        critical_pairs: vec![(0, 1, 3)], // S→I оттаивает на тике 3
        ..Default::default()
    };
    let mut brain = Brain::new(&grown, gains, params);

    let (w0, cnt) = brain.mean_path_weight(0, 1);
    assert!(cnt > 0, "нет синапсов S→I");
    let drive = {
        let mut ext = vec![0.0f32; brain.n];
        for cell in ext.iter_mut().take(40) {
            *cell = 3.0;
        }
        ext
    };
    // До окна (тик 1): дофамин не меняет замороженную пару.
    for _ in 0..100 {
        brain.step(&drive);
    }
    brain.reward_now(1.0);
    let (w1, _) = brain.mean_path_weight(0, 1);
    assert_eq!(w1, w0, "замороженная пара изменилась до окна: {w0} → {w1}");

    // После окна (тик 5): пластичность работает.
    for _ in 0..4 * 100 {
        brain.step(&drive);
    }
    brain.reward_now(1.0);
    let (w2, _) = brain.mean_path_weight(0, 1);
    assert!(w2 != w1, "пара не оттаяла после окна: {w1} → {w2}");
}

#[test]
fn blood_trace_records_events_and_state() {
    use dioma::body::metabolism::{chem, Blood};
    let mut b = Blood::creature(false);
    b.enable_trace(30); // ≥ числа тиков: событие первого тика не вытеснится
    b.ingest_food(50.0);
    for _ in 0..25 {
        b.loci[dioma::body::metabolism::locus::GLUCOSE_DEFICIT] = 100.0;
        b.update();
    }
    let trace = b.trace().expect("trace включен");
    assert_eq!(trace.len(), 25, "потеряны тики: {}", trace.len());
    assert!(trace[0].ate, "событие еды не записано");
    // Кольцо: маленький буфер вытесняет старое.
    let mut b2 = Blood::creature(false);
    b2.enable_trace(5);
    for _ in 0..10 {
        b2.update();
    }
    assert_eq!(b2.trace().unwrap().len(), 5, "буфер не ограничен");
    assert!(trace.iter().all(|t| t.chemicals.len() == chem::N));
    assert!(trace.iter().all(|t| t.loci.len() == dioma::body::metabolism::locus::N));
}

#[test]
fn genome_encodes_gain_matrix() {
    let wild = Genome::wildtype(7);
    let gains = arena_gains(&wild);
    assert_eq!(gains[2][3], wild.w_i, "I→M гейн не из генома");
    assert_eq!(gains[0][2], wild.w_si, "S→I гейн не из генома");
    assert_eq!(gains[0][0], wild.w_ss, "S→S гейн не из генома");
    // Мутация гена меняет матрицу.
    let mut mutant = wild.clone();
    mutant.w_i = 0.0;
    let gm = dioma::world::arena::arena_gains(&mutant);
    assert_eq!(gm[2][3], 0.0);
}

// ─── HAAP-мозг: стабильность, обучение, живость (brain-lab находки) ────────

/// HAAP dt должен реально применяться: template задаёт P::dt, конфиг
/// HaapSynaptic согласован. Регрессия на «перезаливку dt=0.5 из конфига»,
/// которая делала мозг би-стабильным (взрыв/тишина).
#[test]
fn haap_dt_column_stays_at_haap_step() {
    use dioma::engine::neurons::BrainModel;
    let mut genome = Genome::wildtype(1);
    genome.brain_model = BrainModel::Haap;
    let grown = grow_agent(&genome);
    let brain = arena_brain(&grown, &genome);
    let zero = vec![0.0f32; brain.n];
    // Первый step перезаливает dt из конфига при расхождении — проверяем после.
    let mut b = arena_brain(&grown, &genome);
    b.step(&zero);
    let (lo, hi) = b.haap_dt_column().expect("HAAP колонка dt");
    assert!((lo - dioma::engine::neurons::DT_MS_HAAP).abs() < 1e-6, "dt={lo}");
    assert!((hi - lo).abs() < 1e-6, "dt неоднороден: [{lo}; {hi}]");
}

/// Арена на HAAP: агент ест и не судорожит (клип хвоста + гомеостаз +
/// врождённые S→S/S→I + жевание после еды). Судорога = >3000 спайков за тик.
#[test]
fn haap_arena_agent_eats_without_seizure() {
    use dioma::engine::neurons::BrainModel;
    let mut genome = Genome::wildtype(0x0000_AE11);
    genome.brain_model = BrainModel::Haap;
    let grown = grow_agent(&genome);
    let mut brain = arena_brain(&grown, &genome);
    let mut agent = Agent::new(genome.seed);

    let mut eaten = 0;
    for _ in 0..8 {
        let (ticks, worst_tick) =
            dioma::world::arena::run_episode_debug(&mut brain, &mut agent, &genome, true);
        assert!(
            worst_tick < 3000,
            "судорога HAAP-мозга: {worst_tick} спайков за тик"
        );
        if ticks < dioma::world::arena::EPISODE_TIMEOUT_TICKS {
            eaten += 1;
        }
    }
    assert!(eaten >= 5, "агент почти не ест на HAAP: {eaten}/8");
    let (lo, hi) = brain.values_minmax();
    assert!(hi <= 3.0 * 0.15 + 1e-3, "вес выше потолка w_max×scale: {hi}");
    assert!(lo >= 0.0);
}

/// Рефлекс на HAAP учится: дофамин усиливает S→M без гомеостаза каждый тик.
#[test]
fn haap_reflex_dopamine_strengthens_path() {    use dioma::engine::neurons::{gains_from, Brain, PlasticityParams};
    use dioma::engine::grow::{grow, reflex_layout, Steering};
    use dioma::body::physiology;

    // nt уже в Grown от layout (grow ставит знак по inhibitory-флагу).
    let grown = grow(&reflex_layout(), &Steering::default(), 0x00C0_FFEE);
    let gains = gains_from(3, &[(0, 2, 0.45), (0, 1, 0.25), (0, 0, 0.12), (1, 2, 0.6)]);
    let cols =
        physiology::brain_columns(&grown.region_of, physiology::reflex_archetype, &Default::default());
    let params = PlasticityParams {
        homeo_every_ticks: 0, // измеряем среднее пути — гомеостаз его пинит
        frozen_pairs: vec![(0, 0)],
        ..Default::default()
    };
    let mut brain = Brain::new_haap(&grown, gains, params, &cols);

    let n = brain.n;
    let zero = vec![0.0f32; n];
    let trial = |brain: &mut Brain| -> bool {
        let mut ext = zero.clone();
        let mut fired = false;
        for t in 0..400 {
            let stim = if t < 200 { 3.0 } else { 0.0 };
            for cell in ext.iter_mut().take(40) {
                *cell = stim;
            }
            let spikes = brain.step(&ext);
            if spikes[60..n].iter().any(|&s| s != 0) {
                fired = true;
            }
        }
        fired
    };
    assert!(trial(&mut brain), "базовый рефлекс не отвечает");
    let (w0, cnt) = brain.mean_path_weight(0, 2);
    assert!(cnt > 0 && w0 > 0.0, "нет пути S→M");

    for _ in 0..20 {
        assert!(trial(&mut brain), "рефлекс потерян в ходе обучения");
        for _ in 0..1000 {
            brain.step(&zero);
        }
        brain.reward_now(1.0);
    }
    let (w1, _) = brain.mean_path_weight(0, 2);
    assert!(
        w1 > w0 * 1.15,
        "дофамин не усилил S→M на HAAP: {w0:.4} → {w1:.4}"
    );
}

// ─── Органный мозг: HH спина + HAAP средний + LIF кора ─────────────────────

fn spine_conn_test() -> pillars_connectome::ConnectomeData {
    use pillars_connectome::ConnectomeData;
    const NS: usize = 10;
    const NI: usize = 10;
    const NM: usize = 10;
    let n = NS + NI + NM;
    let mut rp = vec![0i32; n + 1];
    let mut ci = Vec::new();
    let mut vals = Vec::new();
    for (s, rp_s) in rp.iter_mut().enumerate().take(NS) {
        let m = NS + NI + s % NM;
        *rp_s = ci.len() as i32;
        ci.push(m as i32);
        vals.push(0.40);
    }
    for n_ in (0..n).rev() {
        rp[n_ + 1] = ci.len() as i32;
    }
    let nt = vec![1.0f32; n];
    let mut conn = ConnectomeData::new(rp, ci, vals, nt, vec![]);
    conn.build_receptor_types();
    conn
}

/// Органный PoC: HH-рефлекс быстрый, кора учится на дофамине, контроль нет.
#[test]
fn organs_reflex_fast_and_cortex_learns() {
    use dioma::engine::neurons::{Brain, PlasticityParams};
    use dioma::body::organs::{Axon, OrganBrain, OrganId};
    use pillars_connectome::{BrainConfig, BrainRunner};

    // A: латентность спинального рефлекса.
    let mut spine = BrainRunner::new(spine_conn_test(), BrainConfig::hh_synaptic(0.05)).unwrap();
    let zero = vec![0.0f32; 30];
    for _ in 0..200 {
        spine.run(1, &zero);
    }
    let pain = {
        let mut e = zero.clone();
        for cell in e.iter_mut().take(10) {
            *cell = 15.0;
        }
        e
    };
    let mut latency: Option<usize> = None;
    for step in 0..400 {
        let buf = spine.run(1, &pain);
        if buf[20..].iter().any(|&s| s != 0) {
            latency = Some(step);
            break;
        }
    }
    let lat = latency.expect("HH рефлекс не отвечает");
    assert!(lat * 5 < 5000, "HH рефлекс медленнее 5 мс: {}", lat as f32 * 0.05);

    // C: кора учится.
    const NP: usize = 12;
    let grown = dioma::Grown {
        edges: {
            let mut e = Vec::new();
            for j in 0..NP {
                for k in [0usize, 4, 8] {
                    e.push(((j + k) % NP, 2 * NP + j, 2.0));
                    e.push((NP + (j + k + 3) % NP, 2 * NP + j, 2.0));
                }
            }
            e
        },
        region_of: {
            let mut r = vec![0usize; NP];
            r.extend(vec![1; NP]);
            r.extend(vec![2; NP]);
            r
        },
        nt: vec![1.0; 3 * NP],
        growth_s: 0.0,
        rule_genes: Vec::new(),
    };
    let gains = {
        let mut g = vec![vec![0.0f32; 3]; 3];
        g[0][2] = 1.0;
        g[1][2] = 1.0;
        g
    };
    let mut organs = OrganBrain::new(spine_conn_test(), grown.clone(), gains.clone(), PlasticityParams::default(), 24);
    organs.add_axon(Axon {
        from: OrganId::Midbrain,
        pre: (0..24).collect(),
        to: OrganId::Cortex,
        post: (0..3 * NP).collect(),
        weight_per_spike: 0.006,
    });
    let tonic = vec![0.12f32; 3 * NP];
    let cue_a = {
        let mut e = tonic.clone();
        for cell in e.iter_mut().take(NP) {
            *cell += 3.0;
        }
        e
    };
    let cue_b = {
        let mut e = tonic.clone();
        for cell in e.iter_mut().skip(NP).take(NP) {
            *cell += 3.0;
        }
        e
    };
    let (w_a0, _) = organs.cortex.mean_path_weight(0, 2);
    let (w_b0, _) = organs.cortex.mean_path_weight(1, 2);
    for _ in 0..12 {
        for _ in 0..20 {
            organs.tick(&dioma::body::organs::WorldInputs { cortex: cue_a.clone(), ..Default::default() });
        }
        organs.cortex.reward_now(1.0);
        organs.cortex.clear_eligibility();
        for _ in 0..20 {
            organs.tick(&dioma::body::organs::WorldInputs { cortex: cue_b.clone(), ..Default::default() });
        }
        organs.cortex.clear_eligibility();
    }
    let (w_a1, _) = organs.cortex.mean_path_weight(0, 2);
    let (w_b1, _) = organs.cortex.mean_path_weight(1, 2);
    assert!(w_a1 > w_a0 * 1.15, "награждённый путь не вырос: {w_a0}→{w_a1}");
    assert!(
        w_a1 > w_b1,
        "кора не различает награду: A={w_a1} vs B={w_b1}"
    );
    assert!((w_b1 - w_b0).abs() < w_a1 - w_a0, "ненаграждённый растёт быстрее");

    // Контроль без дофамина: веса на месте.
    let mut ctrl = Brain::new(&grown, gains, PlasticityParams::default());
    let (c0, _) = ctrl.mean_path_weight(0, 2);
    for _ in 0..12 {
        for _ in 0..20 {
            ctrl.step(&cue_a);
        }
        for _ in 0..20 {
            ctrl.step(&cue_b);
        }
    }
    let (c1, _) = ctrl.mean_path_weight(0, 2);
    assert_eq!(c1, c0, "без дофамина веса изменились");
}
