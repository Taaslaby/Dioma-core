//! -------------------------------------------------------------------------
//! Filename:    src/physiology.rs
//! Class:       GenSuite
//! Purpose:     gene → HAAP columns (PROMETHEUS port)
//! Description: per-neuron physiology as archetype template × expression
//!              genes. Knockout = expression 0 (e.g. no HCN → no rebound,
//!              memory plateaus die). VNC-stability inherited from
//!              taas-checkmate canonical defaults.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------

use pillars::zoo::HAAPModelParamIdx as P;

/// Число колонок HAAP.
pub const N_COLS: usize = 17;

/// Гены экспрессии каналов (0..1+, wildtype = 1). Наследуются и мутируются.
#[derive(Clone, Debug)]
pub struct PhysioGenes {
    /// HCN h-ток: rebound, медленные плато (память).
    pub hcn: f32,
    /// KCNQ M-ток: адаптация, гашение burst'ов.
    pub kcnq: f32,
    /// para NaV: возбудимость (сдвиг порога).
    pub para: f32,
    /// leak K2P: входное сопротивление.
    pub leak: f32,
}

impl Default for PhysioGenes {
    fn default() -> Self {
        Self {
            hcn: 1.0,
            kcnq: 1.0,
            para: 1.0,
            leak: 1.0,
        }
    }
}

/// Архетип нейрона: роль региона задаёт шаблон физиологии.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Archetype {
    /// Сенсор: чуткий реле, слабая адаптация.
    Sensory,
    /// Мотор: VNC-профиль — адаптация против runaway (изначальный рецепт).
    Motor,
    /// Тормозной интернейрон: быстрый, звонкий.
    Interneuron,
    /// Память: сильный h-ток + медленный — плато на секунды.
    Memory,
    /// Пейсмекер: низкий порог + адаптация = ритмические burst'ы.
    Pacemaker,
}

/// Регион арены (5) → архетип.
pub fn arena_archetype(region: usize) -> Archetype {
    match region {
        0 | 1 => Archetype::Sensory,
        2 => Archetype::Interneuron,
        _ => Archetype::Motor,
    }
}

/// Регион когнитивной схемы (9) → архетип.
pub fn cognitive_archetype(region: usize) -> Archetype {
    match region {
        0 | 1 => Archetype::Sensory,
        2 => Archetype::Interneuron,
        3 | 4 => Archetype::Motor,
        5 | 6 => Archetype::Pacemaker, // workspace: burst-гейтинг ignition
        _ => Archetype::Memory,        // WM: плато
    }
}

/// Регион рефлекса (3) → архетип.
pub fn reflex_archetype(region: usize) -> Archetype {
    match region {
        0 => Archetype::Sensory,
        1 => Archetype::Interneuron,
        _ => Archetype::Motor,
    }
}

/// Шаблон архетипа → базовые 17 колонок (VNC-производные; g_l снижен,
/// чтобы мировые токи LIF-масштаба двигали мембрану).
fn template(arch: Archetype) -> [f32; N_COLS] {
    // База = рабочая точка tripod_cpg (проверенная в движке): e_l −60,
    // g_l 0.1, g_m 0.4, tau_u 100, порог −40, dt 0.05, τ_syn 12.
    // Входы порядка 1–3, синаптические веса порядка 5–20.
    // g_h база понижена 0.8→0.35: сильный h-ток на КАЖДОМ нейроне вместе с
    // рекуррентным торможением даёт post-inhibitory rebound-судороги
    // (v до −144 мВ → отдача → PING-петля). Сильный h остаётся у Memory.
    let mut c = [0.0f32; N_COLS];
    let set = |c: &mut [f32; N_COLS], idx: P, v: f32| c[idx as usize] = v;
    set(&mut c, P::dt, crate::engine::neurons::DT_MS_HAAP);
    set(&mut c, P::inv_cm, 1.0);
    set(&mut c, P::g_l, 0.10);
    set(&mut c, P::e_l, -60.0);
    set(&mut c, P::g_m, 0.4);
    set(&mut c, P::e_k, -80.0);
    set(&mut c, P::g_h, 0.35);
    set(&mut c, P::e_h, -30.0);
    set(&mut c, P::inv_tau_u, 1.0 / 100.0);
    set(&mut c, P::inv_tau_h, 1.0 / 120.0);
    set(&mut c, P::v_half_u, -35.0);
    set(&mut c, P::k_u, 5.0);
    set(&mut c, P::v_half_h, -75.0);
    set(&mut c, P::k_h, 6.0);
    set(&mut c, P::v_thresh, -40.0);
    set(&mut c, P::v_reset, -55.0);
    set(&mut c, P::delta_u, 0.02);
    match arch {
        Archetype::Sensory => {
            // Реле: сильная адаптация (delta крупный) ограничивает верх
            // частоты на сильном сигнале; высокий reset — быстрый перезапуск.
            c[P::v_thresh as usize] = -42.0;
            c[P::delta_u as usize] = 0.03;
            c[P::inv_tau_u as usize] = 1.0 / 50.0; // быстрое восстановление
            c[P::v_reset as usize] = -47.0;
        }
        Archetype::Motor => {
            c[P::v_thresh as usize] = -42.0;
            c[P::delta_u as usize] = 0.03;
            c[P::inv_tau_u as usize] = 1.0 / 50.0;
            c[P::v_reset as usize] = -47.0;
        }
        Archetype::Interneuron => {
            c[P::inv_tau_u as usize] = 1.0 / 50.0;
        }
        Archetype::Memory => {
            c[P::g_h as usize] = 0.8; // сильный h-ток — плато на секундах
            c[P::g_m as usize] = 0.15; // burst не гаснет сразу
            c[P::inv_tau_h as usize] = 1.0 / 400.0; // медленный h — секунды
            c[P::v_half_h as usize] = -70.0;
            c[P::v_reset as usize] = -50.0;
            c[P::delta_u as usize] = 0.01;
        }
        Archetype::Pacemaker => {
            // Эндогенный осциллятор (подобран pm-tune, кандидат K):
            // h-ток с полуактивацией у покоя тянет к порогу; burst
            // подсыпает delta_u=0.09, очень медленный (700 мс) M-ток
            // держит длинную паузу. Фон редких «мыслей» (несколько
            // спайков/тик) — ниже порога записи в память; кий же
            // зажигает весь регион через S→WS и пересекает его.
            c[P::v_thresh as usize] = -47.0;
            c[P::g_h as usize] = 1.6;
            c[P::v_half_h as usize] = -56.0;
            c[P::k_h as usize] = 7.0;
            c[P::g_m as usize] = 0.30;
            c[P::delta_u as usize] = 0.09;
            c[P::inv_tau_u as usize] = 1.0 / 700.0;
        }
    }
    c
}

/// Архетип × гены экспрессии → колонки нейрона.
pub fn haap_columns(arch: Archetype, genes: &PhysioGenes) -> [f32; N_COLS] {
    let mut c = template(arch);
    // HCN: масштабирует h-ток (нокаут = нет rebound/плато).
    c[P::g_h as usize] *= genes.hcn.max(0.0);
    // KCNQ: масштабирует M-ток (нокаут = нет адаптации → runaway-риск).
    c[P::g_m as usize] *= genes.kcnq.max(0.0);
    // para: сдвиг порога (низкая экспрессия → труднее возбудить).
    c[P::v_thresh as usize] += (genes.para - 1.0) * PARA_THRESHOLD_SPAN;
    // leak: масштабирует проводимость утечки (больше = тише).
    c[P::g_l as usize] *= genes.leak.max(0.05);
    c
}

/// Колонки для всего мозга по регионам.
///
/// Пейсмекерные регионы разряжаются: эндогенные осцилляторы (~1/3 нейронов,
/// детерминированный хэш) задают спонтанные burst'ы, остальные — тихие реле.
/// Сплошной пейсмекер на весь регион рекуррентная связь синхронизирует в
/// постоянный стрёп: контраст ignition пропадает.
pub fn brain_columns(
    region_of: &[usize],
    archetype_of: impl Fn(usize) -> Archetype,
    genes: &PhysioGenes,
) -> Vec<[f32; N_COLS]> {
    const PACEMAKER_FRACTION: f64 = 0.34;
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    region_of
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let arch = archetype_of(r);
            let arch = if arch == Archetype::Pacemaker {
                let c = counts.entry(r).or_insert(0);
                *c += 1;
                let h =
                    ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33) as f64 / u32::MAX as f64;
                if h < PACEMAKER_FRACTION {
                    Archetype::Pacemaker
                } else {
                    Archetype::Interneuron
                }
            } else {
                arch
            };
            haap_columns(arch, genes)
        })
        .collect()
}

/// Равновесие покоя нейрона с данными колонками: решаем
/// `g_l(V−e_l) + g_m·u∞(V)(V−e_k) + g_h·h∞(V)(V−e_h) = 0`
/// бисекцией по [−120, −20] мВ, гейты в стационаре (как NEURON INITIAL).
/// Возвращает (v, u, h). Без внешнего тока.
pub fn resting_state(c: &[f32; N_COLS]) -> (f32, f32, f32) {
    let (g_l, e_l) = (c[P::g_l as usize], c[P::e_l as usize]);
    let (g_m, e_k) = (c[P::g_m as usize], c[P::e_k as usize]);
    let (g_h, e_h) = (c[P::g_h as usize], c[P::e_h as usize]);
    let (vh_u, k_u) = (c[P::v_half_u as usize], c[P::k_u as usize]);
    let (vh_h, k_h) = (c[P::v_half_h as usize], c[P::k_h as usize]);
    let u_inf = |v: f32| 1.0 / (1.0 + (-(v - vh_u) / k_u).exp());
    let h_inf = |v: f32| 1.0 / (1.0 + ((v - vh_h) / k_h).exp());
    let f = |v: f32| g_l * (v - e_l) + g_m * u_inf(v) * (v - e_k) + g_h * h_inf(v) * (v - e_h);
    let (mut lo, mut hi) = (-120.0f32, -20.0f32);
    debug_assert!(f(lo) < 0.0 && f(hi) > 0.0, "resting_state: нет перемены знака");
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let v = 0.5 * (lo + hi);
    (v, u_inf(v), h_inf(v))
}

/// Залить колонки HAAP в раннер: параметры + гейты в равновесие шаблона.
/// Единая точка инициализации (Brain, органный средний мозг).
pub fn apply_haap_columns(
    runner: &mut pillars_connectome::BrainRunner,
    cols: &[[f32; N_COLS]],
) {
    use pillars::zoo::HAAPModelParamIdx;
    let order = [
        HAAPModelParamIdx::dt,
        HAAPModelParamIdx::inv_cm,
        HAAPModelParamIdx::g_l,
        HAAPModelParamIdx::e_l,
        HAAPModelParamIdx::g_m,
        HAAPModelParamIdx::e_k,
        HAAPModelParamIdx::g_h,
        HAAPModelParamIdx::e_h,
        HAAPModelParamIdx::inv_tau_u,
        HAAPModelParamIdx::inv_tau_h,
        HAAPModelParamIdx::v_half_u,
        HAAPModelParamIdx::k_u,
        HAAPModelParamIdx::v_half_h,
        HAAPModelParamIdx::k_h,
        HAAPModelParamIdx::v_thresh,
        HAAPModelParamIdx::v_reset,
        HAAPModelParamIdx::delta_u,
    ];
    for p in order.iter() {
        let vals: Vec<f32> = cols.iter().map(|row| row[*p as usize]).collect();
        runner.set_haap_param_column(*p as usize, &vals);
    }
    for col in 0..3usize {
        let vals: Vec<f32> = cols
            .iter()
            .map(|row| {
                let (v, u, h) = resting_state(row);
                [v, u, h][col]
            })
            .collect();
        runner.set_haap_state_column(col, &vals);
    }
}

/// Микс для среднего мозга: ~1/3 эндогенных осцилляторов (хэш), остальные
/// тихие реле. Тот же принцип разрежения, что в brain_columns.
pub fn pacemaker_mix_columns(n: usize, genes: &PhysioGenes) -> Vec<[f32; N_COLS]> {
    const PACEMAKER_FRACTION: f64 = 0.34;
    (0..n)
        .map(|i| {
            let h =
                ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33) as f64 / u32::MAX as f64;
            if h < PACEMAKER_FRACTION {
                haap_columns(Archetype::Pacemaker, genes)
            } else {
                haap_columns(Archetype::Interneuron, genes)
            }
        })
        .collect()
}

/// Диапазон сдвига порога от гена para (мВ, в обе стороны).
const PARA_THRESHOLD_SPAN: f32 = 8.0;
