//! -------------------------------------------------------------------------
//! Filename:    src/brain.rs
//! Class:       neuworks
//! Purpose:     agent spiking brain + 3-factor plasticity
//! Description: BrainRunner (LIF-alpha) over CSR with region-gain matrix;
//!
//! delayed & immediate dopamine, soft homeostasis, frozen region pairs
//! (innate reflexes never unlearn), hyperpolarization (memory reset).
//! author: Atlasov anton
//! -------------------------------------------------------------------------

use crate::Grown;
use pillars::api::neurons::DopamineSTDP;
use pillars::zoo::HAAPModelParamIdx as P;
use pillars_connectome::{BrainRunner, ConnectomeData, NeuronConfig};

pub const DT_MS: f32 = 0.5;
pub const BRAIN_STEPS_PER_TICK: usize = 50; // 25 мс на тик мира (LIF)
/// HAAP: канонический dt VNC — тоньше шаг, шире запас устойчивости
/// (интегрирование при 0.5 делало реле би-стабильным: взрыв/тишина).
pub const DT_MS_HAAP: f32 = 0.05;
pub const BRAIN_STEPS_PER_TICK_HAAP: usize = 500; // тот же тик 25 мс
/// Постсинаптическая кинетика HAAP (мс).
const HAAP_TAU_SYN: f32 = 12.0;
/// Перевод мировых токов (масштаб LIF, мВ/шаг) в токи HAAP: при g_l=0.1
/// нужно ~7× чтобы сенсорный сигнал достигал порога, а фон оставался тих.
const HAAP_INPUT_SCALE: f32 = 8.0;
/// Смещение преобразователя LIF→HAAP: вычитает тонический эквивалент, чтобы
/// фон (0.12) оставался тишиной после масштабирования (v_inf ≈ −49).
const HAAP_INPUT_BIAS: f32 = -0.25;
/// Масштаб синаптических весов под HAAP: логнормальные grown-веса (медиана
/// 2, хвост 50+) при пороге в 25 мВ от покоя разгоняют рекуррентные петли —
/// режем, чтобы адаптация успевала (VNC-баланс). Тюнинго-ручки живут в
/// `PlasticityParams::{haap_syn_scale, haap_syn_cap}`.
const HAAP_SYN_SCALE: f32 = 0.15;
/// Потолок одного синапса после масштаба (клип хвоста).
const HAAP_SYN_CAP: f32 = 0.40;

/// Модель нейронов мозга.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrainModel {
    LifAlpha,
    /// HAAP (VNC-стабильность: адаптация против runaway, h-ток для плато).
    Haap,
}

#[derive(Clone, Debug)]
pub struct PlasticityParams {
    pub lr: f32,
    pub w_max: f32,
    pub a_ltp: f32,
    pub a_ltd: f32,
    pub homeo_rate: f32,
    /// Задержка дофамина в тиках мира (50 мс при стандартном тике).
    pub reward_delay_ticks: usize,
    /// Пары регионов, синапсы которых не пластичны (врождённые пути).
    pub frozen_pairs: Vec<(usize, usize)>,
    /// Критические периоды (developmental windows): пара (pre, post)
    /// заморожена до world-тика `thaw_tick`, затем становится пластичной.
    /// Мир детства стабилен, взрослая жизнь — обучаема.
    pub critical_pairs: Vec<(usize, usize, usize)>,
    /// Период мягкого гомеостаза в мировых тиках (0 = только при дофамине,
    /// как раньше). HAAP нуждается в тормозе каждый тик: без рефрактерного
    /// периода и с τ_syn=12 мс пластичные пути уходят в судорогу.
    pub homeo_every_ticks: usize,
    /// Масштаб начальных grown-весов под HAAP (LIF: не используется).
    pub haap_syn_scale: f32,
    /// Потолок ОДНОГО синапса после масштаба: клипует логнормальный хвост
    /// grown-весов (медиана 2, хвост 50+), иначе один хвостовой синапс
    /// единолично пинит постсинаптический нейрон → судорога.
    pub haap_syn_cap: f32,
    /// Домножать ли Δw дофамина на syn_scale (HAAP). false = шаг обучения
    /// в LIF-единицах (lr·D·e), клэмп всё равно w_max·syn_scale.
    pub haap_dopamine_scaled: bool,
}

impl Default for PlasticityParams {
    fn default() -> Self {
        // Рабочая точка аренов: embodied-настройка (LTD почти выключен,
        // иначе длительная сенсорная активность съедает LTP).
        Self {
            lr: 0.15,
            w_max: 6.0,
            a_ltp: 0.1,
            a_ltd: 0.005,
            homeo_rate: 0.25,
            reward_delay_ticks: 2,
            frozen_pairs: Vec::new(),
            critical_pairs: Vec::new(),
            homeo_every_ticks: 0,
            haap_syn_scale: HAAP_SYN_SCALE,
            haap_syn_cap: HAAP_SYN_CAP,
            haap_dopamine_scaled: true,
        }
    }
}

pub struct Brain {
    runner: BrainRunner,
    stdp: DopamineSTDP,
    pre_of: Vec<usize>,
    post_of: Vec<usize>,
    region_of: Vec<usize>,
    /// Гейны по парам регионов: value = gains[pre_region][post_region].
    #[allow(dead_code)] // сохраняем для интроспекции/диффов схем
    gains: Vec<Vec<f32>>,
    target_in_sum: Vec<f32>,
    inh_syn: Vec<bool>,
    /// World-тик (×BRAIN_STEPS_PER_TICK), когда синапс становится
    /// пластичным; usize::MAX = навсегда заморожен (тормозные пресинапсы,
    /// frozen_pairs без окна).
    thaw_at: Vec<usize>,
    pub n: usize,
    pub n_regions: usize,
    model: BrainModel,
    params: PlasticityParams,
    /// Масштаб весов этой модели (HAAP: 0.1 — см. HAAP_SYN_SCALE).
    syn_scale: f32,
    pending_reward: Vec<(usize, f32)>,
    step_count: usize,
}

impl Brain {
    pub fn new(grown: &Grown, gains: Vec<Vec<f32>>, params: PlasticityParams) -> Self {
        Self::with_model(grown, gains, params, BrainModel::LifAlpha, &[])
    }

    /// Мозг на HAAP с пер-neuron физиологией (архетипы × гены экспрессии).
    pub fn new_haap(
        grown: &Grown,
        gains: Vec<Vec<f32>>,
        params: PlasticityParams,
        columns: &[[f32; crate::physiology::N_COLS]],
    ) -> Self {
        Self::with_model(grown, gains, params, BrainModel::Haap, columns)
    }

    fn with_model(
        grown: &Grown,
        gains: Vec<Vec<f32>>,
        params: PlasticityParams,
        model: BrainModel,
        haap_columns: &[[f32; crate::physiology::N_COLS]],
    ) -> Self {
        let n_regions = gains.len();
        let n_total = grown.nt.len();
        let mut per: Vec<Vec<(usize, f32)>> = vec![vec![]; n_total];
        for &(pre, post, w) in &grown.edges {
            let (rp, rq) = (grown.region_of[pre], grown.region_of[post]);
            let gain = gains[rp][rq];
            if gain > 0.0 {
                per[pre].push((post, w * gain));
            }
        }
        let mut row_ptr = vec![0i32; n_total + 1];
        let mut ci = Vec::new();
        let mut vals = Vec::new();
        let haap = model == BrainModel::Haap;
        let (syn_scale, syn_cap) = if haap {
            (params.haap_syn_scale, params.haap_syn_cap)
        } else {
            (1.0, f32::INFINITY)
        };
        for (i, list) in per.iter().enumerate() {
            row_ptr[i] = ci.len() as i32;
            for &(post, w) in list {
                ci.push(post as i32);
                // Масштаб + клип хвоста: один синапс не может доминировать.
                vals.push(if haap { (w * syn_scale).min(syn_cap) } else { w });
            }
        }
        row_ptr[n_total] = ci.len() as i32;
        let conn = ConnectomeData::new(row_ptr, ci, vals, grown.nt.clone(), vec![]);
        let config = match model {
            BrainModel::LifAlpha => NeuronConfig::LifAlpha {
                dt_ms: DT_MS,
                tau_ref: 2.2,
                tau_m: 20.0,
                v_rest: -52.0,
                v_thresh: -45.0,
                v_reset: -52.0,
                tau_syn: 3.0,
                t_delay: 1.8,
            },
            BrainModel::Haap => NeuronConfig::HaapSynaptic {
                dt_ms: DT_MS_HAAP,
                tau_syn: HAAP_TAU_SYN,
            },
        };
        let mut runner = BrainRunner::new(conn, config).expect("BrainRunner init");
        if model == BrainModel::Haap {
            crate::physiology::apply_haap_columns(&mut runner, haap_columns);
        }

        let (n_neurons, n_syn, rp, cc, vv) = {
            let conn = &runner.connectome;
            (
                conn.n_neurons,
                conn.values.len(),
                conn.row_ptr.clone(),
                conn.col_indices.clone(),
                conn.values.clone(),
            )
        };
        let mut pre_of = vec![0usize; n_syn];
        let mut post_of = vec![0usize; n_syn];
        let mut inh_syn = vec![false; n_syn];
        for i in 0..n_neurons {
            for idx in rp[i] as usize..rp[i + 1] as usize {
                pre_of[idx] = i;
                post_of[idx] = cc[idx] as usize;
                inh_syn[idx] = grown.nt[i] < 0.0;
            }
        }
        // Пластичен с рождения (0); usize::MAX — навсегда заморожен
        // (только явные frozen_pairs): дофамин и гомеостаз их не трогают.
        // Тормозные пресинапсы (I→M) ОСТАЮТСЯ пластичными: награда ослабляет
        // торможение через LTD — без этого усиленный S→I запирает моторы.
        const THAW_NEVER: usize = usize::MAX;
        let mut thaw_at = vec![0usize; n_syn];
        for idx in 0..n_syn {
            let pair = (grown.region_of[pre_of[idx]], grown.region_of[post_of[idx]]);
            if params.frozen_pairs.contains(&pair) {
                thaw_at[idx] = THAW_NEVER;
                continue;
            }
            if let Some(&(_, _, thaw_tick)) = params
                .critical_pairs
                .iter()
                .find(|&&(a, b, _)| (a, b) == pair)
            {
                thaw_at[idx] = thaw_tick.saturating_mul(BRAIN_STEPS_PER_TICK);
            }
        }
        let mut target_in_sum = vec![0.0f32; n_neurons];
        for idx in 0..n_syn {
            if !inh_syn[idx] {
                target_in_sum[post_of[idx]] += vv[idx];
            }
        }
        let dt_for_model = if model == BrainModel::Haap { DT_MS_HAAP } else { DT_MS };
        let mut stdp = DopamineSTDP::new(n_syn, dt_for_model);
        stdp.w_max = params.w_max;
        stdp.a_ltp = params.a_ltp;
        stdp.a_ltd = params.a_ltd;
        Self {
            runner,
            stdp,
            pre_of,
            post_of,
            region_of: grown.region_of.clone(),
            gains,
            target_in_sum,
            inh_syn,
            thaw_at,
            n: n_neurons,
            n_regions,
            model,
            params,
            syn_scale,
            pending_reward: Vec::new(),
            step_count: 0,
        }
    }

    /// Шаг интеграции модели (мс).
    pub fn dt_ms(&self) -> f32 {
        if self.model == BrainModel::Haap {
            DT_MS_HAAP
        } else {
            DT_MS
        }
    }

    /// Диагностика HAAP: фактический dt колонки популяции (до/после run —
    /// BrainRunner перезаливает dt из конфига при расхождении).
    pub fn haap_dt_column(&self) -> Option<(f32, f32)> {
        if self.model != BrainModel::Haap {
            return None;
        }
        let col = self.runner.get_haap_param_column(P::dt as usize);
        col.iter().fold(None, |acc: Option<(f32, f32)>, &v| match acc {
            None => Some((v, v)),
            Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
        })
    }

    /// Диагностика HAAP: (min v, max v, среднее u) — живая мембрана.
    /// `None` для не-HAAP мозга.
    pub fn haap_state_stats(&self) -> Option<(f32, f32, f32)> {
        if self.model != BrainModel::Haap {
            return None;
        }
        let v = self.runner.get_haap_state_column(0);
        let u = self.runner.get_haap_state_column(1);
        let mut acc: Option<(f32, f32, f64)> = None;
        for i in 0..v.len() {
            acc = match acc {
                None => Some((v[i], v[i], u[i] as f64)),
                Some((lo, hi, us)) => Some((lo.min(v[i]), hi.max(v[i]), us + u[i] as f64)),
            };
        }
        acc.map(|(lo, hi, us)| (lo, hi, (us / v.len().max(1) as f64) as f32))
    }

    /// Подшагов мозга на тик мира.
    pub fn steps_per_tick(&self) -> usize {
        if self.model == BrainModel::Haap {
            BRAIN_STEPS_PER_TICK_HAAP
        } else {
            BRAIN_STEPS_PER_TICK
        }
    }

    /// Один шаг динамики + STDP-следы + отложенные награды.
    pub fn step(&mut self, ext: &[f32]) -> Vec<i32> {
        let spikes = match self.model {
            BrainModel::Haap => {
                let scaled: Vec<f32> =
                    ext.iter().map(|&x| x * HAAP_INPUT_SCALE + HAAP_INPUT_BIAS).collect();
                self.runner.run(1, &scaled).to_vec()
            }
            BrainModel::LifAlpha => self.runner.run(1, ext).to_vec(),
        };
        let n_syn = self.pre_of.len();
        let mut pre_s = vec![0i32; n_syn];
        let mut post_s = vec![0i32; n_syn];
        for idx in 0..n_syn {
            pre_s[idx] = spikes[self.pre_of[idx]];
            post_s[idx] = spikes[self.post_of[idx]];
        }
        self.stdp.step(&pre_s, &post_s);
        self.step_count += 1;

        // Мягкий гомеостаз по расписанию модели: при sums == target
        // множитель ровно 1.0 — контроль без дофамина бит-в-бит неподвижен.
        let period = self.params.homeo_every_ticks.max(1) * self.steps_per_tick();
        if self.params.homeo_every_ticks > 0 && self.step_count.is_multiple_of(period) {
            self.normalize();
        }

        let due: Vec<f32> = {
            let now = self.step_count;
            let mut keep = Vec::new();
            let mut due = Vec::new();
            for &(at, d) in &self.pending_reward {
                if now >= at {
                    due.push(d);
                } else {
                    keep.push((at, d));
                }
            }
            self.pending_reward = keep;
            due
        };
        if !due.is_empty() {
            let dopamine: f32 = due.iter().sum();
            self.apply_dopamine(dopamine);
            self.normalize();
        }
        spikes
    }

    /// Обнулить eligibility замороженных синапсов (учитывая окна оттаивания).
    fn zero_frozen_eligibility(&mut self) {
        let now = self.step_count;
        for (idx, &thaw) in self.thaw_at.iter().enumerate() {
            if now < thaw {
                self.stdp.eligibility[idx] = 0.0;
            }
        }
    }

    /// Границы нейронов региона [start, end) — блоки сплошные, см.
    /// [`Grown::region_bounds`].
    pub fn region_bounds(&self, region: usize) -> (usize, usize) {
        let start = self
            .region_of
            .iter()
            .position(|&r| r == region)
            .unwrap_or(0);
        let end = self
            .region_of
            .iter()
            .rposition(|&r| r == region)
            .map_or(start, |p| p + 1);
        (start, end)
    }

    /// Запланировать дофамин (применится через reward_delay_steps).
    pub fn schedule_reward(&mut self, dopamine: f32) {
        let at = self.step_count + self.steps_per_tick() * self.params.reward_delay_ticks;
        self.pending_reward.push((at, dopamine));
    }

    /// Задержка награды в тиках мира + 1 (длительность «жевания»).
    pub fn reward_delay_ticks(&self) -> usize {
        self.params.reward_delay_ticks + 1
    }

    /// Применить дофамин немедленно (протоколы с фиксацией веса после пробы).
    pub fn reward_now(&mut self, dopamine: f32) {
        self.apply_dopamine(dopamine);
        self.normalize();
    }

    /// Обнулить eligibility-следы всех синапсов: окно допустимости закрывается
    /// принудительно (иначе след не-награждённой пробы доживает до следующей
    /// награды и обобщается на неё).
    pub fn clear_eligibility(&mut self) {
        self.stdp.clear_eligibility();
    }

    /// Дофаминовая фиксация Δw = η·D·e — канонический SIMD-кернел движка
    /// (`dopamine_weight_update`): клэмп применяется только к синапсам с
    /// e ≠ 0, замороженные (e обнулён `zero_frozen_eligibility`) и врождённые
    /// хвосты > w_max сохраняются бит-в-бит.
    fn apply_dopamine(&mut self, dopamine: f32) {
        self.zero_frozen_eligibility();
        let w_cap = self.params.w_max * self.syn_scale;
        let lr = if self.params.haap_dopamine_scaled {
            self.params.lr * self.syn_scale
        } else {
            self.params.lr
        };
        let values = &mut self.runner.connectome.values;
        let n = values.len().min(self.stdp.eligibility.len());
        self.stdp.w_min = 0.0;
        self.stdp.w_max = w_cap;
        self.stdp.apply_dopamine(&mut values[..n], dopamine, lr);
    }

    /// Мягкий гомеостаз: суммы входящих ВОЗБУЖДАЮЩИХ весов каждого нейрона
    /// медленно тянутся к стартовым (полная нормализация отменяла бы дофамин).
    /// Тормозные пресинапсы не скейлятся (знак в сумме ломал бы масштаб).
    /// Навсегда-замороженные (thaw=usize::MAX) не трогаются: врождённые дуги
    /// неподвижны. Критическое окно тоже защищено (now < thaw).
    fn normalize(&mut self) {
        let mut sums = vec![0.0f32; self.n];
        for idx in 0..self.pre_of.len() {
            if !self.inh_syn[idx] {
                sums[self.post_of[idx]] += self.runner.connectome.values[idx];
            }
        }
        let rate = self.params.homeo_rate;
        let now = self.step_count;
        for idx in 0..self.pre_of.len() {
            if self.inh_syn[idx] {
                continue;
            }
            if now < self.thaw_at[idx] {
                continue;
            }
            let post = self.post_of[idx];
            let target = self.target_in_sum[post];
            if sums[post] > 1e-6 && target > 1e-6 {
                let scale = 1.0 + rate * (target / sums[post] - 1.0);
                if scale.is_finite() {
                    self.runner.connectome.values[idx] *= scale;
                }
            }
        }
    }

    /// Гиперполяризовать выбранные регионы на N шагов (сброс памяти,
    /// нейромодуляторное торможение).
    pub fn hyperpolarize(&mut self, regions: &[usize], steps: usize, current: f32) {
        let ext: Vec<f32> = self
            .region_of
            .iter()
            .map(|r| if regions.contains(r) { current } else { 0.0 })
            .collect();
        for _ in 0..steps {
            self.step(&ext);
        }
    }

    /// Число синапсов CSR (для диагностики/сводок).
    pub fn n_syn(&self) -> usize {
        self.pre_of.len()
    }

    /// Копия текущих значений CSR (сохранение существа).
    pub fn csr_values(&self) -> Vec<f32> {
        self.runner.connectome.values.clone()
    }

    /// Перезаписать значения CSR (наследование выученных весов). Длины
    /// должны совпадать с построенным CSR.
    pub fn set_csr_values(&mut self, vals: &[f32]) {
        let n = vals.len().min(self.runner.connectome.values.len());
        self.runner.connectome.values[..n].copy_from_slice(&vals[..n]);
    }

    /// Мин/макс всех синаптических весов (интроспекция/отладка).
    pub fn values_minmax(&self) -> (f32, f32) {
        self.runner
            .connectome
            .values
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            })
    }

    /// Средний вес пути region_a → region_b (и число синапсов).
    pub fn mean_path_weight(&self, a: usize, b: usize) -> (f32, usize) {
        let mut sum = 0.0f32;
        let mut cnt = 0usize;
        for idx in 0..self.pre_of.len() {
            if self.region_of[self.pre_of[idx]] == a && self.region_of[self.post_of[idx]] == b {
                sum += self.runner.connectome.values[idx];
                cnt += 1;
            }
        }
        (if cnt > 0 { sum / cnt as f32 } else { 0.0 }, cnt)
    }

    /// Сумма спайков нейронов региона за шаг.
    pub fn region_spikes(&self, spikes: &[i32], region: usize) -> i32 {
        let (a, b) = self.region_bounds(region);
        spikes[a..b].iter().sum()
    }
}

/// Гейн-матрица n×n, заполненная по правилу (pre, post) → gain.
pub fn gains_from(n_regions: usize, rules: &[(usize, usize, f32)]) -> Vec<Vec<f32>> {
    let mut g = vec![vec![0.0f32; n_regions]; n_regions];
    for &(a, b, v) in rules {
        g[a][b] = v;
    }
    g
}
