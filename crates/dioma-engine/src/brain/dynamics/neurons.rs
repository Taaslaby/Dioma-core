//! # Creature brain
//!
//! Мост между игровым существом и SiMuse. `Brain` хранит CSR-коннектом,
//! запускает CPU/GPU-динамику и применяет трёхфакторную пластичность:
//! спайки + eligibility + дофамин.
//!
//! Важная граница состояния:
//!
//! - `Grown` — генетическая проводка при рождении;
//! - текущие CSR-веса — опыт жизни;
//! - `memory_lt`/snapshot — долговременная память для сна и Болдуина.
//!
//! Все методы, связанные с backend, должны сохранять одинаковую игровую
//! семантику на CPU и GPU.

use crate::brain::gpu::GpuDynamics;
use crate::brain::gpu::LearnGpu;
use crate::Grown;
use pillars::api::learn::brain::{Brain as PlastBrain, StdpConfig};
use pillars::math::fast_exp::fast_exp;
use pillars::zoo::HAAPModelParamIdx as P;
use pillars_connectome::{BrainRunner, ConnectomeData, NeuronConfig};

pub const DT_MS: f32 = 0.5;
/// Число LIF-подшагов в одном игровом тике (100 × 0.5 мс = 50 мс).
/// (×2: тик 50 мс — мировой тик удвоен, 25 → 50 мс, как в C3
/// `App::GetWorldTickInterval() == 50`; нейронного времени за тик стало
/// вдвое больше, а за секунду — столько же.)
pub const BRAIN_STEPS_PER_TICK: usize = 100;
/// HAAP: канонический dt VNC — тоньше шаг, шире запас устойчивости
/// (интегрирование при 0.5 делало реле би-стабильным: взрыв/тишина).
pub const DT_MS_HAAP: f32 = 0.05;
pub const BRAIN_STEPS_PER_TICK_HAAP: usize = 1000; // 1000 × 0.05 мс = 50 мс (×2: тик 50 мс)
/// Постсинаптическая кинетика HAAP (мс).
const HAAP_TAU_SYN: f32 = 12.0;
/// Перевод мировых токов (масштаб LIF, мВ/шаг) в токи HAAP: при `g_l = 0.1`
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
/// Настройки обучения мозга и ограничения изменения синапсов.
pub struct PlasticityParams {
    pub lr: f32,
    pub w_max: f32,
    pub a_ltp: f32,
    pub a_ltd: f32,
    pub homeo_rate: f32,
    /// Задержка дофамина в тиках мира (50 мс при тике 50 мс).
    /// (÷2: тик 50 мс — подшагов на тик стало вдвое больше, поэтому 1 тик
    /// даёт ту же задержку в подшагах/реальном времени, что 2 тика при 25 мс.)
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
    /// Регионы-ПРЕ, чьи исходящие рёбра исключены из мягкого гомеостаза
    /// (target_in_sum / normalize / sleep_consolidate). Стадия B: исходящие
    /// рёбра ассоциативного хаба (ASSOC_E/ASSOC_I) не должны участвовать
    /// в гомеостазе ядра рефлексов — иначе необученные веса хаба в моторах
    /// раздувают цель нормализации, и после LTD ядра (наказание стеной)
    /// восстановительная масса утекает в тихие рёбра хаба вместо гоминга
    /// (гоминг гаснет: замерено на стадии B, см. отчёт). Пластичность
    /// (дофамин) хаба не затрагивается — он учится как обычно.
    pub homeo_exclude_pre: Vec<usize>,
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
            reward_delay_ticks: 1,
            frozen_pairs: Vec::new(),
            critical_pairs: Vec::new(),
            homeo_every_ticks: 0,
            haap_syn_scale: HAAP_SYN_SCALE,
            haap_syn_cap: HAAP_SYN_CAP,
            haap_dopamine_scaled: true,
            homeo_exclude_pre: Vec::new(),
        }
    }
}

/// Подключаемая динамика мембраны: CPU (SIMD-раннер pillars-connectome), GPU
/// (wgpu-сеть pillars-wgpu, alpha-lif) или «пул» — общий GPU-раннер всей
/// арены (см. `brain::gpu_pool::BrainPool`). Пластичность/гомеостаз/сон
/// живут в `Brain` поверх — смена динамики не меняет семантику обучения.
enum Dynamics {
    Cpu(BrainRunner),
    Gpu(GpuDynamics),
    /// Мембрана считается БАТЧ-пулом арены (блочно-диагональная сеть):
    /// здесь только зеркало весов — источник правды для пластичности/
    /// гомеостаза/сна/наследственности (смещения существа в общей сети
    /// хранит пул, `gpu_pool::PoolSlot`). Спайки приходят через
    /// `Brain::ingest_batched_flat`, а не из своего раннера.
    Pooled(PooledDynamics),
}

/// Зеркало весов существа в батч-пуле (`Dynamics::Pooled`).
pub struct PooledDynamics {
    /// CSR-коннектом существа: `values` — каноническое зеркало весов
    /// (источник правды для пластичности/гомеостаза/сна); перед шагом пул
    /// перезаливает блок в GPU, если зеркало менялось.
    conn: ConnectomeData,
    /// Зеркало менялось с прошлого аплоада?
    dirty: bool,
}

impl PooledDynamics {
    fn values(&self) -> &[f32] {
        &self.conn.values
    }

    fn values_mut(&mut self) -> &mut [f32] {
        self.dirty = true;
        &mut self.conn.values
    }

    fn import_values(&mut self, vals: &[f32]) -> Result<(), String> {
        self.conn.import_values(vals)?;
        self.dirty = true;
        Ok(())
    }

    fn save_baked<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.conn.save_baked(w)
    }
}

impl Dynamics {
    /// Один спайковый подшаг: возвращает пер-нейронные спайки (0/1).
    fn run_step(&mut self, ext: &[f32]) -> &[i32] {
        match self {
            Dynamics::Cpu(r) => r.run(1, ext),
            Dynamics::Gpu(g) => g.run_step(ext),
            // Пул гоняет ВСЕХ существ одним шагом — одиночный подшаг тут
            // недостижим (прожигание инстинктов в пуле не поддерживается);
            // тихо молчим, как GpuDynamics при сбое.
            Dynamics::Pooled(_) => {
                debug_assert!(false, "Dynamics::Pooled::run_step — пул не умеет одиночных шагов");
                EMPTY_SPIKES.get_or_init(Vec::new).as_slice()
            }
        }
    }

    /// Пачка `n` подшагов: заполняет `out` n векторами пер-нейронных спайков.
    /// CPU-раннер гонит n шагов одним вызовом `run(n, ext)` (тот же цикл, что
    /// n × run(1) — бит-в-бит); GPU — батчем с одним ридбеком на пачку.
    fn run_steps_into(&mut self, ext: &[f32], n: usize, out: &mut Vec<Vec<i32>>) {
        match self {
            Dynamics::Cpu(r) => {
                out.clear();
                let nn = r.n_neurons();
                let flat = r.run(n, ext);
                for k in 0..n {
                    let mut v = Vec::with_capacity(nn);
                    v.extend_from_slice(&flat[k * nn..(k + 1) * nn]);
                    out.push(v);
                }
            }
            Dynamics::Gpu(g) => g.run_steps_into(ext, n, out),
            Dynamics::Pooled(_) => {
                debug_assert!(false, "Dynamics::Pooled::run_steps_into — пул не умеет пачек");
                out.clear();
                let nn = ext.len();
                for _ in 0..n {
                    out.push(vec![0i32; nn]);
                }
            }
        }
    }

    /// Каноническое зеркало весов (источник правды для спайков).
    fn values(&self) -> &[f32] {
        match self {
            Dynamics::Cpu(r) => &r.connectome.values,
            Dynamics::Gpu(g) => g.values(),
            Dynamics::Pooled(p) => p.values(),
        }
    }

    fn values_mut(&mut self) -> &mut [f32] {
        match self {
            Dynamics::Cpu(r) => &mut r.connectome.values,
            Dynamics::Gpu(g) => g.values_mut(),
            Dynamics::Pooled(p) => p.values_mut(),
        }
    }

    fn export_values(&self) -> Vec<f32> {
        self.values().to_vec()
    }

    fn save_baked<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            Dynamics::Cpu(r) => r.connectome.save_baked(w),
            Dynamics::Gpu(g) => g.save_baked(w),
            Dynamics::Pooled(p) => p.save_baked(w),
        }
    }

    fn import_values(&mut self, vals: &[f32]) -> Result<(), String> {
        match self {
            Dynamics::Cpu(r) => r.connectome.import_values(vals),
            Dynamics::Gpu(g) => g.import_values(vals),
            Dynamics::Pooled(p) => p.import_values(vals),
        }
    }

    /// Диагностические колонки HAAP (GPU/пул — только LIF → пусто).
    fn get_haap_param_column(&self, col: usize) -> Vec<f32> {
        match self {
            Dynamics::Cpu(r) => r.get_haap_param_column(col),
            Dynamics::Gpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    fn get_haap_state_column(&self, col: usize) -> Vec<f32> {
        match self {
            Dynamics::Cpu(r) => r.get_haap_state_column(col),
            Dynamics::Gpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }
}

/// Переиспользуемый пустой буфер для тихих шагов пула.
static EMPTY_SPIKES: std::sync::OnceLock<Vec<i32>> = std::sync::OnceLock::new();
/// Пустое зеркало весов для `pooled_weights` не-пулевых мозгов.
static EMPTY_WEIGHTS: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

pub struct Brain {
    dynamics: Dynamics,
    /// Пластичность: ленивая дофаминовая STDP (pillars::api::learn::brain).
    /// Веса здесь — зеркало `dynamics.values()`; награда обновляет зеркало,
    /// затем синкается обратно в динамику (источник правды для спайков).
    plast: PlastBrain,
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
    /// Кэш промасштабированного входа HAAP (без аллокации на подшаг).
    haap_cache: Vec<f32>,
    /// Переиспользуемые буферы для `normalize`/`zero_frozen_eligibility`.
    norm_sums: Vec<f32>,
    frozen_keep: Vec<bool>,
    /// Долговременная копия весов для сна-восстановления (`memory_recall`).
    memory_lt: Vec<f32>,
    /// GPU-обучение целиком: eligibility/награды живут на GPU (dopamine.wgsl),
    /// ридбеки за тик — только моторные суммы. true = Dynamics::Gpu + learning.
    gpu_learn: bool,
    /// Переиспользуемый буфер подшага для `ingest_batched_flat` (пул):
    /// спайки из общего flat-буфера конвертируются u32 → i32 без аллокаций.
    pool_spikes: Vec<i32>,
    // ── Тик-пластичность (mean-field STDP: один проход по синапсам за тик) ──
    /// Счётчик спайков нейрона за текущий тик.
    tick_spike_count: Vec<u32>,
    /// Σ индексов подшагов спайков (средний момент спайка = sum/count).
    tick_spike_sum_s: Vec<u64>,
    /// Зеркала пред/пост следов на ТЕКУЩИЙ подшаг (ядро: t = t·d + sp).
    tick_trace_pre: Vec<f32>,
    tick_trace_post: Vec<f32>,
    /// Траектории следов за тик (n×S, ряд = подшаг): для причинных членов
    /// LTP/LTD — след пре на момент среднего спайка поста и наоборот.
    tick_trace_pre_hist: Vec<f32>,
    tick_trace_post_hist: Vec<f32>,
    /// Номер подшага внутри тика (0..S).
    tick_row: usize,
    /// Тик отмечался спайками (CPU-путь)? Пул/GPU-пути — без apply.
    tick_active: bool,
}

impl Brain {
    pub fn new(grown: &Grown, gains: Vec<Vec<f32>>, params: PlasticityParams) -> Self {
        Self::with_model(grown, gains, params, BrainModel::LifAlpha, &[], false)
    }

    /// Мозг на GPU-динамике (pillars-wgpu, alpha-lif): та же пластичность,
    /// гомеостаз, сон и Болдуин — но мембрана считается на GPU. При
    /// отсутствии адаптера тихо падает на CPU-динамику (`try_new` вернул
    /// ошибку → `with_model` собрал `Dynamics::Cpu`).
    pub fn new_gpu(grown: &Grown, gains: Vec<Vec<f32>>, params: PlasticityParams) -> Self {
        Self::with_model(grown, gains, params, BrainModel::LifAlpha, &[], true)
    }

    /// Мозг на HAAP с пер-neuron физиологией (архетипы × гены экспрессии).
    pub fn new_haap(
        grown: &Grown,
        gains: Vec<Vec<f32>>,
        params: PlasticityParams,
        columns: &[[f32; crate::biology::physiology::N_COLS]],
    ) -> Self {
        Self::with_model(grown, gains, params, BrainModel::Haap, columns, false)
    }

    fn with_model(
        grown: &Grown,
        gains: Vec<Vec<f32>>,
        params: PlasticityParams,
        model: BrainModel,
        haap_columns: &[[f32; crate::biology::physiology::N_COLS]],
        gpu: bool,
    ) -> Self {
        // STDP-следы затухают в денормали → SIMD замедляется ×10; FTZ+DAZ
        // сбрасывает их в ноль (см. brain::perf). Один раз на поток.
        crate::brain::perf::flush_denormals();
        let n_regions = gains.len();
        let haap = model == BrainModel::Haap;
        let (syn_scale, syn_cap) = if haap {
            (params.haap_syn_scale, params.haap_syn_cap)
        } else {
            (1.0, f32::INFINITY)
        };
        let conn = build_connectome(grown, &gains, haap, syn_scale, syn_cap);
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
        // Эффективный lr: HAAP-дофамин домножается на syn_scale (иначе шаг
        // обучения в LIF-единицах несовместим с масштабом HAAP-весов).
        let eff_lr = if haap && params.haap_dopamine_scaled {
            params.lr * syn_scale
        } else {
            params.lr
        };
        let dt_for_model = if model == BrainModel::Haap {
            DT_MS_HAAP
        } else {
            DT_MS
        };
        // Plast-зеркало и маски — ДО передачи conn в динамику (движок
        // забирает conn во владение: BrainRunner/GpuDynamics клонируют/едят
        // его целиком); снапшот весов — для memory_lt (наследственность).
        let (plast, pre_of, post_of, inh_syn, thaw_at, target_in_sum) =
            plast_and_masks(&conn, grown, &params, eff_lr, dt_for_model, syn_scale);
        let (n_neurons, memory_lt) = (conn.n_neurons, conn.values.clone());
        // Динамика: GPU (wgpu, только LIF) с тихим CPU-фолбэком, либо CPU SIMD.
        // GpuDynamics::try_new клонирует conn — цена один раз на сборку мозга.
        let mut dynamics = if gpu {
            match GpuDynamics::try_new(conn.clone(), &config) {
                Ok(g) => Dynamics::Gpu(g),
                Err(e) => {
                    eprintln!("[dioma] GPU-динамика недоступна ({e}), фолбэк на CPU");
                    let mut runner = BrainRunner::new(conn, config).expect("BrainRunner init");
                    if model == BrainModel::Haap {
                        crate::biology::physiology::apply_haap_columns(&mut runner, haap_columns);
                    }
                    Dynamics::Cpu(runner)
                }
            }
        } else {
            let mut runner = BrainRunner::new(conn, config).expect("BrainRunner init");
            if model == BrainModel::Haap {
                crate::biology::physiology::apply_haap_columns(&mut runner, haap_columns);
            }
            Dynamics::Cpu(runner)
        };
        // GPU-обучение целиком: если динамика на GPU (LIF), включаем
        // dopamine-eligibility на GPU — те же константы распада, что CPU plast
        // (fast_exp, dt модели, tau из StdpConfig ниже). Ридбеки за тик —
        // только моторные суммы (8 байт).
        let gpu_learn = match &mut dynamics {
            Dynamics::Gpu(g) => {
                g.enable_learning(LearnGpu {
                    d_pre: fast_exp(-dt_for_model / 20.0),
                    d_post: fast_exp(-dt_for_model / 20.0),
                    d_e: fast_exp(-dt_for_model / 1000.0),
                    a_ltp: params.a_ltp,
                    a_ltd: params.a_ltd,
                    lr: eff_lr,
                    w_min: 0.0,
                    w_cap: params.w_max * syn_scale,
                });
                true
            }
            _ => false,
        };
        Self {
            dynamics,
            plast,
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
            haap_cache: if model == BrainModel::Haap { vec![0.0; n_neurons] } else { Vec::new() },
            norm_sums: Vec::new(),
            frozen_keep: Vec::new(),
            memory_lt,
            gpu_learn,
            pool_spikes: Vec::new(),
            tick_spike_count: Vec::new(),
            tick_spike_sum_s: Vec::new(),
            tick_trace_pre: Vec::new(),
            tick_trace_post: Vec::new(),
            tick_trace_pre_hist: Vec::new(),
            tick_trace_post_hist: Vec::new(),
            tick_row: 0,
            tick_active: false,
        }
    }

    /// Мозг в батч-пуле арены (`brain::gpu_pool::BrainPool`): мембрана
    /// считается ОБЩИМ GPU-раннером всех существ (блочно-диагональная сеть),
    /// здесь — пластичность/гомеостаз/сон на CPU поверх зеркала весов.
    /// `csr` регистрируется в пуле ВЫЗЫВАЮЩИМ (`BrainPool::register`, см.
    /// `creature::assemble_pooled`); смещения блока хранит пул. Прожигание
    /// инстинктов в пуле не поддерживается (одиночные шаги динамики
    /// недостижимы — пул гоняет всех разом; у дефолтного генома инстинктов
    /// нет, так что обычные существа пула рождаются «пустыми»).
    pub fn new_pooled(
        grown: &Grown,
        gains: Vec<Vec<f32>>,
        params: PlasticityParams,
        csr: ConnectomeData,
    ) -> Self {
        crate::brain::perf::flush_denormals();
        let n_regions = gains.len();
        let eff_lr = params.lr;
        // Пул — только LIF: syn_scale = 1 (как у CPU/GPU LIF-мозгов).
        let (plast, pre_of, post_of, inh_syn, thaw_at, target_in_sum) =
            plast_and_masks(&csr, grown, &params, eff_lr, DT_MS, 1.0);
        let n = csr.n_neurons;
        let dynamics = Dynamics::Pooled(PooledDynamics {
            conn: csr,
            dirty: false,
        });
        let memory_lt = plast.weights().to_vec();
        Self {
            dynamics,
            plast,
            pre_of,
            post_of,
            region_of: grown.region_of.clone(),
            gains,
            target_in_sum,
            inh_syn,
            thaw_at,
            n,
            n_regions,
            model: BrainModel::LifAlpha,
            params,
            syn_scale: 1.0,
            pending_reward: Vec::new(),
            step_count: 0,
            haap_cache: Vec::new(),
            norm_sums: Vec::new(),
            frozen_keep: Vec::new(),
            memory_lt,
            gpu_learn: false,
            pool_spikes: Vec::new(),
            tick_spike_count: Vec::new(),
            tick_spike_sum_s: Vec::new(),
            tick_trace_pre: Vec::new(),
            tick_trace_post: Vec::new(),
            tick_trace_pre_hist: Vec::new(),
            tick_trace_post_hist: Vec::new(),
            tick_row: 0,
            tick_active: false,
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

    /// Рабочий потолок пластичности весов (params.w_max): дофамин клэмпит
    /// синапсы с ненулевой eligibility в [0, w_max·syn_scale]. Прожигание
    /// инстинктов при рождении может поднять его выше (`reward_now_with_cap`).
    pub fn plasticity_w_max(&self) -> f32 {
        self.params.w_max * self.syn_scale
    }

    /// Динамика на GPU? (диагностика/пробы; HAAP всегда CPU). `Pooled` тоже
    /// считает мембрану на GPU (общий раннер арены).
    pub fn on_gpu(&self) -> bool {
        matches!(self.dynamics, Dynamics::Gpu(_) | Dynamics::Pooled(_))
    }

    /// DEBUG: CPU-столбец v (сверка с GPU в диагностике паритета).
    pub fn debug_cpu_v(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Cpu(r) => r.get_v().to_vec(),
            // Пул: у мозга нет своего раннера — диагностические колонки пусты.
            Dynamics::Gpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: CPU-столбец g (проводимость, только LifAlpha).
    pub fn debug_cpu_g(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Cpu(r) => r.get_alpha_g(),
            Dynamics::Gpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: CPU-счётчик рефрактерности (только LifAlpha).
    pub fn debug_cpu_refr(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Cpu(r) => r.get_alpha_refr(),
            Dynamics::Gpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: CPU-вход, потреблённый на последнем шаге (только LifAlpha).
    pub fn debug_cpu_in(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Cpu(r) => r.get_alpha_debug_in(),
            Dynamics::Gpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: CPU-кольцо аксональной задержки (только LifAlpha).
    pub fn debug_cpu_delay(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Cpu(r) => r.get_alpha_delay(),
            Dynamics::Gpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: список заспайкавших пресинапсов последнего шага (только LifAlpha).
    pub fn debug_cpu_fired(&self) -> Vec<usize> {
        match &self.dynamics {
            Dynamics::Cpu(r) => r.get_alpha_fired(),
            Dynamics::Gpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: знак торможения по каждому синапсу (только для ручного скеттера).
    pub fn debug_inh_syn(&self) -> Vec<bool> {
        self.inh_syn.clone()
    }

    /// DEBUG: CPU-позиция кольца задержки (только LifAlpha).
    pub fn debug_cpu_delay_pos(&self) -> usize {
        match &self.dynamics {
            Dynamics::Cpu(r) => r.get_alpha_delay_pos(),
            Dynamics::Gpu(_) | Dynamics::Pooled(_) => 0,
        }
    }

    /// DEBUG: ring-буфер GPU-динамики (пусто на CPU).
    pub fn debug_gpu_ring(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Gpu(g) => g.debug_ring(),
            Dynamics::Cpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: спайки последнего шага GPU-динамики (пусто на CPU).
    pub fn debug_gpu_spikes(&self) -> Vec<i32> {
        match &self.dynamics {
            Dynamics::Gpu(g) => g.debug_spikes(),
            Dynamics::Cpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: вход скеттера GPU для пост-нейрона (src, вес, знак, задержка).
    pub fn debug_gpu_scatter_row(&self, post: usize) -> Vec<(u32, f32, f32, u32)> {
        match &self.dynamics {
            Dynamics::Gpu(g) => g.debug_scatter_row(post),
            Dynamics::Cpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: буфер весов GPU (weights_b).
    pub fn debug_gpu_weights(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Gpu(g) => g.debug_weights(),
            Dynamics::Cpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: состояния GPU-популяции (пусто на CPU).
    pub fn debug_gpu_states(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Gpu(g) => g.debug_states(),
            Dynamics::Cpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: i_in GPU-популяции (пусто на CPU).
    pub fn debug_gpu_i_in(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Gpu(g) => g.debug_i_in(),
            Dynamics::Cpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: slot GPU-сети.
    pub fn debug_gpu_slot(&self) -> u32 {
        match &self.dynamics {
            Dynamics::Gpu(g) => g.debug_slot(),
            Dynamics::Cpu(_) | Dynamics::Pooled(_) => 0,
        }
    }

    /// DEBUG: eligibility-буфер GPU (сверка с CPU plast пошагово).
    pub fn debug_gpu_elig(&self) -> Vec<f32> {
        match &self.dynamics {
            Dynamics::Gpu(g) => g.debug_elig(),
            Dynamics::Cpu(_) | Dynamics::Pooled(_) => Vec::new(),
        }
    }

    /// DEBUG: per-neuron трассы GPU (pre, post).
    pub fn debug_gpu_traces(&self) -> (Vec<f32>, Vec<f32>) {
        match &self.dynamics {
            Dynamics::Gpu(g) => g.debug_traces(),
            Dynamics::Cpu(_) | Dynamics::Pooled(_) => (Vec::new(), Vec::new()),
        }
    }

    /// DEBUG: eligibility CPU-plast, приведённая к текущему шагу (сверка с
    /// GPU пошагово). Мутабельная — fold_eligibility меняет last_sync.
    pub fn debug_plast_elig(&mut self) -> Vec<f32> {
        self.plast.fold_eligibility();
        self.plast.eligibility().to_vec()
    }

    /// DEBUG: трассы CPU-plast (pre, post) — сверка с GPU пошагово.
    pub fn debug_plast_traces(&self) -> (Vec<f32>, Vec<f32>) {
        self.plast.traces()
    }

    /// Диагностика HAAP: фактический dt колонки популяции (до/после run —
    /// BrainRunner перезаливает dt из конфига при расхождении).
    pub fn haap_dt_column(&self) -> Option<(f32, f32)> {
        if self.model != BrainModel::Haap {
            return None;
        }
        let col = self.dynamics.get_haap_param_column(P::dt as usize);
        col.iter()
            .fold(None, |acc: Option<(f32, f32)>, &v| match acc {
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
        let v = self.dynamics.get_haap_state_column(0);
        let u = self.dynamics.get_haap_state_column(1);
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
    //
    // Делегирует `step_into`; владеющая версия. Хорячий путь: до 500 раз на
    // тик существа (HAAP). `step_into` переиспользует буфер вызывающего и
    // устраняет копию `n_neurons` на подшаг.
    pub fn step(&mut self, ext: &[f32]) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.n);
        self.step_into(ext, &mut out);
        out
    }

    /// Неаллоцирующий шаг: пишет спайки в `out` (переиспользуя его буфер,
    /// удлиняя при необходимости). `step()` = `step_into` + owned.
    /// Хорячий путь: вызывается до 500 раз на тик существа (HAAP), и `.to_vec()`
    /// в `step()` аллоцировал бы копию `n_neurons` int'ов каждый подшаг.
    pub fn step_into(&mut self, ext: &[f32], out: &mut Vec<i32>) {
        // HAAP-вход масштабируем в переиспользуемый кэш (без аллокации),
        // затем сразу копируем спайки в `out`, отпуская borrow на runner,
        // чтобы ниже можно было взять `&mut self.plast`.
        debug_assert_eq!(ext.len(), self.n, "выход мозга != n");
        if self.model == BrainModel::Haap {
            debug_assert_eq!(self.haap_cache.len(), ext.len(), "haap_cache size mismatch");
            for (dst, &x) in self.haap_cache.iter_mut().zip(ext.iter()) {
                *dst = x * HAAP_INPUT_SCALE + HAAP_INPUT_BIAS;
            }
        }
        // Копируем спайки в `out`, чтобы перестать держать borrow на runner.
        out.clear();
        if self.gpu_learn {
            // GPU: один подшаг + eligibility на GPU (step_into-путь:
            // инстинкты/гиперполяризация). Спайки скачиваем — вызывающий
            // их потребляет.
            if let Dynamics::Gpu(g) = &mut self.dynamics {
                let input = if self.model == BrainModel::Haap {
                    &self.haap_cache
                } else {
                    ext
                };
                if let Ok(spikes) = g.run_step_learn(input) {
                    out.extend_from_slice(spikes);
                    self.post_substep_tick();
                    return;
                }
            }
        }
        if self.model == BrainModel::Haap {
            let spikes = self.dynamics.run_step(&self.haap_cache);
            out.extend_from_slice(spikes);
        } else {
            let spikes = self.dynamics.run_step(ext);
            out.extend_from_slice(spikes);
        }
        self.post_substep(out);
    }

    /// Пачка подшагов одним вызовом: динамика гонит все `n` шагов (GPU — без
    /// ожиданий на подшаг), затем пластичность/гомеостаз/награды применяются
    /// по каждому вектору спайков в том же порядке, что в `step_into`.
    /// Семантически бит-в-бит равна n последовательным `step_into` (внутри
    /// тика веса константны — награды и гомеостаз живут на границах тиков).
    /// Для GPU-обучения (`gpu_learn`) eligibility уже посчитан на GPU внутри
    /// `run_tick_learn` — здесь только счётчик/гомеостаз/награды.
    pub fn step_batch(&mut self, ext: &[f32], n: usize, hist: &mut Vec<Vec<i32>>) {
        debug_assert_eq!(ext.len(), self.n, "выход мозга != n");
        if self.gpu_learn {
            // GPU: eligibility живёт на GPU (step_many_learn), ридбек не нужен.
            if let Dynamics::Gpu(g) = &mut self.dynamics {
                if let Ok((_, _)) = g.run_tick_learn(ext, n, 0, 0, 0, 0) {
                    for _ in 0..n {
                        self.post_substep_tick();
                    }
                    return;
                }
                // сбой GPU — тихий фолбэк на CPU-путь ниже
            }
        }
        // HAAP-вход масштабируем один раз на пачку (вход внутри тика один и тот же).
        if self.model == BrainModel::Haap {
            debug_assert_eq!(self.haap_cache.len(), ext.len(), "haap_cache size mismatch");
            for (dst, &x) in self.haap_cache.iter_mut().zip(ext.iter()) {
                *dst = x * HAAP_INPUT_SCALE + HAAP_INPUT_BIAS;
            }
            self.dynamics.run_steps_into(&self.haap_cache, n, hist);
        } else {
            self.dynamics.run_steps_into(ext, n, hist);
        }
        for v in hist.iter() {
            self.post_substep(v);
        }
    }

    /// Батч-тик с моторными суммами: возвращает (ml, mr) — суммы спайков
    /// моторных пулов за тик. GPU-путь (`gpu_learn`) считает их на GPU и
    /// отдаёт ОДНИМ ридбеком (8 байт) вместо hist; CPU-путь — как раньше
    /// через `step_batch` + hist. `hist` заполняется только на CPU-пути.
    pub fn step_batch_motor(
        &mut self,
        ext: &[f32],
        n: usize,
        ml: (usize, usize),
        mr: (usize, usize),
        hist: &mut Vec<Vec<i32>>,
    ) -> (i32, i32) {
        debug_assert_eq!(ext.len(), self.n, "выход мозга != n");
        if self.gpu_learn {
            if let Dynamics::Gpu(g) = &mut self.dynamics {
                if let Ok((a, b)) =
                    g.run_tick_learn(ext, n, ml.0 as u32, ml.1 as u32, mr.0 as u32, mr.1 as u32)
                {
                    for _ in 0..n {
                        self.post_substep_tick();
                    }
                    return (a as i32, b as i32);
                }
            }
        }
        // CPU-путь (и сбой GPU): полный hist, суммы считаем здесь.
        self.step_batch(ext, n, hist);
        let mut ml_sum = 0i32;
        let mut mr_sum = 0i32;
        for v in hist.iter() {
            ml_sum += v[ml.0..ml.1].iter().sum::<i32>();
            mr_sum += v[mr.0..mr.1].iter().sum::<i32>();
        }
        (ml_sum, mr_sum)
    }

    /// Пулевой путь: проглотить `spt` подшагов спайков из ОБЩЕГО flat-буфера
    /// батч-пула (все существа в одной сети; `neuron_off` — смещение этого
    /// существа). Пластичность/счётчик/гомеостаз/отложенные награды — ровно
    /// как в `step_batch` (CPU-путь), моторные суммы — из локальных границ.
    /// Порядок применения подшагов бит-в-бит тот же, что у последовательного
    /// тика, поэтому траектория обучения пула эквивалентна CPU-семантике
    /// (спайки GPU могут отличаться в 1 ulp — см. отчёт).
    pub fn ingest_batched_flat(
        &mut self,
        flat: &[u32],
        spt: usize,
        neuron_off: usize,
        ml: (usize, usize),
        mr: (usize, usize),
    ) -> (i32, i32) {
        let n_total = if spt > 0 {
            flat.len() / spt
        } else {
            0
        };
        debug_assert!(neuron_off + self.n <= n_total, "смещение пула вне flat");
        let mut ml_sum = 0i32;
        let mut mr_sum = 0i32;
        // Буфер вынимаем из self (post_substep требует &mut self, а спайки —
        // займ из того же self): заполнение из flat, возврат после цикла —
        // переиспользование без аллокации на подшаг.
        let mut spikes = std::mem::take(&mut self.pool_spikes);
        for k in 0..spt {
            let base = k * n_total + neuron_off;
            for &s in &flat[base + ml.0..base + ml.1] {
                ml_sum += s as i32;
            }
            for &s in &flat[base + mr.0..base + mr.1] {
                mr_sum += s as i32;
            }
            spikes.clear();
            spikes.extend(flat[base..base + self.n].iter().map(|&s| s as i32));
            self.post_substep(&spikes);
        }
        self.pool_spikes = spikes;
        (ml_sum, mr_sum)
    }

    /// Зеркало весов для батч-пула + флаг «менялось» (арена передаёт пулу
    /// перед шагом; см. `brain::gpu_pool::BrainPool::sync_weights`).
    pub fn pooled_weights(&self) -> (&[f32], bool) {
        match &self.dynamics {
            Dynamics::Pooled(p) => (p.values(), p.dirty),
            _ => (EMPTY_WEIGHTS.get_or_init(|| Vec::new()).as_slice(), false),
        }
    }

    /// Сбросить флаг dirty пула после заливки весов в GPU.
    pub fn pooled_clear_dirty(&mut self) {
        if let Dynamics::Pooled(p) = &mut self.dynamics {
            p.dirty = false;
        }
    }

    /// Пост-обработка одного подшага: пластичность, счётчик, гомеостаз,
    /// отложенные награды — общий код для `step_into` и `step_batch`.
    /// Пост-обработка одного подшага (CPU-путь): пластичность потребляет
    /// пер-нейронные спайки, затем счётчик/гомеостаз/награды.
    fn post_substep(&mut self, spikes: &[i32]) {
        self.tick_note_spikes(spikes);
        self.post_substep_tick();
    }

    // ─── Тик-пластичность: mean-field STDP, один проход за тик ───────────
    //
    // Профилирование (760н): per-substep `plast.step` = 96% времени мозга
    // (5.5M событий/тик — рекуррентная кора не разрежена). Замена: за тик
    // копим статистику спайков (счётчик + средний момент) и траектории
    // следов (n×S, дёшево), затем ОДИН проход по синапсам с причинными
    // членами: LTP — след ПРЕ на момент среднего спайка ПОСТА (каузальность
    // «после»), LTD — след ПОСТА на момент среднего спайка ПРЕ. Потолок
    // события ×count. Eligibility затухает одним d_e^S за тик (тот же
    // per-second темп: 0.9995^100 = 0.951). Run-vs-run детерминизм — без
    // планировщика; поведение отличается от per-substep (крупнее бины
    // времени) — эталоны/тесты переякорены (авторизация владельца).

    /// Подшаг: накопить статистику спайков и продвинуть зеркала следов.
    fn tick_note_spikes(&mut self, spikes: &[i32]) {
        let n = self.n;
        let spt = self.steps_per_tick();
        if self.tick_spike_count.len() != n || self.tick_trace_pre_hist.len() != n * spt {
            self.tick_spike_count = vec![0; n];
            self.tick_spike_sum_s = vec![0; n];
            self.tick_trace_pre = vec![0.0; n];
            self.tick_trace_post = vec![0.0; n];
            self.tick_trace_pre_hist = vec![0.0; n * spt];
            self.tick_trace_post_hist = vec![0.0; n * spt];
            self.tick_row = 0;
        }
        let (dt, tau_pre, tau_post) = {
            let c = self.plast.stdp_config();
            (c.dt, c.tau_pre, c.tau_post)
        };
        let d_pre = fast_exp(-dt / tau_pre);
        let d_post = fast_exp(-dt / tau_post);
        let row = self.tick_row;
        for (i, &sp) in spikes.iter().enumerate() {
            let add = if sp > 0 { 1.0 } else { 0.0 };
            // Зеркала следов: порядок ядра (fma: t·d + sp) — 1 ulp-совместимо.
            let tp = &mut self.tick_trace_pre[i];
            *tp = tp.mul_add(d_pre, add);
            let tpo = &mut self.tick_trace_post[i];
            *tpo = tpo.mul_add(d_post, add);
            self.tick_trace_pre_hist[row * n + i] = *tp;
            self.tick_trace_post_hist[row * n + i] = *tpo;
            if sp > 0 {
                self.tick_spike_count[i] += 1;
                self.tick_spike_sum_s[i] += row as u64;
            }
        }
        self.tick_row = (row + 1) % spt.max(1);
        self.tick_active = true;
    }

    /// Конец тика: один проход по всем синапсам — mean-field STDP.
    ///
    /// Для синапса (pre→post): LTP = a_ltp·след(pre, mean-момент спайка
    /// post)·c_post; LTD = a_ltd·след(post, mean-момент спайка pre)·c_pre;
    /// eligibility = (elig + LTP − LTD)·d_e^S. Замороженные (thaw_at) — 0.
    fn tick_plasticity_apply(&mut self, spt: usize) {
        let n = self.n;
        if self.tick_spike_count.len() != n || self.tick_trace_pre_hist.len() != n * spt {
            return;
        }
        let (dt, tau_pre, tau_post, tau_e, a_ltp, a_ltd) = {
            let c = self.plast.stdp_config();
            (c.dt, c.tau_pre, c.tau_post, c.tau_e, c.a_ltp, c.a_ltd)
        };
        let d_e = fast_exp(-dt / tau_e);
        let d_e_tick = d_e.powi(spt as i32);
        let now = self.step_count;
        let mut elig: Vec<f32> = self.plast.eligibility().to_vec();
        if elig.len() != self.pre_of.len() {
            return; // рассинхрон (не должен случаться)
        }
        // Средние моменты спайков за тик (подшаги 0..S-1).
        let mean_s = |count: u32, sum: u64| -> usize {
            if count > 0 {
                ((sum as f32 / count as f32).round() as usize).min(spt - 1)
            } else {
                0
            }
        };
        let mean_of_pre: Vec<usize> = (0..n)
            .map(|i| mean_s(self.tick_spike_count[i], self.tick_spike_sum_s[i]))
            .collect();
        for idx in 0..self.pre_of.len() {
            let pre = self.pre_of[idx];
            let post = self.post_of[idx];
            // Замороженные синапсы: eligibility держим нулевой (окна оттаивания).
            if self.thaw_at.len() > idx && self.thaw_at[idx] > now {
                elig[idx] = 0.0;
                continue;
            }
            let cpre = self.tick_spike_count[pre];
            let cpost = self.tick_spike_count[post];
            let mut acc = elig[idx] * d_e_tick;
            if cpost > 0 {
                // LTP: пост стрелял — след ПРЕ на его средний момент.
                let m_post = mean_s(cpost, self.tick_spike_sum_s[post]);
                let tpre_at = self.tick_trace_pre_hist[m_post * n + pre];
                acc += a_ltp * tpre_at * cpost as f32;
            }
            if cpre > 0 {
                let m_pre = mean_of_pre[pre];
                let tpost_at = self.tick_trace_post_hist[m_pre * n + post];
                acc -= a_ltd * tpost_at * cpre as f32;
            }
            elig[idx] = acc;
        }
        self.plast.set_eligibility(&elig);
        // Сброс счётчиков тика (зеркала следов НЕ сбрасываются — непрерывны).
        for v in self.tick_spike_count.iter_mut() {
            *v = 0;
        }
        for v in self.tick_spike_sum_s.iter_mut() {
            *v = 0;
        }
    }

    /// Пост-обработка подшага БЕЗ пластичности (GPU-путь: eligibility уже на
    /// GPU в `step_many_learn`/`run_step_learn`): только счётчик, гомеостаз
    /// и отложенные награды — общий код для `step_into`/`step_batch` GPU.
    fn post_substep_tick(&mut self) {
        self.step_count += 1;

        // Граница тика: набралось S подшагов — ОДИН проход тик-пластичности
        // (mean-field STDP, см. `tick_plasticity_apply`). Только CPU-путь:
        // пул/GPU ведут eligibility своими механизмами (tick_active).
        let spt = self.steps_per_tick();
        if self.tick_active && self.step_count.is_multiple_of(spt) {
            self.tick_plasticity_apply(spt);
        }

        // Мягкий гомеостаз по расписанию модели (см. `step`).
        let period = self.params.homeo_every_ticks.max(1) * self.steps_per_tick();
        if self.params.homeo_every_ticks > 0 && self.step_count.is_multiple_of(period) {
            self.normalize();
        }

        // Отложенные награды: двойной проход без аллокации в горячем случае
        // (когда `pending_reward` пуст — частый случай между событиями).
        if self.pending_reward.is_empty() {
            return;
        }
        let now = self.step_count;
        let mut keep = Vec::new();
        let mut due_sum = 0.0f32;
        for &(at, d) in &self.pending_reward {
            if now >= at {
                due_sum += d;
            } else {
                keep.push((at, d));
            }
        }
        self.pending_reward = keep;
        if due_sum != 0.0 {
            self.apply_dopamine(due_sum);
            self.normalize();
        }
    }

    /// Обнулить eligibility замороженных синапсов (учитывая окна оттаивания).
    fn zero_frozen_eligibility(&mut self) {
        let now = self.step_count;
        self.frozen_keep.clear();
        self.frozen_keep.reserve(self.thaw_at.len());
        self.frozen_keep
            .extend(self.thaw_at.iter().map(|&thaw| now >= thaw));
        if self.gpu_learn {
            if let Dynamics::Gpu(g) = &self.dynamics {
                let _ = g.restrict_gpu(&self.frozen_keep);
            }
        } else {
            self.plast.restrict_eligibility(&self.frozen_keep);
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

    /// Запланировать дофамин с внешним гейтом [0..1] (метаболический бюджет:
    /// стресс глушит фиксацию). Гейт применяется в момент доставки.
    pub fn schedule_reward_gated(&mut self, dopamine: f32, gate: f32) {
        let at = self.step_count + self.steps_per_tick() * self.params.reward_delay_ticks;
        self.pending_reward
            .push((at, dopamine * gate.clamp(0.0, 1.0)));
    }

    /// Задержка награды в тиках мира + 1 (длительность «жевания»).
    pub fn reward_delay_ticks(&self) -> usize {
        self.params.reward_delay_ticks + 1
    }

    /// Применить дофамин немедленно с внешним модулятором [0..1]
    /// (метаболический бюджет тела: стресс глушит обучение).
    pub fn reward_now_gated(&mut self, dopamine: f32, gate: f32) {
        self.apply_dopamine(dopamine * gate.clamp(0.0, 1.0));
        self.normalize();
    }

    /// Применить дофамин немедленно (протоколы с фиксацией веса после пробы).
    pub fn reward_now(&mut self, dopamine: f32) {
        self.apply_dopamine(dopamine);
        self.normalize();
    }

    /// Возвращает текущий счётчик шагов мозга.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Возвращает текущие параметры пластичности.
    pub fn plasticity_params(&self) -> &PlasticityParams {
        &self.params
    }

    /// Применить дофамин немедленно с явным потолком клэмпа весов
    /// (используется прожиганием инстинктов при рождении — см.
    /// `Instinct::burn_into_with_cap`).
    pub fn reward_now_with_cap(&mut self, dopamine: f32, w_cap: f32) {
        self.apply_dopamine_capped(dopamine, w_cap);
        self.normalize();
    }

    /// Обнулить eligibility-следы всех синапсов: окно допустимости закрывается
    /// принудительно (иначе след не-награждённой пробы доживает до следующей
    /// награды и обобщается на неё).
    pub fn clear_eligibility(&mut self) {
        if self.gpu_learn {
            if let Dynamics::Gpu(g) = &self.dynamics {
                let _ = g.clear_gpu();
            }
        } else {
            self.plast.clear_eligibility();
        }
    }

    /// Оставить eligibility только на синапсах, чьи пре И пост входят в
    /// `stimulated` (нейроны, стимулируемые инстинктом). Используется
    /// прожиганием: инстинкт фиксирует СВОЮ дугу целиком, а побочные пути
    /// (например тормозные I→M, возбуждённые ко-активацией) награду не
    /// получают — иначе врождённый рефлекс усиливал бы заодно торможение
    /// и глушил бы собственный драйв.
    pub fn restrict_eligibility(&mut self, stimulated: &[bool]) {
        let keep: Vec<bool> = self
            .pre_of
            .iter()
            .zip(self.post_of.iter())
            .map(|(&pre, &post)| {
                stimulated.get(pre).copied().unwrap_or(false)
                    && stimulated.get(post).copied().unwrap_or(false)
            })
            .collect();
        if self.gpu_learn {
            if let Dynamics::Gpu(g) = &self.dynamics {
                let _ = g.restrict_gpu(&keep);
            }
        } else {
            self.plast.restrict_eligibility(&keep);
        }
    }

    /// Дофаминовая фиксация Δw = η·D·e — канонический SIMD-кернел движка
    /// (`dopamine_weight_update`): клэмп применяется только к синапсам с
    /// e ≠ 0, замороженные (e обнулён `zero_frozen_eligibility`) и врождённые
    /// хвосты > w_max сохраняются бит-в-бит.
    fn apply_dopamine(&mut self, dopamine: f32) {
        let w_cap = self.params.w_max * self.syn_scale;
        self.apply_dopamine_capped(dopamine, w_cap);
    }

    /// То же, но с явным потолком клэмпа: рождение/сон могут позволить
    /// прожигание ВЫШЕ рабочего `w_max` (у grown-коннектома хвосты весов
    /// 50+, а рабочий потолок пластичности 6 — обычный дофамин их бы
    /// срезал, см. `burn_into_with_cap`).
    fn apply_dopamine_capped(&mut self, dopamine: f32, w_cap: f32) {
        self.zero_frozen_eligibility();
        if self.gpu_learn {
            // GPU: reward-кернел применяет Δw = lr·dopamine·elig (клэмп
            // [w_min, w_cap]); затем веса скачиваются в зеркало (источник
            // правды для гомеостаза/наследственности).
            if let Dynamics::Gpu(g) = &mut self.dynamics {
                if g.reward_gpu(dopamine, w_cap).is_ok() {
                    // Зеркало plast тоже обновляем — Болдуин/learned_gain
                    // читают его.
                    let n = self.plast.weights().len().min(g.values().len());
                    self.plast.weights_mut()[..n].copy_from_slice(&g.values()[..n]);
                    return;
                }
            }
            // сбой GPU — фолбэк на CPU-пластичность ниже
        }
        // Награда живёт только внутри токена обучения (сессия = событие
        // дофамина); ленивый кернел сворачивает eligibility в этот момент.
        {
            let mut tok = self.plast.begin_learning();
            tok.reward_capped(dopamine, w_cap);
            tok.end();
        }
        self.sync_plast_to_runner();
    }

    /// Скопировать веса зеркала пластичности в runner (источник правды для
    /// спайков) — после каждого дофаминового обновления.
    fn sync_plast_to_runner(&mut self) {
        let pw = self.plast.weights();
        let rv = self.dynamics.values_mut();
        let n = rv.len().min(pw.len());
        rv[..n].copy_from_slice(&pw[..n]);
    }

    /// Обратный sync (динамика → зеркало): после гомеостаза, который пишет
    /// веса напрямую, чтобы следующий дофамин не затёр выравнивание.
    fn sync_runner_to_plast(&mut self) {
        let rv = self.dynamics.values();
        let pw = self.plast.weights_mut();
        let n = pw.len().min(rv.len());
        pw[..n].copy_from_slice(&rv[..n]);
    }

    /// Мягкий гомеостаз: суммы входящих ВОЗБУЖДАЮЩИХ весов каждого нейрона
    /// медленно тянутся к стартовым (полная нормализация отменяла бы дофамин).
    /// Тормозные пресинапсы не скейлятся (знак в сумме ломал бы масштаб).
    /// Навсегда-замороженные (thaw=usize::MAX) не трогаются: врождённые дуги
    /// неподвижны. Критическое окно тоже защищено (now < thaw).
    fn normalize(&mut self) {
        // Переиспользуемый буфер: заполняем нулями разом, затем инкрементально.
        self.norm_sums.clear();
        self.norm_sums.resize(self.n, 0.0f32);
        let sums = &mut self.norm_sums;
        for idx in 0..self.pre_of.len() {
            if self.inh_syn[idx]
                || self.params.homeo_exclude_pre.contains(&self.region_of[self.pre_of[idx]])
            {
                continue;
            }
            sums[self.post_of[idx]] += self.dynamics.values()[idx];
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
            if self.params.homeo_exclude_pre.contains(&self.region_of[self.pre_of[idx]]) {
                continue;
            }
            let post = self.post_of[idx];
            let target = self.target_in_sum[post];
            if sums[post] > 1e-6 && target > 1e-6 {
                let scale = 1.0 + rate * (target / sums[post] - 1.0);
                if scale.is_finite() {
                    self.dynamics.values_mut()[idx] *= scale;
                }
            }
        }
        self.sync_runner_to_plast();
    }

    /// Гиперполяризовать выбранные регионы на N шагов (сброс памяти,
    /// нейромодуляторное торможение).
    pub fn hyperpolarize(&mut self, regions: &[usize], steps: usize, current: f32) {
        let ext: Vec<f32> = self
            .region_of
            .iter()
            .map(|r| if regions.contains(r) { current } else { 0.0 })
            .collect();
        let mut spikes = Vec::with_capacity(self.n);
        for _ in 0..steps {
            self.step_into(&ext, &mut spikes);
        }
    }

    /// Число синапсов CSR (для диагностики/сводок).
    pub fn n_syn(&self) -> usize {
        self.pre_of.len()
    }

    /// Копия текущих значений CSR (сохранение существа).
    /// Делегирует в канонический снапшот движка.
    pub fn csr_values(&self) -> Vec<f32> {
        self.dynamics.export_values()
    }

    /// Нативный бинарный дамп (TAAS) — открывается ConnScope.
    pub fn save_baked<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.dynamics.save_baked(w)
    }

    /// Триплеты CSR (pre, post, вес) в каноническом порядке — для синка
    /// графа при тропической перестройке (`plasticity::sync_weights_from_brain`).
    pub fn csr_triplets(&self) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        (
            self.pre_of.clone(),
            self.post_of.clone(),
            self.dynamics.values().to_vec(),
        )
    }

    /// Сон-консолидация: цель гомеостаза подтягивается к текущим суммам
    /// входов — выученное за бодрствование закрепляется как новая «норма»
    /// и больше не вымывается мягкой нормализацией. Вызывать в фазах покоя.
    pub fn sleep_consolidate(&mut self, alpha: f32) {
        let mut sums = vec![0.0f32; self.n];
        for idx in 0..self.pre_of.len() {
            if !self.inh_syn[idx]
                && !self
                    .params
                    .homeo_exclude_pre
                    .contains(&self.region_of[self.pre_of[idx]])
            {
                sums[self.post_of[idx]] += self.dynamics.values()[idx];
            }
        }
        for (post, sum) in sums.iter().enumerate() {
            if *sum > 1e-6 {
                self.target_in_sum[post] += alpha * (*sum - self.target_in_sum[post]);
            }
        }
    }

    /// Снять долговременную копию весов (начало сна): текущее состояние
    /// становится эталоном, к которому сон будет возвращать дрейф.
    pub fn memory_snapshot(&mut self) {
        self.memory_lt = self.dynamics.values().to_vec();
    }

    /// Восстановление во сне: веса подтягиваются к долговременной копии —
    /// синаптический дрейф дня стирается, выученный рисунок восстанавливается.
    pub fn memory_recall(&mut self, alpha: f32) {
        let n = self.memory_lt.len().min(self.dynamics.values().len());
        for i in 0..n {
            let lt = self.memory_lt[i];
            let w = self.dynamics.values()[i] + alpha * (lt - self.dynamics.values()[i]);
            self.dynamics.values_mut()[i] = w;
            self.plast.weights_mut()[i] = w;
        }
    }

    /// Сон-консолидация весов: сильные выученные дельты (относительно
    /// снимка рождения `memory_lt`) закрепляются в долговременную память
    /// и слегка усиливаются, слабые (шум/дрейф гомеостаза) отсекаются —
    /// вес тянется обратно к снимку. Эталон Болдуина `memory_lt` НЕ
    /// трогается: выученное за жизнь продолжает накапливаться между
    /// рождениями, и `learned_gain` видит весь прирост.
    pub fn consolidate_sleep(&mut self, strong_min: f32, reinforce: f32, prune: f32) {
        let n = self.memory_lt.len().min(self.dynamics.values().len());
        if n == 0 {
            return;
        }
        let lt = self.memory_lt.clone();
        let vals = self.dynamics.values().to_vec();
        let mut out = vals.clone();
        for i in 0..n {
            let d = vals[i] - lt[i];
            if d.abs() >= strong_min {
                // Сильное выученное — в долговременную: усилить дельту.
                out[i] = lt[i] + d * (1.0 + reinforce);
            } else {
                // Слабый шум — обрезать: вернуть к снимку рождения.
                out[i] = lt[i] + d * (1.0 - prune);
            }
        }
        let _ = self.set_csr_values(&out);
    }

    /// Среднеквадратичное расхождение весов с долговременной копией
    /// (диагностика дрейфа).
    pub fn memory_drift_rms(&self) -> f32 {
        let n = self.memory_lt.len().min(self.dynamics.values().len());
        if n == 0 {
            return 0.0;
        }
        let s: f32 = (0..n)
            .map(|i| {
                let d = self.dynamics.values()[i] - self.memory_lt[i];
                d * d
            })
            .sum();
        (s / n as f32).sqrt()
    }

    /// Суммарный выученный прирост весов пары регионов относительно снимка
    /// рождения (`memory_lt`): Σ(текущие − эталон) по синапсам pre→post.
    /// Положительный = путь, усиленный обучением за жизнь (эффект Болдуина:
    /// его можно свернуть в инстинкт-ген потомку); отрицательный = выученное
    /// ослабление. Эталон снимается при сборке и при рождении
    /// (см. `memory_snapshot`), так что «выученным» считается только то,
    /// что изменилось после.
    pub fn learned_gain(&self, pre_region: usize, post_region: usize) -> f32 {
        let (rb, re) = self.region_bounds(pre_region);
        let (pb, pe) = self.region_bounds(post_region);
        let values = self.dynamics.values();
        let lt = &self.memory_lt;
        let n = values.len().min(lt.len());
        let mut gain = 0.0f32;
        for idx in 0..self.pre_of.len() {
            if idx >= n {
                break;
            }
            let pre = self.pre_of[idx];
            let post = self.post_of[idx];
            if (rb..re).contains(&pre) && (pb..pe).contains(&post) {
                gain += values[idx] - lt[idx];
            }
        }
        gain
    }

    /// Перезаписать значения CSR (наследование выученных весов). Длины
    /// должны совпадать с построенным CSR — иначе движок отвергнет снапшот.
    pub fn set_csr_values(&mut self, vals: &[f32]) -> Result<(), String> {
        self.dynamics.import_values(vals)?;
        let n = self.plast.weights().len().min(vals.len());
        self.plast.weights_mut()[..n].copy_from_slice(&vals[..n]);
        Ok(())
    }

    /// Мин/макс всех синаптических весов (интроспекция/отладка).
    pub fn values_minmax(&self) -> (f32, f32) {
        self.dynamics
            .values()
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
                sum += self.dynamics.values()[idx];
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

/// Построить CSR-коннектом из grown-проводки × гейны анатомии. Общий код
/// сборки мозга: `Brain::with_model` (CPU/GPU/HAAP) и батч-пул арены
/// (`brain::gpu_pool::BrainPool::register`). Рёбра с нулевым гейном
/// выпадают; HAAP дополнительно масштабирует веса (`syn_scale`) и клипует
/// хвост (`syn_cap`) — LIF-путь идёт без масштаба, бит-в-бит как раньше.
pub(crate) fn build_connectome(
    grown: &Grown,
    gains: &[Vec<f32>],
    haap: bool,
    syn_scale: f32,
    syn_cap: f32,
) -> ConnectomeData {
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
    for (i, list) in per.iter().enumerate() {
        row_ptr[i] = ci.len() as i32;
        for &(post, w) in list {
            ci.push(post as i32);
            // Масштаб + клип хвоста: один синапс не может доминировать.
            vals.push(if haap {
                (w * syn_scale).min(syn_cap)
            } else {
                w
            });
        }
    }
    row_ptr[n_total] = ci.len() as i32;
    ConnectomeData::new(row_ptr, ci, vals, grown.nt.clone(), vec![])
}

/// Построить plast-зеркало и маски из готового CSR: `pre_of`/`post_of`/
/// `inh_syn`/`thaw_at`/`target_in_sum` + `PlastBrain` (ленивый дофаминовый
/// STDP). Общий код CPU/GPU/пула — смена динамики не меняет пластичность.
/// `syn_scale` — рабочий масштаб весов модели (HAAP < 1, LIF/пул = 1):
/// потолок `w_max` в конфиге plast домножается на него.
fn plast_and_masks(
    conn: &ConnectomeData,
    grown: &Grown,
    params: &PlasticityParams,
    eff_lr: f32,
    dt_for_model: f32,
    syn_scale: f32,
) -> (PlastBrain, Vec<usize>, Vec<usize>, Vec<bool>, Vec<usize>, Vec<f32>) {
    let n_neurons = conn.n_neurons;
    let n_syn = conn.values.len();
    let rp = &conn.row_ptr;
    let cc = &conn.col_indices;
    let vv = &conn.values;
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
        if !inh_syn[idx]
            && !params
                .homeo_exclude_pre
                .contains(&grown.region_of[pre_of[idx]])
        {
            target_in_sum[post_of[idx]] += vv[idx];
        }
    }
    let pre_u32: Vec<u32> = pre_of.iter().map(|&p| p as u32).collect();
    let post_u32: Vec<u32> = post_of.iter().map(|&p| p as u32).collect();
    let plast = PlastBrain::new(
        n_neurons,
        &pre_u32,
        &post_u32,
        vv.clone(),
        StdpConfig {
            dt: dt_for_model,
            tau_pre: 20.0,
            tau_post: 20.0,
            tau_e: 1000.0,
            a_ltp: params.a_ltp,
            a_ltd: params.a_ltd,
            lr: eff_lr,
            w_min: 0.0,
            // Рабочий потолок: LIF — w_max, HAAP — w_max·syn_scale.
            w_max: params.w_max * syn_scale,
        },
    );
    (plast, pre_of, post_of, inh_syn, thaw_at, target_in_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_grown() -> Grown {
        let edges = vec![
            (0, 3, 1.5),
            (1, 4, 1.2),
            (2, 5, 1.8),
            (0, 4, 0.9),
            (1, 3, 0.7),
        ];
        Grown {
            edges,
            region_of: vec![0, 0, 0, 1, 1, 1],
            nt: vec![1.0; 6],
            growth_s: 0.0,
            rule_genes: Vec::new(),
            instincts: Vec::new(),
        }
    }

    fn homeo_params() -> PlasticityParams {
        PlasticityParams {
            homeo_every_ticks: 1,
            ..Default::default()
        }
    }

    #[test]
    fn sleep_consolidation_anchors_learning() {
        let grown = toy_grown();
        let gains = gains_from(2, &[(0, 1, 1.0)]);
        let mean_w = |b: &Brain| {
            let (_, _, v) = b.csr_triplets();
            v.iter().sum::<f32>() / v.len() as f32
        };

        // «Обучение»: удвоили все веса.
        let mut learned = Brain::new(&grown, gains.clone(), homeo_params());
        let m0 = mean_w(&learned);
        let (_, _, vals) = learned.csr_triplets();
        let doubled: Vec<f32> = vals.iter().map(|&v| v * 2.0).collect();
        learned.set_csr_values(&doubled).unwrap();

        // A: закрепил во сне — гомеостаз держит новую норму.
        let mut sleeper = Brain::new(&grown, gains.clone(), homeo_params());
        sleeper.set_csr_values(&doubled).unwrap();
        sleeper.sleep_consolidate(1.0);
        let zero = vec![0.0f32; 6];
        for _ in 0..160 {
            sleeper.step(&zero);
        }
        let ma = mean_w(&sleeper);

        // B: не спал — мягкая нормализация тянет обратно к старой норме.
        let mut awake = Brain::new(&grown, gains, homeo_params());
        awake.set_csr_values(&doubled).unwrap();
        for _ in 0..160 {
            awake.step(&zero);
        }
        let mb = mean_w(&awake);

        assert!(
            ma > m0 * 1.6,
            "консолидированный должен держать выученное: m0={m0} ma={ma}"
        );
        assert!(
            mb < ma - m0 * 0.2,
            "без сна выученное вымывается: ma={ma} mb={mb}"
        );
    }

    /// Сон-консолидация весов: сильное выученное закрепляется и усиливается,
    /// слабый шум отсекается. Эталон рождения (`memory_lt`) не трогается.
    #[test]
    fn sleep_consolidation_reinforces_strong_prunes_weak() {
        let grown = toy_grown();
        let gains = gains_from(2, &[(0, 1, 1.0)]);
        let mut b = Brain::new(&grown, gains, homeo_params());
        let lt = b.dynamics.values().to_vec(); // снимок рождения

        // 0-й синапс сильно выучен (+2), остальные — шум (+0.05).
        let mut vals = lt.clone();
        vals[0] += 2.0;
        for v in vals.iter_mut().skip(1) {
            *v += 0.05;
        }
        b.set_csr_values(&vals).unwrap();

        b.consolidate_sleep(0.5, 0.2, 0.8);
        let out = b.dynamics.values().to_vec();

        let d0 = out[0] - lt[0];
        assert!(
            d0 > 2.0 + 0.2,
            "сильное выученное усиливается при закреплении: d0={d0}"
        );
        for i in 1..lt.len() {
            let d = (out[i] - lt[i]).abs();
            assert!(d < 0.02, "слабый шум отсекается во сне: синапс {i} d={d}");
        }
        // Эталон Болдуина не сдвинут: выученное продолжает накапливаться.
        assert_eq!(b.memory_lt, lt);
    }
}
