//! GPU-динамика мозга на pillars-wgpu (alpha-lif LIF).
//!
//! `GpuDynamics` заменяет CPU-раннер (`pillars_connectome::BrainRunner`)
//! внутри `engine::neurons::Brain`: мембранная динамика считается на GPU
//! (wgpu). Пластичность — ДВА режима:
//! - без `enable_learning` — как раньше: спайки скачиваются, пластичность/
//!   гомеостаз/сон/Болдуин живут на CPU поверх зеркала весов (`conn.values`);
//! - с `enable_learning` (по умолчанию для GPU-существ) — ОБУЧЕНИЕ ЦЕЛИКОМ
//!   НА GPU: dopamine-трассы/eligibility считаются в `step_many_learn` тем же
//!   submit, награды — через `reward_gpu` (кернел `reward_kernel`), заморозка/
//!   прожигание — `restrict_gpu`/`clear_gpu`. Ридбек за тик — ТОЛЬКО моторные
//!   суммы (8 байт), спайки не скачиваются вовсе. Семантика бит-в-бит равна
//!   CPU-plast (доказано gpu_probe/gpu_substep: wmaxdiff=0, |Δ|=0.0%).
//!
//! Семантика подшага — та же, что у `run_lif_alpha`: старт `v=v_rest,
//! g=0, refr=refr_steps` (вне рефрактерности), строгий порог, сброс `v` и
//! `g` на спайке, аксональная задержка через ring-буфер.
//!
//! Один `GpuDynamics` = одна особь (один pop сети). Бачевание популяции в
//! одну сеть с offset-окнами — следующий шаг (`WgpuNetwork` уже умеет
//! несколько pops; здесь коннектом не батчится, а зеркало весов пушится
//! в GPU перед каждым шагом).

use pillars::zoo::neurons::ALPHA_LIF_WGSL_SOURCE;
use pillars_connectome::{ConnectomeData, NeuronConfig};
use pillars_wgpu::{WgpuContext, WgpuNetwork, WgpuPop};

/// AlphaLif-WGSL с ЯВНЫМ fma в узлах v/g: компилятор шейдера волен
/// сворачивать `a*b+c` в fma, а rust без fast-math — нет, и это разводит
/// CPU/GPU на 1 ulp (ложные спайки у порога, расхождение обучения).
/// CPU-раннер (pillars alpha_lif::step, run_lif_alpha-scatter) считает те же
/// узлы через `mul_add` — обе стороны гарантированно fma (IEEE-определён).
/// Один инстанс на процесс (кэш): шейдер одинаков для всех особей модели.
/// Публичный внутри крейта: батч-пул арены (gpu_pool.rs) использует тот же
/// шейдер для своей сети.
pub(crate) static ALPHA_LIF_FMA_WGSL: std::sync::OnceLock<String> = std::sync::OnceLock::new();

pub(crate) fn alpha_lif_fma_wgsl() -> &'static str {
    ALPHA_LIF_FMA_WGSL
        .get_or_init(|| {
            let mut wgsl = ALPHA_LIF_WGSL_SOURCE.to_string();
            wgsl = wgsl.replace(
                "var v_new: f32 = (v + ((*P)[1] * integrand));",
                "var v_new: f32 = fma((*P)[1], integrand, v);",
            );
            wgsl.replace(
                "var g_new: f32 = ((g_old * (*P)[6]) + delayed);",
                "var g_new: f32 = fma(g_old, (*P)[6], delayed);",
            )
        })
        .as_str()
}

/// Один спайковый подшаг динамики на GPU.
pub struct GpuDynamics {
    net: WgpuNetwork,
    /// CSR-коннектом: `values` — каноническое зеркало весов (CPU-источник
    /// правды; перед каждым шагом пушится в GPU через `set_weights`).
    conn: ConnectomeData,
    /// Выходной буфер спайков с GPU (u32) → i32 для `Brain`.
    spikes_u32: Vec<u32>,
    spikes: Vec<i32>,
    /// Индексы всех нейронов для `inject_current` (0..n).
    indices: Vec<u32>,
    /// Зеркало менялось с прошлого аплоада? set_weights делается только по
    /// dirty — награды/гомеостаз живут на границах тиков, внутри тика веса
    /// константны, так что один аплоад на тик (не на подшаг).
    dirty: bool,
    /// Параметры dopamine-обучения (GPU-путь «обучение целиком»): константы
    /// распада трасс/eligibility как в pillars `brain.rs` (fast_exp).
    learn: Option<LearnGpu>,
}

/// Параметры GPU-обучения: константы распада и амплитуды STDP (те же, что
/// CPU plast `StdpConfig` — см. `PlasticityParams`). При `None` GPU-путь
/// работает как раньше: пластичность на CPU по скачанным спайкам.
#[derive(Clone, Copy)]
pub struct LearnGpu {
    pub d_pre: f32,
    pub d_post: f32,
    pub d_e: f32,
    pub a_ltp: f32,
    pub a_ltd: f32,
    pub lr: f32,
    pub w_min: f32,
    pub w_cap: f32,
}

impl GpuDynamics {
    /// Собрать GPU-динамику из коннектома и конфига модели. Только
    /// `NeuronConfig::LifAlpha` — остальные модели (HAAP и пр.) остаются
    /// на CPU-раннере. Ошибка = нет адаптера/драйвера/поддержки WGSL —
    /// вызывающий делает CPU-фолбэк.
    pub fn try_new(conn: ConnectomeData, config: &NeuronConfig) -> Result<Self, String> {
        let (dt, tau_ref, tau_m, v_rest, v_thresh, v_reset, tau_syn, t_delay) = match config {
            NeuronConfig::LifAlpha {
                dt_ms,
                tau_ref,
                tau_m,
                v_rest,
                v_thresh,
                v_reset,
                tau_syn,
                t_delay,
            } => (
                *dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, *tau_syn, *t_delay,
            ),
            _ => return Err("GpuDynamics: поддерживается только LifAlpha".into()),
        };
        let n = conn.n_neurons;
        let n_syn = conn.n_synapses;
        if n == 0 || n_syn == 0 {
            return Err("GpuDynamics: пустой коннектом".into());
        }

        // AlphaLif: 3 состояния (v, g, refr_cnt) и 7 параметров, SoA со stride,
        // ПАДДИТСЯ до workgroup-размера (64) — как требует сеть: колонки
        // читаются как states[k*stride + i]. Инициализация как в run_lif_alpha:
        // v=v_rest, g=0, refr=refr_steps (старт вне рефрактерности).
        let stride = n.div_ceil(64) * 64;
        let refr_steps = (tau_ref / dt).round().max(1.0);
        let time_factor = dt / tau_m;
        let syn_decay = 1.0 - dt / tau_syn;
        let mut states = vec![0.0f32; 3 * stride];
        for i in 0..n {
            states[i] = v_rest;
            states[stride + i] = 0.0;
            states[2 * stride + i] = refr_steps;
        }
        let mut params = vec![0.0f32; 7 * stride];
        for i in 0..n {
            params[i] = dt;
            params[stride + i] = time_factor;
            params[2 * stride + i] = v_rest;
            params[3 * stride + i] = v_thresh;
            params[4 * stride + i] = v_reset;
            params[5 * stride + i] = refr_steps;
            params[6 * stride + i] = syn_decay;
        }

        let pop = WgpuPop {
            wgsl: alpha_lif_fma_wgsl(),
            entry: "alphalif_kernel_wgsl",
            n,
            ns: 3,
            np: 7,
            states,
            params,
        };
        let row_ptr: Vec<u32> = conn.row_ptr.iter().map(|&x| x as u32).collect();
        let col_idx: Vec<u32> = conn.col_indices.iter().map(|&x| x as u32).collect();
        // Аксональная задержка: CPU-раннер скеттерит спайки ПРОШЛОГО шага
        // (alpha_fired) в только что прочитанный слот, который перечитается
        // через size = steps_delay + 1 шагов → эффективная задержка
        // spike→arrival = 1 + size = steps_delay + 2. GPU-сеть скеттерит
        // ТОЖЕ спайки прошлого шага (spikes_prev) → d = steps_delay + 1:
        // fired на шаге t → записан на шаге t+1 → прочитан на t+1+d.
        let steps_delay = (t_delay / dt).round().max(0.0) as u32;
        // DEBUG-подбор: DIOMA_GPU_DELAY переопределяет задержку.
        let d = std::env::var("DIOMA_GPU_DELAY")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(steps_delay + 1);
        let delays = vec![d.min(31) as u8; n_syn];
        let ctx = WgpuContext::new_blocking()?;
        let net = WgpuNetwork::new(ctx, &[pop], &row_ptr, &col_idx, &conn.values, &delays, dt)?;
        // Знаки торможения пресинапсов (nt_types): +1 возбуждающий, −1
        // тормозной — как `nt_sign` в run_lif_alpha (иначе тормозные пути
        // с положительными весами возбуждают пост-нейрон → судорога).
        let signs: Vec<f32> = (0..n)
            .map(|i| {
                if conn.nt_types.get(i).copied().unwrap_or(1.0) < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect();
        net.set_nt_signs(&signs)?;

        Ok(Self {
            net,
            conn,
            spikes_u32: vec![0; n],
            spikes: Vec::with_capacity(n),
            indices: (0..n as u32).collect(),
            dirty: false,
            learn: None,
        })
    }

    /// Включить GPU-обучение: eligibility живёт на GPU (dopamine.wgsl),
    /// ридбеки за тик — только моторные суммы (8 байт). Параметры — те же,
    /// что CPU plast `StdpConfig` (dt, tau, a_ltp, a_ltd, lr, w_max).
    pub fn enable_learning(&mut self, l: LearnGpu) {
        self.learn = Some(l);
    }

    /// Один спайковый подшаг: пушит зеркало весов в GPU, инжектит внешний
    /// ток всех нейронов, считает шаг, скачивает спайки. Ошибки GPU
    /// (device loss и т.п.) логируются и возвращают пустой кадр — существо
    /// затихает, но не паникует в игровом цикле.
    /// Один спайковый подшаг: пушит зеркало весов в GPU (если менялось),
    /// инжектит внешний ток всех нейронов, считает шаг, скачивает спайки.
    /// Ошибки GPU (device loss и т.п.) логируются и возвращают пустой кадр —
    /// существо затихает, но не паникует в игровом цикле.
    pub fn run_step(&mut self, ext: &[f32]) -> &[i32] {
        debug_assert_eq!(ext.len(), self.conn.n_neurons, "выход мозга != n");
        let ok = self.upload_weights_if_dirty()
            // Внешний ток — СКАЧОК НАПРЯЖЕНИЯ (v += ext), как в run_lif_alpha;
            // ring-инъекция (inject_current) шла бы в проводимость g.
            && self.net.inject_voltage(0, &self.indices, ext).is_ok()
            && self.net.step().is_ok()
            && self.net.download_spikes(&mut self.spikes_u32).is_ok();
        if !ok {
            eprintln!("[dioma] GpuDynamics::run_step — сбой GPU, тихий кадр");
            self.spikes.clear();
            return &self.spikes;
        }
        self.spikes.clear();
        self.spikes
            .extend(self.spikes_u32.iter().map(|&s| s as i32));
        &self.spikes
    }

    /// Пачка `n` подшагов одним батчем: все шаги уходят в очередь БЕЗ ожиданий
    /// (v_jump свёрнут в step, веса — один аплоад по dirty), ридбек спайков —
    /// ОДИН map_async на всю пачку. Внутри тика веса константны (награды и
    /// гомеостаз — на границах тиков), поэтому пачка бит-в-бит равна n
    /// последовательных `run_step`. `out` заполняется n векторами по n_neurons.
    pub fn run_steps_into(&mut self, ext: &[f32], n: usize, out: &mut Vec<Vec<i32>>) {
        debug_assert_eq!(ext.len(), self.conn.n_neurons, "выход мозга != n");
        out.clear();
        if n == 0 {
            return;
        }
        let first = self.net.step_counter();
        let ok = self.upload_weights_if_dirty()
            // Вход тика константен — один v_jump на пачку (step_many диспатчит
            // его на каждом подшаге с теми же аргументами).
            && self.net.inject_voltage(0, &self.indices, ext).is_ok();
        if !ok {
            eprintln!("[dioma] GpuDynamics::run_steps — сбой GPU (weights/v_jump)");
            for _ in 0..n {
                out.push(vec![0i32; self.conn.n_neurons]);
            }
            return;
        }
        if let Err(e) = self.net.step_many(n) {
            eprintln!("[dioma] GpuDynamics::run_steps — сбой GPU: {e}");
            for _ in 0..n {
                out.push(vec![0i32; self.conn.n_neurons]);
            }
            return;
        }
        let mut flat = vec![0u32; n * self.conn.n_neurons];
        if self.net.download_spikes_hist(first, n, &mut flat).is_err() {
            eprintln!("[dioma] GpuDynamics::run_steps — сбой ридбека");
            for _ in 0..n {
                out.push(vec![0i32; self.conn.n_neurons]);
            }
            return;
        }
        let nn = self.conn.n_neurons;
        for k in 0..n {
            let mut v = Vec::with_capacity(nn);
            v.extend(flat[k * nn..(k + 1) * nn].iter().map(|&s| s as i32));
            out.push(v);
        }
    }

    /// Залить веса в GPU только если зеркало менялось с прошлого раза.
    fn upload_weights_if_dirty(&mut self) -> bool {
        if self.dirty {
            if self.net.set_weights(&self.conn.values).is_err() {
                return false;
            }
            self.dirty = false;
        }
        true
    }

    /// Зеркало весов (источник правды для `Brain`).
    pub fn values(&self) -> &[f32] {
        &self.conn.values
    }

    pub fn values_mut(&mut self) -> &mut [f32] {
        self.dirty = true;
        &mut self.conn.values
    }

    pub fn import_values(&mut self, vals: &[f32]) -> Result<(), String> {
        self.conn.import_values(vals)?;
        self.dirty = true;
        Ok(())
    }

    pub fn save_baked<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.conn.save_baked(w)
    }

    /// DEBUG: скачать ring-буфер сети.
    pub fn debug_ring(&self) -> Vec<f32> {
        self.net.debug_ring()
    }

    /// DEBUG: состояния популяции (v/g/refr SoA, stride как в сети).
    pub fn debug_states(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; 3 * self.conn.n_neurons.div_ceil(64) * 64];
        let _ = self.net.download_states(0, &mut out);
        out
    }

    /// DEBUG: скачивает текущие входные токи после копирования на GPU.
    /// Используется только диагностикой, не игровым hot path.
    pub fn debug_i_in(&self) -> Vec<f32> {
        self.net.debug_i_in()
    }

    /// DEBUG: скачивает спайки последнего шага и переводит их в `i32`.
    /// Синхронный readback здесь допустим только потому, что это debug API.
    pub fn debug_spikes(&self) -> Vec<i32> {
        let mut out = vec![0u32; self.conn.n_neurons];
        if self.net.download_spikes(&mut out).is_err() {
            return Vec::new();
        }
        out.iter().map(|&s| s as i32).collect()
    }

    /// DEBUG: вход скеттера для пост-нейрона (src, вес, знак, задержка).
    pub fn debug_scatter_row(&self, post: usize) -> Vec<(u32, f32, f32, u32)> {
        self.net.debug_scatter_row(post).unwrap_or_default()
    }

    /// DEBUG: скачать буфер весов GPU (weights_b).
    pub fn debug_weights(&self) -> Vec<f32> {
        self.net.weights().unwrap_or_default()
    }

    /// DEBUG: текущий slot сети.
    pub fn debug_slot(&self) -> u32 {
        self.net.slot()
    }

    /// DEBUG: eligibility-буфер GPU (сверка с CPU plast пошагово).
    pub fn debug_elig(&self) -> Vec<f32> {
        self.net.debug_elig()
    }

    /// DEBUG: per-neuron трассы GPU (pre, post).
    pub fn debug_traces(&self) -> (Vec<f32>, Vec<f32>) {
        self.net.debug_traces()
    }

    // ─── GPU-обучение целиком ─────────────────────────────────────────────

    /// Батч-тик с обучением на GPU: все `n` подшагов + dopamine-eligibility
    /// одним submit, ридбек — только моторные суммы (2 × u32). Возвращает
    /// (ml, mr) — суммы спайков моторных пулов за тик.
    /// Требует включённого `enable_learning`; иначе возвращает ошибку.
    pub fn run_tick_learn(
        &mut self,
        ext: &[f32],
        n: usize,
        ml_lo: u32,
        ml_hi: u32,
        mr_lo: u32,
        mr_hi: u32,
    ) -> Result<(u32, u32), String> {
        let l = *self
            .learn
            .as_ref()
            .ok_or_else(|| "run_tick_learn: GPU-обучение не включено".to_string())?;
        debug_assert_eq!(ext.len(), self.conn.n_neurons, "выход мозга != n");
        if !self.upload_weights_if_dirty() {
            return Err("run_tick_learn: upload весов".into());
        }
        self.net.inject_voltage(0, &self.indices, ext)?;
        self.net.step_many_learn(
            n, l.d_pre, l.d_post, l.d_e, l.a_ltp, l.a_ltd, ml_lo, ml_hi, mr_lo, mr_hi,
        )?;
        self.net.download_motor_sums()
    }

    /// Один подшаг с eligibility на GPU (step_into-путь: инстинкты,
    /// гиперполяризация). Возвращает спайки шага.
    pub fn run_step_learn(&mut self, ext: &[f32]) -> Result<&[i32], String> {
        let l = *self
            .learn
            .as_ref()
            .ok_or_else(|| "run_step_learn: GPU-обучение не включено".to_string())?;
        if !self.upload_weights_if_dirty() {
            return Err("run_step_learn: upload весов".into());
        }
        self.net.inject_voltage(0, &self.indices, ext)?;
        // Один подшаг + eligibility в одном submit (батч n=1).
        self.net
            .step_many_learn(1, l.d_pre, l.d_post, l.d_e, l.a_ltp, l.a_ltd, 0, 0, 0, 0)?;
        self.spikes.clear();
        self.spikes_u32.clear();
        self.spikes_u32.resize(self.conn.n_neurons, 0);
        self.net.download_spikes(&mut self.spikes_u32)?;
        self.spikes
            .extend(self.spikes_u32.iter().map(|&s| s as i32));
        Ok(&self.spikes)
    }

    /// Дофаминовая награда на GPU: `Δw = lr·dopamine·elig`, клэмп
    /// [w_min, w_cap]. Затем скачивает веса в зеркало (источник правды для
    /// гомеостаза/наследственности). `w_cap` явный — прожигание инстинктов
    /// может превышать рабочий потолок.
    pub fn reward_gpu(&mut self, dopamine: f32, w_cap: f32) -> Result<(), String> {
        let l = self
            .learn
            .as_ref()
            .ok_or_else(|| "reward_gpu: GPU-обучение не включено".to_string())?;
        self.net.reward(l.lr, dopamine, l.w_min, w_cap)?;
        self.sync_weights_from_gpu()
    }

    /// Заморозить eligibility по маске (keep[k]=false → elig[k]=0):
    /// инстинкты/критические окна.
    pub fn restrict_gpu(&self, keep: &[bool]) -> Result<(), String> {
        self.net.restrict_eligibility(keep)
    }

    /// Обнулить eligibility всех синапсов (окно закрыто).
    pub fn clear_gpu(&self) -> Result<(), String> {
        self.net.clear_eligibility()
    }

    /// Скачать веса GPU в зеркало `conn.values` (после награды).
    pub fn sync_weights_from_gpu(&mut self) -> Result<(), String> {
        let w = self.net.weights()?;
        let n = self.conn.values.len().min(w.len());
        self.conn.values[..n].copy_from_slice(&w[..n]);
        Ok(())
    }
}
