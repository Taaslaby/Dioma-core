//! Батч-пул GPU-динамики: ОДНА WgpuNetwork на всю арену.
//!
//! `BrainPool` конкатенирует CSR всех существ в блочно-диагональный коннектом
//! (нейроны существа i сдвинуты на сумму предыдущих размеров, синапсы — на
//! сумму предыдущих синапсов), гонит ВСЕХ существ одним `step_many` за тик и
//! скачивает историю спайков одним ридбеком. Пластичность каждого существа
//! остаётся на CPU (режим `GpuDynamics` без `enable_learning`): это
//! минимальная семантика игры — per-creature `reward_now_gated`/eligibility-
//! restriction живут в `Brain::plast` поверх зеркала весов, а пул только
//! перезаливает dirty-зеркала в GPU перед шагом.
//!
//! Почему не per-block dopamine на GPU: `dopamine.wgsl` pillars-wgpu
//! применяет награду к ВСЕМ синапсам одним скаляром, а `restrict_eligibility`
//! ОБНУЛЯЕТ eligibility вне маски — per-creature награда на общей сети
//! уничтожила бы следы соседей. Без изменений SiMuse (запрещены) честный
//! per-block reward недостижим — пластичность на CPU.
//!
//! Ограничения (документированные отклонения от CPU-пути):
//! - спавн существа пересобирает сеть (новый device: буферы WgpuNetwork
//!   фиксированы при сборке) — мембраны ВСЕХ существ сбрасываются к покою;
//! - спящее существо не замораживается, как на CPU, а получает нулевой ток —
//!   его мембрана оседает к v_rest (следы пластичности при этом не идут);
//! - инстинкты/прожигание в пуле не поддерживаются (одиночные шаги
//!   недостижимы; у дефолтного генома инстинктов нет).
//!
//! Детерминизм: одна машина + фиксированные сиды → бит-в-бит одинаковые
//! трассы (порядок скеттера per-post тот же, что у CPU; GPU-путь может
//! отличаться от CPU в 1 ulp — это допускается, run-vs-run — нет).

use crate::brain::gpu::alpha_lif_fma_wgsl;
use pillars_connectome::{ConnectomeData, NeuronConfig};
use pillars_wgpu::{WgpuContext, WgpuNetwork, WgpuPop};

/// Смещения существа в общей сети пула (см. `Brain::new_pooled`).
#[derive(Clone, Copy, Debug)]
pub struct PoolSlot {
    /// Смещение нейронов существа в общей сети.
    pub neuron_off: usize,
    /// Число нейронов существа.
    pub n: usize,
    /// Смещение синапсов существа в общем CSR.
    pub syn_off: usize,
    /// Число синапсов существа.
    pub n_syn: usize,
}

/// Параметры LIF-модели пула: те же, что `Brain::with_model` для LifAlpha
/// (dt 0.5 мс, tau_ref 2.2, tau_m 20, v_rest/thresh/reset −52/−45/−52,
/// tau_syn 3, t_delay 1.8). Пул — только LIF (HAAP остаётся на CPU).
fn pool_config() -> NeuronConfig {
    NeuronConfig::LifAlpha {
        dt_ms: 0.5,
        tau_ref: 2.2,
        tau_m: 20.0,
        v_rest: -52.0,
        v_thresh: -45.0,
        v_reset: -52.0,
        tau_syn: 3.0,
        t_delay: 1.8,
    }
}

/// Максимум подшагов на ОДИН `step_many`/`step_many_learn`/ридбек истории.
/// ИСПРАВЛЕНИЕ (perf-охота 42f30ed+): pillars-wgpu молча зажимает пачку до
/// `MAX_HIST = 64` (кольцо истории спайков), а `download_spikes_hist` с
/// count > 64 возвращает ошибку. `BrainPool::step` звал их с spt = 100
/// (BRAIN_STEPS_PER_TICK): тик пула молча жил 64/100 подшагами, затем вечно
/// падал на ридбеке («сбой ридбека, тихий тик» → все мембраны нули). Тик
/// режется на куски ≤ 64: порядок подшагов тот же, v_jump переинжектится
/// перед каждым куском (внутри пачки он прикладывается на КАЖДОМ подшаге —
/// семантика входа не меняется), история скачивается до перезаписи кольца.
pub(crate) const GPU_MAX_STEPS_PER_SUBMIT: usize = 64;

pub struct BrainPool {
    /// Готовый device для первой сборки сети (`try_new` поднимает заранее,
    /// `new` — лениво). Потребляется `ensure_built`; пересборка после спавна
    /// создаёт новый (буферы WgpuNetwork фиксированы при сборке).
    ctx: Option<WgpuContext>,
    net: Option<WgpuNetwork>,
    slots: Vec<PoolSlot>,
    // Блочно-диагональные массивы (источник для пересборки сети при спавне).
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    /// Полное зеркало весов (копии dirty-блоков существ перед шагом).
    values: Vec<f32>,
    delays: Vec<u8>,
    /// Знаки торможения пресинапсов (per neuron, как `nt_sign`).
    nt: Vec<f32>,
    n_neurons: usize,
    /// Сеть менялась с прошлого шага (спавн) — пересобрать перед шагом.
    needs_rebuild: bool,
    /// Сбой GPU — тихо замолчать, не паниковать в игровом цикле.
    failed: bool,
    /// Аксональная задержка (шаги): steps_delay + 1, как в `GpuDynamics`
    /// (DIOMA_GPU_DELAY переопределяет для отладки паритета).
    delay: u8,
    // Переиспользуемые буферы тика.
    indices: Vec<u32>,
    ext_full: Vec<f32>,
    flat: Vec<u32>,
}

impl BrainPool {
    /// Создать пул с ПРОВЕРКОЙ адаптера (для `Arena::new_pooled`): поднимает
    /// device сразу, чтобы тихо упасть на CPU-арену при его отсутствии.
    /// Сеть всё равно собирается лениво, при первом шаге (на этом же device).
    pub fn try_new() -> Result<Self, String> {
        let ctx = WgpuContext::new_blocking()?;
        let mut p = Self::new();
        p.ctx = Some(ctx);
        Ok(p)
    }

    /// Создать пул: адаптер поднимается лениво, при первом шаге (см. `rebuild`).
    pub fn new() -> Self {
        // Константы LIF из конфига (см. GpuDynamics::try_new — те же формулы).
        let NeuronConfig::LifAlpha { dt_ms, t_delay, .. } = pool_config() else {
            unreachable!()
        };
        let steps_delay = (t_delay / dt_ms).round().max(0.0) as u32;
        let delay = std::env::var("DIOMA_GPU_DELAY")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(steps_delay + 1)
            .min(31) as u8;
        Self {
            ctx: None,
            net: None,
            slots: Vec::new(),
            row_ptr: vec![0u32],
            col_idx: Vec::new(),
            values: Vec::new(),
            delays: Vec::new(),
            nt: Vec::new(),
            n_neurons: 0,
            needs_rebuild: false,
            failed: false,
            delay,
            indices: Vec::new(),
            ext_full: Vec::new(),
            flat: Vec::new(),
        }
    }

    /// Число зарегистрированных существ.
    pub fn n_creatures(&self) -> usize {
        self.slots.len()
    }

    /// Слоты в порядке регистрации (для разрезки flat-буфера по существам).
    pub fn slots(&self) -> &[PoolSlot] {
        &self.slots
    }

    /// Зарегистрировать существо в пуле: дописать его CSR в блочно-
    /// диагональный коннектом и пометить сеть на пересборку. Возвращает
    /// смещения для `Brain::new_pooled`.
    pub fn register(&mut self, csr: &ConnectomeData) -> PoolSlot {
        let neuron_off = self.n_neurons;
        let syn_off = self.values.len();
        let n = csr.n_neurons;
        let n_syn = csr.values.len();
        // Блок нейронов: строка (off + i) начинается со своих рёбер, сдвинутых
        // на syn_off (в общем CSR); колонки — на neuron_off (общая сеть).
        let mut shifted: Vec<u32> = Vec::with_capacity(n + 1);
        for &r in &csr.row_ptr {
            shifted.push(r as u32 + syn_off as u32);
        }
        self.row_ptr.pop(); // последний (общий счётчик) пересчитается ниже
        self.row_ptr.extend_from_slice(&shifted);
        self.col_idx.extend(
            csr.col_indices
                .iter()
                .map(|&c| c as u32 + neuron_off as u32),
        );
        self.values.extend_from_slice(&csr.values);
        self.delays.extend(std::iter::repeat_n(self.delay, n_syn));
        self.nt.extend(csr.nt_types.iter().map(|&t| if t < 0.0 { -1.0 } else { 1.0 }));
        let slot = PoolSlot { neuron_off, n, syn_off, n_syn };
        self.slots.push(slot);
        self.n_neurons += n;
        self.needs_rebuild = true;
        self.failed = false;
        slot
    }

    /// Перезалить веса в GPU, если хоть одно зеркало менялось. `mirrors` —
    /// в порядке слотов: (зеркало, dirty). Перед первым шагом пересобирает сеть.
    pub fn sync_weights(&mut self, mirrors: &[(&[f32], bool)]) -> Result<(), String> {
        debug_assert_eq!(mirrors.len(), self.slots.len());
        self.ensure_built()?;
        let mut any = false;
        for (slot, (m, dirty)) in self.slots.iter().zip(mirrors.iter()) {
            if *dirty {
                debug_assert_eq!(m.len(), slot.n_syn);
                self.values[slot.syn_off..slot.syn_off + slot.n_syn].copy_from_slice(m);
                any = true;
            }
        }
        if any {
            self.net.as_ref().expect("net built").set_weights(&self.values)?;
        }
        Ok(())
    }

    /// Один тик всех существ: инъекция токов → `spt` подшагов → история
    /// спайков. `exts[i]` — внешний ток существа i (None — спящий: нулевой
    /// блок, мембрана оседает к покою). Возвращает плоский буфер спайков
    /// `[подшаг][нейрон]` (вся сеть) и слоты — арена режет его по существам
    /// (`Brain::ingest_batched_flat`). При сбое GPU — нулевой буфер (все
    /// существа замолкают), тихий лог.
    pub fn step(&mut self, exts: &[Option<&[f32]>], spt: usize) -> (&[u32], &[PoolSlot]) {
        debug_assert_eq!(exts.len(), self.slots.len());
        if self.slots.is_empty() {
            return (&[], &self.slots);
        }
        if let Err(e) = self.ensure_built() {
            if !self.failed {
                eprintln!("[dioma] BrainPool::step — сбой сборки сети: {e}");
                self.failed = true;
            }
        }
        if self.failed {
            self.flat.resize(spt * self.n_neurons, 0);
            return (&self.flat, &self.slots);
        }
        // Полный вход тика: блоки существ (спящие — нули).
        self.ext_full.fill(0.0);
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(e) = exts.get(i).and_then(|o| *o) {
                debug_assert_eq!(e.len(), slot.n, "длина ext существа != n");
                self.ext_full[slot.neuron_off..slot.neuron_off + slot.n]
                    .copy_from_slice(e);
            }
        }
        let net = self.net.as_mut().expect("net built");
        self.flat.resize(spt * self.n_neurons, 0);
        // Пачка режется на куски ≤ GPU_MAX_STEPS_PER_SUBMIT (см. константу):
        // v_jump переинжектится перед каждым куском (внутри пачки он
        // прикладывается на каждом подшаге), история спайков скачивается
        // сразу после куска — до перезаписи кольца MAX_HIST. Порядок
        // подшагов и арифметика не меняются.
        let mut done = 0usize;
        while done < spt {
            let chunk = (spt - done).min(GPU_MAX_STEPS_PER_SUBMIT);
            let first = net.step_counter();
            let ok = net
                .inject_voltage(0, &self.indices, &self.ext_full)
                .is_ok()
                && net.step_many(chunk).is_ok();
            if !ok {
                eprintln!("[dioma] BrainPool::step — сбой GPU, тихий тик");
                self.failed = true;
                self.flat.fill(0);
                return (&self.flat, &self.slots);
            }
            let rows = &mut self.flat[done * self.n_neurons..(done + chunk) * self.n_neurons];
            if net.download_spikes_hist(first, chunk, rows).is_err() {
                eprintln!("[dioma] BrainPool::step — сбой ридбека, тихий тик");
                self.failed = true;
                self.flat.fill(0);
                return (&self.flat, &self.slots);
            }
            done += chunk;
        }
        (&self.flat, &self.slots)
    }

    /// Собрать (или пересобрать после спавна) WgpuNetwork. Новый device:
    /// буферы сети фиксированы при сборке, resize невозможен без изменений
    /// SiMuse. Спавны редки (старт арены/клик), хитч документирован.
    fn ensure_built(&mut self) -> Result<(), String> {
        if self.net.is_some() && !self.needs_rebuild {
            return Ok(());
        }
        let NeuronConfig::LifAlpha { dt_ms, tau_ref, tau_m, v_rest, v_thresh, v_reset, tau_syn, .. } =
            pool_config()
        else {
            unreachable!()
        };
        let n = self.n_neurons;
        if n == 0 {
            return Err("BrainPool: пустой пул".into());
        }
        let stride = n.div_ceil(64) * 64;
        let refr_steps = (tau_ref / dt_ms).round().max(1.0);
        let time_factor = dt_ms / tau_m;
        let syn_decay = 1.0 - dt_ms / tau_syn;
        let mut states = vec![0.0f32; 3 * stride];
        let mut params = vec![0.0f32; 7 * stride];
        for i in 0..n {
            states[i] = v_rest;
            states[stride + i] = 0.0;
            states[2 * stride + i] = refr_steps;
            params[i] = dt_ms;
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
        // Device из `try_new`, либо новый (ленивый путь / пересборка после
        // спавна: буферы WgpuNetwork фиксированы при сборке, resize без
        // изменений SiMuse невозможен — спавны редки, хитч документирован).
        let ctx = match self.ctx.take() {
            Some(ctx) => ctx,
            None => WgpuContext::new_blocking()?,
        };
        let net = WgpuNetwork::new(ctx, &[pop], &self.row_ptr, &self.col_idx, &self.values, &self.delays, dt_ms)?;
        net.set_nt_signs(&self.nt)?;
        self.indices = (0..n as u32).collect();
        self.ext_full = vec![0.0f32; n];
        self.net = Some(net);
        self.needs_rebuild = false;
        Ok(())
    }
}
