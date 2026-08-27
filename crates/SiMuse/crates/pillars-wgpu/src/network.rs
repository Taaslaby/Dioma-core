//! WgpuNetwork — гетерогенная сеть на wgpu compute (порт pillars-cuda CompiledNetwork).
//!
//! Шаг (бит-в-бит как `pillars-cuda/src/network.rs:942 launch_network_step`):
//!   1. copy  `ring[slot] → i_in`            (copy_slot_kernel)
//!   2. pops  каждый pop своим кернелом       (states/params/i_in/spikes с offset-окнами)
//!   3. zero  `ring[slot] = 0`               (zero_slot_kernel)
//!   4. scatter спайки → `ring[(slot+d)%32]`  (scatter_kernel, CSR + задержки)

use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::engine::WgpuContext;

const WGS: u32 = 64;
const MAX_SLOTS: u32 = 32;
/// Слотов истории спайков: ≥ максимальной пачки подшагов тика (LIF 50,
/// HAAP 500 — но HAAP живёт на CPU, GPU-пачка = 50).
const MAX_HIST: usize = 64;

/// Популяция нейронов: автоген-шейдер `#[dual_neuron]` (столбцовый SoA).
#[derive(Clone)]
pub struct WgpuPop {
    pub wgsl: &'static str,
    pub entry: &'static str,
    pub n: usize,
    pub ns: usize,
    pub np: usize,
    pub states: Vec<f32>,
    pub params: Vec<f32>,
}

pub struct WgpuNetwork {
    ctx: WgpuContext,
    n_total: usize,
    n_syn: usize,
    pops: Vec<PopDevice>,
    // глобальные буферы (используются в reset/download; остальные живут в bind groups)
    ring: wgpu::Buffer,
    i_in: wgpu::Buffer,
    spikes: wgpu::Buffer,
    // Спайки ПРОШЛОГО шага — их скеттерит scatter (семантика CPU run_lif_alpha:
    // alpha_fired от прошлого шага). Если скеттерить текущие, нормализация/
    // дофамин между шагами t и t+1 пишет одни и те же спайки с разными
    // весами → фаза ring расходится с CPU (диагностика dioma).
    spikes_prev: wgpu::Buffer,
    // История спайков (кольцо MAX_HIST × n_total): каждый шаг копирует свои
    // спайки в слот step%MAX_HIST. Позволяет скачать пачку шагов ОДНИМ
    // ридбеком (dioma батчит тик 50 шагов) вместо map_async+wait на подшаг.
    spikes_hist: wgpu::Buffer,
    hist_rb: wgpu::Buffer,
    // v_jump отложен: inject_voltage пишет аргументы, step() диспатчит кернел
    // в тот же compute pass — нет отдельного submit на подшаг.
    vjump_pending: Vec<bool>,
    trace: wgpu::Buffer,
    weights: wgpu::Buffer,
    // dopamine-обучение (R-STDP с per-synapse eligibility)
    trace_pre: wgpu::Buffer,
    trace_post: wgpu::Buffer,
    elig: wgpu::Buffer,
    // GPU-обучение: аккумулятор моторных сумм (2 × u32 atomic) + readback
    motor_acc: wgpu::Buffer,
    motor_rb: wgpu::Buffer,
    motor_args: wgpu::Buffer,   // MotorArgs { n, ml_lo, ml_hi, mr_lo, mr_hi, pad } (32 байта)
    keep: wgpu::Buffer,         // restrict-маска eligibility (u32 на синапс)
    // uniform args
    scatter_args: wgpu::Buffer, // ScatterArgs { n, slots, cur_slot, _pad } — обновляется каждый шаг
    scatter_args_staging: wgpu::Buffer, // слоты пачки шагов (step_many: один submit на тик)
    trace_args: wgpu::Buffer,   // TraceArgs { n, decay, _pad, _pad }
    stdp_args: wgpu::Buffer,    // StdpArgs { n, _pad, lr_ltp, lr_ltd, w_min, w_max } (24 байта)
    dopa_trace_args: wgpu::Buffer, // DopaTraceArgs { n, d_pre, d_post, _pad }
    elig_args: wgpu::Buffer,       // EligArgs { n_syn, d_e, a_ltp, a_ltd }
    reward_args: wgpu::Buffer,     // RewardArgs { n_syn, lr_dopa, w_min, w_cap }
    inject_args: wgpu::Buffer,  // InjectArgs { n, count, cur_slot, _pad } — обновляется в inject_current
    // инъекция (сенсорный ввод)
    inj_idx: wgpu::Buffer,
    inj_val: wgpu::Buffer,
    // знаки торможения пресинапсов (nt) для scatter — как run_lif_alpha
    signs: wgpu::Buffer,
    // v_jump: внешний ток как скачок v (semantics run_lif_alpha external_I)
    vjump_args: wgpu::Buffer,
    vjump_bgs: Vec<wgpu::BindGroup>,
    // readback
    spikes_rb: wgpu::Buffer,
    weights_rb: wgpu::Buffer,
    states_rb: wgpu::Buffer,
    // CPU-копии обратного CSR (для debug-пересчёта скеттера и тестов)
    rev_ptr: Vec<u32>,
    rev_src: Vec<u32>,
    rev_edge: Vec<u32>,
    rev_d: Vec<u32>,
    // bind groups
    copy_bg: wgpu::BindGroup,
    scatter_bg: wgpu::BindGroup,
    inject_bg: wgpu::BindGroup,
    trace_bg: wgpu::BindGroup,
    stdp_bg: wgpu::BindGroup,
    dopa_trace_bg: wgpu::BindGroup,
    elig_bg: wgpu::BindGroup,
    reward_bg: wgpu::BindGroup,
    motor_bg: wgpu::BindGroup,
    traces_motor_bg: wgpu::BindGroup,
    restrict_bg: wgpu::BindGroup,
    slot: u32,
    step: usize,
}

struct PopDevice {
    bind_group: wgpu::BindGroup,
    entry: &'static str,
    n: usize,
    ns: usize,
    stride: usize,
    i_slice: u64,
    states: wgpu::Buffer,
}

fn dispatch_groups(n: usize) -> u32 {
    (n as u32).div_ceil(WGS)
}

impl WgpuNetwork {
    pub fn new(
        mut ctx: WgpuContext,
        pops: &[WgpuPop],
        row_ptr: &[u32],
        col_idx: &[u32],
        weights: &[f32],
        delays: &[u8],
        dt: f32,
    ) -> Result<Self, String> {
        let n_total: usize = pops.iter().map(|p| p.n).sum();
        if n_total == 0 {
            return Err("WgpuNetwork: no neurons".into());
        }
        let n_syn = col_idx.len();

        // ── пайплайны ──
        for p in pops {
            if ctx.get_pipeline(p.entry).is_none() {
                ctx.compile(p.entry, p.wgsl, p.entry)?;
            }
        }
        let scatter_wgsl = include_str!("../shaders/csr_scatter.wgsl");
        for (name, entry) in [
            ("scatter_kernel", "scatter_kernel"),
            ("copy_slot_kernel", "copy_slot_kernel"),
            ("zero_slot_kernel", "zero_slot_kernel"),
            ("inject_kernel", "inject_kernel"),
            ("v_jump_kernel", "v_jump_kernel"),
        ] {
            if ctx.get_pipeline(name).is_none() {
                ctx.compile(name, scatter_wgsl, entry)?;
            }
        }
        let dop_wgsl = include_str!("../shaders/dopamine.wgsl");
        for (name, entry) in [
            ("trace_update_kernel", "trace_update_kernel"),
            ("csr_stdp_kernel", "csr_stdp_kernel"),
            ("dopa_traces_kernel", "dopa_traces_kernel"),
            ("elig_update_kernel", "elig_update_kernel"),
            ("reward_kernel", "reward_kernel"),
            ("motor_sum_kernel", "motor_sum_kernel"),
            ("traces_motor_kernel", "traces_motor_kernel"),
            ("restrict_elig_kernel", "restrict_elig_kernel"),
        ] {
            if ctx.get_pipeline(name).is_none() {
                ctx.compile(name, dop_wgsl, entry)?;
            }
        }

        let device = &ctx.device;

        // ── глобальные буферы (нужны раньше pop-биндингов) ──
        let ring = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ring"),
            contents: cast_slice(&vec![0.0f32; MAX_SLOTS as usize * n_total]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let i_in = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("i_in"),
            contents: cast_slice(&vec![0.0f32; n_total]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });        let spikes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spikes_all"),
            contents: cast_slice(&vec![0u32; n_total]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let spikes_prev = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spikes_prev"),
            contents: cast_slice(&vec![0u32; n_total]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let spikes_hist = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spikes_hist"),
            contents: cast_slice(&vec![0u32; MAX_HIST * n_total]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let hist_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hist_rb"),
            size: (MAX_HIST * n_total * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let trace = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trace"),
            contents: cast_slice(&vec![0.0f32; n_total]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let row_ptr_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("row_ptr"),
            contents: cast_slice(row_ptr),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let col_idx_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("col_idx"),
            contents: cast_slice(col_idx),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let weights_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("weights"),
            contents: cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let delays_u32: Vec<u32> = delays.iter().map(|&d| d as u32).collect();
        // ── in-edge CSR (обратный граф) для детерминированного per-post
        //    скеттера. Обход src по возрастанию → списки входящих рёбер
        //    каждого post уже отсортированы по источнику: порядок
        //    аккумуляции совпадает с CPU run_lif_alpha (alpha_fired
        //    строится по возрастанию) — бит-в-бит. ──
        let mut rev_ptr = vec![0u32; n_total + 1];
        for &c in col_idx {
            rev_ptr[c as usize + 1] += 1;
        }
        for i in 0..n_total {
            rev_ptr[i + 1] += rev_ptr[i];
        }
        let mut cursor = rev_ptr.clone();
        let mut rev_src = vec![0u32; n_syn];
        let mut rev_edge = vec![0u32; n_syn];
        let mut rev_d = vec![0u32; n_syn];
        for src in 0..n_total {
            let s = src as u32;
            for e in row_ptr[src] as usize..row_ptr[src + 1] as usize {
                let post = col_idx[e] as usize;
                let k = cursor[post] as usize;
                rev_src[k] = s;
                rev_edge[k] = e as u32;
                rev_d[k] = delays_u32[e];
                cursor[post] += 1;
            }
        }
        let rev_ptr_cpu = rev_ptr.clone();
        let rev_src_cpu = rev_src.clone();
        let rev_edge_cpu = rev_edge.clone();
        let rev_d_cpu = rev_d.clone();
        let rev_ptr_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rev_ptr"),
            contents: cast_slice(&rev_ptr),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rev_src_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rev_src"),
            contents: cast_slice(&rev_src),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rev_edge_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rev_edge"),
            contents: cast_slice(&rev_edge),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rev_d_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rev_d"),
            contents: cast_slice(&rev_d),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // per-synapse pre-индекс (для dopamine eligibility) — из CSR row_ptr
        let mut pre_of = vec![0u32; n_syn];
        for pre in 0..n_total {
            pre_of[row_ptr[pre] as usize..row_ptr[pre + 1] as usize].fill(pre as u32);
        }
        let pre_of_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pre_of"),
            contents: cast_slice(&pre_of),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let trace_pre = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trace_pre"),
            contents: cast_slice(&vec![0.0f32; n_total]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let trace_post = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trace_post"),
            contents: cast_slice(&vec![0.0f32; n_total]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let elig = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("elig"),
            contents: cast_slice(&vec![0.0f32; n_syn.max(1)]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        // GPU-обучение: моторные суммы (2 × u32 atomic) + restrict-маска.
        let motor_acc = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motor_acc"),
            contents: cast_slice(&[0u32, 0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let motor_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motor_rb"),
            size: 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let motor_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motor_args"),
            contents: cast_slice(&[0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let keep = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("keep"),
            contents: cast_slice(&vec![1u32; n_syn.max(1)]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let restrict_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("restrict_args"),
            contents: cast_slice(&[n_syn as u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── uniform args + readback ──
        let scatter_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scatter_args"),
            contents: cast_slice(&[n_total as u32, MAX_SLOTS, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // Слоты пачки (n × 16 байт): заполняется на CPU, копируется в
        // scatter_args перед каждым шагом внутри одного encoder.
        let scatter_args_staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scatter_args_staging"),
            contents: cast_slice(&vec![0u32; MAX_HIST * 4]),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let trace_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trace_args"),
            contents: cast_slice(&[n_total as u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let stdp_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stdp_args"),
            contents: &{
                let mut b = [0u8; 24];
                b[0..4].copy_from_slice(&(n_total as u32).to_le_bytes());
                b
            },
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let dopa_trace_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dopa_trace_args"),
            contents: cast_slice(&[n_total as u32, 0.0f32.to_bits(), 0.0f32.to_bits(), 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let elig_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("elig_args"),
            contents: cast_slice(&[n_syn as u32, 0.0f32.to_bits(), 0.0f32.to_bits(), 0.0f32.to_bits()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let reward_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reward_args"),
            contents: cast_slice(&[n_syn as u32, 0.0f32.to_bits(), 0.0f32.to_bits(), 0.0f32.to_bits()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let inject_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inject_args"),
            contents: cast_slice(&[n_total as u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        // inj_idx/inj_val — входные буферы инъекции: размер n_total, чтобы
        // inject_current мог адресовать любую долю популяции (сенсоры = все
        // нейроны существа диомы). Размер 1 был латентным багом для count > 1.
        let inj_idx = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inj_idx"),
            contents: cast_slice(&vec![0u32; n_total]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let inj_val = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inj_val"),
            contents: cast_slice(&vec![0.0f32; n_total]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Знаки пресинапсов для scatter: дефолт +1 (всё возбуждающее),
        // диома ставит через `set_nt_signs` (nt_types коннектома).
        let signs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("signs"),
            contents: cast_slice(&vec![1.0f32; n_total]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        // v_jump_args { n, count, stride, _pad } — обновляется в inject_voltage.
        let vjump_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vjump_args"),
            contents: cast_slice(&[0u32, 0u32, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let spikes_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spikes_rb"),
            size: (n_total * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let weights_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights_rb"),
            size: (n_syn.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // readback для download_states — размер = максимальный pop (ns * stride)
        let states_rb_size = pops
            .iter()
            .map(|p| {
                let stride = p.n.div_ceil(WGS as usize) * WGS as usize;
                p.ns * stride
            })
            .max()
            .unwrap_or(0);
        let states_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("states_rb"),
            size: (states_rb_size.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ── вспомогательные bind groups (layout'ы copy и zero РАЗНЫЕ: ring read vs read_write) ──
        let copy_bg = {
            let layout = ctx
                .get_pipeline("copy_slot_kernel")
                .expect("copy pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("copy_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ring.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: i_in.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: scatter_args.as_entire_binding() },
                ],
            })
        };
        let scatter_bg = {
            let layout = ctx
                .get_pipeline("scatter_kernel")
                .expect("scatter pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scatter_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: rev_ptr_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: rev_src_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: rev_edge_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: weights_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: rev_d_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: spikes_prev.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: ring.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: scatter_args.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: signs.as_entire_binding() },
                ],
            })
        };
        let inject_bg = {
            let layout = ctx
                .get_pipeline("inject_kernel")
                .expect("inject pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inject_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ring.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: inj_idx.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: inj_val.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: inject_args.as_entire_binding() },
                ],
            })
        };
        let trace_bg = {
            let layout = ctx
                .get_pipeline("trace_update_kernel")
                .expect("trace pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("trace_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: trace.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: spikes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: trace_args.as_entire_binding() },
                ],
            })
        };
        let stdp_bg = {
            let layout = ctx
                .get_pipeline("csr_stdp_kernel")
                .expect("stdp pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("stdp_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: weights_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: row_ptr_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: col_idx_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: trace.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: stdp_args.as_entire_binding() },
                ],
            })
        };
        let dopa_trace_bg = {
            let layout = ctx
                .get_pipeline("dopa_traces_kernel")
                .expect("dopa_trace pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dopa_trace_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: trace_pre.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: trace_post.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: spikes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: dopa_trace_args.as_entire_binding() },
                ],
            })
        };
        let elig_bg = {
            let layout = ctx
                .get_pipeline("elig_update_kernel")
                .expect("elig pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("elig_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: elig.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: pre_of_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: col_idx_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: trace_pre.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: trace_post.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: spikes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: elig_args.as_entire_binding() },
                ],
            })
        };
        let reward_bg = {
            let layout = ctx
                .get_pipeline("reward_kernel")
                .expect("reward pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reward_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: weights_b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: elig.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: reward_args.as_entire_binding() },
                ],
            })
        };
        let motor_bg = {
            let layout = ctx
                .get_pipeline("motor_sum_kernel")
                .expect("motor pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("motor_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: motor_acc.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: spikes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: motor_args.as_entire_binding() },
                ],
            })
        };
        let restrict_bg = {
            let layout = ctx
                .get_pipeline("restrict_elig_kernel")
                .expect("restrict pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("restrict_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: elig.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: keep.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: restrict_args.as_entire_binding() },
                ],
            })
        };
        let traces_motor_bg = {
            let layout = ctx
                .get_pipeline("traces_motor_kernel")
                .expect("traces_motor pipeline")
                .get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("traces_motor_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: trace_pre.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: trace_post.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: spikes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: dopa_trace_args.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: motor_acc.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: motor_args.as_entire_binding() },
                ],
            })
        };

        // ── популяции: буферы + bind group (i_in/spikes — offset-окна) ──
        let mut pop_devs = Vec::with_capacity(pops.len());
        let mut offset = 0u64;
        for p in pops {
            let stride = p.n.div_ceil(WGS as usize) * WGS as usize;
            let states = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pop_states"),
                contents: cast_slice(&p.states),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pop_params"),
                contents: cast_slice(&p.params),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            // WgslArgs { dt: f32, n: u32, stride: u32 } — 16 байт
            let mut args_bytes = [0u8; 16];
            args_bytes[0..4].copy_from_slice(&dt.to_le_bytes());
            args_bytes[4..8].copy_from_slice(&(p.n as u32).to_le_bytes());
            args_bytes[8..12].copy_from_slice(&(stride as u32).to_le_bytes());
            let args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pop_args"),
                contents: &args_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let layout = ctx
                .get_pipeline(p.entry)
                .expect("pop pipeline compiled")
                .get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pop_bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: states.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &i_in,
                            offset: offset * 4,
                            size: std::num::NonZeroU64::new(p.n as u64 * 4),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &spikes,
                            offset: offset * 4,
                            size: std::num::NonZeroU64::new(p.n as u64 * 4),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 4, resource: args.as_entire_binding() },
                ],
            });
            pop_devs.push(PopDevice {
                bind_group,
                entry: p.entry,
                n: p.n,
                ns: p.ns,
                stride,
                i_slice: offset,
                states,
            });
            offset += p.n as u64;
        }

        // v_jump-биндинги: по одному на популяцию (states буфер pop-локальный),
        // inj_idx/inj_val/vjump_args — общие.
        let vjump_bgs = pop_devs
            .iter()
            .map(|pd| {
                let layout = ctx
                    .get_pipeline("v_jump_kernel")
                    .expect("v_jump pipeline")
                    .get_bind_group_layout(0);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("vjump_bg"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: pd.states.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: inj_idx.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: inj_val.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: vjump_args.as_entire_binding() },
                    ],
                })
            })
            .collect();

        Ok(Self {
            ctx,
            n_total,
            n_syn,
            pops: pop_devs,
            ring,
            i_in,
            spikes,
            spikes_prev,
            spikes_hist,
            hist_rb,
            vjump_pending: vec![false; pops.len()],
            trace,
            weights: weights_b,
            trace_pre,
            trace_post,
            elig,
            motor_acc,
            motor_rb,
            motor_args,
            keep,
            scatter_args,
            scatter_args_staging,
            trace_args,
            stdp_args,
            dopa_trace_args,
            elig_args,
            reward_args,
            inject_args,
            signs,
            vjump_args,
            vjump_bgs,
            inj_idx,
            inj_val,
            spikes_rb,
            weights_rb,
            states_rb,
            rev_ptr: rev_ptr_cpu,
            rev_src: rev_src_cpu,
            rev_edge: rev_edge_cpu,
            rev_d: rev_d_cpu,
            copy_bg,
            scatter_bg,
            inject_bg,
            trace_bg,
            stdp_bg,
            dopa_trace_bg,
            elig_bg,
            reward_bg,
            motor_bg,
            traces_motor_bg,
            restrict_bg,
            slot: 0,
            step: 0,
        })
    }

    pub fn n_total(&self) -> usize {
        self.n_total
    }
    pub fn n_syn(&self) -> usize {
        self.n_syn
    }
    pub fn slot(&self) -> u32 {
        self.slot
    }
    /// Число популяций.
    pub fn n_pops(&self) -> usize {
        self.pops.len()
    }
    /// Число нейронов в популяции `pop`.
    pub fn pop_n(&self, pop: usize) -> usize {
        self.pops.get(pop).map_or(0, |p| p.n)
    }
    /// Счётчик шагов (сбрасывается в `reset`).
    pub fn step_counter(&self) -> usize {
        self.step
    }

    /// Один шаг динамики: vjump → copy → pops → scatter (submit на шаг).
    pub fn step(&mut self) -> Result<(), String> {
        self.step_many(1)
    }

    /// `n` шагов динамики ОДНИМ submit (один encoder): диспатчи всех шагов
    /// идут подряд в порядке очереди, аргументы скеттера (слоты) — через
    /// staging-буфер. Для батч-тика dioma (50 подшагов, константный вход) это
    /// убирает 49 сабмитов на тик — основной оверхед wgpu-пути на малом мозге.
    pub fn step_many(&mut self, n: usize) -> Result<(), String> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let n = n.min(MAX_HIST);
        // Слоты всех шагов пачки (ScatterArgs { n, slots, cur_slot, _pad }).
        let mut args = Vec::with_capacity(n * 4);
        for k in 0..n {
            args.extend_from_slice(&[
                self.n_total as u32,
                MAX_SLOTS,
                (self.slot + k as u32) % MAX_SLOTS,
                0u32,
            ]);
        }
        queue.write_buffer(&self.scatter_args_staging, 0, cast_slice(&args));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_step_many"),
        });
        for k in 0..n {
            // Скеттер-аргументы шага k (слот) — внутри encoder, до его пасса.
            encoder.copy_buffer_to_buffer(
                &self.scatter_args_staging,
                (k as u64) * 16,
                &self.scatter_args,
                0,
                16,
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vjump+copy+pop+scatter"),
                    timestamp_writes: None,
                });
                // 0. v_jump (внешний ток → v) — отложенный inject_voltage:
                //    один на пачку (вход тика константен). ДО pops (v += ext).
                for (pop, &pending) in self.vjump_pending.iter().enumerate() {
                    if pending {
                        pass.set_pipeline(
                            self.ctx.get_pipeline("v_jump_kernel").expect("v_jump pipeline"),
                        );
                        pass.set_bind_group(0, &self.vjump_bgs[pop], &[]);
                        pass.dispatch_workgroups(dispatch_groups(self.pops[pop].n), 1, 1);
                    }
                }
                // 1. copy ring[slot] → i_in
                pass.set_pipeline(self.ctx.get_pipeline("copy_slot_kernel").expect("copy pipeline"));
                pass.set_bind_group(0, &self.copy_bg, &[]);
                pass.dispatch_workgroups(dispatch_groups(self.n_total), 1, 1);
                // 2. pops (i_in/spikes — offset-окна)
                for pop in &self.pops {
                    let pipeline = self.ctx.get_pipeline(pop.entry).expect("pop pipeline");
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &pop.bind_group, &[]);
                    pass.dispatch_workgroups(dispatch_groups(pop.n), 1, 1);
                }
                // 3. scatter спайки → ring[(slot+d)%32] (zero текущего слота
                //    влит в скеттер — минус один диспатч на шаг).
                if self.n_syn > 0 {
                    pass.set_pipeline(self.ctx.get_pipeline("scatter_kernel").expect("scatter pipeline"));
                    pass.set_bind_group(0, &self.scatter_bg, &[]);
                    pass.dispatch_workgroups(dispatch_groups(self.n_total), 1, 1);
                }
            }
            // Продвинуть скеттер-источник: прошлые спайки ← текущие.
            encoder.copy_buffer_to_buffer(
                &self.spikes,
                0,
                &self.spikes_prev,
                0,
                (self.n_total * 4) as u64,
            );
            // История спайков: слот (self.step + k) % MAX_HIST.
            encoder.copy_buffer_to_buffer(
                &self.spikes,
                0,
                &self.spikes_hist,
                (((self.step + k) % MAX_HIST) * self.n_total * 4) as u64,
                (self.n_total * 4) as u64,
            );
        }
        self.vjump_pending.iter_mut().for_each(|p| *p = false);
        queue.submit(Some(encoder.finish()));
        self.slot = (self.slot + n as u32) % MAX_SLOTS;
        self.step += n;
        Ok(())
    }

    /// `n` шагов динамики + dopamine-eligibility ОДНИМ submit — путь
    /// «GPU-обучение»: подшаги и пластичность живут в одном encoder, ридбеки
    /// вообще не нужны (моторные суммы — через `download_motor_sums`).
    ///
    /// Порядок на подшаге совпадает с CPU `Brain::step` (pillars brain.rs):
    /// трассы распадаются/накапливаются по спайкам ТЕКУЩЕГО шага, затем
    /// eligibility обновляется по этим же трассам. Веса не трогаются внутри
    /// тика (награды/гомеостаз — на границах), поэтому пачка бит-в-бит
    /// равна n последовательных `step()` + `eligibility_step()`.
    ///
    /// Моторные суммы: спайки [ml_lo, ml_hi) → acc[0], [mr_lo, mr_hi) → acc[1]
    /// (atomicAdd u32 — детерминированная сумма независимо от порядка
    /// инвокаций). Аккумулятор ОБНУЛЯЕТСЯ в начале вызова.
    pub fn step_many_learn(
        &mut self,
        n: usize,
        d_pre: f32,
        d_post: f32,
        d_e: f32,
        a_ltp: f32,
        a_ltd: f32,
        ml_lo: u32,
        ml_hi: u32,
        mr_lo: u32,
        mr_hi: u32,
    ) -> Result<(), String> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let n = n.min(MAX_HIST);
        // DopaTraceArgs { n, d_pre, d_post, _pad }
        queue.write_buffer(
            &self.dopa_trace_args,
            0,
            cast_slice(&[self.n_total as u32, d_pre.to_bits(), d_post.to_bits(), 0u32]),
        );
        // EligArgs { n_syn, d_e, a_ltp, a_ltd }
        queue.write_buffer(
            &self.elig_args,
            0,
            cast_slice(&[
                self.n_syn as u32,
                d_e.to_bits(),
                a_ltp.to_bits(),
                a_ltd.to_bits(),
            ]),
        );
        // MotorArgs { n, ml_lo, ml_hi, mr_lo, mr_hi, pad } — 32 байта.
        queue.write_buffer(
            &self.motor_args,
            0,
            cast_slice(&[
                self.n_total as u32,
                ml_lo,
                ml_hi,
                mr_lo,
                mr_hi,
                0u32,
                0u32,
                0u32,
            ]),
        );
        // Обнулить аккумулятор моторных сумм.
        queue.write_buffer(&self.motor_acc, 0, cast_slice(&[0u32, 0u32]));

        let mut args = Vec::with_capacity(n * 4);
        for k in 0..n {
            args.extend_from_slice(&[
                self.n_total as u32,
                MAX_SLOTS,
                (self.slot + k as u32) % MAX_SLOTS,
                0u32,
            ]);
        }
        queue.write_buffer(&self.scatter_args_staging, 0, cast_slice(&args));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_step_many_learn"),
        });
        for k in 0..n {
            encoder.copy_buffer_to_buffer(
                &self.scatter_args_staging,
                (k as u64) * 16,
                &self.scatter_args,
                0,
                16,
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vjump+copy+pop+scatter+elig+motor"),
                    timestamp_writes: None,
                });
                for (pop, &pending) in self.vjump_pending.iter().enumerate() {
                    if pending {
                        pass.set_pipeline(
                            self.ctx.get_pipeline("v_jump_kernel").expect("v_jump pipeline"),
                        );
                        pass.set_bind_group(0, &self.vjump_bgs[pop], &[]);
                        pass.dispatch_workgroups(dispatch_groups(self.pops[pop].n), 1, 1);
                    }
                }
                pass.set_pipeline(self.ctx.get_pipeline("copy_slot_kernel").expect("copy pipeline"));
                pass.set_bind_group(0, &self.copy_bg, &[]);
                pass.dispatch_workgroups(dispatch_groups(self.n_total), 1, 1);
                for pop in &self.pops {
                    let pipeline = self.ctx.get_pipeline(pop.entry).expect("pop pipeline");
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &pop.bind_group, &[]);
                    pass.dispatch_workgroups(dispatch_groups(pop.n), 1, 1);
                }
                if self.n_syn > 0 {
                    pass.set_pipeline(self.ctx.get_pipeline("scatter_kernel").expect("scatter pipeline"));
                    pass.set_bind_group(0, &self.scatter_bg, &[]);
                    pass.dispatch_workgroups(dispatch_groups(self.n_total), 1, 1);
                }
                // dopamine-eligibility: fused-кернел — трассы по спайкам
                // текущего шага + моторные суммы (один диспатч вместо двух),
                // затем per-synapse eligibility (порядок как в `eligibility_step`).
                pass.set_pipeline(self.ctx.get_pipeline("traces_motor_kernel").expect("traces_motor pipeline"));
                pass.set_bind_group(0, &self.traces_motor_bg, &[]);
                pass.dispatch_workgroups(dispatch_groups(self.n_total), 1, 1);
                if self.n_syn > 0 {
                    pass.set_pipeline(self.ctx.get_pipeline("elig_update_kernel").expect("elig pipeline"));
                    pass.set_bind_group(0, &self.elig_bg, &[]);
                    pass.dispatch_workgroups(dispatch_groups(self.n_syn), 1, 1);
                }
            }
            encoder.copy_buffer_to_buffer(
                &self.spikes,
                0,
                &self.spikes_prev,
                0,
                (self.n_total * 4) as u64,
            );
        }
        self.vjump_pending.iter_mut().for_each(|p| *p = false);
        queue.submit(Some(encoder.finish()));
        self.slot = (self.slot + n as u32) % MAX_SLOTS;
        self.step += n;
        Ok(())
    }

    /// Скачать моторные суммы (2 × u32) — ОДИН ридбек на тик (8 байт).
    pub fn download_motor_sums(&self) -> Result<(u32, u32), String> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motor_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.motor_acc, 0, &self.motor_rb, 0, 8);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        self.motor_rb
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = self.motor_rb.slice(..).get_mapped_range();
        let src: &[u32] = bytemuck::cast_slice(&view);
        let out = (src[0], src[1]);
        drop(view);
        self.motor_rb.unmap();
        Ok(out)
    }

    /// Обнулить eligibility всех синапсов (окно закрыто / заморозка).
    pub fn clear_eligibility(&self) -> Result<(), String> {
        self.ctx
            .queue
            .write_buffer(&self.elig, 0, cast_slice(&vec![0.0f32; self.n_syn.max(1)]));
        Ok(())
    }

    /// Оставить eligibility только на синапсах с keep[k]==true (заморозка
    /// инстинктов/прожигание): остальные обнуляются. Один диспатч, без
    /// ридбеков.
    pub fn restrict_eligibility(&self, keep: &[bool]) -> Result<(), String> {
        if keep.len() < self.n_syn {
            return Err(format!(
                "restrict_eligibility: keep {} < n_syn {}",
                keep.len(),
                self.n_syn
            ));
        }
        let mask: Vec<u32> = keep.iter().take(self.n_syn).map(|&k| k as u32).collect();
        self.ctx.queue.write_buffer(&self.keep, 0, cast_slice(&mask));
        if self.n_syn == 0 {
            return Ok(());
        }
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_restrict"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("restrict_elig"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.get_pipeline("restrict_elig_kernel").expect("restrict pipeline"));
            pass.set_bind_group(0, &self.restrict_bg, &[]);
            pass.dispatch_workgroups(dispatch_groups(self.n_syn), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Батч-ридбек истории спайков: `count` шагов начиная с `first` — ОДИН
    /// map_async+wait вместо одного на подшаг. `out` ≥ count × n_total u32.
    pub fn download_spikes_hist(&self, first: usize, count: usize, out: &mut [u32]) -> Result<(), String> {
        if count > MAX_HIST {
            return Err(format!("download_spikes_hist: count {count} > MAX_HIST {MAX_HIST}"));
        }
        let need = count * self.n_total;
        if out.len() < need {
            return Err(format!(
                "download_spikes_hist: out {} < {need} (count {count} × n {})",
                out.len(),
                self.n_total
            ));
        }
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hist_copy"),
        });
        for k in 0..count {
            let slot = (first + k) % MAX_HIST;
            encoder.copy_buffer_to_buffer(
                &self.spikes_hist,
                (slot * self.n_total * 4) as u64,
                &self.hist_rb,
                (k * self.n_total * 4) as u64,
                (self.n_total * 4) as u64,
            );
        }
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        self.hist_rb
            .slice(..(need * 4) as u64)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = self.hist_rb.slice(..(need * 4) as u64).get_mapped_range();
        let src: &[u32] = bytemuck::cast_slice(&view);
        out[..need].copy_from_slice(src);
        drop(view);
        self.hist_rb.unmap();
        Ok(())
    }

    /// Скачать состояния популяции `pop` (SoA, длина `ns * stride`).
    pub fn download_states(&self, pop: usize, out: &mut [f32]) -> Result<(), String> {
        let p = self
            .pops
            .get(pop)
            .ok_or_else(|| format!("WgpuNetwork: pop {pop} out of range"))?;
        let len = p.ns * p.stride;
        if out.len() != len {
            return Err(format!("download_states: {} != {len}", out.len()));
        }
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_states_dl"),
        });
        encoder.copy_buffer_to_buffer(&p.states, 0, &self.states_rb, 0, (len * 4) as u64);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        self.states_rb
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = self.states_rb.slice(..).get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&view);
        out.copy_from_slice(&src[..len]);
        drop(view);
        self.states_rb.unmap();
        Ok(())
    }

    /// Сброс всех популяций к начальным состояниям + очистка ring/spikes/trace.
    /// `pops` — те же спеки, что передавались в [`WgpuNetwork::new`].
    pub fn reset(&mut self, pops: &[WgpuPop]) -> Result<(), String> {
        if pops.len() != self.pops.len() {
            return Err(format!(
                "reset: {} pops != {}",
                pops.len(),
                self.pops.len()
            ));
        }
        let queue = &self.ctx.queue;
        for (dev, spec) in self.pops.iter().zip(pops.iter()) {
            if dev.n != spec.n || dev.ns != spec.ns {
                return Err(format!(
                    "reset: pop spec mismatch (n {} vs {}, ns {} vs {})",
                    dev.n, spec.n, dev.ns, spec.ns
                ));
            }
            queue.write_buffer(&dev.states, 0, cast_slice(&spec.states));
        }
        queue.write_buffer(
            &self.ring,
            0,
            cast_slice(&vec![0.0f32; MAX_SLOTS as usize * self.n_total]),
        );
        queue.write_buffer(&self.spikes, 0, cast_slice(&vec![0u32; self.n_total]));
        queue.write_buffer(&self.spikes_prev, 0, cast_slice(&vec![0u32; self.n_total]));
        queue.write_buffer(&self.trace, 0, cast_slice(&vec![0.0f32; self.n_total]));
        queue.write_buffer(&self.trace_pre, 0, cast_slice(&vec![0.0f32; self.n_total]));
        queue.write_buffer(&self.trace_post, 0, cast_slice(&vec![0.0f32; self.n_total]));
        queue.write_buffer(&self.elig, 0, cast_slice(&vec![0.0f32; self.n_syn.max(1)]));
        queue.write_buffer(&self.motor_acc, 0, cast_slice(&[0u32, 0u32]));
        queue.write_buffer(&self.keep, 0, cast_slice(&vec![1u32; self.n_syn.max(1)]));
        self.slot = 0;
        self.step = 0;
        Ok(())
    }

    /// STDP: обновить следы и веса по текущим спайкам (после step()).
    /// Би-в-бит как cuda `trace_update_kernel` + `csr_stdp_kernel`.
    pub fn stdp_step(
        &mut self,
        trace_decay: f32,
        lr_ltp: f32,
        lr_ltd: f32,
        w_min: f32,
        w_max: f32,
    ) -> Result<(), String> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        // TraceArgs { n, decay, _pad, _pad }
        let mut targs = [0u32; 4];
        targs[0] = self.n_total as u32;
        targs[1] = trace_decay.to_bits();
        queue.write_buffer(&self.trace_args, 0, cast_slice(&targs));
        // StdpArgs { n, _pad, lr_ltp, lr_ltd, w_min, w_max } — 24 байта
        let mut sargs = [0u8; 24];
        sargs[0..4].copy_from_slice(&(self.n_total as u32).to_le_bytes());
        sargs[8..12].copy_from_slice(&lr_ltp.to_le_bytes());
        sargs[12..16].copy_from_slice(&lr_ltd.to_le_bytes());
        sargs[16..20].copy_from_slice(&w_min.to_le_bytes());
        sargs[20..24].copy_from_slice(&w_max.to_le_bytes());
        queue.write_buffer(&self.stdp_args, 0, &sargs);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_stdp"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trace+stdp"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.get_pipeline("trace_update_kernel").expect("trace pipeline"));
            pass.set_bind_group(0, &self.trace_bg, &[]);
            pass.dispatch_workgroups(dispatch_groups(self.n_total), 1, 1);
            if self.n_syn > 0 {
                pass.set_pipeline(self.ctx.get_pipeline("csr_stdp_kernel").expect("stdp pipeline"));
                pass.set_bind_group(0, &self.stdp_bg, &[]);
                pass.dispatch_workgroups(dispatch_groups(self.n_total), 1, 1);
            }
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Dopamine-шаг: обновить пер-нейронные следы и per-synapse eligibility
    /// по текущим спайкам (после `step()`). Би-в-бит `pillars::api::learn::brain`
    /// `step`: traces с двумя tau, eligibility с событиями LTP/LTD.
    ///
    /// `d_pre = exp(-dt/tau_pre)`, `d_post = exp(-dt/tau_post)`, `d_e = exp(-dt/tau_e)`
    /// — константы распада, вычисляемые вызывающим кодом (как в brain.rs).
    pub fn eligibility_step(
        &mut self,
        d_pre: f32,
        d_post: f32,
        d_e: f32,
        a_ltp: f32,
        a_ltd: f32,
    ) -> Result<(), String> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        // DopaTraceArgs { n, d_pre, d_post, _pad }
        let mut dtargs = [0u32; 4];
        dtargs[0] = self.n_total as u32;
        dtargs[1] = d_pre.to_bits();
        dtargs[2] = d_post.to_bits();
        queue.write_buffer(&self.dopa_trace_args, 0, cast_slice(&dtargs));
        // EligArgs { n_syn, d_e, a_ltp, a_ltd }
        let mut eargs = [0u32; 4];
        eargs[0] = self.n_syn as u32;
        eargs[1] = d_e.to_bits();
        eargs[2] = a_ltp.to_bits();
        eargs[3] = a_ltd.to_bits();
        queue.write_buffer(&self.elig_args, 0, cast_slice(&eargs));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_elig"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dopa_traces+elig"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.get_pipeline("dopa_traces_kernel").expect("dopa_trace pipeline"));
            pass.set_bind_group(0, &self.dopa_trace_bg, &[]);
            pass.dispatch_workgroups(dispatch_groups(self.n_total), 1, 1);
            if self.n_syn > 0 {
                pass.set_pipeline(self.ctx.get_pipeline("elig_update_kernel").expect("elig pipeline"));
                pass.set_bind_group(0, &self.elig_bg, &[]);
                pass.dispatch_workgroups(dispatch_groups(self.n_syn), 1, 1);
            }
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Применить дофамин: `Δw = lr·dopamine·elig`, клэмп `[w_min, w_cap]`;
    /// синапсы с `Δw == 0` не трогаются (врождённые веса сохраняются).
    /// Би-в-бит `pillars::api::learn::brain::reward_dopamine`.
    pub fn reward(&mut self, lr: f32, dopamine: f32, w_min: f32, w_cap: f32) -> Result<(), String> {
        if self.n_syn == 0 {
            return Ok(());
        }
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        // RewardArgs { n_syn, lr_dopa, w_min, w_cap }
        let mut rargs = [0u32; 4];
        rargs[0] = self.n_syn as u32;
        rargs[1] = (lr * dopamine).to_bits();
        rargs[2] = w_min.to_bits();
        rargs[3] = w_cap.to_bits();
        queue.write_buffer(&self.reward_args, 0, cast_slice(&rargs));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_reward"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.get_pipeline("reward_kernel").expect("reward pipeline"));
            pass.set_bind_group(0, &self.reward_bg, &[]);
            pass.dispatch_workgroups(dispatch_groups(self.n_syn), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Скачать спайки последнего шага (бит-в-бит как cuda download_spikes).
    pub fn download_spikes(&self, out: &mut [u32]) -> Result<(), String> {
        assert_eq!(out.len(), self.n_total);
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("spikes_copy"),
        });
        encoder.copy_buffer_to_buffer(
            &self.spikes,
            0,
            &self.spikes_rb,
            0,
            (self.n_total * 4) as u64,
        );
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        self.spikes_rb
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = self.spikes_rb.slice(..).get_mapped_range();
        let src: &[u32] = bytemuck::cast_slice(&view);
        out.copy_from_slice(src);
        drop(view);
        self.spikes_rb.unmap();
        Ok(())
    }

    /// Скачать текущие веса синапсов (наследственность/чекпоинты диомы).
    pub fn weights(&self) -> Result<Vec<f32>, String> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("weights_copy"),
        });
        encoder.copy_buffer_to_buffer(
            &self.weights,
            0,
            &self.weights_rb,
            0,
            (self.n_syn.max(1) * 4) as u64,
        );
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        self.weights_rb
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = self.weights_rb.slice(..).get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&view);
        let out = src[..self.n_syn].to_vec();
        drop(view);
        self.weights_rb.unmap();
        Ok(out)
    }

    /// Сенсорный ввод: inject значения в нейроны (как cuda `inject_current`).
    /// Значения попадают в текущий слот ring и приходят на следующем шаге.
    /// `indices` — локальные индексы внутри популяции `pop`.
    pub fn inject_current(&mut self, pop: usize, indices: &[u32], values: &[f32]) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("inject_current: indices/values length mismatch".into());
        }
        if indices.is_empty() {
            return Ok(());
        }
        let p = self
            .pops
            .get(pop)
            .ok_or_else(|| format!("inject_current: pop {pop} out of range"))?;
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let global: Vec<u32> = indices
            .iter()
            .map(|&i| {
                assert!((i as u64) < p.n as u64, "inject_current: index {i} out of range");
                (p.i_slice + i as u64) as u32
            })
            .collect();
        queue.write_buffer(&self.inj_idx, 0, cast_slice(&global));
        queue.write_buffer(&self.inj_val, 0, cast_slice(values));
        // InjectArgs { n, count, cur_slot, _pad }
        queue.write_buffer(
            &self.inject_args,
            0,
            cast_slice(&[self.n_total as u32, indices.len() as u32, self.slot, 0u32]),
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu_inject"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("inject"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.ctx.get_pipeline("inject_kernel").expect("inject pipeline"));
            pass.set_bind_group(0, &self.inject_bg, &[]);
            pass.dispatch_workgroups(dispatch_groups(indices.len()), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// DEBUG: скачать i_in (вход популяции после copy_slot).
    pub fn debug_i_in(&self) -> Vec<f32> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dbg_iin_rb"),
            size: (self.n_total * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dbg_iin_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.i_in, 0, &rb, 0, (self.n_total * 4) as u64);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = rb.slice(..).get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&view);
        let out = src.to_vec();
        drop(view);
        rb.unmap();
        out
    }

    /// DEBUG: скачать eligibility-буфер (сверка с CPU plast пошагово).
    pub fn debug_elig(&self) -> Vec<f32> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dbg_elig_rb"),
            size: (self.n_syn.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dbg_elig_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.elig, 0, &rb, 0, (self.n_syn.max(1) * 4) as u64);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = rb.slice(..).get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&view);
        let out = src[..self.n_syn].to_vec();
        drop(view);
        rb.unmap();
        out
    }

    /// DEBUG: скачать per-neuron трассы (pre и post) — сверка с CPU.
    pub fn debug_traces(&self) -> (Vec<f32>, Vec<f32>) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dbg_trace_rb"),
            size: (self.n_total.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dbg_trace_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.trace_pre, 0, &rb, 0, (self.n_total.max(1) * 4) as u64);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = rb.slice(..).get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&view);
        let pre = src[..self.n_total].to_vec();
        drop(view);
        rb.unmap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dbg_trace_copy2"),
        });
        encoder.copy_buffer_to_buffer(&self.trace_post, 0, &rb, 0, (self.n_total.max(1) * 4) as u64);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = rb.slice(..).get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&view);
        let post = src[..self.n_total].to_vec();
        drop(view);
        rb.unmap();
        (pre, post)
    }

    /// DEBUG: скачать ring-буфер (для диагностики scatter).
    pub fn debug_ring(&self) -> Vec<f32> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dbg_ring_rb"),
            size: (self.n_total * MAX_SLOTS as usize * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dbg_ring_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.ring, 0, &rb, 0, (self.n_total * MAX_SLOTS as usize * 4) as u64);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = rb.slice(..).get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&view);
        let out = src.to_vec();
        drop(view);
        rb.unmap();
        out
    }

    /// Знаки торможения пресинапсов (nt_types коннектома): +1 возбуждающий,
    /// −1 тормозной. Применяются в scatter как `ring += w·sign` — та же
    /// семантика, что у `run_lif_alpha` (без неё тормозные синапсы с
    /// положительными весами возбуждают пост-нейрон).
    pub fn set_nt_signs(&self, signs: &[f32]) -> Result<(), String> {
        if signs.len() != self.n_total {
            return Err(format!(
                "set_nt_signs: {} != n_total {}",
                signs.len(),
                self.n_total
            ));
        }
        self.ctx.queue.write_buffer(&self.signs, 0, cast_slice(signs));
        Ok(())
    }

    /// DEBUG: вход скеттера для пост-нейрона `post` — (src, вес, знак, задержка)
    /// в порядке rev-массивов (по возрастанию источника), как читает кернел.
    /// Веса — живой weights_b (скачивается), знаки — signs_b (скачивается).
    pub fn debug_scatter_row(&self, post: usize) -> Result<Vec<(u32, f32, f32, u32)>, String> {
        let w = self.weights()?;
        let signs = self.debug_signs()?;
        let mut out = Vec::new();
        let start = self.rev_ptr[post] as usize;
        let end = self.rev_ptr[post + 1] as usize;
        for k in start..end {
            let src = self.rev_src[k];
            let e = self.rev_edge[k] as usize;
            out.push((src, w[e], signs[src as usize], self.rev_d[k]));
        }
        Ok(out)
    }

    /// DEBUG: скачать знаки торможения (signs_b).
    pub fn debug_signs(&self) -> Result<Vec<f32>, String> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dbg_signs_rb"),
            size: (self.n_total * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dbg_signs_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.signs, 0, &rb, 0, (self.n_total * 4) as u64);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = rb.slice(..).get_mapped_range();
        let src: &[f32] = bytemuck::cast_slice(&view);
        let out = src.to_vec();
        drop(view);
        rb.unmap();
        Ok(out)
    }

    /// Внешний ток как СКАЧОК НАПРЯЖЕНИЯ (v += value) — семантика
    /// `run_lif_alpha` (external_I прямо в v), а не ring→i_in проводимости.
    /// `indices` — локальные индексы внутри популяции `pop` (без i_slice:
    /// states буфер pop-локальный). Используется диомой для сенсорного входа
    /// в alpha-lif мозг; `inject_current` остаётся для токовых моделей (LIF).
    pub fn inject_voltage(&mut self, pop: usize, indices: &[u32], values: &[f32]) -> Result<(), String> {
        if indices.len() != values.len() {
            return Err("inject_voltage: indices/values length mismatch".into());
        }
        if indices.is_empty() {
            return Ok(());
        }
        let p = self
            .pops
            .get(pop)
            .ok_or_else(|| format!("inject_voltage: pop {pop} out of range"))?;
        for &i in indices {
            assert!((i as u64) < p.n as u64, "inject_voltage: index {i} out of range");
        }
        let queue = &self.ctx.queue;
        queue.write_buffer(&self.inj_idx, 0, cast_slice(indices));
        queue.write_buffer(&self.inj_val, 0, cast_slice(values));
        // VJumpArgs { n, count, stride, _pad } — диспатч отложен в step().
        queue.write_buffer(
            &self.vjump_args,
            0,
            cast_slice(&[p.n as u32, indices.len() as u32, p.stride as u32, 0u32]),
        );
        self.vjump_pending[pop] = true;
        Ok(())
    }

    /// Дождаться завершения всех поданных команд (для батч-бенчмарков без
    /// ридбека: держит очередь пустой, не блокируя на каждом подшаге).
    pub fn wait_idle(&self) {
        self.ctx.device.poll(wgpu::Maintain::Wait);
    }

    /// Загрузить веса с CPU (рождение ребёнка, Болдуин, чекпоинты).
    pub fn set_weights(&self, weights: &[f32]) -> Result<(), String> {
        if weights.len() != self.n_syn {
            return Err(format!(
                "set_weights: {} != n_syn {}",
                weights.len(),
                self.n_syn
            ));
        }
        // Per-post скеттер ремапит in-edge → out-edge через rev_edge и читает
        // живой weights_b — отдельная синхронизация обратной копии не нужна.
        self.ctx.queue.write_buffer(&self.weights, 0, cast_slice(weights));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LIF_WGSL;

    /// CPU-эталон шага сети (зеркало launch_network_step): copy → pops → zero → scatter.
    /// Скеттерит спайки ПРОШЛОГО шага (`prev_spikes`) — семантика CPU
    /// run_lif_alpha (alpha_fired), которую воспроизводит GPU-сеть.
    fn cpu_step(
        s1: &mut [f32], s2: &mut [f32], p1: &[f32], p2: &[f32], n1: usize, n2: usize,
        stride1: usize, stride2: usize, dt: f32, ring: &mut [f32], slot: usize,
        row_ptr: &[u32], col_idx: &[u32], weights: &[f32], delays: &[u8],
        prev_spikes: &[u32],
    ) -> Vec<u32> {
        let n_total = n1 + n2;
        let mut i_in = vec![0.0f32; n_total];
        for i in 0..n_total {
            i_in[i] = ring[slot * n_total + i];
        }
        let mut spikes = vec![0u32; n_total];
        let step_pop = |s: &mut [f32], p: &[f32], stride: usize, n: usize, base: usize, i_in: &[f32], spikes: &mut [u32], dt: f32| {
            for i in 0..n {
                let v = s[i];
                let mut refr = s[stride + i];
                if refr > 0.0 {
                    refr = (refr - dt).max(0.0);
                    s[stride + i] = refr;
                    continue;
                }
                let dv = (p[2 * stride + i] - v + i_in[base + i]) * p[1 * stride + i] * p[0 * stride + i];
                let v_new = v + dv;
                if v_new >= p[3 * stride + i] {
                    spikes[base + i] = 1;
                    s[i] = p[4 * stride + i];
                    s[stride + i] = p[5 * stride + i];
                } else {
                    s[i] = v_new;
                    s[stride + i] = 0.0;
                }
            }
        };
        step_pop(s1, p1, stride1, n1, 0, &i_in, &mut spikes, dt);
        step_pop(s2, p2, stride2, n2, n1, &i_in, &mut spikes, dt);
        for i in 0..n_total {
            ring[slot * n_total + i] = 0.0;
        }
        for pre in 0..n_total {
            if prev_spikes[pre] == 0 {
                continue;
            }
            for k in row_ptr[pre] as usize..row_ptr[pre + 1] as usize {
                let post = col_idx[k] as usize;
                let d = delays[k].max(1) as usize;
                let tgt = (slot + d) % 32;
                ring[tgt * n_total + post] += weights[k];
            }
        }
        spikes
    }

    /// Полный parity-гейт на GPU: 2 популяции, кольцо с задержками, STDP, round-trip весов.
    /// Пропускается (не падает), если GPU-адаптера нет.
    #[test]
    fn gpu_network_parity() {
        let ctx = match WgpuContext::new_blocking() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("SKIP: no GPU adapter ({e})");
                return;
            }
        };
        let n1 = 64usize;
        let n2 = 32usize;
        let n_total = n1 + n2;
        let dt = 0.5f32;
        let stride1 = n1.div_ceil(64) * 64;
        let stride2 = n2.div_ceil(64) * 64;

        let mut s1 = vec![0.0f32; 2 * stride1];
        s1[0] = -54.6;
        let s2 = vec![0.0f32; 2 * stride2];
        let mut p1 = vec![0.0f32; 6 * stride1];
        let mut p2 = vec![0.0f32; 6 * stride2];
        for (p, stride) in [(&mut p1, stride1), (&mut p2, stride2)] {
            for i in 0..stride {
                p[0 * stride + i] = dt;
                p[1 * stride + i] = 0.1;
                p[2 * stride + i] = -70.0;
                p[3 * stride + i] = -54.0;
                p[4 * stride + i] = -70.0;
                p[5 * stride + i] = 2.0;
            }
        }

        let mut row_ptr = vec![0u32; n_total + 1];
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();
        let mut delays = Vec::new();
        for pre in 0..n_total {
            if pre < n1 {
                col_idx.push(((pre + 1) % n1) as u32);
                weights.push(3.0);
                delays.push(1u8);
                col_idx.push(pre as u32);
                weights.push(1.0);
                delays.push(3u8);
                if pre < n2 {
                    col_idx.push((n1 + pre) as u32);
                    weights.push(2.0);
                    delays.push(1u8);
                }
            }
            row_ptr[pre + 1] = col_idx.len() as u32;
        }
        let n_syn = col_idx.len();
        let pop1 = WgpuPop { wgsl: LIF_WGSL, entry: "lif_kernel_wgsl", n: n1, ns: 2, np: 6, states: s1.clone(), params: p1.clone() };
        let pop2 = WgpuPop { wgsl: LIF_WGSL, entry: "lif_kernel_wgsl", n: n2, ns: 2, np: 6, states: s2.clone(), params: p2.clone() };
        let pop1_spec = pop1.clone();
        let pop2_spec = pop2.clone();
        let mut net = WgpuNetwork::new(ctx, &[pop1, pop2], &row_ptr, &col_idx, &weights, &delays, dt)
            .expect("network init");

    /// Знаки торможения доезжают до scatter: сеть S(0)→I(1) возбуждающая,
        // ── 200 шагов динамики: суммы спайков бит-в-бит ──
        let steps = 200usize;
        let mut gpu_total = 0u32;
        let mut gpu_spikes = vec![0u32; n_total];
        let mut c_s1 = s1.clone();
        let mut c_s2 = s2.clone();
        let mut ring = vec![0.0f32; 32 * n_total];
        let mut slot = 0usize;
        let mut cpu_total = 0u32;
        let mut cpu_prev = vec![0u32; n_total];
        for _ in 0..steps {
            net.step().expect("step");
            net.download_spikes(&mut gpu_spikes).expect("download");
            gpu_total += gpu_spikes.iter().sum::<u32>();
            let sp = cpu_step(&mut c_s1, &mut c_s2, &p1, &p2, n1, n2, stride1, stride2, dt, &mut ring, slot, &row_ptr, &col_idx, &weights, &delays, &cpu_prev);
            cpu_total += sp.iter().sum::<u32>();
            cpu_prev = sp;
            slot = (slot + 1) % 32;
        }
        assert_eq!(gpu_total, cpu_total, "спайки GPU ≠ CPU (динамика сети)");
        // состояния после первой фазы (без STDP)
        let mut st1 = vec![0.0f32; 2 * stride1];
        net.download_states(0, &mut st1).expect("dl_states_phase1");
        let mut md = 0.0f32;
        for i in 0..2 * stride1 {
            md = md.max((st1[i] - c_s1[i]).abs());
        }
        assert_eq!(md, 0.0, "состояния pop0 после фазы 1 GPU ≠ CPU (max_diff={md})");

        // ── 60 шагов STDP: веса бит-в-бит ──
        let trace_decay = 0.9f32;
        let lr_ltp = 0.01f32;
        let lr_ltd = 0.005f32;
        let w_min = -5.0f32;
        let w_max = 5.0f32;
        // синхронный старт обеих сторон: GPU reset + свежие CPU-состояния
        net.reset(&[pop1_spec.clone(), pop2_spec.clone()]).expect("reset_stdp");
        let mut cpu_trace = vec![0.0f32; n_total];
        let mut cpu_weights = weights.clone();
        let mut cpu_ring = vec![0.0f32; 32 * n_total];
        let mut cpu_slot = 0usize;
        let mut c_s1 = s1.clone();
        let mut c_s2 = s2.clone();
        let mut cpu_prev = vec![0u32; n_total];
        for _ in 0..60 {
            net.step().expect("step2");
            net.download_spikes(&mut gpu_spikes).expect("dl2");
            let sp = cpu_step(&mut c_s1, &mut c_s2, &p1, &p2, n1, n2, stride1, stride2, dt, &mut cpu_ring, cpu_slot, &row_ptr, &col_idx, &cpu_weights, &delays, &cpu_prev);
            cpu_prev = sp;
            cpu_slot = (cpu_slot + 1) % 32;
            net.stdp_step(trace_decay, lr_ltp, lr_ltd, w_min, w_max).expect("stdp");
            for i in 0..n_total {
                cpu_trace[i] = cpu_trace[i] * trace_decay + if gpu_spikes[i] != 0 { 1.0 } else { 0.0 };
            }
            for pre in 0..n_total {
                let tp = cpu_trace[pre];
                for k in row_ptr[pre] as usize..row_ptr[pre + 1] as usize {
                    let post = col_idx[k] as usize;
                    let w = cpu_weights[k] + lr_ltp * tp - lr_ltd * cpu_trace[post];
                    cpu_weights[k] = w.clamp(w_min, w_max);
                }
            }
        }
        let gpu_weights = net.weights().expect("weights");
        let mut max_diff = 0.0f32;
        for k in 0..n_syn {
            max_diff = max_diff.max((gpu_weights[k] - cpu_weights[k]).abs());
        }
        assert_eq!(max_diff, 0.0, "STDP-веса GPU ≠ CPU (max_diff={max_diff})");

        // ── round-trip set_weights → weights ──
        net.set_weights(&cpu_weights).expect("set_weights");
        let rt = net.weights().expect("weights2");
        assert_eq!(rt, cpu_weights, "round-trip весов нарушен");

        // ── accessors: n_pops / pop_n / step_counter ──
        assert_eq!(net.n_pops(), 2);
        assert_eq!(net.pop_n(0), n1);
        assert_eq!(net.pop_n(1), n2);
        assert_eq!(net.pop_n(99), 0);
        assert_eq!(net.step_counter(), 60);

        // ── download_states: состояния pop0 совпадают с CPU-эталоном ──
        let mut gpu_st = vec![0.0f32; 2 * stride1];
        net.download_states(0, &mut gpu_st).expect("dl_states");
        let mut max_diff = 0.0f32;
        for i in 0..stride1 {
            max_diff = max_diff.max((gpu_st[i] - c_s1[i]).abs());
            max_diff = max_diff.max((gpu_st[stride1 + i] - c_s1[stride1 + i]).abs());
        }
        assert_eq!(max_diff, 0.0, "состояния pop0 GPU ≠ CPU (max_diff={max_diff})");

        // ── reset: состояния вернулись к начальным, счётчик обнулён ──
        net.reset(&[pop1_spec.clone(), pop2_spec.clone()]).expect("reset");
        assert_eq!(net.step_counter(), 0);
        assert_eq!(net.slot(), 0);
        let mut after = vec![0.0f32; 2 * stride1];
        net.download_states(0, &mut after).expect("dl_states2");
        let s1_init = &pop1_spec.states;
        let mut max_diff = 0.0f32;
        for i in 0..2 * stride1 {
            max_diff = max_diff.max((after[i] - s1_init[i]).abs());
        }
        assert_eq!(max_diff, 0.0, "reset не вернул состояния pop0 (max_diff={max_diff})");
        // после reset сеть снова оживает с теми же спайками за первые шаги
        let mut s_total = 0u32;
        for _ in 0..10 {
            net.step().expect("step_after_reset");
            net.download_spikes(&mut gpu_spikes).expect("dl_after_reset");
            s_total += gpu_spikes.iter().sum::<u32>();
        }
        assert!(s_total > 0, "сеть не ожила после reset");

        // ── dopamine-обучение: eligibility + reward против CPU-эталона ──
        let d_pre = (-dt / 20.0).exp();   // tau_pre = 20
        let d_post = (-dt / 10.0).exp();  // tau_post = 10
        let d_e = (-dt / 5.0).exp();      // tau_e = 5
        let a_ltp = 0.05f32;
        let a_ltd = 0.03f32;
        let lr = 0.15f32;
        let w_min = -5.0f32;
        let w_cap = 5.0f32;
        net.reset(&[pop1_spec.clone(), pop2_spec.clone()]).expect("reset_dopa");
        let mut cpu_trace_pre = vec![0.0f32; n_total];
        let mut cpu_trace_post = vec![0.0f32; n_total];
        let mut cpu_elig = vec![0.0f32; n_syn];
        // GPU-веса после STDP-фазы — обновлённые (reset веса не трогает),
        // CPU-эталон должен стартовать с тех же самых.
        let mut cpu_weights = cpu_weights.clone();
        let mut cpu_ring = vec![0.0f32; 32 * n_total];
        let mut cpu_slot = 0usize;
        let mut c_s1 = s1.clone();
        let mut c_s2 = s2.clone();
        let mut cpu_prev = vec![0u32; n_total];
        let mut changed = false;
        for t in 0..80 {
            net.step().expect("step_dopa");
            net.download_spikes(&mut gpu_spikes).expect("dl_dopa");
            let sp = cpu_step(&mut c_s1, &mut c_s2, &p1, &p2, n1, n2, stride1, stride2, dt, &mut cpu_ring, cpu_slot, &row_ptr, &col_idx, &cpu_weights, &delays, &cpu_prev);
            cpu_prev = sp;
            cpu_slot = (cpu_slot + 1) % 32;
            // eligibility (GPU и CPU синхронно, спайки — с GPU)
            net.eligibility_step(d_pre, d_post, d_e, a_ltp, a_ltd).expect("elig");
            for n in 0..n_total {
                let s = if gpu_spikes[n] != 0 { 1.0 } else { 0.0 };
                cpu_trace_pre[n] = cpu_trace_pre[n] * d_pre + s;
                cpu_trace_post[n] = cpu_trace_post[n] * d_post + s;
            }
            for k in 0..n_syn {
                // найти pre по CSR (k-й синапс)
                let mut pre = 0usize;
                for p in 0..n_total {
                    if (row_ptr[p] as usize) <= k && k < (row_ptr[p + 1] as usize) {
                        pre = p;
                        break;
                    }
                }
                let post = col_idx[k] as usize;
                let ltp = if gpu_spikes[post] != 0 { a_ltp * cpu_trace_pre[pre] } else { 0.0 };
                let ltd = if gpu_spikes[pre] != 0 { a_ltd * cpu_trace_post[post] } else { 0.0 };
                cpu_elig[k] = (cpu_elig[k] + ltp - ltd) * d_e;
            }
            // награда каждые 25 шагов
            if t % 25 == 24 {
                let dopamine = 2.0f32;
                net.reward(lr, dopamine, w_min, w_cap).expect("reward");
                for k in 0..n_syn {
                    let delta = lr * dopamine * cpu_elig[k];
                    if delta != 0.0 {
                        changed = true;
                        cpu_weights[k] = (cpu_weights[k] + delta).clamp(w_min, w_cap);
                    }
                }
            }
        }
        assert!(changed, "dopamine-обучение не изменило ни одного веса");
        let gpu_weights = net.weights().expect("weights_dopa");
        let mut max_diff = 0.0f32;
        for k in 0..n_syn {
            max_diff = max_diff.max((gpu_weights[k] - cpu_weights[k]).abs());
        }
        assert_eq!(max_diff, 0.0, "dopamine-веса GPU ≠ CPU (max_diff={max_diff})");

        // ── decay тихих шагов: eligibility затухает бит-в-бит без спайков ──
        // (диома: эпизод между наградами — следы не должны «висеть»).
        let mut c_tp = cpu_trace_pre.clone();
        let mut c_tpo = cpu_trace_post.clone();
        let mut c_elig = cpu_elig.clone();
        for _ in 0..10 {
            net.step().expect("quiet_step");
            net.download_spikes(&mut gpu_spikes).expect("quiet_dl");
            net.eligibility_step(d_pre, d_post, d_e, a_ltp, a_ltd).expect("quiet_elig");
            assert!(gpu_spikes.iter().all(|&s| s == 0), "ожидались тихие шаги");
            for n in 0..n_total {
                c_tp[n] *= d_pre;
                c_tpo[n] *= d_post;
            }
            for k in 0..n_syn {
                c_elig[k] *= d_e;
            }
        }
        // тихие шаги не должны добавлять спайки, но decay должен быть точен
        net.reward(lr, 1.0, w_min, w_cap).expect("quiet_reward");
        let gw = net.weights().expect("quiet_weights");
        let mut md2 = 0.0f32;
        for k in 0..n_syn {
            let cpu_w = (cpu_weights[k] + lr * 1.0 * c_elig[k]).clamp(w_min, w_cap);
            md2 = md2.max((gw[k] - cpu_w).abs());
        }
        assert_eq!(md2, 0.0, "decay-тихих шагов GPU ≠ CPU (max_diff={md2})");
    }

    /// Знаки торможения доезжают до scatter: сеть S(0)→I(1) возбуждающая,
    /// I(1)→M(2) тормозная (sign −1). Скачок на S даёт спайк S, затем M
    /// должен получить ОТРИЦАТЕЛЬНЫЙ вклад в ring (торможение), а не
    /// положительный.
    #[test]
    fn nt_signs_reach_scatter() {
        use crate::LIF_WGSL;
        let ctx = WgpuContext::new_blocking().expect("adapter");
        let n = 3usize; // S, I, M
        let stride = n.div_ceil(64) * 64; // сеть паддит stride до workgroup-размера
        // states: v=v_rest, refr=0 (LIF: refr=0 → не рефрактерен)
        let mut states = vec![0.0f32; 2 * stride];
        for i in 0..n {
            states[0 * stride + i] = -70.0;
        }
        let mut params = vec![0.0f32; 6 * stride];
        for i in 0..n {
            params[0 * stride + i] = 0.5;
            params[1 * stride + i] = 0.05;
            params[2 * stride + i] = -70.0;
            params[3 * stride + i] = -54.0;
            params[4 * stride + i] = -70.0;
            params[5 * stride + i] = 0.0;
        }
        // S→I вес 3.0, I→M вес 3.0
        let row_ptr = vec![0u32, 1, 2, 2];
        let col_idx = vec![1u32, 2];
        let weights = vec![3.0f32, 3.0];
        let delays = vec![1u8, 1];
        let pop = WgpuPop { wgsl: LIF_WGSL, entry: "lif_kernel_wgsl", n, ns: 2, np: 6, states, params };
        let mut net = WgpuNetwork::new(ctx, &[pop], &row_ptr, &col_idx, &weights, &delays, 0.5).expect("net");
        net.set_nt_signs(&[1.0, -1.0, 1.0]).expect("signs");
        // Скеттер работает со спайками ПРОШЛОГО шага (как CPU alpha_fired):
        // S спайкает на шаге 1, вклад S→I попадает в ring на шаге 2 и
        // доезжает до I на шаге 3.
        net.inject_voltage(0, &[0], &[25.0]).expect("vjump");
        net.step().expect("step1"); // S спайкает; prev пуст → скеттер молчит
        let ring = net.debug_ring();
        let s0: f32 = ring[1 * n..1 * n + n].iter().sum();
        assert_eq!(s0, 0.0, "на шаге 1 скеттер пуст (prev=0), got {s0}");
        net.step().expect("step2"); // скеттер prev=[S] → ring[2][I] += 3.0·(+1)
        let ring = net.debug_ring();
        let s1: f32 = ring[2 * n + 1];
        assert_eq!(s1, 3.0, "S→I возбуждающий вклад должен быть +3.0, got {s1}");
        // Толкаем I над порогом: v = −70 + 30 = −40 > −54 → спайк на шаге 3.
        net.inject_voltage(0, &[1], &[30.0]).expect("vjump_i");
        net.step().expect("step3"); // I получает +3 (i_in) и спайкает
        net.step().expect("step4"); // скеттер prev=[I] → ring[4][M] += 3.0·(−1)
        let ring = net.debug_ring();
        let s2: f32 = ring[4 * n + 2];
        assert_eq!(s2, -3.0, "I→M должен тормозить (−3.0), got {s2}");
    }
}
