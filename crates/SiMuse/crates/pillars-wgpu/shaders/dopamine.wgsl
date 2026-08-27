// dopamine.wgsl — STDP + R-STDP (dopamine-обучение) кернелы.
// STDP-часть бит-в-бит pillars-cuda network.rs STDP_SOURCE; dopamine-часть
// бит-в-бит pillars::api::learn::brain (traces → eligibility → reward).

// ── STDP (bit-identical to pillars-cuda network.rs STDP_SOURCE) ──
struct TraceArgs { n: u32, decay: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read_write> trace: array<f32>;
@group(0) @binding(1) var<storage, read> spikes: array<u32>;
@group(0) @binding(2) var<uniform> targs: TraceArgs;

// trace = trace * decay + spike (cuda trace_update_kernel)
@compute @workgroup_size(64)
fn trace_update_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= targs.n) { return; }
    trace[i] = trace[i] * targs.decay + select(0.0, 1.0, spikes[i] != 0u);
}

struct StdpArgs { n: u32, _pad0: u32, lr_ltp: f32, lr_ltd: f32, w_min: f32, w_max: f32 };
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_idx: array<u32>;
@group(0) @binding(3) var<storage, read> trace_in: array<f32>;
@group(0) @binding(4) var<uniform> sargs: StdpArgs;

// w[k] = clamp(w[k] + lr_ltp * trace[pre] - lr_ltd * trace[post]) (cuda csr_stdp_kernel)
@compute @workgroup_size(64)
fn csr_stdp_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pre = gid.x; if (pre >= sargs.n) { return; }
    let start = row_ptr[pre];
    let end = row_ptr[pre + 1u];
    let tp = trace_in[pre];
    for (var k = start; k < end; k = k + 1u) {
        let post = col_idx[k];
        if (post >= sargs.n) { continue; }
        let tn = trace_in[post];
        let w = weights[k] + sargs.lr_ltp * tp - sargs.lr_ltd * tn;
        weights[k] = clamp(w, sargs.w_min, sargs.w_max);
    }
}

// ── dopamine-обучение (R-STDP с per-synapse eligibility, бит-в-бит
//    pillars::api::learn::brain): traces → eligibility → reward ──

// trace_pre/post[n] = trace * d + spike (два отдельных tau, как brain.rs step)
struct DopaTraceArgs { n: u32, d_pre: f32, d_post: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> tp_pre: array<f32>;
@group(0) @binding(1) var<storage, read_write> tp_post: array<f32>;
@group(0) @binding(2) var<storage, read> spikes_in: array<u32>;
@group(0) @binding(3) var<uniform> dtargs: DopaTraceArgs;

// ЯВНЫЙ fma: `tp * d + spike` компилятор сворачивает в fused-умножение
// (одно округление), а rust без fast-math — нет → 1 ulp дрейфа трасс →
// расхождение eligibility → ложные Δw. CPU (pillars brain.rs) считает те же
// узлы через `mul_add` — обе стороны гарантированно fma (IEEE-определён).
@compute @workgroup_size(64)
fn dopa_traces_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x; if (n >= dtargs.n) { return; }
    let add = select(0.0, 1.0, spikes_in[n] != 0u);
    tp_pre[n] = fma(tp_pre[n], dtargs.d_pre, add);
    tp_post[n] = fma(tp_post[n], dtargs.d_post, add);
}

// e[k] = (e[k] + A_ltp·tp_pre[pre]·spike[post] − A_ltd·tp_post[post]·spike[pre])·d_e
// per-synapse, dense по всем синапсам (бит-в-бит ленивой версии — decay
// распределён по шагам, но произведение констант то же).
//
// Порядок сложения LTP/LTD ВАЖЕН для бит-в-бит: ленивый кернел (pillars
// brain.rs) ставит события в порядке индексов нейронов — для синапса с
// pre=p, post=q: если p<q, LTD-событие (нейрон p как PRE) попадает раньше
// LTP (нейрон q как POST) → e − ltd + ltp; иначе e + ltp − ltd. Воспроизводим
// ту же последовательность операций, иначе f32-rounding разведёт пути.
struct EligArgs { n_syn: u32, d_e: f32, a_ltp: f32, a_ltd: f32 };
@group(0) @binding(0) var<storage, read_write> elig: array<f32>;
@group(0) @binding(1) var<storage, read> pre_of: array<u32>;
@group(0) @binding(2) var<storage, read> col_idx_e: array<u32>;
@group(0) @binding(3) var<storage, read> tp_pre_e: array<f32>;
@group(0) @binding(4) var<storage, read> tp_post_e: array<f32>;
@group(0) @binding(5) var<storage, read> spikes_e: array<u32>;
@group(0) @binding(6) var<uniform> eargs: EligArgs;

@compute @workgroup_size(64)
fn elig_update_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x; if (k >= eargs.n_syn) { return; }
    let pre = pre_of[k];
    let post = col_idx_e[k];
    var ltp = 0.0;
    if (spikes_e[post] != 0u) { ltp = eargs.a_ltp * tp_pre_e[pre]; }
    var ltd = 0.0;
    if (spikes_e[pre] != 0u) { ltd = eargs.a_ltd * tp_post_e[post]; }
    var e_new: f32;
    if (pre < post) {
        // LTD-событие раньше LTP (как в ленивом кернеле при p<q).
        e_new = (elig[k] - ltd) + ltp;
    } else {
        e_new = (elig[k] + ltp) - ltd;
    }
    elig[k] = e_new * eargs.d_e;
}

// ── моторные суммы: спайки регионов [ml_lo, ml_hi) → acc[0],
//    [mr_lo, mr_hi) → acc[1] (u32 atomicAdd — сумма целых детерминирована
//    независимо от порядка инвокаций). Обнуляется вызывающим на старте тика.
struct MotorArgs { n: u32, ml_lo: u32, ml_hi: u32, mr_lo: u32, mr_hi: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read_write> motor_acc: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> spikes_m: array<u32>;
@group(0) @binding(2) var<uniform> margs: MotorArgs;

@compute @workgroup_size(64)
fn motor_sum_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= margs.n) { return; }
    if (spikes_m[i] == 0u) { return; }
    if (i >= margs.ml_lo && i < margs.ml_hi) { atomicAdd(&motor_acc[0], 1u); }
    if (i >= margs.mr_lo && i < margs.mr_hi) { atomicAdd(&motor_acc[1], 1u); }
}

// Fused: трассы + моторные суммы в одном кернеле (этап 2 — меньше лаунчей
// на тик). Оба итерируют по нейронам и читают одни спайки; layout —
// суперпозиция dopa_traces (bind 0-3) и motor (bind 4-6).
@group(0) @binding(0) var<storage, read_write> tp_pre_f: array<f32>;
@group(0) @binding(1) var<storage, read_write> tp_post_f: array<f32>;
@group(0) @binding(2) var<storage, read> spikes_f: array<u32>;
@group(0) @binding(3) var<uniform> dtargs_f: DopaTraceArgs;
@group(0) @binding(4) var<storage, read_write> motor_acc_f: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> margs_f: MotorArgs;

@compute @workgroup_size(64)
fn traces_motor_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x; if (n >= dtargs_f.n) { return; }
    let add = select(0.0, 1.0, spikes_f[n] != 0u);
    tp_pre_f[n] = fma(tp_pre_f[n], dtargs_f.d_pre, add);
    tp_post_f[n] = fma(tp_post_f[n], dtargs_f.d_post, add);
    if (add == 1.0) {
        if (n >= margs_f.ml_lo && n < margs_f.ml_hi) { atomicAdd(&motor_acc_f[0], 1u); }
        if (n >= margs_f.mr_lo && n < margs_f.mr_hi) { atomicAdd(&motor_acc_f[1], 1u); }
    }
}

// ── рестрикт eligibility: keep[k]==0 → elig[k]=0 (заморозка/прожигание) ──
struct RestrictArgs { n_syn: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(0) var<storage, read_write> elig_rs: array<f32>;
@group(0) @binding(1) var<storage, read> keep_m: array<u32>;
@group(0) @binding(2) var<uniform> rargs_r: RestrictArgs;

@compute @workgroup_size(64)
fn restrict_elig_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x; if (k >= rargs_r.n_syn) { return; }
    if (keep_m[k] == 0u) { elig_rs[k] = 0.0; }
}

// Δw = lr·dopamine·e[k], клэмп [w_min, w_cap]; Δw == 0 пропускается
// (врождённые веса не трогаем — как brain.rs reward_dopamine).
struct RewardArgs { n_syn: u32, lr_dopa: f32, w_min: f32, w_cap: f32 };
@group(0) @binding(0) var<storage, read_write> weights_r: array<f32>;
@group(0) @binding(1) var<storage, read> elig_r: array<f32>;
@group(0) @binding(2) var<uniform> rargs: RewardArgs;

@compute @workgroup_size(64)
fn reward_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x; if (k >= rargs.n_syn) { return; }
    let delta = rargs.lr_dopa * elig_r[k];
    if (delta == 0.0) { return; }
    weights_r[k] = clamp(weights_r[k] + delta, rargs.w_min, rargs.w_cap);
}
