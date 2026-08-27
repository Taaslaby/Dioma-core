//! GPEA with **conductance-based synapses** on GPU — the full biophysics.
//!
//! Mirrors the CPU `hh_synaptic` path in `pillars-connectome` (see
//! `sim/hh.rs::sim_hh_synaptic_pillars` + `kernels/synapse/kinetics.rs`):
//!
//! ```text
//! step t (captured graph):
//!   1. decay      — g_r[i] *= exp(-dt/τ_r)        for all 7 receptor types
//!   2. current    — i_syn[i] = ext[i] + Σ_r g_r[i]·(E_rev_r − V[i])
//!                   (NMDA carries the Mg²⁺ voltage block B(V) = 1/(1+0.33·e^(−0.06V)))
//!   3. neuron     — gpea_step (same #[dual_neuron] source as CPU)
//!   4. scatter    — g[receptor][post] += values[edge]  for each fired pre
//!                   (immediate jump — the CPU conductance path has no delay)
//! ```
//!
//! Unlike the current-based [`super::gpea::GpeaGpu`], there is no delay ring
//! and no gain hack: the CSR `values` are **conductances** used verbatim, and
//! each synapse carries a `receptor_types[u8]` (AMPA/NMDA/GABA_A/GABA_B/nAChR/
//! GluCl/mAChR — built via `ConnectomeData::build_receptor_types*`). The
//! neuron kernel is bit-identical to the CPU (`GPEA_CUDA_SOURCE`); the scatter
//! is an atomic push (tolerance parity, like `AlphaLifGpu`).

use std::ffi::c_void;

use crate::engine::{
    check_cu, compile_source_function, device_context, module_function, CUDA_CALL_LOCK,
};
use crate::ffi::{
    driver, CUdeviceptr, CUfunction, CUgraph, CUgraphExec, CUmodule, CUstream,
    CU_GRAPH_INSTANTIATE_DEFAULT, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, CU_STREAM_DEFAULT,
};
use pillars::zoo::neurons::GPEA_CUDA_SOURCE;

use super::gpea::{GpeaOutput, GpeaParams};

const BLOCK: u32 = 256;

/// Number of receptor types (AMPA, NMDA, GABA_A, GABA_B, nAChR, GluCl, mAChR).
const N_RECEPTORS: usize = 7;

/// Device args layout.
const ARG_DT: usize = 0;
const ARG_N: usize = 1;
const ARG_STRIDE: usize = 2;
const ARG_HAS_EXT: usize = 3;
const N_ARGS: usize = 4;

const AUX_SOURCE: &str = r#"
// ── decay: g_r[i] *= factor_r (7 receptor columns) ──
extern "C" __global__ void gpea_syn_decay_kernel(
    float* g, const float* factors, const float* args)
{
    const int n = (int)args[1];
    const int stride = (int)args[2];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    #pragma unroll
    for (int r = 0; r < 7; r++) g[r * stride + i] *= factors[r];
}

// ── synaptic current: i_syn[i] = ext[i] + Σ_r g_r[i]·(E_rev_r − V[i]) ──
// Reversal potentials (mV): AMPA=0, NMDA=0, GABA_A=−65, GABA_B=−90,
// nAChR=0, GluCl=−70, mAChR=−90. NMDA (r=1) carries the voltage-dependent
// Mg²⁺ block B(V) = 1/(1+0.33·exp(−0.06·V)) — same as the CPU path.
extern "C" __global__ void gpea_syn_current_kernel(
    const float* g, const float* states, const float* ext, float* i_syn, const float* args)
{
    const int n = (int)args[1];
    const int stride = (int)args[2];
    const int has_ext = (int)args[3];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = states[i]; // column 0 = v
    float cur = has_ext ? ext[i] : 0.0f;
    float gampa  = g[0 * stride + i];
    float gnmda  = g[1 * stride + i];
    float ggaba_a = g[2 * stride + i];
    float ggaba_b = g[3 * stride + i];
    float gnachr = g[4 * stride + i];
    float gglucl = g[5 * stride + i];
    float gmachr = g[6 * stride + i];
    float nmda_arg = fminf(fmaxf(-0.06f * v, -10.0f), 10.0f);
    float nmda_b = 1.0f / (1.0f + 0.33f * expf(nmda_arg));
    cur += gampa   * (0.0f   - v);
    cur += gnmda   * nmda_b * (0.0f - v);
    cur += ggaba_a * (-65.0f - v);
    cur += ggaba_b * (-90.0f - v);
    cur += gnachr  * (0.0f   - v);
    cur += gglucl  * (-70.0f - v);
    cur += gmachr  * (-90.0f - v);
    i_syn[i] = cur;
}

// ── neuron step (reads the i_syn buffer) + count ──
extern "C" __global__ void gpea_syn_runner_kernel(
    float* states, const float* params, const float* i_syn, int* spikes, const float* args,
    unsigned char* active, int* total)
{
    const float dt = args[0];
    const int n = (int)args[1];
    const int stride = (int)args[2];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float S[9];
    #pragma unroll
    for (int k = 0; k < 9; k++) S[k] = states[k * stride + i];
    float P[14];
    #pragma unroll
    for (int k = 0; k < 14; k++) P[k] = params[k * stride + i];
    float spike = 0.0f;
    gpea_step(S, P, i_syn[i], dt, &spike);
    #pragma unroll
    for (int k = 0; k < 9; k++) states[k * stride + i] = S[k];
    int spk = (spike != 0.0f) ? 1 : 0;
    spikes[i] = spk;
    if (spk) { active[i] = 1; atomicAdd(total, 1); }
}

// ── immediate conductance jump: g[receptor][post] += values[edge] ──
// Reads the CURRENT step's spikes (stream-ordered after the runner — the CPU
// conductance path jumps in the same step, no delay).
extern "C" __global__ void gpea_syn_scatter_kernel(
    const int* row_ptr, const int* col_indices, const float* values,
    const unsigned char* receptor_types, const int* spikes, float* g, const float* args)
{
    const int n = (int)args[1];
    const int stride = (int)args[2];
    int pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre >= n) return;
    if (spikes[pre] == 0) return;
    int start = row_ptr[pre];
    int end = row_ptr[pre + 1];
    for (int k = start; k < end; ++k) {
        int post = col_indices[k];
        int r = (int)receptor_types[k];
        atomicAdd(&g[r * stride + post], values[k]);
    }
}
"#;

/// Persistent GPU worker for GPEA with conductance-based synapses.
///
/// Device buffers stay in VRAM across `run_steps` calls. No delay ring: the
/// CPU `hh_synaptic` path jumps conductances in the same step as the spike, so
/// this worker does the same (single spike buffer, stream-ordered scatter).
pub struct GpeaGpuSynaptic {
    _module: CUmodule,

    // device buffers
    d_states: CUdeviceptr,          // 9 * stride * 4
    d_params: CUdeviceptr,          // 14 * stride * 4
    d_g: CUdeviceptr,               // 7 * stride * 4 (conductance columns)
    d_i_syn: CUdeviceptr,           // stride * 4
    d_spikes: CUdeviceptr,          // n * 4
    d_args: CUdeviceptr,            // N_ARGS * 4
    d_factors: CUdeviceptr,         // 7 * 4 (decay factors exp(-dt/τ_r))
    d_row_ptr: CUdeviceptr,         // (n + 1) * 4
    d_col_indices: CUdeviceptr,     // E * 4
    d_values: CUdeviceptr,          // E * 4
    d_receptor_types: CUdeviceptr,  // E bytes
    d_ext: CUdeviceptr,             // n * 4
    d_active: CUdeviceptr,          // n bytes
    d_total: CUdeviceptr,           // 4 bytes

    graph_exec: CUgraphExec,

    // host-side geometry / state
    n: usize,
    stride: usize,
    dt: f32,
    init_states: [f32; 9],

    brain_ms: f32,
}

impl GpeaGpuSynaptic {
    /// Build the worker from raw CSR connectome arrays + per-synapse receptor
    /// types (one `u8` per edge: AMPA=0, NMDA=1, GABA_A=2, GABA_B=3, nAChR=4,
    /// GluCl=5, mAChR=6) + model params. `values` are conductances used
    /// verbatim (scale 1.0 — like the CPU `hh_synaptic` path).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_ptr: &[i32],
        col_indices: &[i32],
        values: &[f32],
        receptor_types: &[u8],
        params: GpeaParams,
    ) -> Result<Self, String> {
        let n = row_ptr.len().checked_sub(1).ok_or("empty row_ptr")?;
        if n == 0 {
            return Err("GpeaGpuSynaptic: connectome has 0 neurons".into());
        }
        if col_indices.len() != values.len() {
            return Err("GpeaGpuSynaptic: col_indices/values length mismatch".into());
        }
        if receptor_types.len() != values.len() {
            return Err("GpeaGpuSynaptic: receptor_types must be per-synapse".into());
        }

        let stride = n.saturating_add(7) / 8 * 8;

        // Packed param columns (GpeaModel #[param] order) — uniform.
        let mut flat_params = vec![0.0f32; 14 * stride];
        let vals = [
            params.dt_ms,
            params.celsius,
            params.inv_cm,
            params.v_thresh,
            params.area,
            params.gnabar,
            params.gkdrbar,
            params.gl,
            params.el,
            params.gcatbar,
            params.gkcabar,
            params.kca,
            params.vol,
            params.ca_gain,
        ];
        for c in 0..14 {
            let col = &mut flat_params[c * stride..(c + 1) * stride];
            for i in 0..n {
                col[i] = vals[c];
            }
        }

        // Initial state — NEURON finitialize IC (see make_pop in zoo/neurons).
        let init = [
            -60.0, 0.0414, 0.906, 0.2165, 0.355, 0.01325, 5e-5, 0.1821261, 1.0,
        ];
        let mut flat_state = vec![0.0f32; 9 * stride];
        for c in 0..9 {
            let col = &mut flat_state[c * stride..(c + 1) * stride];
            for i in 0..n {
                col[i] = init[c];
            }
        }

        // Decay factors exp(-dt/τ_r) — same τ as kernels/synapse/receptor.rs.
        let taus = [2.0f32, 50.0, 10.0, 200.0, 3.0, 5.0, 150.0]; // AMPA..mAChR
        let factors: Vec<f32> = taus.iter().map(|&t| (-params.dt_ms / t).exp()).collect();

        // ── Compile one NVRTC module ──
        let source = format!("{GPEA_CUDA_SOURCE}\n\n{AUX_SOURCE}");
        let (module, runner_kernel) = compile_source_function(&source, "gpea_syn_runner_kernel")?;
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let decay_kernel = module_function(module, "gpea_syn_decay_kernel")?;
        let current_kernel = module_function(module, "gpea_syn_current_kernel")?;
        let scatter_kernel = module_function(module, "gpea_syn_scatter_kernel")?;

        let mut args_host = [0.0f32; N_ARGS];
        args_host[ARG_DT] = params.dt_ms;
        args_host[ARG_N] = n as f32;
        args_host[ARG_STRIDE] = stride as f32;
        args_host[ARG_HAS_EXT] = 0.0;

        let grid = (n as u32).div_ceil(BLOCK);

        unsafe {
            let d_states = alloc(d, 9 * stride * 4)?;
            let d_params = alloc(d, 14 * stride * 4)?;
            let d_g = alloc(d, N_RECEPTORS * stride * 4)?;
            let d_i_syn = alloc(d, stride * 4)?;
            let d_spikes = alloc(d, n * 4)?;
            let d_args = alloc(d, N_ARGS * 4)?;
            let d_factors = alloc(d, 7 * 4)?;
            let d_row_ptr = alloc(d, (n + 1) * 4)?;
            let d_col_indices = alloc(d, col_indices.len() * 4)?;
            let d_values = alloc(d, values.len() * 4)?;
            let d_receptor_types = alloc(d, receptor_types.len())?;
            let d_ext = alloc(d, n * 4)?;
            let d_active = alloc(d, n)?;
            let d_total = alloc(d, 4)?;

            upload(d, d_states, &flat_state)?;
            upload(d, d_params, &flat_params)?;
            let zeros_g = vec![0.0f32; N_RECEPTORS * stride];
            upload(d, d_g, &zeros_g)?;
            let zeros_i = vec![0.0f32; stride];
            upload(d, d_i_syn, &zeros_i)?;
            let zeros_spikes = vec![0i32; n];
            upload(d, d_spikes, &zeros_spikes)?;
            upload(d, d_args, &args_host)?;
            upload(d, d_factors, &factors)?;
            upload(d, d_row_ptr, row_ptr)?;
            upload(d, d_col_indices, col_indices)?;
            upload(d, d_values, values)?;
            upload(d, d_receptor_types, receptor_types)?;
            let zeros_ext = vec![0.0f32; n];
            upload(d, d_ext, &zeros_ext)?;

            // Capture the per-step sequence [decay, current, runner, scatter]
            // into one graph (immediate propagation — no double buffer).
            let mut cap_stream: CUstream = std::ptr::null_mut();
            check_cu(d, (d.cu_stream_create)(&mut cap_stream, CU_STREAM_DEFAULT))
                .map_err(|e| format!("capture stream create: {e}"))?;

            let graph_exec = capture_graph(
                d,
                cap_stream,
                decay_kernel,
                current_kernel,
                runner_kernel,
                scatter_kernel,
                d_states,
                d_params,
                d_g,
                d_i_syn,
                d_args,
                d_factors,
                d_active,
                d_total,
                d_row_ptr,
                d_col_indices,
                d_values,
                d_receptor_types,
                d_ext,
                d_spikes,
                grid,
            )?;
            let _ = (d.cu_stream_destroy)(cap_stream);

            Ok(Self {
                _module: module,
                d_states,
                d_params,
                d_g,
                d_i_syn,
                d_spikes,
                d_args,
                d_factors,
                d_row_ptr,
                d_col_indices,
                d_values,
                d_receptor_types,
                d_ext,
                d_active,
                d_total,
                graph_exec,
                n,
                stride,
                dt: params.dt_ms,
                init_states: init,
                brain_ms: 0.0,
            })
        }
    }

    /// Number of neurons.
    pub fn n_neurons(&self) -> usize {
        self.n
    }

    /// Run `steps` brain steps, optionally adding `external_i` (µA/cm²) to
    /// `i_syn` every step (length `n` or empty). State persists between calls.
    pub fn run_steps(&mut self, steps: usize, external_i: &[f32]) -> Result<GpeaOutput, String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = std::time::Instant::now();

        unsafe {
            let has_ext = !external_i.is_empty();
            let ext_flag = [if has_ext { 1.0f32 } else { 0.0f32 }];
            upload(d, self.d_args + (ARG_HAS_EXT * 4) as CUdeviceptr, &ext_flag)?;
            let mut ext = vec![0.0f32; self.n];
            if has_ext {
                for (i, slot) in ext.iter_mut().enumerate() {
                    *slot = external_i.get(i).copied().unwrap_or(0.0);
                }
            }
            upload(d, self.d_ext, &ext)?;

            upload(d, self.d_total, &[0i32; 1])?;
            let zeros = vec![0u8; self.n];
            upload(d, self.d_active, &zeros)?;

            for _ in 0..steps {
                check_cu(
                    d,
                    (d.cu_graph_launch)(
                        self.graph_exec,
                        std::ptr::null_mut(), // legacy NULL stream (FIFO)
                    ),
                )?;
            }
            check_cu(d, (d.cu_ctx_synchronize)())?;

            let mut total = [0i32; 1];
            download(d, &mut total, self.d_total)?;
            let mut active = vec![0u8; self.n];
            download(d, &mut active, self.d_active)?;
            let active_count = active.iter().filter(|&&a| a != 0).count();

            let wall = t0.elapsed().as_secs_f64();
            let brain_ms = steps as f32 * self.dt;
            self.brain_ms += brain_ms;
            Ok(GpeaOutput {
                total_spikes: total[0].max(0) as u64,
                active_neurons: active_count as u64,
                brain_ms,
                wall_seconds: wall,
            })
        }
    }

    /// Run `seconds` seconds of brain time.
    pub fn run_seconds(&mut self, seconds: f32) -> Result<GpeaOutput, String> {
        let steps = (seconds * 1000.0 / self.dt).round() as usize;
        self.run_steps(steps, &[])
    }

    /// Download a state column (0 = v, 6 = cai, ...) — for parity checks.
    pub fn download_state(&self, col: usize) -> Result<Vec<f32>, String> {
        if col >= 9 {
            return Err(format!("GpeaGpuSynaptic: state column {col} out of range"));
        }
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut v = vec![0.0f32; self.stride];
        unsafe {
            download(
                d,
                &mut v,
                self.d_states + (col * self.stride * 4) as CUdeviceptr,
            )?;
        }
        v.truncate(self.n);
        Ok(v)
    }

    /// Download the membrane potentials (for parity checks).
    pub fn download_v(&self) -> Result<Vec<f32>, String> {
        self.download_state(0)
    }

    /// Reset state to the NEURON finitialize IC, conductances/spikes zeroed.
    pub fn reset(&mut self) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            let mut flat_state = vec![0.0f32; 9 * self.stride];
            for c in 0..9 {
                let col = &mut flat_state[c * self.stride..(c + 1) * self.stride];
                for i in 0..self.n {
                    col[i] = self.init_states[c];
                }
            }
            upload(d, self.d_states, &flat_state)?;
            let zeros_g = vec![0.0f32; N_RECEPTORS * self.stride];
            upload(d, self.d_g, &zeros_g)?;
            let zeros_spikes = vec![0i32; self.n];
            upload(d, self.d_spikes, &zeros_spikes)?;
        }
        self.brain_ms = 0.0;
        Ok(())
    }
}

impl Drop for GpeaGpuSynaptic {
    fn drop(&mut self) {
        if let Ok(d) = driver().as_ref() {
            if device_context(d).is_ok() {
                let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                unsafe {
                    for &p in &[
                        self.d_states,
                        self.d_params,
                        self.d_g,
                        self.d_i_syn,
                        self.d_spikes,
                        self.d_args,
                        self.d_factors,
                        self.d_row_ptr,
                        self.d_col_indices,
                        self.d_values,
                        self.d_receptor_types,
                        self.d_ext,
                        self.d_active,
                        self.d_total,
                    ] {
                        let _ = (d.cu_mem_free)(p);
                    }
                    let _ = (d.cu_graph_exec_destroy)(self.graph_exec);
                    let _ = (d.cu_module_unload)(self._module);
                }
            }
        }
    }
}

// ── CUDA graph capture ─────────────────────────────────────────────

/// Capture the per-step kernel sequence `[decay, current, runner, scatter]`
/// into a CUDA graph on `stream`, using the fixed device pointers.
#[allow(clippy::too_many_arguments)]
unsafe fn capture_graph(
    d: &crate::ffi::Driver,
    stream: CUstream,
    decay_kernel: CUfunction,
    current_kernel: CUfunction,
    runner_kernel: CUfunction,
    scatter_kernel: CUfunction,
    d_states: CUdeviceptr,
    d_params: CUdeviceptr,
    d_g: CUdeviceptr,
    d_i_syn: CUdeviceptr,
    d_args: CUdeviceptr,
    d_factors: CUdeviceptr,
    d_active: CUdeviceptr,
    d_total: CUdeviceptr,
    d_row_ptr: CUdeviceptr,
    d_col_indices: CUdeviceptr,
    d_values: CUdeviceptr,
    d_receptor_types: CUdeviceptr,
    d_ext: CUdeviceptr,
    d_spikes: CUdeviceptr,
    grid: u32,
) -> Result<CUgraphExec, String> {
    if let Err(e) = check_cu(
        d,
        (d.cu_stream_begin_capture)(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
    ) {
        return Err(format!("graph begin_capture: {e}"));
    }

    // 1. Decay.
    {
        let mut gg = d_g;
        let mut ff = d_factors;
        let mut ar = d_args;
        let mut a: [*mut c_void; 3] = [
            &mut gg as *mut CUdeviceptr as *mut c_void,
            &mut ff as *mut CUdeviceptr as *mut c_void,
            &mut ar as *mut CUdeviceptr as *mut c_void,
        ];
        let rc = (d.cu_launch_kernel)(
            decay_kernel,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            a.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != 0 {
            let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
            return Err(format!(
                "capture decay: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // 2. Current (+ ext).
    {
        let mut gg = d_g;
        let mut st = d_states;
        let mut ex = d_ext;
        let mut is = d_i_syn;
        let mut ar = d_args;
        let mut a: [*mut c_void; 5] = [
            &mut gg as *mut CUdeviceptr as *mut c_void,
            &mut st as *mut CUdeviceptr as *mut c_void,
            &mut ex as *mut CUdeviceptr as *mut c_void,
            &mut is as *mut CUdeviceptr as *mut c_void,
            &mut ar as *mut CUdeviceptr as *mut c_void,
        ];
        let rc = (d.cu_launch_kernel)(
            current_kernel,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            a.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != 0 {
            let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
            return Err(format!(
                "capture current: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // 3. Runner.
    {
        let mut st = d_states;
        let mut pr = d_params;
        let mut is = d_i_syn;
        let mut sp = d_spikes;
        let mut ar = d_args;
        let mut ac = d_active;
        let mut tt = d_total;
        let mut a: [*mut c_void; 7] = [
            &mut st as *mut CUdeviceptr as *mut c_void,
            &mut pr as *mut CUdeviceptr as *mut c_void,
            &mut is as *mut CUdeviceptr as *mut c_void,
            &mut sp as *mut CUdeviceptr as *mut c_void,
            &mut ar as *mut CUdeviceptr as *mut c_void,
            &mut ac as *mut CUdeviceptr as *mut c_void,
            &mut tt as *mut CUdeviceptr as *mut c_void,
        ];
        let rc = (d.cu_launch_kernel)(
            runner_kernel,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            a.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != 0 {
            let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
            return Err(format!(
                "capture runner: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // 4. Scatter (immediate conductance jump, reads current spikes).
    {
        let mut rp = d_row_ptr;
        let mut ci = d_col_indices;
        let mut vl = d_values;
        let mut rt = d_receptor_types;
        let mut sp = d_spikes;
        let mut gg = d_g;
        let mut ar = d_args;
        let mut a: [*mut c_void; 7] = [
            &mut rp as *mut CUdeviceptr as *mut c_void,
            &mut ci as *mut CUdeviceptr as *mut c_void,
            &mut vl as *mut CUdeviceptr as *mut c_void,
            &mut rt as *mut CUdeviceptr as *mut c_void,
            &mut sp as *mut CUdeviceptr as *mut c_void,
            &mut gg as *mut CUdeviceptr as *mut c_void,
            &mut ar as *mut CUdeviceptr as *mut c_void,
        ];
        let rc = (d.cu_launch_kernel)(
            scatter_kernel,
            grid,
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream,
            a.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if rc != 0 {
            let _ = (d.cu_stream_end_capture)(stream, std::ptr::null_mut());
            return Err(format!(
                "capture scatter: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // End capture + instantiate.
    let mut graph: CUgraph = std::ptr::null_mut();
    if let Err(e) = check_cu(d, (d.cu_stream_end_capture)(stream, &mut graph)) {
        return Err(format!("graph end_capture: {e}"));
    }
    let mut exec: CUgraphExec = std::ptr::null_mut();
    let rc = (d.cu_graph_instantiate)(&mut exec, graph, CU_GRAPH_INSTANTIATE_DEFAULT);
    let _ = (d.cu_graph_destroy)(graph);
    if rc != 0 {
        return Err(format!(
            "graph instantiate: {}",
            crate::engine::driver_error(d, rc)
        ));
    }

    Ok(exec)
}

// ── helpers ────────────────────────────────────────────────────────

unsafe fn alloc(d: &crate::ffi::Driver, bytes: usize) -> Result<CUdeviceptr, String> {
    let mut p: CUdeviceptr = 0;
    check_cu(d, (d.cu_mem_alloc)(&mut p, bytes))?;
    Ok(p)
}

unsafe fn upload<T: Sized>(
    d: &crate::ffi::Driver,
    dst: CUdeviceptr,
    src: &[T],
) -> Result<(), String> {
    check_cu(
        d,
        (d.cu_memcpy_htod)(
            dst,
            src.as_ptr() as *const c_void,
            std::mem::size_of_val(src),
        ),
    )
}

unsafe fn download<T: Sized>(
    d: &crate::ffi::Driver,
    dst: &mut [T],
    src: CUdeviceptr,
) -> Result<(), String> {
    check_cu(
        d,
        (d.cu_memcpy_dtoh)(
            dst.as_mut_ptr() as *mut c_void,
            src,
            std::mem::size_of_val(dst),
        ),
    )
}
