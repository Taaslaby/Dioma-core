//! GPEA (biophysical insect neuron, Belmabrouk 2011 / Higuchi 2022) — full
//! GPU worker for the FlyWire connectome.
//!
//! Mirrors the CPU path in `pillars` (`NeuralPopulation<GpeaModel>`, validated
//! against NEURON 9.0.2 in `zoo/neurons/gpea.rs`) as a persistent device
//! worker with a captured CUDA graph, exactly like [`super::alpha_lif`]:
//!
//! ```text
//! step t (captured graph, two variants for the spike double-buffer):
//!   1. external inject  — ring[ring_pos][i] += ext[i]   (guarded by a device flag)
//!   2. neuron step      — gpea_step (same #[dual_neuron] source as CPU)
//!                          + zero the ring slot it just read + count spikes
//!   3. scatter          — ring[ring_pos][post] = Σ w[pre→post]·spike[pre] (atomic push)
//!   4. advance          — ring_pos/step scalars bumped for the next replay
//! ```
//!
//! # Input current semantics
//!
//! The ring slot holds `i_syn` in **µA/cm²** — the GPEA membrane equation is
//! `v_new = (Cm/dt·v + i_syn + Σg·E)/(Cm/dt + Σg)` — so the CSR `values` are
//! per-synapse current jumps in the same units (current-based synapses, like
//! the alpha-LIF worker; the CPU HH path's conductance kinetics is a separate
//! concern). External `ext[i]` is constant current in µA/cm² (the way the
//! flywire_hh CPU experiment drives the sugar GRNs with 20 µA/cm²).
//!
//! # Parity model
//!
//! - **Neuron** — bit-identical: reuses `GPEA_CUDA_SOURCE` (proven vs CPU SIMD
//!   in `pillars-cuda/tests/gpea_nvrtc.rs`: exact spike raster over 100 ms of
//!   1100 Hz firing, rest-phase |Δv| < 2e-2). NVRTC is compiled with
//!   `--fmad=false` so `a*b+c` stays a separate mul+add like the Rust `f32x8`
//!   path.
//! - **Scatter** — atomic spike-driven push (`O(fired_edges)`), NOT bit-exact
//!   (same policy as `AlphaLifGpu` / PyTorch / NEST / GeNN). The parity tests
//!   assert spike-count and membrane-potential *tolerance* parity.

use std::ffi::c_void;

use crate::engine::{
    check_cu, compile_source_function, device_context, module_function, CUDA_CALL_LOCK,
};
use crate::ffi::{
    driver, CUdeviceptr, CUfunction, CUgraph, CUgraphExec, CUmodule, CUstream,
    CU_GRAPH_INSTANTIATE_DEFAULT, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL, CU_STREAM_DEFAULT,
};
use pillars::zoo::neurons::GPEA_CUDA_SOURCE;

const BLOCK: u32 = 256;

/// GpeaModel: 9 states (v, m, h, n, p, q, cai, r, was_below), 14 params.
const N_STATE: usize = 9;
const N_PARAM: usize = 14;

/// Device args layout. Indices 3 (`ring_pos`) and 4 (`step`) are advanced
/// on-device by the `gpea_advance_kernel` at the end of each graph replay;
/// index 6 (`has_ext`) is set once per `run_steps` call; the rest are constant.
const ARG_DT: usize = 0;
const ARG_N: usize = 1;
const ARG_STRIDE: usize = 2;
const ARG_RING_POS: usize = 3;
const ARG_STEP: usize = 4;
const ARG_SIZE: usize = 5;
const ARG_HAS_EXT: usize = 6;
const N_ARGS: usize = 7;

/// Auxiliary `extern "C"` kernels compiled alongside `GPEA_CUDA_SOURCE`.
///
/// All kernels take a shared `const float* args` device buffer for the per-step
/// scalars, so a captured CUDA graph keeps FIXED pointers and re-reads the
/// scalars each replay (with `ring_pos`/`step` advanced on-device).
const AUX_SOURCE: &str = r#"
// ── neuron + count (wrapper around the generated gpea_step) ──
// Reads the delayed i_syn from the ring slot (ring_pos from args), zeroes that
// slot (so the scatter's atomicAdd starts clean), steps the neuron, counts.
extern "C" __global__ void gpea_runner_kernel(
    float* states, const float* params, float* ring, int* spikes, const float* args,
    unsigned char* active, int* total
) {
    const float dt = args[0];
    const int n = (int)args[1];
    const int stride = (int)args[2];
    const int ring_pos = (int)args[3];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float S[9];
    #pragma unroll
    for (int k = 0; k < 9; k++) S[k] = states[k * stride + i];
    float P[14];
    #pragma unroll
    for (int k = 0; k < 14; k++) P[k] = params[k * stride + i];
    float* slot = ring + (size_t)ring_pos * (size_t)n;
    float i_syn = slot[i];
    slot[i] = 0.0f;   // reset the slot for this step's scatter accumulation
    float spike = 0.0f;
    gpea_step(S, P, i_syn, dt, &spike);
    #pragma unroll
    for (int k = 0; k < 9; k++) states[k * stride + i] = S[k];
    int spk = (spike != 0.0f) ? 1 : 0;
    spikes[i] = spk;
    if (spk) { active[i] = 1; atomicAdd(total, 1); }
}

// ── spike-driven atomic scatter: ring[ring_pos][post] += w[pre→post] ──
// Consumes the PREVIOUS step's spike buffer (double-buffer parity).
extern "C" __global__ void gpea_scatter_kernel(
    const int* row_ptr, const int* col_indices, const float* values,
    const int* spikes, float* ring, const float* args
) {
    const int n = (int)args[1];
    const int ring_pos = (int)args[3];
    int pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (pre >= n) return;
    if (spikes[pre] == 0) return;
    float* slot = ring + (size_t)ring_pos * (size_t)n;
    int start = row_ptr[pre];
    int end = row_ptr[pre + 1];
    for (int k = start; k < end; ++k) {
        atomicAdd(&slot[col_indices[k]], values[k]);
    }
}

// ── External current inject into the CURRENT ring slot (µA/cm², i_syn
//    semantics — runs BEFORE the runner reads the slot). No-op when
//    args[6] == 0. ──
extern "C" __global__ void gpea_ext_kernel(float* ring, const float* ext, const float* args) {
    if (args[6] == 0.0f) return;   // ext disabled for this run → skip entirely
    const int n = (int)args[1];
    const int ring_pos = (int)args[3];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ring[(size_t)ring_pos * (size_t)n + i] += ext[i];
}

// ── advance: bump ring_pos / step for the next replay (single thread) ──
// Runs last in the graph (after every reader), so the next replay sees the
// updated scalars. `size` (ring slots) lives at args[5].
extern "C" __global__ void gpea_advance_kernel(float* args) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int size = (int)args[5];
    int ring_pos = (int)args[3];
    args[3] = (float)((ring_pos + 1) % size);
    args[4] += 1.0f;
}
"#;

/// Model parameters — the NEURON-validated GPEA values at 24 °C (Drosophila).
///
/// Conductance maxima are per-type tunable in the fly brain (PNs/LNs differ);
/// kinetics constants are the published `pGPeA_k.mod` values (see
/// `pillars::zoo::neurons::gpea`).
#[derive(Debug, Clone, Copy)]
pub struct GpeaParams {
    pub dt_ms: f32,
    pub celsius: f32,
    pub inv_cm: f32,
    pub v_thresh: f32,
    pub area: f32, // µm² — Ca-dynamics scaling (reference geometry π·40·20)
    pub gnabar: f32, // mS/cm²
    pub gkdrbar: f32, // mS/cm²
    pub gl: f32,      // mS/cm²
    pub el: f32,      // mV
    pub gcatbar: f32, // mS/cm²
    pub gkcabar: f32, // mS/cm²
    pub kca: f32,     // 1/ms
    pub vol: f32,     // L
    pub ca_gain: f32,
    pub t_delay: f32, // ms — synaptic delay (ring size)
}

impl GpeaParams {
    /// NEURON-validated defaults (validation trace in `zoo/neurons/gpea.rs`).
    pub fn flywire() -> Self {
        Self {
            dt_ms: 0.025,
            celsius: 24.0,
            inv_cm: 1.0,
            v_thresh: -20.0,
            area: 2513.2741228718345,
            gnabar: 49.0,
            gkdrbar: 57.0,
            gl: 0.35,
            el: -60.0,
            gcatbar: 5.0,
            gkcabar: 1.0,
            kca: 2.0,
            vol: 3.355e-11,
            ca_gain: 0.1,
            t_delay: 1.8,
        }
    }
}

/// Result of a GPU run — same shape as the CPU worker output.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpeaOutput {
    pub total_spikes: u64,
    pub active_neurons: u64,
    pub brain_ms: f32,
    pub wall_seconds: f64,
}

/// Persistent GPU worker for the biophysical GPEA fly brain. Device buffers
/// stay in VRAM across `run_steps` calls (always-on worker).
pub struct GpeaGpu {
    // kernels (one module, kept alive for the kernels' lifetime)
    _module: CUmodule,

    // device buffers
    d_states: CUdeviceptr,      // N_STATE * stride * 4
    d_params: CUdeviceptr,      // N_PARAM * stride * 4
    d_spikes: [CUdeviceptr; 2], // double-buffered spikes (n * 4 each)
    d_ring: CUdeviceptr,        // size * n * 4
    d_args: CUdeviceptr,        // N_ARGS * 4
    d_row_ptr: CUdeviceptr,     // (n + 1) * 4
    d_col_indices: CUdeviceptr, // E * 4
    d_values: CUdeviceptr,      // E * 4
    d_ext: CUdeviceptr,         // n * 4
    d_active: CUdeviceptr,      // n bytes
    d_total: CUdeviceptr,       // 4 bytes

    // captured graphs (one per spike-buffer parity), replayed on the NULL stream
    graph_execs: [CUgraphExec; 2],

    // host-side geometry / state
    n: usize,
    stride: usize,
    size: usize, // ring slots = steps_delay + 1
    spike_cur: usize,
    dt: f32,
    init_states: [f32; N_STATE],

    brain_ms: f32,
}

impl GpeaGpu {
    /// Build the worker from raw CSR connectome arrays + model params.
    ///
    /// `row_ptr`/`col_indices` are the CSR (pre → post) graph; `values` are the
    /// signed per-synapse current jumps in µA/cm² (used verbatim in the
    /// scatter). `t_delay` from `params` sizes the ring.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        row_ptr: &[i32],
        col_indices: &[i32],
        values: &[f32],
        params: GpeaParams,
    ) -> Result<Self, String> {
        let n = row_ptr.len().checked_sub(1).ok_or("empty row_ptr")?;
        if n == 0 {
            return Err("GpeaGpu: connectome has 0 neurons".into());
        }
        if col_indices.len() != values.len() {
            return Err("GpeaGpu: col_indices/values length mismatch".into());
        }

        // stride = CPU NeuralPopulation capacity (round up to a multiple of 8).
        let stride = n.saturating_add(7) / 8 * 8;
        let steps_delay = (params.t_delay / params.dt_ms).round().max(0.0) as usize;
        let size = steps_delay + 1;

        // Packed param columns in GpeaModel #[param] order:
        // dt, celsius, inv_cm, v_thresh, area, gnabar, gkdrbar, gl, el,
        // gcatbar, gkcabar, kca, vol, caGain — uniform across the population.
        let mut flat_params = vec![0.0f32; N_PARAM * stride];
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
        for c in 0..N_PARAM {
            let col = &mut flat_params[c * stride..(c + 1) * stride];
            for i in 0..n {
                col[i] = vals[c];
            }
        }

        // Initial state — NEURON finitialize IC (see make_pop in zoo/neurons):
        // V=-60, gates at steady state, cai=5e-5 (WRITE-ion reset), r=r_inf,
        // was_below=1. Column-major SoA.
        let init = [
            -60.0, 0.0414, 0.906, 0.2165, 0.355, 0.01325, 5e-5, 0.1821261, 1.0,
        ];
        let mut flat_state = vec![0.0f32; N_STATE * stride];
        for c in 0..N_STATE {
            let col = &mut flat_state[c * stride..(c + 1) * stride];
            for i in 0..n {
                col[i] = init[c];
            }
        }

        // ── Compile one NVRTC module: neuron source + aux kernels ──
        let source = format!("{GPEA_CUDA_SOURCE}\n\n{AUX_SOURCE}");
        let (module, runner_kernel) = compile_source_function(&source, "gpea_runner_kernel")?;
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let scatter_kernel = module_function(module, "gpea_scatter_kernel")?;
        let ext_kernel = module_function(module, "gpea_ext_kernel")?;
        let advance_kernel = module_function(module, "gpea_advance_kernel")?;

        // ── Allocate + upload device buffers ──
        let mut args_host = [0.0f32; N_ARGS];
        args_host[ARG_DT] = params.dt_ms;
        args_host[ARG_N] = n as f32;
        args_host[ARG_STRIDE] = stride as f32;
        args_host[ARG_RING_POS] = (size - 1) as f32; // initial ring read slot
        args_host[ARG_STEP] = 0.0;
        args_host[ARG_SIZE] = size as f32;
        args_host[ARG_HAS_EXT] = 0.0;

        let grid = (n as u32).div_ceil(BLOCK);

        unsafe {
            let d_states = alloc(d, N_STATE * stride * 4)?;
            let d_params = alloc(d, N_PARAM * stride * 4)?;
            let d_spikes = [alloc(d, n * 4)?, alloc(d, n * 4)?];
            let d_ring = alloc(d, size * n * 4)?;
            let d_args = alloc(d, N_ARGS * 4)?;
            let d_row_ptr = alloc(d, (n + 1) * 4)?;
            let d_col_indices = alloc(d, col_indices.len() * 4)?;
            let d_values = alloc(d, values.len() * 4)?;
            let d_ext = alloc(d, n * 4)?;
            let d_active = alloc(d, n)?;
            let d_total = alloc(d, 4)?;

            upload(d, d_states, &flat_state)?;
            upload(d, d_params, &flat_params)?;
            // Zero the spike buffers: the first scatter consumes the *previous*
            // step's buffer, which must start empty.
            let zeros_spikes = vec![0i32; n];
            upload(d, d_spikes[0], &zeros_spikes)?;
            upload(d, d_spikes[1], &zeros_spikes)?;
            let zeros_ring = vec![0.0f32; size * n];
            upload(d, d_ring, &zeros_ring)?;
            upload(d, d_args, &args_host)?;
            upload(d, d_row_ptr, row_ptr)?;
            upload(d, d_col_indices, col_indices)?;
            upload(d, d_values, values)?;
            let zeros_ext = vec![0.0f32; n];
            upload(d, d_ext, &zeros_ext)?;

            // Capture the per-step sequence into two graphs (spike double-buffer
            // parity: graph[0] writes d_spikes[0] / reads d_spikes[1]; graph[1]
            // vice-versa). Captured on one temporary stream, replayed on NULL.
            let mut cap_stream: CUstream = std::ptr::null_mut();
            check_cu(d, (d.cu_stream_create)(&mut cap_stream, CU_STREAM_DEFAULT))
                .map_err(|e| format!("capture stream create: {e}"))?;

            let graph_execs = [
                capture_graph(
                    d,
                    cap_stream,
                    runner_kernel,
                    scatter_kernel,
                    ext_kernel,
                    advance_kernel,
                    d_states,
                    d_params,
                    d_ring,
                    d_args,
                    d_active,
                    d_total,
                    d_row_ptr,
                    d_col_indices,
                    d_values,
                    d_ext,
                    d_spikes[0],
                    d_spikes[1],
                    grid,
                )?,
                capture_graph(
                    d,
                    cap_stream,
                    runner_kernel,
                    scatter_kernel,
                    ext_kernel,
                    advance_kernel,
                    d_states,
                    d_params,
                    d_ring,
                    d_args,
                    d_active,
                    d_total,
                    d_row_ptr,
                    d_col_indices,
                    d_values,
                    d_ext,
                    d_spikes[1],
                    d_spikes[0],
                    grid,
                )?,
            ];
            let _ = (d.cu_stream_destroy)(cap_stream);

            Ok(Self {
                _module: module,
                d_states,
                d_params,
                d_spikes,
                d_ring,
                d_args,
                d_row_ptr,
                d_col_indices,
                d_values,
                d_ext,
                d_active,
                d_total,
                graph_execs,
                n,
                stride,
                size,
                spike_cur: 0,
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

    /// Run `steps` brain steps, optionally adding `external_i` (µA/cm²) to the
    /// current ring slot every step (length `n` or empty). State persists
    /// between calls.
    pub fn run_steps(&mut self, steps: usize, external_i: &[f32]) -> Result<GpeaOutput, String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = std::time::Instant::now();

        unsafe {
            // Normalize + upload the external current (or zeros), and set the
            // device flag the ext kernel checks. Done once per call.
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

            // Zero the run's metrics accumulators.
            upload(d, self.d_total, &[0i32; 1])?;
            let zeros = vec![0u8; self.n];
            upload(d, self.d_active, &zeros)?;

            // Pure async replay loop: the advance kernel bumps ring_pos/step
            // on-device, so the host never stalls per step. Both graphs
            // alternate on the NULL stream (FIFO) → strict per-step ordering.
            for _ in 0..steps {
                check_cu(
                    d,
                    (d.cu_graph_launch)(
                        self.graph_execs[self.spike_cur],
                        std::ptr::null_mut(), // legacy NULL stream (FIFO)
                    ),
                )?;
                self.spike_cur = 1 - self.spike_cur;
            }
            // Hard boundary before the metrics readback.
            check_cu(d, (d.cu_ctx_synchronize)())?;

            // Download metrics.
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
        if col >= N_STATE {
            return Err(format!("GpeaGpu: state column {col} out of range"));
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

    /// Reset state to the NEURON finitialize IC, ring/spikes zeroed, ring
    /// pointer rewound.
    pub fn reset(&mut self) -> Result<(), String> {
        let d = driver().as_ref().map_err(|e| e.clone())?;
        device_context(d)?;
        let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            let mut flat_state = vec![0.0f32; N_STATE * self.stride];
            for c in 0..N_STATE {
                let col = &mut flat_state[c * self.stride..(c + 1) * self.stride];
                for i in 0..self.n {
                    col[i] = self.init_states[c];
                }
            }
            upload(d, self.d_states, &flat_state)?;
            let zeros_ring = vec![0.0f32; self.size * self.n];
            upload(d, self.d_ring, &zeros_ring)?;
            let zeros_spikes = vec![0i32; self.n];
            upload(d, self.d_spikes[0], &zeros_spikes)?;
            upload(d, self.d_spikes[1], &zeros_spikes)?;
            // Rewind the device-side per-step scalars.
            let ring_reset = [(self.size - 1) as f32];
            upload(
                d,
                self.d_args + (ARG_RING_POS * 4) as CUdeviceptr,
                &ring_reset,
            )?;
            let step_reset = [0.0f32];
            upload(d, self.d_args + (ARG_STEP * 4) as CUdeviceptr, &step_reset)?;
        }
        self.spike_cur = 0;
        self.brain_ms = 0.0;
        Ok(())
    }
}

impl Drop for GpeaGpu {
    fn drop(&mut self) {
        if let Ok(d) = driver().as_ref() {
            if device_context(d).is_ok() {
                let _lock = CUDA_CALL_LOCK.lock().unwrap_or_else(|p| p.into_inner());
                unsafe {
                    for &p in &[
                        self.d_states,
                        self.d_params,
                        self.d_spikes[0],
                        self.d_spikes[1],
                        self.d_ring,
                        self.d_args,
                        self.d_row_ptr,
                        self.d_col_indices,
                        self.d_values,
                        self.d_ext,
                        self.d_active,
                        self.d_total,
                    ] {
                        let _ = (d.cu_mem_free)(p);
                    }
                    for &exec in &self.graph_execs {
                        let _ = (d.cu_graph_exec_destroy)(exec);
                    }
                    let _ = (d.cu_module_unload)(self._module);
                }
            }
        }
    }
}

// ── CUDA graph capture ─────────────────────────────────────────────

/// Capture the per-step kernel sequence `[ext, runner, scatter, advance]` into
/// a CUDA graph on `stream`, using the fixed device pointers. The per-step
/// scalars (`ring_pos`, `step`) live in `d_args` and are advanced on-device by
/// the trailing `advance` kernel.
///
/// `spikes_write` is where the runner writes this step's spikes; `spikes_read`
/// is the *previous* step's spikes that the scatter consumes (the double-buffer
/// parity flips per step — hence two captured graphs).
///
/// The `ext` kernel is always captured; whether it does anything is controlled
/// by the `args[6]` flag at replay time.
#[allow(clippy::too_many_arguments)]
unsafe fn capture_graph(
    d: &crate::ffi::Driver,
    stream: CUstream,
    runner_kernel: CUfunction,
    scatter_kernel: CUfunction,
    ext_kernel: CUfunction,
    advance_kernel: CUfunction,
    d_states: CUdeviceptr,
    d_params: CUdeviceptr,
    d_ring: CUdeviceptr,
    d_args: CUdeviceptr,
    d_active: CUdeviceptr,
    d_total: CUdeviceptr,
    d_row_ptr: CUdeviceptr,
    d_col_indices: CUdeviceptr,
    d_values: CUdeviceptr,
    d_ext: CUdeviceptr,
    spikes_write: CUdeviceptr,
    spikes_read: CUdeviceptr,
    grid: u32,
) -> Result<CUgraphExec, String> {
    if let Err(e) = check_cu(
        d,
        (d.cu_stream_begin_capture)(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
    ) {
        return Err(format!("graph begin_capture: {e}"));
    }

    // 1. External inject (always captured; no-op when args[6] == 0).
    {
        let mut rg = d_ring;
        let mut ex = d_ext;
        let mut ar = d_args;
        let mut a: [*mut c_void; 3] = [
            &mut rg as *mut CUdeviceptr as *mut c_void,
            &mut ex as *mut CUdeviceptr as *mut c_void,
            &mut ar as *mut CUdeviceptr as *mut c_void,
        ];
        let rc = (d.cu_launch_kernel)(
            ext_kernel,
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
                "capture ext: {}",
                crate::engine::driver_error(d, rc)
            ));
        }
    }

    // 2. Runner (neuron + zero slot + count).
    {
        let mut st = d_states;
        let mut pr = d_params;
        let mut rg = d_ring;
        let mut sp = spikes_write;
        let mut ar = d_args;
        let mut ac = d_active;
        let mut tt = d_total;
        let mut a: [*mut c_void; 7] = [
            &mut st as *mut CUdeviceptr as *mut c_void,
            &mut pr as *mut CUdeviceptr as *mut c_void,
            &mut rg as *mut CUdeviceptr as *mut c_void,
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

    // 3. Scatter (atomic push) — reads previous spikes.
    {
        let mut rp = d_row_ptr;
        let mut ci = d_col_indices;
        let mut vl = d_values;
        let mut sp = spikes_read;
        let mut rg = d_ring;
        let mut ar = d_args;
        let mut a: [*mut c_void; 6] = [
            &mut rp as *mut CUdeviceptr as *mut c_void,
            &mut ci as *mut CUdeviceptr as *mut c_void,
            &mut vl as *mut CUdeviceptr as *mut c_void,
            &mut sp as *mut CUdeviceptr as *mut c_void,
            &mut rg as *mut CUdeviceptr as *mut c_void,
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

    // 4. Advance ring_pos/step for the next replay (single thread, runs last).
    {
        let mut ar = d_args;
        let mut a: [*mut c_void; 1] = [&mut ar as *mut CUdeviceptr as *mut c_void];
        let rc = (d.cu_launch_kernel)(
            advance_kernel,
            1,
            1,
            1,
            1,
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
                "capture advance: {}",
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
