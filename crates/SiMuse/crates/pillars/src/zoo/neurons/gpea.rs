//! GPEA — biophysical insect neuron model (Na⁺/Kdr/CaT/AHP + Ca dynamics)
//!
//! Port of `pGPeA_k.mod` — the single-compartment channel set used for the
//! **Drosophila whole-brain multi-compartment HH simulation** (Higuchi et al.,
//! bioRxiv 2022.11.01.512969, Kazawa lab). Originally from the moth antennal
//! lobe GPEA model (Belmabrouk et al., PNAS 2011).
//!
//! # Channels
//! ```text
//! I_Na  = gNa·m³·h·(V-E_Na)        fast sodium (sigmoid steady states)
//! I_Kdr = gKdr·n⁴·(V-E_K)          delayed rectifier
//! I_CaT = gCaT·p²·q·(V-E_Ca)       T-type calcium (pacemaking)
//! I_AHP = gKCa·r²·(V-E_K)          Ca-dependent K (SK-like), r = r(cai)
//! I_L   = gL·(V-E_L)               leak
//! ```
//!
//! # Calcium dynamics (NEURON cnexp, exact port)
//! ```text
//! dcai/dt = caGain·(-I_Ca·area·1e-11/(2F·vol) - kca·cai)
//! ```
//! `I_Ca` is frozen from the previous step (NEURON cnexp semantics).
//!
//! # Temperature
//! GPEA kinetics have **no Q10 scaling** — temperature enters only through
//! the Nernst potentials with `T = 273 + celsius − 9.5` (mod convention).
//! Default `celsius = 24` = Drosophila.
//!
//! # Validation
//! Rate functions and REST/PULSE V(t) traces validated against NEURON 9.0.2
//! running the original `pGPeA_k.mod` compiled with `nrnivmodl`.
//!
//! # Architecture
//! Same as [`super::hh`]: `#[dual_neuron]` single source → SIMD/scalar/CUDA,
//! gates via exponential Euler (cnexp), membrane via implicit Euler, edge
//! spike detection.

#![allow(non_snake_case)]

use pillars_macros::dual_neuron;

#[dual_neuron]
pub mod gpea {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use crate::ops::edge_spike;
    use pillars_macros::NeuronModel;

    // ═══════════════════════════════════════════════════════════════
    //  Channel constants — verbatim from pGPeA_k.mod
    // ═══════════════════════════════════════════════════════════════

    // Fast Na⁺:  gNa = gnabar·m³·h
    pub const NA_TH_M: f32 = -38.0;
    pub const NA_TH_H: f32 = -45.5;
    pub const NA_K_M: f32 = -7.0;
    pub const NA_K_H: f32 = 6.4;
    pub const NA_TAU_M0: f32 = 0.001;
    pub const NA_TAU_M1: f32 = 0.1;
    pub const NA_THT_M: f32 = -53.0;
    pub const NA_SIG_M: f32 = -0.7;
    pub const NA_TAU_H0: f32 = 0.0;
    pub const NA_TAU_H1: f32 = 4.5;
    pub const NA_THT_H1: f32 = -50.0;
    pub const NA_SIG_H1: f32 = -15.0;
    pub const NA_THT_H2: f32 = -50.0;
    pub const NA_SIG_H2: f32 = 16.0;

    // Delayed K rectifier:  gKdr = gkdrbar·n⁴
    pub const K_TH_N: f32 = -42.0;
    pub const K_K_N: f32 = -14.0;
    pub const K_TAU_N0: f32 = 0.0;
    pub const K_TAU_N1: f32 = 2.4;
    pub const K_THT_N1: f32 = -40.0;
    pub const K_SIG_N1: f32 = -40.0;
    pub const K_THT_N2: f32 = -40.0;
    pub const K_SIG_N2: f32 = 50.0;

    // Leak
    #[allow(dead_code)] // test-only reference value (param `el` carries it at runtime)
    pub const LEAK_EL: f32 = -60.0;

    // T-type Ca²⁺:  gCaT = gcatbar·p²·q
    pub const CAT_TH_P: f32 = -56.0;
    pub const CAT_TH_Q: f32 = -85.0;
    pub const CAT_K_P: f32 = -6.7;
    pub const CAT_K_Q: f32 = 5.8;
    pub const CAT_TAU_P0: f32 = 5.0;
    pub const CAT_TAU_P1: f32 = 0.33;
    pub const CAT_TAU_Q0: f32 = 0.0;
    pub const CAT_TAU_Q1: f32 = 400.0;
    pub const CAT_THT_P1: f32 = -27.0;
    pub const CAT_SIG_P1: f32 = -10.0;
    pub const CAT_THT_P2: f32 = -102.0;
    pub const CAT_SIG_P2: f32 = 15.0;
    pub const CAT_THT_Q1: f32 = -50.0;
    pub const CAT_SIG_Q1: f32 = -15.0;
    pub const CAT_THT_Q2: f32 = -50.0;
    pub const CAT_SIG_Q2: f32 = 16.0;

    // AHP (Ca-dependent K, SK-like):  gKCa = gkcabar·r²
    pub const AHP_TH_R: f32 = 0.17e-3;
    pub const AHP_K_R: f32 = -0.08e-3;
    pub const AHP_TAU_R: f32 = 2.0;
    // (test-only reference values — the runtime values come from params)
    #[allow(dead_code)]
    pub const AHP_POWER: f32 = 2.0;
    #[allow(dead_code)]
    pub const CA_GAIN: f32 = 0.1;
    #[allow(dead_code)]
    pub const CA_KCA: f32 = 2.0;
    pub const CA_FARADAY: f32 = 96485.0;
    #[allow(dead_code)]
    pub const CA_VOL: f32 = 3.355e-11;
    #[allow(dead_code)]
    pub const CA_AREA: f32 = 2_513.274_2; // π·40·20 µm² (reference geometry)

    // Ionic concentrations (global in the fly model) — mM
    pub const NA_NAI: f32 = 10.0;
    pub const NA_NAO: f32 = 140.0;
    pub const K_KI: f32 = 140.0;
    pub const K_KO: f32 = 5.0;
    pub const CA_CAO: f32 = 2.0;
    pub const GAS_R: f32 = 8.31441;

    // ═══════════════════════════════════════════════════════════════
    //  Gate helpers — exact transcription of evaluate_fct()
    // ═══════════════════════════════════════════════════════════════

    /// 1/(1+exp((x-th)/k)) — the mod's steady-state sigmoid.
    #[inline(always)]
    fn sigmoid<S: SimdLane>(x: S, th: f32, k: f32) -> S {
        S::one() / (S::one() + ((x - S::splat(th)) * S::splat(1.0 / k)).fast_exp())
    }

    /// Exponential Euler: x' = x_inf + (x - x_inf)·exp(-dt/τ).
    #[inline(always)]
    fn exp_euler<S: SimdLane>(x: S, x_inf: S, tau: S, dt: S) -> S {
        let decay = (-dt / tau).fast_exp();
        decay.mul_add(x - x_inf, x_inf)
    }

    /// τm = τm0 + τm1/(1+exp(-(v−tht_m)/sig_m))
    #[inline(always)]
    fn tau_m<S: SimdLane>(v: S) -> S {
        let denom = S::one() + ((v - S::splat(NA_THT_M)) * S::splat(-1.0 / NA_SIG_M)).fast_exp();
        S::splat(NA_TAU_M0) + S::splat(NA_TAU_M1) / denom
    }

    /// τh = τh0 + τh1/(exp(-(v−tht_h1)/sig_h1) + exp(-(v−tht_h2)/sig_h2))
    #[inline(always)]
    fn tau_h<S: SimdLane>(v: S) -> S {
        let e1 = ((v - S::splat(NA_THT_H1)) * S::splat(-1.0 / NA_SIG_H1)).fast_exp();
        let e2 = (-(v - S::splat(NA_THT_H2)) * S::splat(1.0 / NA_SIG_H2)).fast_exp();
        S::splat(NA_TAU_H0) + S::splat(NA_TAU_H1) / (e1 + e2)
    }

    /// τn = τn0 + τn1/(exp(-(v−tht_n1)/sig_n1) + exp(-(v−tht_n2)/sig_n2))
    #[inline(always)]
    fn tau_n<S: SimdLane>(v: S) -> S {
        let e1 = ((v - S::splat(K_THT_N1)) * S::splat(-1.0 / K_SIG_N1)).fast_exp();
        let e2 = (-(v - S::splat(K_THT_N2)) * S::splat(1.0 / K_SIG_N2)).fast_exp();
        S::splat(K_TAU_N0) + S::splat(K_TAU_N1) / (e1 + e2)
    }

    /// τp = τp0 + τp1/(exp(-(v−tht_p1)/sig_p1) + exp(-(v−tht_p2)/sig_p2))
    #[inline(always)]
    fn tau_p<S: SimdLane>(v: S) -> S {
        let e1 = ((v - S::splat(CAT_THT_P1)) * S::splat(-1.0 / CAT_SIG_P1)).fast_exp();
        let e2 = (-(v - S::splat(CAT_THT_P2)) * S::splat(1.0 / CAT_SIG_P2)).fast_exp();
        S::splat(CAT_TAU_P0) + S::splat(CAT_TAU_P1) / (e1 + e2)
    }

    /// τq = τq0 + τq1/(exp(-(v−tht_q1)/sig_q1) + exp(-(v−tht_q2)/sig_q2))
    #[inline(always)]
    fn tau_q<S: SimdLane>(v: S) -> S {
        let e1 = ((v - S::splat(CAT_THT_Q1)) * S::splat(-1.0 / CAT_SIG_Q1)).fast_exp();
        let e2 = (-(v - S::splat(CAT_THT_Q2)) * S::splat(1.0 / CAT_SIG_Q2)).fast_exp();
        S::splat(CAT_TAU_Q0) + S::splat(CAT_TAU_Q1) / (e1 + e2)
    }

    // ═══════════════════════════════════════════════════════════════
    //  Model
    // ═══════════════════════════════════════════════════════════════

    /// GPEA biophysical neuron — single compartment (point body).
    ///
    /// # States (9)
    /// - `v`: membrane potential (mV)
    /// - `m`, `h`: Na⁺ activation/inactivation
    /// - `n`: Kdr activation
    /// - `p`, `q`: CaT activation/inactivation
    /// - `cai`: intracellular Ca²⁺ (mM)
    /// - `r`: AHP (Ca-dependent K) gate
    /// - `was_below`: edge-detection flag
    ///
    /// # Params (14)
    /// - `dt`, `celsius` (T = 273+celsius−9.5 in Nernst), `inv_cm`, `v_thresh`
    /// - `area` (µm², Ca-dynamics scaling)
    /// - `gnabar`, `gkdrbar`, `gl`, `el`, `gcatbar`, `gkcabar` — mS/cm²
    /// - `kca`, `vol`, `caGain` (Ca dynamics)
    ///
    /// Conductance *maxima* are params so populations can be tuned per type
    /// (the paper adjusts them heuristically for PNs/LNs); kinetics constants
    /// are the published values.
    ///
    /// `#[no_clamp]`: `i_syn` — честная плотность тока (µA/cm²); сильный
    /// вход >100 µA/cm² реален (сильная синаптическая драйва в мозге мухи,
    /// токовая инъекция в валидации против NEURON до 500 µA/cm²).
    #[derive(NeuronModel)]
    #[no_clamp]
    pub struct GpeaModel {
        #[state]
        pub v: f32,
        #[state]
        pub m: f32,
        #[state]
        pub h: f32,
        #[state]
        pub n: f32,
        #[state]
        pub p: f32,
        #[state]
        pub q: f32,
        #[state]
        pub cai: f32,
        #[state]
        pub r: f32,
        #[state]
        pub was_below: f32,

        #[param]
        pub dt: f32,
        #[param]
        pub celsius: f32,
        #[param]
        pub inv_cm: f32,
        #[param]
        pub v_thresh: f32,
        #[param]
        pub area: f32,
        #[param]
        pub gnabar: f32,
        #[param]
        pub gkdrbar: f32,
        #[param]
        pub gl: f32,
        #[param]
        pub el: f32,
        #[param]
        pub gcatbar: f32,
        #[param]
        pub gkcabar: f32,
        #[param]
        pub kca: f32,
        #[param]
        pub vol: f32,
        #[param]
        pub caGain: f32,
    }

    impl ComputeNeuron for GpeaModel {
        type State<S: SimdLane> = GpeaState<S>;
        type Params<S: SimdLane> = GpeaParams<S>;

        #[inline(always)]
        fn step<S: SimdLane>(
            state: &mut Self::State<S>,
            params: &Self::Params<S>,
            ctx: &mut StepContext<S>,
        ) {
            // ─── Temperature: Nernst only (no Q10 on kinetics) ─────
            // T = 273 + celsius − 9.5 (mod convention); RT/F in mV
            let rt = S::splat(GAS_R)
                * (S::splat(273.0) + params.celsius - S::splat(9.5))
                / S::splat(CA_FARADAY)
                * S::splat(1000.0);
            let ena = -rt * (S::splat(NA_NAI) / S::splat(NA_NAO)).ln();
            let ek = rt * (S::splat(K_KO) / S::splat(K_KI)).ln();

            let dt = ctx.dt_simd();
            let v = state.v;

            // ─── Gate integration (exponential Euler, NEURON cnexp) ──
            // Steady states & time constants at the CURRENT (old) V.
            // (each helper call is its own `let` — the transpiler inlines
            // helpers only from `let` initializers, no nested helper calls)
            let m_inf = sigmoid(v, NA_TH_M, NA_K_M);
            let m_tau = tau_m(v);
            let m_new = exp_euler(state.m, m_inf, m_tau, dt);

            let h_inf = sigmoid(v, NA_TH_H, NA_K_H);
            let h_tau = tau_h(v);
            let h_new = exp_euler(state.h, h_inf, h_tau, dt);

            let n_inf = sigmoid(v, K_TH_N, K_K_N);
            let n_tau = tau_n(v);
            let n_new = exp_euler(state.n, n_inf, n_tau, dt);

            let p_inf = sigmoid(v, CAT_TH_P, CAT_K_P);
            let p_tau = tau_p(v);
            let p_new = exp_euler(state.p, p_inf, p_tau, dt);

            let q_inf = sigmoid(v, CAT_TH_Q, CAT_K_Q);
            let q_tau = tau_q(v);
            let q_new = exp_euler(state.q, q_inf, q_tau, dt);

            // AHP gate: τr const, r_inf from the CURRENT (old) cai.
            let r_inf = sigmoid(state.cai, AHP_TH_R, AHP_K_R);
            let r_tau = S::splat(AHP_TAU_R);
            let r_new = exp_euler(state.r, r_inf, r_tau, dt);

            // ─── Ca dynamics (cnexp, I_Ca frozen from old state) ─────
            let eca_old = -rt * (state.cai / S::splat(CA_CAO)).ln() / S::splat(2.0);
            let gcat_old = params.gcatbar * state.p * state.p * state.q;
            let ica_old = gcat_old * (v - eca_old);
            // dcai/dt = caGain·(-ica·area·1e-11/(2F·vol) − kca·cai)
            // (mod: ica in mA/cm²; here ica is in µA/cm² → 1e-11 → 1e-14)
            let j = -ica_old * params.area * S::splat(1e-14)
                / (S::splat(2.0) * S::splat(CA_FARADAY) * params.vol);
            let k_ca = params.caGain * params.kca;
            let cai_decay = (-dt * k_ca).fast_exp();
            // cai' = caGain·(j − kca·cai)  →  target = caGain·j/k_ca
            let cai_new =
                state.cai + (S::one() - cai_decay) * (params.caGain * j / k_ca - state.cai);

            // ─── Nernst with NEW cai ──────────────────────────────
            let eca = -rt * (cai_new / S::splat(CA_CAO)).ln() / S::splat(2.0);

            // ─── Effective conductances (new gates) ───────────────
            let g_na = params.gnabar * m_new.cube() * h_new;
            let g_k = params.gkdrbar * n_new.powi::<4>();
            let g_cat = params.gcatbar * p_new * p_new * q_new;
            let g_ahp = params.gkcabar * r_new * r_new;
            let g_leak = params.gl;

            let g_sum = g_na + g_k + g_cat + g_ahp + g_leak;
            let g_e_sum = g_na * ena
                + (g_k + g_ahp) * ek
                + g_cat * eca
                + g_leak * params.el;

            // ─── Membrane: implicit Euler (NEURON derivimplicit) ──
            let c_over_dt = S::one() / (params.inv_cm * dt);
            let v_new = (c_over_dt * v + ctx.i_syn + g_e_sum) / (c_over_dt + g_sum);

            // ─── Spike detection (edge-triggered, no reset) ───────
            let is_above = v_new.cmp_ge(params.v_thresh);
            let (spike, was_below_next) = edge_spike(is_above, state.was_below);
            ctx.emit_spike(spike);

            state.v = v_new;
            state.m = m_new;
            state.h = h_new;
            state.n = n_new;
            state.p = p_new;
            state.q = q_new;
            state.cai = cai_new;
            state.r = r_new;
            state.was_below = was_below_next;
        }
    }
}

pub use gpea::*;

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zoo::Model;
    use crate::NeuralPopulation;

    /// Params for the NEURON reference geometry (soma L=20, diam=40 µm;
    /// area = π·40·20 = 2513.27 µm²) at 24 °C, dt = 0.025 ms.
    fn make_params(num: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; num * GpeaModel::N_PARAMS];
        for i in 0..num {
            p[GpeaModelParamIdx::dt as usize * num + i] = 0.025;
            p[GpeaModelParamIdx::celsius as usize * num + i] = 24.0;
            p[GpeaModelParamIdx::inv_cm as usize * num + i] = 1.0;
            p[GpeaModelParamIdx::v_thresh as usize * num + i] = -20.0;
            p[GpeaModelParamIdx::area as usize * num + i] = CA_AREA;
            // Conductance maxima in mS/cm² (mod: S/cm² ×1000) — matches the
            // crate µA/cm² convention (g·(v−E) → µA/cm²).
            p[GpeaModelParamIdx::gnabar as usize * num + i] = 49.0;
            p[GpeaModelParamIdx::gkdrbar as usize * num + i] = 57.0;
            p[GpeaModelParamIdx::gl as usize * num + i] = 0.35;
            p[GpeaModelParamIdx::el as usize * num + i] = LEAK_EL;
            p[GpeaModelParamIdx::gcatbar as usize * num + i] = 5.0;
            p[GpeaModelParamIdx::gkcabar as usize * num + i] = 1.0;
            p[GpeaModelParamIdx::kca as usize * num + i] = CA_KCA;
            p[GpeaModelParamIdx::vol as usize * num + i] = CA_VOL;
            p[GpeaModelParamIdx::caGain as usize * num + i] = CA_GAIN;
        }
        p
    }

    /// Fresh population at the NEURON reference initial condition:
    /// V=-60 mV, gates at steady state, dt=0.025, 24°C.
    ///
    /// `cai = 5e-5` — NEURON's `finitialize` resets a WRITE-ion (the mod does
    /// `USEION ca WRITE cai`) to the ion's default initial concentration
    /// `cai0 = 5e-5` regardless of any hoc-set value; the reference traces
    /// therefore start from 5e-5. r = r_inf(5e-5) = 0.1821261 (NEURON-verified).
    fn make_pop(num: usize) -> NeuralPopulation<GpeaModel> {
        let mut pop = NeuralPopulation::<GpeaModel>::new(num);
        let p = make_params(1);
        for idx in 0..GpeaModel::N_PARAMS {
            pop.fill_param(idx, p[idx]);
        }
        pop.fill_state(0, -60.0); // v
        pop.fill_state(1, 0.0414); // m_inf(-60)
        pop.fill_state(2, 0.906); // h_inf(-60)
        pop.fill_state(3, 0.2165); // n_inf(-60)
        pop.fill_state(4, 0.355); // p_inf(-60)
        pop.fill_state(5, 0.01325); // q_inf(-60)
        pop.fill_state(6, 5e-5); // cai (NEURON finitialize WRITE-ion reset)
        pop.fill_state(7, 0.1821261); // r_inf(5e-5) — NEURON-verified
        pop.fill_state(8, 1.0); // was_below (edge)
        pop
    }

    /// Step `pop` `steps` times with `i_syn`; returns total spike count.
    fn step_n(pop: &mut NeuralPopulation<GpeaModel>, steps: usize, i_syn: &[f32]) -> i64 {
        let mut spikes = vec![0i32; i_syn.len()];
        let mut total = 0i64;
        for _ in 0..steps {
            pop.step(i_syn, &mut spikes);
            total += spikes.iter().map(|&s| s as i64).sum::<i64>();
        }
        total
    }

    /// Injected current density matching NEURON IClamp 0.5 nA over the
    /// reference section (area = π·40·20 µm² = 2.51327e-5 cm²):
    /// 0.5 nA / 2.51327e-5 cm² = 19.89 µA/cm².
    fn i_amp() -> f32 {
        0.5e-3 / (CA_AREA * 1e-8)
    }

    // ─── NEURON-validated rate functions (pGPeA_k.mod, NEURON 9.0.2) ──

    #[test]
    fn test_gpea_rates_resting_vs_neuron() {
        // Reference values from the compiled mod (evaluate_fct at V=-65, cai=1e-4)
        let v = -65.0f32;
        let cai = 1e-4f32;
        let m_inf = sigmoid_impl(v, NA_TH_M, NA_K_M);
        let h_inf = sigmoid_impl(v, NA_TH_H, NA_K_H);
        let n_inf = sigmoid_impl(v, K_TH_N, K_K_N);
        let p_inf = sigmoid_impl(v, CAT_TH_P, CAT_K_P);
        let q_inf = sigmoid_impl(v, CAT_TH_Q, CAT_K_Q);
        let r_inf = sigmoid_impl(cai, AHP_TH_R, AHP_K_R);
        let tm = tau_m_impl(v);
        let th = tau_h_impl(v);
        let tn = tau_n_impl(v);
        let tp = tau_p_impl(v);
        let tq = tau_q_impl(v);

        // NEURON reference at -65 mV (10 sig figs from gpea_ref.hoc)
        assert_close(m_inf, 0.02069111227, 1e-4);
        assert_close(h_inf, 0.9546474194, 1e-4);
        assert_close(n_inf, 0.1620766655, 1e-4);
        assert_close(p_inf, 0.2069705933, 1e-4);
        assert_close(q_inf, 0.03082031845, 1e-4);
        assert_close(r_inf, 0.2942149722, 1e-4);
        assert_close(tm, 0.1009999964, 1e-3);
        assert_close(th, 1.540321035, 1e-3);
        assert_close(tn, 1.098909804, 1e-3);
        assert_close(tp, 8.077265928, 1e-3);
        assert_close(tq, 136.9174254, 5e-2);
    }

    #[test]
    fn test_gpea_rates_depolarized_vs_neuron() {
        // V=0 mV — Na strongly activated, CaT fully on
        let v = 0.0f32;
        let cai = 1e-4f32;
        assert_close(sigmoid_impl(v, NA_TH_M, NA_K_M), 0.9956298205, 1e-4);
        assert_close(sigmoid_impl(v, NA_TH_H, NA_K_H), 0.0008167381047, 1e-4);
        assert_close(sigmoid_impl(v, CAT_TH_P, CAT_K_P), 0.9997655911, 1e-4);
        assert_close(sigmoid_impl(cai, AHP_TH_R, AHP_K_R), 0.2942149722, 1e-4);
        assert_close(tau_q_impl(v), 14.24726609, 1e-2);
    }

    #[test]
    fn test_gpea_rates_hyperpolarized_vs_neuron() {
        // V=-80 mV — everything closed, slow CaT inactivation gate q
        let v = -80.0f32;
        assert_close(sigmoid_impl(v, NA_TH_M, NA_K_M), 0.002472623157, 1e-4);
        assert_close(sigmoid_impl(v, NA_TH_H, NA_K_H), 0.9954615683, 1e-4);
        assert_close(sigmoid_impl(v, CAT_TH_Q, CAT_K_Q), 0.2969072608, 1e-4);
        assert_close(tau_h_impl(v), 0.6760660476, 1e-3);
        assert_close(tau_n_impl(v), 0.925418813, 1e-3);
        assert_close(tau_p_impl(v), 6.400175291, 1e-3);
        assert_close(tau_q_impl(v), 60.09475979, 5e-2);
    }

    #[test]
    fn test_gpea_r_inf_elevated_cai() {
        // r_inf(5e-4) — NEURON: 0.9840936083
        assert_close(sigmoid_impl(5e-4, AHP_TH_R, AHP_K_R), 0.9840936083, 1e-4);
    }

    // ─── NEURON-validated dynamics ──────────────────────────────

    #[test]
    fn test_gpea_resting_matches_neuron() {
        let num = 8;
        let mut pop = make_pop(num);
        let i_syn = vec![0.0f32; num];
        step_n(&mut pop, 200, &i_syn); // 5 ms
        // NEURON REST at t=5 ms: -61.72048753
        assert_close(pop.state_slice(0)[0], -61.72048753, 0.05);
        step_n(&mut pop, 1800, &i_syn); // → 50 ms
        // NEURON REST at t=50 ms: -61.2831801
        assert_close(pop.state_slice(0)[0], -61.2831801, 0.05);
        step_n(&mut pop, 6000, &i_syn); // → 200 ms
        // NEURON REST at t=200 ms: -61.04569455
        assert_close(pop.state_slice(0)[0], -61.04569455, 0.05);
    }

    #[test]
    fn test_gpea_pulse_matches_neuron() {
        let num = 8;
        let mut pop = make_pop(num);
        let i_off = vec![0.0f32; num];
        let i_on = vec![i_amp(); num];
        step_n(&mut pop, 2000, &i_off); // 0..50 ms (rest)
        // Track first spike after pulse onset: NEURON first at 50.8 ms
        let mut first_spike_ms = -1.0f32;
        let mut spikes = vec![0i32; num];
        for k in 0..4000 {
            pop.step(&i_on, &mut spikes); // 50..150 ms
            let s: i32 = spikes.iter().sum();
            if s > 0 && first_spike_ms < 0.0 {
                first_spike_ms = 50.0 + (k as f32 + 0.5) * 0.025;
            }
        }
        assert!(
            first_spike_ms > 0.0 && (first_spike_ms - 50.8).abs() < 0.5,
            "first spike at {first_spike_ms} ms vs NEURON 50.8 ms"
        );
        // Spike count over the 100 ms pulse: NEURON 37 (edge count, dt=0.025)
        // All `num` neurons are identical copies; NEURON reference is per neuron.
        let mut total = 0i64;
        let mut pop2 = make_pop(num);
        step_n(&mut pop2, 2000, &i_off);
        total += step_n(&mut pop2, 4000, &i_on);
        let per_neuron = total as f64 / num as f64;
        assert!(
            (per_neuron - 37.0).abs() <= 3.0,
            "pulse spike count {per_neuron}/neuron (total {total}/{num}) vs NEURON 37"
        );
    }

    // ─── Behavioral ─────────────────────────────────────────────

    #[test]
    fn test_gpea_silent_at_rest_no_nan() {
        let num = 8;
        let mut pop = make_pop(num);
        let i_syn = vec![0.0f32; num];
        let total = step_n(&mut pop, 8000, &i_syn); // 200 ms
        for &vi in pop.state_slice(0) {
            assert!(vi.is_finite(), "v NaN/inf");
        }
        assert_eq!(total, 0, "no input → no spikes (rest is below threshold)");
    }

    #[test]
    fn test_gpea_spikes_under_current() {
        let num = 8;
        let mut pop = make_pop(num);
        let i_syn = vec![i_amp(); num];
        let total = step_n(&mut pop, 4000, &i_syn); // 100 ms
        assert!(total > 0, "GPEA should fire under 0.5 nA, got {total}");
    }

    #[test]
    fn test_gpea_non_aligned() {
        let num = 13;
        let mut pop = make_pop(num);
        let i_syn = vec![i_amp(); num];
        let total = step_n(&mut pop, 4000, &i_syn);
        assert!(total > 0, "non-aligned GPEA should fire, got {total}");
    }

    #[test]
    fn test_gpea_stress_strong_input() {
        // 0.5 mA/cm² = 500 µA/cm² (25× the pulse current) — NEURON reference:
        // 220 spikes over 200 ms, V oscillates up to +60 mV (fast firing, not
        // a depolarization block — h re-activates each cycle).
        // (crate i_syn units are µA/cm², so 500.0 = 0.5 mA/cm²)
        let num = 8;
        let mut pop = make_pop(num);
        let i_syn = vec![500.0f32; num];
        let total = step_n(&mut pop, 8000, &i_syn); // 200 ms
        for &vi in pop.state_slice(0) {
            assert!(vi.is_finite(), "v NaN/inf");
            assert!(vi < 200.0 && vi > -200.0, "v out of bounds: {vi}");
        }
        let per_neuron = total as f64 / num as f64;
        assert!(
            (per_neuron - 220.0).abs() <= 22.0,
            "stress spike count {per_neuron}/neuron (total {total}/{num}) vs NEURON 220"
        );
    }

    #[test]
    fn test_gpea_population() {
        let num = 8;
        let mut pop = make_pop(num);
        let i_in = vec![i_amp(); num];
        let total = step_n(&mut pop, 4000, &i_in);
        assert!(total > 0, "GPEA population should fire, got {total}");
    }

    // ─── Scalar helpers (same formulas, plain f32) ──────────────

    fn sigmoid_impl(x: f32, th: f32, k: f32) -> f32 {
        1.0 / (1.0 + ((x - th) / k).exp())
    }

    fn tau_m_impl(v: f32) -> f32 {
        NA_TAU_M0 + NA_TAU_M1 / (1.0 + ((v + 53.0) / 0.7).exp())
    }

    fn tau_h_impl(v: f32) -> f32 {
        NA_TAU_H1 / (((v + 50.0) / 15.0).exp() + (-(v + 50.0) / 16.0).exp())
    }

    fn tau_n_impl(v: f32) -> f32 {
        K_TAU_N1 / (((v + 40.0) / 40.0).exp() + (-(v + 40.0) / 50.0).exp())
    }

    fn tau_p_impl(v: f32) -> f32 {
        CAT_TAU_P0 + CAT_TAU_P1 / (((v + 27.0) / 10.0).exp() + (-(v + 102.0) / 15.0).exp())
    }

    fn tau_q_impl(v: f32) -> f32 {
        CAT_TAU_Q1 / (((v + 50.0) / 15.0).exp() + (-(v + 50.0) / 16.0).exp())
    }

    fn assert_close(actual: f32, expected: f32, tol: f32) {
        let d = (actual - expected).abs();
        assert!(
            d <= tol,
            "value {actual} differs from NEURON reference {expected} by {d} (> {tol})"
        );
    }
}
