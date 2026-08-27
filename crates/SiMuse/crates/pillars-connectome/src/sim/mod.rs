//! Brain simulator — универсальный движок для любого CSR коннектома.
//!
//! Поддерживает LIF, Izhikevich, Rate, и GCO (Graph-Coupled Oscillator) модели.
//! Принимает [`ConnectomeData`](crate::data::ConnectomeData) для описания графа.

pub use crate::propagate::*;
pub use crate::types::*;

// ═══════════════════════════════════════════════════════════
//  Pillars bridge helpers (транш 3b → волна 12) — тонкие делегаты
//  в канон `pillars::api::neurons`. Единый источник для runner.rs;
//  тела репака живут ТОЛЬКО в pillars (дубль устранён).
// ═══════════════════════════════════════════════════════════

/// LIF: legacy 5-col params `[dt, tau_m, v_rest, v_thresh, v_reset]` × n →
/// делегат `pillars::api::neurons::step_lif` (репак в LIFModel внутри pillars).
mod gco;
#[cfg(feature = "cuda")]
pub mod gpu;
mod haap;
mod hh;
mod izhikevich;
mod lif;
mod rate;

pub use gco::{sim_gco, sim_gco_record};
pub use haap::{haap_population, sim_haap_pillars};
pub use hh::{
    hh_population_pacemaking, hh_population_standard, sim_hh_pillars, sim_hh_synaptic_pillars,
};
pub use izhikevich::{
    pad_to, sim_izhikevich, sim_izhikevich_dual, sim_izhikevich_synaptic,
    step_izhikevich_dual_pillars, step_izhikevich_pillars,
};
pub use lif::{sim_lif, sim_lif_synaptic, step_lif_pillars};

#[cfg(feature = "cuda")]
pub use gpu::{sim_izhikevich_gpu, sim_lif_gpu};
pub use rate::sim_rate;

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ConnectomeData;
    use pillars::kernels::synapse::{ReceptorType, SynapticKinetics};

    #[test]
    fn spike_queue_basic_push_pop() {
        let mut q = SpikeQueue::new(8);
        assert_eq!(q.len(), 0);
        q.push(5);
        q.push(42);
        q.push(100);
        assert_eq!(q.len(), 3);
        assert_eq!(q.pop(), Some(5));
        assert_eq!(q.pop(), Some(42));
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop(), Some(100));
        assert_eq!(q.pop(), None);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn spike_queue_wrap_around() {
        let mut q = SpikeQueue::new(4);
        // Push until near capacity
        q.push(1);
        q.push(2);
        q.push(3);
        q.push(4);
        assert_eq!(q.len(), 4);
        // Pop all
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.len(), 2);
        // Now push more — wraps around in circular buffer
        q.push(10);
        q.push(11);
        // head is now 6, tail is 2, cap=4, mask=3
        assert_eq!(q.len(), 4); // 6 - 2 = 4
        assert_eq!(q.pop(), Some(3));
        assert_eq!(q.pop(), Some(4));
        assert_eq!(q.pop(), Some(10));
        assert_eq!(q.pop(), Some(11));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn spike_queue_drain() {
        let mut q = SpikeQueue::new(8);
        q.push(1);
        q.push(2);
        q.push(3);
        assert_eq!(q.len(), 3);
        q.clear();
        assert_eq!(q.len(), 0);
        assert_eq!(q.pop(), None);
        // After clear, can still push
        q.push(42);
        assert_eq!(q.pop(), Some(42));
    }

    #[test]
    fn spike_queue_capacity_power_of_two() {
        let q = SpikeQueue::new(5); // next power of 2 = 8
        assert_eq!(q.buffer.len(), 8);
        assert_eq!(q.mask, 7);

        let q2 = SpikeQueue::new(1000); // next power of 2 = 1024
        assert_eq!(q2.buffer.len(), 1024);

        let q3 = SpikeQueue::new(1); // min 4
        assert_eq!(q3.buffer.len(), 4);
    }

    #[test]
    fn push_spikes_to_queue_basic() {
        let spikes = vec![0, 1, 0, 0, 1, 0, 1];
        let mut q = SpikeQueue::new(16);
        let count = push_spikes_to_queue(&spikes, 7, &mut q);
        assert_eq!(count, 3);
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(4));
        assert_eq!(q.pop(), Some(6));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn csr_propagate_from_queue_basic() {
        // Simple ring: 0→1, 1→2, 2→0
        let row_ptr = vec![0i32, 1, 2, 3];
        let col_indices = vec![1, 2, 0];
        let values = vec![0.5f32, 0.5, 0.5];
        let nt_types = vec![1.0f32, 1.0, 1.0];
        let mut I_buf = vec![0.0f32; 3];
        let mut q = SpikeQueue::new(8);

        // Neuron 0 spikes
        q.push(0);
        csr_propagate_from_queue(
            &row_ptr,
            &col_indices,
            &values,
            &nt_types,
            &mut q,
            &mut I_buf,
            1.0,
        );

        // I_buf[1] should get 0.5
        assert!((I_buf[1] - 0.5).abs() < 1e-6, "I_buf[1]={}", I_buf[1]);
        assert_eq!(I_buf[0], 0.0);
        assert_eq!(I_buf[2], 0.0);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn csr_propagate_from_queue_multi_spike() {
        // 0→2, 1→1 (self), 2→0, 3→1
        let row_ptr = vec![0i32, 1, 2, 3, 4];
        let col_indices = vec![2, 1, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let nt_types = vec![1.0f32, -1.0, 1.0, 1.0]; // neuron 1 is inhibitory
        let mut I_buf = vec![0.0f32; 3];
        let mut q = SpikeQueue::new(16);

        q.push(0); // exc → I_buf[2] += 1.0
        q.push(1); // inh → I_buf[1] += 2.0 * -1.0 = -2.0
        q.push(3); // exc → I_buf[1] += 4.0

        csr_propagate_from_queue(
            &row_ptr,
            &col_indices,
            &values,
            &nt_types,
            &mut q,
            &mut I_buf,
            1.0,
        );

        assert!((I_buf[2] - 1.0).abs() < 1e-6, "I_buf[2]={}", I_buf[2]);
        assert!((I_buf[1] - 2.0).abs() < 1e-6, "I_buf[1]={}", I_buf[1]); // -2.0 + 4.0 = 2.0
        assert_eq!(I_buf[0], 0.0);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn csr_propagate_from_queue_inhibitory_scale() {
        let row_ptr = vec![0i32, 2, 2];
        let col_indices = vec![0, 1]; // neuron 0→0 and 0→1
        let values = vec![1.0f32, 1.0];
        let nt_types = vec![1.0f32, 0.0]; // neuron 1 is silent (no synapses anyway)
        let mut I_buf = vec![0.0f32; 2];
        let mut q = SpikeQueue::new(8);

        q.push(0);
        // scale = -0.5
        csr_propagate_from_queue(
            &row_ptr,
            &col_indices,
            &values,
            &nt_types,
            &mut q,
            &mut I_buf,
            -0.5,
        );

        assert!((I_buf[0] - (-0.5)).abs() < 1e-6, "I_buf[0]={}", I_buf[0]);
        assert!((I_buf[1] - (-0.5)).abs() < 1e-6, "I_buf[1]={}", I_buf[1]);
        assert_eq!(q.len(), 0);
    }

    // ─── Synaptic kinetics integration tests ───────────

    fn make_synaptic_connectome() -> ConnectomeData {
        // 3 neurons: 0→1 (exc, AMPA), 1→2 (inh, GABA_A), 2→0 (exc, AMPA)
        let mut conn = ConnectomeData::new(
            vec![0i32, 1, 2, 3],
            vec![1, 2, 0],
            vec![0.5f32, 1.0, 0.8],
            vec![1.0, -1.0, 1.0],
            vec![],
        );
        conn.build_receptor_types();
        conn
    }

    #[test]
    fn test_sim_lif_synaptic_basic() {
        let conn = make_synaptic_connectome();
        assert!(
            !conn.receptor_types.is_empty(),
            "receptor_types must be non-empty"
        );

        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::LIF);
        // Strong drive ensures firing: LIF dv/dt = (I - (V - V_rest)) / tau
        // With I=100, tau=5, dt=0.1: dv ~= 2.0/step → reaches V_thresh=0 in ~26 steps
        let tau: Vec<f32> = vec![5.0; n];
        let rest: Vec<f32> = vec![-52.0; n];
        let thresh: Vec<f32> = vec![0.0; n];
        let reset: Vec<f32> = vec![-60.0; n];
        let params = LifParams::new(&tau, &rest, &thresh, &reset, 0.1, 1.0);
        let external_I = vec![100.0f32; n];
        let n_steps = 100;
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_lif_synaptic(
            &conn,
            &mut state,
            &params,
            &external_I,
            n_steps,
            &mut kinetics,
            1.0,
        );

        let total_spikes: i32 = spikes.iter().sum();
        assert!(
            total_spikes > 0,
            "LIF synaptic should fire with high drive, got {} spikes",
            total_spikes
        );
        assert_eq!(spikes.len(), n_steps * n);
    }

    #[test]
    fn test_sim_izhikevich_synaptic_basic() {
        let conn = make_synaptic_connectome();
        assert!(!conn.receptor_types.is_empty());

        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::Izhikevich);
        let params = IzhikevichParams {
            a: vec![0.02f32; n],
            b: vec![0.2f32; n],
            c: vec![-65.0f32; n],
            d: vec![8.0f32; n],
            dt_ms: 0.5,
            tau_ref: 3.0,
        };
        let external_I = vec![7.0f32; n];
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_izhikevich_synaptic(
            &conn,
            &mut state,
            &params,
            &external_I,
            20,
            &mut kinetics,
            1.0,
        );

        let total_spikes: i32 = spikes.iter().sum();
        assert!(
            total_spikes > 0,
            "Izhikevich synaptic should fire with high drive"
        );
        assert_eq!(spikes.len(), 20 * n);
    }

    #[test]
    fn test_sim_lif_synaptic_conductance_decay() {
        let conn = make_synaptic_connectome();
        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::LIF);
        let tau: Vec<f32> = vec![5.0; n];
        let rest: Vec<f32> = vec![-52.0; n];
        let thresh: Vec<f32> = vec![0.0; n];
        let reset: Vec<f32> = vec![-60.0; n];
        let params = LifParams::new(&tau, &rest, &thresh, &reset, 0.1, 1.0);
        let external_I = vec![0.0f32; n];
        let mut kinetics = SynapticKinetics::new(n);

        kinetics.jump(0, ReceptorType::AMPA, 10.0);
        kinetics.jump(1, ReceptorType::AMPA, 10.0);

        let initial_g = kinetics.g_ampa[0];
        assert!((initial_g - 10.0).abs() < 1e-6, "initial g={}", initial_g);

        let _spikes = sim_lif_synaptic(
            &conn,
            &mut state,
            &params,
            &external_I,
            1,
            &mut kinetics,
            1.0,
        );

        let expected_decay = (-0.1f32 / 2.0).exp();
        assert!(
            kinetics.g_ampa[0] < 10.0 * expected_decay * 1.01,
            "g_ampa[0]={} should have decayed",
            kinetics.g_ampa[0]
        );
        assert!(
            kinetics.g_ampa[0] > 0.0,
            "g_ampa[0] should still be positive"
        );
    }

    #[test]
    fn test_sim_izhikevich_synaptic_receptor_types() {
        let mut conn = make_synaptic_connectome();
        conn.receptor_types = vec![0u8, 1, 2]; // AMPA, NMDA, GABA_A

        let n = conn.n_neurons;
        let mut state = BrainState::new(n, NeuronModel::Izhikevich);
        let params = IzhikevichParams {
            a: vec![0.02f32; n],
            b: vec![0.2f32; n],
            c: vec![-65.0f32; n],
            d: vec![8.0f32; n],
            dt_ms: 0.5,
            tau_ref: 3.0,
        };
        let external_I = vec![10.0f32; n];
        let mut kinetics = SynapticKinetics::new(n);

        let spikes = sim_izhikevich_synaptic(
            &conn,
            &mut state,
            &params,
            &external_I,
            50,
            &mut kinetics,
            1.0,
        );

        let total_spikes: i32 = spikes.iter().sum();
        assert!(
            total_spikes > 0,
            "Izhikevich with mixed receptors should fire"
        );

        let ampa_decay = (-0.5f32 / 2.0).exp().powi(50);
        let nmda_decay = (-0.5f32 / 50.0).exp().powi(50);
        assert!(
            nmda_decay > ampa_decay,
            "NMDA should decay slower than AMPA"
        );
        println!("AMPA decay over 50 steps: {:.6}", ampa_decay);
        println!("NMDA decay over 50 steps: {:.6}", nmda_decay);
    }

    #[test]
    fn test_sim_haap_basic() {
        let conn = make_synaptic_connectome();
        let n = conn.n_neurons;
        // dt=0.05 ms — HAAP (M-current + h-current) is stable at this timestep.
        // 2000 steps = 100 ms.
        let mut pop = haap_population(n, 0.05);
        let external_I = vec![15.0f32; n];
        let n_steps = 2000;

        let spikes = sim_haap_pillars(&conn, &mut pop, &external_I, n_steps, 1.0);

        let total_spikes: i32 = spikes.iter().sum();
        assert!(
            total_spikes > 0,
            "HAAP should fire with high drive (I=15), got {} spikes",
            total_spikes
        );
        assert_eq!(spikes.len(), n_steps * n);
    }

    // ═══════════════════════════════════════════════════════
    //  Integration: HH physics + synaptic chemistry
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_hh_synaptic_vs_current_ratio() {
        // Демонстрация ratio (conductance-based) / (current-based) = (E_rev - V)
        // Для nAChR при V=-65: ratio = 65
        let n = 2;
        let mut kin = SynapticKinetics::new(n);
        let v = vec![-65.0, -65.0];
        let mut i_curr = [0.0, 0.0];
        let mut i_cond = vec![0.0, 0.0];

        // Current-based: nt_types[0]=1, weight=1, scale=1 → I_post = 1*1*1 = 1.0
        i_curr[1] = 1.0; // simulated CSR result

        // Conductance-based: jump + compute
        kin.jump(1, ReceptorType::nAChR, 1.0); // g=1.0
        kin.compute_currents(&v, &mut i_cond);

        let ratio = i_cond[1] / i_curr[1];
        println!("Conductance/Current ratio for nAChR at V=-65: {ratio:.0}× (expected ~65)");
        assert!(
            ratio > 50.0 && ratio < 80.0,
            "Ratio {ratio:.0} should be ~65× for nAChR at rest"
        );
    }

}
