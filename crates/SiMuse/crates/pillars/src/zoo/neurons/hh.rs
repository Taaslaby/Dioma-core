//! HH stub — HH удалён для Brian2-профиля.
//! Операнды vtrap/fast_exp/sigmoid остались в crate::math — собери свою модель через #[derive(NeuronModel)].

use pillars_macros::dual_neuron;

#[dual_neuron]
pub mod hh {
    use crate::core::compute::{ComputeNeuron, StepContext};
    use crate::core::lane::SimdLane;
    use pillars_macros::NeuronModel;

    #[derive(NeuronModel)]
    #[no_clamp]
    pub struct HHModel {
        #[state] pub v: f32,
        #[state] pub m: f32,
        #[state] pub h: f32,
        #[state] pub n_param: f32,
        #[state] pub dummy4: f32,
        #[state] pub dummy5: f32,
        #[param] pub dt: f32,
        #[param] pub g_na: f32,
        #[param] pub g_k: f32,
        #[param] pub g_leak: f32,
        #[param] pub e_na: f32,
        #[param] pub e_k: f32,
        #[param] pub e_leak: f32,
        #[param] pub inv_cm: f32,
        #[param] pub q10: f32,
        #[param] pub v_thresh: f32,
        #[param] pub g_ca_t: f32,
        #[param] pub e_ca_t: f32,
    }

    impl ComputeNeuron for HHModel {
        type State<S: SimdLane> = HHState<S>;
        type Params<S: SimdLane> = HHParams<S>;
        #[inline(always)]
        fn step<S: SimdLane>(_state: &mut HHState<S>, _params: &HHParams<S>, ctx: &mut StepContext<S>) {
            // Stub: HH удалён — no-op. Используй vtrap/fast_exp для своей модели.
            ctx.emit_spike(S::zero().cmp_gt(S::zero()));
        }
    }

    pub const GATE_M: usize = 0;
    pub const GATE_H: usize = 0;
    pub const GATE_N: usize = 0;
    pub struct GateParams;
}

pub use hh::*;
