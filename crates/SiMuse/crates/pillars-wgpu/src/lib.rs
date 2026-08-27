//! pillars-wgpu — WebGPU compute backend (вычисления, не рендер)
//! Порт pillars-cuda на wgpu (Vulkan/Metal/DX12/WebGPU) для SNN.

pub mod engine;
pub mod network;

pub use engine::WgpuContext;
pub use network::{WgpuNetwork, WgpuPop};

// WGSL sources — авто-генерятся #[dual_neuron] как *_WGSL_SOURCE (как *_CUDA_SOURCE)
// Ручные shaders/*.wgsl — fallback, канон — pillars::zoo::*_WGSL_SOURCE
pub use pillars::zoo::neurons::{
    LIF_WGSL_SOURCE, IZHIKEVICH_WGSL_SOURCE, ADEX_WGSL_SOURCE, ALPHA_LIF_WGSL_SOURCE,
    IZHIKEVICH_2007_WGSL_SOURCE, LIFAT_WGSL_SOURCE, RATE_WGSL_SOURCE, SAW_WGSL_SOURCE,
    HAAP_WGSL_SOURCE, HR_WGSL_SOURCE, FHN_WGSL_SOURCE, GLIF3_WGSL_SOURCE, GLIF4_WGSL_SOURCE,
};
pub const LIF_WGSL: &str = pillars::zoo::neurons::LIF_WGSL_SOURCE;
pub const IZHIKEVICH_WGSL: &str = pillars::zoo::neurons::IZHIKEVICH_WGSL_SOURCE;
pub const CSR_SCATTER_WGSL: &str = include_str!("../shaders/csr_scatter.wgsl");

// Re-export pillars models for convenience (как pillars-cuda lib.rs:50)
pub use pillars::zoo::neurons::{
    AdExModel, AlphaLifModel, IzhikevichModel, LIFModel, RateModel, SAWModel,
};
