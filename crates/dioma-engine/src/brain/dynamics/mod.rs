//! Исполнение мозгов: мембранная динамика и численное ядро.
//!
//! - [`neurons`] — `Brain`: CSR-коннектом, CPU/GPU-динамика, пластичность;
//! - [`gpu`] — GPU-динамика одной особи (`GpuDynamics`);
//! - [`gpu_pool`] — батч-пул одной сети на всю арену (`BrainPool`);
//! - [`perf`] — низкоуровневые численные оптимизации (FTZ, SIMD).
pub mod gpu;
pub mod gpu_pool;
pub mod neurons;
pub mod perf;
