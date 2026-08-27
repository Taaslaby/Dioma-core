//! Общие помощники интеграционных offscreen-тестов моста dioma (сим ↔ рендер).
//!
//! Схема — как в crates/dioma-render/tests/common/mod.rs: GPU-устройство
//! создаётся ОДИН раз на процесс (`OnceLock`), чтобы не зависнуть на
//! драйверах при параллельном создании нескольких wgpu-устройств.

#![allow(dead_code)]

use std::sync::{Arc, OnceLock};

/// GPU-пара для offscreen-тестов (`Arc` — wgpu 22 Device/Queue не клонируются).
#[derive(Clone)]
pub struct TestGpu {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

static SHARED_GPU: OnceLock<Option<TestGpu>> = OnceLock::new();

/// Поднимает устройство (fallback-адаптер при отсутствии основного).
/// `None` — GPU недоступен, тест пропускается.
pub fn gpu() -> Option<TestGpu> {
    let init = || {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let mut adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
        if adapter.is_none() {
            adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                force_fallback_adapter: true,
                ..Default::default()
            }));
        }
        let Some(adapter) = adapter else {
            eprintln!("[dioma bridge] тест пропущен: нет GPU-адаптера");
            return None;
        };
        let (device, queue) = match pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("dioma bridge offscreen test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[dioma bridge] тест пропущен: request_device не удался: {e}");
                return None;
            }
        };
        Some(TestGpu { device: Arc::new(device), queue: Arc::new(queue) })
    };
    SHARED_GPU.get_or_init(init).clone()
}
