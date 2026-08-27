//! WgpuContext — device/queue + pipeline cache (аналог cuda engine::CompiledModel)

use std::collections::HashMap;
use wgpu::{ComputePipeline, Device, Queue};

pub struct WgpuContext {
    pub device: Device,
    pub queue: Queue,
    pub adapter_name: String,
    pub device_type: wgpu::DeviceType,
    pipelines: HashMap<String, ComputePipeline>,
}

impl WgpuContext {
    /// Создать headless device (Vulkan/Metal/DX12 — без surface).
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("wgpu: no adapter")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("pillars-wgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| format!("wgpu device: {e}"))?;
        let adapter_name = adapter.get_info().name;
        let device_type = adapter.get_info().device_type;
        Ok(Self {
            device,
            queue,
            adapter_name,
            device_type,
            pipelines: HashMap::new(),
        })
    }

    /// Блокирующий конструктор для sync кода (как cuda driver()).
    pub fn new_blocking() -> Result<Self, String> {
        pollster::block_on(Self::new())
    }

    pub fn get_pipeline(&self, name: &str) -> Option<&ComputePipeline> {
        self.pipelines.get(name)
    }

    pub fn insert_pipeline(&mut self, name: String, pipeline: ComputePipeline) {
        self.pipelines.insert(name, pipeline);
    }

    /// Скомпилировать WGSL compute shader в pipeline.
    pub fn compile(&mut self, name: &str, wgsl: &str, entry: &str) -> Result<(), String> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &shader,
                entry_point: entry,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        self.pipelines.insert(name.to_string(), pipeline);
        Ok(())
    }
}
