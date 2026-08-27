//! Общие помощники интеграционных offscreen-тестов dioma-render.
//!
//! Адаптер: любой доступный; fallback — программный; нет — тест пропускается
//! (CI без видеокарты не падает).

use std::sync::Arc;

use dioma_render::Camera;

/// GPU-пара для offscreen-тестов.
///
/// `device` обёрнут в `Arc`: wgpu 22 `Device` не клонируется, а атлас хранит
/// клон устройства для авто-роста (см. `SpriteAtlas::new`).
pub struct TestGpu {
    pub device: Arc<wgpu::Device>,
    pub queue: wgpu::Queue,
}

/// Стандартный размер offscreen-цели.
pub const W: u32 = 64;
pub const H: u32 = 64;

/// Поднимает устройство (та же схема fallback-адаптера, что в основном тесте).
/// Возвращает `None`, если GPU недоступен — тест надо пропустить.
pub fn gpu() -> Option<TestGpu> {
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
        eprintln!("[dioma-render] тест пропущен: на машине нет ни одного GPU-адаптера");
        return None;
    };
    let (device, queue) = match pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("dioma-render offscreen test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    )) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[dioma-render] тест пропущен: request_device не удался: {e}");
            return None;
        }
    };
    Some(TestGpu { device: Arc::new(device), queue })
}

/// Offscreen-цель `w×h` RGBA8UnormSrgb (RENDER_ATTACHMENT | COPY_SRC) + её view.
pub fn target(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dioma-render offscreen target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Копирует `texture` (w×h, row_pitch = w×4) на CPU и возвращает RGBA8-пиксели.
pub fn readback(gpu: &TestGpu, texture: &wgpu::Texture, w: u32, h: u32) -> Vec<u8> {
    let row_pitch = w * 4;
    let read_back = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dioma-render readback"),
        size: row_pitch as u64 * h as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("offscreen encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &read_back,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(row_pitch),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit(Some(encoder.finish()));

    let slice = read_back.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .expect("map_async должен вызвать callback")
        .expect("map_async ok");
    // Временный BufferView падает в конце выражения — borrow `read_back` закрыт.
    let data = slice.get_mapped_range().to_vec();
    data
}

/// Пиксель (x, y) из readback-данных (RGBA8, строка w×4 байта).
#[allow(dead_code)] // используется не во всех тестовых крейтах
pub fn px(data: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let i = (y * w * 4 + x * 4) as usize;
    [data[i], data[i + 1], data[i + 2], data[i + 3]]
}

/// Сумма RGB — простое «насколько яркий/цветной» сравнение.
#[allow(dead_code)] // используется не во всех тестовых крейтах
pub fn sum_rgb(p: [u8; 4]) -> u32 {
    p[0] as u32 + p[1] as u32 + p[2] as u32
}

/// Стандартная камера: центр (32,32), вьюпорт 64×64, зум 1.
pub fn camera() -> Camera {
    Camera { center: [W as f32 * 0.5, H as f32 * 0.5], zoom: 1.0, viewport: [W as f32, H as f32] }
}

/// Тёмно-синяя заливка, как в основном тесте (линейный 0.02/0.03/0.10).
pub fn navy_clear() -> wgpu::Color {
    wgpu::Color { r: 0.02, g: 0.03, b: 0.10, a: 1.0 }
}
