//! Интеграционные тесты меш-пайплайна: произвольная триангулированная
//! геометрия с текстурой и per-vertex цветом.
//!
//! Всё offscreen (64×64, RGBA8UnormSrgb), та же схема fallback-адаптера, что
//! в остальных тестах; без GPU — пропуск. Меш-проходы открываются с Load,
//! поэтому перед ними пустой спрайтовый проход даёт заливку (как в
//! `effects_offscreen.rs`).

mod common;

use std::sync::Arc;

use common::{camera, gpu, navy_clear, px, readback, sum_rgb, target, H, W};
use dioma_render::{atlas, Mesh, MeshBatch, MeshBlend, Renderer, SpriteAtlas, SpriteBatch};

/// Атлас 8×8, целиком занятый белым квадратом: uv `[0,0,1,1]` = вся текстура,
/// цвет меша проходит насквозь без примеси.
fn white_atlas(device: &Arc<wgpu::Device>, queue: &wgpu::Queue) -> SpriteAtlas {
    let mut atlas = SpriteAtlas::new(device, 8);
    let (w, h, pixels) = atlas::bar_rgba(8, 8, [255, 255, 255, 255], [255, 255, 255, 255], 1.0);
    atlas.add_rgba(queue, "white", w, h, &pixels).unwrap();
    atlas
}

/// Очищает цель заливкой и рисует последовательность mesh-проходов
/// (каждый — своим блендингом). Возвращает RGBA8-пиксели.
///
/// Вызывающий обязан сделать `MeshBatch::upload` перед вызовом (контракт
/// [`Renderer::render_meshes`]).
fn render_passes(
    gpu: &common::TestGpu,
    atlas: &SpriteAtlas,
    passes: &[(&MeshBatch, MeshBlend)],
) -> Vec<u8> {
    let (tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mesh encoder") });
    // Пустой спрайтовый проход даёт заливку: mesh-проходы идут с Load.
    let empty = SpriteBatch::new(&gpu.device, 1);
    renderer.render(&gpu.queue, &mut encoder, &view, atlas, &empty, &camera());
    for (batch, blend) in passes {
        renderer.render_meshes(&mut encoder, &gpu.queue, &view, atlas, batch, &camera(), *blend);
    }
    gpu.queue.submit(Some(encoder.finish()));
    readback(gpu, &tex, W, H)
}

/// ── 1. Квад: красный прямоугольник на тёмно-синей заливке ────────────────

#[test]
fn mesh_quad_renders() {
    let Some(gpu) = gpu() else { return };
    let atlas = white_atlas(&gpu.device, &gpu.queue);

    // Квад почти на весь вьюпорт, красный; белая текстура → цвет проходит.
    let mut batch = MeshBatch::new(&gpu.device, 16, 32);
    batch.push(&Mesh::quad([32.0, 32.0], [56.0, 56.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]));
    batch.upload(&gpu.queue);
    let data = render_passes(&gpu, &atlas, &[(&batch, MeshBlend::Alpha)]);

    let center = px(&data, W, W / 2, H / 2);
    assert!(center[0] > 150, "центр должен быть красным, а это {center:?}");
    assert!(center[1] < 80, "центр должен быть красным, а это {center:?}");
    assert!(center[2] < 80, "центр должен быть красным, а это {center:?}");

    // Угол (квад покрывает не всё): тёмно-синяя заливка.
    let corner = px(&data, W, 2, 2);
    assert!(corner[0] < 80, "угол должен быть заливкой, а это {corner:?}");
    assert!(corner[1] < 80, "угол должен быть заливкой, а это {corner:?}");
    assert!(corner[2] > 60, "угол должен быть синеватым, а это {corner:?}");
}

/// ── 2. UV-маппинг: квад семплирует только угол атласа ─────────────────────

#[test]
fn mesh_uses_atlas_uv() {
    let Some(gpu) = gpu() else { return };
    // 4×4 текстура: первый столбец красный, остальное синее. Она укладывается
    // в самый верхний левый угол атласа 256², поэтому uv_rect [0,0,0.01,0.01]
    // семплирует именно её малый участок.
    let mut atlas = SpriteAtlas::new(&gpu.device, 256);
    let mut pixels = vec![0u8; 4 * 4 * 4];
    for y in 0..4 {
        for x in 0..4 {
            let i = (y * 4 + x) * 4;
            pixels[i] = if x == 0 { 255 } else { 0 };
            pixels[i + 1] = 0;
            pixels[i + 2] = if x == 0 { 0 } else { 255 };
            pixels[i + 3] = 255;
        }
    }
    atlas.add_rgba(&gpu.queue, "corner", 4, 4, &pixels).unwrap();

    // Квад на весь вьюпорт, белый цвет, семплирует ТОЛЬКО угол атласа:
    // левая половина экрана — красный столбец текстуры, правая — синяя.
    let mut batch = MeshBatch::new(&gpu.device, 16, 32);
    batch.push(&Mesh::quad([32.0, 32.0], [64.0, 64.0], [0.0, 0.0, 0.01, 0.01], [1.0, 1.0, 1.0, 1.0]));
    batch.upload(&gpu.queue);
    let data = render_passes(&gpu, &atlas, &[(&batch, MeshBlend::Alpha)]);

    let left = px(&data, W, 16, H / 2);
    assert!(left[0] > 150, "левая половина должна быть красной, а это {left:?}");
    assert!(left[0] > left[2], "в левой половине красный должен доминировать, а это {left:?}");

    let right = px(&data, W, 48, H / 2);
    assert!(right[2] > 150, "правая половина должна быть синей, а это {right:?}");
    assert!(right[2] > right[0], "в правой половине синий должен доминировать, а это {right:?}");
}

/// ── 3. Веер: круглый блоб по центру, углы — заливка ───────────────────────

#[test]
fn mesh_circle_fan_shape() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 256);
    let (w, h, pixels) = atlas::circle_rgba(16, [255, 255, 255, 255]);
    atlas.add_rgba(&gpu.queue, "circle", w, h, &pixels).unwrap();
    let uv = atlas.region("circle").unwrap();

    let mut batch = MeshBatch::new(&gpu.device, 64, 192);
    batch.push(&Mesh::circle_fan([32.0, 32.0], 15.0, 24, uv, [1.0, 1.0, 1.0, 1.0]));
    batch.upload(&gpu.queue);
    let data = render_passes(&gpu, &atlas, &[(&batch, MeshBlend::Alpha)]);

    // Центр веера: белый круг текстуры (r-канал яркий).
    let center = px(&data, W, W / 2, H / 2);
    assert!(center[0] > 150, "центр блоба должен быть ярким, а это {center:?}");

    // Угол далеко за радиусом 15: заливка.
    let corner = px(&data, W, 2, 2);
    assert!(corner[0] < 80, "угол должен остаться заливкой, а это {corner:?}");
    assert!(corner[1] < 80, "угол должен остаться заливкой, а это {corner:?}");
    assert!(corner[2] > 60, "угол должен остаться синеватым, а это {corner:?}");
}

/// ── 4. Аддитивный блендинг: светлеет поверх тусклого квада ────────────────

#[test]
fn mesh_additive_blends() {
    let Some(gpu) = gpu() else { return };
    let atlas = white_atlas(&gpu.device, &gpu.queue);

    // Тусклый полноэкранный квад + аддитивный жёлто-зелёный поверх.
    let dim = Mesh::quad([32.0, 32.0], [64.0, 64.0], [0.0, 0.0, 1.0, 1.0], [0.3, 0.3, 0.3, 1.0]);
    let glow = Mesh::quad([32.0, 32.0], [64.0, 64.0], [0.0, 0.0, 1.0, 1.0], [0.4, 0.4, 0.0, 0.5]);

    // Контроль: только тусклый квад.
    let mut control = MeshBatch::new(&gpu.device, 64, 64);
    control.push(&dim);
    control.upload(&gpu.queue);
    let data_control = render_passes(&gpu, &atlas, &[(&control, MeshBlend::Alpha)]);

    // Тест: тусклый квад + аддитивный проход поверх (та же текстура).
    let mut both = MeshBatch::new(&gpu.device, 64, 64);
    both.push(&dim);
    both.upload(&gpu.queue);
    let mut glow_batch = MeshBatch::new(&gpu.device, 64, 64);
    glow_batch.push(&glow);
    glow_batch.upload(&gpu.queue);
    let data_glow =
        render_passes(&gpu, &atlas, &[(&both, MeshBlend::Alpha), (&glow_batch, MeshBlend::Additive)]);

    let center_control = px(&data_control, W, W / 2, H / 2);
    let center_glow = px(&data_glow, W, W / 2, H / 2);
    assert!(
        sum_rgb(center_glow) > sum_rgb(center_control) + 50,
        "аддитивный проход должен светлеть: {} vs {} ({center_glow:?} vs {center_control:?})",
        sum_rgb(center_glow),
        sum_rgb(center_control)
    );
    assert!(
        center_glow[1] > center_control[1] + 20,
        "зелёный канал должен заметно вырасти: {center_glow:?} vs {center_control:?}"
    );
}
