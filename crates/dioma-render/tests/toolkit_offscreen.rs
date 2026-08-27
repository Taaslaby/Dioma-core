//! Интеграционные тесты debug/profiling-тулкита: флипбук, Y-сортировка по
//! глубине, счётчики профилировщика, текстовый оверлей, wireframe-меши,
//! sanity глифов шрифта.
//!
//! Всё offscreen (64×64, RGBA8UnormSrgb), та же схема fallback-адаптера, что
//! в остальных тестах; без GPU — пропуск. Тест флипбука — чистый CPU, GPU
//! не нужен.

mod common;

use std::sync::Arc;

use common::{camera, gpu, navy_clear, px, readback, sum_rgb, target, H, W};
use dioma_render::{
    atlas, Camera, DebugFont, Flipbook, Light, LightBatch, Mesh, MeshBatch, MeshBlend, Particle,
    ParticleSystem, Renderer, Sprite, SpriteAtlas, SpriteBatch, SpriteFx,
};

/// Атлас 8×8, целиком занятый белым квадратом (uv [0,0,1,1]): цвет проходит
/// насквозь от тинта/цвета вершин.
fn white_atlas(device: &Arc<wgpu::Device>, queue: &wgpu::Queue) -> SpriteAtlas {
    let mut atlas = SpriteAtlas::new(device, 8);
    let (w, h, pixels) = atlas::bar_rgba(8, 8, [255, 255, 255, 255], [255, 255, 255, 255], 1.0);
    atlas.add_rgba(queue, "white", w, h, &pixels).unwrap();
    atlas
}

// ── 1. Флипбук: математика сетки, зацикливание и клампинг (чистый CPU) ─────

#[test]
fn flipbook_loops_and_clamps() {
    // 2×2 сетка региона [0,0,0.5,0.5] → 4 кадра с корректными UV.
    let sheet = [0.0, 0.0, 0.5, 0.5];
    let fb = Flipbook::from_grid(sheet, 2, 2, 0, 4, 10.0, true);
    assert_eq!(fb.len(), 4);
    let expect = [
        [0.0, 0.0, 0.25, 0.25],
        [0.25, 0.0, 0.5, 0.25],
        [0.0, 0.25, 0.25, 0.5],
        [0.25, 0.25, 0.5, 0.5],
    ];
    for (i, uv) in expect.iter().enumerate() {
        for c in 0..4 {
            assert!(
                (fb.frames[i][c] - uv[c]).abs() < 1e-6,
                "кадр {i} должен быть {uv:?}, а это {:?}",
                fb.frames[i]
            );
        }
    }

    // Зацикливание: frame_at(0.0) == frames[0]; (t·10) % 4.
    assert_eq!(fb.frame_at(0.0), fb.frames[0]);
    assert_eq!(fb.frame_at(0.35), fb.frames[3]);
    assert_eq!(fb.frame_at(0.40), fb.frames[0], "4 кадра — снова первый");

    // Поддиапазон: start=1, count=2 берёт кадры 1..2 сетки.
    let sub = Flipbook::from_grid(sheet, 2, 2, 1, 2, 10.0, true);
    assert_eq!(sub.frames, vec![expect[1], expect[2]]);

    // Незацикленный: клампится к последнему кадру.
    let once = Flipbook::new(fb.frames.clone(), 10.0, false);
    assert_eq!(once.frame_at(0.0), once.frames[0]);
    assert_eq!(once.frame_at(100.0), once.frames[3], "после конца — последний кадр");

    // Пустой флипбук: весь атлас.
    let empty = Flipbook::new(vec![], 10.0, true);
    assert!(empty.is_empty());
    assert_eq!(empty.frame_at(1.0), [0.0, 0.0, 1.0, 1.0]);

    // with_frame применяет кадр к спрайту.
    let mut s = Sprite::new([0.0, 0.0], [10.0, 10.0], [0.0, 0.0, 1.0, 1.0]);
    s = s.with_frame(&fb, 0.25);
    assert_eq!(s.uv_rect, fb.frame_at(0.25));
}

// ── 2. Y-сортировка: на одном слое больший depth рисуется последним ────────

#[test]
fn y_sort_draws_top_most() {
    let Some(gpu) = gpu() else { return };
    let atlas = white_atlas(&gpu.device, &gpu.queue);

    let render_pair = |red_depth: f32, blue_depth: f32| -> [u8; 4] {
        let mut batch = SpriteBatch::new(&gpu.device, 2);
        // Один и тот же квад: красный (depth = red_depth) и синий (blue_depth).
        batch.push(
            Sprite::new([W as f32 * 0.5, H as f32 * 0.5], [32.0, 32.0], [0.0, 0.0, 1.0, 1.0])
                .with_tint([1.0, 0.0, 0.0, 1.0])
                .with_depth(red_depth),
        );
        batch.push(
            Sprite::new([W as f32 * 0.5, H as f32 * 0.5], [32.0, 32.0], [0.0, 0.0, 1.0, 1.0])
                .with_tint([0.0, 0.0, 1.0, 1.0])
                .with_depth(blue_depth),
        );
        batch.upload(&gpu.queue);

        let (tex, view) = target(&gpu.device, W, H);
        let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
        renderer.set_clear_color(navy_clear());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("y-sort encoder") });
        renderer.render(&gpu.queue, &mut encoder, &view, &atlas, &batch, &camera());
        gpu.queue.submit(Some(encoder.finish()));
        let data = readback(&gpu, &tex, W, H);
        px(&data, W, W / 2, H / 2)
    };

    // depth 0 < depth 1: синий поверх красного.
    let p = render_pair(0.0, 1.0);
    assert!(p[2] > 150 && p[0] < 80, "при depth 1 поверх должен быть синий, а это {p:?}");

    // Меняем глубины: красный поверх синего.
    let p = render_pair(1.0, 0.0);
    assert!(p[0] > 150 && p[2] < 80, "при depth 1 поверх должен быть красный, а это {p:?}");
}

// ── 3. Профилировщик: счётчики накапливаются по проходам ───────────────────

#[test]
fn profiler_counts_pass() {
    let Some(gpu) = gpu() else { return };
    let atlas = white_atlas(&gpu.device, &gpu.queue);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    renderer.begin_frame(1.0 / 60.0);

    // 3 спрайта: два без эффекта + один с fx (glow).
    let mut sprites = SpriteBatch::new(&gpu.device, 4);
    sprites.push(Sprite::new([20.0, 20.0], [10.0, 10.0], [0.0, 0.0, 1.0, 1.0]));
    sprites.push(Sprite::new([40.0, 20.0], [10.0, 10.0], [0.0, 0.0, 1.0, 1.0]));
    sprites.push(
        Sprite::new([30.0, 40.0], [10.0, 10.0], [0.0, 0.0, 1.0, 1.0])
            .with_fx(SpriteFx::Glow { radius: 3.0, intensity: 0.5, color: [1.0, 1.0, 0.5, 1.0] }),
    );
    sprites.upload(&gpu.queue);

    // 1 меш (квад → 6 индексов), 1 свет, 1 частица.
    let mut meshes = MeshBatch::new(&gpu.device, 16, 32);
    meshes.push(&Mesh::quad([32.0, 32.0], [8.0, 8.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]));
    meshes.upload(&gpu.queue);
    let mut lights = LightBatch::new(&gpu.device, 2);
    lights.push(Light { position: [32.0, 32.0], radius: 20.0, intensity: 0.5, color: [1.0, 1.0, 1.0] });
    lights.upload(&gpu.queue);
    let mut particles = ParticleSystem::new(&gpu.device, 8);
    particles.spawn(Particle::new([32.0, 32.0], [0.0, 0.0], 1.0, 12.0, [1.0, 1.0, 0.8, 1.0]));

    let (_tex, view) = target(&gpu.device, W, H);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("profiler encoder") });
    renderer.render(&gpu.queue, &mut encoder, &view, &atlas, &sprites, &camera());
    renderer.render_meshes(&mut encoder, &gpu.queue, &view, &atlas, &meshes, &camera(), MeshBlend::Alpha);
    renderer.render_lights(&gpu.queue, &mut encoder, &view, &camera(), &lights);
    particles.render(&gpu.queue, &mut encoder, &view, &camera(), &mut renderer);
    gpu.queue.submit(Some(encoder.finish()));

    let stats = *renderer.stats();
    assert!(stats.fps > 0.0, "fps должен быть > 0 после begin_frame(1/60), а это {}", stats.fps);
    assert!(stats.sprites >= 2, "два базовых спрайта, а их {}", stats.sprites);
    assert!(stats.fx_sprites >= 1, "один fx-спрайт, а их {}", stats.fx_sprites);
    assert!(stats.lights >= 1, "один свет, а их {}", stats.lights);
    assert!(stats.particles >= 1, "одна частица, а их {}", stats.particles);
    assert!(stats.draw_calls >= 3, "минимум 3 draw call (спрайты/меш/свет/частицы), а их {}", stats.draw_calls);
    assert_eq!(stats.meshes, 1, "один меш");
    assert!(stats.mesh_indices >= 3, "квад — 6 индексов, а их {}", stats.mesh_indices);
}

// ── 4. Оверлей: текст рисует пиксели в верхнем левом углу ──────────────────

#[test]
fn overlay_text_draws_pixels() {
    let Some(gpu) = gpu() else { return };
    let empty_atlas = SpriteAtlas::new(&gpu.device, 16);
    let font = DebugFont::new(&gpu.device, &gpu.queue);

    let (tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("overlay encoder") });
    // Пустой спрайтовый проход даёт заливку; поверх — строка "Hi" (отступ 8 px).
    let empty_batch = SpriteBatch::new(&gpu.device, 1);
    renderer.render(&gpu.queue, &mut encoder, &view, &empty_atlas, &empty_batch, &camera());
    renderer.render_debug_overlay(
        &mut encoder,
        &gpu.queue,
        &view,
        &font,
        &Camera::new([W as f32, H as f32]),
        &["Hi".to_string()],
    );
    gpu.queue.submit(Some(encoder.finish()));
    let data = readback(&gpu, &tex, W, H);

    // В области текста (верхние 32×32 пикселя) должны быть белые глифы —
    // заметно ярче тёмно-синей заливки.
    let mut lit = 0;
    for y in 0..32 {
        for x in 0..32 {
            if sum_rgb(px(&data, W, x, y)) > 300 {
                lit += 1;
            }
        }
    }
    assert!(lit > 5, "в верхнем левом углу должны быть белые пиксели текста, а их {lit}");
}

// ── 5. Wireframe-меши: пайплайн строится, рендер не паникует ───────────────

#[test]
fn wireframe_mesh_pipeline() {
    let Some(gpu) = gpu() else { return };
    let device = &gpu.device;
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let atlas = white_atlas(device, &gpu.queue);
    let mut renderer = Renderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    renderer.begin_frame(1.0 / 60.0);
    renderer.debug.wireframe_meshes = true;

    let mut meshes = MeshBatch::new(device, 16, 32);
    meshes.push(&Mesh::quad([32.0, 32.0], [20.0, 20.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.3, 0.3, 1.0]));
    meshes.upload(&gpu.queue);

    let (_tex, view) = target(device, W, H);
    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wireframe encoder") });
    let empty_batch = SpriteBatch::new(device, 1);
    renderer.render(&gpu.queue, &mut encoder, &view, &atlas, &empty_batch, &camera());
    renderer.render_meshes(&mut encoder, &gpu.queue, &view, &atlas, &meshes, &camera(), MeshBlend::Alpha);
    gpu.queue.submit(Some(encoder.finish()));
    gpu.device.poll(wgpu::Maintain::Wait);

    let err = pollster::block_on(device.pop_error_scope());
    assert!(err.is_none(), "валидационная ошибка при wireframe-рендере: {err:?}");
    assert_eq!(renderer.stats().meshes, 1, "меш должен быть посчитан в статистике");
}

// ── 6. Шрифт: атлас непустой, глифы различимы, меш строится ────────────────

#[test]
fn font_glyphs_sane() {
    let Some(gpu) = gpu() else { return };
    let font = DebugFont::new(&gpu.device, &gpu.queue);

    assert!(!font.atlas.is_empty(), "атлас шрифта должен содержать регионы глифов");
    assert_eq!(font.atlas.len(), 95, "ровно 95 глифов ASCII 32..=126");

    let a = font.glyph_uv('A');
    let b = font.glyph_uv('B');
    assert_ne!(a, b, "'A' и 'B' — разные UV-прямоугольники");
    assert_eq!(font.glyph_uv('ф'), font.glyph_uv('?'), "не-ASCII символ — на '?'");

    let mesh = font.text_mesh("Hi", 0.0, 0.0, 1.0, [1.0, 1.0, 1.0, 1.0], [W as f32, H as f32]);
    assert!(mesh.vertices.len() > 0, "у видимого текста должны быть вершины");
    assert_eq!(mesh.indices.len(), mesh.vertices.len() * 6 / 4, "квады по 4 вершины / 6 индексов");
}
