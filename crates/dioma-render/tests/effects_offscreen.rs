//! Интеграционные тесты эффектов движка: fx-шейдеры, свет, частицы, пост-обработка.
//!
//! Все тесты offscreen (64×64, RGBA8UnormSrgb), pollster::block_on, та же схема
//! fallback-адаптера, что в `render_offscreen.rs`; без GPU — пропуск.

mod common;

use std::sync::Arc;

use common::{camera, gpu, navy_clear, px, readback, sum_rgb, target, H, W};
use dioma_render::{
    atlas, Light, LightBatch, Particle, ParticleSystem, PostFx, PostProcess, Renderer, Sprite,
    SpriteAtlas, SpriteBatch, SpriteFx,
};

/// Белый круг в атласе (radius 16 → текстура 34×34).
fn white_circle_atlas(device: &Arc<wgpu::Device>, queue: &wgpu::Queue) -> (SpriteAtlas, [f32; 4]) {
    let mut atlas = SpriteAtlas::new(device, 256);
    let (tw, th, pixels) = atlas::circle_rgba(16, [255, 255, 255, 255]);
    atlas.add_rgba(queue, "circle", tw, th, &pixels).expect("круг должен влезть в атлас 256²");
    let uv = atlas.region("circle").unwrap();
    (atlas, uv)
}

/// Спрайт-круг 40×40 по центру кадра.
fn centered_circle_sprite(uv: [f32; 4]) -> Sprite {
    Sprite::new([W as f32 * 0.5, H as f32 * 0.5], [40.0, 40.0], uv)
}

/// Рендерит батч в 64×64 offscreen-цель; возвращает пиксели RGBA8.
fn render_batch(
    gpu: &common::TestGpu,
    batch: &mut SpriteBatch,
    atlas: &SpriteAtlas,
) -> Vec<u8> {
    batch.upload(&gpu.queue);
    let (tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fx encoder") });
    renderer.render(&gpu.queue, &mut encoder, &view, atlas, batch, &camera());
    gpu.queue.submit(Some(encoder.finish()));
    readback(gpu, &tex, W, H)
}

// ── 1. Все пайплайны строятся без валидационных ошибок ─────────────────────

#[test]
fn all_pipelines_build() {
    let Some(gpu) = gpu() else { return };
    let device = &gpu.device;

    // Ловим валидационные ошибки (создание пайплайнов + отрисовка всеми).
    device.push_error_scope(wgpu::ErrorFilter::Validation);

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut renderer = Renderer::new(device, format);
    renderer.set_time(1.25);
    renderer.set_clear_color(navy_clear());

    let mut post = PostProcess::new(device, format);
    post.resize(device, W, H);

    let (atlas, uv) = white_circle_atlas(device, &gpu.queue);

    // Батч: по спрайту на каждый вариант SpriteFx + без эффекта.
    let mut batch = SpriteBatch::new(device, 8);
    batch.push(centered_circle_sprite(uv).with_fx(SpriteFx::None));
    batch.push(
        centered_circle_sprite(uv).with_fx(SpriteFx::Outline { width: 2.0, color: [1.0, 0.2, 0.2, 1.0] }),
    );
    batch.push(
        centered_circle_sprite(uv)
            .with_fx(SpriteFx::Glow { radius: 6.0, intensity: 0.5, color: [1.0, 1.0, 0.4, 1.0] }),
    );
    batch.push(
        centered_circle_sprite(uv)
            .with_fx(SpriteFx::Dissolve { progress: 0.5, edge_width: 0.05, edge_color: [1.0, 0.6, 0.1, 1.0] }),
    );
    batch.push(
        centered_circle_sprite(uv).with_fx(SpriteFx::Wobble { amplitude: 2.0, frequency: 0.5, speed: 3.0 }),
    );
    batch.upload(&gpu.queue);

    // Свет и частицы.
    let mut lights = LightBatch::new(device, 2);
    lights.push(Light { position: [32.0, 32.0], radius: 40.0, intensity: 0.5, color: [1.0, 1.0, 1.0] });
    lights.upload(&gpu.queue);
    let mut particles = ParticleSystem::new(device, 16);
    particles.spawn(Particle::new([32.0, 32.0], [0.0, 0.0], 1.0, 20.0, [1.0, 1.0, 0.8, 1.0]));

    // Рисуем мир в сцену пост-процессора, затем все пост-эффекты.
    let (_, post_view) = target(device, W, H);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("all pipelines encoder") });
    renderer.render(&gpu.queue, &mut encoder, post.scene_view(), &atlas, &batch, &camera());
    renderer.render_lights(&gpu.queue, &mut encoder, post.scene_view(), &camera(), &lights);
    particles.render(&gpu.queue, &mut encoder, post.scene_view(), &camera(), &mut renderer);
    for fx in [
        PostFx::None,
        PostFx::VignetteGrain { vignette: 0.3, grain: 0.05 },
        PostFx::Blur { radius: 1.5 },
    ] {
        post.apply(&gpu.queue, &mut encoder, &post_view, fx, 1.25);
    }
    gpu.queue.submit(Some(encoder.finish()));
    gpu.device.poll(wgpu::Maintain::Wait);

    let err = pollster::block_on(gpu.device.pop_error_scope());
    assert!(err.is_none(), "валидационная ошибка при сборке/рендере: {err:?}");
}

// ── 2. Контур: красное кольцо вокруг белого круга ──────────────────────────

#[test]
fn outline_pixels() {
    let Some(gpu) = gpu() else { return };
    let (atlas, uv) = white_circle_atlas(&gpu.device, &gpu.queue);

    let mut batch = SpriteBatch::new(&gpu.device, 1);
    batch.push(
        centered_circle_sprite(uv)
            .with_fx(SpriteFx::Outline { width: 2.0, color: [1.0, 0.0, 0.0, 1.0] }),
    );
    let data = render_batch(&gpu, &mut batch, &atlas);

    // Сразу за краем круга (~19.5px от центра, квад 40×40): красный контур.
    let ring = px(&data, W, 51, 32);
    assert!(ring[0] > 120, "у края должен быть красный контур, а это {ring:?}");
    assert!(ring[1] < 80, "контур должен быть красным, а это {ring:?}");
    assert!(ring[2] < 80, "контур должен быть красным, а это {ring:?}");

    // Центр: спрайт остался белым.
    let center = px(&data, W, 32, 32);
    assert!(center[0] > 200 && center[1] > 200 && center[2] > 200, "центр должен быть белым, а это {center:?}");
}

// ── 3. Растворение: прогресс 0 — цел, 0.99 — почти всё исчезло ─────────────

#[test]
fn dissolve_progress() {
    let Some(gpu) = gpu() else { return };
    let (atlas, uv) = white_circle_atlas(&gpu.device, &gpu.queue);

    let count_colored = |progress: f32| -> usize {
        let mut batch = SpriteBatch::new(&gpu.device, 1);
        batch.push(
            centered_circle_sprite(uv).with_fx(SpriteFx::Dissolve {
                progress,
                edge_width: 0.05,
                edge_color: [1.0, 0.6, 0.1, 1.0],
            }),
        );
        let data = render_batch(&gpu, &mut batch, &atlas);
        // Маленькая область вокруг центра: сколько «цветных» пикселей.
        let mut count = 0;
        for y in 16..48 {
            for x in 16..48 {
                if sum_rgb(px(&data, W, x, y)) > 250 {
                    count += 1;
                }
            }
        }
        count
    };

    let c0 = count_colored(0.0);
    let c99 = count_colored(0.99);
    assert!(c0 > 600, "при progress=0 круг должен быть целым, а цветных пикселей {c0}");
    assert!(c99 < 60, "при progress=0.99 почти всё должно раствориться, а цветных {c99}");
    assert!(c99 < c0 / 2, "растворённый спрайт должен быть заметно пустее: {c99} vs {c0}");
}

// ── 4. Свечение: у края спрайта ярче, чем без эффекта ──────────────────────

#[test]
fn glow_brightens_around() {
    let Some(gpu) = gpu() else { return };
    let (atlas, uv) = white_circle_atlas(&gpu.device, &gpu.queue);

    // Без эффекта.
    let mut plain = SpriteBatch::new(&gpu.device, 1);
    plain.push(centered_circle_sprite(uv));
    let data_plain = render_batch(&gpu, &mut plain, &atlas);

    // Со свечением.
    let mut glowing = SpriteBatch::new(&gpu.device, 1);
    glowing.push(
        centered_circle_sprite(uv)
            .with_fx(SpriteFx::Glow { radius: 3.0, intensity: 0.8, color: [1.0, 1.0, 0.0, 1.0] }),
    );
    let data_glow = render_batch(&gpu, &mut glowing, &atlas);

    // Точка у края квада (вне круга): свечение заметно добавляет яркости.
    let a = sum_rgb(px(&data_plain, W, 51, 32));
    let b = sum_rgb(px(&data_glow, W, 51, 32));
    assert!(b > a + 100, "свечение должно осветлять край: {b} vs {a}");
}

// ── 5. Частицы: аддитивный мягкий круг ярче заливки ────────────────────────

#[test]
fn particles_additive() {
    let Some(gpu) = gpu() else { return };
    let mut particles = ParticleSystem::new(&gpu.device, 8);
    particles.spawn(Particle::new([32.0, 32.0], [0.0, 0.0], 2.0, 24.0, [1.0, 1.0, 0.8, 1.0]));

    let (tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("particles encoder") });
    renderer.render(&gpu.queue, &mut encoder, &view, &make_empty_atlas(&gpu), &SpriteBatch::new(&gpu.device, 1), &camera());
    particles.render(&gpu.queue, &mut encoder, &view, &camera(), &mut renderer);
    gpu.queue.submit(Some(encoder.finish()));
    let data = readback(&gpu, &tex, W, H);

    let center = px(&data, W, 32, 32);
    assert!(center[0] > 150, "центр частицы должен быть ярким (аддитив), а это {center:?}");
    assert!(center[1] > 150, "центр частицы должен быть ярким (аддитив), а это {center:?}");
}

/// Пустой атлас для проходов, где он не нужен (заглушка для render()).
fn make_empty_atlas(gpu: &common::TestGpu) -> SpriteAtlas {
    SpriteAtlas::new(&gpu.device, 16)
}

// ── 6. Свет: центр ярче угла (аддитивный falloff) ──────────────────────────

#[test]
fn light_adds() {
    let Some(gpu) = gpu() else { return };
    let mut lights = LightBatch::new(&gpu.device, 1);
    lights.push(Light { position: [32.0, 32.0], radius: 40.0, intensity: 1.0, color: [1.0, 1.0, 1.0] });
    lights.upload(&gpu.queue);

    let (tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("light encoder") });
    renderer.render(&gpu.queue, &mut encoder, &view, &make_empty_atlas(&gpu), &SpriteBatch::new(&gpu.device, 1), &camera());
    renderer.render_lights(&gpu.queue, &mut encoder, &view, &camera(), &lights);
    gpu.queue.submit(Some(encoder.finish()));
    let data = readback(&gpu, &tex, W, H);

    let center = sum_rgb(px(&data, W, 32, 32));
    let corner = sum_rgb(px(&data, W, 2, 2));
    assert!(center > corner + 100, "центр света должен быть ярче угла: {center} vs {corner}");
}

// ── 7. Виньетка: углы темнее центра ────────────────────────────────────────

#[test]
fn vignette_darkens_corners() {
    let Some(gpu) = gpu() else { return };
    let device = &gpu.device;

    // Атлас: белая полоса-квадрат (сплошная) для полноэкранного спрайта.
    let mut atlas = SpriteAtlas::new(device, 256);
    let (tw, th, pixels) = atlas::bar_rgba(16, 16, [255, 255, 255, 255], [255, 255, 255, 255], 1.0);
    atlas.add_rgba(&gpu.queue, "white", tw, th, &pixels).unwrap();
    let uv = atlas.region("white").unwrap();

    // Полноэкранный белый квад.
    let mut batch = SpriteBatch::new(device, 1);
    batch.push(Sprite::new([32.0, 32.0], [64.0, 64.0], uv));
    batch.upload(&gpu.queue);

    let mut post = PostProcess::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    post.resize(device, W, H);
    let mut renderer = Renderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());

    let (post_target, post_view) = target(device, W, H);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vignette encoder") });
    renderer.render(&gpu.queue, &mut encoder, post.scene_view(), &atlas, &batch, &camera());
    post.apply(&gpu.queue, &mut encoder, &post_view, PostFx::VignetteGrain { vignette: 0.8, grain: 0.0 }, 0.0);
    gpu.queue.submit(Some(encoder.finish()));
    let data = readback(&gpu, &post_target, W, H);

    let center = sum_rgb(px(&data, W, 32, 32));
    let corner = sum_rgb(px(&data, W, 2, 2));
    assert!(center > 600, "центр должен остаться белым, а это {center}");
    assert!(corner < center, "углы должны затемняться виньеткой: {corner} vs {center}");
}

// ── N. Drop-тень силуэта (`SpriteFx::Shadow`) ──────────────────────────────

/// Тень: мягкое затемнение ПОД спрайтом (смещение вниз экрана), тело чистое.
#[test]
fn shadow_offset_blur() {
    let Some(gpu) = gpu() else { return };
    let (atlas, uv) = white_circle_atlas(&gpu.device, &gpu.queue);
    let mut batch = SpriteBatch::new(&gpu.device, 8);
    // offset (0, -6): uv y = вверх мира = вверх экрана → тень на 6 px НИЖЕ тела.
    batch.push(Sprite::new([W as f32 * 0.5, H as f32 * 0.5], [40.0, 40.0], uv).with_fx(SpriteFx::Shadow {
        offset: [0.0, -6.0],
        radius: 3.0,
        color: [0.0, 0.0, 0.0, 0.9],
    }));
    let data = render_batch(&gpu, &mut batch, &atlas);

    // Тело спрайта — белое (тень под ним не видна).
    let body = px(&data, W, W / 2, H / 2);
    assert!(sum_rgb(body) > 600, "тело белое: {body:?}");
    // Ниже тела (y=центр+14) — тень: темнее navy, но не тело.
    let below = px(&data, W, W / 2, H / 2 + 23);
    assert!(sum_rgb(below) < sum_rgb(body) - 300 && sum_rgb(below) > 80, "под телом тень: {below:?}");
    // Выше тела (y=центр−14) — тени нет (offset ушёл вниз), чистый navy.
    let above = px(&data, W, W / 2, H / 2 - 23);
    assert!(sum_rgb(above) < 200, "над телом чистый фон (navy ~176): {above:?}");
}

// ── N+1. Обесцвечивание (`SpriteFx::Grayscale`) ────────────────────────────

#[test]
fn grayscale_desaturates() {
    let Some(gpu) = gpu() else { return };
    // Красный круг.
    let mut atlas = SpriteAtlas::new(&gpu.device, 256);
    let (tw, th, pixels) = atlas::circle_rgba(16, [255, 40, 40, 255]);
    atlas.add_rgba(&gpu.queue, "red", tw, th, &pixels).expect("влезает");
    let uv = atlas.region("red").unwrap();

    // amount=1: красный канал уравнивается с остальными (серый).
    let mut batch = SpriteBatch::new(&gpu.device, 8);
    batch.push(Sprite::new([W as f32 * 0.5, H as f32 * 0.5], [40.0, 40.0], uv).with_fx(SpriteFx::Grayscale { amount: 1.0 }));
    let data = render_batch(&gpu, &mut batch, &atlas);
    let gray = px(&data, W, W / 2, H / 2);
    assert!(
        gray[0] > 100 && (gray[0] as i32 - gray[1] as i32).abs() < 8 && (gray[1] as i32 - gray[2] as i32).abs() < 8,
        "полностью серый: {gray:?}"
    );

    // amount=0: без изменений — красный доминирует.
    let mut batch2 = SpriteBatch::new(&gpu.device, 8);
    batch2.push(Sprite::new([W as f32 * 0.5, H as f32 * 0.5], [40.0, 40.0], uv).with_fx(SpriteFx::Grayscale { amount: 0.0 }));
    let data2 = render_batch(&gpu, &mut batch2, &atlas);
    let red = px(&data2, W, W / 2, H / 2);
    assert!(red[0] > 180 && red[0] > red[1] + 80, "красный остался красным: {red:?}");
}

// ── N+2. Вертикальный градиент tint ────────────────────────────────────────

#[test]
fn gradient_tint_top_bottom() {
    let Some(gpu) = gpu() else { return };
    let (atlas, uv) = white_circle_atlas(&gpu.device, &gpu.queue);
    let mut batch = SpriteBatch::new(&gpu.device, 8);
    batch.push(
        Sprite::new([W as f32 * 0.5, H as f32 * 0.5], [40.0, 40.0], uv)
            .with_gradient([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
    );
    let data = render_batch(&gpu, &mut batch, &atlas);

    // Верх круга — красный, низ — синий (центральный столбец).
    let top = px(&data, W, W / 2, H / 2 - 10);
    let bottom = px(&data, W, W / 2, H / 2 + 10);
    assert!(top[0] > 150 && top[0] > top[2] + 80, "верх красный: {top:?}");
    assert!(bottom[2] > 150 && bottom[2] > bottom[0] + 80, "низ синий: {bottom:?}");
}

/// Градиент без разницы цветов == обычный tint (регрессия: не темнить низ).
#[test]
fn gradient_equal_is_uniform() {
    let Some(gpu) = gpu() else { return };
    let (atlas, uv) = white_circle_atlas(&gpu.device, &gpu.queue);
    let mut batch = SpriteBatch::new(&gpu.device, 8);
    batch.push(Sprite::new([W as f32 * 0.5, H as f32 * 0.5], [40.0, 40.0], uv));
    let data = render_batch(&gpu, &mut batch, &atlas);
    let top = px(&data, W, W / 2, H / 2 - 10);
    let bottom = px(&data, W, W / 2, H / 2 + 10);
    assert_eq!(top, bottom, "без with_gradient цвет однородный");
}
