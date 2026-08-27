//! Интеграционный тест: offscreen-рендер спрайта в текстуру и чтение пикселей.
//!
//! Работает на локальном GPU «как бы без окна» (offscreen-текстура вместо
//! поверхности). Без GPU тест тривиально проходит: при отсутствии адаптера
//! печатает предупреждение и выходит, чтобы CI без видеокарты не падал.

mod common;

use common::{camera, gpu, navy_clear, px, readback, target, W};
use dioma_render::{atlas, Renderer, Sprite, SpriteAtlas, SpriteBatch};

#[test]
fn offscreen_sprite_center_red_on_navy() {
    let Some(gpu) = gpu() else { return };
    let (target_tex, view) = target(&gpu.device, W, common::H);

    // ── Атлас: один процедурный круг ──
    let mut atlas = SpriteAtlas::new(&gpu.device, 256);
    let (cw, ch, pixels) = atlas::circle_rgba(16, [255, 60, 60, 255]);
    atlas
        .add_rgba(&gpu.queue, "circle", cw, ch, &pixels)
        .expect("круг должен влезть в атлас 256²");
    let uv = atlas.region("circle").expect("круг зарегистрирован в атласе");

    // ── Батч: один красный спрайт по центру, размер 32×32 ──
    let mut batch = SpriteBatch::new(&gpu.device, 1);
    let mut sprite = Sprite::new([W as f32 * 0.5, common::H as f32 * 0.5], [32.0, 32.0], uv);
    sprite.tint = [1.0, 0.0, 0.0, 1.0];
    batch.push(sprite);
    batch.upload(&gpu.queue);

    // ── Рендер в offscreen-текстуру с тёмно-синей заливкой ──
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("offscreen encoder") });
    renderer.render(&gpu.queue, &mut encoder, &view, &atlas, &batch, &camera());
    gpu.queue.submit(Some(encoder.finish()));

    let data = readback(&gpu, &target_tex, W, common::H);

    // Центр: красный спрайт (r > 150, g < 80, b < 80).
    let center = px(&data, W, W / 2, common::H / 2);
    assert!(center[0] > 150, "центр должен быть красным, а это {center:?}");
    assert!(center[1] < 80, "центр должен быть красным, а это {center:?}");
    assert!(center[2] < 80, "центр должен быть красным, а это {center:?}");

    // Угол: тёмно-синяя заливка (линейный 0.02/0.03/0.10 → sRGB ≈ [39,48,89]).
    let corner = px(&data, W, 2, 2);
    assert!(corner[0] < 80, "угол должен быть заливкой, а это {corner:?}");
    assert!(corner[1] < 80, "угол должен быть заливкой, а это {corner:?}");
    assert!(corner[2] > 60, "угол должен быть синеватым, а это {corner:?}");
}
