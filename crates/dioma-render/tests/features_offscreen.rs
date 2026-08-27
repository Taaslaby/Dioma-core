//! Offscreen-тесты новых возможностей: world_to_screen, отсечение спрайтов и
//! тайлов, scissor-рект, debug-шейпы, тьма/ночь (PostFx::Darkness), мипмапы атласа.
//!
//! Без GPU все тесты тривиально проходят (см. `common::gpu`).

mod common;

use common::{camera, gpu, navy_clear, px, readback, target, H, W};
use dioma_render::{
    atlas, Camera, PostFx, PostProcess, Renderer, ScissorRect, Sprite, SpriteAtlas, SpriteBatch,
    TileBatch, Tilemap, TilesetUv,
};

/// Чисто CPU: world_to_screen — обратная к screen_to_world (пикинг в обе стороны).
#[test]
fn camera_world_to_screen_roundtrip() {
    let cam = Camera {
        center: [123.0, -45.0],
        zoom: 2.5,
        viewport: [800.0, 600.0],
    };
    let screen = [333.0, 111.0];
    let world = cam.screen_to_world(screen);
    let back = cam.world_to_screen(world);
    assert!(
        (back[0] - screen[0]).abs() < 0.001,
        "X не сошёлся: {back:?} vs {screen:?}"
    );
    assert!(
        (back[1] - screen[1]).abs() < 0.001,
        "Y не сошёлся: {back:?} vs {screen:?}"
    );

    // visible_world_rect: прямоугольник видимой области вокруг центра.
    let rect = cam.visible_world_rect();
    let half_w = 800.0 * 0.5 / 2.5;
    let half_h = 600.0 * 0.5 / 2.5;
    assert!((rect[0] - (123.0 - half_w)).abs() < 0.001);
    assert!((rect[1] - (-45.0 - half_h)).abs() < 0.001);
    assert!((rect[2] - (123.0 + half_w)).abs() < 0.001);
    assert!((rect[3] - (-45.0 + half_h)).abs() < 0.001);
}

/// Отсечение спрайтов: на GPU уходит только то, что в прямоугольнике.
#[test]
fn sprite_batch_culls_to_visible() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 128);
    let (w, h, px_rgba) = atlas::circle_rgba(8, [255, 255, 255, 255]);
    atlas
        .add_rgba(&gpu.queue, "c", w, h, &px_rgba)
        .expect("круг влезает");
    let uv = atlas.region("c").unwrap();

    let mut batch = SpriteBatch::new(&gpu.device, 16);
    for y in 0..4u32 {
        for x in 0..4u32 {
            batch.push(Sprite::new(
                [16.0 + x as f32 * 8.0, 16.0 + y as f32 * 8.0],
                [8.0, 8.0],
                uv,
            ));
        }
    }
    assert_eq!(batch.len(), 16);

    // Центры 16..40 с шагом 8; прямоугольник [0,28] пропускает 16 и 24 → 2×2.
    batch.set_cull_rect(Some([0.0, 0.0, 28.0, 28.0]));
    batch.upload(&gpu.queue);
    assert_eq!(batch.len_drawn(), 4, "в прямоугольник попадают центры 2×2");

    let (target_tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });
    renderer.render(&gpu.queue, &mut encoder, &view, &atlas, &batch, &camera());
    gpu.queue.submit(Some(encoder.finish()));
    // В статистике кадра — только отсечённые 4 спрайта.
    assert_eq!(
        renderer.stats().sprites,
        4,
        "отсечённые спрайты не рисуются"
    );
    let data = readback(&gpu, &target_tex, W, H);
    // Спрайт с центром (16,16) остался в списке — на экране он в (16,48)
    // из-за Y-флипа (мировая Y вверх = верх экрана).
    let kept = px(&data, W, 16, 48);
    assert!(kept[0] > 150, "уцелевший спрайт белый: {kept:?}");
    // А спрайт с центром (32,32) отсечён — под ним заливка.
    let culled = px(&data, W, W / 2, H / 2);
    assert!(culled[0] < 80, "отсечённый спрайт не рисуется: {culled:?}");

    // Возврат к полному списку — всё рисуется.
    batch.set_cull_rect(None);
    batch.upload(&gpu.queue);
    assert_eq!(batch.len_drawn(), 16);
}

/// Отсечение тайлов: upload_visible заливает только пересекающие кадр.
#[test]
fn tile_batch_upload_visible_counts() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 128);
    let (w, h, px_rgba) = atlas::circle_rgba(8, [255, 255, 255, 255]);
    atlas
        .add_rgba(&gpu.queue, "tile", w, h, &px_rgba)
        .expect("тайл влезает");

    let mut map = Tilemap::new(8, 8, 8.0);
    map.fill_rect(0, 0, 7, 7, 1); // всё заполнено
    let tileset = TilesetUv::new([0.0, 0.0, 1.0, 1.0], 1, 1);
    let mut batch = TileBatch::new(&gpu.device, &map, &tileset);
    assert_eq!(batch.len(), 64, "все непустые тайлы");

    batch.upload(&gpu.queue);
    assert_eq!(batch.len_visible(), 64, "без отсечения — всё");

    // Окошко 24×24 в левом верхнем углу: с расширением на полтайла границы
    // [-4,28], центры 4,12,20,28 → 4×4 = 16 тайлов.
    batch.upload_visible(&gpu.queue, [0.0, 0.0, 24.0, 24.0]);
    assert_eq!(batch.len_visible(), 16, "видны 4×4 = 16 тайлов");
}

/// Scissor: рисуется только правая половина цели, левая остаётся заливкой.
#[test]
fn scissor_clip_renders_inside_only() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 128);
    // Белый texel 4×4: заливаем им весь квад (весь атлас почти прозрачен);
    // 1×1 с linear-фильтром даёт пригашенный цвет на краях квада.
    let white_px = vec![255u8; 4 * 4 * 4];
    atlas
        .add_rgba(&gpu.queue, "white", 4, 4, &white_px)
        .expect("белый texel влезает");

    let mut batch = SpriteBatch::new(&gpu.device, 1);
    batch.push(Sprite::new(
        [32.0, 32.0],
        [64.0, 64.0],
        atlas.region("white").unwrap(),
    ));
    batch.upload(&gpu.queue);

    let (target_tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });
    // Обрезаем правую половину цели.
    renderer.render_with_clip(
        &gpu.queue,
        &mut encoder,
        &view,
        &atlas,
        &batch,
        &camera(),
        Some(ScissorRect {
            x: 32,
            y: 0,
            width: 32,
            height: 64,
        }),
    );
    gpu.queue.submit(Some(encoder.finish()));

    let data = readback(&gpu, &target_tex, W, H);
    let left = px(&data, W, 16, 32);
    assert!(left[0] < 80, "левая половина — заливка: {left:?}");
    let right = px(&data, W, 48, 32);
    assert!(right[0] > 150, "правая половина — белый спрайт: {right:?}");
}

/// Debug-шейпы: зелёная линия видна по центру, углы — чистая заливка.
#[test]
fn debug_shapes_draw_pixels() {
    let Some(gpu) = gpu() else { return };
    let (target_tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());

    renderer.debug_line([8.0, 32.0], [56.0, 32.0], 4.0, [0.0, 1.0, 0.0, 1.0]);
    renderer.debug_circle([32.0, 32.0], 4.0, 16, [1.0, 1.0, 1.0, 1.0]);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });
    renderer.render_debug_shapes(&mut encoder, &gpu.queue, &view, &camera());
    // Повторный вызов с пустым накопителем — no-op: draw_calls не растёт.
    let calls_before = renderer.stats().draw_calls;
    renderer.render_debug_shapes(&mut encoder, &gpu.queue, &view, &camera());
    assert_eq!(
        renderer.stats().draw_calls,
        calls_before,
        "пустой накопитель не рисует"
    );
    gpu.queue.submit(Some(encoder.finish()));

    let data = readback(&gpu, &target_tex, W, H);
    let center = px(&data, W, W / 2, H / 2);
    assert!(center[1] > 150, "центр — зелёная линия: {center:?}");
    assert!(
        center[0] < 120,
        "центр — зелёная линия (r низкий): {center:?}"
    );
    let corner = px(&data, W, 2, 2);
    assert!(corner[1] < 80, "угол — чистая заливка: {corner:?}");
}

/// Тьма/ночь: PostFx::Darkness с strength=1 умножает сцену на tint.
#[test]
fn darkness_tint_dims_scene() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 128);
    let (w, h, px_rgba) = atlas::circle_rgba(8, [255, 60, 60, 255]);
    atlas
        .add_rgba(&gpu.queue, "c", w, h, &px_rgba)
        .expect("круг влезает");

    let mut batch = SpriteBatch::new(&gpu.device, 1);
    let mut sprite = Sprite::new([32.0, 32.0], [64.0, 64.0], atlas.region("c").unwrap());
    sprite.tint = [1.0, 0.0, 0.0, 1.0];
    batch.push(sprite);
    batch.upload(&gpu.queue);

    let (target_tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut post = PostProcess::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    post.resize(&gpu.device, W, H);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });
    // Сцена рисуется в промежуточную текстуру пост-процессора.
    renderer.render(
        &gpu.queue,
        &mut encoder,
        post.scene_view(),
        &atlas,
        &batch,
        &camera(),
    );
    // Тьма: tint 0.5 серый, сила 1 → красный 255→~188 (sRGB от линейного 0.5).
    post.apply(
        &gpu.queue,
        &mut encoder,
        &view,
        PostFx::Darkness {
            tint: [0.5, 0.5, 0.5],
            strength: 1.0,
        },
        0.0,
    );
    gpu.queue.submit(Some(encoder.finish()));

    let data = readback(&gpu, &target_tex, W, H);
    let center = px(&data, W, W / 2, H / 2);
    assert!(
        (150..=230).contains(&center[0]),
        "центр затемнён наполовину: {center:?}"
    );
    let corner = px(&data, W, 2, 2);
    assert!(corner[0] < center[0], "угол темнее центра: {corner:?}");
}

/// Мипмапы атласа: включаются до добавления, атлас получает 4 уровня и
/// отрисовка на малом зуме не падает и даёт цветные регионы.
#[test]
fn atlas_mipmap_pipeline_works() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 64);
    atlas
        .enable_mipmaps()
        .expect("мипмапы включаются на пустом атласе");
    assert_eq!(atlas.texture().mip_level_count(), 4, "64² → 4 уровня (кап)");

    let (gw, gh, green) = atlas::circle_rgba(8, [40, 200, 40, 255]);
    let (mw, mh, magenta) = atlas::circle_rgba(8, [200, 40, 200, 255]);
    atlas
        .add_rgba(&gpu.queue, "green", gw, gh, &green)
        .expect("зелёный влезает");
    atlas
        .add_rgba(&gpu.queue, "magenta", mw, mh, &magenta)
        .expect("магента влезает");

    let mut batch = SpriteBatch::new(&gpu.device, 2);
    batch.push(Sprite::new(
        [24.0, 20.0],
        [20.0, 20.0],
        atlas.region("green").unwrap(),
    ));
    batch.push(Sprite::new(
        [24.0, 44.0],
        [20.0, 20.0],
        atlas.region("magenta").unwrap(),
    ));
    batch.upload(&gpu.queue);

    let (target_tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    // Малый зум → сэмплируются глубокие мипы (без мерцания не проверить, но
    // пиплан должен отработать и цвета не «съесть»).
    let mut cam = camera();
    cam.zoom = 0.2;

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });
    renderer.render(&gpu.queue, &mut encoder, &view, &atlas, &batch, &cam);
    gpu.queue.submit(Some(encoder.finish()));

    let data = readback(&gpu, &target_tex, W, H);
    let has_green = (0..H as usize).step_by(2).any(|y| {
        (0..W as usize).step_by(2).any(|x| {
            let p = px(&data, W, x as u32, y as u32);
            p[1] > 90 && p[1] > p[0] + 30
        })
    });
    assert!(has_green, "зелёный регион виден на глубоком мипе");
}
