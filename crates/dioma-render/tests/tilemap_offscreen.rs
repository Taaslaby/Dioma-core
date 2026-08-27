//! Интеграционные тесты тайлмапа и параллакса: сетка UV тайлсета,
//! TileBatch (рендер квадов тайлов через базовый спрайтовый проход),
//! пропуск пустых клеток, сдвиг параллакс-камеры и ошибки factor вне
//! диапазона.
//!
//! Всё offscreen (64×64, RGBA8UnormSrgb), та же схема fallback-адаптера и
//! ОДНОГО устройства на процесс (tests/common), что и в остальных тестах;
//! без GPU — пропуск.

mod common;

use std::panic::catch_unwind;

use common::{camera, gpu, navy_clear, px, readback, target, H, W};
use dioma_render::{atlas, Camera, Renderer, Sprite, SpriteAtlas, SpriteBatch, TileBatch, Tilemap, TilesetUv};

/// Атлас 32×32 с одним регионом «tiles» 16×16: сетка 2×2 из цветных ячеек
/// 8×8 (1 — красный верх-лево, 2 — зелёный верх-право, 3 — синий низ-лево,
/// 4 — жёлтый низ-право). Возвращает атлас + TilesetUv 2×2 над регионом.
fn tileset_atlas(gpu: &common::TestGpu) -> (SpriteAtlas, TilesetUv) {
    let mut atlas = SpriteAtlas::new(&gpu.device, 32);
    let colors: [[u8; 4]; 4] = [
        [255, 0, 0, 255],   // 1 — красный (верхний левый угол региона)
        [0, 255, 0, 255],   // 2 — зелёный (верхний правый)
        [0, 0, 255, 255],   // 3 — синий (нижний левый)
        [255, 255, 0, 255], // 4 — жёлтый (нижний правый)
    ];
    let mut img = vec![0u8; 16 * 16 * 4];
    for y in 0..16 {
        for x in 0..16 {
            let cell = if y < 8 {
                if x < 8 {
                    0
                } else {
                    1
                }
            } else if x < 8 {
                2
            } else {
                3
            };
            let i = ((y * 16 + x) * 4) as usize;
            img[i..i + 4].copy_from_slice(&colors[cell]);
        }
    }
    atlas.add_rgba(&gpu.queue, "tiles", 16, 16, &img).unwrap();
    let region = atlas.region("tiles").unwrap();
    let tileset = TilesetUv::new(region, 2, 2);
    (atlas, tileset)
}

// ── 1. Tilemap::from_rows + TilesetUv::cell_uv (чистый CPU) ─────────────────

#[test]
fn tilemap_from_rows_and_uv() {
    // 3×2 паттерн из строк: cols из первой строки, row-major.
    let rows: [&[u16]; 2] = [&[1, 2, 3], &[0, 4, 5]];
    let map = Tilemap::from_rows(&rows, 16.0);
    assert_eq!(map.cols(), 3, "cols из первой строки");
    assert_eq!(map.rows(), 2);
    assert_eq!(map.tile_size(), 16.0);
    assert_eq!(map.world_size(), [48.0, 32.0]);
    assert_eq!(map.get(0, 0), 1);
    assert_eq!(map.get(1, 0), 2);
    assert_eq!(map.get(2, 0), 3);
    assert_eq!(map.get(0, 1), 0, "0 — пустая клетка");
    assert_eq!(map.get(2, 1), 5);

    // set + get.
    let mut map = Tilemap::new(3, 2, 16.0);
    map.set(0, 0, 7);
    map.set(2, 1, 9);
    assert_eq!(map.get(0, 0), 7);
    assert_eq!(map.get(2, 1), 9);

    // fill_rect: инклюзивно, кламп к границам.
    map.fill_rect(1, 0, 1, 1, 8);
    assert_eq!(map.get(1, 0), 8);
    assert_eq!(map.get(1, 1), 8);
    assert_eq!(map.get(0, 0), 7, "вне прямоугольника не трогаем");
    map.fill_rect(2, 0, 99, 99, 6); // x1/y1 клампятся к 2/1
    assert_eq!(map.get(2, 0), 6);
    assert_eq!(map.get(2, 1), 6);

    // from_rows со строками разной длины — паника с русским сообщением.
    let bad = catch_unwind(|| Tilemap::from_rows(&[&[1, 2], &[3]], 16.0));
    assert!(bad.is_err(), "строки разной длины должны паниковать");
    let msg = bad
        .unwrap_err()
        .downcast_ref::<String>()
        .map(|s| s.clone())
        .unwrap_or_default();
    assert!(msg.contains("длину"), "сообщение паники русское и осмысленное: {msg}");

    // TilesetUv: сетка 2×2 в регионе 0.5×0.5.
    let ts = TilesetUv::new([0.25, 0.25, 0.75, 0.75], 2, 2);
    assert_eq!(ts.cell_count(), 4);
    assert_eq!(ts.cell_uv(0), [0.0, 0.0, 1.0, 1.0], "индекс 0 = пустая клетка → весь атлас");
    let c1 = ts.cell_uv(1);
    let c2 = ts.cell_uv(2);
    let c3 = ts.cell_uv(3);
    assert_ne!(c1, c2, "ячейки 1 и 2 — разные прямоугольники");
    assert_ne!(c1, c3, "ячейки 1 и 3 — разные прямоугольники");
    assert_ne!(c2, c3, "ячейки 2 и 3 — разные прямоугольники");
    // Ячейка 1 — верхний левый угол сетки: u0 < u1, v0 < v1, квадрат.
    assert!(c1[0] < c1[2] && c1[1] < c1[3], "клетка 1: u0 < u1, v0 < v1, а это {c1:?}");
    assert!(
        (c1[2] - c1[0] - (c1[3] - c1[1])).abs() < 1e-6,
        "клетка 1 квадратная (ширина == высота), а это {c1:?}"
    );
    assert_eq!(c1, [0.25, 0.25, 0.5, 0.5], "клетка 1 — верхняя левая");
    assert_eq!(c2, [0.5, 0.25, 0.75, 0.5], "клетка 2 — верхняя правая");
    assert_eq!(c3, [0.25, 0.5, 0.5, 0.75], "клетка 3 — нижняя левая");
}

// ── 2. TileBatch рисует тайлы цветного блока; снаружи — заливка ────────────

#[test]
fn tile_batch_draws_tiles() {
    let Some(gpu) = gpu() else { return };
    let (atlas, tileset) = tileset_atlas(&gpu);

    // Тайлмап 4×4, тайл 16 px; цветной блок 2×2 из клеток (1,1)..(2,2):
    // 1 — красный, 2 — зелёный, 3 — синий, 4 — жёлтый.
    let mut map = Tilemap::new(4, 4, 16.0);
    map.set(1, 1, 1);
    map.set(2, 1, 2);
    map.set(1, 2, 3);
    map.set(2, 2, 4);

    let mut tile_batch = TileBatch::new(&gpu.device, &map, &tileset);
    assert_eq!(tile_batch.len(), 4, "4 непустых тайла из 16");
    tile_batch.upload(&gpu.queue);

    let (tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tilemap encoder") });
    renderer.render(&gpu.queue, &mut encoder, &view, &atlas, tile_batch.sprite_batch(), &camera());
    gpu.queue.submit(Some(encoder.finish()));
    let data = readback(&gpu, &tex, W, H);

    // Внутри блока — цвет тайла. Камера (32,32), зум 1: screen_x = world_x,
    // screen_y = 64 − world_y (см. sprite.wgsl: NDC Y вверх = верх кадра,
    // `(world.y − center.y)` без минуса). Ряд данных 0 — наименьший мировой
    // Y — рендерится в НИЗУ экрана, последний ряд — вверху.
    let red = px(&data, W, 24, 40); // клетка (1,1): world (24, 24) → экран (24, 40)
    assert!(red[0] > 150 && red[1] < 80 && red[2] < 80, "красный тайл (1,1), а это {red:?}");
    let green = px(&data, W, 40, 40); // клетка (2,1): world (40, 24) → экран (40, 40)
    assert!(green[1] > 150 && green[0] < 80 && green[2] < 80, "зелёный тайл (2,1), а это {green:?}");
    let blue = px(&data, W, 24, 24); // клетка (1,2): world (24, 40) → экран (24, 24)
    assert!(blue[2] > 150 && blue[0] < 80 && blue[1] < 80, "синий тайл (1,2), а это {blue:?}");
    let yellow = px(&data, W, 40, 24); // клетка (2,2): world (40, 40) → экран (40, 24)
    assert!(
        yellow[0] > 150 && yellow[1] > 150 && yellow[2] < 80,
        "жёлтый тайл (2,2), а это {yellow:?}"
    );
    let outside = px(&data, W, 8, 56); // клетка (0,3): world (8, 56) → экран (8, 56), пустая
    assert!(
        outside[2] > 60 && outside[0] < 80 && outside[1] < 80,
        "снаружи блока — тёмно-синяя заливка, а это {outside:?}"
    );
}

// ── 3. Пустые клетки не рисуются: len() и пиксели вне тайлов ───────────────

#[test]
fn tile_batch_skips_empty() {
    let Some(gpu) = gpu() else { return };
    let (atlas, tileset) = tileset_atlas(&gpu);

    let mut map = Tilemap::new(4, 4, 16.0);
    map.set(1, 1, 1); // красный
    map.set(2, 1, 2); // зелёный
    map.set(3, 3, 3); // синий

    let mut tile_batch = TileBatch::new(&gpu.device, &map, &tileset);
    assert_eq!(tile_batch.len(), 3, "из 16 клеток нарисовано ровно 3 тайла");
    tile_batch.upload(&gpu.queue);

    let (tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tilemap skip encoder") });
    renderer.render(&gpu.queue, &mut encoder, &view, &atlas, tile_batch.sprite_batch(), &camera());
    gpu.queue.submit(Some(encoder.finish()));
    let data = readback(&gpu, &tex, W, H);

    // Поставленные тайлы нарисованы: screen_x = world_x, screen_y = 64 − world_y.
    let red = px(&data, W, 24, 40); // (1,1): world (24, 24) → экран (24, 40)
    assert!(red[0] > 150 && red[1] < 80, "тайл (1,1) нарисован, а это {red:?}");
    let blue = px(&data, W, 56, 8); // (3,3): world (56, 56) → экран (56, 8)
    assert!(blue[2] > 150 && blue[0] < 80, "тайл (3,3) нарисован, а это {blue:?}");

    // Пустые клетки остались заливкой.
    let empty1 = px(&data, W, 8, 56); // (0,0): world (8, 8) → экран (8, 56)
    assert!(
        empty1[2] > 60 && empty1[0] < 80 && empty1[1] < 80,
        "пустая клетка (0,0) — заливка, а это {empty1:?}"
    );
    let empty2 = px(&data, W, 56, 24); // (3,2): world (56, 40) → экран (56, 24)
    assert!(
        empty2[2] > 60 && empty2[0] < 80 && empty2[1] < 80,
        "пустая клетка (3,2) — заливка, а это {empty2:?}"
    );
}

// ── 4. Параллакс сдвигает слой: красный ближе к центру, чем синий ──────────

#[test]
fn parallax_shifts_layers() {
    let Some(gpu) = gpu() else { return };
    // Белый атлас 8×8: цвет проходит от тинта спрайта.
    let mut atlas = SpriteAtlas::new(&gpu.device, 8);
    let (w, h, white) = atlas::bar_rgba(8, 8, [255, 255, 255, 255], [255, 255, 255, 255], 1.0);
    atlas.add_rgba(&gpu.queue, "white", w, h, &white).unwrap();
    let uv = [0.0, 0.0, 1.0, 1.0];

    // Камера пан-нута вправо: центр (48, 32) вместо (32, 32), вьюпорт 64×64.
    let cam = Camera { center: [48.0, 32.0], zoom: 1.0, viewport: [W as f32, H as f32] };

    // Оба спрайта — на одной мировой позиции (20, 32), размер 8×8.
    let mk = |tint: [f32; 4]| Sprite::new([20.0, 32.0], [8.0, 8.0], uv).with_tint(tint);

    // Обычный рендер (factor 1.0): синий. Экран: x = 32 + (20−48) = 4.
    let mut normal = SpriteBatch::new(&gpu.device, 1);
    normal.push(mk([0.0, 0.0, 1.0, 1.0]));
    normal.upload(&gpu.queue);

    // Параллакс (factor 0.5): красный. Камера слоя центр (24, 16):
    // x = 32 + (20−24) = 28 — ближе к центру экрана (32), чем синий.
    let mut layer = SpriteBatch::new(&gpu.device, 1);
    layer.push(mk([1.0, 0.0, 0.0, 1.0]));
    layer.upload(&gpu.queue);

    let (tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parallax encoder") });
    // Сначала обычный проход (очищает цель), затем параллакс-слой поверх (Load).
    renderer.render(&gpu.queue, &mut encoder, &view, &atlas, &normal, &cam);
    renderer
        .render_parallax(&mut encoder, &gpu.queue, &view, &atlas, &layer, &cam, 0.5)
        .expect("factor 0.5 в диапазоне");
    gpu.queue.submit(Some(encoder.finish()));
    let data = readback(&gpu, &tex, W, H);

    // Синий (обычный): колонка 4, ряд 32. Экран: screen_x = 32+(x−48),
    // screen_y = 32−(y−32) (Y вверх), поэтому world (20,32) → экран (4, 32).
    let blue = px(&data, W, 4, 32);
    assert!(blue[2] > 150 && blue[0] < 80, "обычный спрайт — синий, а это {blue:?}");
    // Красный (параллакс, factor 0.5): камера слоя центр (24, 16) →
    // world (20,32) → экран (32+(20−24), 32−(32−16)) = (28, 16).
    let red = px(&data, W, 28, 16);
    assert!(red[0] > 150 && red[1] < 80 && red[2] < 80, "параллакс-спрайт — красный, а это {red:?}");
    // Относительные позиции: красный между синим и центром экрана (32).
    assert!(4 < 28 && 28 < 32, "красная колонка должна быть между синей и центром");
    let red_dist = (28i32 - 32).abs();
    let blue_dist = (4i32 - 32).abs();
    assert!(red_dist < blue_dist, "параллакс-спрайт ближе к центру, чем обычный ({red_dist} vs {blue_dist})");
}

// ── 5. factor вне 0.0..=1.0 — ошибка; границы — ок ──────────────────────────

#[test]
fn parallax_bad_factor_errors() {
    let Some(gpu) = gpu() else { return };
    let atlas = SpriteAtlas::new(&gpu.device, 8);
    let batch = SpriteBatch::new(&gpu.device, 1);
    let (_tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("parallax err encoder") });
    let cam = camera();

    let err = renderer.render_parallax(&mut encoder, &gpu.queue, &view, &atlas, &batch, &cam, 1.5);
    assert!(err.is_err(), "factor 1.5 должен быть ошибкой");
    assert!(
        err.unwrap_err().contains("0.0..=1.0"),
        "сообщение об ошибке должно упоминать диапазон"
    );

    let err = renderer.render_parallax(&mut encoder, &gpu.queue, &view, &atlas, &batch, &cam, -0.1);
    assert!(err.is_err(), "factor −0.1 должен быть ошибкой");

    let err = renderer
        .render_parallax(&mut encoder, &gpu.queue, &view, &atlas, &batch, &cam, f32::NAN);
    assert!(err.is_err(), "NaN должен быть ошибкой");

    // Границы диапазона — валидны (ничего не рисуется: батч пустой).
    renderer.render_parallax(&mut encoder, &gpu.queue, &view, &atlas, &batch, &cam, 0.0).unwrap();
    renderer.render_parallax(&mut encoder, &gpu.queue, &view, &atlas, &batch, &cam, 1.0).unwrap();

    gpu.queue.submit(Some(encoder.finish()));
}

// ── 6. Регресс: мировая Y вверх = верх экрана (соглашение сайд-вью) ────────

/// Один красный спрайт ВЫШЕ центра камеры (world.y 48 > center.y 32) должен
/// отрендериться в ВЕРХНЕЙ половине изображения (ряды < 32), а
/// `Camera::screen_to_world` должен быть согласован с рендером: верх экрана
/// (ряд 16) — это больший Y мира, низ (ряд 48) — меньший.
#[test]
fn world_up_is_screen_up() {
    let Some(gpu) = gpu() else { return };

    // Белый атлас 8×8 → цвет спрайта проходит от тинта.
    let mut atlas = SpriteAtlas::new(&gpu.device, 8);
    let (w, h, white) = atlas::bar_rgba(8, 8, [255, 255, 255, 255], [255, 255, 255, 255], 1.0);
    atlas.add_rgba(&gpu.queue, "white", w, h, &white).unwrap();
    let uv = [0.0, 0.0, 1.0, 1.0];

    // Камера: центр (32, 32), вьюпорт 64×64, зум 1 — как common::camera().
    let cam = camera();

    // Красный спрайт 16×16 в world (32, 48): Y > center.y → выше центра экрана.
    let mut batch = SpriteBatch::new(&gpu.device, 1);
    batch.push(Sprite::new([32.0, 48.0], [16.0, 16.0], uv).with_tint([1.0, 0.0, 0.0, 1.0]));
    batch.upload(&gpu.queue);

    let (tex, view) = target(&gpu.device, W, H);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("orientation encoder") });
    renderer.render(&gpu.queue, &mut encoder, &view, &atlas, &batch, &cam);
    gpu.queue.submit(Some(encoder.finish()));
    let data = readback(&gpu, &tex, W, H);

    // Красные пиксели — только в верхней половине (ряды < 32), в нижней — нет.
    let mut lit_top = 0;
    let mut lit_bottom = 0;
    for y in 0..H {
        for x in 0..W {
            let p = px(&data, W, x, y);
            if p[0] > 150 && p[1] < 80 && p[2] < 80 {
                if y < H / 2 {
                    lit_top += 1;
                } else {
                    lit_bottom += 1;
                }
            }
        }
    }
    assert!(lit_top > 0, "красный спрайт должен быть в верхней половине, а красных пикселей там {lit_top}");
    assert_eq!(lit_bottom, 0, "в нижней половине красного быть не должно, а их {lit_bottom}");

    // screen_to_world согласован с рендером: верх экрана = больший Y мира.
    let up = cam.screen_to_world([32.0, 16.0]);
    assert!(up[1] > 32.0, "верх экрана → world.y > center.y, а это {up:?}");
    assert_eq!(up, [32.0, 48.0], "screen (32,16) → world ровно (32,48), а это {up:?}");
    let down = cam.screen_to_world([32.0, 48.0]);
    assert!(down[1] < 32.0, "низ экрана → world.y < center.y, а это {down:?}");
    assert_eq!(down, [32.0, 16.0], "screen (32,48) → world ровно (32,16), а это {down:?}");

    // world_to_ndc совпадает с шейдером: Y вверх = положительный NDC.
    let ndc = cam.world_to_ndc([32.0, 48.0]);
    assert!(ndc[1] > 0.0, "world выше центра → положительный NDC Y, а это {ndc:?}");
    assert_eq!(ndc, [0.0, 0.5], "world (32,48) → NDC (0, +0.5), а это {ndc:?}");
}
