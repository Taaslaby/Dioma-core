//! Offscreen-тесты UI-каркаса: панель/кнопка/бар рисуются, текст читаем и
//! не перевёрнут, клик по кнопке распознаётся. Рендер — в offscreen-текстуру
//! 160×160 (readback по пикселям), взаимодействие — через `UiInput`.
//!
//! Геометрия кадра (см. `build_frame`): панель 110×126 якорится в
//! TopLeft с полем 16 → (16,16); содержимое — метка «Тест», кнопка
//! «Кнопка» (y ≈ 53, высота 36), бар «Бар» (y ≈ 114, высота 14).

mod common;

use common::*;
use dioma_render::ui::{layout, UiInput, UiSystem};
use dioma_render::{Renderer, SpriteAtlas, SpriteBatch};

const VP: [f32; 2] = [256.0, 160.0]; // 256×4 кратно COPY_BYTES_PER_ROW_ALIGNMENT
const W: u32 = 256;
const H: u32 = 160;
/// Центр кнопки (панель (16,16) + поля 12 + метка ~15+10 → кнопка y≈53..89).
const BTN_CENTER: [f32; 2] = [71.0, 71.0];

struct UiGpu {
    gpu: TestGpu,
    atlas: SpriteAtlas,
    ui: UiSystem,
    renderer: Renderer,
}

fn setup() -> Option<UiGpu> {
    let gpu = gpu()?;
    let mut atlas = SpriteAtlas::new(&gpu.device, 64);
    let ui = UiSystem::new(&gpu.device, &gpu.queue, &mut atlas);
    let mut renderer = Renderer::new(&gpu.device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_clear_color(navy_clear());
    Some(UiGpu { gpu, atlas, ui, renderer })
}

/// Один кадр: панель (метка + кнопка + бар); возвращает ответ кнопки.
fn frame(g: &mut UiGpu, input: UiInput) -> dioma_render::ui::Response {
    let mut btn = dioma_render::ui::Response::default();
    let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input, VP);
    f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
        p.label("Тест");
        btn = p.button("Кнопка");
        p.progress_bar("Бар", 0.6);
    });
    f.upload(&g.gpu.queue);
    btn
}

fn render(g: &mut UiGpu) -> Vec<u8> {
    let (tex, view) = target(&g.gpu.device, W, H);
    let mut encoder = g
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ui test") });
    let empty = SpriteAtlas::new(&g.gpu.device, 16);
    let empty_batch = SpriteBatch::new(&g.gpu.device, 1);
    g.renderer
        .render(&g.gpu.queue, &mut encoder, &view, &empty, &empty_batch, &camera());
    g.renderer
        .render_ui(&g.gpu.queue, &mut encoder, &view, &g.atlas, g.ui.batch(), VP);
    g.gpu.queue.submit(Some(encoder.finish()));
    readback(&g.gpu, &tex, W, H)
}

/// Панель рисуется: светлая рамка на краю, тёмная заливка внутри.
#[test]
fn panel_pixels() {
    let Some(mut g) = setup() else { return };
    let out = frame(&mut g, UiInput::default());
    assert!(!out.hovered);
    let data = render(&mut g);

    // Край панели (17, 83) — рамка светлая; фон снаружи (8, 83) — navy.
    let border = px(&data, W, 17, 83);
    let outside = px(&data, W, 8, 83);
    assert!(sum_rgb(border) > 250, "рамка светлая: {border:?}");
    assert!(sum_rgb(outside) < 60, "вне панели — фон: {outside:?}");

    // Заливка панели тёмная (текст/виджеты сюда не попадают): (100, 30).
    let fill = px(&data, W, 100, 30);
    assert!(sum_rgb(fill) < 120, "заливка тёмная: {fill:?}");
}

/// Текст «Тест» читаем: чернила есть, у «Т» перекладина сверху (не перевёрнут).
#[test]
fn text_ink_and_orientation() {
    let Some(mut g) = setup() else { return };
    frame(&mut g, UiInput::default());
    let data = render(&mut g);

    // Область метки: x 28..114, y 26..46.
    let ink = |r: [u32; 4]| -> usize {
        let mut n = 0;
        for y in r[1]..r[3] {
            for x in r[0]..r[2] {
                if sum_rgb(px(&data, W, x, y)) > 350 {
                    n += 1;
                }
            }
        }
        n
    };
    let all = ink([28, 26, 114, 46]);
    assert!(all > 15, "чернила метки есть: {all}");

    // «Т» — первый глиф (~14 px шириной): перекладина сверху → чернил в
    // верхней половине заметно больше, чем в нижней (там только ножка).
    let top = ink([28, 26, 44, 36]);
    let bottom = ink([28, 36, 44, 46]);
    assert!(top > bottom * 2, "«Т» не перевёрнут: верх {top} vs низ {bottom}");
}

/// Кнопка: hover подсвечивает, клик (нажатие+отпускание) распознаётся.
#[test]
fn button_hover_click_and_fill() {
    let Some(mut g) = setup() else { return };

    // Мимо кнопки — ни hover, ни клик.
    let far = frame(&mut g, UiInput { cursor: [5.0, 5.0], ..Default::default() });
    assert!(!far.hovered && !far.clicked);

    // Нажатие → отпускание на кнопке = клик.
    let press = frame(
        &mut g,
        UiInput { cursor: BTN_CENTER, down: true, just_pressed: true, just_released: false },
    );
    assert!(press.hovered, "курсор на кнопке");
    assert!(!press.clicked, "отпускания ещё не было");
    let release = frame(
        &mut g,
        UiInput { cursor: BTN_CENTER, down: false, just_pressed: false, just_released: true },
    );
    assert!(release.clicked, "клик завершён на кнопке");

    // Отпускание ВНЕ кнопки — клик не засчитан (нажатие было внутри, но
    // курсор ушёл): поведение «клик строго внутри».
    let press2 = frame(
        &mut g,
        UiInput { cursor: BTN_CENTER, down: true, just_pressed: true, just_released: false },
    );
    assert!(press2.hovered);
    let release2 = frame(
        &mut g,
        UiInput { cursor: [5.0, 5.0], down: false, just_pressed: false, just_released: true },
    );
    assert!(!release2.clicked, "отпускание вне кнопки — не клик");

    // Пиксельно: заливка кнопки светлее заливки панели (BTN tint).
    let data = render(&mut g);
    let btn_fill = px(&data, W, 108, 71); // правая часть кнопки, без текста
    let panel_fill = px(&data, W, 100, 30);
    assert!(
        sum_rgb(btn_fill) > sum_rgb(panel_fill) + 60,
        "кнопка светлее панели: {btn_fill:?} vs {panel_fill:?}"
    );
}

/// Бар с прогрессом 0.6: зелёная заливка присутствует в своей полосе.
#[test]
fn progress_bar_fill_pixels() {
    let Some(mut g) = setup() else { return };
    frame(&mut g, UiInput::default());
    let data = render(&mut g);

    // Полоса бара y ≈ 115..127 (высота 14, внутренняя заливка 12).
    let mut green = 0;
    for y in 116..126 {
        for x in 40..70 {
            let p = px(&data, W, x, y);
            if p[1] > 90 && p[1] > p[0] + 20 && p[1] > p[2] + 20 {
                green += 1;
            }
        }
    }
    assert!(green > 30, "зелёная заливка бара видна: {green}");
}

/// ВРЕМЕННЫЙ дамп: ASCII-арт кадра UI (яркость), для отладки геометрии.
#[test]
fn debug_dump_ascii() {
    let Some(mut g) = setup() else { return };
    frame(&mut g, UiInput { cursor: BTN_CENTER, down: false, just_pressed: false, just_released: false });
    let data = render(&mut g);
    let chars: Vec<char> = " .:-=+*#%@".chars().collect();
    for y in (12..132).step_by(2) {
        let mut row = String::new();
        for x in 12..126 {
            let p = px(&data, W, x, y);
            let b = ((p[0] as u32 + p[1] as u32 + p[2] as u32) / 3) as usize;
            let c = chars[(b.min(255) * chars.len()) / 256];
            row.push(c);
        }
        println!("{y:3}|{row}");
    }
}
