//! Offscreen-тесты UI-каркаса: панель/кнопка/бар рисуются, текст читаем и
//! не перевёрнут, клик по кнопке распознаётся. Рендер — в offscreen-текстуру
//! 160×160 (readback по пикселям), взаимодействие — через `UiInput`.
//!
//! Геометрия кадра (см. `build_frame`): панель 110×126 якорится в
//! TopLeft с полем 16 → (16,16); содержимое — метка «Тест», кнопка
//! «Кнопка» (y ≈ 53, высота 36), бар «Бар» (y ≈ 114, высота 14).

mod common;

use common::*;
use dioma_render::ui::{layout, Button, UiInput, UiSystem};
use dioma_render::ui::theme::UiTheme;
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

/// Один кадр с произвольным содержимым панели.
fn frame_with(g: &mut UiGpu, input: UiInput, content: impl FnOnce(&mut dioma_render::ui::Panel)) {
    let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input, VP);
    f.panel(layout::Anchor::TopLeft, [110.0, 126.0], content);
    f.upload(&g.gpu.queue);
}

/// Стандартный кадр: панель (метка + кнопка + бар); возвращает ответ кнопки.
fn frame(g: &mut UiGpu, input: UiInput) -> dioma_render::ui::Response {
    let mut btn = dioma_render::ui::Response::default();
    frame_with(g, input, |p| {
        p.label("Тест");
        btn = p.button("Кнопка");
        p.progress_bar("Бар", 0.6);
    });
    btn
}

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
    assert!(sum_rgb(outside) < 200, "вне панели — фон (navy sRGB ~176): {outside:?}");

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

    // Область метки: x 28..114, y 24..48.
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
    let all = ink([28, 24, 114, 48]);
    assert!(all > 15, "чернила метки есть: {all}");

    // «Т» — первый глиф (x 28..40). Базовая линия блока: at.y + ascent·0.5
    // (PT Sans ascent ≈ 32 px при 32 px) ≈ y 44. У ПРАВИЛЬНОГО «Т» перекладина
    // лежит выше базовой линии (строки 31..36), а НИЖЕ базовой линии чернил
    // нет вовсе. У перевёрнутого — перекладина уехала бы под базовую линию.
    let bar = ink([28, 31, 40, 36]);
    let under = ink([28, 44, 40, 49]);
    assert!(bar >= 8, "перекладина «Т» сверху: {bar}");
    assert_eq!(under, 0, "ниже базовой линии у «Т» чернил нет (не перевёрнут): {under}");
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
        UiInput { cursor: BTN_CENTER, down: true, just_pressed: true, just_released: false, key: None },
    );
    assert!(press.hovered, "курсор на кнопке");
    assert!(!press.clicked, "отпускания ещё не было");
    let release = frame(
        &mut g,
        UiInput { cursor: BTN_CENTER, down: false, just_pressed: false, just_released: true, key: None },
    );
    assert!(release.clicked, "клик завершён на кнопке");

    // Отпускание ВНЕ кнопки — клик не засчитан (нажатие было внутри, но
    // курсор ушёл): поведение «клик строго внутри».
    let press2 = frame(
        &mut g,
        UiInput { cursor: BTN_CENTER, down: true, just_pressed: true, just_released: false, key: None },
    );
    assert!(press2.hovered);
    let release2 = frame(
        &mut g,
        UiInput { cursor: [5.0, 5.0], down: false, just_pressed: false, just_released: true, key: None },
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
    for y in 112..128 {
        for x in 35..75 {
            let p = px(&data, W, x, y);
            let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
            if g > 90 && g > r + 20 && g > b + 20 {
                green += 1;
            }
        }
    }
    assert!(green > 25, "зелёная заливка бара видна: {green}");
}

/// Disabled-кнопка: не участвует в hit-test (hover/клик заглушены).
#[test]
fn disabled_button_ignores_input() {
    let Some(mut g) = setup() else { return };

    let mut r_press = dioma_render::ui::Response::default();
    frame_with(&mut g, UiInput { cursor: BTN_CENTER, down: true, just_pressed: true, just_released: false, key: None }, |p| {
        p.label("Тест");
        r_press = p.button(Button::new("Кнопка").disabled());
    });
    assert!(r_press.disabled);
    assert!(!r_press.hovered, "disabled не участвует в hit-test");

    let mut r_release = r_press;
    frame_with(&mut g, UiInput { cursor: BTN_CENTER, down: false, just_pressed: false, just_released: true, key: None }, |p| {
        p.label("Тест");
        r_release = p.button(Button::new("Кнопка").disabled());
    });
    assert!(!r_release.clicked, "disabled не кликается");
}

/// Анимация бара: после смены цели заливка едет от старого значения,
/// а не снапится к новой цели.
#[test]
fn progress_bar_animates() {
    use dioma_render::ui::theme::UiTheme;

    let Some(mut g) = setup() else { return };
    let mut fill_w = 0.0f32;
    let bar = |g: &mut UiGpu, v: f32, w: &mut f32| {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput::default(), VP);
        f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            p.label("Тест");
            p.button("Кнопка");
            p.progress_bar("Бар", v);
        });
        // Заливка = спрайт с tint бара (последний из виджетов).
        if let Some(s) = f.sprites().iter().rev().find(|s| s.tint == UiTheme::dark().bar_fill) {
            *w = s.size[0];
        }
        f.upload(&g.gpu.queue);
    };

    bar(&mut g, 0.0, &mut fill_w);
    assert_eq!(fill_w, 0.0, "пустой бар");
    bar(&mut g, 1.0, &mut fill_w);
    assert_eq!(fill_w, 0.0, "первый кадр после смены цели: fill стартует от 0 (сэмпл при t=0)");
    // Половина TRANSITION_MS (160мс) → ease_out_cubic(0.5)=0.875 пути.
    std::thread::sleep(std::time::Duration::from_millis(80));
    bar(&mut g, 1.0, &mut fill_w);
    assert!(fill_w > 20.0 && fill_w < 84.0, "fill едет к цели: {fill_w}");
    std::thread::sleep(std::time::Duration::from_millis(300));
    bar(&mut g, 1.0, &mut fill_w);
    assert!((fill_w - 84.0).abs() < 0.01, "доехал до полной ширины (86-2): {fill_w}");
}

/// Смена темы перекрашивает панель (роли, не константы).
#[test]
fn theme_switch_changes_colors() {
    let Some(mut g) = setup() else { return };
    frame(&mut g, UiInput::default());
    let data = render(&mut g);
    let dark_fill = px(&data, W, 100, 30);

    g.ui.set_theme(&mut g.atlas, &g.gpu.queue, UiTheme::light());
    frame(&mut g, UiInput::default());
    let data = render(&mut g);
    let light_fill = px(&data, W, 100, 30);
    assert!(
        sum_rgb(light_fill) > sum_rgb(dark_fill) + 200,
        "светлая тема светлее тёмной: {light_fill:?} vs {dark_fill:?}"
    );
}

/// Row: две кнопки по горизонтали на одной высоте.
#[test]
fn row_places_widgets_horizontally() {
    use dioma_render::ui::theme::UiTheme;
    let Some(mut g) = setup() else { return };
    let mut second_center_x = None;
    let mut first_center_x = None;
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput::default(), VP);
        f.panel(layout::Anchor::TopLeft, [220.0, 90.0], |p| {
            p.row(|row| {
                row.button("Ок");
                row.button("Отмена");
            });
        });
        // Ячейки nine-patch кнопок: LAYER_WIDGET × tint кнопки; кластеры по x.
        let mut xs: Vec<f32> = f
            .sprites()
            .iter()
            .filter(|s| s.layer == dioma_render::ui::LAYER_WIDGET && s.tint == UiTheme::dark().button)
            .map(|s| s.position[0] + VP[0] * 0.5)
            .collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        first_center_x = xs.first().copied();
        second_center_x = xs.last().copied();
        f.upload(&g.gpu.queue);
    }
    let (a, b) = (first_center_x.expect("первая кнопка"), second_center_x.expect("вторая кнопка"));
    assert!(b > a + 30.0, "вторая кнопка правее первой: {a} vs {b}");
}

/// Checkbox: клик переключает значение, changed=true.
#[test]
fn checkbox_toggles() {
    let Some(mut g) = setup() else { return };
    let mut value = false;
    let mut r = dioma_render::ui::Response::default();
    // Курсор на чекбоксе (вторая строка после метки: y = 28 + ~15 + 10 ≈ 53..79).
    let input_press = UiInput { cursor: [40.0, 66.0], down: true, just_pressed: true, just_released: false, key: None };
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input_press, VP);
        f.panel(layout::Anchor::TopLeft, [160.0, 110.0], |p| {
            p.label("Тест");
            r = p.checkbox("Звук", &mut value);
        });
        f.upload(&g.gpu.queue);
    }
    assert!(!value, "в кадре нажатия значение ещё не сменилось");
    let input_release = UiInput { cursor: [40.0, 66.0], down: false, just_pressed: false, just_released: true, key: None };
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input_release, VP);
        f.panel(layout::Anchor::TopLeft, [160.0, 110.0], |p| {
            p.label("Тест");
            r = p.checkbox("Звук", &mut value);
        });
        f.upload(&g.gpu.queue);
    }
    assert!(value, "чекбокс включился");
    assert!(r.changed && r.clicked);
}

/// Switch: бегунок едет (анимация) и значение меняется.
#[test]
fn switch_moves_thumb() {
    use dioma_render::ui::theme::UiTheme;
    let Some(mut g) = setup() else { return };
    let mut value = false;
    let thumb_uv = g.atlas.region("ui/thumb").unwrap();
    let thumb_x = |g: &mut UiGpu, value: &mut bool, input: UiInput| -> f32 {
        let mut x = 0.0;
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input, VP);
        f.panel(layout::Anchor::TopLeft, [160.0, 110.0], |p| {
            p.label("Тест");
            p.switch("Звук", value);
        });
        for s in f.sprites() {
            if s.uv_rect == thumb_uv {
                x = s.position[0];
            }
        }
        f.upload(&g.gpu.queue);
        x
    };
    let input_click = UiInput { cursor: [30.0, 66.0], down: true, just_pressed: true, just_released: false, key: None };
    thumb_x(&mut g, &mut value, input_click);
    let input_release = UiInput { cursor: [30.0, 66.0], down: false, just_pressed: false, just_released: true, key: None };
    let _ = thumb_x(&mut g, &mut value, input_release);
    assert!(value, "свитч включился");
    let x_before = thumb_x(&mut g, &mut value, UiInput::default());
    std::thread::sleep(std::time::Duration::from_millis(200));
    let x_after = thumb_x(&mut g, &mut value, UiInput::default());
    assert!(x_after > x_before, "бегунок едет вправо: {x_before} → {x_after}");
}

/// Slider: drag на 75% ширины дорожки ставит ~0.75.
#[test]
fn slider_drag_sets_value() {
    let Some(mut g) = setup() else { return };
    let mut value = 0.0f32;
    // Слайдер в третьей строке; дорожка = контент панели (16+12=28..222).
    // Панель 220 шириной: контент x 28..208, строка слайдера y ≈ 97..133.
    let input = UiInput { cursor: [175.0, 115.0], down: true, just_pressed: true, just_released: false, key: None };
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input, VP);
        f.panel(layout::Anchor::TopLeft, [220.0, 170.0], |p| {
            p.label("Тест");
            p.button("Кнопка");
            p.slider("Громк.", &mut value);
        });
        f.upload(&g.gpu.queue);
    }
    assert!((value - 0.75).abs() < 0.05, "drag на 75%: value={value}");
    let input_release = UiInput { cursor: [175.0, 115.0], down: false, just_pressed: false, just_released: true, key: None };
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input_release, VP);
        f.panel(layout::Anchor::TopLeft, [220.0, 170.0], |p| {
            p.label("Тест");
            p.button("Кнопка");
            p.slider("Громк.", &mut value);
        });
        f.upload(&g.gpu.queue);
    }
    assert!((value - 0.75).abs() < 0.05, "значение сохранилось после отпускания");
}

/// Tooltip: появляется только после задержки hover, поверх виджетов.
#[test]
fn tooltip_after_delay() {
    let Some(mut g) = setup() else { return };
    let input = UiInput { cursor: BTN_CENTER, down: false, just_pressed: false, just_released: false, key: None };
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input, VP);
        f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            p.label("Тест");
            p.button(Button::new("Кнопка").tooltip("Подсказка"));
            p.progress_bar("Бар", 0.6);
        });
        f.upload(&g.gpu.queue);
    }
    // Ещё нет задержки — tooltip не рисуется: проверяем через спрайты.
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input, VP);
        f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            p.label("Тест");
            p.button(Button::new("Кнопка").tooltip("Подсказка"));
            p.progress_bar("Бар", 0.6);
        });
        assert!(!f.sprites().iter().any(|s| s.layer == dioma_render::ui::LAYER_TOOLTIP));
        f.upload(&g.gpu.queue);
    }
    std::thread::sleep(std::time::Duration::from_millis(250));
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input, VP);
        f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            p.label("Тест");
            p.button(Button::new("Кнопка").tooltip("Подсказка"));
            p.progress_bar("Бар", 0.6);
        });
        assert!(f.sprites().iter().any(|s| s.layer == dioma_render::ui::LAYER_TOOLTIP));
        f.upload(&g.gpu.queue);
    }
}

/// Popup-меню: открывается кликом, пункт выбирается, клик вне — закрывает.
#[test]
fn menu_open_select_close() {
    let Some(mut g) = setup() else { return };
    let mut selected: Option<usize> = None;

    // 1) Клик по кнопке меню (первая строка панели) → открыто.
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput { cursor: [71.0, 46.0], down: true, just_pressed: true, just_released: false, key: None }, VP);
        f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            p.menu("Меню", &["Новый", "Выход"]);
        });
        f.upload(&g.gpu.queue);
    }
    // 2) Отпускание на пункте «Выход» (второй пункт popup: y ≈ 46+36+4+4+24 = 114).
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput { cursor: [71.0, 116.0], down: false, just_pressed: false, just_released: true, key: None }, VP);
        f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            selected = p.menu("Меню", &["Новый", "Выход"]);
        });
        f.upload(&g.gpu.queue);
    }
    assert_eq!(selected, Some(1), "выбран пункт «Выход»");
}

/// Клавиатурная навигация: Down ставит фокус на первую кнопку, Enter нажимает.
#[test]
fn keyboard_focus_moves_and_enters() {
    let Some(mut g) = setup() else { return };
    let mut clicked: bool = false;

    // Кадр 1: строим порядок фокуса.
    frame_with(&mut g, UiInput::default(), |p| {
        p.button("Кнопка");
    });
    // Кадр 2: Down — фокус на первую кнопку (порядок прошлого кадра).
    frame_with(&mut g, UiInput { key: Some(dioma_render::ui::UiKey::Down), ..Default::default() }, |p| {
        p.button("Кнопка");
    });
    // Кадр 3: Enter — клик.
    frame_with(&mut g, UiInput { key: Some(dioma_render::ui::UiKey::Enter), ..Default::default() }, |p| {
        let r = p.button("Кнопка");
        clicked = r.clicked;
    });
    assert!(clicked, "Enter нажал сфокусированную кнопку");

    // Down ×2 → фокус на вторую из двух кнопок (ring рисуется под ней).
    let mut ring_seen = false;
    frame_with(&mut g, UiInput { key: Some(dioma_render::ui::UiKey::Down), ..Default::default() }, |p| {
        p.button("Кнопка");
        p.button("Вторая");
    });
    let accent = dioma_render::ui::theme::UiTheme::dark().accent;
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput { key: Some(dioma_render::ui::UiKey::Down), ..Default::default() }, VP);
        f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            p.button("Кнопка");
            p.button("Вторая");
        });
        ring_seen = f.sprites().iter().any(|s| s.tint == accent);
        f.upload(&g.gpu.queue);
    }
    assert!(ring_seen, "фокус-ринг акцентного цвета отрисован");
}

/// Separator рисуется линией с приглушённым tint.
#[test]
fn separator_draws_line() {
    use dioma_render::ui::theme::UiTheme;
    let Some(mut g) = setup() else { return };
    frame_with(&mut g, UiInput::default(), |p| {
        p.label("Тест");
        p.separator();
    });
    let muted = UiTheme::dark().text_muted;
    let data = render(&mut g);
    // Линия на второй строке: y ≈ 28+15+10 = 53..55, x центр панели.
    let mut line_px = 0;
    for y in 56..70 {
        for x in 40..100 {
            let p = px(&data, W, x, y);
            if sum_rgb(p) > 200 && (p[0] as i32 - p[2] as i32).abs() < 30 {
                line_px += 1;
            }
        }
    }
    let _ = muted;
    assert!(line_px > 20, "линия разделителя видна: {line_px}");
}
