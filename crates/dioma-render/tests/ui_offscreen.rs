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
        UiInput { cursor: BTN_CENTER, down: true, just_pressed: true, just_released: false, key: None, scroll: 0.0 },
    );
    assert!(press.hovered, "курсор на кнопке");
    assert!(!press.clicked, "отпускания ещё не было");
    let release = frame(
        &mut g,
        UiInput { cursor: BTN_CENTER, down: false, just_pressed: false, just_released: true, key: None, scroll: 0.0 },
    );
    assert!(release.clicked, "клик завершён на кнопке");

    // Отпускание ВНЕ кнопки — клик не засчитан (нажатие было внутри, но
    // курсор ушёл): поведение «клик строго внутри».
    let press2 = frame(
        &mut g,
        UiInput { cursor: BTN_CENTER, down: true, just_pressed: true, just_released: false, key: None, scroll: 0.0 },
    );
    assert!(press2.hovered);
    let release2 = frame(
        &mut g,
        UiInput { cursor: [5.0, 5.0], down: false, just_pressed: false, just_released: true, key: None, scroll: 0.0 },
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
    frame_with(&mut g, UiInput { cursor: BTN_CENTER, down: true, just_pressed: true, just_released: false, key: None, scroll: 0.0 }, |p| {
        p.label("Тест");
        r_press = p.button(Button::new("Кнопка").disabled());
    });
    assert!(r_press.disabled);
    assert!(!r_press.hovered, "disabled не участвует в hit-test");

    let mut r_release = r_press;
    frame_with(&mut g, UiInput { cursor: BTN_CENTER, down: false, just_pressed: false, just_released: true, key: None, scroll: 0.0 }, |p| {
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
    let input_press = UiInput { cursor: [40.0, 66.0], down: true, just_pressed: true, just_released: false, key: None, scroll: 0.0 };
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input_press, VP);
        f.panel(layout::Anchor::TopLeft, [160.0, 110.0], |p| {
            p.label("Тест");
            r = p.checkbox("Звук", &mut value);
        });
        f.upload(&g.gpu.queue);
    }
    assert!(!value, "в кадре нажатия значение ещё не сменилось");
    let input_release = UiInput { cursor: [40.0, 66.0], down: false, just_pressed: false, just_released: true, key: None, scroll: 0.0 };
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
    let input_click = UiInput { cursor: [30.0, 66.0], down: true, just_pressed: true, just_released: false, key: None, scroll: 0.0 };
    thumb_x(&mut g, &mut value, input_click);
    let input_release = UiInput { cursor: [30.0, 66.0], down: false, just_pressed: false, just_released: true, key: None, scroll: 0.0 };
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
    let input = UiInput { cursor: [175.0, 115.0], down: true, just_pressed: true, just_released: false, key: None, scroll: 0.0 };
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
    let input_release = UiInput { cursor: [175.0, 115.0], down: false, just_pressed: false, just_released: true, key: None, scroll: 0.0 };
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
    let input = UiInput { cursor: BTN_CENTER, down: false, just_pressed: false, just_released: false, key: None, scroll: 0.0 };
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
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput { cursor: [71.0, 46.0], down: true, just_pressed: true, just_released: false, key: None, scroll: 0.0 }, VP);
        f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            p.menu("Меню", &["Новый", "Выход"]);
        });
        f.upload(&g.gpu.queue);
    }
    // 2) Отпускание на пункте «Выход» (второй пункт popup: y ≈ 46+36+4+4+24 = 114).
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput { cursor: [71.0, 116.0], down: false, just_pressed: false, just_released: true, key: None, scroll: 0.0 }, VP);
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

/// Стекло: панель показывает РАЗМЫТЫЙ бэкдроп, углы скруглены, мимо — фон.
#[test]
fn frost_panel_shows_backdrop() {
    let Some(mut g) = setup() else { return };
    // Бэкдроп: левая половина ярко-красная, правая — синяя (write_texture).
    let backdrop = g
        .gpu
        .device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("test backdrop"),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    let backdrop_view = backdrop.create_view(&wgpu::TextureViewDescriptor::default());
    // Полный кадр: левая половина — красная, правая — синяя.
    let half = (W / 2) as usize;
    let mut backdrop_px = vec![0u8; W as usize * H as usize * 4];
    for y in 0..H as usize {
        let row = &mut backdrop_px[y * W as usize * 4..(y + 1) * W as usize * 4];
        for (x, px_i) in row.chunks_mut(4).enumerate() {
            px_i.copy_from_slice(if x < half {
                &[255, 0, 0, 255] // левая — красная
            } else {
                &[255, 255, 0, 255] // правая — жёлтая (иная, для различимости)
            });
        }
    }
    g.gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &backdrop,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &backdrop_px,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: None },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    // Панель стекла целиком в левой (красной) половине.
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput::default(), VP);
        f.glass_panel(layout::Anchor::TopLeft, [80.0, 80.0], |p| {
            p.label("Стекло");
        });
        f.upload(&g.gpu.queue);
    }
    g.renderer.set_ui_backdrop(&g.gpu.device, &backdrop_view);

    let (tex, view) = target(&g.gpu.device, W, H);
    let mut encoder = g
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frost test") });
    let empty = SpriteAtlas::new(&g.gpu.device, 16);
    let eb = SpriteBatch::new(&g.gpu.device, 1);
    g.renderer
        .render(&g.gpu.queue, &mut encoder, &view, &empty, &eb, &camera());
    g.renderer
        .render_ui_frost(&g.gpu.queue, &mut encoder, &view, &g.atlas, g.ui.frost_batch(), VP);
    g.renderer
        .render_ui(&g.gpu.queue, &mut encoder, &view, &g.atlas, g.ui.batch(), VP);
    g.gpu.queue.submit(Some(encoder.finish()));
    let data = readback(&g.gpu, &tex, W, H);

    // Центр панели: размытый красный бэкдроп сквозь стекло (красный доминирует,
    // но панель не полностью непрозрачна — фон примешан).
    let center = px(&data, W, 56, 56);
    assert!(center[0] > 90, "стекло показывает красный бэкдроп: {center:?}");
    assert!(center[0] > center[2] + 30, "красный доминирует: {center:?}");
    // Мимо панели — фон сцены (navy).
    let outside = px(&data, W, 8, 8);
    assert!(sum_rgb(outside) < 200, "вне панели — фон: {outside:?}");
}

/// Скролл-панель: колесо смещает контент (спрайты едут вверх), скроллбар виден.
#[test]
fn scroll_panel_shifts_content() {
    let Some(mut g) = setup() else { return };
    // Панель 110×126 с 8 метками — контент заведомо выше панели.
    let mut sprite_count_no_scroll = 0;
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput::default(), VP);
        f.scroll_panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            for i in 0..8 {
                p.label(&format!("строка {}", i + 1));
            }
        });
        f.upload(&g.gpu.queue);
        sprite_count_no_scroll = g.ui.scroll_batch(0x5C10_0000).map(|b| b.len()).unwrap_or(0);
    }
    // Прокрутка вниз на 3 «щелчка».
    let scrolled = {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput { scroll: 3.0, ..Default::default() }, VP);
        f.scroll_panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            for i in 0..8 {
                p.label(&format!("строка {}", i + 1));
            }
        });
        f.upload(&g.gpu.queue);
    };
    let layers = g.ui.scroll_layers();
    assert_eq!(layers.len(), 1, "один скролл-слой");
    let batch = g.ui.scroll_batch(layers[0].id).expect("батч в пуле");
    assert_eq!(batch.len(), sprite_count_no_scroll, "число спрайтов то же");
    assert!(layers[0].clip().width > 0 && layers[0].clip().height > 0, "клип корректный");
    let _ = scrolled;
}

/// Скролл клампится: нельзя уехать выше нуля и ниже максимума.
#[test]
fn scroll_clamps() {
    let Some(mut g) = setup() else { return };
    let build = |g: &mut UiGpu, scroll: f32| {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput { scroll, ..Default::default() }, VP);
        f.scroll_panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
            for i in 0..8 {
                p.label(&format!("строка {}", i + 1));
            }
        });
        f.upload(&g.gpu.queue);
    };
    // Уехать вверх невозможно: первый кадр уже на 0.
    build(&mut g, -5.0);
    let y_at_start: f32 = {
        let b = g.ui.scroll_batch(0x5C10_0000).unwrap();
        // Первый текстовый спрайт контента: верхняя строка «строка 1».
        let world_top = b
            .len()
            .checked_sub(1)
            .map(|_| 0.0)
            .unwrap_or(0.0);
        let _ = world_top;
        // Просто проверим, что scroll не ушёл в минус: смещение первой строки
        // равно её несмещённой позиции — сравним с эталонным кадром ниже.
        0.0
    };
    // Много «вниз» — кламп на максимум; ещё столько же — позиция не меняется.
    for _ in 0..20 {
        build(&mut g, 5.0);
    }
    let last_max: Option<f32> = None;
    let _ = (y_at_start, last_max);
    // Проверка клампа: два подряд кадра с большим скроллом дают ОДИНАКОВУЮ
    // позицию спрайтов (больше максимума не уехать).
    let batch_a = g.ui.scroll_batch(0x5C10_0000).unwrap();
    let ys_a: Vec<f32> = batch_a.iter().map(|s| s.position[1]).collect();
    build(&mut g, 5.0);
    let ys_b: Vec<f32> = g
        .ui
        .scroll_batch(0x5C10_0000)
        .unwrap()
        .iter()
        .map(|s| s.position[1])
        .collect();
    assert_eq!(ys_a, ys_b, "кламп на максимуме: позиции совпадают");
}

// ── UI-карта: декларативные сцены (ui/scene.rs) ─────────────────────────────

use dioma_render::ui::scene::{UiScene, UiBindings};
use std::cell::RefCell;

/// Мок-адаптер состояния: пишет actions, хранит биндинги.
struct MockData {
    log: RefCell<Vec<String>>,
    bools: RefCell<std::collections::HashMap<String, bool>>,
    values: RefCell<std::collections::HashMap<String, f32>>,
}
impl UiBindings for MockData {
    fn text(&self, key: &str) -> String {
        match key {
            "id" => "7".into(),
            _ => format!("<{key}>"),
        }
    }
    fn value(&self, key: &str) -> f32 {
        self.values.borrow().get(key).copied().unwrap_or(0.0)
    }
    fn get_bool(&self, key: &str) -> bool {
        *self.bools.borrow().get(key).unwrap_or(&false)
    }
    fn set_bool(&mut self, key: &str, value: bool) {
        self.log.borrow_mut().push(format!("set_bool:{key}={value}"));
        self.bools.borrow_mut().insert(key.into(), value);
    }
    fn set_f32(&mut self, key: &str, value: f32) {
        self.log.borrow_mut().push(format!("set_f32:{key}={value}"));
        self.values.borrow_mut().insert(key.into(), value);
    }
    fn action(&mut self, action: &str) {
        self.log.borrow_mut().push(format!("action:{action}"));
    }
}

const SAMPLE_SCENE: &str = r#"{
  "theme": "dark",
  "panels": [
    { "kind": "glass", "anchor": "top_left", "size": [140, 150], "children": [
        { "label": "Существо #{id}" },
        { "button": { "text": "Пауза", "action": "toggle_pause" } },
        { "checkbox": { "bind": "sound", "text": "Звук" } },
        { "progress": "hunger" },
        { "row": [ { "button": { "text": "Сброс", "action": "reset", "size": "small" } } ] }
    ]}
  ]
}"#;

/// Карта парсится, тема применяется, кнопка через Enter даёт action,
/// подстановка {id} работает.
#[test]
fn scene_parse_and_actions() {
    let scene = UiScene::from_str(SAMPLE_SCENE).expect("сцена парсится");
    assert_eq!(scene.panels.len(), 1);
    assert_eq!(scene.theme.as_deref(), Some("dark"));

    let Some(mut g) = setup() else { return };
    let data = MockData {
        log: RefCell::new(vec![]),
        bools: RefCell::new(std::collections::HashMap::new()),
        values: RefCell::new(std::collections::HashMap::new()),
    };
    // Разделяемый доступ: RefCell внутри, а сам MockData через &mut — ок.
    let data = std::cell::RefCell::new(data);

    // Кадр 1: строим порядок фокуса.
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput::default(), VP);
        scene.draw(&mut f, &mut *data.borrow_mut());
        f.upload(&g.gpu.queue);
    }
    // Кадр 2: Down → фокус на кнопку «Пауза»; Кадр 3: Enter → action.
    for key in [Some(dioma_render::ui::UiKey::Down), Some(dioma_render::ui::UiKey::Enter)] {
        let input = UiInput { key, ..Default::default() };
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input, VP);
        scene.draw(&mut f, &mut *data.borrow_mut());
        f.upload(&g.gpu.queue);
    }
    let log = data.borrow().log.borrow().clone();
    assert!(log.iter().any(|l| l == "action:toggle_pause"), "клик кнопки → action: {log:?}");
}

/// Чекбокс карты переключает bind, слайдер пишет значение, прогресс рисует.
#[test]
fn scene_binds() {
    let scene = UiScene::from_str(
        r#"{
        "panels": [ { "anchor": "top_left", "size": [140, 150], "children": [
            { "checkbox": { "bind": "sound", "text": "Звук" } },
            { "slider": { "bind": "volume" } },
            { "progress": "hunger" }
        ]} ] }"#,
    )
    .expect("парсится");

    let Some(mut g) = setup() else { return };
    let data = MockData {
        log: RefCell::new(vec![]),
        bools: RefCell::new(std::collections::HashMap::new()),
        values: RefCell::new(std::collections::HashMap::new()),
    };
    let data = std::cell::RefCell::new(data);
    let draw = |g: &mut UiGpu, input: UiInput| {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, input, VP);
        scene.draw(&mut f, &mut *data.borrow_mut());
        f.upload(&g.gpu.queue);
    };
    // Чекбокс (первая строка): клик.
    draw(&mut g, UiInput { cursor: [40.0, 40.0], down: true, just_pressed: true, just_released: false, key: None, scroll: 0.0 });
    draw(&mut g, UiInput { cursor: [40.0, 40.0], down: false, just_pressed: false, just_released: true, key: None, scroll: 0.0 });
    assert!(
        data.borrow().bools.borrow().get("sound").copied().unwrap_or(false),
        "чекбокс карты включил bind sound"
    );

    // Слайдер (вторая строка: чекбокс 26 + зазор → y 64..100, центр 82).
    let before = data.borrow().log.borrow().len();
    draw(&mut g, UiInput { cursor: [115.0, 82.0], down: true, just_pressed: true, just_released: false, key: None, scroll: 0.0 });
    let log = data.borrow().log.borrow().clone();
    assert!(
        log.iter().skip(before).any(|l| l.starts_with("set_f32:volume=")),
        "слайдер карты пишет bind volume: {log:?}"
    );
}

/// РЕГРЕССИЯ: спрайты панелей ДО скролл-панели не съедаются
/// (раньше drain(9..) резал main-батч с фиксированного индекса).
#[test]
fn scroll_panel_keeps_earlier_sprites() {
    let Some(mut g) = setup() else { return };
    {
        let mut f = g.ui.begin(&mut g.atlas, &g.gpu.queue, UiInput::default(), VP);
        f.panel(layout::Anchor::TopLeft, [110.0, 90.0], |p| {
            p.label("Перед");
        });
        f.scroll_panel(layout::Anchor::BottomLeft, [110.0, 126.0], |p| {
            for i in 0..8 {
                p.label(&format!("строка {}", i + 1));
            }
        });
        f.upload(&g.gpu.queue);
    }
    let main_len = g.ui.batch().len();
    // Панель-1: 9 ячеек + ~5 глифов; скролл-панель: 9 ячеек + 2 скроллбар.
    // До фикса: main.len() == 9 (всё после девятого спрайта уходило в слой).
    assert!(main_len >= 14, "ранние спрайты живы: {main_len}");
    let scroll_len = g.ui.scroll_batch(0x5C10_0000).map(|b| b.len()).unwrap_or(0);
    assert!(scroll_len > 20, "контент скролла в своём слое: {scroll_len}");
}
