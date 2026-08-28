//! ВРЕМЕННЫЙ дамп спрайтов UI-кадра (удалить после отладки).
mod common;
use common::*;
use dioma_render::ui::{layout, UiInput, UiSystem};
use dioma_render::SpriteAtlas;

#[test]
fn dump_ui_sprites() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 64);
    let mut ui = UiSystem::new(&gpu.device, &gpu.queue, &mut atlas);
    let vp = [256.0, 160.0];
    let mut f = ui.begin(&mut atlas, &gpu.queue, UiInput::default(), vp);
    f.panel(layout::Anchor::TopLeft, [110.0, 126.0], |p| {
        p.label("Тест");
        p.button("Кнопка");
        p.progress_bar("Бар", 0.6);
    });
    println!("sprite count = {}", f.sprite_count());
    for (i, s) in f.sprites().iter().enumerate() {
        // Обратный перевод мирового центра в экранный.
        let sx = s.position[0] + vp[0] * 0.5;
        let sy = vp[1] * 0.5 - s.position[1];
        println!(
            "{i:2} screen=({sx:7.1},{sy:7.1}) size=({:6.1},{:6.1}) layer={:6.1} uv=[{:.3},{:.3},{:.3},{:.3}] tint={:?}",
            s.size[0], s.size[1], s.layer, s.uv_rect[0], s.uv_rect[1], s.uv_rect[2], s.uv_rect[3], s.tint
        );
    }
}
