//! Экранная геометрия UI: прямоугольники и якоря.
//!
//! Координаты UI — экранные пиксели, ось Y ВНИЗ (как у winit-событий) —
//! в отличие от мира рендера (Y вверх). Перевод в мировые координаты
//! identity-камеры ([`crate::Camera::new`]) выполняют [`crate::ui`]-модули
//! при создании спрайтов, снаружи он не нужен.

/// Прямоугольник в экранных пикселях (Y вниз): `x`/`y` — верхний левый угол.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// Точка внутри прямоугольника (границы включены).
    pub fn contains(&self, p: [f32; 2]) -> bool {
        p[0] >= self.x && p[0] <= self.x + self.w && p[1] >= self.y && p[1] <= self.y + self.h
    }

    /// Прямоугольник, суженный со всех сторон на `px` (не меньше нуля по размеру).
    pub fn inner(&self, px: f32) -> Rect {
        Rect {
            x: self.x + px,
            y: self.y + px,
            w: (self.w - 2.0 * px).max(0.0),
            h: (self.h - 2.0 * px).max(0.0),
        }
    }

    /// Мировой центр для identity-камеры (`Camera::new(viewport)`): экранная
    /// Y вниз → мировая Y вверх.
    pub fn world_center(&self, viewport: [f32; 2]) -> [f32; 2] {
        [
            self.x + self.w * 0.5 - viewport[0] * 0.5,
            viewport[1] * 0.5 - (self.y + self.h * 0.5),
        ]
    }

    /// Верхняя часть прямоугольника высотой `h`.
    pub fn split_top(&self, h: f32) -> Rect {
        Rect::new(self.x, self.y, self.w, h.min(self.h))
    }

    /// Левая часть прямоугольника шириной `w`.
    pub fn split_left(&self, w: f32) -> Rect {
        Rect::new(self.x, self.y, w.min(self.w), self.h)
    }
}

/// Отступы по четырём сторонам (px).
#[derive(Clone, Copy, Debug, Default)]
pub struct Edges {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

/// Сторона, с которой попап предпочитает появиться.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Side {
    Above,
    Below,
    Left,
    Right,
}

/// Позиционер попапов (tooltip/popup-menu) — заимствовано из gpui-component
/// (`Positioner`): предпочитаемая сторона → флип, если не влезает →
/// clamp внутрь вьюпорта с отступом `margin`.
pub fn place_popup(
    anchor: Rect,
    popup: [f32; 2],
    viewport: [f32; 2],
    preferred: Side,
    margin: f32,
) -> Rect {
    let fits = |side: Side| -> bool {
        match side {
            Side::Above => anchor.y >= popup[1] + margin,
            Side::Below => viewport[1] - (anchor.y + anchor.h) >= popup[1] + margin,
            Side::Left => anchor.x >= popup[0] + margin,
            Side::Right => viewport[0] - (anchor.x + anchor.w) >= popup[0] + margin,
        }
    };
    let side = if fits(preferred) {
        preferred
    } else {
        let flipped = match preferred {
            Side::Above => Side::Below,
            Side::Below => Side::Above,
            Side::Left => Side::Right,
            Side::Right => Side::Left,
        };
        if fits(flipped) { flipped } else { preferred }
    };

    // Основная ось: вплотную к якорю со стороны `side` (уже проверено/флипнуто).
    let (x, y) = match side {
        Side::Above => (0.0, anchor.y - popup[1] - margin),
        Side::Below => (0.0, anchor.y + anchor.h + margin),
        Side::Left => (anchor.x - popup[0] - margin, 0.0),
        Side::Right => (anchor.x + anchor.w + margin, 0.0),
    };
    // Поперечная ось: центр якоря.
    let (mut x, mut y) = if matches!(side, Side::Above | Side::Below) {
        (anchor.x + (anchor.w - popup[0]) * 0.5, y)
    } else {
        (x, anchor.y + (anchor.h - popup[1]) * 0.5)
    };

    // Clamp внутрь вьюпорта (попап меньше вьюпорта предполагается).
    x = x.clamp(margin, (viewport[0] - popup[0] - margin).max(margin));
    y = y.clamp(margin, (viewport[1] - popup[1] - margin).max(margin));
    Rect::new(x, y, popup[0], popup[1])
}

/// Якорь панели на экране: к какому краю приколочен виджет.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

/// Прямоугольник `size`, приколоченный к краю экрана с отступом `margin` px.
pub fn anchored_rect(anchor: Anchor, size: [f32; 2], viewport: [f32; 2], margin: f32) -> Rect {
    let [vw, vh] = viewport;
    let [w, h] = size;
    let (x, y) = match anchor {
        Anchor::TopLeft => (margin, margin),
        Anchor::TopCenter => ((vw - w) * 0.5, margin),
        Anchor::TopRight => (vw - w - margin, margin),
        Anchor::CenterLeft => (margin, (vh - h) * 0.5),
        Anchor::Center => ((vw - w) * 0.5, (vh - h) * 0.5),
        Anchor::CenterRight => (vw - w - margin, (vh - h) * 0.5),
        Anchor::BottomLeft => (margin, vh - h - margin),
        Anchor::BottomCenter => ((vw - w) * 0.5, vh - h - margin),
        Anchor::BottomRight => (vw - w - margin, vh - h - margin),
    };
    Rect::new(x, y, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_boundaries_inclusive() {
        let r = Rect::new(10.0, 20.0, 30.0, 40.0);
        assert!(r.contains([10.0, 20.0]));
        assert!(r.contains([40.0, 60.0]));
        assert!(r.contains([25.0, 40.0]));
        assert!(!r.contains([9.9, 20.0]));
        assert!(!r.contains([40.1, 60.0]));
    }

    #[test]
    fn inner_clamps_to_zero() {
        let r = Rect::new(0.0, 0.0, 10.0, 8.0);
        let i = r.inner(3.0);
        assert_eq!([i.x, i.y, i.w, i.h], [3.0, 3.0, 4.0, 2.0]);
        let i2 = r.inner(10.0); // сужение больше размера — нулевые размеры
        assert_eq!([i2.w, i2.h], [0.0, 0.0]);
    }

    #[test]
    fn anchors_are_inside_viewport() {
        let vp = [1280.0, 720.0];
        let size = [280.0, 220.0];
        for (a, expected_corner) in [
            (Anchor::TopLeft, [16.0, 16.0]),
            (Anchor::TopRight, [1280.0 - 16.0 - 280.0, 16.0]),
            (Anchor::BottomRight, [1280.0 - 16.0 - 280.0, 720.0 - 16.0 - 220.0]),
            (Anchor::Center, [(1280.0 - 280.0) / 2.0, (720.0 - 220.0) / 2.0]),
        ] {
            let r = anchored_rect(a, size, vp, 16.0);
            assert!((r.x - expected_corner[0]).abs() < 1e-4, "{a:?}: {r:?}");
            assert!((r.y - expected_corner[1]).abs() < 1e-4, "{a:?}: {r:?}");
        }
    }

    #[test]
    fn place_popup_prefers_side_and_clamps() {
        let vp = [400.0, 300.0];
        let anchor = Rect::new(180.0, 280.0, 40.0, 20.0);
        let size = [100.0, 30.0];
        // Снизу якоря места нет (якорь у нижнего края) → флип наверх.
        let r = place_popup(anchor, size, vp, Side::Below, 8.0);
        assert!((r.y + r.h) < anchor.y + 0.1, "флип наверх: r.y={}", r.y);
        // Клэмп по краям вьюпорта: центр якоря выталкивает попап за левый край —
        // попап прижат к margin.
        let r2 = place_popup(Rect::new(10.0, 100.0, 10.0, 10.0), size, vp, Side::Below, 8.0);
        assert!((r2.x - 8.0).abs() < 0.1, "клэмп влево: r2.x={}", r2.x);
        // Справа достаточно места → как просили.
        let r3 = place_popup(Rect::new(180.0, 100.0, 40.0, 20.0), size, vp, Side::Right, 8.0);
        assert!((r3.x - (180.0 + 40.0 + 8.0)).abs() < 0.1, "right side: r3.x={}", r3.x);
    }

    #[test]
    fn splits_are_inside_parent() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        let top = r.split_top(20.0);
        assert_eq!([top.w, top.h], [100.0, 20.0]);
        let left = r.split_left(30.0);
        assert_eq!([left.x, left.h], [10.0, 50.0]);
        // Перебор не вылезает.
        assert_eq!(r.split_top(100.0).h, 50.0);
        assert_eq!(r.split_left(200.0).w, 100.0);
    }

    #[test]
    fn world_center_flips_y() {
        // Экран 100×100, rect сверху (y=0) → мир снизу... нет: экранная Y вниз,
        // мировая вверх, значит верх экрана = больший мировой Y.
        let vp = [100.0, 100.0];
        let top = Rect::new(40.0, 0.0, 20.0, 10.0);
        let bottom = Rect::new(40.0, 90.0, 20.0, 10.0);
        assert!(top.world_center(vp)[1] > bottom.world_center(vp)[1]);
        let c = Rect::new(40.0, 40.0, 20.0, 20.0).world_center(vp);
        assert_eq!(c, [0.0, 0.0]);
    }
}
