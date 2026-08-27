//! 2D-камера: центр в мировых координатах, зум, вьюпорт в пикселях.

/// Минимальный/максимальный зум (защита от деления на ноль и расходящегося зума).
const MIN_ZOOM: f32 = 0.05;
const MAX_ZOOM: f32 = 64.0;

/// 2D-камера для спрайтов.
///
/// Мировые координаты — пиксели, ось Y вверх. `viewport` — размер целевой
/// поверхности в пикселях. Зум 1.0 = один мировой пиксель на один экранный.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub center: [f32; 2],
    pub zoom: f32,
    pub viewport: [f32; 2],
}

impl Camera {
    /// Камера в центре мира (0,0) с зумом 1.0 и заданным вьюпортом.
    pub fn new(viewport: [f32; 2]) -> Self {
        Self {
            center: [0.0, 0.0],
            zoom: 1.0,
            viewport,
        }
    }

    /// Панорамирование на `delta` мировых единиц (ось Y вверх).
    pub fn pan(&mut self, delta: [f32; 2]) {
        self.center[0] += delta[0];
        self.center[1] += delta[1];
    }

    /// Зум с фиксацией мировой точки под экранным курсором.
    ///
    /// `screen` — координаты курсора в пикселях (origin — верхний левый угол).
    pub fn zoom_at(&mut self, factor: f32, screen: [f32; 2]) {
        let before = self.screen_to_world(screen);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let after = self.screen_to_world(screen);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
    }

    /// Экранные пиксели (Y вниз) → мировые координаты (Y вверх). Для пикинга.
    pub fn screen_to_world(&self, screen: [f32; 2]) -> [f32; 2] {
        [
            self.center[0] + (screen[0] - self.viewport[0] * 0.5) / self.zoom,
            self.center[1] - (screen[1] - self.viewport[1] * 0.5) / self.zoom,
        ]
    }

    /// Мировые координаты → NDC (то же преобразование, что в вершинном шейдере).
    /// CPU-сторона для пикинга.
    pub fn world_to_ndc(&self, world: [f32; 2]) -> [f32; 2] {
        let half_w = self.viewport[0] * 0.5 / self.zoom;
        let half_h = self.viewport[1] * 0.5 / self.zoom;
        [
            (world[0] - self.center[0]) / half_w,
            -(world[1] - self.center[1]) / half_h,
        ]
    }

    /// Uniform-данные камеры: `[center.x, center.y, zoom, viewport.x, viewport.y]`.
    pub fn uniform_data(&self) -> [f32; 5] {
        [
            self.center[0],
            self.center[1],
            self.zoom,
            self.viewport[0],
            self.viewport[1],
        ]
    }
}
