//! Климат мира: поля температуры/света/питания, диффундирующие по сетке
//! поверх карты. Комнаты — источники (их CA кормит поле), диффузия разносит
//! тепло и свет по всему миру, растения растут там, где тепло+светло+сытно
//! (`fertility`).
//!
//! Диффузия идёт через `prometheus_grow::DiffusionBackend`: CPU (f64, rayon
//! на больших сетках) или GPU (wgpu compute, f32) — бэкенд взаимозаменяем,
//! арена выбирает один на всю жизнь мира. Это настоящий дом GPU-диффузии
//! прометея: шаг на такт — копейки, в отличие от роста коннектома, где
//! поле надо читать на каждый шаг роста.

use crate::env::map::{Substance, WorldMap};
use prometheus_grow::{DiffusionBackend, Grid, Vec3};

/// Диффузия и распад на шаг (0..1). Тепло течёт медленно и долго держится,
/// свет — быстро и гаснет без окон, питание — средне.
const DIFF: (f64, f64, f64) = (0.10, 0.25, 0.15);
const DECAY: (f64, f64, f64) = (0.005, 0.05, 0.02);

/// Климат: три поля на общей сетке + диффузионный бэкенд.
pub struct Climate {
    grid: Grid,
    pub nx: usize,
    pub ny: usize,
    /// Мировые единицы на клетку.
    pub cell: f64,
    temp: Vec<f64>,
    light: Vec<f64>,
    nutrient: Vec<f64>,
    c2: Vec<f64>,
    backend: DiffusionBackend,
    pub tick: u64,
}

impl Climate {
    /// Сетка климата по границам карты с шагом `cell` (мировые единицы).
    pub fn over(world: &WorldMap, cell: f64, backend: DiffusionBackend) -> Self {
        let mut x0 = f32::INFINITY;
        let mut y0 = f32::INFINITY;
        let mut x1 = f32::NEG_INFINITY;
        let mut y1 = f32::NEG_INFINITY;
        for r in &world.rooms {
            x0 = x0.min(r.rect.min.x);
            y0 = y0.min(r.rect.min.y);
            x1 = x1.max(r.rect.max.x);
            y1 = y1.max(r.rect.max.y);
        }
        let nx = ((x1 - x0) / cell as f32).ceil().max(1.0) as usize;
        let ny = ((y1 - y0) / cell as f32).ceil().max(1.0) as usize;
        // nz=3: прометеевский 7-точечный лапласиан требует интерьера — при
        // nz=1 интерьера нет вообще. Поле держим z-однородным (зеркалим
        // средний слой после каждого шага), z-вклад лапласиана = 0 → это
        // честная 2D-диффузия на 3D-пайплайне.
        let grid = Grid::new(nx, ny, 3, cell, Vec3::new(x0 as f64, y0 as f64, 0.0));
        let vol = grid.volume();
        Self {
            grid,
            nx,
            ny,
            cell,
            temp: vec![0.0; vol],
            light: vec![0.0; vol],
            nutrient: vec![0.0; vol],
            c2: vec![0.0; vol],
            backend,
            tick: 0,
        }
    }

    /// Климат без карты (тесты/пробы): явные границы.
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32, cell: f64, backend: DiffusionBackend) -> Self {
        let nx = ((x1 - x0) / cell as f32).ceil().max(1.0) as usize;
        let ny = ((y1 - y0) / cell as f32).ceil().max(1.0) as usize;
        let grid = Grid::new(nx, ny, 3, cell, Vec3::new(x0 as f64, y0 as f64, 0.0));
        let vol = grid.volume();
        Self {
            grid,
            nx,
            ny,
            cell,
            temp: vec![0.0; vol],
            light: vec![0.0; vol],
            nutrient: vec![0.0; vol],
            c2: vec![0.0; vol],
            backend,
            tick: 0,
        }
    }

    /// Комнаты карты — источники: их CA (0..255) кормит поля климата.
    pub fn seed_from_rooms(&mut self, world: &WorldMap) {
        let (ox, oy) = (self.grid.origin.x as f32, self.grid.origin.y as f32);
        for r in &world.rooms {
            let ca = r.ca;
            let x0 = ((r.rect.min.x - ox) / self.cell as f32).max(0.0) as usize;
            let y0 = ((r.rect.min.y - oy) / self.cell as f32).max(0.0) as usize;
            let x1 = ((r.rect.max.x - ox) / self.cell as f32).min(self.nx as f32 - 1.0) as usize;
            let y1 = ((r.rect.max.y - oy) / self.cell as f32).min(self.ny as f32 - 1.0) as usize;
            let t = (ca[Substance::Temperature.idx()] / 255.0) as f64;
            let l = (ca[Substance::Light.idx()] / 255.0) as f64;
            let n = (ca[Substance::Nutrients.idx()] / 255.0) as f64;
            for x in x0..=x1 {
                for y in y0..=y1 {
                    let i = self.flat(x, y);
                    self.temp[i] = (self.temp[i] + t).min(1.0);
                    self.light[i] = (self.light[i] + l).min(1.0);
                    self.nutrient[i] = (self.nutrient[i] + n).min(1.0);
                }
            }
        }
    }

    /// Один шаг климата: диффузия + распад по всем трём полям.
    pub fn step(&mut self) {
        let (dt, dl, dn) = DIFF;
        let (kt, kl, kn) = DECAY;
        diffuse_field(&mut self.backend, &self.grid, &mut self.temp, &mut self.c2, dt, kt);
        mirror_bounds(&self.grid, &mut self.temp);
        diffuse_field(&mut self.backend, &self.grid, &mut self.light, &mut self.c2, dl, kl);
        mirror_bounds(&self.grid, &mut self.light);
        diffuse_field(&mut self.backend, &self.grid, &mut self.nutrient, &mut self.c2, dn, kn);
        mirror_bounds(&self.grid, &mut self.nutrient);
        self.tick += 1;
    }

    #[inline]
    fn flat(&self, x: usize, y: usize) -> usize {
        // Интерьер (z=1): прометеевский пайплайн обновляет только его,
        // границы z=0/z=2 замораживает. Сиды должны ложиться в живой слой.
        self.grid.flat(x, y, 1)
    }

    /// Поля в мировой точке (0..1). Вне сетки — граница.
    pub fn sample(&self, x: f32, y: f32) -> (f32, f32, f32) {
        let p = Vec3::new(x as f64, y as f64, 0.0);
        (
            self.grid.trilinear(&self.temp, p) as f32,
            self.grid.trilinear(&self.light, p) as f32,
            self.grid.trilinear(&self.nutrient, p) as f32,
        )
    }

    /// Плодородность точки: растения растут там, где тепло И светло И сытно.
    /// Геометрическое среднее — ни один фактор не может быть нулём.
    pub fn fertility(&self, x: f32, y: f32) -> f32 {
        let (t, l, n) = self.sample(x, y);
        (t * l * n).cbrt()
    }

    /// Средняя температура (диагностика «мир потеплел»).
    pub fn mean_temp(&self) -> f32 {
        (self.temp.iter().sum::<f64>() / self.temp.len().max(1) as f64) as f32
    }

    /// Клетка с максимумом тепла (диагностика распространения).
    pub fn hot_cells(&self, above: f64) -> usize {
        self.temp.iter().filter(|&&v| v > above).count()
    }
}

/// Один шаг диффузии поля через бэкенд (CPU или GPU) с распадом.
///
/// `alpha` — доля разности с соседом, переносимая за такт (0..1/6 — предел
/// стабильности явной схемы Эйлера для 7-точечного лапласиана). Если α
/// больше предела, шаг дробится на подшаги — общий перенос за такт тот же.
fn diffuse_field(
    backend: &mut DiffusionBackend,
    grid: &Grid,
    field: &mut Vec<f64>,
    c2: &mut Vec<f64>,
    alpha: f64,
    decay: f64,
) {
    let dx2 = grid.cell * grid.cell;
    // Стабильность: per-neighbor mixing = dt*diff/dx2 = alpha/n ≤ 0.15.
    let n = ((alpha / 0.15).ceil().max(1.0)) as usize;
    let diff = alpha * dx2 / n as f64;
    let deg = decay / n as f64;
    for _ in 0..n {
        backend.diffuse(grid, field, c2, 1.0, diff, deg, 0.0, 1.0);
        std::mem::swap(field, c2);
    }
}

/// Зеркалирование граней (граничное условие Неймана, замкнутый ящик):
/// прометеевский пайплайн обновляет только интерьер, все грани замораживает.
/// Тиражируем приграничный слой на грань — тепло доходит до стен и не
/// утекает. Поле z-однородно (2D-режим на 3D-сетке), поэтому зеркалим
/// средний слой на z-грани и все x/y-грани по всем z.
fn mirror_bounds(grid: &Grid, field: &mut [f64]) {
    for x in 0..grid.nx {
        for y in 0..grid.ny {
            let mid = grid.flat(x, y, 1);
            let v = field[mid];
            field[grid.flat(x, y, 0)] = v;
            field[grid.flat(x, y, 2)] = v;
        }
    }
    for y in 0..grid.ny {
        for k in 0..grid.nz {
            field[grid.flat(0, y, k)] = field[grid.flat(1, y, k)];
            field[grid.flat(grid.nx - 1, y, k)] = field[grid.flat(grid.nx - 2, y, k)];
        }
    }
    for x in 0..grid.nx {
        for k in 0..grid.nz {
            field[grid.flat(x, 0, k)] = field[grid.flat(x, 1, k)];
            field[grid.flat(x, grid.ny - 1, k)] = field[grid.flat(x, grid.ny - 2, k)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus_grow::DiffusionBackend;

    fn backend_cpu() -> DiffusionBackend {
        DiffusionBackend::cpu()
    }

    /// Центральная комната — тёплый источник, угловые — холодные
    /// (как в арене: сид каждый шаг).
    fn warm_room() -> (Climate, WorldMap) {
        let mut world = WorldMap::default();
        world.add_room(8.0, 8.0, 12.0, 12.0);
        world.add_room(0.0, 0.0, 6.0, 6.0);
        world.add_room(14.0, 14.0, 20.0, 20.0);
        for r in &mut world.rooms {
            // Центральная комната (8..12) — тёплая.
            if (r.rect.min.x - 10.0).abs() < 3.0 && (r.rect.min.y - 10.0).abs() < 3.0 {
                r.ca[Substance::Temperature.idx()] = 200.0;
            }
        }
        let cl = Climate::over(&world, 2.0, backend_cpu());
        (cl, world)
    }

    #[test]
    fn heat_spreads_from_warm_room_into_cold() {
        let (mut cl, world) = warm_room();
        let (nx, ny) = (cl.nx, cl.ny);
        for _ in 0..60 {
            cl.seed_from_rooms(&world);
            cl.step();
        }
        let center = cl.flat(nx / 2, ny / 2);
        let corner = cl.flat(0, 0);
        assert!(
            cl.temp[center] > 0.7,
            "в источнике тепло: {}",
            cl.temp[center]
        );
        assert!(
            cl.temp[corner] > 0.05,
            "тепло дотекло в холодный угол: {}",
            cl.temp[corner]
        );
        assert!(
            cl.temp[corner] < cl.temp[center],
            "у источника теплее, чем вдали: corner={} center={}",
            cl.temp[corner],
            cl.temp[center]
        );
    }

    #[test]
    fn fertility_requires_all_factors() {
        // Только тепло в центре — света и питания нет → плодородие ~0.
        let (mut cl, mut world) = warm_room();
        for _ in 0..20 {
            cl.seed_from_rooms(&world);
            cl.step();
        }
        assert!(
            cl.fertility(10.0, 10.0) < 0.2,
            "без света и питания растения не растут: {}",
            cl.fertility(10.0, 10.0)
        );
        // Всё сразу — тёплая И светлая И сытная комната даёт урожай.
        for r in &mut world.rooms {
            if (r.rect.min.x - 10.0).abs() < 3.0 && (r.rect.min.y - 10.0).abs() < 3.0 {
                r.ca[Substance::Light.idx()] = 200.0;
                r.ca[Substance::Nutrients.idx()] = 200.0;
            }
        }
        for _ in 0..20 {
            cl.seed_from_rooms(&world);
            cl.step();
        }
        assert!(
            cl.fertility(10.0, 10.0) > 0.5,
            "тепло+свет+питание = урожай: {}",
            cl.fertility(10.0, 10.0)
        );
    }

    /// GPU-диффузия (если адаптер есть) сходится с CPU в допуске: f32 на
    /// GPU против f64 на CPU дают небольшой дрейф, не расхождение.
    #[test]
    fn gpu_diffusion_matches_cpu() {
        let gpu = pollster::block_on(DiffusionBackend::new());
        // Нет GPU — тест тривиален (тот же CPU-путь), честно.
        let (mut a, mut b) = (Climate::new(0.0, 0.0, 40.0, 40.0, 2.0, backend_cpu()), Climate::new(0.0, 0.0, 40.0, 40.0, 2.0, gpu));
        // Одинаковый источник.
        for i in 0..a.temp.len() {
            let v = ((i as f64 * 7919.0) % 1000.0) / 1000.0 * (i % 7) as f64 / 6.0;
            a.temp[i] = v;
            b.temp[i] = v;
        }
        for _ in 0..20 {
            a.step();
            b.step();
        }
        let mut max_err = 0.0f64;
        for i in 0..a.temp.len() {
            max_err = max_err.max((a.temp[i] - b.temp[i]).abs());
        }
        assert!(
            max_err < 0.05,
            "GPU-диффузия сходится с CPU: max_err={max_err}"
        );
    }
}
