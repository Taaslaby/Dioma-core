//! Научные тулзы: запись и чтение активности мозга существа.
//!
//! Чистые структуры данных — без wgpu и окна: движок остаётся headless,
//! а визуализация (пример `examples/sci.rs`) читает эти записи. Всё тянется
//! ЧЕРЕЗ API существа:
//!
//! * спайки — [`crate::creature::Creature::tick`] → [`spike_hist`](crate::creature::Creature::spike_hist)
//!   (по одному вектору на подшаг мозга);
//! * мембранный потенциал — [`Brain::debug_cpu_v`](crate::engine::neurons::Brain::debug_cpu_v);
//! * регионы — [`Brain::region_bounds`](crate::engine::neurons::Brain::region_bounds).
//!
//! ## Что внутри
//!
//! * [`SpikeRaster`] — кольцевой буфер «время × нейроны» (классический
//!   спайк-растр, по столбцу на подшаг);
//! * [`Trace`] — кольцевой буфер сэмплов мембранного потенциала с
//!   автодиапазоном (min/max);
//! * [`Activity`] — скользящее окно скорости спайков по нейронам и регионам.

use std::collections::VecDeque;

/// Спайк-растр: кольцевой буфер столбцов «время × нейроны».
///
/// Каждый `push` — один столбец (подшаг мозга): `spikes[i]` — число спайков
/// нейрона `i` в этом подшаге. При переполнении окна самый старый столбец
/// вытесняется. Питается из `Creature::spike_hist`.
#[derive(Clone, Debug)]
pub struct SpikeRaster {
    n_neurons: usize,
    window: usize,
    columns: VecDeque<Vec<i32>>,
    total_spikes: u64,
}

impl SpikeRaster {
    /// Растр на `n_neurons` нейронов с окном `window` столбцов.
    pub fn new(n_neurons: usize, window: usize) -> Self {
        Self { n_neurons, window: window.max(1), columns: VecDeque::with_capacity(window), total_spikes: 0 }
    }

    /// Добавляет столбец спайков подшага. Паникует (русское сообщение) при
    /// несовпадении длины — это признак рассинхронизации с мозгом.
    pub fn push(&mut self, spikes: &[i32]) {
        assert_eq!(
            spikes.len(),
            self.n_neurons,
            "SpikeRaster::push: столбец длины {}, ожидалось {} (мозг сменил размер?)",
            spikes.len(),
            self.n_neurons
        );
        self.total_spikes += spikes.iter().map(|&s| s.max(0) as u64).sum::<u64>();
        self.columns.push_back(spikes.to_vec());
        while self.columns.len() > self.window {
            self.columns.pop_front();
        }
    }

    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Ёмкость окна в столбцах.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Сколько столбцов накоплено (≤ окна).
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Столбец `i` от старейшего (0) к свежайшему.
    pub fn column(&self, i: usize) -> Option<&[i32]> {
        self.columns.get(i).map(|c| c.as_slice())
    }

    /// Итерация столбцов от старейшего к свежайшему.
    pub fn iter(&self) -> impl Iterator<Item = &[i32]> {
        self.columns.iter().map(|c| c.as_slice())
    }

    /// Суммарное число спайков за всё время записи.
    pub fn total_spikes(&self) -> u64 {
        self.total_spikes
    }

    /// Число спайков нейрона за текущее окно (для активностей и гистограмм).
    pub fn spike_count(&self, neuron: usize) -> u32 {
        self.columns.iter().map(|c| c[neuron].max(0) as u32).sum()
    }
}

/// Трейс мембранного потенциала: кольцевой буфер сэмплов «нейроны × время».
///
/// Каждый `push` — один сэмпл (тик): `values[i]` — потенциал нейрона `i` в
/// этот момент. Автодиапазон `min/max` обновляется инкрементально и хранит
/// ГЛОБАЛЬНЫЙ минимум/максимум с момента создания (для стабильной шкалы
/// осциллографа сбросьте трейс через [`Self::reset`]).
#[derive(Clone, Debug)]
pub struct Trace {
    n: usize,
    window: usize,
    samples: VecDeque<Vec<f32>>,
    min: f32,
    max: f32,
}

impl Trace {
    /// Трейс на `n` нейронов с окном `window` сэмплов.
    pub fn new(n: usize, window: usize) -> Self {
        Self {
            n,
            window: window.max(1),
            samples: VecDeque::with_capacity(window),
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }

    /// Добавляет сэмпл потенциалов тика.
    pub fn push(&mut self, values: &[f32]) {
        assert_eq!(
            values.len(),
            self.n,
            "Trace::push: сэмпл длины {}, ожидалось {}",
            values.len(),
            self.n
        );
        for &v in values {
            if v.is_finite() {
                self.min = self.min.min(v);
                self.max = self.max.max(v);
            }
        }
        self.samples.push_back(values.to_vec());
        while self.samples.len() > self.window {
            self.samples.pop_front();
        }
    }

    /// Очищает историю и автодиапазон (шкала пересчитается заново).
    pub fn reset(&mut self) {
        self.samples.clear();
        self.min = f32::INFINITY;
        self.max = f32::NEG_INFINITY;
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// Ёмкость окна в сэмплах.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Сколько сэмплов накоплено (≤ окна).
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Сэмпл `i` от старейшего (0) к свежайшему.
    pub fn sample(&self, i: usize) -> Option<&[f32]> {
        self.samples.get(i).map(|s| s.as_slice())
    }

    /// Ряд потенциалов нейрона `neuron` от старейшего к свежайшему.
    pub fn series(&self, neuron: usize) -> Vec<f32> {
        self.samples.iter().map(|s| s[neuron]).collect()
    }

    /// Глобальный минимум (не `NaN`; `f32::INFINITY` при пустом трейсе).
    pub fn min(&self) -> f32 {
        self.min
    }

    /// Глобальный максимум (не `NaN`; `-INFINITY` при пустом трейсе).
    pub fn max(&self) -> f32 {
        self.max
    }
}

/// Активность: скорость спайков по нейронам и регионам за скользящее окно.
///
/// Питается целыми тиками: `push_tick` принимает пачку столбцов подшагов
/// тика (как `Creature::spike_hist`), суммирует спайки по каждому нейрону и
/// держит кольцо тиков. `rate` нормирована на один тик (спайков/тик).
#[derive(Clone, Debug)]
pub struct Activity {
    n_neurons: usize,
    window_ticks: usize,
    /// Кольцо тиков: сумма спайков по нейронам за тик.
    ticks: VecDeque<Vec<u32>>,
    /// Диапазоны регионов `(start, end)` — для группировки гистограмм.
    region_ranges: Vec<(usize, usize)>,
    /// Имена регионов (совпадают по индексу с `region_ranges`).
    region_names: Vec<String>,
}

impl Activity {
    /// Счётчик на `n_neurons` нейронов, окно `window_ticks` тиков.
    ///
    /// `regions` — пары `(имя, диапазон)`, например построенные из
    /// `Brain::region_bounds` для дефолтного существа.
    pub fn new(n_neurons: usize, window_ticks: usize, regions: Vec<(String, (usize, usize))>) -> Self {
        let region_ranges: Vec<(usize, usize)> = regions.iter().map(|(_, r)| *r).collect();
        let region_names: Vec<String> = regions.into_iter().map(|(n, _)| n).collect();
        Self { n_neurons, window_ticks: window_ticks.max(1), ticks: VecDeque::with_capacity(window_ticks.max(1)), region_ranges, region_names }
    }

    /// Записывает тик: пачка столбцов подшагов (`SpikeRaster`-совместимая).
    pub fn push_tick(&mut self, columns: &[Vec<i32>]) {
        let mut per_neuron = vec![0u32; self.n_neurons];
        for col in columns {
            assert_eq!(
                col.len(),
                self.n_neurons,
                "Activity::push_tick: столбец длины {}, ожидалось {}",
                col.len(),
                self.n_neurons
            );
            for (dst, &s) in per_neuron.iter_mut().zip(col) {
                *dst += s.max(0) as u32;
            }
        }
        self.ticks.push_back(per_neuron);
        while self.ticks.len() > self.window_ticks {
            self.ticks.pop_front();
        }
    }

    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    /// Сколько тиков в окне.
    pub fn len(&self) -> usize {
        self.ticks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ticks.is_empty()
    }

    /// Скорость нейрона `neuron`: спайков за тик в среднем по окну.
    pub fn rate(&self, neuron: usize) -> f32 {
        if self.ticks.is_empty() {
            return 0.0;
        }
        let sum: u32 = self.ticks.iter().map(|t| t[neuron]).sum();
        sum as f32 / self.ticks.len() as f32
    }

    /// Средняя скорость по региону `region` (спайков/тик на нейрон).
    pub fn region_rate(&self, region: usize) -> f32 {
        let (start, end) = self.region_ranges[region];
        if start >= end || self.ticks.is_empty() {
            return 0.0;
        }
        let sum: u64 = self
            .ticks
            .iter()
            .map(|t| t[start..end].iter().map(|&s| s as u64).sum::<u64>())
            .sum();
        sum as f32 / (self.ticks.len() as f32 * (end - start) as f32)
    }

    /// Имя региона `region`.
    pub fn region_name(&self, region: usize) -> &str {
        &self.region_names[region]
    }

    /// Сколько регионов задано.
    pub fn n_regions(&self) -> usize {
        self.region_ranges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_keeps_window_and_counts() {
        let mut r = SpikeRaster::new(3, 2);
        r.push(&[1, 0, 2]);
        r.push(&[0, 1, 0]);
        r.push(&[1, 1, 1]);
        assert_eq!(r.len(), 2, "окно 2 столбца");
        assert_eq!(r.column(0).unwrap(), &[0, 1, 0], "старый столбец вытеснен");
        assert_eq!(r.total_spikes(), 7, "сумма за всё время (1+2+1+1+1+1)");
        assert_eq!(r.spike_count(2), 1, "нейрон 2 по окну: 0 + 1");
        assert_eq!(r.n_neurons(), 3);
    }

    #[test]
    #[should_panic(expected = "столбец длины")]
    fn raster_rejects_wrong_width() {
        let mut r = SpikeRaster::new(3, 8);
        r.push(&[1, 2]);
    }

    #[test]
    fn trace_tracks_autoscale_and_order() {
        let mut t = Trace::new(2, 3);
        assert!(!t.min().is_finite(), "пустой трейс: диапазона нет");
        t.push(&[-52.0, -45.0]);
        t.push(&[-50.0, -30.0]);
        t.push(&[-55.0, -48.0]);
        t.push(&[-51.0, -47.0]); // вытесняет первый
        assert_eq!(t.len(), 3);
        assert_eq!(t.min(), -55.0);
        assert_eq!(t.max(), -30.0);
        assert_eq!(t.series(0), vec![-50.0, -55.0, -51.0], "старейший → свежайший");
        t.reset();
        assert!(t.is_empty() && !t.min().is_finite());
    }

    #[test]
    fn activity_rates_by_region() {
        // 4 нейрона: [0,1] — регион "S", [2,3] — регион "M".
        let mut a = Activity::new(4, 3, vec![("S".to_string(), (0, 2)), ("M".to_string(), (2, 4))]);
        a.push_tick(&[vec![1, 0, 0, 2], vec![1, 0, 0, 2]]); // нейрон 0: 2 спайка/тик
        a.push_tick(&[vec![0, 0, 0, 0]]);
        assert_eq!(a.len(), 2);
        assert!((a.rate(0) - 1.0).abs() < 1e-6, "нейрон 0: 2 спайка за 2 тика");
        assert!((a.region_rate(1) - 1.0).abs() < 1e-6, "M: 4 спайка / (2 тика × 2 нейрона)");
        assert!((a.region_rate(0) - 0.5).abs() < 1e-6, "S: 2 спайка / (2 × 2)");
        assert_eq!(a.region_name(1), "M");
        assert_eq!(a.n_regions(), 2);
    }
}
