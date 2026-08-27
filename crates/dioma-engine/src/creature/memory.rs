//! Память места (гиппокампоподобный слой).
//!
//! Существо запоминает, в каком направлении была еда, по грубому контексту
//! места (`Sense::place` — грубая клетка карты, которую даёт хост) и своему
//! курсу (`Sense::heading`). Когда запаха нет (градиента нет), память
//! подсказывает, куда рулить, чтобы вернуться к знакомой еде. Во сне
//! (`consolidate`) слабые места забываются — память не копит мусор.
//!
//! Направления квантуются в `N_BINS` бинов (по 45°). Направление еды
//! запоминается ОТНОСИТЕЛЬНО курса: бин, в который существо смотрело,
//! когда ело. При возврате с другим курсом руль тянет выровнять курс
//! на запомненный бин.

use std::collections::HashMap;

/// Число бинов направления (8 × 45°).
pub const N_BINS: usize = 8;
/// Радиус бина в радианах.
pub const BIN_RAD: f32 = std::f32::consts::TAU / N_BINS as f32;
/// Минимум «удачных» посещений места, чтобы память считалась настоящей
/// (один случайный приём пищи не создаёт память).
const MIN_HITS: u32 = 3;

/// Гиппокампоподобная память мест.
#[derive(Clone, Debug, Default)]
pub struct Hippocampus {
    /// place → веса бинов (сила воспоминания «еда в этом направлении»).
    places: HashMap<u32, [f32; N_BINS]>,
    /// Сколько раз место запоминалось с едой (фильтр шума).
    hits: HashMap<u32, u32>,
    /// Сколько всего воспоминаний накоплено за жизнь.
    total: u32,
}

impl Hippocampus {
    /// Запомнить: в этом месте, глядя в направлении `heading` (радианы),
    /// была еда. Вызывается при каждом «съел».
    pub fn remember(&mut self, place: u32, heading: f32) {
        let bin = bin_of(heading);
        let e = self.places.entry(place).or_insert([0.0; N_BINS]);
        e[bin] += 1.0;
        *self.hits.entry(place).or_insert(0) += 1;
        self.total += 1;
    }

    /// Вспомнить сильнейшее направление еды в этом месте. Возвращает
    /// центр бина (радианы) и силу (доля всех воспоминаний места).
    /// `None` — место незнакомо или память слишком слабая.
    pub fn recall(&self, place: u32) -> Option<(f32, f32)> {
        let weights = self.places.get(&place)?;
        let hits = *self.hits.get(&place)?;
        if hits < MIN_HITS {
            return None;
        }
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 {
            return None;
        }
        let (best, &w) = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))?;
        Some((bin_center(best), w / sum))
    }

    /// Консолидация во сне: забыть места с единичными удачами и ослабить
    /// старые воспоминания (прореживание — мозг не копит шум).
    pub fn consolidate(&mut self) {
        let weak: Vec<u32> = self
            .hits
            .iter()
            .filter(|(_, &h)| h < MIN_HITS)
            .map(|(&p, _)| p)
            .collect();
        for p in weak {
            self.places.remove(&p);
            self.hits.remove(&p);
        }
        // Старые воспоминания слегка затухают (полураспад силы).
        for w in self.places.values_mut() {
            for v in w.iter_mut() {
                *v *= 0.9;
            }
        }
    }

    /// Сколько мест запомнено (диагностика).
    pub fn places_remembered(&self) -> usize {
        self.places.len()
    }

    /// Сколько всего воспоминаний за жизнь.
    pub fn total(&self) -> u32 {
        self.total
    }
}

/// Бин направления (0..N_BINS).
pub fn bin_of(heading: f32) -> usize {
    ((heading.rem_euclid(std::f32::consts::TAU) / BIN_RAD) as usize) % N_BINS
}

/// Центр бина в радианах.
pub fn bin_center(bin: usize) -> f32 {
    (bin as f32 + 0.5) * BIN_RAD
}

/// Руль из памяти: на сколько повернуть ([-1..1], +1 = вправо), чтобы
/// выровнять текущий курс `heading` на запомненное направление `target`.
/// Конвенция: heading — радианы против часовой от +x; «вправо» = по часовой.
pub fn steer_toward(heading: f32, target: f32) -> f32 {
    let err = (heading - target).rem_euclid(std::f32::consts::TAU);
    let err = if err > std::f32::consts::PI {
        err - std::f32::consts::TAU
    } else {
        err
    };
    (err / std::f32::consts::PI).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remembers_and_recalls_direction_at_place() {
        let mut h = Hippocampus::default();
        assert!(h.recall(5).is_none(), "незнакомое место — нет памяти");
        // Ели трижды, глядя на восток (heading 0).
        for _ in 0..3 {
            h.remember(5, 0.0);
        }
        let (dir, strength) = h.recall(5).expect("место запомнено");
        assert_eq!(bin_of(dir), bin_of(0.0), "направление = куда смотрели");
        assert!(strength > 0.5, "восток доминирует: {strength}");
        assert_eq!(h.places_remembered(), 1);
        assert_eq!(h.total(), 3);
    }

    #[test]
    fn one_lucky_meal_is_not_a_memory() {
        let mut h = Hippocampus::default();
        h.remember(7, 1.0);
        assert!(h.recall(7).is_none(), "один приём пищи — ещё не память");
    }

    #[test]
    fn consolidate_forgets_weak_places() {
        let mut h = Hippocampus::default();
        for _ in 0..3 {
            h.remember(1, 0.0); // сильное место
        }
        h.remember(2, 1.0); // слабое место (1 удача)
        h.consolidate();
        assert!(h.recall(1).is_some(), "сильное место пережило сон");
        assert!(h.recall(2).is_none(), "слабое место забыто во сне");
    }

    #[test]
    fn steer_toward_aligns_heading() {
        // Смотрим на север (π/2), еда была на востоке (0): поворот вправо (+).
        let s = steer_toward(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(s > 0.3, "на восток от севера — вправо: {s}");
        // Смотрим на север (π/2), еда на западе (π): поворот влево (−).
        let s2 = steer_toward(std::f32::consts::FRAC_PI_2, std::f32::consts::PI);
        assert!(s2 < -0.3, "на запад от севера — влево: {s2}");
        // Уже смотрим на еду — руля нет.
        assert!(steer_toward(0.5, 0.5).abs() < 1e-4);
    }
}
