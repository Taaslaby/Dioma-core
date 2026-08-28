//! Анимации UI: keyed transition поверх immediate-mode.
//!
//! Заимствовано из gpui-component (`crates/base/src/motion.rs`): значение
//! интерполируется к цели по ключу `(id, канал)`; состояние переживает кадры,
//! при смене цели в полёте движение стартует от ТЕКУЩЕГО значения (без
//! скачков), первый кадр — мгновенный снап.
//!
//! Каналы нужны, чтобы анимировать несколько величин одного виджета
//! (`(id, CH_FILL)`, `(id, CH_HOVER)`) независимо.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// ease_out_cubic — дефолт gpui-component: быстро стартует, мягко доезжает.
pub fn ease_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(3)
}

/// Каналы анимации одного виджета.
pub mod channels {
    /// Заливка прогресс-бара (0..1).
    pub const FILL: u8 = 0;
    /// Подсветка hover кнопки (0..1).
    pub const HOVER: u8 = 1;
    /// Позиция бегунка switch (0..1).
    pub const THUMB: u8 = 2;
}

/// Сэмплирует интерполяцию анимации в момент `now`.
fn sample(a: &Anim, now: Instant) -> f32 {
    let t = now.duration_since(a.started).as_secs_f32() / a.dur.as_secs_f32().max(1e-6);
    a.value + (a.target - a.value) * ease_out_cubic(t)
}

/// Одна анимация: едет от текущего значения к цели за `dur`.
#[derive(Clone, Copy, Debug)]
struct Anim {
    value: f32,
    target: f32,
    started: Instant,
    dur: Duration,
}

/// Хранилище анимаций, живёт в [`crate::ui::UiSystem`] между кадрами.
#[derive(Default)]
pub struct Motion {
    anims: HashMap<(u64, u8), Anim>,
}

impl Motion {
    pub fn new() -> Self {
        Self::default()
    }

    /// Текущее значение по каналу, плавно едущее к `target` за `dur`.
    ///
    /// * первый вызов — мгновенный снап к цели (без «выезда из нуля»);
    /// * смена цели в полёте — старт от текущего значения;
    /// * `reduced_motion` — всегда снап (глобальный флаг доступности).
    pub fn animate(
        &mut self,
        id: u64,
        channel: u8,
        target: f32,
        dur: Duration,
        now: Instant,
        reduced_motion: bool,
    ) -> f32 {
        const EPS: f32 = 0.001;
        let Some(a) = self.anims.get_mut(&(id, channel)) else {
            // Первый кадр: сразу к цели, анимации нет.
            self.anims.insert((id, channel), Anim { value: target, target, started: now, dur });
            return target;
        };
        if reduced_motion || dur.is_zero() || (a.value - target).abs() < EPS {
            a.value = target;
            a.target = target;
            return target;
        }
        if a.target != target {
            // Смена цели в полёте: стартуем от текущего сэмпла.
            a.value = sample(a, now);
            a.target = target;
            a.started = now;
        }
        let v = sample(a, now);
        a.value = v;
        v
    }

    /// Забыть анимации виджета (виджет исчез — состояние не копится).
    pub fn forget(&mut self, id: u64) {
        self.anims.retain(|(wid, _), _| *wid != id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_frame_snaps_to_target() {
        let mut m = Motion::new();
        let now = Instant::now();
        let v = m.animate(1, channels::FILL, 0.7, Duration::from_millis(160), now, false);
        assert_eq!(v, 0.7, "первый кадр — снап, без выезда из нуля");
    }

    #[test]
    fn eases_toward_target_without_overshoot() {
        let mut m = Motion::new();
        let now = Instant::now();
        m.animate(1, channels::FILL, 0.0, Duration::from_millis(160), now, false);
        // Задаём цель в t=0, через 80мс (половина пути) ease_out_cubic(0.5)=0.875.
        m.animate(1, channels::FILL, 1.0, Duration::from_millis(160), now, false);
        let v = m.animate(1, channels::FILL, 1.0, Duration::from_millis(160), now + Duration::from_millis(80), false);
        assert!((v - 0.875).abs() < 0.05, "v={v}");
        // В конце — точно цель.
        let v2 = m.animate(1, channels::FILL, 1.0, Duration::from_millis(160), now + Duration::from_millis(400), false);
        assert!((v2 - 1.0).abs() < 0.001, "в конце — цель: v2={v2}");
    }

    #[test]
    fn target_change_starts_from_current_value() {
        let mut m = Motion::new();
        let now = Instant::now();
        m.animate(1, channels::HOVER, 0.0, Duration::from_millis(160), now, false);
        m.animate(1, channels::HOVER, 1.0, Duration::from_millis(160), now, false);
        // На середине пути значение между 0 и 1.
        let mid = m.animate(1, channels::HOVER, 1.0, Duration::from_millis(160), now + Duration::from_millis(80), false);
        assert!(mid > 0.5 && mid < 1.0, "в полёте: {mid}");
        // Разворот цели на полпути — старт от текущего значения, скачка нет.
        let back = m.animate(1, channels::HOVER, 0.0, Duration::from_millis(160), now + Duration::from_millis(100), false);
        let after = m.animate(1, channels::HOVER, 0.0, Duration::from_millis(160), now + Duration::from_millis(101), false);
        assert!(back > 0.7, "без скачка к нулю: {back}");
        assert!(after < back, "поехали назад: {after} < {back}");
    }

    #[test]
    fn reduced_motion_snaps() {
        let mut m = Motion::new();
        let now = Instant::now();
        m.animate(1, channels::FILL, 0.0, Duration::from_millis(160), now, false);
        let v = m.animate(1, channels::FILL, 1.0, Duration::from_millis(160), now + Duration::from_millis(10), true);
        assert_eq!(v, 1.0, "reduced motion — мгновенный снап");
    }

    #[test]
    fn channels_are_independent() {
        let mut m = Motion::new();
        let now = Instant::now();
        m.animate(1, channels::FILL, 0.0, Duration::from_millis(160), now, false);
        m.animate(1, channels::HOVER, 1.0, Duration::from_millis(160), now, false);
        m.animate(1, channels::FILL, 1.0, Duration::from_millis(160), now, false);
        let f = m.animate(1, channels::FILL, 1.0, Duration::from_millis(160), now + Duration::from_millis(80), false);
        assert!(f > 0.5, "канал FILL едет к своей цели независимо: {f}");
    }
}
