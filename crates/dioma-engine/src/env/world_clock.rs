//! Мировые часы: время суток, день в сезоне и сезон — всё выводится из
//! мирового тика арены (`Arena.tick`), отдельного счётчика нет.
//!
//! Формулы и константы — как в Creatures 3 (`engine/World.cpp:80-87, 882-992`):
//!
//! - день = 20 реальных минут (24 000 тиков при тике 50 мс);
//! - 5 фаз суток: рассвет / утро / день / вечер / ночь — граница
//!   `phase = (tick % TICKS_PER_DAY) / (TICKS_PER_DAY / 5)`;
//! - сезон = 3 дня, 4 сезона, год = 12 дней.
//!
//! Освещённость фазы (`light_multiplier`) гейтит свет комнат в арене: ночью
//! мир гаснет, и ягоды (плодородность = тепло × свет × питание) растут
//! медленнее — как в C3, где свет движет ростом растений.
//!
//! ВАЖНО: сон существ НЕ привязан к ночи. В C3 существа вообще не спят;
//! наш усталостный сон (fatigue-химия, `sim::mod::SLEEP_AT_FATIGUE`) — наша
//! фича, и он работает независимо от фазы суток. Ночь влияет только на свет.

/// Мировой тик, мс (как в C3: `App::GetWorldTickInterval() == 50`,
/// `engine/App.h:157`). Транш 2: 25 мс → 50 мс.
pub const TICK_MS: u32 = 50;

/// Длина суток в реальных минутах (C3: `myDayLengthInMinutes = 20`).
pub const DAY_LENGTH_MINUTES: u32 = 20;

/// Тиков в сутках: `(1000 / 50) × 60 × 20` = 24 000 тиков.
pub const TICKS_PER_DAY: u64 = (1000 / TICK_MS) as u64 * 60 * DAY_LENGTH_MINUTES as u64;

/// Дней в сезоне (C3: `mySeasonLengthInDays = 3`).
pub const DAYS_PER_SEASON: u32 = 3;

/// Число сезонов (C3: `mySeasonCount = 4`).
pub const SEASONS: u32 = 4;

/// Дней в году: 4 сезона × 3 дня (C3: `myYearLengthInDays`).
pub const YEAR_DAYS: u32 = DAYS_PER_SEASON * SEASONS;

/// Фаза суток (0..4): рассвет / утро / день / вечер / ночь.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimeOfDay {
    Dawn = 0,
    Morning = 1,
    Afternoon = 2,
    Evening = 3,
    Night = 4,
}

/// Сезон (0..3): весна / лето / осень / зима.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Season {
    Spring = 0,
    Summer = 1,
    Autumn = 2,
    Winter = 3,
}

impl TimeOfDay {
    pub fn idx(self) -> usize {
        self as usize
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl Season {
    pub fn idx(self) -> usize {
        self as usize
    }
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Освещённость фазы суток — множитель света комнат (0..1). Константы
/// подобраны под визуал и рост: рассвет едва светлеет, ночь почти гасит.
pub const PHASE_LIGHT: [f64; 5] = [0.4, 0.8, 1.0, 0.6, 0.15];

/// Статус-объект без состояния: мировые часы — чистые функции от тика.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorldClock;

impl WorldClock {
    /// Фаза суток из тика: `(tick % TICKS_PER_DAY) / (TICKS_PER_DAY / 5)`.
    pub fn time_of_day(tick: u64) -> TimeOfDay {
        const N: u64 = 5;
        let interval = tick % TICKS_PER_DAY;
        match (interval / (TICKS_PER_DAY / N)) as usize {
            0 => TimeOfDay::Dawn,
            1 => TimeOfDay::Morning,
            2 => TimeOfDay::Afternoon,
            3 => TimeOfDay::Evening,
            _ => TimeOfDay::Night,
        }
    }

    /// День внутри сезона (0..=2): `(tick / TICKS_PER_DAY) % DAYS_PER_SEASON`.
    pub fn day_in_season(tick: u64) -> u32 {
        ((tick / TICKS_PER_DAY) % DAYS_PER_SEASON as u64) as u32
    }

    /// Сезон (0..=3): `(tick / (TICKS_PER_DAY × 3)) % 4`.
    pub fn season(tick: u64) -> Season {
        let season_ticks = TICKS_PER_DAY * DAYS_PER_SEASON as u64;
        match ((tick / season_ticks) % SEASONS as u64) as usize {
            0 => Season::Spring,
            1 => Season::Summer,
            2 => Season::Autumn,
            _ => Season::Winter,
        }
    }

    /// Прогресс внутри фазы 0..1 (для рендера: плавный переход сумерек).
    pub fn day_phase_fraction(tick: u64) -> f32 {
        let phase_len = TICKS_PER_DAY / 5;
        ((tick % TICKS_PER_DAY) % phase_len) as f32 / phase_len as f32
    }

    /// Множитель света фазы суток: ночью мир гаснет (см. [`PHASE_LIGHT`]).
    pub fn light_multiplier(tick: u64) -> f64 {
        PHASE_LIGHT[Self::time_of_day(tick).idx()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_cycle_through_the_day() {
        let phase = TICKS_PER_DAY / 5;
        assert_eq!(TICKS_PER_DAY, 24_000, "день = 20 минут при тике 50 мс");
        assert_eq!(WorldClock::time_of_day(0), TimeOfDay::Dawn);
        assert_eq!(WorldClock::time_of_day(phase - 1), TimeOfDay::Dawn);
        assert_eq!(WorldClock::time_of_day(phase), TimeOfDay::Morning);
        assert_eq!(WorldClock::time_of_day(2 * phase), TimeOfDay::Afternoon);
        assert_eq!(WorldClock::time_of_day(3 * phase), TimeOfDay::Evening);
        assert_eq!(WorldClock::time_of_day(4 * phase), TimeOfDay::Night);
        assert_eq!(
            WorldClock::time_of_day(TICKS_PER_DAY - 1),
            TimeOfDay::Night
        );
        assert_eq!(
            WorldClock::time_of_day(TICKS_PER_DAY),
            TimeOfDay::Dawn,
            "сутки зациклились"
        );
    }

    #[test]
    fn days_seasons_and_years_wrap() {
        let day = TICKS_PER_DAY;
        assert_eq!(WorldClock::day_in_season(0), 0);
        assert_eq!(WorldClock::day_in_season(day), 1);
        assert_eq!(WorldClock::day_in_season(2 * day), 2);
        assert_eq!(WorldClock::day_in_season(3 * day), 0, "сезон из 3 дней");

        assert_eq!(WorldClock::season(0), Season::Spring);
        assert_eq!(WorldClock::season(3 * day), Season::Summer);
        assert_eq!(WorldClock::season(6 * day), Season::Autumn);
        assert_eq!(WorldClock::season(9 * day), Season::Winter);
        assert_eq!(WorldClock::season(12 * day), Season::Spring, "год зациклился");
    }

    #[test]
    fn phase_fraction_grows_and_resets() {
        let phase = TICKS_PER_DAY / 5;
        assert_eq!(WorldClock::day_phase_fraction(0), 0.0);
        assert!((WorldClock::day_phase_fraction(phase / 2) - 0.5).abs() < 1e-6);
        assert!(WorldClock::day_phase_fraction(phase - 1) < 1.0);
        assert_eq!(WorldClock::day_phase_fraction(phase), 0.0, "граница фазы");
    }

    #[test]
    fn night_light_is_dimmer_than_afternoon() {
        let afternoon = WorldClock::light_multiplier(2 * TICKS_PER_DAY / 5);
        let night = WorldClock::light_multiplier(4 * TICKS_PER_DAY / 5);
        assert_eq!(afternoon, 1.0, "днём свет полный");
        assert!(night < 0.2, "ночью свет почти гаснет: {night}");
    }
}
