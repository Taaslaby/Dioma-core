//! -------------------------------------------------------------------------
//! Filename:    src/mind.rs
//! Class:       neuworks
//! Purpose:     bridge body → karmac SpikingBrain → behavior
//! Description: кровь/температура/сенсорика OrganSim → 8 семантических
//!              каналов SpikingBrain (GNW → базальные ганглии → WTA) —
//!              назад приходит действие: сидеть / искать / подойти /
//!              бежать / бродить. Награды (еда +, стена −) учат приор.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use taas_checkmate::karmac::brain::{BrainPrior, SpikingBrain};

/// Словарь действий существа (индексы = выходные каналы SpikingBrain).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreatureAction {
    Explore = 0,
    SeekFood = 1,
    Approach = 2,
    Flee = 3,
    Rest = 4,
}

impl CreatureAction {
    pub fn from_index(i: usize) -> Self {
        match i {
            1 => Self::SeekFood,
            2 => Self::Approach,
            3 => Self::Flee,
            4 => Self::Rest,
            _ => Self::Explore,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Explore => "бродит",
            Self::SeekFood => "ищет еду",
            Self::Approach => "подходит",
            Self::Flee => "бежит",
            Self::Rest => "сидит",
        }
    }
}

/// Приор существа: какие каналы поддерживают какие действия.
///
/// Каналы:
/// * c0 hunger      — голод [0..1]
/// * c1 freshness   — бодрость (1 − усталость)
/// * c2 smell_L     — запах еды слева
/// * c3 smell_R     — запах еды справа
/// * c4 pain        — свежая боль/удар
/// * c5 rest_comfort— комфорт покоя (сытость × нет боли)
/// * c6 fever       — лихорадка [0..1]
/// * c7 glucose     — запас глюкозы
///
/// Действия: Explore←freshness · SeekFood←hunger · Approach←smell L/R ·
/// Flee←pain · Rest←rest_comfort.
pub fn creature_prior() -> BrainPrior {
    BrainPrior {
        action_support: [
            [1, 1], // Explore  ← freshness (бюджет блуждания)
            [0, 0], // SeekFood ← hunger
            [2, 3], // Approach ← smell L/R
            [4, 4], // Flee     ← pain
            [5, 5], // Rest     ← rest_comfort
        ],
        confusion: [
            (0, 5), // Explore от комфорта (бродит когда пора спать)
            (1, 7), // Seek от глюкозы (идёт к еде сытый)
            (3, 0), // Flee от голода (паникует на пустой желудок)
            (4, 1), // Rest от бодрости (дремлет на бодряке)
        ],
    }
}

/// Каналы восприятия из состояния тела.
#[derive(Clone, Copy)]
pub struct BodyChannels {
    pub hunger: f32,
    pub freshness: f32,
    pub smell_l: f32,
    pub smell_r: f32,
    pub pain: f32,
    pub rest_comfort: f32,
    pub fever: f32,
    pub glucose: f32,
}

/// Разум существа: SpikingBrain + последнее решение.
pub struct CreatureMind {
    brain: SpikingBrain,
    /// Последнее выбранное действие.
    pub last_action: CreatureAction,
    /// Гистерезис покоя: сколько тиков ещё держать «сидим» после Rest.
    pub rest_hold_ticks: usize,
}

impl CreatureMind {
    pub fn new(seed: u64) -> Self {
        Self {
            brain: SpikingBrain::train_init_with(seed, creature_prior()),
            last_action: CreatureAction::Explore,
            rest_hold_ticks: 0,
        }
    }

    /// Каналы из состояния OrganSim. `fatigue01`, `fever01` — нормировки;
    /// `sated01` — сытость (1 = полная); `pain_recent` затухает сам.
    #[allow(clippy::too_many_arguments)]
    pub fn channels(
        hunger01: f32,
        fatigue01: f32,
        smell_l: f32,
        smell_r: f32,
        pain_recent: f32,
        sated01: f32,
        fever01: f32,
        glucose01: f32,
    ) -> [f32; 8] {
        [
            hunger01.clamp(0.0, 1.0),
            (1.0 - fatigue01).clamp(0.0, 1.0),
            smell_l.clamp(0.0, 1.0),
            smell_r.clamp(0.0, 1.0),
            pain_recent.clamp(0.0, 1.0),
            (sated01 * (1.0 - pain_recent)).clamp(0.0, 1.0),
            fever01.clamp(0.0, 1.0),
            glucose01.clamp(0.0, 1.0),
        ]
    }

    /// Решение по каналам; возвращает действие и держит гистерезис покоя.
    pub fn decide(&mut self, channels: &[f32; 8]) -> CreatureAction {
        let idx = self.brain.step(channels);
        let action = CreatureAction::from_index(idx);
        // Гистерезис: Rest держится, пока каналы комфорта не рухнули.
        if action == CreatureAction::Rest {
            self.rest_hold_ticks = self.rest_hold_ticks.max(40);
        } else if self.rest_hold_ticks > 0 && channels[5] > 0.25 {
            self.rest_hold_ticks -= 1;
            return CreatureAction::Rest;
        } else {
            self.rest_hold_ticks = 0;
        }
        self.last_action = action;
        action
    }

    /// Дофаминовое подкрепление последнего решения.
    pub fn reward(&mut self, r: f32) {
        self.brain.apply_reward(r);
    }

    /// Тепловая карта весов W2 для GUI.
    pub fn weight_heat(&self) -> Vec<f32> {
        self.brain.weight_heat()
    }

    /// Средние частоты выходных пулов по действиям.
    pub fn action_rates(&self) -> &[f32] {
        self.brain.action_rates()
    }

    /// Текущий дофамин.
    pub fn dopamine(&self) -> f32 {
        self.brain.dopamine()
    }

    /// EMA-базлайн награды.
    pub fn baseline(&self) -> f32 {
        self.brain.baseline()
    }

    /// Гейт регионов (кто выиграл гонку моторов).
    pub fn region_gate(&self) -> &[f32] {
        self.brain.region_gate()
    }
}
