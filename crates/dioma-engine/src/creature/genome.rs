//! -------------------------------------------------------------------------
//! Filename:    src/creature/genome.rs
//! Purpose:     НАСЛЕДСТВЕННОСТЬ существа — геном, который мутирует
//! Description: геном = рост (engine::grow::Steering) + инстинкты +
//!              гены правил Пифии + гены метаболизма. Родитель → ребёнок:
//!              каждый ген дрейфует (мутация), коннектом РАСТЁТ ЗАНОВО из
//!              мутировавшего Steering и нового сида. Разделение на врождённое
//!              и выученное: выученные веса мозга родителя НЕ наследуются
//!              (обнуляются рождением), а врождённые инстинкты прожигаются в
//!              свежий мозг при рождении (первый сон, `Instinct::burn_into`) —
//!              ребёнок рождается с рефлексами родителей.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use crate::body::metabolism::Blood;
use crate::engine::grow::Steering;
use crate::engine::instincts::Instinct;
use crate::engine::rules::RuleGene;
use crate::rng::XorShift;

/// Сигма мутации генома роста (относительная, на параметр).
const STEERING_SIGMA: f64 = 0.05;
/// Сигма мутации генов правил Пифии (абсолютная, как в dynasty).
const RULE_SIGMA: f32 = 0.08;
/// Сигма мутации инстинктов (относительная).
const INSTINCT_SIGMA: f32 = 0.10;
/// Сигма мутации генов метаболизма (относительная).
const METAB_SIGMA: f32 = 0.10;

/// Пол существа. Значения совпадают с C3 (`SkeletonConstants.h`:
/// `enum {MALE=1, FEMALE=2}`; `Genome.cpp mySex`). Пол наследуется в геноме,
/// но НЕ мутирует: в C3 ген пола — первый ген потомка и читается из СЛУЧАЙНО
/// выбранного родителя («sex gene, so must pick randomly!», `Genome.cpp:328`).
/// У нас «случайность» заменена детерминированным хэшем сида: один сид —
/// один пол, но по диапазону сидов полы чередуются как случайные.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sex {
    Male = 1,
    Female = 2,
}

impl Default for Sex {
    fn default() -> Self {
        Sex::Male
    }
}

impl Sex {
    /// Пол из сида существа: детерминированно (чёт/нечёт splitmix64-хэша).
    /// Соседние сиды не коррелируют по полу.
    pub fn from_seed(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        if z & 1 == 0 {
            Sex::Male
        } else {
            Sex::Female
        }
    }

    pub fn idx(self) -> usize {
        self as usize
    }
}

/// Гены метаболизма: параметры органа «мозг» (Blood::BrainOrgan).
/// Мутируют как всё остальное — тело эволюционирует вместе с мозгом.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetabolismGene {
    /// Глюкозы на единицу активности (спайк моторов за тик).
    pub cost_per_activity: f32,
    /// Урон за тик при пустой глюкозе.
    pub starve_damage: f32,
    /// Глюкоза на единицу дофамина при фиксации веса (цена обучения).
    pub cost_per_dopamine: f32,
    /// Восстановление повреждения за тик при сытости.
    pub heal_rate: f32,
}

impl Default for MetabolismGene {
    fn default() -> Self {
        // Точка отсчёта — дефолтный орган `Blood::creature`.
        // cost_per_activity НЕ ×2 (активность сама удваивается от подшагов
        // мозга), остальные — те же значения, что в Blood::creature.
        Self {
            cost_per_activity: 0.0015,
            starve_damage: 0.004,
            cost_per_dopamine: 0.4,
            heal_rate: 0.0001,
        }
    }
}

impl MetabolismGene {
    /// Мягкая мутация: относительный дрейф каждого параметра, всё ≥ 0.
    pub fn mutate(&self, rng: &mut XorShift, sigma: f32) -> Self {
        let mut drift = |v: f32| (v * (1.0 + rng.gauss() as f32 * sigma)).max(0.0);
        Self {
            cost_per_activity: drift(self.cost_per_activity),
            starve_damage: drift(self.starve_damage),
            cost_per_dopamine: drift(self.cost_per_dopamine),
            heal_rate: drift(self.heal_rate),
        }
    }

    /// Применить гены к свежей крови (вместо дефолтного органа).
    pub fn apply_to(&self, blood: &mut Blood) {
        blood.organ.cost_per_activity = self.cost_per_activity;
        blood.organ.starve_damage = self.starve_damage;
        blood.organ.cost_per_dopamine = self.cost_per_dopamine;
        blood.organ.heal_rate = self.heal_rate;
    }
}

/// Геном существа: всё, что наследуется и мутирует при размножении.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CreatureGenome {
    /// Геном роста: параметры выращивания коннектома. Мутация → другая
    /// топология и веса связей (тело мозга растёт заново).
    pub steering: Steering,
    /// Гены правил Пифии на регион (наследуются и дрейфуют).
    pub rule_genes: Vec<RuleGene>,
    /// Врождённые рефлексы (дофаминовые прошивки, прожигаются при рождении).
    pub instincts: Vec<Instinct>,
    /// Гены метаболизма тела.
    pub metabolism: MetabolismGene,
    /// Пол существа (1 = самец, 2 = самка, как в C3). Наследуется в геноме,
    /// но НЕ мутирует — пол задаётся сидом при рождении (`Sex::from_seed`).
    pub sex: Sex,
}

impl CreatureGenome {
    /// Потомок генома: каждый ген дрейфует от родительского значения.
    /// Детерминированно от сида (один сид — один потомок).
    /// Пол не мутирует (см. [`Sex`]): у C3 ген пола не дрейфует —
    /// он заново выбирается случайно при каждом наследовании.
    pub fn mutate(&self, rng: &mut XorShift) -> Self {
        Self {
            steering: self.steering.mutate(rng, STEERING_SIGMA),
            rule_genes: self
                .rule_genes
                .iter()
                .map(|g| g.mutate(rng, RULE_SIGMA))
                .collect(),
            instincts: self
                .instincts
                .iter()
                .map(|i| i.mutate(rng, INSTINCT_SIGMA))
                .collect(),
            metabolism: self.metabolism.mutate(rng, METAB_SIGMA),
            sex: self.sex,
        }
    }

    /// Эффект Болдуина: добавить инстинкты, свёрнутые из выученных путей
    /// родителя. Каждый дрейфует как обычный ген (тем же rng потомка) —
    /// приобретённый рефлекс становится наследственным, но не копируется
    /// бит-в-бит, а наследуется с изменчивостью.
    pub fn add_learned_instincts(&mut self, learned: Vec<Instinct>, rng: &mut XorShift) {
        self.instincts
            .extend(learned.into_iter().map(|i| i.mutate(rng, INSTINCT_SIGMA)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::XorShift;

    #[test]
    fn mutation_drifts_every_gene_class() {
        let base = CreatureGenome::default();
        let mut rng = XorShift::new(1234);
        let mut changed_steering = false;
        let mut changed_metab = false;
        for _ in 0..8 {
            let child = base.mutate(&mut rng);
            changed_steering |= child.steering != base.steering;
            changed_metab |= child.metabolism != base.metabolism;
        }
        assert!(
            changed_steering,
            "геном роста должен дрейфовать за 8 поколений"
        );
        assert!(changed_metab, "гены метаболизма должны дрейфовать");
    }

    #[test]
    fn mutation_is_deterministic_per_seed() {
        let base = CreatureGenome::default();
        let a = base.mutate(&mut XorShift::new(77));
        let b = base.mutate(&mut XorShift::new(77));
        assert_eq!(a, b, "один сид — один потомок генома");
    }

    #[test]
    fn metabolism_keeps_positive_and_bounded() {
        let mut rng = XorShift::new(5);
        let base = MetabolismGene::default();
        for _ in 0..200 {
            let m = base.mutate(&mut rng, 1.0); // жёсткая мутация — стресс-тест
            assert!(m.cost_per_activity >= 0.0);
            assert!(m.starve_damage >= 0.0);
            assert!(m.cost_per_dopamine >= 0.0);
            assert!(m.heal_rate >= 0.0);
        }
    }

    /// Пол не дрейфует: как в C3, ген пола не мутирует — пол задаётся
    /// заново при наследовании (у нас — из сида потомка).
    #[test]
    fn mutate_keeps_sex() {
        let base = CreatureGenome {
            sex: Sex::Female,
            ..CreatureGenome::default()
        };
        for _ in 0..16 {
            let child = base.mutate(&mut XorShift::new(42));
            assert_eq!(
                child.sex, Sex::Female,
                "мутация не должна менять пол (C3: пол не дрейфует)"
            );
        }
    }

    /// Пол из сида детерминирован и чередуется: один сид — один пол,
    /// по диапазону сидов встречаются оба пола (как случайный выбор
    /// родителя в C3, но без ГСЧ).
    #[test]
    fn sex_from_seed_is_deterministic_and_mixed() {
        for seed in [0u64, 1, 7, 42, 99] {
            assert_eq!(
                Sex::from_seed(seed),
                Sex::from_seed(seed),
                "один сид — один пол"
            );
        }
        let males = (0..64).filter(|&s| Sex::from_seed(s) == Sex::Male).count();
        assert!(
            males > 0 && males < 64,
            "по диапазону сидов должны быть оба пола: самцов {males}/64"
        );
    }
}
