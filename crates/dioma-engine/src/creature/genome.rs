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
        Self {
            cost_per_activity: 0.0015,
            starve_damage: 0.002,
            cost_per_dopamine: 0.4,
            heal_rate: 0.000_05,
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
}

impl CreatureGenome {
    /// Потомок генома: каждый ген дрейфует от родительского значения.
    /// Детерминированно от сида (один сид — один потомок).
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
        }
    }

    /// Эффект Болдуина: добавить инстинкты, свёрнутые из выученных путей
    /// родителя. Каждый дрейфует как обычный ген (тем же rng потомка) —
    /// приобретённый рефлекс становится наследственным, но не копируется
    /// бит-в-бит, а наследуется с изменчивостью.
    pub fn add_learned_instincts(
        &mut self,
        learned: Vec<Instinct>,
        rng: &mut XorShift,
    ) {
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
}
