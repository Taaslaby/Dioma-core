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
use crate::engine::grow::{stage_c_lite_layout, Steering};
use crate::engine::instincts::Instinct;
use crate::engine::rules::RuleGene;
use crate::rng::XorShift;

/// Потолок инстинктов в геноме: один и тот же, что у свёртки выученных путей
/// (`creature::LEARNED_TOP_N`) — кроссовер не должен плодить инстинкты без
/// меры, иначе геном растёт экспоненциально от поколения к поколению.
const CROSSOVER_INSTINCT_CAP: usize = super::LEARNED_TOP_N;

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

    /// Кроссовер двух геномов (половое размножение): ребёнок получает гены
    /// ОБОИХ родителей. Прецедент — `RegGenome::crossover`: per-gene
    /// разрешение, детерминированно от rng (который посеян сидом потомка —
    /// один сид → один потомок пары).
    ///
    /// Семантика по классам генов:
    /// - `steering` — по-параметровый выбор родителя (50/50 на параметр);
    /// - `rule_genes` — по-генный выбор с выравниванием по индексу; если
    ///   длины разные, лишние слоты длинного родителя наследуются с p=0.5
    ///   (иначе у длинного генома не было бы шанса передать хвост целиком);
    /// - `instincts` — ОБЪЕДИНЕНИЕ инстинктов обоих родителей: дубликаты по
    ///   паре (pre_region, post_region) схлопываются в копию с высшим
    ///   дофамином, итог обрезается до `LEARNED_TOP_N` сильнейших — у
    ///   потомка не может быть больше рефлексов, чем у одного родителя после
    ///   свёртки выученного;
    /// - `metabolism` — по-полевой выбор родителя (как steering);
    /// - `sex` — НЕ берётся от родителей: пол задаётся сидом потомка при
    ///   рождении (`Sex::from_seed(child_seed)`, правило C3 — «sex gene, so
    ///   must pick randomly!»; у нас случайность заменена хэшем сида). Здесь
    ///   остаётся пол `self` — фактическое значение перезапишет сборка.
    pub fn crossover(&self, other: &Self, rng: &mut XorShift) -> Self {
        Self {
            steering: crossover_steering(&self.steering, &other.steering, rng),
            rule_genes: crossover_rules(&self.rule_genes, &other.rule_genes, rng),
            instincts: crossover_instincts(&self.instincts, &other.instincts),
            metabolism: crossover_metabolism(&self.metabolism, &other.metabolism, rng),
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

/// По-параметровый выбор родителя (50/50 на параметр через rng).
fn crossover_steering(a: &Steering, b: &Steering, rng: &mut XorShift) -> Steering {
    let pick = |rng: &mut XorShift| rng.unit() < 0.5;
    let f = |x: f64, y: f64, rng: &mut XorShift| if pick(rng) { x } else { y };
    Steering {
        noise_weight: f(a.noise_weight, b.noise_weight, rng),
        gradient_weight: f(a.gradient_weight, b.gradient_weight, rng),
        old_dir_weight: f(a.old_dir_weight, b.old_dir_weight, rng),
        turn_rate_deg: f(a.turn_rate_deg, b.turn_rate_deg, rng),
        stop_concentration: f(a.stop_concentration, b.stop_concentration, rng),
        bump_max: f(a.bump_max, b.bump_max, rng),
        bump_sigma: f(a.bump_sigma, b.bump_sigma, rng),
        synapse_dist: f(a.synapse_dist, b.synapse_dist, rng),
        grow_steps: if pick(rng) { a.grow_steps } else { b.grow_steps },
    }
}

/// По-генный выбор правил с выравниванием по индексу; лишние слоты длинного
/// родителя наследуются с p=0.5 (иначе длина генома всегда сжималась бы к
/// короткому родителю, и хвост правил терялся бы систематически).
fn crossover_rules(a: &[RuleGene], b: &[RuleGene], rng: &mut XorShift) -> Vec<RuleGene> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        match (a.get(i), b.get(i)) {
            (Some(x), Some(y)) => out.push(if rng.unit() < 0.5 { *x } else { *y }),
            (Some(x), None) | (None, Some(x)) => {
                if rng.unit() < 0.5 {
                    out.push(*x);
                }
            }
            (None, None) => unreachable!("слот {i} есть у одного из родителей по построению"),
        }
    }
    out
}

/// Объединение инстинктов обоих родителей: конфликт по паре (pre, post)
/// решается в пользу копии с высшим дофамином, итог обрезается до
/// `LEARNED_TOP_N` сильнейших (защита от экспоненциального роста генома).
fn crossover_instincts(a: &[Instinct], b: &[Instinct]) -> Vec<Instinct> {
    let mut out: Vec<Instinct> = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    for gene in b {
        let key = instinct_pair_key(gene);
        match out.iter().position(|g| instinct_pair_key(g) == key) {
            Some(i) => {
                if gene.dopamine > out[i].dopamine {
                    out[i] = gene.clone();
                }
            }
            None => out.push(gene.clone()),
        }
    }
    out.sort_by(|x, y| y.dopamine.total_cmp(&x.dopamine));
    out.truncate(CROSSOVER_INSTINCT_CAP);
    out
}

/// По-полевой выбор генов метаболизма (как steering).
fn crossover_metabolism(a: &MetabolismGene, b: &MetabolismGene, rng: &mut XorShift) -> MetabolismGene {
    let pick = |rng: &mut XorShift| rng.unit() < 0.5;
    let f = |x: f32, y: f32, rng: &mut XorShift| if pick(rng) { x } else { y };
    MetabolismGene {
        cost_per_activity: f(a.cost_per_activity, b.cost_per_activity, rng),
        starve_damage: f(a.starve_damage, b.starve_damage, rng),
        cost_per_dopamine: f(a.cost_per_dopamine, b.cost_per_dopamine, rng),
        heal_rate: f(a.heal_rate, b.heal_rate, rng),
    }
}

/// Ключ инстинкта: пара (pre_region, post_region) рефлекса «сенсор → мотор».
/// Инстинкт строится как блок pre-нейронов, затем блок post-нейронов
/// (`creature::learned_instincts`), поэтому первый вход — начало pre-региона,
/// последний — конец post-региона. Регион нейрона — по фиксированной
/// раскладке `stage_c_lite_layout` (все существа движка собираются по ней;
/// ядро 0..6 не сдвинуто — инстинкты рефлексов валидны).
pub(crate) fn instinct_pair_key(i: &Instinct) -> (usize, usize) {
    let (Some(first), Some(last)) = (i.inputs.first(), i.inputs.last()) else {
        return (usize::MAX, usize::MAX);
    };
    (neuron_region(first.0), neuron_region(last.0))
}

/// Регион нейрона по раскладке `stage_c_lite_layout` (кумулятивные счётчики).
fn neuron_region(n: usize) -> usize {
    let layout = stage_c_lite_layout();
    let mut acc = 0usize;
    for (i, r) in layout.regions.iter().enumerate() {
        if n < acc + r.count {
            return i;
        }
        acc += r.count;
    }
    layout.regions.len()
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

    /// Кроссовер детерминирован по сиду rng: два вызова с тем же rng-сидом —
    /// один и тот же потомок (как `RegGenome::crossover` по seed).
    #[test]
    fn crossover_is_deterministic_per_seed() {
        let a = CreatureGenome::default();
        let mut b = CreatureGenome::default();
        b.steering.bump_max = 8.0;
        b.rule_genes = vec![RuleGene::spiker(0.4, 0.9, 0.5, 1.0)];
        let c0 = a.crossover(&b, &mut XorShift::new(77));
        let c1 = a.crossover(&b, &mut XorShift::new(77));
        assert_eq!(c0, c1, "один сид — один потомок кроссовера");
        // Разные сиды могут дать разных потомков (хотя бы по одному параметру
        // на большом наборе сидов — проверяем, что rng реально участвует).
        let different = (0..32)
            .filter(|&s| {
                a.crossover(&b, &mut XorShift::new(s))
                    != a.crossover(&b, &mut XorShift::new(s + 1))
            })
            .count();
        assert!(different > 0, "кроссовер должен зависеть от сида rng");
    }

    /// Кроссовер смешивает гены ОБОИХ родителей: у родителей различимы все
    /// классы генов — потомок содержит значения и от A, и от B.
    #[test]
    fn crossover_mixes_genes_from_both_parents() {
        let mut a = CreatureGenome::default();
        let mut b = CreatureGenome::default();
        // Отличимые значения по каждому классу.
        a.steering.bump_max = 2.0;
        b.steering.bump_max = 9.0;
        a.metabolism.heal_rate = 0.001;
        b.metabolism.heal_rate = 0.009;
        a.rule_genes = vec![
            RuleGene::spiker(0.1, 0.2, 0.3, 1.0),
            RuleGene::spiker(0.5, 0.6, 0.7, 2.0),
        ];
        b.rule_genes = vec![
            RuleGene::spiker(1.1, 1.2, 1.3, 3.0),
            RuleGene::spiker(1.5, 1.6, 1.7, 4.0),
        ];
        a.instincts = vec![Instinct {
            inputs: vec![(0, 4.0), (80, 8.0)], // SL→ML
            dopamine: 2.0,
        }];
        b.instincts = vec![Instinct {
            inputs: vec![(40, 4.0), (100, 8.0)], // WL→MR
            dopamine: 5.0,
        }];

        // Много сидов: для каждого класса проверяем, что встречаются и
        // «от A», и «от B» (по достаточно большому диапазону).
        let mut from_a_bump = false;
        let mut from_b_bump = false;
        let mut from_a_heal = false;
        let mut from_b_heal = false;
        let mut from_a_rule = false;
        let mut from_b_rule = false;
        let mut saw_a_instinct = false;
        let mut saw_b_instinct = false;
        for s in 0..64 {
            let c = a.crossover(&b, &mut XorShift::new(s));
            if (c.steering.bump_max - 2.0).abs() < 1e-9 {
                from_a_bump = true;
            }
            if (c.steering.bump_max - 9.0).abs() < 1e-9 {
                from_b_bump = true;
            }
            if (c.metabolism.heal_rate - 0.001).abs() < 1e-9 {
                from_a_heal = true;
            }
            if (c.metabolism.heal_rate - 0.009).abs() < 1e-9 {
                from_b_heal = true;
            }
            if c.rule_genes.iter().any(|g| g.p[0] < 0.5) {
                from_a_rule = true;
            }
            if c.rule_genes.iter().any(|g| g.p[0] > 1.0) {
                from_b_rule = true;
            }
            if c.instincts.iter().any(|i| i.dopamine < 3.0) {
                saw_a_instinct = true;
            }
            if c.instincts.iter().any(|i| i.dopamine > 4.0) {
                saw_b_instinct = true;
            }
        }
        assert!(from_a_bump && from_b_bump, "steering должен браться от обоих");
        assert!(from_a_heal && from_b_heal, "metabolism должен браться от обоих");
        assert!(from_a_rule && from_b_rule, "rule_genes должны браться от обоих");
        assert!(
            saw_a_instinct && saw_b_instinct,
            "instincts — объединение обоих родителей"
        );
        // Каждый потомок — ровно с сильнейшим инстинктом каждой пары: здесь
        // пары разные (SL→ML и WL→MR), объединение = 2 инстинкта.
        let c = a.crossover(&b, &mut XorShift::new(5));
        assert_eq!(c.instincts.len(), 2, "обе пары должны выжить в объединении");
        assert_eq!(c.sex, a.sex, "пол потомка не выбирается кроссовером");
    }

    /// Объединение инстинктов ограничено: сколько бы рефлексов ни несли
    /// родители (в т.ч. дубликаты одной пары), у потомка не больше
    /// `LEARNED_TOP_N`, и дубликат пары схлопывается в копию с высшим
    /// дофамином.
    #[test]
    fn crossover_caps_instinct_union() {
        let mk = |pre: usize, post: usize, dopamine: f32| Instinct {
            inputs: vec![(pre, 4.0), (post, 8.0)],
            dopamine,
        };
        let mut a = CreatureGenome::default();
        // 6 инстинктов: дубликаты пар SL→ML (2 шт) и WL→MR (2 шт) + ещё пары.
        a.instincts = vec![
            mk(0, 80, 1.0),   // SL→ML
            mk(0, 80, 7.0),   // SL→ML (сильнее — должна выжить копия)
            mk(40, 100, 2.0), // WL→MR
            mk(40, 100, 3.0), // WL→MR (сильнее)
            mk(20, 100, 4.0), // SR→MR
            mk(52, 80, 5.0),  // WR→ML
        ];
        let mut b = CreatureGenome::default();
        b.instincts = vec![
            mk(0, 80, 6.0),   // SL→ML (слабее 9.0 — должна проиграть)
            mk(20, 100, 8.0), // SR→MR (сильнее 4.0)
            mk(0, 80, 9.0),   // SL→ML — максимум пары
            mk(2, 100, 0.5),  // (2 = SL, 100 = MR): ключ (0,6) — отдельная пара
        ];
        let c = a.crossover(&b, &mut XorShift::new(42));
        assert!(
            c.instincts.len() <= crate::creature::LEARNED_TOP_N,
            "объединение обрезано до LEARNED_TOP_N: {}",
            c.instincts.len()
        );
        // Пара SL→ML схлопнулась в копию с высшим дофамином (9.0 от B).
        let sl_ml = c
            .instincts
            .iter()
            .find(|i| i.inputs[0].0 == 0 && i.inputs.last().unwrap().0 == 80)
            .expect("пара SL→ML должна остаться в объединении");
        assert!(
            (sl_ml.dopamine - 9.0).abs() < 1e-6,
            "дубликат пары должен взять высший дофамин: {}",
            sl_ml.dopamine
        );
        // Итог отсортирован по дофамину убыванию.
        for w in c.instincts.windows(2) {
            assert!(w[0].dopamine >= w[1].dopamine, "инстинкты по убыванию дофамина");
        }
    }
}
