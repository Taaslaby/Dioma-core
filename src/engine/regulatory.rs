//! -------------------------------------------------------------------------
//! Filename:    src/engine/regulatory.rs
//! Class:       neuworks
//! Purpose:     регуляторные гены — дозирование экспрессии без изменения кода
//! Description: RegGenome не кодирует структуру (её растит grow) и не
//!              программирует динамику (это RuleGene). Он МОДУЛИРУЕТ:
//!              гейн пластичности по регионам, силу инстинктов, скорость
//!              тропической перестройки, пороги родительского доминирования.
//!              Гены взаимодействуют (эпистаз): итоговый множитель = база ×
//!              произведение активных модуляторов. Наследуется с мутацией;
//!              у каждого гена бит доминантности для полового кроссовера.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use crate::rng::XorShift;

/// Что модулирует регуляторный ген.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Множитель lr пластичности региона (0..2, 1 = нейтрально).
    Plasticity,
    /// Множитель силы инстинктов при прожиге (0..2).
    Instinct,
    /// Множитель числа переносов тропической перестройки за вызов (0..2).
    Rewire,
    /// Порог сытости: во сколько раз дольше сытость держится (0..2).
    Satiety,
}

impl Target {
    pub const ALL: [Target; 4] = [
        Target::Plasticity,
        Target::Instinct,
        Target::Rewire,
        Target::Satiety,
    ];

    /// Число системных слотов (для сериализации).
    pub const N: usize = 4;

    pub fn idx(self) -> usize {
        match self {
            Target::Plasticity => 0,
            Target::Instinct => 1,
            Target::Rewire => 2,
            Target::Satiety => 3,
        }
    }
}

/// Один регуляторный ген: цель + регион + сила + доминантность.
///
/// `region` = usize::MAX означает «все регионы» (системный модулятор).
/// `on_tick`/`off_tick` — окно экспрессии по возрасту (в мировых тиках):
/// ген активен только когда `age >= on_tick && age < off_tick`.
/// Это даёт онтогенез бесплатно: юношеские гены пластичности гаснут,
/// взрослые гены консерватизма включаются. usize::MAX = «всегда».
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegGene {
    pub target: Target,
    /// Регион действия; usize::MAX — системный.
    pub region: usize,
    /// Множитель эффекта (1.0 = нейтрально). Клэмпится в [0..2].
    pub strength: f32,
    /// true — доминантный аллель: в кроссовере перекрывает рецессивного партнёра.
    pub dominant: bool,
    /// Тик рождения гена (возрастная экспрессия, см. [`RegGenome::expression_at`]).
    pub on_tick: usize,
    /// Тик выключения; usize::MAX — никогда.
    pub off_tick: usize,
}

impl Default for RegGene {
    fn default() -> Self {
        Self {
            target: Target::Plasticity,
            region: usize::MAX,
            strength: 1.0,
            dominant: false,
            on_tick: 0,
            off_tick: usize::MAX,
        }
    }
}

impl RegGene {
    /// Юношеский ген: активен первые `duration` тиков жизни.
    pub fn juvenile(target: Target, strength: f32, duration: usize) -> Self {
        Self {
            target,
            strength,
            off_tick: duration.max(1),
            ..Default::default()
        }
    }

    /// Взрослый ген: включается после `onset` тиков и остаётся навсегда.
    pub fn adult(target: Target, strength: f32, onset: usize) -> Self {
        Self {
            target,
            strength,
            on_tick: onset,
            ..Default::default()
        }
    }

    /// Активен ли ген в данном возрасте.
    pub fn active_at(&self, age: usize) -> bool {
        age >= self.on_tick && age < self.off_tick
    }

    /// Мутация: дрейф силы, изредка флип доминантности, цели и окна.
    pub fn mutate(&self, rng: &mut XorShift, sigma: f32) -> Self {
        let mut g = *self;
        g.strength = (g.strength + rng.gauss() as f32 * sigma).clamp(0.0, 2.0);
        if rng.unit() < 0.05 {
            g.dominant = !g.dominant;
        }
        if rng.unit() < 0.02 {
            // Редкий сдвиг цели: ген начинает регулировать другое.
            let all = Target::ALL;
            g.target = all[(rng.next_u64() as usize) % all.len()];
        }
        if rng.unit() < 0.04 {
            // Дрейф окна экспрессии: онтогенетический тайминг сам эволюционирует.
            let shift = rng.gauss() as f32 * sigma * 2000.0;
            let onset = (g.on_tick as f32 + shift).max(0.0) as usize;
            let offset = if g.off_tick == usize::MAX {
                usize::MAX
            } else {
                ((g.off_tick as f32 + shift).max(onset as f32 + 100.0)) as usize
            };
            g.on_tick = onset;
            g.off_tick = offset;
        }
        g
    }
}

/// Регуляторный геном: фиксированный набор слотов (по одному на цель),
/// плюс опциональные per-region уточнения. Детерминирован по сиду.
#[derive(Clone, Debug, PartialEq)]
pub struct RegGenome {
    /// Системные гены (region = usize::MAX), ровно по одному на Target.
    pub systemic: [RegGene; Target::N],
    /// Per-region уточнения (пустой = только системные).
    pub regional: Vec<RegGene>,
}

impl Default for RegGenome {
    fn default() -> Self {
        Self {
            systemic: std::array::from_fn(|i| RegGene {
                target: Target::ALL[i],
                ..Default::default()
            }),
            regional: Vec::new(),
        }
    }
}

impl RegGenome {
    /// Дикий тип: всё нейтрально.
    pub fn wildtype() -> Self {
        Self::default()
    }

    /// Итоговый множитель цели в регионе на данном возрасте: база(=1) ×
    /// системный × региональный, только активные в этом возрасте гены.
    /// Эпистаз: два гена 1.4 дают 1.96, а не среднее — сильные аллели усиливают
    /// друг друга нелинейно.
    pub fn expression_at(&self, target: Target, region: usize, age: usize) -> f32 {
        let mut m = 1.0f32;
        let sys = &self.systemic[target.idx()];
        if sys.active_at(age) {
            m *= sys.strength;
        }
        for g in &self.regional {
            if g.target == target
                && (g.region == region || g.region == usize::MAX)
                && g.active_at(age)
            {
                m *= g.strength;
            }
        }
        m.clamp(0.0, 4.0)
    }

    /// Множитель без учёта возраста (возраст = 0, все окна открыты).
    pub fn expression(&self, target: Target, region: usize) -> f32 {
        self.expression_at(target, region, 0)
    }

    /// Мутация всего генома.
    pub fn mutate(&self, rng: &mut XorShift, sigma: f32) -> Self {
        let mut g = self.clone();
        for slot in g.systemic.iter_mut() {
            *slot = slot.mutate(rng, sigma);
        }
        for gene in g.regional.iter_mut() {
            *gene = gene.mutate(rng, sigma);
        }
        // Редкая дупликация: системный ген получает региональную копию
        // (субфункционализация — копия дивергирует мутацией).
        if rng.unit() < 0.03 && g.regional.len() < 16 {
            let src = &self.systemic[(rng.next_u64() as usize) % Target::N];
            let mut copy = *src;
            copy.region = (rng.next_u64() as usize) % 8;
            copy.strength = (copy.strength + rng.gauss() as f32 * sigma).clamp(0.0, 2.0);
            g.regional.push(copy);
        }
        g
    }

    /// Кроссовер с доминантностью: на каждый слот побеждает доминантный
    /// аллель; если оба доминантны или оба рецессивны — берётся A/B чередованием.
    pub fn crossover(&self, other: &Self, seed: u64) -> Self {
        let mut child = Self::default();
        for i in 0..Target::N {
            let (a, b) = (&self.systemic[i], &other.systemic[i]);
            child.systemic[i] = match (a.dominant, b.dominant) {
                (true, false) => *a,
                (false, true) => *b,
                _ => {
                    // Тай-брейк по сиду, как в merge_regional: при равной
                    // доминантности выбор А/В не привязан к чётности таргета
                    // (иначе чётные таргеты всегда наследовались бы от А,
                    // нечётные — от В, по всей популяции).
                    if (seed >> (i % 32)) & 1 == 0 {
                        *a
                    } else {
                        *b
                    }
                }
            };
        }
        // Региональные: объединение, конфликты по (target, region) решает доминантность.
        child.regional = merge_regional(&self.regional, &other.regional, seed);
        child
    }
}

fn merge_regional(a: &[RegGene], b: &[RegGene], seed: u64) -> Vec<RegGene> {
    let mut out: Vec<RegGene> = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    for gene in b {
        match out
            .iter()
            .position(|g| g.target == gene.target && g.region == gene.region)
        {
            Some(i) => {
                let resident = out[i];
                out[i] = match (resident.dominant, gene.dominant) {
                    (true, false) => resident,
                    (false, true) => *gene,
                    _ => {
                        let pick_a = (seed >> (gene.region % 32)) & 1 == 0;
                        if pick_a {
                            resident
                        } else {
                            *gene
                        }
                    }
                };
            }
            None => out.push(*gene),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildtype_is_neutral() {
        let g = RegGenome::wildtype();
        for t in Target::ALL {
            assert!((g.expression(t, 0) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn epistasis_multiplies_not_averages() {
        let mut g = RegGenome::wildtype();
        g.systemic[Target::Plasticity.idx()].strength = 1.5;
        g.regional.push(RegGene {
            target: Target::Plasticity,
            region: 3,
            strength: 1.5,
            ..Default::default()
        });
        // Системный 1.5 × региональный 1.5 = 2.25 (не среднее 1.5).
        assert!((g.expression(Target::Plasticity, 3) - 2.25).abs() < 1e-5);
        // Другой регион — только системный.
        assert!((g.expression(Target::Plasticity, 0) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn crossover_tiebreak_uses_seed_not_target_parity() {
        let mut a = RegGenome::wildtype();
        let mut b = RegGenome::wildtype();
        // Оба родителя рецессивны, силы различаются на всех системных таргетах.
        for i in 0..Target::N {
            a.systemic[i].strength = 0.4;
            a.systemic[i].dominant = false;
            b.systemic[i].strength = 1.6;
            b.systemic[i].dominant = false;
        }
        let c0 = a.crossover(&b, 0);
        let c1 = a.crossover(&b, 1);
        let c0_again = a.crossover(&b, 0);
        assert_eq!(c0, c0_again, "кроссовер детерминирован по сиду");
        // Сиды различаются в бите 0 → выбор на таргете 0 (Plasticity) обязан
        // разойтись; раньше чётный индекс всегда брал родителя A.
        assert_ne!(
            c0.systemic[0].strength, c1.systemic[0].strength,
            "тай-брейк системных таргетов должен зависеть от сида"
        );
    }

    #[test]
    fn dominance_overrides_recessive() {
        let mut a = RegGenome::wildtype();
        let mut b = RegGenome::wildtype();
        a.systemic[0].strength = 0.5;
        a.systemic[0].dominant = false;
        b.systemic[0].strength = 1.8;
        b.systemic[0].dominant = true;
        let child = a.crossover(&b, 42);
        assert!(
            (child.systemic[0].strength - 1.8).abs() < 1e-6,
            "доминантный должен победить"
        );
    }

    #[test]
    fn juvenile_gene_expires_with_age() {
        let mut g = RegGenome::wildtype();
        g.regional
            .push(RegGene::juvenile(Target::Plasticity, 2.0, 1000));
        assert!((g.expression_at(Target::Plasticity, 0, 0) - 2.0).abs() < 1e-6);
        assert!((g.expression_at(Target::Plasticity, 0, 999) - 2.0).abs() < 1e-6);
        assert!(
            (g.expression_at(Target::Plasticity, 0, 1000) - 1.0).abs() < 1e-6,
            "юношеский ген должен погаснуть"
        );
    }

    #[test]
    fn adult_gene_switches_on() {
        let mut g = RegGenome::wildtype();
        g.regional.push(RegGene::adult(Target::Rewire, 0.3, 5000));
        assert!((g.expression_at(Target::Rewire, 0, 4999) - 1.0).abs() < 1e-6);
        assert!((g.expression_at(Target::Rewire, 0, 5000) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn mutation_stays_in_bounds() {
        let mut rng = XorShift::new(7);
        let g = RegGenome::wildtype()
            .mutate(&mut rng, 1.5)
            .mutate(&mut rng, 1.5);
        for slot in &g.systemic {
            assert!((0.0..=2.0).contains(&slot.strength));
        }
    }
}
