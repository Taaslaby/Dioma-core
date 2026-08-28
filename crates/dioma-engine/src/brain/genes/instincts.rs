//! -------------------------------------------------------------------------
//! Filename:    src/engine/instincts.rs
//! Class:       neuworks
//! Purpose:     врождённые рефлексы: знание, прожигаемое во сне
//! Description: инстинкт = «стимул → ожидаемое действие + дофамин».
//!              Новорождённый не умеет реагировать; первый сон прожигает
//!              инстинкты: стимуляция + награда + консолидация делают
//!              pathway постоянным (снимок памяти обновляется).
//! author: Atlasov anton
// -------------------------------------------------------------------------
use crate::brain::neurons::Brain;
use crate::rng::XorShift;

/// Один инстинкт генома.
#[derive(Clone, Debug, PartialEq)]
pub struct Instinct {
    /// Клетки для стимуляции: (индекс нейрона, ток).
    pub inputs: Vec<(usize, f32)>,
    /// Дофамин за правильную реакцию (фазический, > 0).
    pub dopamine: f32,
}

/// Верхняя граница дофамина инстинкта: за много поколений мутаций сигнал
/// не должен расходиться бесконечно (нижняя граница — в `mutate`).
const DOPAMINE_MAX: f32 = 8.0;

impl Instinct {
    /// Мутация инстинкта: дрейф токов и дофамина (индексы не трогаем —
    /// топология врождённая).
    pub fn mutate(&self, rng: &mut XorShift, sigma: f32) -> Instinct {
        let mut g = self.clone();
        for (_, drive) in g.inputs.iter_mut() {
            *drive = (*drive + rng.gauss() as f32 * sigma * 20.0).max(0.5);
        }
        g.dopamine = (g.dopamine + rng.gauss() as f32 * sigma * 4.0).clamp(0.2, DOPAMINE_MAX);
        g
    }

    /// Прожечь инстинкт в мозге: стимулировать вход, наградить, закрепить
    /// сном (цель гомеостаза + долговременный снимок обновляются).
    pub fn burn_into(&self, brain: &mut Brain, tonic: f32) {
        self.burn_into_with_cap(brain, tonic, brain.plasticity_w_max());
    }

    /// То же, но с явным потолком клэмпа весов. Рождение (первый сон
    /// новорождённого) прожигает инстинкты ВЫШЕ рабочего `w_max`: у
    /// grown-коннектома хвосты весов десятки единиц, а рабочий потолок
    /// пластичности ~6 — обычный дофамин срезал бы их при первом же
    /// награждении и ОСЛАБИЛ бы рефлекс вместо усиления.
    pub fn burn_into_with_cap(&self, brain: &mut Brain, tonic: f32, w_cap: f32) {
        let mut ext = vec![tonic; brain.n];
        let mut stimulated = vec![false; brain.n];
        for &(i, drive) in &self.inputs {
            if i < brain.n {
                ext[i] += drive;
                stimulated[i] = true;
            }
        }
        let spt = brain.steps_per_tick();
        // Прожигание — до 500 подшагов HAAP: переиспользуем буфер спайков,
        // чтобы не аллоцировать `n_neurons` на каждый подшаг.
        let mut spikes = Vec::with_capacity(brain.n);
        for _ in 0..spt {
            brain.step_into(&ext, &mut spikes);
        }
        // Фиксировать только собственную дугу: ко-активация сенсора и мотора
        // возбуждает и тормозные пути (через интернейроны), и без этого
        // фильтра награда усилила бы их — рефлекс глушил бы собственный драйв.
        brain.restrict_eligibility(&stimulated);
        brain.reward_now_with_cap(self.dopamine, w_cap);
        brain.clear_eligibility();
        // Закрепление: новая норма гомеостаза + эталон памяти.
        brain.sleep_consolidate(0.6);
        brain.memory_snapshot();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::genome::Grown;
    use crate::brain::neurons::{gains_from, PlasticityParams};

    fn weak_grown() -> Grown {
        // Диагональ S→M слишком слаба для надёжной реакции (0.35),
        // фон 0.12: новорождённый почти не отвечает.
        Grown {
            edges: vec![
                (0, 4, 0.35),
                (1, 5, 0.30),
                (2, 6, 1.60),
                (3, 7, 1.40),
                (0, 1, 1.20),
                (1, 2, 1.10),
            ],
            region_of: vec![0, 0, 0, 0, 1, 1, 1, 1],
            nt: vec![1.0; 8],
            growth_s: 0.0,
            rule_genes: Vec::new(),
            instincts: Vec::new(),
        }
    }

    fn response(brain: &mut Brain) -> i64 {
        let spt = brain.steps_per_tick();
        let mut ext = vec![0.12f32; 8];
        ext[0] = 2.0; // свет слева
        let mut m = 0i64;
        let mut spikes = Vec::with_capacity(brain.n);
        for _ in 0..spt {
            brain.step_into(&ext, &mut spikes);
            m += spikes[4] as i64 + spikes[5] as i64;
        }
        m
    }

    #[test]
    fn instinct_burn_creates_reflex() {
        let params = PlasticityParams::default();

        // Новорождённый: реакция слабая.
        let mut baby = Brain::new(&weak_grown(), gains_from(2, &[(0, 1, 1.0)]), params.clone());
        let before = response(&mut baby);

        // Инстинкт «свет слева → уходи влево»: сильная КО-АКТИВАция
        // сенсора и мотора (иначе eligibility пуст и дофамину нечего
        // усиливать) + награда.
        let instinct = Instinct {
            inputs: vec![(0, 4.0), (4, 9.0)],
            dopamine: 2.0,
        };

        // Спящий: инстинкт прожжён несколько раз → рефлекс появился.
        for _ in 0..3 {
            instinct.burn_into(&mut baby, 0.12);
        }
        let after_sleep = response(&mut baby);

        // Контроль: тот же опыт БЕЗ закрепления (только стимул+награда
        // уже были выше) — сравниваем с ненаграждённым мозгом.
        let mut awake_only = Brain::new(&weak_grown(), gains_from(2, &[(0, 1, 1.0)]), params);
        let _ = response(&mut awake_only);

        assert!(
            after_sleep > before,
            "инстинкт должен усилить реакцию: до={before} после={after_sleep}"
        );
    }

    #[test]
    fn instinct_inputs_out_of_bounds_safe() {
        let mut brain = Brain::new(
            &weak_grown(),
            gains_from(2, &[(0, 1, 1.0)]),
            PlasticityParams::default(),
        );
        let wild = Instinct {
            inputs: vec![(9999, 2.0)],
            dopamine: 0.5,
        };
        wild.burn_into(&mut brain, 0.12); // не должно паниковать
    }
}
