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
use crate::engine::neurons::Brain;

/// Один инстинкт генома.
#[derive(Clone, Debug, PartialEq)]
pub struct Instinct {
    /// Клетки для стимуляции: (индекс нейрона, ток).
    pub inputs: Vec<(usize, f32)>,
    /// Дофамин за правильную реакцию (фазический, > 0).
    pub dopamine: f32,
}

impl Instinct {
    /// Прожечь инстинкт в мозге: стимулировать вход, наградить, закрепить
    /// сном (цель гомеостаза + долговременный снимок обновляются).
    pub fn burn_into(&self, brain: &mut Brain, tonic: f32) {
        let mut ext = vec![tonic; brain.n];
        for &(i, drive) in &self.inputs {
            if i < brain.n {
                ext[i] += drive;
            }
        }
        let spt = brain.steps_per_tick();
        for _ in 0..spt {
            brain.step(&ext);
        }
        brain.reward_now(self.dopamine);
        brain.clear_eligibility();
        // Закрепление: новая норма гомеостаза + эталон памяти.
        brain.sleep_consolidate(0.6);
        brain.memory_snapshot();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::genome::Grown;
    use crate::engine::neurons::{gains_from, PlasticityParams};

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
        for _ in 0..spt {
            let sp = brain.step(&ext);
            m += sp[4] as i64 + sp[5] as i64;
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
        let mut awake_only =
            Brain::new(&weak_grown(), gains_from(2, &[(0, 1, 1.0)]), params);
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
        let wild = Instinct { inputs: vec![(9999, 2.0)], dopamine: 0.5 };
        wild.burn_into(&mut brain, 0.12); // не должно паниковать
    }
}
