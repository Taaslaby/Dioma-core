//! Programmable cell populations on the headless BioVM.
//!
//! This module is organism-agnostic: it runs cell programs and returns
//! [`BioEvent`]s. Applying events to tissue/blood lives in the body layer
//! (`body::cell_chemistry`).

use super::biovm::{
    BioEvent, BioEventKind, BioPopulation, BioProgram, Instruction, OutputBinding, SubstanceId,
};

mod state {
    pub const SIGNAL: usize = 0;
    pub const CALCIUM: usize = 1;
    pub const VESICLES: usize = 2;
    pub const SECRETION: usize = 3;
    pub const COUNT: usize = 4;
}

mod register {
    pub const SIGNAL: u8 = 0;
    pub const ACTIVATION: u8 = 1;
    pub const CALCIUM: u8 = 2;
    pub const VESICLES: u8 = 3;
    pub const RELEASE_GAIN: u8 = 4;
    pub const RELEASE: u8 = 5;
    pub const RECOVERY_RATE: u8 = 6;
    pub const RECOVERY: u8 = 7;
}

const CALCIUM_RESPONSE: f32 = 0.25;
const VESICLE_RECOVERY_RATE: f32 = 0.05;

/// Parameters of a homogeneous secretory cell population.
#[derive(Clone, Copy, Debug)]
pub struct SecretoryCellConfig {
    pub activation_threshold: f32,
    pub release_gain: f32,
    pub calcium_response: f32,
    pub vesicle_recovery_rate: f32,
    pub substance: SubstanceId,
}

/// A homogeneous population of programmable secretory cells.
///
/// This is the first vertical BioVM slice in dioma. The runtime owns local
/// cell state, while the caller applies returned events to an organism or
/// world chemical field.
pub struct BioCellRuntime {
    program: BioProgram,
    population: BioPopulation,
}

fn build_secretory_program(config: SecretoryCellConfig) -> BioProgram {
    use register as r;
    use state as s;

    let mut program = BioProgram::new(s::COUNT, 0);
    let instructions = [
        // Receptor activation: max(signal - threshold, 0).
        Instruction::LoadState {
            dst: r::SIGNAL,
            state: s::SIGNAL,
        },
        Instruction::Threshold {
            dst: r::ACTIVATION,
            source: r::SIGNAL,
            baseline: config.activation_threshold,
            gain: 1.0,
        },
        // Calcium follows receptor activation with a finite response time.
        Instruction::LoadState {
            dst: r::CALCIUM,
            state: s::CALCIUM,
        },
        Instruction::Relax {
            dst: r::CALCIUM,
            target: r::ACTIVATION,
            alpha: config.calcium_response,
        },
        Instruction::StoreState {
            state: s::CALCIUM,
            src: r::CALCIUM,
        },
        // Release is calcium-driven and cannot exceed the available pool.
        Instruction::LoadState {
            dst: r::VESICLES,
            state: s::VESICLES,
        },
        Instruction::Const {
            dst: r::RELEASE_GAIN,
            value: config.release_gain,
        },
        Instruction::Mul {
            dst: r::RELEASE,
            lhs: r::CALCIUM,
            rhs: r::RELEASE_GAIN,
        },
        Instruction::Mul {
            dst: r::RELEASE,
            lhs: r::RELEASE,
            rhs: r::VESICLES,
        },
        Instruction::Min {
            dst: r::RELEASE,
            lhs: r::RELEASE,
            rhs: r::VESICLES,
        },
        Instruction::StoreState {
            state: s::SECRETION,
            src: r::RELEASE,
        },
        // Empty vesicle capacity recovers while release depletes the pool.
        Instruction::Const {
            dst: r::RECOVERY_RATE,
            value: config.vesicle_recovery_rate,
        },
        Instruction::Const {
            dst: r::RECOVERY,
            value: 1.0,
        },
        Instruction::Sub {
            dst: r::RECOVERY,
            lhs: r::RECOVERY,
            rhs: r::VESICLES,
        },
        Instruction::Mul {
            dst: r::RECOVERY,
            lhs: r::RECOVERY,
            rhs: r::RECOVERY_RATE,
        },
        Instruction::Sub {
            dst: r::VESICLES,
            lhs: r::VESICLES,
            rhs: r::RELEASE,
        },
        Instruction::Add {
            dst: r::VESICLES,
            lhs: r::VESICLES,
            rhs: r::RECOVERY,
        },
        Instruction::Clamp01 { reg: r::VESICLES },
        Instruction::StoreState {
            state: s::VESICLES,
            src: r::VESICLES,
        },
    ];

    for instruction in instructions {
        program
            .push(instruction)
            .expect("secretory program uses a static validated layout");
    }
    program
        .bind_output(OutputBinding {
            state: s::SECRETION,
            kind: BioEventKind::Secrete,
            substance: Some(config.substance),
            threshold: 0.0,
            scale: 1.0,
        })
        .expect("secretion output uses a static validated layout");
    program
}

impl BioCellRuntime {
    pub fn secretory(n_cells: usize, threshold: f32, gain: f32, substance: SubstanceId) -> Self {
        Self::secretory_with(
            n_cells,
            SecretoryCellConfig {
                activation_threshold: threshold,
                release_gain: gain,
                calcium_response: CALCIUM_RESPONSE,
                vesicle_recovery_rate: VESICLE_RECOVERY_RATE,
                substance,
            },
        )
    }

    pub fn secretory_with(n_cells: usize, config: SecretoryCellConfig) -> Self {
        let program = build_secretory_program(config);

        let mut population = BioPopulation::new(&program, n_cells);
        population.state_mut(state::VESICLES).fill(1.0);
        Self {
            program,
            population,
        }
    }

    pub fn n_cells(&self) -> usize {
        self.population.n_cells()
    }

    pub fn set_signal(&mut self, signal: &[f32]) {
        self.population.set_input(state::SIGNAL, signal);
    }

    pub fn step(&mut self) -> Vec<BioEvent> {
        self.population.step_events(&self.program)
    }

    pub fn secretion(&self) -> &[f32] {
        self.population.state(state::SECRETION)
    }

    pub fn calcium(&self) -> &[f32] {
        self.population.state(state::CALCIUM)
    }

    pub fn vesicles(&self) -> &[f32] {
        self.population.state(state::VESICLES)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secretory_cells_emit_only_above_threshold() {
        let mut cells = BioCellRuntime::secretory(4, 0.5, 2.0, SubstanceId(1));
        cells.set_signal(&[0.2, 0.5, 0.8, 1.0]);

        let events = cells.step();

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].cell, 2);
        assert_eq!(events[0].kind, BioEventKind::Secrete);
        assert!(events[0].value > 0.0);
        assert!(events[1].value > events[0].value);
        assert!(cells.vesicles()[2] < 1.0);
    }

    #[test]
    fn vesicles_recover_after_release() {
        let mut cells = BioCellRuntime::secretory(1, 0.1, 1.0, SubstanceId(1));
        cells.set_signal(&[1.0]);
        let first = cells.step()[0].value;
        let depleted = cells.vesicles()[0];
        cells.set_signal(&[0.0]);
        for _ in 0..200 {
            cells.step();
        }

        assert!(first > 0.0);
        assert!(depleted < 1.0);
        assert!(cells.vesicles()[0] > depleted);
    }
}
