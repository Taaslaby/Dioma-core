//! Neutral biological virtual machine facade.
//!
//! `BioVm` executes column-major state for a population of equivalent cells.
//! The facade deliberately has no neuron, spike, brain, or game concepts.
//! Existing TNMT opcodes remain the execution backend; this module gives the
//! generic subset a stable biological vocabulary.

use super::vm::Opcode;

/// Errors raised while compiling a biological program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BioVmError {
    StateOutOfRange(usize),
    ParameterOutOfRange(usize),
    RegisterOutOfRange(u8),
    TooManyRegisters,
    /// Snapshot was written by an incompatible format version.
    SnapshotVersionMismatch {
        expected: u16,
        found: u16,
    },
    /// Snapshot cell count differs from the target population.
    SnapshotCellCountMismatch {
        expected: usize,
        found: usize,
    },
}

/// Format version of [`BioPopulationSnapshot`]. Bump on any state-layout
/// change; `restore` refuses older snapshots instead of misreading bytes.
pub const BIOVM_STATE_VERSION: u16 = 1;

/// Serializable state of one homogeneous cell population.
///
/// Programs are not captured here — the host is responsible for knowing
/// which [`BioProgram`] a population runs (store a program id alongside).
#[derive(Clone, Debug, PartialEq)]
pub struct BioPopulationSnapshot {
    pub version: u16,
    pub n_cells: usize,
    pub states: Vec<f32>,
    pub params: Vec<f32>,
}

/// A request emitted by a cell program for the owning simulation.
/// The host decides whether and how to apply it to the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BioEventKind {
    Secrete,
    Transport,
    Divide,
    Differentiate,
    Contract,
}

/// Host-defined substance identifier. BioVM does not assign chemical meaning
/// to the number; the embedding simulation owns the mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubstanceId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BioEvent {
    pub cell: usize,
    pub kind: BioEventKind,
    pub substance: Option<SubstanceId>,
    pub value: f32,
}

/// Maps a state slot to an event channel when its value crosses `threshold`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputBinding {
    pub state: usize,
    pub kind: BioEventKind,
    pub substance: Option<SubstanceId>,
    pub threshold: f32,
    pub scale: f32,
}

/// A neutral instruction for local cell biology.
///
/// Registers `14` and `15` are reserved as compiler scratch registers by
/// compound instructions such as `Relax` and `Threshold`.
#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    LoadState {
        dst: u8,
        state: usize,
    },
    StoreState {
        state: usize,
        src: u8,
    },
    LoadParam {
        dst: u8,
        param: usize,
    },
    Const {
        dst: u8,
        value: f32,
    },
    Add {
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    Sub {
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    Mul {
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    Exp {
        dst: u8,
        src: u8,
    },
    Clamp01 {
        reg: u8,
    },
    Max {
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    Min {
        dst: u8,
        lhs: u8,
        rhs: u8,
    },
    /// `dst += (target - dst) * alpha`.
    Relax {
        dst: u8,
        target: u8,
        alpha: f32,
    },
    /// `dst = max(0, source - baseline) * gain`.
    Threshold {
        dst: u8,
        source: u8,
        baseline: f32,
        gain: f32,
    },
    /// `dst = conductance * (potential - reversal)`.
    OhmicFlow {
        dst: u8,
        conductance: u8,
        potential: u8,
        reversal: u8,
    },
    /// `concentration = baseline + (concentration - baseline) * exp(-dt / tau)`.
    DiffusionDecay {
        concentration: u8,
        baseline: f32,
        tau: f32,
        dt: f32,
    },
}

/// Compiled straight-line biological program.
#[derive(Clone, Debug)]
pub struct BioProgram {
    state_count: usize,
    parameter_count: usize,
    instructions: Vec<Opcode>,
    outputs: Vec<OutputBinding>,
}

/// Owned state for one homogeneous population of biological cells.
///
/// State and parameter storage is column-major: all cells for one slot are
/// contiguous. Inputs are copied into state slots before execution, which
/// keeps the VM executor independent from any particular world or transport
/// system.
#[derive(Clone, Debug)]
pub struct BioPopulation {
    n_cells: usize,
    states: Vec<f32>,
    params: Vec<f32>,
}

impl BioPopulation {
    pub fn new(program: &BioProgram, n_cells: usize) -> Self {
        Self {
            n_cells,
            states: vec![0.0; program.state_count * n_cells],
            params: vec![0.0; program.parameter_count * n_cells],
        }
    }

    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    pub fn state(&self, slot: usize) -> &[f32] {
        self.column(&self.states, slot)
    }

    pub fn state_mut(&mut self, slot: usize) -> &mut [f32] {
        let start = slot * self.n_cells;
        &mut self.states[start..start + self.n_cells]
    }

    pub fn parameter_mut(&mut self, slot: usize) -> &mut [f32] {
        let start = slot * self.n_cells;
        &mut self.params[start..start + self.n_cells]
    }

    pub fn set_input(&mut self, state_slot: usize, input: &[f32]) {
        assert_eq!(input.len(), self.n_cells);
        self.state_mut(state_slot).copy_from_slice(input);
    }

    pub fn step(&mut self, program: &BioProgram) {
        assert_eq!(program.state_count * self.n_cells, self.states.len());
        assert_eq!(program.parameter_count * self.n_cells, self.params.len());
        program.execute(&mut self.states, &self.params, self.n_cells);
    }

    /// Run one local biology step and translate bound outputs into events.
    pub fn step_events(&mut self, program: &BioProgram) -> Vec<BioEvent> {
        self.step(program);
        let mut events = Vec::new();
        for &binding in program.outputs() {
            for (cell, &raw) in self.state(binding.state).iter().enumerate() {
                if raw > binding.threshold {
                    events.push(BioEvent {
                        cell,
                        kind: binding.kind,
                        substance: binding.substance,
                        value: raw * binding.scale,
                    });
                }
            }
        }
        events
    }

    fn column<'a>(&self, storage: &'a [f32], slot: usize) -> &'a [f32] {
        let start = slot * self.n_cells;
        &storage[start..start + self.n_cells]
    }

    /// Capture the full cell state for saving.
    pub fn snapshot(&self) -> BioPopulationSnapshot {
        BioPopulationSnapshot {
            version: BIOVM_STATE_VERSION,
            n_cells: self.n_cells,
            states: self.states.clone(),
            params: self.params.clone(),
        }
    }

    /// Restore previously saved state, validating format compatibility.
    pub fn restore(&mut self, snapshot: &BioPopulationSnapshot) -> Result<(), BioVmError> {
        if snapshot.version != BIOVM_STATE_VERSION {
            return Err(BioVmError::SnapshotVersionMismatch {
                expected: BIOVM_STATE_VERSION,
                found: snapshot.version,
            });
        }
        if snapshot.n_cells != self.n_cells {
            return Err(BioVmError::SnapshotCellCountMismatch {
                expected: self.n_cells,
                found: snapshot.n_cells,
            });
        }
        let expected_len = self.states.len() + self.params.len();
        let found_len = snapshot.states.len() + snapshot.params.len();
        if found_len != expected_len {
            return Err(BioVmError::SnapshotCellCountMismatch {
                expected: expected_len,
                found: found_len,
            });
        }
        self.states = snapshot.states.clone();
        self.params = snapshot.params.clone();
        Ok(())
    }
}

impl BioProgram {
    pub fn new(state_count: usize, parameter_count: usize) -> Self {
        Self {
            state_count,
            parameter_count,
            instructions: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn state_count(&self) -> usize {
        self.state_count
    }

    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub fn instructions(&self) -> &[Opcode] {
        &self.instructions
    }

    pub fn outputs(&self) -> &[OutputBinding] {
        &self.outputs
    }

    pub fn bind_output(&mut self, binding: OutputBinding) -> Result<(), BioVmError> {
        if binding.state >= self.state_count {
            return Err(BioVmError::StateOutOfRange(binding.state));
        }
        self.outputs.push(binding);
        Ok(())
    }

    pub fn push(&mut self, instruction: Instruction) -> Result<(), BioVmError> {
        self.validate(instruction)?;
        let mut compiled = Vec::new();
        match instruction {
            Instruction::LoadState { dst, state } => compiled.push(Opcode::LoadState(dst, state)),
            Instruction::StoreState { state, src } => compiled.push(Opcode::StoreState(state, src)),
            Instruction::LoadParam { dst, param } => compiled.push(Opcode::LoadParam(dst, param)),
            Instruction::Const { dst, value } => compiled.push(Opcode::LoadConst(dst, value)),
            Instruction::Add { dst, lhs, rhs } => compiled.push(Opcode::MathAdd(dst, lhs, rhs)),
            Instruction::Sub { dst, lhs, rhs } => compiled.push(Opcode::MathSub(dst, lhs, rhs)),
            Instruction::Mul { dst, lhs, rhs } => compiled.push(Opcode::MathMul(dst, lhs, rhs)),
            Instruction::Exp { dst, src } => compiled.push(Opcode::MathFastExp(dst, src)),
            Instruction::Clamp01 { reg } => compiled.push(Opcode::MathClamp01(reg)),
            Instruction::Max { dst, lhs, rhs } => compiled.push(Opcode::MathMax(dst, lhs, rhs)),
            Instruction::Min { dst, lhs, rhs } => compiled.push(Opcode::MathMin(dst, lhs, rhs)),
            Instruction::Relax { dst, target, alpha } => {
                compiled.extend([
                    Opcode::MathSub(14, target, dst),
                    Opcode::LoadConst(15, alpha),
                    Opcode::MathMul(14, 14, 15),
                    Opcode::MathAdd(dst, dst, 14),
                ]);
            }
            Instruction::Threshold {
                dst,
                source,
                baseline,
                gain,
            } => {
                compiled.extend([
                    Opcode::LoadConst(14, baseline),
                    Opcode::MathSub(dst, source, 14),
                    Opcode::LoadConst(15, 0.0),
                    Opcode::MathMax(dst, dst, 15),
                    Opcode::LoadConst(15, gain),
                    Opcode::MathMul(dst, dst, 15),
                ]);
            }
            Instruction::OhmicFlow {
                dst,
                conductance,
                potential,
                reversal,
            } => {
                compiled.push(Opcode::PhysOhm(dst, conductance, potential, reversal));
            }
            Instruction::DiffusionDecay {
                concentration,
                baseline,
                tau,
                dt,
            } => {
                compiled.push(Opcode::PhysDiffusionDecay(concentration, baseline, tau, dt));
            }
        }
        self.instructions.extend(compiled);
        Ok(())
    }

    fn validate(&self, instruction: Instruction) -> Result<(), BioVmError> {
        let state_check = |index: usize| {
            if index < self.state_count {
                Ok(())
            } else {
                Err(BioVmError::StateOutOfRange(index))
            }
        };
        let param = |index: usize| {
            if index < self.parameter_count {
                Ok(())
            } else {
                Err(BioVmError::ParameterOutOfRange(index))
            }
        };
        let reg = |index: u8| {
            if index < 16 {
                Ok(())
            } else {
                Err(BioVmError::RegisterOutOfRange(index))
            }
        };
        match instruction {
            Instruction::LoadState { dst, state } => {
                reg(dst)?;
                state_check(state)
            }
            Instruction::StoreState { state, src } => {
                state_check(state)?;
                reg(src)
            }
            Instruction::LoadParam { dst, param: index } => {
                reg(dst)?;
                param(index)
            }
            Instruction::Const { dst, .. } | Instruction::Clamp01 { reg: dst } => reg(dst),
            Instruction::Add { dst, lhs, rhs }
            | Instruction::Sub { dst, lhs, rhs }
            | Instruction::Mul { dst, lhs, rhs }
            | Instruction::Max { dst, lhs, rhs }
            | Instruction::Min { dst, lhs, rhs } => {
                reg(dst)?;
                reg(lhs)?;
                reg(rhs)
            }
            Instruction::Exp { dst, src } => {
                reg(dst)?;
                reg(src)
            }
            Instruction::Relax { dst, target, .. } => {
                reg(dst)?;
                reg(target)
            }
            Instruction::Threshold { dst, source, .. } => {
                reg(dst)?;
                reg(source)
            }
            Instruction::OhmicFlow {
                dst,
                conductance,
                potential,
                reversal,
            } => {
                reg(dst)?;
                reg(conductance)?;
                reg(potential)?;
                reg(reversal)
            }
            Instruction::DiffusionDecay { concentration, .. } => reg(concentration),
        }
    }

    /// Execute the program over a column-major population.
    ///
    /// `states` must have `state_count * n_cells` values and `params`, when
    /// used, must have `parameter_count * n_cells` values.
    pub fn execute(&self, states: &mut [f32], params: &[f32], n_cells: usize) {
        assert!(states.len() >= self.state_count * n_cells);
        assert!(params.len() >= self.parameter_count * n_cells);
        unsafe {
            super::vm::execute(
                states.as_mut_ptr(),
                params.as_ptr(),
                n_cells,
                &self.instructions,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_cell_program_updates_state() {
        let mut program = BioProgram::new(2, 0);
        program
            .push(Instruction::LoadState { dst: 0, state: 0 })
            .unwrap();
        program
            .push(Instruction::LoadState { dst: 1, state: 1 })
            .unwrap();
        program
            .push(Instruction::Relax {
                dst: 0,
                target: 1,
                alpha: 0.25,
            })
            .unwrap();
        program
            .push(Instruction::StoreState { state: 0, src: 0 })
            .unwrap();

        let n = 9;
        let mut states = vec![0.0; 2 * n];
        states[n..].fill(1.0);
        program.execute(&mut states, &[], n);

        assert!(states[0] > 0.24 && states[0] < 0.26);
        assert_eq!(states[n], 1.0);
    }

    #[test]
    fn compiler_rejects_invalid_state_and_parameter() {
        let mut program = BioProgram::new(1, 1);
        assert_eq!(
            program.push(Instruction::LoadState { dst: 0, state: 1 }),
            Err(BioVmError::StateOutOfRange(1))
        );
        assert_eq!(
            program.push(Instruction::LoadParam { dst: 0, param: 1 }),
            Err(BioVmError::ParameterOutOfRange(1))
        );
    }

    #[test]
    fn population_applies_input_and_diffusion_program() {
        let mut program = BioProgram::new(2, 0);
        program
            .push(Instruction::LoadState { dst: 0, state: 0 })
            .unwrap();
        program
            .push(Instruction::DiffusionDecay {
                concentration: 0,
                baseline: 0.0,
                tau: 10.0,
                dt: 1.0,
            })
            .unwrap();
        program
            .push(Instruction::StoreState { state: 0, src: 0 })
            .unwrap();

        let mut cells = BioPopulation::new(&program, 9);
        cells.set_input(0, &[1.0; 9]);
        cells.step(&program);

        let expected = (-0.1f32).exp();
        assert!(cells
            .state(0)
            .iter()
            .all(|&value| (value - expected).abs() < 1e-4));
    }

    #[test]
    fn population_emits_host_events_from_bound_outputs() {
        let mut program = BioProgram::new(1, 0);
        program
            .push(Instruction::LoadState { dst: 0, state: 0 })
            .unwrap();
        program
            .push(Instruction::StoreState { state: 0, src: 0 })
            .unwrap();
        program
            .bind_output(OutputBinding {
                state: 0,
                kind: BioEventKind::Secrete,
                substance: Some(SubstanceId(7)),
                threshold: 0.5,
                scale: 2.0,
            })
            .unwrap();

        let mut cells = BioPopulation::new(&program, 9);
        cells.set_input(0, &[0.0, 0.8, 0.2, 1.0, 0.0, 0.0, 0.6, 0.1, 0.0]);
        let events = cells.step_events(&program);

        assert_eq!(events.len(), 3);
        assert_eq!(
            events[0],
            BioEvent {
                cell: 1,
                kind: BioEventKind::Secrete,
                substance: Some(SubstanceId(7)),
                value: 1.6,
            }
        );
        assert_eq!(events[1].cell, 3);
        assert_eq!(events[2].cell, 6);
    }

    #[test]
    fn snapshot_restore_roundtrip_and_version_guard() {
        let mut program = BioProgram::new(2, 0);
        program
            .push(Instruction::LoadState { dst: 0, state: 0 })
            .unwrap();
        program
            .push(Instruction::StoreState { state: 0, src: 0 })
            .unwrap();

        let mut cells = BioPopulation::new(&program, 4);
        cells.set_input(0, &[1.0, 2.0, 3.0, 4.0]);

        let snapshot = cells.snapshot();
        assert_eq!(snapshot.version, BIOVM_STATE_VERSION);

        // Wipe, then restore: state must come back exactly.
        let mut fresh = BioPopulation::new(&program, 4);
        fresh.restore(&snapshot).unwrap();
        assert_eq!(fresh.state(0), cells.state(0));

        // Incompatible versions are rejected, not misread.
        let mut stale = snapshot.clone();
        stale.version = BIOVM_STATE_VERSION + 1;
        assert_eq!(
            fresh.restore(&stale),
            Err(BioVmError::SnapshotVersionMismatch {
                expected: BIOVM_STATE_VERSION,
                found: BIOVM_STATE_VERSION + 1,
            })
        );

        // Wrong population size is rejected.
        let mut other = BioPopulation::new(&program, 8);
        assert_eq!(
            other.restore(&snapshot),
            Err(BioVmError::SnapshotCellCountMismatch {
                expected: 8,
                found: 4,
            })
        );
    }
}
