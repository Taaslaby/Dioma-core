//! -------------------------------------------------------------------------
//! Filename:    src/engine/behavior.rs
//! Class:       neuworks
//! Purpose:     гены поведенческих программ — врождённая «прошивка» поведения
//! Description: существо исполняет не хардкод, а программу из инструкций над
//!              слотами: сенсоры → вычисления → действия (курс, драйв,
//!              прыжок, еда). Программа линейна (как BioVM) — предикатные
//!              записи IfAct без циклов и goto дают детерминизм и .taas-
//!              совместимость. Программы наследуются и мутируют: поведение
//!              эволюционирует, как гены правил и регуляторный геном.
//!
//! Слои: engine знает только ВМ и слоты; сенсоры пишет «тело/мир»
//! (организм или скриптовый хост) перед каждым тиком; действия читают после
//! execute().
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------

use crate::vm::vm::Opcode as T;
use crate::rng::XorShift;

/// Количество регистров ВМ (совпадает с logic::vm).
pub const N_REG: usize = 16;
/// Регистры 14/15 — кэш компилятора (Relax/Threshold); общие — 0..13.
const SCRATCH: u8 = 14;
const SCRATCH2: u8 = 15;

/// Слот-контракт программы: СЕНСОРЫ и ДЕЙСТВИЯ (фиксированные номера).
///
/// Слой, который исполняет программу (организм или скриптовый хост),
/// пишет сенсоры → execute() → читает действия.
pub mod slots {
    // ── Сенсоры (0..8) — пишет мир/тело ──
    /// Голод [0..1] (нормированный из крови).
    pub const HUNGER: usize = 0;
    /// Глюкоза [0..1] (нормированная).
    pub const GLUCOSE: usize = 1;
    /// Запах еды слева [0..1].
    pub const SMELL_L: usize = 2;
    /// Запах еды справа [0..1].
    pub const SMELL_R: usize = 3;
    /// Касание стены/края [0..1].
    pub const TOUCH: usize = 4;
    /// Одиночество [0..1].
    pub const LONELY: usize = 5;
    /// Страх [0..1].
    pub const FEAR: usize = 6;
    /// Темперамент [0..2] (личность из наследия).
    pub const TEMPO: usize = 7;
    /// Сколько сенсоров.
    pub const N_SENSORS: usize = 8;

    // ── Действия (8..12) — читает мир ──
    /// Курс [−1..1]: знак — направление.
    pub const STEER: usize = 8;
    /// Мощность движения [0..1].
    pub const DRIVE: usize = 9;
    /// Прыжок [0..1].
    pub const JUMP: usize = 10;
    /// Есть [0..1].
    pub const EAT: usize = 11;
    /// Сколько действий.
    pub const N_ACTIONS: usize = 4;
    /// Всего слотов состояния.
    pub const N_STATE: usize = 12;
}

fn slot_ok(state: usize) -> bool {
    state < slots::N_STATE
}

/// Инструкция поведенческой программы (доменный набор).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BehaviorOp {
    /// Читает слот в регистр.
    Sensor {
        dst: u8,
        state: usize,
    },
    /// Безусловная запись в слот (действие).
    Action {
        state: usize,
        src: u8,
    },
    /// Предикат: если регистр cond > 0, записать src в слот.
    IfAct {
        cond: u8,
        state: usize,
        src: u8,
    },
    Const {
        dst: u8,
        value: f32,
    },
    Add {
        dst: u8,
        a: u8,
        b: u8,
    },
    Sub {
        dst: u8,
        a: u8,
        b: u8,
    },
    Mul {
        dst: u8,
        a: u8,
        b: u8,
    },
    Max {
        dst: u8,
        a: u8,
        b: u8,
    },
    Min {
        dst: u8,
        a: u8,
        b: u8,
    },
    Clamp01 {
        reg: u8,
    },
    /// Экспоненциальное затухание/насыщение: exp(src).
    Exp {
        dst: u8,
        src: u8,
    },
    /// dst += (target − dst) · alpha — плавное притяжение.
    Relax {
        dst: u8,
        target: u8,
        alpha: f32,
    },
    /// dst = max(0, src − baseline) · gain.
    Threshold {
        dst: u8,
        src: u8,
        baseline: f32,
        gain: f32,
    },
}

/// Параметры мутаций поведенческой программы.
#[derive(Clone, Copy, Debug)]
pub struct BehaviorMutate {
    /// Вероятность вставить случайную инструкцию.
    pub insert: f64,
    /// Вероятность удалить инструкцию.
    pub delete: f64,
    /// Вероятность заменить инструкцию на случайную.
    pub replace: f64,
    /// Вероятность дрейфа констант за вызов.
    pub const_ops: f64,
    /// Сигма дрейфа констант.
    pub sigma: f32,
    /// Максимальная длина программы.
    pub max_len: usize,
}

impl Default for BehaviorMutate {
    fn default() -> Self {
        Self {
            insert: 0.05,
            delete: 0.03,
            replace: 0.10,
            const_ops: 0.7,
            sigma: 0.15,
            max_len: 128,
        }
    }
}

/// Валидация: регистры только общие (0..13), слоты в диапазоне.
fn validate(ops: &[BehaviorOp]) -> Result<(), String> {
    let reg_ok = |r: u8| -> Result<(), String> {
        if r < 14 {
            Ok(())
        } else {
            Err(format!("регистр {r} >= 14 — служебный"))
        }
    };
    for op in ops {
        match op {
            BehaviorOp::Sensor { dst, state } => {
                reg_ok(*dst)?;
                if !slot_ok(*state) {
                    return Err(format!("слот {state} вне диапазона"));
                }
            }
            BehaviorOp::Action { state, src } => {
                reg_ok(*src)?;
                if !slot_ok(*state) {
                    return Err(format!("слот {state} вне диапазона"));
                }
            }
            BehaviorOp::IfAct { cond, state, src } => {
                reg_ok(*cond)?;
                reg_ok(*src)?;
                if !slot_ok(*state) {
                    return Err(format!("слот {state} вне диапазона"));
                }
            }
            BehaviorOp::Const { dst, .. } => reg_ok(*dst)?,
            BehaviorOp::Add { dst, a, b }
            | BehaviorOp::Sub { dst, a, b }
            | BehaviorOp::Mul { dst, a, b }
            | BehaviorOp::Max { dst, a, b }
            | BehaviorOp::Min { dst, a, b } => {
                reg_ok(*dst)?;
                reg_ok(*a)?;
                reg_ok(*b)?;
            }
            BehaviorOp::Clamp01 { reg } => reg_ok(*reg)?,
            BehaviorOp::Exp { dst, src } => {
                reg_ok(*dst)?;
                reg_ok(*src)?;
            }
            BehaviorOp::Relax { dst, target, .. } => {
                reg_ok(*dst)?;
                reg_ok(*target)?;
            }
            BehaviorOp::Threshold { dst, src, .. } => {
                reg_ok(*dst)?;
                reg_ok(*src)?;
            }
        }
    }
    Ok(())
}

/// Компиляция инструкции → байткод ВМ (линейный, без переходов).
fn compile_op(op: BehaviorOp) -> Result<Vec<T>, String> {
    let out = match op {
        BehaviorOp::Sensor { dst, state } => vec![T::LoadState(dst, state)],
        BehaviorOp::Action { state, src } => vec![T::StoreState(state, src)],
        BehaviorOp::IfAct { cond, state, src } => {
            vec![T::PredOn(cond), T::StoreState(state, src), T::PredOff]
        }
        BehaviorOp::Const { dst, value } => vec![T::LoadConst(dst, value)],
        BehaviorOp::Add { dst, a, b } => vec![T::MathAdd(dst, a, b)],
        BehaviorOp::Sub { dst, a, b } => vec![T::MathSub(dst, a, b)],
        BehaviorOp::Mul { dst, a, b } => vec![T::MathMul(dst, a, b)],
        BehaviorOp::Max { dst, a, b } => vec![T::MathMax(dst, a, b)],
        BehaviorOp::Min { dst, a, b } => vec![T::MathMin(dst, a, b)],
        BehaviorOp::Clamp01 { reg } => vec![T::MathClamp01(reg)],
        BehaviorOp::Exp { dst, src } => vec![T::MathFastExp(dst, src)],
        BehaviorOp::Relax { dst, target, alpha } => vec![
            T::MathSub(SCRATCH, target, dst),
            T::LoadConst(SCRATCH2, alpha),
            T::MathMul(SCRATCH, SCRATCH, SCRATCH2),
            T::MathAdd(dst, dst, SCRATCH),
        ],
        BehaviorOp::Threshold {
            dst,
            src,
            baseline,
            gain,
        } => vec![
            T::LoadConst(SCRATCH, baseline),
            T::MathSub(dst, src, SCRATCH),
            T::LoadConst(SCRATCH2, 0.0),
            T::MathMax(dst, dst, SCRATCH2),
            T::LoadConst(SCRATCH2, gain),
            T::MathMul(dst, dst, SCRATCH2),
        ],
    };
    Ok(out)
}

/// Поведенческая программа: инструкции + скомпилированный байткод.
#[derive(Clone, Debug)]
pub struct BehaviorProgram {
    ops: Vec<BehaviorOp>,
    bytecode: Vec<T>,
}

impl PartialEq for BehaviorProgram {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops
    }
}

impl BehaviorProgram {
    /// Собрать из инструкций (валидация + компиляция).
    pub fn from_ops(ops: Vec<BehaviorOp>) -> Result<Self, String> {
        validate(&ops)?;
        let mut bytecode = Vec::new();
        for &op in &ops {
            bytecode.extend(compile_op(op)?);
        }
        Ok(Self { ops, bytecode })
    }

    /// Прогон над популяцией из `n` особей (column-major).
    ///
    /// `states` — размер ≥ `slots::N_STATE · n`; сенсоры должны быть
    /// заполнены вызывающим слоем до вызова.
    pub fn execute(&self, states: &mut [f32], n: usize) {
        assert!(states.len() >= slots::N_STATE * n, "буфер состояния мал");
        unsafe { crate::vm::vm::execute(states.as_mut_ptr(), core::ptr::null(), n, &self.bytecode) }
    }

    /// Возвращает исходные операции программы для инспекции и мутаций.
    pub fn ops(&self) -> &[BehaviorOp] {
        &self.ops
    }
    /// Число инструкций в программе.
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    /// Проверяет, является ли программа пустой.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Сериализация в компактный массив 32-битных слов (для TAAS и наследства).
    pub fn to_u32(&self) -> Vec<u32> {
        pack_all(&self.ops)
    }

    /// Восстанавливает программу из компактного представления `u32`.
    /// Возвращает ошибку, если поток содержит неизвестную операцию или
    /// обрывается посреди инструкции.
    pub fn from_u32(words: &[u32]) -> Result<Self, String> {
        let ops = unpack_all(words)?;
        Self::from_ops(ops)
    }

    /// Мутация программы: вставка/удаление/замена инструкций + дрейф констант.
    pub fn mutate(&self, rng: &mut XorShift, p: &BehaviorMutate) -> Self {
        let mut ops: Vec<BehaviorOp> = self.ops.clone();
        if ops.len() < p.max_len && rng.unit() < p.insert {
            let i = (rng.next_u64() as usize) % (ops.len() + 1);
            ops.insert(i, random_op(rng));
        }
        if ops.len() > 2 && rng.unit() < p.delete {
            let i = (rng.next_u64() as usize) % ops.len();
            ops.remove(i);
        }
        if !ops.is_empty() && rng.unit() < p.replace {
            let i = (rng.next_u64() as usize) % ops.len();
            ops[i] = random_op(rng);
        }
        if rng.unit() < p.const_ops {
            for op in &mut ops {
                match op {
                    BehaviorOp::Const { value, .. } => {
                        *value = (*value + rng.gauss() as f32 * p.sigma).clamp(-4.0, 4.0);
                    }
                    BehaviorOp::Threshold { baseline, gain, .. } => {
                        *baseline = (*baseline + rng.gauss() as f32 * p.sigma).clamp(-4.0, 4.0);
                        *gain = (*gain + rng.gauss() as f32 * p.sigma).clamp(0.001, 4.0);
                    }
                    BehaviorOp::Relax { alpha, .. } => {
                        *alpha = (*alpha + rng.gauss() as f32 * p.sigma).clamp(0.0, 1.0);
                    }
                    _ => {}
                }
            }
        }
        Self::from_ops(ops).unwrap_or_else(|_| self.clone())
    }
}

fn rnd_reg(rng: &mut XorShift) -> u8 {
    (rng.next_u64() % 14) as u8
}

fn rnd_sensor(rng: &mut XorShift) -> usize {
    (rng.next_u64() % slots::N_SENSORS as u64) as usize
}

fn rnd_action(rng: &mut XorShift) -> usize {
    slots::N_SENSORS + (rng.next_u64() % slots::N_ACTIONS as u64) as usize
}

/// Случайная валидная инструкция (для мутаций и стартовых программ).
fn random_op(rng: &mut XorShift) -> BehaviorOp {
    match rng.next_u64() % 12 {
        0 => BehaviorOp::Sensor {
            dst: rnd_reg(rng),
            state: rnd_sensor(rng),
        },
        1 => BehaviorOp::Action {
            state: rnd_action(rng),
            src: rnd_reg(rng),
        },
        2 => BehaviorOp::IfAct {
            cond: rnd_reg(rng),
            state: rnd_action(rng),
            src: rnd_reg(rng),
        },
        3 => BehaviorOp::Const {
            dst: rnd_reg(rng),
            value: rng.unit() as f32 * 2.0 - 1.0,
        },
        4 => BehaviorOp::Add {
            dst: rnd_reg(rng),
            a: rnd_reg(rng),
            b: rnd_reg(rng),
        },
        5 => BehaviorOp::Sub {
            dst: rnd_reg(rng),
            a: rnd_reg(rng),
            b: rnd_reg(rng),
        },
        6 => BehaviorOp::Mul {
            dst: rnd_reg(rng),
            a: rnd_reg(rng),
            b: rnd_reg(rng),
        },
        7 => BehaviorOp::Max {
            dst: rnd_reg(rng),
            a: rnd_reg(rng),
            b: rnd_reg(rng),
        },
        8 => BehaviorOp::Min {
            dst: rnd_reg(rng),
            a: rnd_reg(rng),
            b: rnd_reg(rng),
        },
        9 => BehaviorOp::Clamp01 { reg: rnd_reg(rng) },
        10 => BehaviorOp::Exp {
            dst: rnd_reg(rng),
            src: rnd_reg(rng),
        },
        _ => BehaviorOp::Relax {
            dst: rnd_reg(rng),
            target: rnd_reg(rng),
            alpha: rng.unit() as f32 * 0.5,
        },
    }
}

/// Упаковка: тег 8 бит, три байта операндов; константы — следующим словом.
fn pack_all(ops: &[BehaviorOp]) -> Vec<u32> {
    let mut out = Vec::new();
    for op in ops {
        match op {
            BehaviorOp::Sensor { dst, state } => out.push(pack(1, *dst, *state as u8, 0)),
            BehaviorOp::Action { state, src } => out.push(pack(2, *src, *state as u8, 0)),
            BehaviorOp::IfAct { cond, state, src } => out.push(pack(3, *cond, *state as u8, *src)),
            BehaviorOp::Const { dst, value } => {
                out.push(pack(4, *dst, 0, 0));
                out.push(value.to_bits());
            }
            BehaviorOp::Add { dst, a, b } => out.push(pack(5, *dst, *a, *b)),
            BehaviorOp::Sub { dst, a, b } => out.push(pack(6, *dst, *a, *b)),
            BehaviorOp::Mul { dst, a, b } => out.push(pack(7, *dst, *a, *b)),
            BehaviorOp::Max { dst, a, b } => out.push(pack(8, *dst, *a, *b)),
            BehaviorOp::Min { dst, a, b } => out.push(pack(9, *dst, *a, *b)),
            BehaviorOp::Clamp01 { reg } => out.push(pack(10, *reg, 0, 0)),
            BehaviorOp::Exp { dst, src } => out.push(pack(11, *dst, *src, 0)),
            BehaviorOp::Relax { dst, target, alpha } => {
                out.push(pack(12, *dst, *target, 0));
                out.push(alpha.to_bits());
            }
            BehaviorOp::Threshold {
                dst,
                src,
                baseline,
                gain,
            } => {
                out.push(pack(13, *dst, *src, 0));
                out.push(baseline.to_bits());
                out.push(gain.to_bits());
            }
        }
    }
    out
}

fn pack(tag: u32, a: u8, b: u8, c: u8) -> u32 {
    (tag << 24) | ((a as u32) << 16) | ((b as u32) << 8) | c as u32
}

fn unpack_all(words: &[u32]) -> Result<Vec<BehaviorOp>, String> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let read_f32 = |i: &mut usize| -> Result<f32, String> {
        if *i >= words.len() {
            return Err("обрыв константы в сериализации".into());
        }
        let v = f32::from_bits(words[*i]);
        *i += 1;
        Ok(v)
    };
    while i < words.len() {
        let w = words[i];
        i += 1;
        let tag = (w >> 24) as u8;
        let a = (w >> 16) as u8;
        let b = (w >> 8) as u8;
        let c = w as u8;
        match tag {
            1 => out.push(BehaviorOp::Sensor {
                dst: a,
                state: b as usize,
            }),
            2 => out.push(BehaviorOp::Action {
                state: b as usize,
                src: a,
            }),
            3 => out.push(BehaviorOp::IfAct {
                cond: a,
                state: b as usize,
                src: c,
            }),
            4 => {
                let value = read_f32(&mut i)?;
                out.push(BehaviorOp::Const { dst: a, value });
            }
            5 => out.push(BehaviorOp::Add { dst: a, a: b, b: c }),
            6 => out.push(BehaviorOp::Sub { dst: a, a: b, b: c }),
            7 => out.push(BehaviorOp::Mul { dst: a, a: b, b: c }),
            8 => out.push(BehaviorOp::Max { dst: a, a: b, b: c }),
            9 => out.push(BehaviorOp::Min { dst: a, a: b, b: c }),
            10 => out.push(BehaviorOp::Clamp01 { reg: a }),
            11 => out.push(BehaviorOp::Exp { dst: a, src: b }),
            12 => {
                let alpha = read_f32(&mut i)?;
                out.push(BehaviorOp::Relax {
                    dst: a,
                    target: b,
                    alpha,
                });
            }
            13 => {
                let baseline = read_f32(&mut i)?;
                let gain = read_f32(&mut i)?;
                out.push(BehaviorOp::Threshold {
                    dst: a,
                    src: b,
                    baseline,
                    gain,
                });
            }
            _ => return Err(format!("неизвестный тег {tag} в сериализации")),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn states(n: usize) -> Vec<f32> {
        vec![0.0; slots::N_STATE * n]
    }

    /// Сенсор → действие напрямую (голод → драйв), по лейнам.
    #[test]
    fn hunger_drives() {
        let prog = BehaviorProgram::from_ops(vec![
            BehaviorOp::Sensor {
                dst: 0,
                state: slots::HUNGER,
            },
            BehaviorOp::Action {
                state: slots::DRIVE,
                src: 0,
            },
        ])
        .unwrap();
        let mut s = states(2);
        s[slots::HUNGER * 2] = 0.7;
        s[slots::HUNGER * 2 + 1] = 0.3;
        prog.execute(&mut s, 2);
        assert!((s[slots::DRIVE * 2] - 0.7).abs() < 1e-6);
        assert!((s[slots::DRIVE * 2 + 1] - 0.3).abs() < 1e-6);
    }

    /// Предикатное действие: есть только при голоде > 0.8.
    #[test]
    fn conditional_action() {
        let prog = BehaviorProgram::from_ops(vec![
            BehaviorOp::Sensor {
                dst: 0,
                state: slots::HUNGER,
            },
            BehaviorOp::Const { dst: 1, value: 0.8 },
            BehaviorOp::Sub { dst: 2, a: 0, b: 1 },
            BehaviorOp::Const { dst: 3, value: 1.0 },
            BehaviorOp::IfAct {
                cond: 2,
                state: slots::EAT,
                src: 3,
            },
        ])
        .unwrap();
        let mut s = states(2);
        s[slots::HUNGER * 2] = 0.9; // голоден → есть
        s[slots::HUNGER * 2 + 1] = 0.5; // сыт → не есть
        prog.execute(&mut s, 2);
        assert_eq!(s[slots::EAT * 2], 1.0);
        assert_eq!(s[slots::EAT * 2 + 1], 0.0);
    }

    /// SIMD-чанк (8) и скалярный хвост (11) дают один результат.
    #[test]
    fn simd_and_scalar_tail_agree() {
        let prog = BehaviorProgram::from_ops(vec![
            BehaviorOp::Sensor {
                dst: 0,
                state: slots::HUNGER,
            },
            BehaviorOp::Threshold {
                dst: 0,
                src: 0,
                baseline: 0.5,
                gain: 1.0,
            },
            BehaviorOp::Action {
                state: slots::DRIVE,
                src: 0,
            },
        ])
        .unwrap();
        let n = 11;
        let mut s = states(n);
        for i in 0..n {
            s[slots::HUNGER * n + i] = i as f32 / (n - 1) as f32;
        }
        prog.execute(&mut s, n);
        for i in 0..n {
            let h = i as f32 / (n - 1) as f32;
            let want = (h - 0.5).max(0.0);
            assert!(
                (s[slots::DRIVE * n + i] - want).abs() < 1e-5,
                "lane {i}: {want} vs {}",
                s[slots::DRIVE * n + i]
            );
        }
    }

    /// Детерминизм: два прогона совпадают бит-в-бит.
    #[test]
    fn deterministic_runs() {
        let prog = BehaviorProgram::from_ops(vec![
            BehaviorOp::Sensor {
                dst: 0,
                state: slots::HUNGER,
            },
            BehaviorOp::Sensor {
                dst: 1,
                state: slots::SMELL_L,
            },
            BehaviorOp::Sub { dst: 2, a: 0, b: 1 },
            BehaviorOp::Action {
                state: slots::STEER,
                src: 2,
            },
            BehaviorOp::Const { dst: 3, value: 0.9 },
            BehaviorOp::Mul { dst: 4, a: 0, b: 3 },
            BehaviorOp::Min { dst: 4, a: 4, b: 3 },
            BehaviorOp::Action {
                state: slots::DRIVE,
                src: 4,
            },
        ])
        .unwrap();
        let n = 9;
        let mut s1 = states(n);
        for i in 0..n {
            s1[slots::HUNGER * n + i] = 0.2 + 0.07 * i as f32;
            s1[slots::SMELL_L * n + i] = 0.1 + 0.05 * i as f32;
        }
        let mut s2 = s1.clone();
        prog.execute(&mut s1, n);
        prog.execute(&mut s2, n);
        assert_eq!(s1, s2);
    }

    /// Сериализация: from_ops → u32 → from_u32 → те же инструкции.
    #[test]
    fn serialization_roundtrip() {
        let ops = vec![
            BehaviorOp::Sensor { dst: 0, state: 0 },
            BehaviorOp::Threshold {
                dst: 1,
                src: 0,
                baseline: 0.3,
                gain: 1.7,
            },
            BehaviorOp::IfAct {
                cond: 1,
                state: 11,
                src: 2,
            },
            BehaviorOp::Const {
                dst: 2,
                value: 0.65,
            },
            BehaviorOp::Relax {
                dst: 3,
                target: 1,
                alpha: 0.2,
            },
            BehaviorOp::Exp { dst: 4, src: 3 },
            BehaviorOp::Sub { dst: 5, a: 4, b: 2 },
            BehaviorOp::Action { state: 9, src: 5 },
        ];
        let prog = BehaviorProgram::from_ops(ops.clone()).unwrap();
        let words = prog.to_u32();
        let back = BehaviorProgram::from_u32(&words).unwrap();
        assert_eq!(back, prog);
        assert_eq!(back.ops(), &ops[..]);
    }

    /// Мутация: 400 поколений — все прогоны валидны, длина ограничена.
    #[test]
    fn mutation_always_valid() {
        let mut rng = XorShift::new(123);
        let mut prog = BehaviorProgram::from_ops(vec![
            BehaviorOp::Sensor { dst: 0, state: 0 },
            BehaviorOp::Action { state: 9, src: 0 },
        ])
        .unwrap();
        let params = BehaviorMutate::default();
        for round in 0..400 {
            prog = prog.mutate(&mut rng, &params);
            assert!(
                BehaviorProgram::from_ops(prog.ops().to_vec()).is_ok(),
                "round {round}: мутация дала невалидную прогу"
            );
            assert!(prog.len() <= params.max_len);
            assert!(prog.len() >= 2);
        }
    }

    /// Пустая программа — no-op.
    #[test]
    fn empty_program_is_noop() {
        let prog = BehaviorProgram::from_ops(Vec::new()).unwrap();
        assert!(prog.is_empty());
        let mut s = states(3);
        s[slots::DRIVE * 3 + 1] = 0.5;
        prog.execute(&mut s, 3);
        assert_eq!(s[slots::DRIVE * 3 + 1], 0.5);
    }
}
