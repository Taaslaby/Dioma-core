//! -------------------------------------------------------------------------
//! Filename:    src/pythia.rs
//! Class:       neuworks
//! Purpose:     геномные правила динамики клеток (движок «Пифия»)
//! Description: маленькая стековая ВМ: правило = программа из опкодов,
//!              компилируется из гена (шаблон + параметры). Гены мутируют —
//!              эволюция меняет ДИНАМИКУ клеток, а не только веса связей.
//!              Переменные клетки: Act, Inp, Out, S3..S6, Troph (трофический
//!              сигнал: награда снижает порог возбуждения).
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use crate::rng::XorShift;

/// Слоты переменных клетки.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Var {
    Act,   // активность (0..1)
    Inp,   // внешний вход за тик
    Out,   // выходной импульс (затухает)
    S3,    // свободный (привыкание)
    S4,    // свободный (динамический порог / candidacy)
    S5,    // счётчик рефрактерности
    S6,    // флаг победителя конкуренции
    Troph, // трофический сигнал (награда), затухает
}

impl Var {
    fn idx(self) -> usize {
        match self {
            Var::Act => 0,
            Var::Inp => 1,
            Var::Out => 2,
            Var::S3 => 3,
            Var::S4 => 4,
            Var::S5 => 5,
            Var::S6 => 6,
            Var::Troph => 7,
        }
    }
}

/// Опкод правила. Бинарные операции: push a; push b; op → a OP b.
#[derive(Clone, Debug)]
pub enum Op {
    Load(Var),
    Const(f32),
    Store(Var),
    Add,
    Sub,
    Mul,
    Max,
    Min,
    /// v ← v + rate·(top − v): плавное притяжение переменной к вершине стека.
    Tend(Var, f32),
    Clamp01,
    /// top += равномерный шум ±amp.
    Noise(f32),
    IfPosGoto(usize),
    Goto(usize),
    Stop,
    /// Спайк: если top > порога из `thr_var` (и не в рефрактере) — Out=1 и
    /// старт рефрактера, иначе Out затухает. Push Out.
    Fire { thr_var: Var, refr: f32, decay: f32 },
    /// Записать top в S4 как candidacy для конкуренции этого тика.
    Compete,
}

/// Шаблоны правил (видовой признак, мутации не меняют).
pub mod templates {
    pub const INTEGRATOR: u8 = 0;
    pub const SPIKER: u8 = 1;
    pub const HABITUATOR: u8 = 2;
    pub const CONSOLIDATOR: u8 = 3;
}

/// Ген правила: шаблон + параметры. Живёт в геноме существа рядом со связями.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuleGene {
    pub template: u8,
    /// Смысл параметров зависит от шаблона.
    pub p: [f32; 8],
}

impl Default for RuleGene {
    fn default() -> Self {
        // Интегратор: leak 0.9, gain 0.3.
        Self { template: templates::INTEGRATOR, p: [0.9, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] }
    }
}

impl RuleGene {
    /// SPIKER: p=[leak, gain, thr, refr].
    pub fn spiker(leak: f32, gain: f32, thr: f32, refr: f32) -> Self {
        Self { template: templates::SPIKER, p: [leak, gain, thr, refr, 0.0, 0.0, 0.0, 0.0] }
    }

    /// CONSOLIDATOR: p=[leak, gain, thr0, k_troph], refr=3.
    pub fn consolidator(thr0: f32, k_troph: f32) -> Self {
        Self {
            template: templates::CONSOLIDATOR,
            p: [0.85, 0.35, thr0, k_troph, 0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Мягкая мутация параметров (шаблон — видовой признак).
    pub fn mutate(&self, rng: &mut XorShift, sigma: f32) -> RuleGene {
        let mut g = *self;
        for v in g.p.iter_mut() {
            *v = (*v + rng.gauss() as f32 * sigma).clamp(0.0, 2.0);
        }
        g
    }
}

fn fire(refr: f32) -> Op {
    Op::Fire { thr_var: Var::S4, refr: refr.max(1.0), decay: 0.55 }
}

/// Компиляция гена → программа. Один ген = одна динамика клетки.
pub fn compile(gene: &RuleGene) -> Vec<Op> {
    let p = gene.p;
    match gene.template {
        templates::SPIKER => {
            let mut prog = vec![Op::Const(p[2]), Op::Store(Var::S4)];
            prog.extend(integrate(p[0], p[1], 0.0));
            prog.extend([
                Op::Load(Var::Act),
                fire(p[3]),
                Op::Store(Var::Out),
                Op::Stop,
            ]);
            prog
        }
        templates::HABITUATOR => {
            // Привыкание: S3 → уровень недавнего входа; эффективный множитель
            // gain_eff = max(gain·(1 − 0.7·S3), 0.15). Порог p[3], рефрактер p[4].
            let mut prog = vec![
                Op::Load(Var::Inp),
                Op::Const(0.0),
                Op::Max,
                Op::Tend(Var::S3, 0.02),
            ];
            prog.push(Op::Load(Var::Act));
            prog.push(Op::Const(p[0]));
            prog.push(Op::Mul);
            prog.push(Op::Load(Var::Inp));
            prog.extend([
                Op::Const(1.0),
                Op::Load(Var::S3),
                Op::Const(0.7),
                Op::Mul,
                Op::Sub,
                Op::Const(p[1]),
                Op::Mul,
                Op::Const(0.15),
                Op::Max,
                Op::Mul, // Inp · gain_eff
                Op::Add,
                Op::Clamp01,
                Op::Store(Var::Act),
                Op::Const(p[2]),
                Op::Store(Var::S4),
                Op::Load(Var::Act),
                fire(p[4]),
                Op::Store(Var::Out),
                Op::Stop,
            ]);
            prog
        }
        templates::CONSOLIDATOR => {
            // Трофический сигнал снижает порог: eff_thr = max(thr0·(1 − k·Troph), 0.05).
            let mut prog = vec![
                Op::Load(Var::Troph),
                Op::Const(0.95),
                Op::Mul,
                Op::Store(Var::Troph),
                Op::Const(1.0),
                Op::Load(Var::Troph),
                Op::Const(p[3]),
                Op::Mul,
                Op::Sub,
                Op::Const(p[2]),
                Op::Mul,
                Op::Const(0.05),
                Op::Max,
                Op::Store(Var::S4),
            ];
            prog.extend(integrate(p[0], p[1], 0.0));
            prog.extend([
                Op::Load(Var::Act),
                fire(p[4].max(3.0)),
                Op::Store(Var::Out),
                Op::Stop,
            ]);
            prog
        }
        // INTEGRATOR и всё неизвестное — утечка + накопление.
        _ => {
            let mut prog = integrate(p[0], p[1], p[2]);
            prog.extend([Op::Load(Var::Act), Op::Store(Var::Out), Op::Stop]);
            prog
        }
    }
}

/// Тело интегратора: Act ← clamp01(Act·leak + Inp·gain [+ noise]).
fn integrate(leak: f32, gain: f32, noise: f32) -> Vec<Op> {
    let mut v = vec![Op::Load(Var::Act), Op::Const(leak.clamp(0.0, 1.0)), Op::Mul];
    v.push(Op::Load(Var::Inp));
    if gain != 1.0 {
        v.push(Op::Const(gain));
        v.push(Op::Mul);
    }
    v.push(Op::Add);
    if noise > 0.0 {
        v.push(Op::Noise(noise));
    }
    v.extend([Op::Clamp01, Op::Store(Var::Act)]);
    v
}

/// Состояние одной клетки: 8 переменных.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellState(pub [f32; 8]);

/// Юнит Пифии: N клеток с одним правилом (один регион-исполнитель).
pub struct PythiaUnit {
    rule: Vec<Op>,
    cells: Vec<CellState>,
    rng: XorShift,
    cursor: u64,
    has_compete: bool,
}

impl PythiaUnit {
    pub fn from_gene(gene: &RuleGene, n_cells: usize, seed: u64) -> Self {
        let rule = compile(gene);
        let has_compete = rule.iter().any(|op| matches!(op, Op::Compete));
        Self {
            rule,
            cells: vec![CellState::default(); n_cells],
            rng: XorShift::new(seed | 1),
            cursor: 0,
            has_compete,
        }
    }

    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// Вход за тик (Inp каждой клетки).
    pub fn set_inputs(&mut self, inputs: &[f32]) {
        for (c, &x) in self.cells.iter_mut().zip(inputs) {
            c.0[Var::Inp.idx()] = x;
        }
    }

    /// Награда клетке: Troph ← max(Troph, f).
    pub fn reward_cell(&mut self, i: usize, f: f32) {
        if let Some(c) = self.cells.get_mut(i) {
            c.0[Var::Troph.idx()] = c.0[Var::Troph.idx()].max(f);
        }
    }

    pub fn out(&self, i: usize) -> f32 {
        self.cells[i].0[Var::Out.idx()]
    }

    /// Победитель последней конкуренции (S6 == 1).
    pub fn winner(&self) -> Option<usize> {
        self.cells.iter().position(|c| c.0[Var::S6.idx()] >= 0.5)
    }

    /// Один тик: прогон правила по всем клеткам + разрешение конкуренции.
    pub fn step(&mut self) -> Vec<f32> {
        let mut stack: Vec<f32> = Vec::with_capacity(8);
        for cell in self.cells.iter_mut() {
            stack.clear();
            let mut pc = 0usize;
            let mut budget = self.rule.len() * 16 + 16;
            while pc < self.rule.len() && budget > 0 {
                budget -= 1;
                match &self.rule[pc] {
                    Op::Load(v) => stack.push(cell.0[v.idx()]),
                    Op::Const(x) => stack.push(*x),
                    Op::Store(v) => {
                        if let Some(x) = stack.pop() {
                            cell.0[v.idx()] = x;
                        }
                    }
                    Op::Add => binop(&mut stack, |a, b| a + b),
                    Op::Sub => binop(&mut stack, |a, b| a - b),
                    Op::Mul => binop(&mut stack, |a, b| a * b),
                    Op::Max => binop(&mut stack, f32::max),
                    Op::Min => binop(&mut stack, f32::min),
                    Op::Tend(v, rate) => {
                        let target = stack.last().copied().unwrap_or(0.0);
                        let s = v.idx();
                        cell.0[s] += rate * (target - cell.0[s]);
                    }
                    Op::Clamp01 => {
                        if let Some(x) = stack.last_mut() {
                            *x = x.clamp(0.0, 1.0);
                        }
                    }
                    Op::Noise(amp) => {
                        self.cursor += 1;
                        let u = self.rng.unit() as f32 - 0.5;
                        if let Some(x) = stack.last_mut() {
                            *x += 2.0 * amp * u;
                        }
                    }
                    Op::IfPosGoto(line) => {
                        if stack.pop().is_some_and(|x| x > 0.0) {
                            pc = *line;
                            continue;
                        }
                    }
                    Op::Goto(line) => {
                        pc = *line;
                        continue;
                    }
                    Op::Stop => break,
                    Op::Fire { thr_var, refr, decay } => {
                        let cand = stack.pop().unwrap_or(0.0);
                        let thr = cell.0[thr_var.idx()];
                        let o = Var::Out.idx();
                        let r = Var::S5.idx();
                        if cell.0[r] > 0.5 {
                            cell.0[r] -= 1.0;
                            cell.0[o] *= decay;
                        } else if cand > thr {
                            cell.0[o] = 1.0;
                            cell.0[r] = *refr;
                        } else {
                            cell.0[o] *= decay;
                        }
                        stack.push(cell.0[o]);
                    }
                    Op::Compete => {
                        cell.0[Var::S4.idx()] = stack.last().copied().unwrap_or(0.0);
                    }
                }
                pc += 1;
            }
        }
        if self.has_compete {
            let mut best = 0usize;
            for (i, c) in self.cells.iter().enumerate() {
                if c.0[Var::S4.idx()] > self.cells[best].0[Var::S4.idx()] {
                    best = i;
                }
            }
            for (i, c) in self.cells.iter_mut().enumerate() {
                c.0[Var::S6.idx()] = if i == best { 1.0 } else { 0.0 };
            }
        }
        self.cells.iter().map(|c| c.0[Var::Out.idx()]).collect()
    }
}

fn binop(stack: &mut Vec<f32>, f: impl Fn(f32, f32) -> f32) {
    if stack.len() >= 2 {
        let b = stack.pop().unwrap();
        let a = stack.pop().unwrap();
        stack.push(f(a, b));
    }
}

/// Конкуренция кандидатов: явный лидер со счётом ≥ min_score и отрывом ≥ lead.
pub fn compete(scores: &[f32], min_score: f32, lead: f32) -> Option<usize> {
    if scores.is_empty() {
        return None;
    }
    let mut best = 0usize;
    for (i, &s) in scores.iter().enumerate() {
        if s > scores[best] {
            best = i;
        }
    }
    if scores[best] < min_score {
        return None;
    }
    let second = scores
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != best)
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    (scores[best] - second.max(0.0) >= lead).then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrator_tends_to_input() {
        let g = RuleGene { template: templates::INTEGRATOR, p: [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
        let mut u = PythiaUnit::from_gene(&g, 1, 7);
        u.set_inputs(&[1.0]);
        let mut last = 0.0f32;
        for _ in 0..50 {
            last = u.step()[0];
        }
        assert!((last - 1.0).abs() < 0.15, "integrator must approach input, got {last}");
    }

    #[test]
    fn spiker_threshold_gates_output() {
        // Стационарная точка интегратора: Act_ss = inp·gain/(1−leak) = 2·inp.
        // Порог выше 1.0 недостижим (clamp01) — клетка молчит навсегда.
        let quiet = RuleGene::spiker(0.8, 0.4, 1.5, 2.0);
        let loud = RuleGene::spiker(0.8, 0.4, 0.35, 2.0);
        let mut q = PythiaUnit::from_gene(&quiet, 1, 1);
        let mut l = PythiaUnit::from_gene(&loud, 1, 1);
        for _ in 0..40 {
            q.set_inputs(&[0.45]);
            l.set_inputs(&[0.45]);
            q.step();
            l.step();
        }
        let run = |u: &mut PythiaUnit| -> f32 {
            (0..20).map(|_| { u.set_inputs(&[0.45]); u.step()[0] }).sum()
        };
        let qo = run(&mut q);
        let lo = run(&mut l);
        assert!(qo < 0.5, "unreachable threshold must stay quiet, got {qo}");
        assert!(lo > 5.0, "low threshold must fire, got {lo}");
    }

    #[test]
    fn troph_lowers_threshold() {
        // Act_ss = 0.30·0.35/(1−0.85) ≈ 0.70. До награды порог 0.9 недостижим,
        // после Troph=1 порог max(0.9·(1−0.8), 0.05)=0.18 — клетка стреляет.
        let g = RuleGene::consolidator(0.9, 0.8);
        let mut u = PythiaUnit::from_gene(&g, 1, 3);
        for _ in 0..30 {
            u.set_inputs(&[0.30]);
            u.step();
        }
        let before: f32 = (0..10).map(|_| { u.set_inputs(&[0.30]); u.step()[0] }).sum();
        u.reward_cell(0, 1.0);
        let after: f32 = (0..10).map(|_| { u.set_inputs(&[0.30]); u.step()[0] }).sum();
        assert!(
            before < 0.5 && after > 5.0,
            "reward must unlock firing: before={before} after={after}"
        );
    }

    #[test]
    fn compete_picks_clear_leader() {
        assert_eq!(compete(&[0.9, 0.2], 0.3, 0.05), Some(0));
        assert_eq!(compete(&[0.2, 0.9], 0.3, 0.05), Some(1));
        assert_eq!(compete(&[0.1, 0.12], 0.3, 0.05), None);
        assert_eq!(compete(&[0.9, 0.85], 0.3, 0.2), None);
        assert_eq!(compete(&[], 0.3, 0.05), None);
    }

    #[test]
    fn mutation_moves_params_only() {
        let g = RuleGene::spiker(0.8, 0.4, 0.5, 2.0);
        let mut rng = XorShift::new(42);
        let m = g.mutate(&mut rng, 0.05);
        assert_eq!(m.template, g.template);
        assert!(m.p.iter().zip(g.p.iter()).any(|(a, b)| a != b));
    }
}
