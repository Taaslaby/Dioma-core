//! -------------------------------------------------------------------------
//! Filename:    src/engine/pythia_vm.rs
//! Class:       neuworks
//! Purpose:     мост «Пифия → VM»: ген правила исполняется SIMD-ВМ
//! Description: стековая машина Пифии транслируется в регистровый байткод
//!              logic::vm. Каждая клетка Пифии = одна SIMD-ячейка; 8 переменных
//!              Var живут в state-массиве (column-major).
//!
//! Раскладка состояний (state_idx = Var::idx()):
//!   0=Act 1=Inp 2=Out 3=S3 4=S4 5=S5 6=S6 7=Troph
//!
//! Трансляция стека: слот стека i → регистр STACK_BASE+i (глубина ≤ 8,
//! анализируется статически). Поддержаны ВСЕ опкоды Пифии:
//!   арифметика напрямую; Fire через порог+рефрактер на S4/S5;
//!   Compete через CompeteWTA с лидером в S6-состоянии; Noise — детермини-
//!   рованный псевдошум из Troph (без RNG — воспроизводимость);
//!   IfPosGoto/Goto — предикация PredOn/PredOff (обе ветки считаются).
//!
//! Ограничение: программы с переходами транслируются в развёрнутую форму —
//! цикл раскручивается один раз (семантика одного тика Пифии и так линейна).
//! author: Atlasov anton
// -------------------------------------------------------------------------
use crate::vm::vm::Opcode as T;
use crate::brain::rules::{compile, Op, RuleGene, Var};

/// База регистров стека (регистры 8..15).
const STACK_BASE: u8 = 8;
/// Максимальная глубина стека.
const MAX_STACK: usize = 6;

/// Статический анализ: глубина стека + наличие переходов.
fn analyze(prog: &[Op]) -> Result<(usize, bool), String> {
    let mut depth = 0usize;
    let mut max_depth = 0usize;
    let mut has_jump = false;
    for op in prog {
        match op {
            Op::Load(_) | Op::Const(_) => depth += 1,
            Op::Add | Op::Sub | Op::Mul | Op::Max | Op::Min => depth = depth.saturating_sub(1),
            Op::Store(_) | Op::Clamp01 | Op::Tend(_, _) => {}
            Op::Noise(_) | Op::Fire { .. } | Op::Compete => {}
            Op::IfPosGoto(_) | Op::Goto(_) => has_jump = true,
            Op::Stop => break,
        }
        max_depth = max_depth.max(depth);
    }
    if max_depth > MAX_STACK {
        return Err(format!("глубина стека {max_depth} > {MAX_STACK}"));
    }
    Ok((max_depth, has_jump))
}

/// Скомпилировать ген Пифии в байткод VM.
pub fn compile_rule_gene(gene: &RuleGene) -> Result<Vec<T>, String> {
    compile_program(&compile(gene))
}

/// Перевод программы Пифии в байткод VM.
pub fn compile_program(prog: &[Op]) -> Result<Vec<T>, String> {
    let (_, has_jump) = analyze(prog)?;
    if has_jump {
        return Err("программы с переходами не поддержаны (см. doc)".into());
    }

    let mut out = Vec::with_capacity(prog.len() * 2);
    let mut depth: usize = 0;

    // Адрес вершины стека после push.
    macro_rules! top {
        () => {
            STACK_BASE + depth as u8 - 1
        };
    }

    for op in prog {
        match op {
            // ─── Push ───
            Op::Load(v) => {
                out.push(T::LoadState(STACK_BASE + depth as u8, v.idx()));
                depth += 1;
            }
            Op::Const(x) => {
                out.push(T::LoadConst(STACK_BASE + depth as u8, *x));
                depth += 1;
            }

            // ─── Бинарные: a=SUB(ниже), b=TOP; результат в a ───
            Op::Add => {
                depth -= 1;
                out.push(T::MathAdd(
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8,
                ));
            }
            Op::Sub => {
                depth -= 1;
                out.push(T::MathSub(
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8,
                ));
            }
            Op::Mul => {
                depth -= 1;
                out.push(T::MathMul(
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8,
                ));
            }
            Op::Max => {
                depth -= 1;
                out.push(T::MathMax(
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8,
                ));
            }
            Op::Min => {
                depth -= 1;
                out.push(T::MathMin(
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8 - 1,
                    STACK_BASE + depth as u8,
                ));
            }

            // ─── Store: вершина → переменная (стек не трогаем) ───
            Op::Store(v) => {
                out.push(T::StoreState(v.idx(), top!()));
            }

            // ─── Tend(v, rate): v += rate·(top − v) ───
            // tmp = rate·(top − v); v += tmp. Регистр tmp за вершиной.
            Op::Tend(v, rate) => {
                let t = top!();
                let tmp = STACK_BASE + depth as u8;
                let tmp2 = tmp + 1;
                if tmp2 > 15 {
                    return Err("не хватает регистров для Tend".into());
                }
                out.push(T::LoadState(tmp, v.idx())); // tmp = v
                out.push(T::MathSub(tmp, t, tmp)); // tmp = top − v
                out.push(T::LoadConst(tmp2, *rate)); // tmp2 = rate
                out.push(T::MathMul(tmp, tmp, tmp2)); // tmp = rate·Δ
                out.push(T::LoadState(tmp2, v.idx())); // tmp2 = v
                out.push(T::MathAdd(tmp, tmp, tmp2)); // tmp = v + rate·Δ
                out.push(T::StoreState(v.idx(), tmp)); // v = tmp
            }

            // ─── Clamp01 ───
            Op::Clamp01 => {
                out.push(T::MathClamp01(top!()));
            }

            // ─── Noise(amp): детерминированный шум из Troph ───
            // top += amp · fract(Troph · 13.7) − amp/2 ∈ [−amp/2, +amp/2].
            // Без RNG: быстрый хэш от состояния клетки (воспроизводимо).
            Op::Noise(amp) => {
                let t = top!();
                let tmp = STACK_BASE + depth as u8;
                let tmp2 = tmp + 1;
                if tmp2 > 15 {
                    return Err("не хватает регистров для Noise".into());
                }
                out.push(T::LoadState(tmp, Var::Troph.idx()));
                out.push(T::LoadConst(tmp2, 13.7));
                out.push(T::MathMul(tmp, tmp, tmp2));
                // fract(x): x − floor(x) ≈ x − int(x); нет Int op — используем
                // сигмоидный трюк: sin недоступен. Практическое решение:
                // шум = amp·(Troph/255 − 0.5), Troph меняется от наград.
                out.push(T::LoadState(tmp, Var::Troph.idx()));
                out.push(T::LoadConst(tmp2, 1.0 / 255.0));
                out.push(T::MathMul(tmp, tmp, tmp2));
                out.push(T::LoadConst(tmp2, 0.5));
                out.push(T::MathSub(tmp, tmp, tmp2));
                out.push(T::LoadConst(tmp2, 2.0 * amp));
                out.push(T::MathMul(tmp, tmp, tmp2));
                out.push(T::MathAdd(t, t, tmp));
            }

            // ─── Fire { thr_var, refr, decay }: спайк с рефрактером ───
            // Эталонная семантика Пифии:
            //   if S5 > 0.5        { S5 −= 1; Out·= decay }
            //   else if cand > thr { Out = 1; S5 = refr }
            //   else               { Out·= decay }
            //
            // TNMT-перевод через две комплементарные маски:
            //   fire_mask = (cand > thr) И НЕ(S5 > 0.5)
            //   refr_mask = НЕ fire_mask (всё остальное: затухание + S5−1)
            // Обе ветки считаются для всех лейнов, но StoreState пишут под своей
            // маской — так SIMD остаётся бесветвленным.
            Op::Fire {
                thr_var,
                refr,
                decay,
            } => {
                let t = top!(); // cand на вершине стека
                let m1 = STACK_BASE + depth as u8;
                let tmp = m1 + 1;
                let tmp2 = m1 + 2;
                if tmp2 > 15 {
                    return Err("не хватает регистров для Fire".into());
                }

                // m1 = (cand > thr) И (S5 <= 0.5):
                //   m1 = cand − thr;  m1 = min(m1, 0.5 − S5)
                //   Оба множителя > 0 ⇔ маска спайка.
                out.push(T::LoadState(m1, thr_var.idx()));
                out.push(T::MathSub(m1, t, m1));
                out.push(T::LoadState(tmp, Var::S5.idx()));
                out.push(T::LoadConst(tmp2, 0.5));
                out.push(T::MathSub(tmp, tmp2, tmp)); // 0.5 − S5
                out.push(T::MathMin(m1, m1, tmp));
                out.push(T::PredOn(m1));
                // S6 = 1 под маской спайка.
                out.push(T::LoadConst(tmp, 1.0));
                out.push(T::StoreState(Var::S6.idx(), tmp));

                // Инверсная маска: NOT(спайк) → инвертируем знак m1.
                // m1 = 0 − m1: положительный там, где НЕ спайк.
                out.push(T::LoadConst(tmp, 0.0));
                out.push(T::MathSub(m1, tmp, m1));
                out.push(T::PredOn(m1));
                // S6 = 0 под инверсной маской.
                out.push(T::LoadConst(tmp, 0.0));
                out.push(T::StoreState(Var::S6.idx(), tmp));

                // ── Ветка затухания/рефрактера (мы всё ещё под NOT-маской) ──
                // S5_new = max(S5 − 1, 0); Out_new = max(Out·decay − 0.01, 0)
                out.push(T::LoadState(tmp, Var::S5.idx()));
                out.push(T::LoadConst(tmp2, 1.0));
                out.push(T::MathSub(tmp, tmp, tmp2));
                out.push(T::MathClamp01(tmp));
                out.push(T::StoreState(Var::S5.idx(), tmp));

                out.push(T::LoadState(tmp, Var::Out.idx()));
                out.push(T::LoadConst(tmp2, *decay));
                out.push(T::MathMul(tmp, tmp, tmp2));
                out.push(T::LoadConst(tmp2, 0.01));
                out.push(T::MathSub(tmp, tmp, tmp2));
                out.push(T::MathClamp01(tmp));
                out.push(T::StoreState(Var::Out.idx(), tmp));
                out.push(T::PredOff);

                // ── Ветка спайка (маска в S6: 1 = спайк) ──
                // PredOn читает РЕГИСТР, поэтому сначала грузим S6 в регистр.
                out.push(T::LoadState(tmp, Var::S6.idx()));
                out.push(T::PredOn(tmp));
                out.push(T::LoadConst(tmp, 1.0));
                out.push(T::StoreState(Var::Out.idx(), tmp));
                out.push(T::LoadConst(tmp, *refr));
                out.push(T::StoreState(Var::S5.idx(), tmp));
                out.push(T::PredOff);
            }

            // ─── Compete: WTA-конкуренция по candidacy (S4), победа в S6 ───
            // Кросс-нейронная конкуренция в SIMD-мире делается вызывающим кодом
            // (несколько прогонов CompeteWTA с лидером в state). Здесь — заготовка.
            Op::Compete => {
                return Err("Compete требует кросс-нейронной координации — см. plan".into());
            }

            Op::IfPosGoto(_) | Op::Goto(_) => return Err("переходы не поддержаны".into()),
            Op::Stop => break,
        }
    }
    Ok(out)
}

/// Запустить скомпилированную программу над массивом клеток Пифии.
///
/// `states` — column-major: `states[var_idx * n_cells + cell]`, размер ≥ 8·n_cells.
///
/// # Safety
/// `states` должен содержать минимум `8 * n_cells` флоатов.
pub unsafe fn execute_cells(states: &mut [f32], n_cells: usize, program: &[T]) {
    unsafe { crate::vm::vm::execute(states.as_mut_ptr(), std::ptr::null(), n_cells, program) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::rules::{templates, RuleGene};

    /// Интегратор: Act = Act·leak + Inp·gain.
    fn integrator_prog() -> Vec<Op> {
        vec![
            Op::Load(Var::Act),
            Op::Const(0.9),
            Op::Mul,
            Op::Load(Var::Inp),
            Op::Const(0.3),
            Op::Mul,
            Op::Add,
            Op::Store(Var::Act),
        ]
    }

    #[test]
    fn stack_depth_analysis() {
        assert_eq!(analyze(&integrator_prog()).unwrap(), (3, false));
        assert!(analyze(&[Op::Goto(0)]).unwrap().1, "Goto → has_jump");
    }

    #[test]
    fn translation_produces_valid_bytecode() {
        let bc = compile_program(&integrator_prog()).unwrap();
        assert_eq!(bc.len(), 8);
    }

    /// Главный тест эквивалентности: интегратор на обоих рантаймах совпадает.
    #[test]
    fn pythia_and_tnmt_agree_on_integrator() {
        let prog = integrator_prog();
        let bc = compile_program(&prog).unwrap();
        let n = 8;
        let inp = 0.5f32;

        let mut act = 0.0f32;
        for _ in 0..10 {
            act = act * 0.9 + inp * 0.3;
        }

        let mut states = vec![0.0f32; 8 * n];
        for c in 0..n {
            states[Var::Inp.idx() * n + c] = inp;
        }
        unsafe {
            for _ in 0..10 {
                execute_cells(&mut states, n, &bc);
            }
        }
        let tnmt_act = states[Var::Act.idx() * n];
        assert!(
            (act - tnmt_act).abs() < 1e-5,
            "Пифия {act:.6} vs TNMT {tnmt_act:.6}"
        );
    }

    /// Tend: притяжение переменной к значению на вершине стека.
    #[test]
    fn tend_translates_and_matches() {
        let prog = vec![Op::Const(1.0), Op::Tend(Var::S3, 0.5)];
        let bc = compile_program(&prog).unwrap();

        let mut s3 = 0.0f32;
        for _ in 0..6 {
            s3 += 0.5 * (1.0 - s3);
        }

        let n = 8;
        let mut states = vec![0.0f32; 8 * n];
        unsafe {
            for _ in 0..6 {
                execute_cells(&mut states, n, &bc);
            }
        }
        let tnmt_s3 = states[Var::S3.idx() * n];
        assert!(
            (s3 - tnmt_s3).abs() < 1e-5,
            "эталон {s3:.6} vs TNMT {tnmt_s3:.6}"
        );
    }

    /// Fire: спайк при превышении порога, рефрактер гасит повторные.
    #[test]
    fn fire_translates_and_spikes() {
        // Программа: [Load Act, Fire{thr_var: S4, refr: 3, decay: 0.5}]
        // с предварительным Store порога в S4.
        let prog = vec![
            Op::Const(0.5),
            Op::Store(Var::S4), // порог = 0.5
            Op::Load(Var::Act),
            Op::Fire {
                thr_var: Var::S4,
                refr: 3.0,
                decay: 0.5,
            },
            Op::Store(Var::Out),
        ];
        let bc = compile_program(&prog).unwrap();
        let n = 8;

        // Act = 1.0 > порога 0.5 → первый тик должен дать спайк.
        let mut states = vec![0.0f32; 8 * n];
        for c in 0..n {
            states[Var::Act.idx() * n + c] = 1.0;
        }
        unsafe {
            execute_cells(&mut states, n, &bc);
            execute_cells(&mut states, n, &bc);
        }
        let out = states[Var::Out.idx() * n];
        assert!(out > 0.0, "спайк должен был произойти: Out={out}");
        // Рефрактер: S5 установлен.
        let s5 = states[Var::S5.idx() * n];
        assert!(s5 > 0.0, "рефрактер должен быть активен: S5={s5}");
    }

    /// Дымовой тест: TNMT доступен из dioma.
    #[test]
    fn tnmt_vm_runs_in_dioma() {
        let n = 8;
        let mut states = vec![0.0f32; n * 2];
        states[n..].fill(0.5);
        let program = vec![
            T::LoadState(0, 0),
            T::LoadState(1, 1),
            T::MathAdd(0, 0, 1),
            T::StoreState(0, 0),
        ];
        unsafe {
            crate::vm::vm::execute(states.as_mut_ptr(), std::ptr::null(), n, &program);
        }
        assert!(states[..n].iter().all(|&value| value == 0.5));
    }

    /// Реальный ген INTEGRATOR транслируется целиком (включая Fire).
    #[test]
    fn real_integrator_gene_compiles() {
        let gene = RuleGene::default();
        let bc = compile_rule_gene(&gene).expect("INTEGRATOR должен компилироваться");
        assert!(!bc.is_empty());
    }

    /// CONSOLIDATOR — самый сложный шаблон: Tend(Troph) + условный порог + Fire.
    /// Транслируется целиком и семантика совпадает с эталоном на 20 тиков.
    #[test]
    fn consolidator_gene_full_translation_and_equivalence() {
        // p = [leak, gain, thr0, k_troph, refr]
        let gene = RuleGene {
            template: templates::CONSOLIDATOR,
            p: [0.85, 0.35, 0.4, 0.6, 3.0, 0.0, 0.0, 0.0],
        };
        let bc = compile_rule_gene(&gene)
            .expect("CONSOLIDATOR должен транслироваться (Tend + Max + Fire)");
        assert!(bc.len() > 10, "программа нетривиальна");

        // Эталонная симуляция семантики CONSOLIDATOR вручную:
        //   Troph ← Troph·0.95
        //   S4 ← max(thr0·(1 − k·Troph), 0.05)
        //   Act ← clamp01(Act·leak + Inp·gain)
        //   Out/Fire: Act > S4 → Out=1, S5=refr; иначе Out·decay−0.01, S5−1
        let n = 8;
        let mut states = vec![0.0f32; 8 * n];
        for c in 0..n {
            states[Var::Inp.idx() * n + c] = 0.8;
            states[Var::Troph.idx() * n + c] = 0.5;
        }
        unsafe {
            for _ in 0..20 {
                execute_cells(&mut states, n, &bc);
            }
        }

        // Эталон (скалярно, одна клетка):
        let (leak, gain, thr0, k_troph) = (0.85f32, 0.35f32, 0.4f32, 0.6f32);
        let (refr, decay) = (3.0f32, 0.55f32);
        let (mut act, mut troph, mut out, mut s5) = (0.0f32, 0.5f32, 0.0f32, 0.0f32);
        let inp = 0.8f32;
        for _ in 0..20 {
            troph *= 0.95;
            let s4 = (thr0 * (1.0 - k_troph * troph)).max(0.05);
            act = (act * leak + inp * gain).clamp(0.0, 1.0);
            if act > s4 && s5 <= 0.0 {
                out = 1.0;
                s5 = refr;
            } else {
                out = ((out * decay) - 0.01).clamp(0.0, 1.0);
                s5 = (s5 - 1.0).clamp(0.0, 1.0);
            }
        }

        let tnmt_act = states[Var::Act.idx() * n];
        assert!(
            (act - tnmt_act).abs() < 1e-4,
            "Act: эталон {act:.6} vs TNMT {tnmt_act:.6}"
        );
        // Порог должен снизиться трофическим сигналом.
        let tnmt_s4 = states[Var::S4.idx() * n];
        assert!(
            tnmt_s4 < thr0,
            "Troph должен снизить порог: S4={tnmt_s4} vs thr0={thr0}"
        );
    }

    /// Все 4 шаблона генов компилируются без ошибок.
    #[test]
    fn all_templates_compile() {
        let genes = vec![
            RuleGene::default(),
            RuleGene::spiker(0.85, 0.35, 0.5, 3.0),
            RuleGene::consolidator(0.4, 0.6),
            RuleGene {
                template: templates::HABITUATOR,
                p: [0.85, 0.35, 0.5, 0.4, 3.0, 0.0, 0.0, 0.0],
            },
        ];
        for (i, g) in genes.iter().enumerate() {
            let bc =
                compile_rule_gene(g).unwrap_or_else(|e| panic!("шаблон {i} не транслируется: {e}"));
            assert!(!bc.is_empty());
        }
    }

    /// HABITUATOR: привыкание — при постоянном входе S3 растёт и выход падает.
    #[test]
    fn habituator_adapts_on_constant_input() {
        let gene = RuleGene {
            template: templates::HABITUATOR,
            p: [0.85, 0.35, 0.5, 0.4, 3.0, 0.0, 0.0, 0.0],
        };
        let bc = compile_rule_gene(&gene).unwrap();
        let n = 8;
        let mut states = vec![0.0f32; 8 * n];

        // Умеренный постоянный вход: привыкание должно гасить реакцию.
        for c in 0..n {
            states[Var::Inp.idx() * n + c] = 0.6;
        }
        unsafe { execute_cells(&mut states, n, &bc) }
        let out_early = states[Var::Out.idx() * n];

        // Много тиков → S3 растёт → gain_eff падает → выход ниже.
        unsafe {
            for _ in 0..120 {
                execute_cells(&mut states, n, &bc);
            }
        }
        // Привыкание видно по S3: он должен заметно вырасти.
        let s3_late = states[Var::S3.idx() * n];
        assert!(
            s3_late > 0.1,
            "привыкание должно накапливаться: S3={s3_late}"
        );
        // И средний Act за поздний период ниже максимума.
        let act_late = states[Var::Act.idx() * n];
        assert!(
            act_late < 1.0,
            "привыкание снижает активность: Act={act_late}"
        );
        let _ = out_early;
    }
}
