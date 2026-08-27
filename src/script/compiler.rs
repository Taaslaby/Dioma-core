//! Компилятор (порт a1-script): токены → байткод. Как в CAOS всё префиксно:
//! у каждой команды фиксированная арность, поэтому разбор линейный, без
//! приоритетов операций.
//!
//! Управление: `doif … [else …] endi`, `repn <n> … repe`, `wait <тики>`,
//! `stop`. Переменные: локальные `va0..va99`, глобальные мира `vg0..vg99`.

use crate::script::lexer::{lex, Tok};
use crate::script::registry::CommandSet;
use crate::script::value::Value;

/// Область переменных.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scope {
    Local,
    Global,
}

#[derive(Debug, Clone)]
pub enum Op {
    Const(Value),
    /// Вызов команды из реестра (арность берётся из определения).
    Cmd(u32),
    Load(Scope, u8),
    Store(Scope, u8),
    /// Снять условие; ложь → прыжок вперёд.
    DoIf(usize),
    /// Ложная ветка прыгает за `endi`.
    Else(usize),
    /// Снять число повторов; ноль → прыжок за `repe`.
    LoopBegin(usize),
    /// Декрементировать счётчик кадра; не исчерпан → назад к телу цикла.
    LoopEnd { back: usize },
    /// Блокировка: снять тики и приостановить скрипт.
    Wait,
    Stop,
}

#[derive(Debug, Clone, Default)]
pub struct Program {
    pub ops: Vec<Op>,
}

pub fn compile(src: &str, cmds: &CommandSet) -> Result<Program, String> {
    let toks = lex(src)?;
    let mut c = Compiler {
        toks: &toks,
        pos: 0,
        prog: Program::default(),
        jumps: Vec::new(),
        loops: Vec::new(),
        cmds,
    };
    c.block()?;
    if !c.jumps.is_empty() {
        return Err("не закрыт doif".into());
    }
    if !c.loops.is_empty() {
        return Err("не закрыт repn".into());
    }
    Ok(c.prog)
}

struct Compiler<'a> {
    toks: &'a [Tok],
    pos: usize,
    prog: Program,
    /// Незакрытые точки патча: 'd' = DoIf, 'e' = Else, 'l' = LoopBegin-skip.
    jumps: Vec<(char, usize)>,
    loops: Vec<usize>,
    cmds: &'a CommandSet,
}

impl<'a> Compiler<'a> {
    fn block(&mut self) -> Result<(), String> {
        while self.pos < self.toks.len() {
            let name = match &self.toks[self.pos] {
                Tok::Ident(s) => s.clone(),
                t => return Err(format!("ожидается команда, найдено {t:?}")),
            };
            self.pos += 1;
            match name.as_str() {
                "doif" => {
                    self.expr(&name)?;
                    let at = self.prog.ops.len();
                    self.prog.ops.push(Op::DoIf(0));
                    self.jumps.push(('d', at));
                }
                "else" => {
                    let doif_at = match self.jumps.last() {
                        Some(('d', at)) => *at,
                        _ => return Err("else без doif".into()),
                    };
                    // запись 'd' снята ниже: её переход уже ведёт на ветку else
                    self.jumps.pop();
                    let at = self.prog.ops.len();
                    self.prog.ops.push(Op::Else(0));
                    // ложный путь приходит на начало ветки else (за оп-код Else)
                    self.patch(doif_at, at + 1);
                    self.jumps.push(('e', at));
                }
                "endi" => {
                    let here = self.prog.ops.len();
                    match self.jumps.pop() {
                        // без else: ложный путь прыгает за endi
                        Some(('d', at)) => self.patch(at, here),
                        // с else: сам else прыгает за endi
                        Some(('e', at)) => self.patch(at, here),
                        _ => return Err("endi без doif".into()),
                    }
                }
                "repn" => {
                    self.expr(&name)?;
                    let at = self.prog.ops.len();
                    self.prog.ops.push(Op::LoopBegin(0));
                    self.loops.push(at);
                    self.jumps.push(('l', at));
                }
                "repe" => {
                    let begin = self.loops.pop().ok_or("repe без repn")?;
                    // патчим пропуск пустого цикла
                    match self.jumps.pop() {
                        Some(('l', at)) => {
                            debug_assert_eq!(at, begin);
                            let here = self.prog.ops.len();
                            self.patch(at, here);
                        }
                        _ => return Err("repe без repn".into()),
                    }
                    self.prog.ops.push(Op::LoopEnd { back: begin + 1 });
                }
                "wait" => {
                    self.expr(&name)?;
                    self.prog.ops.push(Op::Wait);
                }
                "stop" => self.prog.ops.push(Op::Stop),
                _ if name == "setv" => {
                    let (scope, n) = self.var_ref()?;
                    self.args(&name)?;
                    self.prog
                        .ops
                        .push(match scope {
                            Scope::Local => Op::Store(Scope::Local, n),
                            Scope::Global => Op::Store(Scope::Global, n),
                        });
                }
                _ => {
                    self.args(&name)?;
                    let idx = self.cmds.index_of(&name).ok_or_else(|| {
                        format!("неизвестная команда {name:?}")
                    })?;
                    self.prog.ops.push(Op::Cmd(idx));
                }
            }
        }
        Ok(())
    }

    /// Разбирает аргументы команды по арности из реестра.
    fn args(&mut self, name: &str) -> Result<(), String> {
        let arity = if name == "doif" || name == "repn" || name == "wait" || name == "setv" {
            1 // значение; переменная setv уже разобрана отдельно
        } else {
            self.cmds
                .arity_of(name)
                .ok_or_else(|| format!("неизвестная команда {name:?}"))?
        };
        for _ in 0..arity {
            self.expr(name)?;
        }
        Ok(())
    }

    fn expr(&mut self, caller: &str) -> Result<(), String> {
        let t = self.toks.get(self.pos).ok_or_else(|| {
            format!("{caller}: неожиданный конец в выражении")
        })?;
        match t {
            Tok::Int(i) => {
                self.prog.ops.push(Op::Const(Value::Int(*i)));
                self.pos += 1;
            }
            Tok::Float(f) => {
                self.prog.ops.push(Op::Const(Value::Float(*f)));
                self.pos += 1;
            }
            Tok::Str(s) => {
                self.prog.ops.push(Op::Const(Value::Str(s.clone())));
                self.pos += 1;
            }
            Tok::Ident(name) => {
                let name = name.clone();
                self.pos += 1;
                if let Some((scope, n)) = parse_var(&name) {
                    self.prog.ops.push(Op::Load(scope, n));
                    return Ok(());
                }
                if name == "setv" {
                    return Err("setv нельзя использовать как значение".into());
                }
                let arity = self.cmds.arity_of(&name).ok_or_else(|| {
                    format!("неизвестная команда {name:?}")
                })?;
                for _ in 0..arity {
                    self.expr(&name)?;
                }
                let idx = self.cmds.index_of(&name).unwrap();
                self.prog.ops.push(Op::Cmd(idx));
            }
        }
        Ok(())
    }

    /// Читает имя переменной после setv.
    fn var_ref(&mut self) -> Result<(Scope, u8), String> {
        let t = self
            .toks
            .get(self.pos)
            .ok_or("setv: нет имени переменной")?;
        self.pos += 1;
        match t {
            Tok::Ident(n) => parse_var(n)
                .ok_or_else(|| format!("setv: {n:?} не переменная (va0..99 / vg0..99)")),
            other => Err(format!("setv: {other:?} не переменная")),
        }
    }

    fn patch(&mut self, at: usize, target: usize) {
        match &mut self.prog.ops[at] {
            Op::DoIf(t) | Op::Else(t) | Op::LoopBegin(t) => *t = target,
            _ => unreachable!("патч не-перехода"),
        }
    }
}

/// `va7` / `vg12` → область + индекс.
fn parse_var(name: &str) -> Option<(Scope, u8)> {
    let (prefix, rest) = name.split_at(2);
    let scope = match prefix {
        "va" => Scope::Local,
        "vg" => Scope::Global,
        _ => return None,
    };
    let n: u8 = rest.parse().ok()?;
    (n < 100).then_some((scope, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script::registry::CommandSet;

    #[test]
    fn vars() {
        assert_eq!(parse_var("va13"), Some((Scope::Local, 13)));
        assert_eq!(parse_var("vg0"), Some((Scope::Global, 0)));
        assert_eq!(parse_var("vx0"), None);
        assert_eq!(parse_var("va100"), None);
    }

    #[test]
    fn compile_errors() {
        let cs = CommandSet::builder().finish();
        assert!(compile("nosuchcmd", &cs).is_err());
        assert!(compile("doif 1 out \"x\"", &cs).is_err()); // нет endi
        assert!(compile("endi", &cs).is_err());
        assert!(compile("repe", &cs).is_err());
        assert!(compile("setv zz 1", &cs).is_err());
    }
}
