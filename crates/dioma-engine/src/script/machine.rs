//! Исполнитель байткода (порт a1-script). Один `Machine` — один живой
//! скрипт-поток агента (аналог MacroScript из C2e): умеет приостанавливаться
//! на `wait`.
//!
//! Программу компилирует хост один раз и хранит у агента; машина держит
//! только состояние исполнения (pc, стек, переменные).

use crate::script::compiler::{compile, Op, Program, Scope};
use crate::script::registry::{CommandSet, Ctx};
use crate::script::value::Value;

const NVARS: usize = 100;

/// Глобальные переменные мира (`vg0..vg99`) — общие для всех машин.
#[derive(Debug, Clone)]
pub struct GVars(pub Vec<Value>);

impl Default for GVars {
    fn default() -> Self {
        GVars(vec![Value::Nul; NVARS])
    }
}

#[derive(Debug, Clone)]
struct LoopFrame {
    remaining: u64,
}

/// Чем кончился прогон машины.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Flow {
    /// Скрипт дошёл до конца или `stop`.
    Halted,
    /// Встречен `wait`: продолжение после кулдауна.
    Blocked,
}

#[derive(Debug, Clone)]
pub struct Machine {
    pc: usize,
    stack: Vec<Value>,
    locals: Vec<Value>,
    loops: Vec<LoopFrame>,
    cooldown: u64,
    halted: bool,
}

impl Default for Machine {
    fn default() -> Self {
        Self::new()
    }
}

impl Machine {
    pub fn new() -> Self {
        Machine {
            pc: 0,
            stack: Vec::new(),
            locals: vec![Value::Nul; NVARS],
            loops: Vec::new(),
            cooldown: 0,
            halted: true,
        }
    }

    /// Готовит машину к новому запуску программы.
    pub fn start(&mut self) {
        self.pc = 0;
        self.stack.clear();
        self.loops.clear();
        self.cooldown = 0;
        self.halted = false;
    }

    pub fn is_halted(&self) -> bool {
        self.halted
    }

    pub fn is_cooling(&self) -> bool {
        self.cooldown > 0
    }

    /// Тик кулдауна после `wait`; true — пора снова звать `run`.
    pub fn tick_cooldown(&mut self) -> bool {
        if self.cooldown == 0 {
            return false;
        }
        self.cooldown -= 1;
        self.cooldown == 0
    }

    /// Исполняет программу до конца, блокировки или ошибки.
    /// Перед первым вызовом нужно позвать [`Machine::start`].
    pub fn run(
        &mut self,
        prog: &Program,
        cmds: &CommandSet,
        gvars: &mut GVars,
        ctx: &mut Ctx<'_>,
    ) -> Result<Flow, String> {
        while !self.halted && self.pc < prog.ops.len() {
            match &prog.ops[self.pc] {
                Op::Const(v) => {
                    self.stack.push(v.clone());
                    self.pc += 1;
                }
                Op::Cmd(idx) => {
                    let def = cmds.get(*idx);
                    let argc = def.args as usize;
                    let at = self
                        .stack
                        .len()
                        .checked_sub(argc)
                        .ok_or_else(|| format!("стек недобран под {}", def.name))?;
                    let args = self.stack.split_off(at);
                    let v = (def.f)(ctx, &args)?;
                    self.stack.push(v);
                    self.pc += 1;
                }
                Op::Load(scope, n) => {
                    let v = match scope {
                        Scope::Local => self.locals[*n as usize].clone(),
                        Scope::Global => gvars.0[*n as usize].clone(),
                    };
                    self.stack.push(v);
                    self.pc += 1;
                }
                Op::Store(scope, n) => {
                    let v = self.stack.pop().unwrap_or(Value::Nul);
                    match scope {
                        Scope::Local => self.locals[*n as usize] = v,
                        Scope::Global => gvars.0[*n as usize] = v,
                    }
                    self.pc += 1;
                }
                Op::DoIf(t) => {
                    let c = self.stack.pop().unwrap_or(Value::Nul);
                    self.pc = if c.truthy() { self.pc + 1 } else { *t };
                }
                Op::Else(t) => self.pc = *t,
                Op::LoopBegin(skip) => {
                    let n = self.stack.pop().unwrap_or(Value::Nul).as_int()?;
                    if n <= 0 {
                        self.pc = *skip;
                    } else {
                        self.loops.push(LoopFrame {
                            remaining: n as u64,
                        });
                        self.pc += 1;
                    }
                }
                Op::LoopEnd { back } => {
                    let again = match self.loops.last_mut() {
                        Some(f) => {
                            f.remaining -= 1;
                            f.remaining > 0
                        }
                        None => return Err("LoopEnd без кадра цикла".into()),
                    };
                    self.pc = if again {
                        *back
                    } else {
                        self.loops.pop();
                        self.pc + 1
                    };
                }
                Op::Wait => {
                    let t = self.stack.pop().unwrap_or(Value::Nul).as_int()?;
                    self.pc += 1;
                    if t > 0 {
                        self.cooldown = t as u64;
                        return Ok(Flow::Blocked);
                    }
                }
                Op::Stop => self.halted = true,
            }
        }
        self.halted = true;
        Ok(Flow::Halted)
    }

    /// Компилирует исходник и сразу запускает с нуля — для одноразовых скриптов.
    pub fn run_source(
        src: &str,
        cmds: &CommandSet,
        gvars: &mut GVars,
        ctx: &mut Ctx<'_>,
    ) -> Result<Flow, String> {
        let prog = compile(src, cmds)?;
        let mut m = Machine::new();
        m.start();
        m.run(&prog, cmds, gvars, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::script::registry::Services;
    use std::cell::RefCell;

    /// Хост, собирающий вывод `out` в вектор.
    struct LogHost(RefCell<Vec<String>>);
    impl Services for LogHost {
        fn entity_pos(&mut self, _id: u64) -> Option<(f32, f32)> {
            None
        }
        fn entity_move(&mut self, _id: u64, _dx: f32, _dy: f32) {}
        fn entity_teleport(&mut self, _id: u64, _x: f32, _y: f32) {}
        fn entity_kill(&mut self, _id: u64) {}
        fn entity_class(&mut self, _id: u64) -> Option<(u8, u8)> {
            None
        }
        fn find_nearest(&mut self, _c: (u8, u8), _near: u64) -> Option<u64> {
            None
        }
        fn send_message(&mut self, _f: u64, _t: u64, _s: u8, _p: i64) {}
        fn rand_range(&mut self, lo: i64, _hi: i64) -> i64 {
            lo
        }
        fn log(&mut self, msg: String) {
            self.0.borrow_mut().push(msg);
        }
    }

    fn ctx_with<'a>(svc: &'a mut dyn Services) -> Ctx<'a> {
        Ctx {
            target: 7,
            owner: 7,
            from: 0,
            param: 0,
            services: svc,
        }
    }

    #[test]
    fn arithmetic_and_vars() {
        let cs = CommandSet::builder().with_builtins().finish();
        let mut gv = GVars::default();
        let mut svc = LogHost(RefCell::new(vec![]));
        let out = Machine::run_source(
            "setv va0 mul add 2 3 4 setv vg1 add va0 10 out vg1",
            &cs,
            &mut gv,
            &mut ctx_with(&mut svc),
        )
        .unwrap();
        assert_eq!(out, Flow::Halted);
        assert_eq!(gv.0[1], Value::Int(30));
    }

    #[test]
    fn doif_else() {
        let cs = CommandSet::builder().with_builtins().finish();
        let mut gv = GVars::default();

        // ложная ветка
        let mut svc = LogHost(RefCell::new(vec![]));
        Machine::run_source(
            "doif gt 1 2 out \"да\" else out \"нет\" endi",
            &cs,
            &mut gv,
            &mut ctx_with(&mut svc),
        )
        .unwrap();
        assert_eq!(svc.0.into_inner(), vec!["нет".to_string()]);

        // истинная ветка
        let mut svc = LogHost(RefCell::new(vec![]));
        Machine::run_source(
            "doif gt 5 2 out \"да\" else out \"нет\" endi",
            &cs,
            &mut gv,
            &mut ctx_with(&mut svc),
        )
        .unwrap();
        assert_eq!(svc.0.into_inner(), vec!["да".to_string()]);
    }

    #[test]
    fn nested_doif_and_loop() {
        let cs = CommandSet::builder().with_builtins().finish();
        let mut gv = GVars::default();
        let mut svc = LogHost(RefCell::new(vec![]));
        Machine::run_source(
            "setv va9 0 repn 3 setv va9 add va9 2 repe \
             doif eq va9 6 repn 2 out \"к\" repe endi",
            &cs,
            &mut gv,
            &mut ctx_with(&mut svc),
        )
        .unwrap();
        assert_eq!(svc.0.into_inner(), vec!["к".to_string(), "к".to_string()]);
    }

    #[test]
    fn wait_blocks_until_cooldown() {
        let cs = CommandSet::builder().with_builtins().finish();
        let mut gv = GVars::default();
        let prog = compile("out \"a\" wait 3 out \"b\"", &cs).unwrap();

        let mut svc = LogHost(RefCell::new(vec![]));
        let mut m = Machine::new();
        m.start();
        assert_eq!(
            m.run(&prog, &cs, &mut gv, &mut ctx_with(&mut svc)).unwrap(),
            Flow::Blocked
        );
        assert_eq!(svc.0.into_inner(), vec!["a".to_string()]);

        for _ in 0..2 {
            assert!(!m.tick_cooldown());
            // пока охлаждается — не запускаем
        }
        assert!(m.tick_cooldown());
        let mut svc = LogHost(RefCell::new(vec![]));
        assert_eq!(
            m.run(&prog, &cs, &mut gv, &mut ctx_with(&mut svc)).unwrap(),
            Flow::Halted
        );
        assert_eq!(svc.0.into_inner(), vec!["b".to_string()]);
    }

    #[test]
    fn globals_shared_between_machines() {
        let cs = CommandSet::builder().with_builtins().finish();
        let mut gv = GVars::default();
        let mut svc = ();
        Machine::run_source("setv vg42 77", &cs, &mut gv, &mut ctx_with(&mut svc)).unwrap();
        let mut svc = LogHost(RefCell::new(vec![]));
        Machine::run_source("out vg42", &cs, &mut gv, &mut ctx_with(&mut svc)).unwrap();
        assert_eq!(svc.0.into_inner(), vec!["77".to_string()]);
    }

    #[test]
    fn stack_underflow_is_error_not_panic() {
        let cs = CommandSet::builder().with_builtins().finish();
        let mut gv = GVars::default();
        let mut svc = ();
        // add ждёт два аргумента, получает ноль
        assert!(Machine::run_source("add", &cs, &mut gv, &mut ctx_with(&mut svc)).is_err());
    }
}
