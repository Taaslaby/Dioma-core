//! Реестр команд и хост-сервисы (порт a1-script). Ядро ВМ ничего не знает о
//! мире: команды получают только `Ctx` с сырыми id и ссылку на `Services`,
//! который реализует верхний слой (агенты или приложение).

use crate::logic::script::value::Value;
use std::collections::HashMap;

/// Что команда может попросить у мира. Реализуется хостом.
pub trait Services {
    fn entity_pos(&mut self, id: u64) -> Option<(f32, f32)>;
    fn entity_move(&mut self, id: u64, dx: f32, dy: f32);
    fn entity_teleport(&mut self, id: u64, x: f32, y: f32);
    fn entity_kill(&mut self, id: u64);
    /// Классификация агента: `(family, genus)`.
    fn entity_class(&mut self, id: u64) -> Option<(u8, u8)>;
    /// Ближайший живой агент класса `(family, genus)` к агенту `near`.
    fn find_nearest(&mut self, class: (u8, u8), near: u64) -> Option<u64>;
    /// Положить сообщение в очередь цели.
    fn send_message(&mut self, from: u64, to: u64, slot: u8, param: i64);
    /// Детерминированный рандом хоста: целое `[lo, hi]`.
    fn rand_range(&mut self, lo: i64, hi: i64) -> i64;
    fn log(&mut self, msg: String);
}

/// Заглушка для чистых тестов ВМ.
impl Services for () {
    fn entity_pos(&mut self, _id: u64) -> Option<(f32, f32)> {
        None
    }
    fn entity_move(&mut self, _id: u64, _dx: f32, _dy: f32) {}
    fn entity_teleport(&mut self, _id: u64, _x: f32, _y: f32) {}
    fn entity_kill(&mut self, _id: u64) {}
    fn entity_class(&mut self, _id: u64) -> Option<(u8, u8)> {
        None
    }
    fn find_nearest(&mut self, _class: (u8, u8), _near: u64) -> Option<u64> {
        None
    }
    fn send_message(&mut self, _from: u64, _to: u64, _slot: u8, _param: i64) {}
    fn rand_range(&mut self, lo: i64, _hi: i64) -> i64 {
        lo
    }
    fn log(&mut self, _msg: String) {}
}

/// Контекст вызова команды.
pub struct Ctx<'a> {
    /// Агент, чей скрипт исполняется (TARG в терминах CAOS).
    pub target: u64,
    /// Владелец скрипта — не меняется `settarg`'ом (OWNR).
    pub owner: u64,
    /// Кто вызвал/прислал сообщение (FROM).
    pub from: u64,
    /// Параметр события, запустившего скрипт.
    pub param: i64,
    pub services: &'a mut dyn Services,
}

type CmdFn = fn(&mut Ctx<'_>, &[Value]) -> Result<Value, String>;

#[derive(Clone)]
pub struct CommandDef {
    pub name: &'static str,
    pub args: u8,
    pub f: CmdFn,
}

/// Таблица команд. Компилятор ищет арность по имени, ВМ вызывает по индексу.
#[derive(Clone, Default)]
pub struct CommandSet {
    defs: Vec<CommandDef>,
    index: HashMap<String, u32>,
}

impl CommandSet {
    pub fn builder() -> Builder {
        Builder::default()
    }

    pub fn arity_of(&self, name: &str) -> Option<u8> {
        self.index.get(name).map(|&i| self.defs[i as usize].args)
    }

    pub fn index_of(&self, name: &str) -> Option<u32> {
        self.index.get(name).copied()
    }

    pub fn get(&self, idx: u32) -> &CommandDef {
        &self.defs[idx as usize]
    }
}

#[derive(Default)]
pub struct Builder {
    set: CommandSet,
}

impl Builder {
    /// Базовая арифметика/сравнения/вывод — удобно иметь всегда.
    pub fn with_builtins(mut self) -> Self {
        for d in builtins() {
            self = self.cmd(d.name, d.args, d.f);
        }
        self
    }

    pub fn cmd(mut self, name: &'static str, args: u8, f: CmdFn) -> Self {
        assert!(
            !self.set.index.contains_key(name),
            "команда {name:?} уже есть"
        );
        let idx = self.set.defs.len() as u32;
        self.set.defs.push(CommandDef { name, args, f });
        self.set.index.insert(name.to_string(), idx);
        self
    }

    pub fn finish(self) -> CommandSet {
        self.set
    }
}

fn num2(a: &[Value]) -> Result<(f64, f64), String> {
    Ok((a[0].as_num()?, a[1].as_num()?))
}

fn cmp_values(a: &[Value], ord: fn(std::cmp::Ordering) -> bool) -> Result<Value, String> {
    use std::cmp::Ordering;
    let o = match (&a[0], &a[1]) {
        (Value::Str(x), Value::Str(y)) => x.cmp(y),
        _ => {
            let (x, y) = num2(a)?;
            x.partial_cmp(&y).unwrap_or(Ordering::Equal)
        }
    };
    Ok(Value::Int(ord(o) as i64))
}

/// Встроенные команды: математика, сравнения, строки, лог, цель скрипта.
pub fn builtins() -> Vec<CommandDef> {
    vec![
        CommandDef {
            name: "add",
            args: 2,
            f: |_, a| Ok(Value::num(num2(a)?.0 + num2(a)?.1)),
        },
        CommandDef {
            name: "sub",
            args: 2,
            f: |_, a| Ok(Value::num(num2(a)?.0 - num2(a)?.1)),
        },
        CommandDef {
            name: "mul",
            args: 2,
            f: |_, a| Ok(Value::num(num2(a)?.0 * num2(a)?.1)),
        },
        CommandDef {
            name: "div",
            args: 2,
            f: |_, a| {
                let (x, y) = num2(a)?;
                if y == 0.0 {
                    Err("деление на ноль".into())
                } else {
                    Ok(Value::num(x / y))
                }
            },
        },
        CommandDef {
            name: "mod",
            args: 2,
            f: |_, a| {
                let (x, y) = num2(a)?;
                if y == 0.0 {
                    Err("деление на ноль".into())
                } else {
                    Ok(Value::num(x % y))
                }
            },
        },
        CommandDef {
            name: "min",
            args: 2,
            f: |_, a| Ok(Value::num(num2(a)?.0.min(num2(a)?.1))),
        },
        CommandDef {
            name: "max",
            args: 2,
            f: |_, a| Ok(Value::num(num2(a)?.0.max(num2(a)?.1))),
        },
        CommandDef {
            name: "abs",
            args: 1,
            f: |_, a| Ok(Value::num(a[0].as_num()?.abs())),
        },
        CommandDef {
            name: "neg",
            args: 1,
            f: |_, a| Ok(Value::num(-a[0].as_num()?)),
        },
        CommandDef {
            name: "eq",
            args: 2,
            f: |_, a| cmp_values(a, |o| o == std::cmp::Ordering::Equal),
        },
        CommandDef {
            name: "ne",
            args: 2,
            f: |_, a| cmp_values(a, |o| o != std::cmp::Ordering::Equal),
        },
        CommandDef {
            name: "lt",
            args: 2,
            f: |_, a| cmp_values(a, |o| o == std::cmp::Ordering::Less),
        },
        CommandDef {
            name: "gt",
            args: 2,
            f: |_, a| cmp_values(a, |o| o == std::cmp::Ordering::Greater),
        },
        CommandDef {
            name: "le",
            args: 2,
            f: |_, a| cmp_values(a, |o| o != std::cmp::Ordering::Greater),
        },
        CommandDef {
            name: "ge",
            args: 2,
            f: |_, a| cmp_values(a, |o| o != std::cmp::Ordering::Less),
        },
        CommandDef {
            name: "and",
            args: 2,
            f: |_, a| Ok(Value::Int((a[0].truthy() && a[1].truthy()) as i64)),
        },
        CommandDef {
            name: "or",
            args: 2,
            f: |_, a| Ok(Value::Int((a[0].truthy() || a[1].truthy()) as i64)),
        },
        CommandDef {
            name: "not",
            args: 1,
            f: |_, a| Ok(Value::Int(!a[0].truthy() as i64)),
        },
        CommandDef {
            name: "cat",
            args: 2,
            f: |_, a| {
                Ok(Value::Str(format!(
                    "{}{}",
                    scalar_text(&a[0]),
                    scalar_text(&a[1])
                )))
            },
        },
        CommandDef {
            name: "out",
            args: 1,
            f: |c, a| {
                c.services.log(scalar_text(&a[0]));
                Ok(Value::Nul)
            },
        },
        // TARG/FROM/OWNR как в CAOS: цель скрипта можно менять на лету
        CommandDef {
            name: "targ",
            args: 0,
            f: |c, _| Ok(Value::Entity(c.target)),
        },
        CommandDef {
            name: "ownr",
            args: 0,
            f: |c, _| Ok(Value::Entity(c.owner)),
        },
        // параметр текущего события
        CommandDef {
            name: "param",
            args: 0,
            f: |c, _| Ok(Value::Int(c.param)),
        },
        CommandDef {
            name: "from",
            args: 0,
            f: |c, _| Ok(Value::Entity(c.from)),
        },
        CommandDef {
            name: "settarg",
            args: 1,
            f: |c, a| {
                c.target = a[0].as_entity()?;
                Ok(Value::Nul)
            },
        },
    ]
}

fn scalar_text(v: &Value) -> String {
    match v {
        Value::Str(s) => s.clone(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => format!("{f:.3}").trim_end_matches('0').to_string(),
        Value::Entity(id) => format!("#{id}"),
        Value::Nul => "null".into(),
    }
}
