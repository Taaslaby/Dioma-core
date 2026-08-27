//! -------------------------------------------------------------------------
//! Filename:    src/script/mod.rs
//! Class:       scripting
//! Purpose:     CAOS-подобная скрипт-машина (порт a1-script)
//! Description: слои: лексер → компилятор (префиксные команды фиксированной
//!               арности) → машина с блокировками `wait`. Команды подключаются
//!               хостом через registry::CommandSet + registry::Services —
//!               ядро про агентов ничего не знает.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
pub mod compiler;
pub mod lexer;
pub mod machine;
pub mod registry;
pub mod value;

pub use compiler::{compile, Op, Program, Scope};
pub use machine::{Flow, GVars, Machine};
pub use registry::{CommandDef, CommandSet, Ctx, Services};
pub use value::Value;
