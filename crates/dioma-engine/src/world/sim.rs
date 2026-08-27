//! -------------------------------------------------------------------------
//! Filename:    src/world/sim.rs
//! Purpose:     world:map owner, agents, events
//! Description:
//!
//!
//! Advances the simulation and provides engine commands to scripts.
//! Borrowing scheme: mutable world state resides in the private `Inner` state;
//! before execution,
//! //! the script is **extracted** from the agent's slot—this allows a command, via services,
//! //! to access the entire world, including the agent currently executing it.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------

use crate::env::map::WorldMap;
use crate::env::math::Vec2;
use crate::env::{AgentId, Tick};
use crate::rng::XorShift;
use crate::script::{CommandDef, CommandSet, Ctx, Flow, GVars, Machine, Services};
use crate::world::agent::{slot, Agent, Cls, Message, Script};
use std::collections::HashMap;

/// Отложенное сообщение.
struct Pending {
    due: Tick,
    to: AgentId,
    msg: Message,
}

/// Всё мутируемое состояние мира, кроме карты/реестра команд/gvars —
/// вынесено, чтобы сервисы команд могли взять это одним &mut.
struct Inner {
    agents: HashMap<AgentId, Agent>,
    /// Порядок обхода = возрастание id, детерминированно.
    order: Vec<AgentId>,
    pending: Vec<Pending>,
    rng: XorShift,
    time: Tick,
    next_id: u64,
    /// Журнал событий такта: вывод `out`, смерти, ошибки скриптов.
    log: Vec<String>,
}

pub struct World {
    pub map: WorldMap,
    pub cmds: CommandSet,
    gvars: GVars,
    inner: Inner,
}

impl World {
    pub fn new(seed: u64, map: WorldMap) -> Self {
        let mut b = CommandSet::builder().with_builtins();
        for d in engine_commands() {
            b = b.cmd(d.name, d.args, d.f);
        }
        World {
            map,
            cmds: b.finish(),
            gvars: GVars::default(),
            inner: Inner {
                agents: HashMap::new(),
                order: Vec::new(),
                pending: Vec::new(),
                rng: XorShift::new(seed),
                time: Tick(0),
                next_id: 1,
                log: Vec::new(),
            },
        }
    }

    // ── создание и настройка ────────────────────────────────────────────

    pub fn spawn(&mut self, cls: Cls, pos: Vec2, sprite: &str) -> AgentId {
        let id = AgentId(self.inner.next_id);
        self.inner.next_id += 1;
        self.inner
            .agents
            .insert(id, Agent::new(id, cls, pos, sprite));
        self.inner.order.push(id);
        id
    }

    /// Компилирует исходник и прикрепляет программу к слоту агента.
    /// Машина запускается заново для каждого нового события этого слота.
    pub fn attach_script(&mut self, id: AgentId, slot_idx: u8, src: &str) -> Result<(), String> {
        let prog = crate::script::compile(src, &self.cmds)?;
        let ag = self
            .inner
            .agents
            .get_mut(&id)
            .ok_or_else(|| format!("агент {id:?} не найден"))?;
        ag.scripts[slot_idx as usize] = Some(Script {
            prog,
            machine: Machine::new(),
        });
        Ok(())
    }

    pub fn set_timer(&mut self, id: AgentId, period: u32) -> Result<(), String> {
        self.inner
            .agents
            .get_mut(&id)
            .ok_or_else(|| format!("агент {id:?} не найден"))?
            .timer_period = period;
        Ok(())
    }

    // ── сообщения ───────────────────────────────────────────────────────

    /// Событие «тронули»: кладёт Activate1 в ящик (сработает на след. такте).
    pub fn activate(&mut self, id: AgentId) {
        if let Some(ag) = self.inner.agents.get_mut(&id) {
            ag.receive(Message {
                from: AgentId::NONE,
                slot: slot::ACTIVATE1,
                param: 0,
            });
        }
    }

    pub fn now(&self) -> Tick {
        self.inner.time
    }

    /// Сообщение с задержкой в один тик (как в C2e).
    pub fn send(&mut self, from: AgentId, to: AgentId, slot_idx: u8, param: i64) {
        if to.is_none() || to == from || !self.inner.agents.contains_key(&to) {
            return;
        }
        let due = Tick(self.inner.time.0 + 1);
        self.inner.pending.push(Pending {
            due,
            to,
            msg: Message {
                from,
                slot: slot_idx,
                param,
            },
        });
    }

    // ── главный цикл ────────────────────────────────────────────────────

    /// Продвигает объектный мир на один тик.
    ///
    /// Сначала обновляется химия карты, затем доставляются отложенные
    /// сообщения, исполняются скрипты и таймеры. Физика выполняется последней,
    /// чтобы скрипты текущего тика видели согласованное состояние мира.
    pub fn step(&mut self) {
        self.inner.time.0 += 1;
        self.inner.log.clear();
        self.map.step();

        // 1. отложенные сообщения этого такта — в почтовые ящики
        let due: Vec<(AgentId, Message)> = self
            .inner
            .pending
            .iter()
            .filter(|p| p.due == self.inner.time)
            .map(|p| (p.to, p.msg.clone()))
            .collect();
        self.inner.pending.retain(|p| p.due > self.inner.time);
        for (to, m) in due {
            if let Some(ag) = self.inner.agents.get_mut(&to) {
                ag.receive(m);
            }
        }

        // 2. задания скриптов: продолжения ожидавших + события такта
        let mut jobs: Vec<(AgentId, u8, AgentId, i64, bool)> = Vec::new();
        for &id in &self.inner.order.clone() {
            let Some(ag) = self.inner.agents.get_mut(&id) else {
                continue;
            };
            for si in 0..slot::N {
                // тик кулдауна у всех живых машин
                if let Some(s) = ag.scripts[si].as_mut() {
                    s.machine.tick_cooldown();
                }
                // машина на паузе и остыла — пора продолжать
                if let Some(s) = &ag.scripts[si] {
                    if !s.machine.is_halted() && !s.machine.is_cooling() {
                        jobs.push((id, si as u8, id, 0, false));
                    }
                }
            }
            for m in ag.drain_inbox() {
                jobs.push((id, m.slot, m.from, m.param, true));
            }
            if ag.tick_timer() {
                jobs.push((id, slot::TIMER, id, 0, true));
            }
        }

        // 3. исполнение: скрипт вынимаем из слота, мир остаётся целым
        for (id, slot_idx, from, param, restart) in jobs {
            let Some(mut script) = self
                .inner
                .agents
                .get_mut(&id)
                .and_then(|ag| ag.scripts[slot_idx as usize].take())
            else {
                continue;
            };
            if restart {
                script.machine.start();
            }
            let _ = self.exec(id, from, param, &mut script);
            if let Some(ag) = self.inner.agents.get_mut(&id) {
                if ag.alive {
                    ag.scripts[slot_idx as usize] = Some(script);
                } else {
                    let name = ag.sprite.clone();
                    self.inner.order.retain(|&x| x != id);
                    self.inner.agents.remove(&id);
                    self.inner.log.push(format!("#{id} ({name}) умер"));
                }
            }
        }

        // 4. физика: интеграция, трение, стены/переходы между комнатами
        for &id in &self.inner.order {
            let Some(ag) = self.inner.agents.get_mut(&id) else {
                continue;
            };
            let cand = ag.pos + ag.vel;
            if self.map.room_at(cand).is_some() {
                ag.pos = cand;
            } else if let Some(cur) = self.map.room_at(ag.pos) {
                let r = &self.map.rooms[cur.0 as usize].rect;
                ag.pos = Vec2::new(
                    cand.x.clamp(r.min.x, r.max.x),
                    cand.y.clamp(r.min.y, r.max.y),
                );
            }
            // сопротивление среды: толчок затухает за несколько тиков
            // (²: тик 50 мс — 0.25 = 0.5²: за секунду затухание то же,
            // что при тике 25 мс: 0.25²⁰ = 0.5⁴⁰).
            ag.vel = ag.vel * 0.25;
        }
    }

    /// Гонит машину один раз (до конца или блокировки `wait`); ошибки пишет
    /// в журнал мира. Продолжение после `wait` перепланируется в `step`.
    fn exec(
        &mut self,
        target: AgentId,
        from: AgentId,
        param: i64,
        script: &mut Script,
    ) -> Result<(), String> {
        let cmds = &self.cmds;
        let gvars = &mut self.gvars;
        let mut svc = WorldSvc {
            inner: &mut self.inner,
        };
        let mut ctx = Ctx {
            target: target.0,
            owner: target.0,
            from: from.0,
            param,
            services: &mut svc,
        };
        match script.machine.run(&script.prog, cmds, gvars, &mut ctx) {
            Ok(Flow::Halted | Flow::Blocked) => Ok(()),
            Err(e) => {
                let msg = format!("скрипт #{target}: {e}");
                self.inner.log.push(msg);
                Err(e)
            }
        }
    }

    // ── наблюдение ──────────────────────────────────────────────────────

    pub fn agent(&self, id: AgentId) -> Option<&Agent> {
        self.inner.agents.get(&id)
    }

    pub fn agent_mut(&mut self, id: AgentId) -> Option<&mut Agent> {
        self.inner.agents.get_mut(&id)
    }

    /// Убить агента немедленно (семантика команды `die`): снимает живого,
    /// дальше он фильтруется везде (еда/обоняние/рост). Используется
    /// эволюцией для сброса ягод к фиксированному набору между поколениями.
    pub fn kill(&mut self, id: AgentId) {
        if let Some(a) = self.inner.agents.get_mut(&id) {
            a.alive = false;
        }
    }

    pub fn all_ids(&self) -> &[AgentId] {
        &self.inner.order
    }

    /// Журнал событий последнего такта (вывод `out`, смерти, ошибки).
    pub fn log(&self) -> &[String] {
        &self.inner.log
    }

    /// Глобальная переменная скриптов `vgN` как целое (для тестов/отладки).
    pub fn gvar(&self, n: usize) -> i64 {
        match &self.gvars.0[n] {
            crate::script::Value::Int(i) => *i,
            _ => 0,
        }
    }

    pub fn find_nearest(&self, class: (u8, u8), near: AgentId) -> Option<AgentId> {
        let origin = self.inner.agents.get(&near)?.pos;
        self.inner
            .agents
            .values()
            .filter(|a| {
                a.alive && a.cls.family == class.0 && a.cls.genus == class.1 && a.id != near
            })
            .min_by(|a, b| {
                a.pos
                    .dist(origin)
                    .partial_cmp(&b.pos.dist(origin))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a.id)
    }
}

// ── сервисы для скрипт-команд ───────────────────────────────────────────

struct WorldSvc<'w> {
    inner: &'w mut Inner,
}

impl Services for WorldSvc<'_> {
    fn entity_pos(&mut self, id: u64) -> Option<(f32, f32)> {
        self.inner
            .agents
            .get(&AgentId(id))
            .map(|a| (a.pos.x, a.pos.y))
    }

    fn entity_move(&mut self, id: u64, dx: f32, dy: f32) {
        if let Some(a) = self.inner.agents.get_mut(&AgentId(id)) {
            a.vel += Vec2::new(dx, dy);
        }
    }

    fn entity_teleport(&mut self, id: u64, x: f32, y: f32) {
        if let Some(a) = self.inner.agents.get_mut(&AgentId(id)) {
            a.pos = Vec2::new(x, y);
            a.vel = Vec2::ZERO;
        }
    }

    fn entity_kill(&mut self, id: u64) {
        if let Some(a) = self.inner.agents.get_mut(&AgentId(id)) {
            a.alive = false;
        }
    }

    fn entity_class(&mut self, id: u64) -> Option<(u8, u8)> {
        self.inner
            .agents
            .get(&AgentId(id))
            .map(|a| (a.cls.family, a.cls.genus))
    }

    fn find_nearest(&mut self, class: (u8, u8), near: u64) -> Option<u64> {
        let origin = self.inner.agents.get(&AgentId(near))?.pos;
        self.inner
            .agents
            .values()
            .filter(|a| {
                a.alive && a.cls.family == class.0 && a.cls.genus == class.1 && a.id.0 != near
            })
            .min_by(|a, b| {
                a.pos
                    .dist(origin)
                    .partial_cmp(&b.pos.dist(origin))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a.id.0)
    }

    fn send_message(&mut self, from: u64, to: u64, slot: u8, param: i64) {
        if to == from || to == 0 || !self.inner.agents.contains_key(&AgentId(to)) {
            return;
        }
        let due = Tick(self.inner.time.0 + 1);
        self.inner.pending.push(Pending {
            due,
            to: AgentId(to),
            msg: Message {
                from: AgentId(from),
                slot,
                param,
            },
        });
    }

    fn rand_range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo + 1).max(1) as u64;
        lo + (self.inner.rng.next_u64() % span) as i64
    }

    fn log(&mut self, msg: String) {
        self.inner.log.push(msg);
    }
}

// ── команды движка (аналог хендлеров CAOS) ─────────────────────────────

fn engine_commands() -> Vec<CommandDef> {
    use crate::script::Value;
    vec![
        // позиция цели: posx / posy
        CommandDef {
            name: "posx",
            args: 0,
            f: |c, _| {
                Ok(Value::Float(
                    c.services.entity_pos(c.target).unwrap_or_default().0,
                ))
            },
        },
        CommandDef {
            name: "posy",
            args: 0,
            f: |c, _| {
                Ok(Value::Float(
                    c.services.entity_pos(c.target).unwrap_or_default().1,
                ))
            },
        },
        // толкнуть цель: nudge dx dy
        CommandDef {
            name: "nudge",
            args: 2,
            f: |c, a| {
                c.services
                    .entity_move(c.target, a[0].as_num()? as f32, a[1].as_num()? as f32);
                Ok(Value::Nul)
            },
        },
        // телепорт цели: mvto x y
        CommandDef {
            name: "mvto",
            args: 2,
            f: |c, a| {
                c.services
                    .entity_teleport(c.target, a[0].as_num()? as f32, a[1].as_num()? as f32);
                Ok(Value::Nul)
            },
        },
        // класс цели: fam → семейство, gen → род
        CommandDef {
            name: "fam",
            args: 0,
            f: |c, _| {
                Ok(Value::Int(
                    c.services.entity_class(c.target).unwrap_or((0, 0)).0 as i64,
                ))
            },
        },
        CommandDef {
            name: "gen",
            args: 0,
            f: |c, _| {
                Ok(Value::Int(
                    c.services.entity_class(c.target).unwrap_or((0, 0)).1 as i64,
                ))
            },
        },
        // ближайший агент класса к цели: near <fam> <gen> → entity/null
        CommandDef {
            name: "near",
            args: 2,
            f: |c, a| {
                let cls = (a[0].as_int()? as u8, a[1].as_int()? as u8);
                Ok(match c.services.find_nearest(cls, c.target) {
                    Some(id) => Value::Entity(id),
                    None => Value::Nul,
                })
            },
        },
        // послать сообщение: mesg <кому-entity> <слот> <параметр>
        // (отправитель — владелец скрипта, даже если TARG переключён)
        CommandDef {
            name: "mesg",
            args: 3,
            f: |c, a| {
                c.services.send_message(
                    c.owner,
                    a[0].as_entity()?,
                    a[1].as_int()? as u8,
                    a[2].as_int()?,
                );
                Ok(Value::Nul)
            },
        },
        // убить цель
        CommandDef {
            name: "die",
            args: 0,
            f: |c, _| {
                c.services.entity_kill(c.target);
                Ok(Value::Nul)
            },
        },
        // детерминированный рандом мира: rndi lo hi
        CommandDef {
            name: "rndi",
            args: 2,
            f: |c, a| {
                let lo = a[0].as_int()?;
                let hi = a[1].as_int()?;
                Ok(Value::Int(c.services.rand_range(lo, hi)))
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::agent::slot;

    fn two_room_map() -> WorldMap {
        let mut m = WorldMap::default();
        let _a = m.add_room(0.0, 0.0, 30.0, 20.0);
        let _b = m.add_room(30.0, 0.0, 60.0, 20.0);
        m
    }

    fn critter_cls() -> Cls {
        Cls::new(1, 1, 1)
    }
    fn food_cls() -> Cls {
        Cls::new(1, 2, 1)
    }

    #[test]
    fn message_triggers_script_of_receiver() {
        let mut w = World::new(1, two_room_map());
        let a = w.spawn(critter_cls(), Vec2::new(5.0, 5.0), "critter");
        let f = w.spawn(food_cls(), Vec2::new(25.0, 5.0), "berry");
        // зверь: найди еду и отправь ей слот EAT с параметром 7
        w.attach_script(a, slot::ACTIVATE1, "mesg near 1 2 4 7")
            .unwrap();
        // еда: при EAT — крикнуть параметр и умереть
        w.attach_script(f, slot::EAT, "out param die").unwrap();

        w.activate(a);
        w.step(); // такт 1: зверь находит еду и кладёт сообщение в очередь
        assert!(w.agent(f).is_some());
        w.step(); // такт 2: еда получает EAT, кричит и умирает

        assert!(w.agent(f).is_none(), "еда должна быть съедена");
        assert!(
            w.log().iter().any(|s| s.contains("7")),
            "в журнале должен быть параметр 7: {:?}",
            w.log()
        );
    }

    #[test]
    fn timer_fires_periodically_and_wait_blocks() {
        let mut w = World::new(2, two_room_map());
        let a = w.spawn(critter_cls(), Vec2::new(5.0, 5.0), "critter");
        w.set_timer(a, 3).unwrap();
        // каждый запуск: старт++, пауза 2 тика, завершение++
        w.attach_script(
            a,
            slot::TIMER,
            "setv vg0 add vg0 1 wait 2 setv vg1 add vg1 1",
        )
        .unwrap();

        for _ in 0..12 {
            w.step();
        }
        // старты на тактах 3,6,9,12; завершения на 5,8,11
        assert_eq!(w.gvar(0), 4, "стартов");
        assert_eq!(w.gvar(1), 3, "завершений после wait");
    }

    #[test]
    fn walls_stop_movement() {
        // одна комната: справа — настоящая стена
        let mut w = World::new(3, {
            let mut m = WorldMap::default();
            m.add_room(0.0, 0.0, 30.0, 20.0);
            m
        });
        let a = w.spawn(critter_cls(), Vec2::new(5.0, 10.0), "critter");
        w.attach_script(a, slot::ACTIVATE1, "nudge 100 0").unwrap();
        w.activate(a);
        for _ in 0..50 {
            w.step();
        }
        let ag = w.agent(a).unwrap();
        assert_eq!(ag.pos.x, 30.0, "должен упереться в правую стену");
    }

    #[test]
    fn agent_can_cross_to_second_room() {
        let mut w = World::new(4, two_room_map());
        let a = w.spawn(critter_cls(), Vec2::new(25.0, 10.0), "critter");
        w.set_timer(a, 1).unwrap();
        w.attach_script(a, slot::TIMER, "nudge 1 0").unwrap();
        for _ in 0..20 {
            w.step();
        }
        let pos = w.agent(a).unwrap().pos;
        assert!(pos.x > 30.0, "должен уйти во вторую комнату, pos={pos:?}");
    }

    #[test]
    fn self_destruct_removes_agent() {
        let mut w = World::new(5, two_room_map());
        let a = w.spawn(critter_cls(), Vec2::new(5.0, 5.0), "ghost");
        w.attach_script(a, slot::ACTIVATE1, "die").unwrap();
        w.activate(a);
        w.step();
        assert!(w.agent(a).is_none());
        assert!(!w.all_ids().contains(&a));
    }

    #[test]
    fn determinism_same_seed_same_track() {
        let run = |seed: u64| -> Vec<(f32, f32)> {
            let mut w = World::new(seed, two_room_map());
            let a = w.spawn(critter_cls(), Vec2::new(15.0, 10.0), "c");
            w.set_timer(a, 1).unwrap();
            w.attach_script(a, slot::TIMER, "nudge rndi -1 1 rndi -1 1")
                .unwrap();
            for _ in 0..30 {
                w.step();
            }
            let p = w.agent(a).unwrap().pos;
            vec![(p.x, p.y)]
        };
        assert_eq!(run(77), run(77));
        // разные сиды почти наверняка разойдутся
        assert_ne!(run(1), run(2));
    }
}
