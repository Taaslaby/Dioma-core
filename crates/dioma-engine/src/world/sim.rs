//! World — упрощённый контейнер агентов без скриптов.
//! Легаси VM (a1-script) удалена: ягода — прямой `BerryState`, без `Machine`.
//! Мир хранит агентов, карту и детерминированный RNG; `step` — только физика.

use crate::env::map::WorldMap;
use crate::env::math::Vec2;
use crate::env::{AgentId, Tick};
use crate::rng::XorShift;
use crate::world::agent::{Agent, Cls, Message};
use std::collections::HashMap;

struct Pending {
    due: Tick,
    to: AgentId,
    msg: Message,
}

struct Inner {
    agents: HashMap<AgentId, Agent>,
    order: Vec<AgentId>,
    pending: Vec<Pending>,
    rng: XorShift,
    time: Tick,
    next_id: u64,
    log: Vec<String>,
}

pub struct World {
    pub map: WorldMap,
    inner: Inner,
}

impl World {
    pub fn new(seed: u64, map: WorldMap) -> Self {
        World {
            map,
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

    pub fn spawn(&mut self, cls: Cls, pos: Vec2, sprite: &str) -> AgentId {
        let id = AgentId(self.inner.next_id);
        self.inner.next_id += 1;
        self.inner.agents.insert(id, Agent::new(id, cls, pos, sprite));
        self.inner.order.push(id);
        id
    }

    // Легаси API — заглушки для совместимости (скрипты удалены)
    pub fn attach_script(&mut self, _id: AgentId, _slot: u8, _src: &str) -> Result<(), String> { Ok(()) }
    pub fn set_timer(&mut self, _id: AgentId, _period: u32) -> Result<(), String> { Ok(()) }
    pub fn activate(&mut self, _id: AgentId) {}
    pub fn gvar(&self, _n: usize) -> i64 { 0 }

    pub fn now(&self) -> Tick { self.inner.time }

    pub fn send(&mut self, from: AgentId, to: AgentId, slot: u8, param: i64) {
        if to.is_none() || to == from || !self.inner.agents.contains_key(&to) { return; }
        let due = Tick(self.inner.time.0 + 1);
        self.inner.pending.push(Pending { due, to, msg: Message { from, slot, param } });
    }

    pub fn step(&mut self) {
        self.inner.time.0 += 1;
        self.inner.log.clear();
        self.map.step();
        // доставка отложенных — без VM, просто дропаем (ягода не использует)
        let due: Vec<(AgentId, Message)> = self.inner.pending.iter()
            .filter(|p| p.due == self.inner.time).map(|p| (p.to, p.msg.clone())).collect();
        self.inner.pending.retain(|p| p.due > self.inner.time);
        let _ = due; // сообщения без скриптов игнорируются

        // физика
        for &id in &self.inner.order.clone() {
            let Some(ag) = self.inner.agents.get_mut(&id) else { continue };
            let cand = ag.pos + ag.vel;
            if self.map.room_at(cand).is_some() {
                ag.pos = cand;
            } else if let Some(cur) = self.map.room_at(ag.pos) {
                let r = &self.map.rooms[cur.0 as usize].rect;
                ag.pos = Vec2::new(cand.x.clamp(r.min.x, r.max.x), cand.y.clamp(r.min.y, r.max.y));
            }
            ag.vel = ag.vel * 0.25;
        }
        // уборка мёртвых (kill ставит alive=false)
        let dead: Vec<AgentId> = self.inner.agents.iter().filter(|(_,a)| !a.alive).map(|(id,_)| *id).collect();
        for id in dead {
            self.inner.order.retain(|&x| x != id);
            self.inner.agents.remove(&id);
        }
    }

    pub fn agent(&self, id: AgentId) -> Option<&Agent> { self.inner.agents.get(&id) }
    pub fn agent_mut(&mut self, id: AgentId) -> Option<&mut Agent> { self.inner.agents.get_mut(&id) }
    pub fn kill(&mut self, id: AgentId) { if let Some(a) = self.inner.agents.get_mut(&id) { a.alive = false; } }
    pub fn all_ids(&self) -> &[AgentId] { &self.inner.order }
    pub fn log(&self) -> &[String] { &self.inner.log }
    pub fn find_nearest(&self, class: (u8,u8), near: AgentId) -> Option<AgentId> {
        let origin = self.inner.agents.get(&near)?.pos;
        self.inner.agents.values()
            .filter(|a| a.alive && a.cls.family==class.0 && a.cls.genus==class.1 && a.id!=near)
            .min_by(|a,b| a.pos.dist(origin).partial_cmp(&b.pos.dist(origin)).unwrap_or(std::cmp::Ordering::Equal))
            .map(|a| a.id)
    }
}
