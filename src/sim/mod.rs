//! -------------------------------------------------------------------------
//! Filename:    src/sim/mod.rs
//! Class:       sim
//! Purpose:     АРЕНА — композиция среды, мира объектов и существ
//! Description: существа (creature::Creature) живут в комнатах (env::map),
//!               еду им дают СКРИПТОВЫЕ ягоды (world::World: слот EAT →
//!               «хрум» → die). Ягоды растут от плодородности комнаты
//!               (питательные вещества × свет из CA) и дают потомство.
//!               Слой связывает независимые модули и больше ничего не знает.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use crate::creature::{Action, Creature, Sense};
use crate::env::map::WorldMap;
use crate::env::math::Vec2;
use crate::env::{AgentId, Substance};
use crate::world::{Cls, World, SLOT_EAT};
use std::f32::consts::TAU;

pub mod evolution;
pub use evolution::{Evolution, EvolutionParams, GenerationStats};

/// Радиус «рта»: на этом расстоянии еда считается съедобной.
pub const EAT_RADIUS: f32 = 2.0;
/// Радиус, в котором существо «видит» соседа за едой (социальное обучение).
const SOCIAL_RADIUS: f32 = 10.0;
/// Максимальный поворот за тик (радианы).
pub const MAX_TURN: f32 = 0.3;
/// Затухание запаха с дистанцией: 60/(60+d).
const SMELL_FALLOFF: f32 = 60.0;
/// Размер клетки памяти места (грубая привязка «где была еда»).
const PLACE_CELL: f32 = 8.0;
/// Дальность ощущения стены (юнитов) и шаг пробы.
const WALL_LOOKAHEAD: f32 = 8.0;
const WALL_PROBE_STEP: f32 = 1.0;

// ─── Ягода: класс (1, 2) — «съедобное» ────────────────────────────────────
pub const BERRY_FAMILY: u8 = 1;
pub const BERRY_GENUS: u8 = 2;
/// Скрипт ягоды на EAT: хрумнуть (не умирает — еда возобновляемая).
pub const BERRY_EATEN: &str = "out \"хрум\"";
/// Тиков «отрастания» съеденной ягоды: vars[1] > 0 = ягода пустая,
/// не пахнет и не съедобна, пока счётчик не дойдёт до нуля. Без этого
/// сильные охотники истребляют урожай до размножения ягод и голодают —
/// а food_eaten перестаёт быть мерой навыка.
const BERRY_REGROW_TICKS: f32 = 150.0;
/// Рост за тик: vars[0] += плодородность × GROW_RATE.
const GROW_RATE: f32 = 1.0;
/// Накопленный рост, при котором ягода даёт потомство.
const SPAWN_AT: f32 = 12.0;
/// Питательные вещества комнаты, съедаемые за тик роста (рост не бесплатен).
const NUTRIENT_COST: f32 = 0.02;
/// Разброс потомства вокруг родителя.
const SPAWN_SPREAD: f32 = 3.0;
/// Потолок популяции ягод (защита от экспоненты).
const MAX_BERRIES: usize = 24;

/// Существо в арене: позиция, курс и сам контроллер.
pub struct Critter {
    pub id: AgentId,
    pub pos: Vec2,
    /// Курс в радианах (0 = +X).
    pub heading: f32,
    pub creature: Creature,
    /// Сколько раз существо ударилось о стену (счётчик для наблюдения/тестов).
    pub wall_hits: u64,
    /// Видел ли соседа за едой на прошлом такте (социальное обучение:
    /// заполняет sense.see_eat на следующий тик и сбрасывается).
    pub see_eat: bool,
}

/// Арена: мир объектов (ягоды, карта) + существа. Детерминирована по сиду.
pub struct Arena {
    pub world: World,
    pub critters: Vec<Critter>,
    pub tick: u64,
    /// Следующий свободный id существа (эволюция выделяет id новорождённым).
    pub(crate) next_id: u64,
    /// Тиков отрастания съеденной ягоды (можно тюнить через эволюцию).
    pub berry_regrow: f32,
}

impl Arena {
    pub fn new(seed: u64, map: WorldMap) -> Self {
        Arena {
            world: World::new(seed, map),
            critters: Vec::new(),
            tick: 0,
            next_id: 1,
            berry_regrow: BERRY_REGROW_TICKS,
        }
    }

    pub fn spawn(&mut self, pos: Vec2, heading: f32, creature_seed: u64) -> AgentId {
        self.spawn_creature(pos, heading, Creature::new(creature_seed))
    }

    /// Посадить готовое существо (контрольные протоколы: `lr = 0` и т.п.).
    pub fn spawn_creature(&mut self, pos: Vec2, heading: f32, creature: Creature) -> AgentId {
        let id = AgentId(self.next_id);
        self.next_id += 1;
        self.critters.push(        Critter {
            id,
            pos,
            heading,
            creature,
            wall_hits: 0,
            see_eat: false,
        });
        id
    }

    /// Посадить ягоду: скриптовый агент мира со слотом EAT («хрум» → die).
    pub fn add_berry(&mut self, pos: Vec2) -> AgentId {
        let id = self.world.spawn(Cls::new(BERRY_FAMILY, BERRY_GENUS, 1), pos, "berry");
        self.world
            .attach_script(id, SLOT_EAT, BERRY_EATEN)
            .expect("EAT-скрипт ягоды");
        id
    }

    /// Живая ягода готова к еде (не в стадии отрастания).
    fn berry_edible(ag: &crate::world::Agent) -> bool {
        ag.alive && ag.cls.family == BERRY_FAMILY && ag.cls.genus == BERRY_GENUS && ag.vars[1] <= 0.0
    }

    /// Живая ягода без проверки съедобности (для роста/счётчиков).
    fn berry_any(ag: &crate::world::Agent) -> bool {
        ag.alive && ag.cls.family == BERRY_FAMILY && ag.cls.genus == BERRY_GENUS
    }

    /// Живые ягоды в мире (включая отрастающие — они занимают место).
    pub fn berries(&self) -> Vec<AgentId> {
        self.world
            .all_ids()
            .iter()
            .filter(|&&id| self.world.agent(id).map(Self::berry_any).unwrap_or(false))
            .copied()
            .collect()
    }

    /// Один такт: мир (сообщения/скрипты/физика) → существа → еда → рост.
    pub fn step(&mut self) {
        self.tick += 1;
        // 0. Мир: доставка сообщений (в т.ч. вчерашний EAT → «хрум» → die),
        //    таймеры, скрипты и физика комнат.
        self.world.step();

        // 1. Сенсоры (по живым ягодам мира + память о соседе за едой) → решения.
        let senses: Vec<Sense> = self
            .critters
            .iter()
            .map(|c| self.sense(c.pos, c.heading, c.see_eat))
            .collect();
        for c in &mut self.critters {
            c.see_eat = false; // окно наблюдения закрылось
        }
        let mut actions: Vec<(usize, Action)> = Vec::with_capacity(self.critters.len());
        for (i, c) in self.critters.iter_mut().enumerate() {
            actions.push((i, c.creature.tick(&senses[i])));
        }

        // 2. Движение по комнатам (стены/переходы — физика мира).
        //    Удар о стену = кандидат вне всех комнат: позиция зажимается на
        //    край, а существу сообщается удар (боль + негативный дофамин).
        for &(i, ref action) in &actions {
            let c = &mut self.critters[i];
            c.heading = (c.heading + action.steer * MAX_TURN).rem_euclid(TAU);
            let dir = Vec2::new(c.heading.cos(), c.heading.sin());
            let step = dir * (action.drive * c.creature.speed());
            let cand = c.pos + step;
            if self.world.map.room_at(cand).is_some() {
                c.pos = cand;
            } else if let Some(room) = self.world.map.room_at(c.pos) {
                let r = &self.world.map.rooms[room.0 as usize].rect;
                c.pos = Vec2::new(
                    cand.x.clamp(r.min.x, r.max.x),
                    cand.y.clamp(r.min.y, r.max.y),
                );
                c.creature.hit_wall();
                c.wall_hits += 1;
            }
        }

        // 3. Еда: рот открыт → послать ягоде сообщение EAT (хрумнёт на след. такте).
        let mut eaters: Vec<Vec2> = Vec::new();
        for &(i, ref action) in &actions {
            if !action.eat {
                continue;
            }
            let pos = self.critters[i].pos;
            // Съедобная (не отрастающая) ягода в радиусе рта: хрумнуть,
            // ягода уходит в стадию отрастания (vars[1] = таймер).
            if let Some(&bid) = self
                .berries()
                .iter()
                .find(|&&id| {
                    self.world.agent(id).map(Self::berry_edible).unwrap_or(false)
                        && self.world.agent(id).unwrap().pos.dist(pos) <= EAT_RADIUS
                })
            {
                self.world.send(AgentId::NONE, bid, SLOT_EAT, 0);
                if let Some(ag) = self.world.agent_mut(bid) {
                    ag.vars[1] = self.berry_regrow;
                }
                eaters.push(pos);
            }
        }
        // Социальное обучение: соседи в радиусе видят едущего и получают
        // зеркальный сигнал на следующий такт (see_eat → sense.see_eat).
        if !eaters.is_empty() {
            for c in &mut self.critters {
                if c.see_eat {
                    continue;
                }
                if eaters.iter().any(|e| e.dist(c.pos) <= SOCIAL_RADIUS) {
                    c.see_eat = true;
                }
            }
        }

        // 4. Рост ягод от плодородности комнаты (питание × свет из CA).
        //    Съеденные ягоды отрастают: vars[1] тикает вниз к нулю.
        //    Сначала собираем состояние (только чтение), потом мутируем —
        //    чтобы не держать заимствование мира.
        let mut growth: Vec<(AgentId, Vec2, f32, f32)> = Vec::new();
        for &id in self.world.all_ids() {
            let Some(ag) = self.world.agent(id) else {
                continue;
            };
            if !Self::berry_any(ag) {
                continue;
            }
            let fert = self.fertility(ag.pos);
            if fert > 0.0 || ag.vars[1] > 0.0 {
                growth.push((id, ag.pos, fert, ag.vars[1]));
            }
        }
        for (id, pos, fert, regrow) in growth {
            // Съеденная ягода отрастает: таймер тикает вниз.
            if regrow > 0.0 {
                if let Some(ag) = self.world.agent_mut(id) {
                    ag.vars[1] = (ag.vars[1] - 1.0).max(0.0);
                }
            }
            let room = self.world.map.room_at(pos);
            let grown = {
                let ag = self.world.agent_mut(id).expect("ягода жива");
                ag.vars[0] += fert * GROW_RATE;
                let done = ag.vars[0] >= SPAWN_AT;
                if done {
                    ag.vars[0] = 0.0;
                }
                done
            };
            // Рост съедает питание комнаты.
            if let Some(rid) = room {
                let ca = &mut self.world.map.rooms[rid.0 as usize].ca[Substance::Nutrients.idx()];
                *ca = (*ca - NUTRIENT_COST).max(0.0);
            }
            // Потомство рядом с родителем (детерминированный разброс).
            if grown && self.berries().len() < MAX_BERRIES {
                let offset = deterministic_jitter(id.0, self.tick, SPAWN_SPREAD);
                let b = self.world.spawn(
                    Cls::new(BERRY_FAMILY, BERRY_GENUS, 1),
                    pos + offset,
                    "berry",
                );
                let _ = self.world.attach_script(b, SLOT_EAT, BERRY_EATEN);
            }
        }
    }

    /// Обоняние + направленное ощущение стен: запах ягод и близость стены
    /// слева/справа относительно курса + «еда у рта».
    /// Сенсоры существа: запах ягод (живые), стены, еда у рта, память места
    /// и зеркальный сигнал `see_eat` (видел соседа за едой на прошлом такте).
    fn sense(&self, pos: Vec2, heading: f32, see_eat: bool) -> Sense {
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        let mut near = false;
        for &id in self.world.all_ids() {
            let Some(ag) = self.world.agent(id) else { continue };
            // Отрастающие ягоды не пахнут и не «у рта» — иначе существо
            // залипало бы у пустой ягоды.
            if !Self::berry_edible(ag) {
                continue;
            }
            let d = ag.pos - pos;
            let dist = d.len();
            let strength = (SMELL_FALLOFF / (SMELL_FALLOFF + dist)).clamp(0.0, 1.0);
            if dist <= EAT_RADIUS {
                near = true;
            }
            let mut rel = d.y.atan2(d.x) - heading;
            rel = (rel + std::f32::consts::PI).rem_euclid(TAU) - std::f32::consts::PI;
            if rel < 0.0 {
                l += strength * (-rel.sin()).max(0.0);
            } else {
                r += strength * rel.sin().max(0.0);
            }
        }

        // Стены: пробы вбок (±90°) и по диагонали вперёд (±45°) — стена прямо
        // по курсу попадает в обе диагональные пробы, боковая — в свою.
        let fwd = Vec2::new(heading.cos(), heading.sin());
        let left = Vec2::new(fwd.y, -fwd.x);
        let right = Vec2::new(-fwd.y, fwd.x);
        let ahead_l = Vec2::new(
            (heading - std::f32::consts::FRAC_PI_4).cos(),
            (heading - std::f32::consts::FRAC_PI_4).sin(),
        );
        let ahead_r = Vec2::new(
            (heading + std::f32::consts::FRAC_PI_4).cos(),
            (heading + std::f32::consts::FRAC_PI_4).sin(),
        );
        let wall_l = self.wall_probe(pos, left).max(self.wall_probe(pos, ahead_l));
        let wall_r = self.wall_probe(pos, right).max(self.wall_probe(pos, ahead_r));

        // Память места: грубая клетка карты (квадрат PLACE_CELL) + курс.
        let cx = (pos.x / PLACE_CELL).floor() as i64;
        let cy = (pos.y / PLACE_CELL).floor() as i64;
        let place = (cx.wrapping_mul(7349) ^ cy.wrapping_mul(7919)) as u32;
        Sense {
            smell_l: l.min(1.0),
            smell_r: r.min(1.0),
            wall_l,
            wall_r,
            food_near: near,
            wall: false,
            place,
            heading,
            see_eat,
        }
    }

    /// Близость стены в направлении `dir`: 1.0 у самой стены, затухает к 0 на
    /// `WALL_LOOKAHEAD` (стена = точка вне всех комнат — та же семантика, что
    /// у блокировки движения).
    fn wall_probe(&self, pos: Vec2, dir: Vec2) -> f32 {
        let mut d = WALL_PROBE_STEP;
        while d <= WALL_LOOKAHEAD {
            if self.world.map.room_at(pos + dir * d).is_none() {
                return 1.0 - (d / WALL_LOOKAHEAD);
            }
            d += WALL_PROBE_STEP;
        }
        0.0
    }

    /// Плодородность точки: питательные вещества × свет комнаты (0..1).
    fn fertility(&self, pos: Vec2) -> f32 {
        let Some(rid) = self.world.map.room_at(pos) else {
            return 0.0;
        };
        let ca = self.world.map.rooms[rid.0 as usize].ca;
        (ca[Substance::Nutrients.idx()] / 255.0) * (ca[Substance::Light.idx()] / 255.0)
    }
}

/// Детерминированный разброс потомства из (id родителя, такт) — без ГСЧ.
fn deterministic_jitter(a: u64, b: u64, spread: f32) -> Vec2 {
    let h = |s: u64| -> f32 {
        let x = s.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ s.rotate_right(17);
        ((x >> 40) & 0xFFFF) as f32 / 65536.0 * 2.0 - 1.0
    };
    Vec2::new(
        h(a ^ b) * spread,
        h(a.rotate_left(23) ^ b.rotate_left(7)) * spread,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::metabolism::chem;

    #[test]
    fn creature_hunts_berry_and_it_crunches() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 80.0, 20.0);
        let mut arena = Arena::new(42, map);
        arena.spawn(Vec2::new(5.0, 10.0), 0.0, 42);
        // Голод сразу, чтобы не ждать сотни тиков синтеза HUNGER.
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        let berry = arena.add_berry(Vec2::new(35.0, 10.0));

        for _ in 0..250 {
            arena.step();
            // Съедена = ушла в стадию отрастания (vars[1] > 0).
            if arena
                .world
                .agent(berry)
                .map(|a| a.vars[1] > 0.0)
                .unwrap_or(false)
            {
                break;
            }
        }
        // Ещё один шаг: EAT-сообщение доставляется в world.step следующего
        // такта — только тогда скрипт «хрум» выполнится и попадёт в журнал.
        arena.step();

        assert!(
            arena
                .world
                .agent(berry)
                .map(|a| a.alive && a.vars[1] > 0.0)
                .unwrap_or(false),
            "ягода должна быть съедена и отрастать (не исчезать)"
        );
        assert!(
            arena.world.log().iter().any(|s| s.contains("хрум")),
            "ягода должна хрумнуть: {:?}",
            arena.world.log()
        );
        assert!(
            arena.critters[0].creature.glucose() > 128.0,
            "еда должна дать питание"
        );
    }

    /// Охота за стеной: существо с направленными сенсорами стены объезжает
    /// L-препятствие и доедает в обоих триалах (мозг живёт между триалами,
    /// кровь сбрасывается). Раньше (без сенсора) тот же удар выжигал гоминг,
    /// и второй триал голодал у стены.
    #[test]
    fn detour_hunt_survives_trials_with_wall_sense() {
        let mut map = WorldMap::default();
        // L-форма: нижний правый квадрант (x>20, y<10) — пустота (стена).
        map.add_room(0.0, 0.0, 20.0, 20.0);
        map.add_room(20.0, 10.0, 60.0, 20.0);
        let mut a = Arena::new(42, map);
        a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
        for _ in 0..2 {
            a.critters[0].creature.blood = crate::body::metabolism::Blood::creature(false);
            a.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
            let berry = a.add_berry(Vec2::new(55.0, 15.0));
            let h0 = a.critters[0].wall_hits;
            let mut ate = false;
            for _ in 0..800 {
                a.step();
                // Съедена = отрастает (vars[1] > 0).
                if a
                    .world
                    .agent(berry)
                    .map(|ag| ag.vars[1] > 0.0)
                    .unwrap_or(false)
                {
                    ate = true;
                    break;
                }
            }
            assert!(ate, "существо должно объехать стену и съесть ягоду");
            assert!(
                a.critters[0].wall_hits - h0 <= 5,
                "объезд почти без ударов: {} ударов",
                a.critters[0].wall_hits - h0
            );
            a.critters[0].pos = Vec2::new(5.0, 5.0);
            a.critters[0].heading = 0.0;
        }
    }

    /// Сквозная проводка «стена → боль → мозг»: существо в углу упирается
    /// в стены, арена считает удар и поднимает ноцицепцию в крови
    /// (bump_wall → PAIN). В углу отворот невозможен — сенсоры стены дают
    /// компенсирующие сигналы, и удар гарантирован.
    #[test]
    fn wall_hit_counts_and_raises_pain() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut arena = Arena::new(7, map);
        arena.spawn(Vec2::new(39.5, 1.5), 0.0, 7);
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        let pain0 = arena.critters[0].creature.blood.conc[chem::PAIN];

        // Гоним до первого удара о стену (угол: удар на первом же такте).
        let mut guard = 0;
        while arena.critters[0].wall_hits == 0 && guard < 200 {
            arena.step();
            guard += 1;
        }
        assert!(
            arena.critters[0].wall_hits > 0,
            "существо должно упереться в стену и получить удары"
        );
        // bump_wall ставит EVENT_WALL, а боль эмитируется в blood.update()
        // следующего тика — один шаг, чтобы эмиттер сработал.
        arena.step();
        let pain = arena.critters[0].creature.blood.conc[chem::PAIN];
        assert!(
            pain > pain0 + 5.0,
            "удар о стену должен поднять боль: {pain0} → {pain}"
        );
    }

    #[test]
    fn berries_grow_only_in_fertile_rooms() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0); // плодородная
        map.add_room(40.0, 0.0, 80.0, 20.0); // бедная
        let mut arena = Arena::new(1, map);
        // Почти без распада веществ, чтобы питание не ушло за время теста.
        arena.world.map.rates.decay = [0.0005; 4];
        let nutr = Substance::Nutrients.idx();
        let light = Substance::Light.idx();
        arena.world.map.rooms[0].ca[nutr] = 200.0;
        arena.world.map.rooms[0].ca[light] = 200.0;

        let rich = arena.add_berry(Vec2::new(20.0, 10.0)); // плодородная
        let poor = arena.add_berry(Vec2::new(60.0, 10.0)); // бедная
        let n0 = arena.berries().len();

        for _ in 0..600 {
            arena.step();
        }

        let n1 = arena.berries().len();
        assert!(
            n1 > n0,
            "в плодородной комнате ягоды должны размножиться: {n0} → {n1}"
        );
        assert!(
            arena.world.agent(poor).is_some() && arena.world.agent(rich).is_some(),
            "родители не должны исчезать без еды"
        );
        assert!(
            arena.world.map.rooms[0].ca[nutr] < 200.0,
            "рост должен съедать питание комнаты"
        );
        assert!(n1 <= MAX_BERRIES, "популяция ограничена капом: {n1}");
    }

    #[test]
    fn arena_is_deterministic_from_seeds() {
        let run = || -> Vec<(f32, f32)> {
            let mut map = WorldMap::default();
            map.add_room(0.0, 0.0, 60.0, 20.0);
            let mut a = Arena::new(42, map);
            a.spawn(Vec2::new(5.0, 10.0), 0.0, 42);
            a.add_berry(Vec2::new(30.0, 10.0));
            for _ in 0..50 {
                a.step();
            }
            a.critters
                .iter()
                .map(|c| (c.pos.x, c.pos.y))
                .collect()
        };
        assert_eq!(run(), run(), "один сид — одна траектория");
    }
}
