//! -------------------------------------------------------------------------
//! Filename:    src/sim/evolution.rs
//! Class:       sim::evolution
//! Purpose:     ЭВОЛЮЦИЯ — популяция охотников на арене
//! Description: поколение = N тиков арены, где голодные умирают (глюкоза →
//!              урон → смерть), а сытые в конце размножаются через
//!              `Creature::spawn_child` — с эффектом Болдуина: выученные за
//!              жизнь пути лучших охотников сворачиваются в инстинкт-гены
//!              потомкам. Следующее поколение рождается с более сильным
//!              врождённым рефлексом охоты. Детерминированно по сиду.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use crate::body::metabolism::chem;
use crate::env::map::WorldMap;
use crate::env::math::Vec2;
use crate::env::AgentId;
use super::{Arena, Critter};

/// Голод новорождённого: свежий организм (HUNGER ≈ 30) начинает охоту
/// только через сотни тиков распада глюкозы — новорождённый эволюционной
/// арены рождается голодным (как и положено), и отбор сразу работает.
const NEWBORN_HUNGER: f32 = 200.0;
/// Стартовые позиции базовых ягод (доли комнаты, сетка 2×4).
const BERRY_SPOTS: [(f32, f32); 8] = [
    (0.2, 0.25),
    (0.4, 0.25),
    (0.6, 0.25),
    (0.8, 0.25),
    (0.2, 0.75),
    (0.4, 0.75),
    (0.6, 0.75),
    (0.8, 0.75),
];

/// Параметры эволюционной арены.
#[derive(Clone, Debug)]
pub struct EvolutionParams {
    /// Целевой размер популяции (число существ в начале каждого поколения).
    pub population: usize,
    /// Длительность поколения в тиках (окно естественного отбора).
    pub generation_ticks: u64,
    /// Каждые `food_per_child` порций еды у родителя дают +1 ребёнка.
    pub food_per_child: u32,
    /// Потолок детей у одного родителя.
    pub max_children: usize,
    /// Тиков отрастания съеденной ягоды (короткий регроу = еда всегда
    /// доступна, food_eaten ближе к «чистому навыку»; длинный — экология
    /// с голодными окнами).
    pub berry_regrow: f32,
    /// Целевой запас ягод на старте поколения (фиксированные точки).
    pub target_berries: usize,
}

impl Default for EvolutionParams {
    fn default() -> Self {
        Self {
            population: 8,
            generation_ticks: 1200,
            food_per_child: 6,
            max_children: 4,
            berry_regrow: 40.0,
            target_berries: 8,
        }
    }
}

/// Итоги одного поколения: кого отобрала жизнь, что досталось потомкам.
#[derive(Clone, Debug)]
pub struct GenerationStats {
    pub generation: u64,
    /// Сколько существ дожило до конца поколения.
    pub survivors: usize,
    /// Сколько детей родилось в конце поколения.
    pub born: usize,
    /// Среднее съеденное выжившими (порций).
    pub avg_food_eaten: f64,
    /// Максимум съеденного среди выживших.
    pub max_food_eaten: u32,
    /// Сколько выживших носят инстинкт-ген (свёрнутый выученный рефлекс).
    pub with_instinct: usize,
    /// Средний дофамин сильнейшего инстинкта по популяции (мера силы
    /// врождённого рефлекса; 0 = инстинктов нет вообще).
    pub instinct_dopamine_avg: f64,
}

/// Эволюционная арена: оборачивает `Arena` и управляет поколениями.
pub struct Evolution {
    pub arena: Arena,
    pub params: EvolutionParams,
    /// Сид всей линии (инициализация + размножение детерминированы им).
    pub seed: u64,
    /// Текущий номер поколения (0 = стартовое, до первого размножения).
    pub generation: u64,
    /// Статистика всех прошедших поколений.
    pub history: Vec<GenerationStats>,
    /// Исходные концентрации CA комнат: каждое поколение стартует в свежей
    /// среде (плодородие восстанавливается — «новый сезон»).
    room_ca: Vec<[f32; 4]>,
}

impl Evolution {
    /// Арена с начальной популяцией: существа из сидов, расставленные
    /// детерминированной сеткой по первой комнате карты.
    pub fn new(seed: u64, map: WorldMap, params: EvolutionParams) -> Self {
        let mut arena = Arena::new(seed ^ 0x517C_1B72_70DE_942A, map);
        arena.berry_regrow = params.berry_regrow;
        let pop = params.population.max(1);
        let room_ca = arena
            .world
            .map
            .rooms
            .iter()
            .map(|r| r.ca)
            .collect();
        let mut evo = Evolution {
            arena,
            params,
            seed,
            generation: 0,
            history: Vec::new(),
            room_ca,
        };
        evo.spawn_initial(pop);
        evo
    }

    /// Начальная популяция: сетка по первой комнате, у всех свой сид
    /// (линия `seed + i` — каждый род уникален), все голодны.
    fn spawn_initial(&mut self, pop: usize) {
        for i in 0..pop {
            let pos = self.grid_pos(i);
            let creature_seed = self.seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u64);
            let heading = deterministic_heading(self.seed, i as u64);
            let id = self.arena.spawn(pos, heading, creature_seed);
            // Новорождённый голоден — иначе первые ~700 тиков никто не ест.
            if let Some(c) = self.arena.critters.iter_mut().find(|c| c.id == id) {
                c.creature.blood.conc[chem::HUNGER] = NEWBORN_HUNGER;
            }
        }
    }

    /// Один такт: мир + существа, затем голодные (урон ≥ 1) убираются.
    /// Умершие не едят и не размножаются — это и есть естественный отбор.
    pub fn step(&mut self) {
        self.arena.step();
        self.arena.critters.retain(|c| c.creature.alive());
    }

    /// Прогнать одно поколение: `generation_ticks` шагов естественного
    /// отбора, затем сытые размножаются, голодные/мёртвые не оставляют
    /// потомства. Возвращает итоги поколения.
    pub fn run_generation(&mut self) -> GenerationStats {
        self.fresh_environment();
        for _ in 0..self.params.generation_ticks {
            self.step();
            if self.arena.critters.is_empty() {
                break; // всё поколение вымерло — размножать некого
            }
        }
        let mut stats = self.collect_stats();
        stats.born = self.reproduce();
        self.generation += 1;
        self.history.push(stats.clone());
        stats
    }

    /// «Новый сезон»: восстановить плодородие комнат до исходного и ПОЛНОСТЬЮ
    /// пересадить ягоды в фиксированные точки базового набора. Среда каждого
    /// поколения бит-в-бит одинакова — разница в food_eaten отражает только
    /// геномы (включая свёрнутые инстинкты), а не удачу с ягодами.
    fn fresh_environment(&mut self) {
        for (room, base) in self.arena.world.map.rooms.iter_mut().zip(&self.room_ca) {
            room.ca.copy_from_slice(base);
        }
        for id in self.arena.berries() {
            self.arena.world.kill(id);
        }
        if let Some(room) = self.arena.world.map.rooms.first().map(|r| r.rect) {
            let n = self.params.target_berries.min(BERRY_SPOTS.len());
            for &(fx, fy) in &BERRY_SPOTS[..n] {
                let pos = Vec2::new(
                    room.min.x + (room.max.x - room.min.x) * fx,
                    room.min.y + (room.max.y - room.min.y) * fy,
                );
                self.arena.add_berry(pos);
            }
        }
    }

    /// Прогнать `n` поколений подряд.
    pub fn run(&mut self, n: u64) {
        for _ in 0..n {
            self.run_generation();
        }
    }

    /// Итоги текущего поколения (до размножения): по выжившим.
    fn collect_stats(&self) -> GenerationStats {
        let mut avg = 0.0f64;
        let mut max = 0u32;
        let mut with_instinct = 0usize;
        let mut dop_sum = 0.0f64;
        let n = self.arena.critters.len();
        for c in &self.arena.critters {
            let f = c.creature.food_eaten();
            avg += f as f64;
            max = max.max(f);
            let instincts = &c.creature.genome.instincts;
            if !instincts.is_empty() {
                with_instinct += 1;
                let best = instincts
                    .iter()
                    .map(|i| i.dopamine)
                    .fold(0.0f32, f32::max);
                dop_sum += best as f64;
            }
        }
        GenerationStats {
            generation: self.generation,
            survivors: n,
            born: 0,
            avg_food_eaten: if n > 0 { avg / n as f64 } else { 0.0 },
            max_food_eaten: max,
            with_instinct,
            instinct_dopamine_avg: if with_instinct > 0 {
                dop_sum / with_instinct as f64
            } else {
                0.0
            },
        }
    }

    /// Размножение сытых: выжившие ранжируются по съеденному. Ребёнок НЕ
    /// даётся бесплатно — `food_eaten / food_per_child` детей: голодавший
    /// (food < food_per_child) не оставляет потомства вовсе, и его гены
    /// вымирают. Если детей меньше целевой популяции, лучшие родители
    /// добирают до потолка — популяция не пустеет, но геномы следующего
    /// поколения приходят от реально сытых. Через `spawn_child`: геном
    /// мутирует, выученные пути сворачиваются в инстинкты (эффект
    /// Болдуина). Дети сажаются на ту же сетку позиций, что и стартовое
    /// поколение: отбор измеряет геномы, а не удачу с местом рождения.
    /// Возвращает число родившихся.
    fn reproduce(&mut self) -> usize {
        let survivors: Vec<Critter> = std::mem::take(&mut self.arena.critters);
        let mut ranked = survivors;
        ranked.sort_by_key(|a| std::cmp::Reverse(a.creature.food_eaten()));
        let fpc = self.params.food_per_child.max(1);

        // Распределение детей: первый проход — за реально съеденное
        // (голодавшие не размножаются), добор — лучшим до потолка.
        let mut alloc: Vec<usize> = ranked.iter().map(|_| 0).collect();
        for (i, parent) in ranked.iter().enumerate() {
            alloc[i] = (parent.creature.food_eaten() / fpc)
                .min(self.params.max_children as u32) as usize;
        }
        let mut need = self.params.population.saturating_sub(alloc.iter().sum::<usize>());
        for (i, _) in ranked.iter().enumerate() {
            if need == 0 {
                break;
            }
            let room = self.params.max_children.saturating_sub(alloc[i]);
            let add = room.min(need);
            alloc[i] += add;
            need -= add;
        }

        let mut children: Vec<Critter> = Vec::new();
        let mut child_idx = 0u64;
        'outer: for (i, &n) in alloc.iter().enumerate() {
            for _ in 0..n {
                if children.len() >= self.params.population {
                    break 'outer;
                }
                self.push_child(&ranked[i], &mut children, &mut child_idx);
            }
        }
        let born = children.len();
        self.arena.critters = children;
        born
    }

    /// Один потомок родителя (через `spawn_child`) на сетку-позицию.
    fn push_child(&mut self, parent: &Critter, children: &mut Vec<Critter>, child_idx: &mut u64) {
        let child_seed = self.child_seed(parent.creature.seed, *child_idx);
        let mut creature = parent.creature.spawn_child(child_seed);
        // Сброс сид потомка на его собственный (spawn_child уже строит из
        // child_seed, но тело/голод — свежие, как новорождённый).
        creature.seed = child_seed;
        // Новорождённый голоден — иначе первая половина поколения уходит
        // на «просыпание» метаболизма.
        creature.blood.conc[chem::HUNGER] = NEWBORN_HUNGER;
        let pos = self.grid_pos(*child_idx as usize);
        let id = AgentId(self.arena.next_id);
        self.arena.next_id += 1;
        children.push(Critter {
            id,
            pos,
            heading: deterministic_heading(child_seed, 0),
            creature,
            wall_hits: 0,
            see_eat: false,
        });
        *child_idx += 1;
    }

    /// Позиция на стартовой сетке (та же, что у начальной популяции).
    fn grid_pos(&self, i: usize) -> Vec2 {
        let Some(r) = self
            .arena
            .world
            .map
            .rooms
            .first()
            .map(|room| room.rect)
        else {
            return Vec2::new(0.0, 0.0);
        };
        let cols = (self.params.population as f32).sqrt().ceil().max(1.0) as usize;
        let (w, h) = (r.max.x - r.min.x, r.max.y - r.min.y);
        let (dx, dy) = (w / cols as f32, h / cols as f32);
        let (cx, cy) = (i % cols, i / cols);
        Vec2::new(
            r.min.x + dx * (cx as f32 + 0.5),
            r.min.y + dy * (cy as f32 + 0.5),
        )
    }

    /// Сид потомка: детерминирован из (сида родителя, поколения, индекса
    /// ребёнка) — один и тот же прогон даёт одних и тех же потомков.
    fn child_seed(&self, parent_seed: u64, idx: u64) -> u64 {
        parent_seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.generation.wrapping_mul(0x517C_1B72))
            .wrapping_add(idx.wrapping_mul(0xB5A4_0E51))
    }

}

/// Детерминированный начальный курс из сида.
fn deterministic_heading(seed: u64, i: u64) -> f32 {
    let h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.rotate_left(17);
    ((h >> 40) & 0xFFFF) as f32 / 65536.0 * std::f32::consts::TAU
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::Substance;

    /// Конфиг, на котором эволюция стабильно улучшает охоту (проверен в
    /// debug и release — числа бит-в-бит совпадают).
    fn probe_map() -> WorldMap {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 80.0, 40.0);
        map.rates.decay = [0.0005; 4];
        let nutr = Substance::Nutrients.idx();
        let light = Substance::Light.idx();
        map.rooms[0].ca[nutr] = 220.0;
        map.rooms[0].ca[light] = 220.0;
        map
    }

    fn probe_params() -> EvolutionParams {
        // Минимальный конфиг, где рост устойчив (проверен в debug и release
        // — числа совпадают): 2 существа, короткие поколения.
        EvolutionParams {
            population: 2,
            generation_ticks: 800,
            food_per_child: 3,
            max_children: 3,
            berry_regrow: 5.0,
            target_berries: 16,
        }
    }

    /// Эволюция охотников: врождённый рефлекс (инстинкт, свёрнутый из
    /// выученного родителя) распространяется по популяции — от нуля до всех
    /// с максимальным дофамином, — а средняя добыча лучшего поколения
    /// заметно выше стартового (сытые размножаются, голодные умирают).
    #[test]
    fn evolution_spreads_reflex_and_improves_hunting() {
        let mut evo = Evolution::new(1, probe_map(), probe_params());
        let mut stats = Vec::new();
        for _ in 0..3 {
            stats.push(evo.run_generation());
        }

        // gen0 — стартовое, без инстинктов: врождённого рефлекса ещё нет.
        assert_eq!(stats[0].with_instinct, 0, "поколение 0 не носит инстинктов");
        // Каждое поколение должно выживать (популяция не вымирает).
        for s in &stats {
            assert!(s.survivors > 0, "поколение {} вымерло", s.generation);
        }

        // Врождённый рефлекс растёт: к последнему поколению ВСЕ существа
        // носят инстинкт с дофамином у потолка (рефлекс материализован).
        let last = &stats[2];
        assert_eq!(
            last.with_instinct, last.survivors,
            "к финалу все носят инстинкт: {}/{}",
            last.with_instinct, last.survivors
        );
        assert!(
            last.instinct_dopamine_avg > 7.0,
            "сила врождённого рефлекса у потолка: {}",
            last.instinct_dopamine_avg
        );

        // Охота улучшается: лучшая средняя добыча последующих поколений
        // заметно выше стартовой (навык наследуется и копится).
        let g0 = stats[0].avg_food_eaten;
        let later_best = stats[1..]
            .iter()
            .map(|s| s.avg_food_eaten)
            .fold(0.0f64, f64::max);
        assert!(
            later_best > g0 * 1.5,
            "лучшее поколение ест заметно больше стартового: {g0:.1} → {later_best:.1}"
        );
    }

    /// Эволюция детерминирована: два одинаковых прогона — одинаковая
    /// история поколений (добыча и распространение инстинктов).
    #[test]
    fn evolution_is_deterministic_from_seed() {
        let run = || -> Vec<(f64, usize, f64)> {
            let mut evo = Evolution::new(1, probe_map(), probe_params());
            (0..2)
                .map(|_| {
                    let s = evo.run_generation();
                    (s.avg_food_eaten, s.with_instinct, s.instinct_dopamine_avg)
                })
                .collect()
        };
        assert_eq!(run(), run(), "один сид — одна эволюционная история");
    }
}

