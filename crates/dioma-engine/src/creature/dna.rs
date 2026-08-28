//! -------------------------------------------------------------------------
//! Filename:    src/creature/dna.rs
//! Purpose:     РАСШИРЯЕМЫЙ ГЕНОМ — генная машина в духе Creatures
//! Description: типизированный список генов (шапка + payload) + онтогенез,
//!              сцепка с полом, мутация/кроссовер по классам и побайтовая
//!              сериализация «DNA1». Новые классы генов-органелл (расцветка
//!              Coat, ANN-глаз Eye, память Memory) добавляются в `GeneKind`,
//!              не трогая старую структуру; незнакомые классы будущих
//!              траншей сохраняются как `Raw` без потерь.
//!              ПАРАЛЛЕЛЬНЫЙ ПУТЬ (этот транш): `CreatureGenome` остаётся
//!              живым читателем поведения (40+ мест читают его поля напрямую);
//!              `Creature::dna: Genome` идёт РЯДОМ — зеркала legacy-классов
//!              синхронизируются из `CreatureGenome` при каждом размножении
//!              (`sync_legacy`), новые классы (Coat/Eye/Memory/Raw)
//!              эволюционируют ТОЛЬКО машиной. Поведение игры бит-в-бит:
//!              все rng-потребления легаси-кода идут ДО операций машины
//!              (см. `Creature::spawn_child` / `spawn_child_pair`) — число
//!              и порядок draw'ов не меняются.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use crate::brain::grow::Steering;
use crate::brain::instincts::Instinct;
use crate::brain::regulatory::{RegGene, RegGenome, Target};
use crate::brain::rules::RuleGene;
use crate::rng::XorShift;

use super::genome::{CreatureGenome, MetabolismGene, Sex};
use super::genome::{INSTINCT_SIGMA, METAB_SIGMA, RULE_SIGMA, STEERING_SIGMA};

/// Версия формата ДНК (сериализация генома). Ломающие изменения формата
/// поднимают версию; загрузка несовместимой версии — ошибка.
pub const DNA_FORMAT_VERSION: u16 = 1;

/// Класс гена — «орган» в геноме. Дополняемый: незнакомые классы будущих
/// траншей разбираются как [`GeneKind::Raw`] и сохраняются как есть.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneKind {
    /// Геном роста: параметры выращивания коннектома (`brain::grow::Steering`).
    Steering,
    /// Гены правил Пифии на регион.
    Rules,
    /// Врождённые рефлексы (эффект Болдуина).
    Instincts,
    /// Гены метаболизма тела.
    Metabolism,
    /// Регуляторный геном (сила существующих механизмов).
    Regulatory,
    /// Расцветка (рендер-ген: движок не использует, следующий транш читает).
    Coat,
    /// Вариант ANN-глаза: id модели из будущего зоопарка `.dbw` (заготовка).
    Eye,
    /// Вариант памяти (LSTM-вариант и т.п.; заготовка).
    Memory,
    /// Незнакомый/будущий класс: число — исходный kind из байтового формата.
    Raw(u16),
}

impl GeneKind {
    /// Код класса в байтовом формате. Для `Raw(k)` — само `k` (незнакомый
    /// kind сохраняется как есть и читается обратно как `Raw`).
    pub fn to_u16(self) -> u16 {
        match self {
            GeneKind::Steering => 0,
            GeneKind::Rules => 1,
            GeneKind::Instincts => 2,
            GeneKind::Metabolism => 3,
            GeneKind::Regulatory => 4,
            GeneKind::Coat => 5,
            GeneKind::Eye => 6,
            GeneKind::Memory => 7,
            GeneKind::Raw(k) => k,
        }
    }

    /// Класс по коду байтового формата; неизвестные коды — [`GeneKind::Raw`].
    /// Примечание: `Raw(k)` с известным `k` (0..=7) НЕ кругл в байтах (запишется
    /// как известный класс) — такие значения не конструируются; Raw — только
    /// для неизвестных кодов.
    pub fn from_u16(k: u16) -> Self {
        match k {
            0 => GeneKind::Steering,
            1 => GeneKind::Rules,
            2 => GeneKind::Instincts,
            3 => GeneKind::Metabolism,
            4 => GeneKind::Regulatory,
            5 => GeneKind::Coat,
            6 => GeneKind::Eye,
            7 => GeneKind::Memory,
            other => GeneKind::Raw(other),
        }
    }
}

/// Сцепка гена с полом (как в Creatures: гены на половых хромосомах выражены
/// только у одного пола). Значения 0/1/2 совпадают с семантикой `Sex` C3
/// (MALE=1, FEMALE=2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SexLink {
    /// Не сцеплен: выражен у обоих полов.
    Ignore = 0,
    /// Выражен только у самцов.
    Male = 1,
    /// Выражен только у самок.
    Female = 2,
}

/// Шапка гена — в духе Creatures: класс + онтогенез + сцепка с полом +
/// мутабельность.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeneHeader {
    pub kind: GeneKind,
    /// Версия/подвид payload (для `Raw` — идентификатор исходного класса).
    pub variant: u16,
    /// Тик мира, с которого ген выражен (совместимо с `usize::MAX`).
    pub turn_on: u64,
    /// Тик, с которого ген гаснет; `u64::MAX` = бессрочно.
    pub turn_off: u64,
    pub sex_link: SexLink,
    /// Участвует ли ген в мутации/кроссовере.
    pub mutable: bool,
}

impl GeneHeader {
    /// Шапка по умолчанию: выражена всегда, без сцепки с полом, мутабельна.
    pub fn new(kind: GeneKind) -> Self {
        Self {
            kind,
            variant: 0,
            turn_on: 0,
            turn_off: u64::MAX,
            sex_link: SexLink::Ignore,
            mutable: true,
        }
    }
}

/// Расцветка (рендер-ген). Каналы 0..1 (клэмп). Движок не читает — ген
/// живёт в геноме, мутирует и наследуется; выразит рендер-транш.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoatGene {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Default for CoatGene {
    fn default() -> Self {
        // Неокрашенный (белый) — нейтральный дикий тип.
        Self {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        }
    }
}

impl CoatGene {
    /// Каналы из [0..1] (внешние значения клэмпятся).
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Дрейф по σ с клэмпом 0..1: канал гуляет, но не выходит из диапазона.
    pub fn mutate(&self, rng: &mut XorShift) -> Self {
        let mut drift = |v: f32| (v + rng.gauss() as f32 * COAT_SIGMA).clamp(0.0, 1.0);
        Self {
            r: drift(self.r),
            g: drift(self.g),
            b: drift(self.b),
        }
    }

    /// Кроссовер по-канально: каждый канал — 50/50 от одного из родителей.
    pub fn crossover(&self, other: &Self, rng: &mut XorShift) -> Self {
        let mut pick = |x: f32, y: f32| if rng.unit() < 0.5 { x } else { y };
        Self {
            r: pick(self.r, other.r),
            g: pick(self.g, other.g),
            b: pick(self.b, other.b),
        }
    }
}

/// Вариант ANN-глаза: `variant` — id модели из будущего зоопарка `.dbw`
/// (пока ДАННЫЕ только; выражение — следующий транш). `gain` — гейн глаза.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeGene {
    pub variant: u16,
    pub gain: f32,
}

impl Default for EyeGene {
    fn default() -> Self {
        // variant 0 = заглушка (зоопарка .dbw ещё нет), gain 1.0 = нейтральный.
        Self {
            variant: 0,
            gain: 1.0,
        }
    }
}

impl EyeGene {
    /// Дрейф gain (σ, клэмп ≥ 0); вариант НЕ меняется мутацией.
    pub fn mutate(&self, rng: &mut XorShift) -> Self {
        Self {
            variant: self.variant,
            gain: (self.gain + rng.gauss() as f32 * EYE_GAIN_SIGMA).max(0.0),
        }
    }

    /// Кроссовер по-полевой: variant и gain — 50/50 от одного из родителей.
    pub fn crossover(&self, other: &Self, rng: &mut XorShift) -> Self {
        Self {
            variant: if rng.unit() < 0.5 {
                self.variant
            } else {
                other.variant
            },
            gain: if rng.unit() < 0.5 {
                self.gain
            } else {
                other.gain
            },
        }
    }
}

/// Вариант памяти (LSTM-варианты и т.п.): `variant` — id варианта,
/// `tau` — постоянная времени (тики затухания). Данные только — выражение
/// следующий транш.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryGene {
    pub variant: u16,
    pub tau: f32,
}

impl Default for MemoryGene {
    fn default() -> Self {
        // variant 0 = заглушка, tau 100 тиков — документированный дефолт
        // (движок не использует до транша выражений).
        Self {
            variant: 0,
            tau: 100.0,
        }
    }
}

impl MemoryGene {
    /// Дрейф tau (относительный, ≥ 0); вариант НЕ меняется мутацией.
    pub fn mutate(&self, rng: &mut XorShift) -> Self {
        Self {
            variant: self.variant,
            tau: (self.tau * (1.0 + rng.gauss() as f32 * MEM_TAU_SIGMA)).max(0.0),
        }
    }

    /// Кроссовер по-полевой: variant и tau — 50/50 от одного из родителей.
    pub fn crossover(&self, other: &Self, rng: &mut XorShift) -> Self {
        Self {
            variant: if rng.unit() < 0.5 {
                self.variant
            } else {
                other.variant
            },
            tau: if rng.unit() < 0.5 {
                self.tau
            } else {
                other.tau
            },
        }
    }
}

/// Payload гена: существующие типы НЕ дублируются (переиспользуются как есть),
/// новые — Coat/Eye/Memory, незнакомые — `Raw`.
#[derive(Clone, Debug, PartialEq)]
pub enum GenePayload {
    Steering(Steering),
    Rules(Vec<RuleGene>),
    Instincts(Vec<Instinct>),
    Metabolism(MetabolismGene),
    Regulatory(RegGenome),
    Coat(CoatGene),
    Eye(EyeGene),
    Memory(MemoryGene),
    /// Незнакомые/будущие гены: сохраняются как есть (без интерпретации).
    Raw { data: Vec<f32> },
}

impl GenePayload {
    /// Мутация payload: диспетчер по классу. Существующие классы дрейфуют
    /// своими мутаторами (те же сигмы, что в `CreatureGenome::mutate`),
    /// Coat — дрейф каналов, Eye/Memory — дрейф gain/tau, Raw — не мутирует.
    fn mutate(&self, rng: &mut XorShift) -> GenePayload {
        match self {
            GenePayload::Steering(s) => GenePayload::Steering(s.mutate(rng, STEERING_SIGMA)),
            GenePayload::Rules(rs) => GenePayload::Rules(
                rs.iter().map(|g| g.mutate(rng, RULE_SIGMA)).collect(),
            ),
            GenePayload::Instincts(ins) => GenePayload::Instincts(
                ins.iter().map(|i| i.mutate(rng, INSTINCT_SIGMA)).collect(),
            ),
            GenePayload::Metabolism(m) => GenePayload::Metabolism(m.mutate(rng, METAB_SIGMA)),
            GenePayload::Regulatory(reg) => GenePayload::Regulatory(reg.mutate(rng, REG_SIGMA)),
            GenePayload::Coat(c) => GenePayload::Coat(c.mutate(rng)),
            GenePayload::Eye(e) => GenePayload::Eye(e.mutate(rng)),
            GenePayload::Memory(m) => GenePayload::Memory(m.mutate(rng)),
            GenePayload::Raw { .. } => self.clone(),
        }
    }

    /// По-канальное/по-полевое смешивание НОВЫХ классов (Coat/Eye/Memory);
    /// остальные классы на уровне генома смешиваются по-генным выбором
    /// родителя (легаси-семантика кроссовера живёт в `CreatureGenome`).
    fn crossover(&self, other: &Self, rng: &mut XorShift) -> GenePayload {
        match (self, other) {
            (GenePayload::Coat(a), GenePayload::Coat(b)) => {
                GenePayload::Coat(a.crossover(b, rng))
            }
            (GenePayload::Eye(a), GenePayload::Eye(b)) => GenePayload::Eye(a.crossover(b, rng)),
            (GenePayload::Memory(a), GenePayload::Memory(b)) => {
                GenePayload::Memory(a.crossover(b, rng))
            }
            _ => self.clone(),
        }
    }
}

/// Ген: шапка + payload.
#[derive(Clone, Debug, PartialEq)]
pub struct Gene {
    pub header: GeneHeader,
    pub payload: GenePayload,
}

/// Расширяемый геном: список генов + пол (пол — ВНЕ списка генов, как в C3,
/// где ген пола читается отдельно и выбирается заново при наследовании).
#[derive(Clone, Debug, PartialEq)]
pub struct Genome {
    pub genes: Vec<Gene>,
    pub sex: Sex,
}

impl Default for Genome {
    /// Дикий тип: зеркала дефолтного легаси-генома + дефолтные органеллы
    /// (Coat/Eye/Memory). Каждый организм рождается с полным набором классов.
    fn default() -> Self {
        Self::from_creature_genome(&CreatureGenome::default())
    }
}

impl Genome {
    /// Обратный адаптер: `CreatureGenome` → ДНК. Каждый непустой legacy-класс
    /// сворачивается в ОДИН ген (Rules/Instincts — весь список в payload),
    /// пустые списки генов не дают. Дополнительно добавляются дефолтные
    /// органеллы (Coat/Eye/Memory) — миграция без дублирования данных,
    /// носители новых классов появляются у каждого организма.
    pub fn from_creature_genome(old: &CreatureGenome) -> Self {
        let mut genes = legacy_genes_from(old);
        genes.push(Gene {
            header: GeneHeader::new(GeneKind::Coat),
            payload: GenePayload::Coat(CoatGene::default()),
        });
        genes.push(Gene {
            header: GeneHeader::new(GeneKind::Eye),
            payload: GenePayload::Eye(EyeGene::default()),
        });
        genes.push(Gene {
            header: GeneHeader::new(GeneKind::Memory),
            payload: GenePayload::Memory(MemoryGene::default()),
        });
        Genome {
            genes,
            sex: old.sex,
        }
    }

    /// Прямой адаптер: ДНК → старая структура (единственный читатель
    /// поведения на переходный период). Фильтрует по онтогенезу
    /// (`turn_on`/`turn_off`), сцепке с полом и собирает `CreatureGenome`:
    /// первый выраженный ген класса для одиночных (Steering/Metabolism),
    /// конкатенация payload'ов для списковых (Rules/Instincts). Незнакомые
    /// классы (Coat/Eye/Memory/Raw/Regulatory) в старой структуре слотов не
    /// имеют и пропускаются.
    pub fn to_creature_genome(&self, age: u64) -> CreatureGenome {
        let mut g = CreatureGenome {
            sex: self.sex,
            ..CreatureGenome::default()
        };
        let mut steering_taken = false;
        let mut metab_taken = false;
        for gene in self.expressed_at(age) {
            if !sex_link_ok(gene.header.sex_link, self.sex) {
                continue;
            }
            match &gene.payload {
                GenePayload::Steering(s) if !steering_taken => {
                    g.steering = s.clone();
                    steering_taken = true;
                }
                GenePayload::Rules(rs) => g.rule_genes.extend_from_slice(rs),
                GenePayload::Instincts(ins) => g.instincts.extend(ins.iter().cloned()),
                GenePayload::Metabolism(m) if !metab_taken => {
                    g.metabolism = *m;
                    metab_taken = true;
                }
                _ => {}
            }
        }
        g
    }

    /// Синхронизация зеркал legacy-классов (параллельный путь): гены
    /// Steering/Rules/Instincts/Metabolism заменяются на свежие из легаси-
    /// генома (единственный источник правды на переходный период), пол —
    /// тоже из легаси. Новые классы (Coat/Eye/Memory/Raw/Regulatory) не
    /// трогаются — их эволюционирует только машина.
    pub fn sync_legacy(&mut self, old: &CreatureGenome) {
        self.genes.retain(|g| {
            !matches!(
                g.header.kind,
                GeneKind::Steering
                    | GeneKind::Rules
                    | GeneKind::Instincts
                    | GeneKind::Metabolism
            )
        });
        self.genes.extend(legacy_genes_from(old));
        self.sex = old.sex;
    }

    /// Онтогенез: гены, выраженные в возрасте `age` (фильтр `turn_on`/`turn_off`;
    /// `turn_off = u64::MAX` — бессрочно). Сцепка с полом здесь НЕ фильтруется —
    /// она применяется в `to_creature_genome` и при чтении.
    pub fn expressed_at(&self, age: u64) -> impl Iterator<Item = &Gene> {
        self.genes
            .iter()
            .filter(move |g| g.header.turn_on <= age && age < g.header.turn_off)
    }

    /// Мутация: диспетчер по `header.mutable` — мутабельные гены вызывают
    /// payload-ную мутацию (см. [`GenePayload::mutate`]), Raw и не-мутабельные
    /// не трогаются. Пол НЕ мутирует (C3: ген пола не дрейфует — задаётся
    /// заново при наследовании). Детерминированно от rng: один сид — один
    /// потомок.
    pub fn mutate(&self, rng: &mut XorShift) -> Self {
        let mut out = self.clone();
        for gene in out.genes.iter_mut() {
            if gene.header.mutable {
                gene.payload = gene.payload.mutate(rng);
            }
        }
        out
    }

    /// Кроссовер двух геномов: выравнивание по классу ([`GeneKind`]) —
    /// гены одного класса скрещиваются по-слотно (слот = индекс внутри
    /// класса), отсутствующий у одного родителя ген наследуется от другого.
    ///
    /// Семантика слота:
    /// - оба гена доступны (см. секс-линк) и мутабельны → 50/50 выбор
    ///   родителя; для новых классов (Coat/Eye/Memory) — по-канальное/
    ///   по-полевое смешивание payload;
    /// - `Raw` и не-мутабельные гены в кроссовере НЕ выбираются — наследуются
    ///   от родителя A целиком;
    /// - секс-линк: ген Male/Female проходит, только если пол ребёнка
    ///   совпадает; иначе берётся ген второго родителя (если его сцепка
    ///   совпадает); если оба гена выключены для пола ребёнка — слот
    ///   не наследуется (ген выключен у потомка этого пола);
    /// - пол ребёнка — «рабочая гипотеза» для секс-линка: детерминированно
    ///   из первого draw кроссовера (C3: пол выбирается при наследовании);
    ///   фактический пол потомка при рождении перезаписывается из сида
    ///   (`Sex::from_seed`), а выключенные сцепкой гены отфильтрует
    ///   `to_creature_genome`.
    ///
    /// Детерминированно от rng: один сид — один потомок пары.
    pub fn crossover(&self, other: &Self, rng: &mut XorShift) -> Self {
        let child_sex = Sex::from_seed(rng.next_u64());
        let a_groups = group_by_kind(&self.genes);
        let b_groups = group_by_kind(&other.genes);
        let mut genes: Vec<Gene> = Vec::new();
        for (kind, a_list) in &a_groups {
            let b_list = b_groups
                .iter()
                .find(|(k, _)| k == kind)
                .map(|(_, l)| l);
            let n = a_list.len().max(b_list.map_or(0, |l| l.len()));
            for i in 0..n {
                if let Some(g) =
                    resolve_slot(a_list.get(i).copied(), b_list.and_then(|l| l.get(i).copied()), child_sex, rng)
                {
                    genes.push(g);
                }
            }
        }
        // Классы, которых нет у A — целиком от B (по-слотно, с фильтром
        // секс-линка: выключенный для пола ребёнка ген не наследуется).
        for (kind, b_list) in &b_groups {
            if a_groups.iter().any(|(k, _)| k == kind) {
                continue;
            }
            for g in b_list {
                if sex_link_ok(g.header.sex_link, child_sex) {
                    genes.push((*g).clone());
                }
            }
        }
        Genome {
            genes,
            sex: child_sex,
        }
    }

    /// Первый ген класса (для эволюционной метрики и будущих траншей).
    pub fn gene_of(&self, kind: GeneKind) -> Option<&Gene> {
        self.genes.iter().find(|g| g.header.kind == kind)
    }

    /// Расцветка существа (`None` — гена нет в геноме).
    /// Природная изменчивость окраса: джиттер Coat от сида особи (±0.15 на
    /// канал, clamp 0..1). Даёт стартовую вариацию для отбора в живой арене —
    /// иначе у всех особей одинаковый дефолтный окрас и эволюции цвета нет.
    pub fn jitter_coat(&mut self, seed: u64) {
        let j = |k: u64| -> f32 {
            let h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(k.wrapping_mul(0xBF58_476D_1CE4_E5B9));
            let v = ((h ^ (h >> 30)) & 0xFFFF) as f32 / 65535.0;
            (v - 0.5) * 0.3
        };
        let base = self.coat().unwrap_or_default();
        let cl = |v: f32| v.clamp(0.0, 1.0);
        self.set_coat(CoatGene::new(
            cl(base.r + j(1)),
            cl(base.g + j(2)),
            cl(base.b + j(3)),
        ));
    }

    pub fn coat(&self) -> Option<CoatGene> {
        self.gene_of(GeneKind::Coat)
            .map(|g| match &g.payload {
                GenePayload::Coat(c) => *c,
                _ => unreachable!("класс Coat несёт CoatGene"),
            })
    }

    /// Глаз-ген (`None` — гена нет).
    pub fn eye(&self) -> Option<EyeGene> {
        self.gene_of(GeneKind::Eye)
            .map(|g| match &g.payload {
                GenePayload::Eye(e) => *e,
                _ => unreachable!("класс Eye несёт EyeGene"),
            })
    }

    /// Память-ген (`None` — гена нет).
    pub fn memory(&self) -> Option<MemoryGene> {
        self.gene_of(GeneKind::Memory)
            .map(|g| match &g.payload {
                GenePayload::Memory(m) => *m,
                _ => unreachable!("класс Memory несёт MemoryGene"),
            })
    }

    /// Data-векторы всех Raw-генов (в порядке появления в геноме).
    pub fn raw_data(&self) -> impl Iterator<Item = &[f32]> {
        self.genes.iter().filter_map(|g| match &g.payload {
            GenePayload::Raw { data } => Some(data.as_slice()),
            _ => None,
        })
    }

    /// Заменить (или добавить) ген расцветки — для тестов и будущих траншей.
    pub fn set_coat(&mut self, coat: CoatGene) {
        if let Some(g) = self.genes.iter_mut().find(|g| g.header.kind == GeneKind::Coat) {
            g.payload = GenePayload::Coat(coat);
        } else {
            self.genes.push(Gene {
                header: GeneHeader::new(GeneKind::Coat),
                payload: GenePayload::Coat(coat),
            });
        }
    }

    /// Сериализация в байты: magic «DNA1» + версия + пол + число генов +
    /// на ген: шапка + payload. Полная симметрия save/load, побайтовая
    /// детерминированность (один геном — одни байты; float'ы пишутся в
    /// точности типа: f32 → 4 байта LE, f64 → 8 байт LE).
    pub fn to_dna_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"DNA1");
        out.extend_from_slice(&DNA_FORMAT_VERSION.to_le_bytes());
        out.push(self.sex as u8);
        out.extend_from_slice(&(self.genes.len() as u32).to_le_bytes());
        for gene in &self.genes {
            write_gene(&mut out, gene);
        }
        out
    }

    /// Десериализация из байтов [`Self::to_dna_bytes`]: строгое чтение —
    /// magic, версия, пол, число генов; несовпадение/обрыв — ошибка.
    pub fn from_dna_bytes(bytes: &[u8]) -> Result<Self, DnaError> {
        let mut r = Reader { b: bytes, pos: 0 };
        if r.take(4)? != b"DNA1" {
            return Err(DnaError::BadMagic);
        }
        let version = r.u16()?;
        if version != DNA_FORMAT_VERSION {
            return Err(DnaError::UnsupportedVersion(version));
        }
        let sex = match r.u8()? {
            1 => Sex::Male,
            2 => Sex::Female,
            v => return Err(DnaError::InvalidSex(v)),
        };
        let n = r.u32()? as usize;
        let mut genes = Vec::with_capacity(n);
        for _ in 0..n {
            genes.push(read_gene(&mut r)?);
        }
        Ok(Genome { genes, sex })
    }
}

/// Ошибки разбора байтового формата «DNA1».
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DnaError {
    /// Магические байты не «DNA1» — это не наш формат.
    BadMagic,
    /// Версия формата новее/старше, чем умеет эта сборка.
    UnsupportedVersion(u16),
    /// Неверный код пола (допустимы 1 = Male, 2 = Female).
    InvalidSex(u8),
    /// Неверная сцепка с полом (допустимы 0 = Ignore, 1 = Male, 2 = Female).
    InvalidSexLink(u8),
    /// Байты закончились раньше, чем закончился геном (обрыв/порча).
    Truncated,
}

/// Сигма мутации расцветки (абсолютный дрейф канала за мутацию).
const COAT_SIGMA: f32 = 0.05;
/// Сигма мутации гейна глаза (абсолютный дрейф, клэмп ≥ 0).
const EYE_GAIN_SIGMA: f32 = 0.05;
/// Сигма мутации tau памяти (относительный дрейф, ≥ 0).
const MEM_TAU_SIGMA: f32 = 0.05;
/// Сигма мутации регуляторного генома (в `CreatureGenome` регуляторных генов
/// нет — сигма для автономного использования машины).
const REG_SIGMA: f32 = 0.05;

/// Гены-зеркала legacy-классов из легаси-генома (общий строитель для
/// `from_creature_genome` и `sync_legacy`). Каждый класс — ОДИН ген;
/// пустые списки правил/инстинктов генов не дают.
fn legacy_genes_from(old: &CreatureGenome) -> Vec<Gene> {
    let mut genes = Vec::with_capacity(4);
    genes.push(Gene {
        header: GeneHeader::new(GeneKind::Steering),
        payload: GenePayload::Steering(old.steering.clone()),
    });
    if !old.rule_genes.is_empty() {
        genes.push(Gene {
            header: GeneHeader::new(GeneKind::Rules),
            payload: GenePayload::Rules(old.rule_genes.clone()),
        });
    }
    if !old.instincts.is_empty() {
        genes.push(Gene {
            header: GeneHeader::new(GeneKind::Instincts),
            payload: GenePayload::Instincts(old.instincts.clone()),
        });
    }
    genes.push(Gene {
        header: GeneHeader::new(GeneKind::Metabolism),
        payload: GenePayload::Metabolism(old.metabolism),
    });
    genes
}

/// Выражен ли ген с данной сцепкой у существа данного пола.
fn sex_link_ok(link: SexLink, sex: Sex) -> bool {
    match link {
        SexLink::Ignore => true,
        SexLink::Male => sex == Sex::Male,
        SexLink::Female => sex == Sex::Female,
    }
}

/// Группировка генов по классу (порядок первого появления сохраняется) —
/// выравнивание кроссовера.
fn group_by_kind(genes: &[Gene]) -> Vec<(GeneKind, Vec<&Gene>)> {
    let mut out: Vec<(GeneKind, Vec<&Gene>)> = Vec::new();
    for g in genes {
        match out.iter_mut().find(|(k, _)| *k == g.header.kind) {
            Some((_, list)) => list.push(g),
            None => out.push((g.header.kind, vec![g])),
        }
    }
    out
}

/// Разрешение одного слота кроссовера (см. доку `Genome::crossover`).
fn resolve_slot(
    a: Option<&Gene>,
    b: Option<&Gene>,
    child_sex: Sex,
    rng: &mut XorShift,
) -> Option<Gene> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) => sex_link_ok(x.header.sex_link, child_sex).then(|| x.clone()),
        (None, Some(y)) => sex_link_ok(y.header.sex_link, child_sex).then(|| y.clone()),
        (Some(x), Some(y)) => {
            let x_ok = sex_link_ok(x.header.sex_link, child_sex);
            let y_ok = sex_link_ok(y.header.sex_link, child_sex);
            if !x_ok && !y_ok {
                return None;
            }
            if !x_ok {
                return Some(y.clone());
            }
            if !y_ok {
                return Some(x.clone());
            }
            // Оба доступны: Raw и не-мутабельные в кроссовере не выбираются —
            // наследуются от родителя A целиком.
            let raw = matches!(x.header.kind, GeneKind::Raw(_));
            if raw || !x.header.mutable || !y.header.mutable {
                return Some(x.clone());
            }
            // Новые классы: по-канальное/по-полевое смешивание payload
            // (шапка — от A).
            if matches!(
                x.header.kind,
                GeneKind::Coat | GeneKind::Eye | GeneKind::Memory
            ) {
                let payload = x.payload.crossover(&y.payload, rng);
                return Some(Gene {
                    header: x.header,
                    payload,
                });
            }
            if rng.unit() < 0.5 {
                Some(x.clone())
            } else {
                Some(y.clone())
            }
        }
    }
}

// ─── Сериализация «DNA1» ──────────────────────────────────────────────

/// Пишет ген целиком: шапка (kind u16 + variant u16 + turn_on u64 +
/// turn_off u64 + sex_link u8 + mutable u8) + payload.
fn write_gene(out: &mut Vec<u8>, g: &Gene) {
    out.extend_from_slice(&g.header.kind.to_u16().to_le_bytes());
    out.extend_from_slice(&g.header.variant.to_le_bytes());
    out.extend_from_slice(&g.header.turn_on.to_le_bytes());
    out.extend_from_slice(&g.header.turn_off.to_le_bytes());
    out.push(g.header.sex_link as u8);
    out.push(g.header.mutable as u8);
    match &g.payload {
        GenePayload::Steering(s) => {
            // 8 f64 + grow_steps (u64) — в порядке полей структуры.
            out.extend_from_slice(&s.noise_weight.to_le_bytes());
            out.extend_from_slice(&s.gradient_weight.to_le_bytes());
            out.extend_from_slice(&s.old_dir_weight.to_le_bytes());
            out.extend_from_slice(&s.turn_rate_deg.to_le_bytes());
            out.extend_from_slice(&s.stop_concentration.to_le_bytes());
            out.extend_from_slice(&s.bump_max.to_le_bytes());
            out.extend_from_slice(&s.bump_sigma.to_le_bytes());
            out.extend_from_slice(&s.synapse_dist.to_le_bytes());
            out.extend_from_slice(&(s.grow_steps as u64).to_le_bytes());
        }
        GenePayload::Rules(rs) => {
            out.extend_from_slice(&(rs.len() as u32).to_le_bytes());
            for r in rs {
                out.push(r.template);
                for p in &r.p {
                    out.extend_from_slice(&p.to_le_bytes());
                }
            }
        }
        GenePayload::Instincts(ins) => {
            out.extend_from_slice(&(ins.len() as u32).to_le_bytes());
            for i in ins {
                out.extend_from_slice(&(i.inputs.len() as u32).to_le_bytes());
                for (idx, drive) in &i.inputs {
                    out.extend_from_slice(&(*idx as u64).to_le_bytes());
                    out.extend_from_slice(&drive.to_le_bytes());
                }
                out.extend_from_slice(&i.dopamine.to_le_bytes());
            }
        }
        GenePayload::Metabolism(m) => {
            out.extend_from_slice(&m.cost_per_activity.to_le_bytes());
            out.extend_from_slice(&m.starve_damage.to_le_bytes());
            out.extend_from_slice(&m.cost_per_dopamine.to_le_bytes());
            out.extend_from_slice(&m.heal_rate.to_le_bytes());
        }
        GenePayload::Regulatory(reg) => {
            for slot in &reg.systemic {
                write_reg_gene(out, slot);
            }
            out.extend_from_slice(&(reg.regional.len() as u32).to_le_bytes());
            for rg in &reg.regional {
                write_reg_gene(out, rg);
            }
        }
        GenePayload::Coat(c) => {
            out.extend_from_slice(&c.r.to_le_bytes());
            out.extend_from_slice(&c.g.to_le_bytes());
            out.extend_from_slice(&c.b.to_le_bytes());
        }
        GenePayload::Eye(e) => {
            out.extend_from_slice(&e.variant.to_le_bytes());
            out.extend_from_slice(&e.gain.to_le_bytes());
        }
        GenePayload::Memory(m) => {
            out.extend_from_slice(&m.variant.to_le_bytes());
            out.extend_from_slice(&m.tau.to_le_bytes());
        }
        GenePayload::Raw { data } => {
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            for x in data {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
}

/// Один регуляторный ген: target u8 + region u64 + strength f32 +
/// dominant u8 + on_tick u64 + off_tick u64.
fn write_reg_gene(out: &mut Vec<u8>, rg: &RegGene) {
    out.push(rg.target.idx() as u8);
    out.extend_from_slice(&(rg.region as u64).to_le_bytes());
    out.extend_from_slice(&rg.strength.to_le_bytes());
    out.push(rg.dominant as u8);
    out.extend_from_slice(&(rg.on_tick as u64).to_le_bytes());
    out.extend_from_slice(&(rg.off_tick as u64).to_le_bytes());
}

/// Курсор строгого чтения байтов: любой выход за границы — `Truncated`.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], DnaError> {
        if self.pos + n > self.b.len() {
            return Err(DnaError::Truncated);
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, DnaError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, DnaError> {
        let b: [u8; 2] = self.take(2)?.try_into().expect("2 байта");
        Ok(u16::from_le_bytes(b))
    }
    fn u32(&mut self) -> Result<u32, DnaError> {
        let b: [u8; 4] = self.take(4)?.try_into().expect("4 байта");
        Ok(u32::from_le_bytes(b))
    }
    fn u64(&mut self) -> Result<u64, DnaError> {
        let b: [u8; 8] = self.take(8)?.try_into().expect("8 байт");
        Ok(u64::from_le_bytes(b))
    }
    fn f32(&mut self) -> Result<f32, DnaError> {
        let b: [u8; 4] = self.take(4)?.try_into().expect("4 байта");
        Ok(f32::from_le_bytes(b))
    }
    fn f64(&mut self) -> Result<f64, DnaError> {
        let b: [u8; 8] = self.take(8)?.try_into().expect("8 байт");
        Ok(f64::from_le_bytes(b))
    }
}

/// Читает один ген: шапка + payload по классу. Незнакомый kind → `Raw(k)`
/// (variant шапки — идентификатор), данные читаются как есть.
fn read_gene(r: &mut Reader) -> Result<Gene, DnaError> {
    let kind = GeneKind::from_u16(r.u16()?);
    let variant = r.u16()?;
    let turn_on = r.u64()?;
    let turn_off = r.u64()?;
    let sex_link = match r.u8()? {
        0 => SexLink::Ignore,
        1 => SexLink::Male,
        2 => SexLink::Female,
        v => return Err(DnaError::InvalidSexLink(v)),
    };
    let mutable = r.u8()? != 0;
    let header = GeneHeader {
        kind,
        variant,
        turn_on,
        turn_off,
        sex_link,
        mutable,
    };
    let payload = match kind {
        GeneKind::Steering => GenePayload::Steering(Steering {
            noise_weight: r.f64()?,
            gradient_weight: r.f64()?,
            old_dir_weight: r.f64()?,
            turn_rate_deg: r.f64()?,
            stop_concentration: r.f64()?,
            bump_max: r.f64()?,
            bump_sigma: r.f64()?,
            synapse_dist: r.f64()?,
            grow_steps: r.u64()? as usize,
        }),
        GeneKind::Rules => {
            let n = r.u32()? as usize;
            let mut rs = Vec::with_capacity(n);
            for _ in 0..n {
                let template = r.u8()?;
                let mut p = [0.0f32; 8];
                for v in p.iter_mut() {
                    *v = r.f32()?;
                }
                rs.push(RuleGene { template, p });
            }
            GenePayload::Rules(rs)
        }
        GeneKind::Instincts => {
            let n = r.u32()? as usize;
            let mut ins = Vec::with_capacity(n);
            for _ in 0..n {
                let m = r.u32()? as usize;
                let mut inputs = Vec::with_capacity(m);
                for _ in 0..m {
                    inputs.push((r.u64()? as usize, r.f32()?));
                }
                ins.push(Instinct {
                    inputs,
                    dopamine: r.f32()?,
                });
            }
            GenePayload::Instincts(ins)
        }
        GeneKind::Metabolism => GenePayload::Metabolism(MetabolismGene {
            cost_per_activity: r.f32()?,
            starve_damage: r.f32()?,
            cost_per_dopamine: r.f32()?,
            heal_rate: r.f32()?,
        }),
        GeneKind::Regulatory => {
            let mut systemic: [RegGene; Target::N] =
                std::array::from_fn(|_| RegGene::default());
            for slot in systemic.iter_mut() {
                *slot = read_reg_gene(r)?;
            }
            let n = r.u32()? as usize;
            let mut regional = Vec::with_capacity(n);
            for _ in 0..n {
                regional.push(read_reg_gene(r)?);
            }
            GenePayload::Regulatory(RegGenome { systemic, regional })
        }
        GeneKind::Coat => GenePayload::Coat(CoatGene {
            r: r.f32()?,
            g: r.f32()?,
            b: r.f32()?,
        }),
        GeneKind::Eye => GenePayload::Eye(EyeGene {
            variant: r.u16()?,
            gain: r.f32()?,
        }),
        GeneKind::Memory => GenePayload::Memory(MemoryGene {
            variant: r.u16()?,
            tau: r.f32()?,
        }),
        GeneKind::Raw(_) => {
            let n = r.u32()? as usize;
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                data.push(r.f32()?);
            }
            GenePayload::Raw { data }
        }
    };
    Ok(Gene { header, payload })
}

fn read_reg_gene(r: &mut Reader) -> Result<RegGene, DnaError> {
    let target = match r.u8()? {
        0 => Target::Plasticity,
        1 => Target::Instinct,
        2 => Target::Rewire,
        3 => Target::Satiety,
        _ => return Err(DnaError::Truncated), // валидных таргетов 4
    };
    Ok(RegGene {
        target,
        region: r.u64()? as usize,
        strength: r.f32()?,
        dominant: r.u8()? != 0,
        on_tick: r.u64()? as usize,
        off_tick: r.u64()? as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Полный геном со всеми классами (включая Raw и нестандартные Coat)
    /// — образец для round-trip'ов.
    fn sample_genome() -> Genome {
        let mut old = CreatureGenome::default();
        old.steering.bump_max = 6.5;
        old.rule_genes = vec![RuleGene::spiker(0.1, 0.2, 0.3, 1.0)];
        old.instincts = vec![Instinct {
            inputs: vec![(0, 4.0), (80, 8.0)],
            dopamine: 3.0,
        }];
        old.metabolism.heal_rate = 0.002;
        old.sex = Sex::Female;
        let mut g = Genome::from_creature_genome(&old);
        for gene in g.genes.iter_mut() {
            if gene.header.kind == GeneKind::Coat {
                gene.payload = GenePayload::Coat(CoatGene::new(0.9, 0.4, 0.1));
            }
        }
        g.genes.push(Gene {
            header: GeneHeader::new(GeneKind::Eye),
            payload: GenePayload::Eye(EyeGene {
                variant: 3,
                gain: 1.5,
            }),
        });
        g.genes.push(Gene {
            header: GeneHeader::new(GeneKind::Memory),
            payload: GenePayload::Memory(MemoryGene {
                variant: 7,
                tau: 250.0,
            }),
        });
        let mut reg = RegGenome::wildtype();
        reg.systemic[Target::Plasticity.idx()].strength = 1.4;
        reg.regional.push(RegGene {
            target: Target::Satiety,
            region: 3,
            strength: 0.7,
            dominant: true,
            on_tick: 100,
            off_tick: usize::MAX,
        });
        g.genes.push(Gene {
            header: GeneHeader::new(GeneKind::Regulatory),
            payload: GenePayload::Regulatory(reg),
        });
        g.genes.push(Gene {
            header: GeneHeader::new(GeneKind::Raw(999)),
            payload: GenePayload::Raw {
                data: vec![1.0, -2.5, 3.25],
            },
        });
        g.genes.push(Gene {
            header: GeneHeader {
                kind: GeneKind::Coat,
                variant: 0,
                turn_on: 100,
                turn_off: u64::MAX,
                sex_link: SexLink::Ignore,
                mutable: true,
            },
            payload: GenePayload::Coat(CoatGene::new(0.2, 0.8, 0.5)),
        });
        g
    }

    /// Genome → bytes → Genome: идентичность по ВСЕМ классам, включая Raw
    /// и Coat (побайтовая симметрия save/load).
    #[test]
    fn dna_roundtrip_bytes() {
        let g = sample_genome();
        let bytes = g.to_dna_bytes();
        let back = Genome::from_dna_bytes(&bytes).expect("валидные байты DNA1");
        assert_eq!(back, g, "байтовый round-trip: все классы, включая Raw и Coat");
    }

    /// Якорь формата: первые байты ПРОСТОГО генома (дефолтный легаси →
    /// 5 генов: Steering, Metabolism, Coat, Eye, Memory) фиксированы. Любая
    /// несовместимая правка формата ломает тест громко.
    #[test]
    fn dna_bytes_header_anchor() {
        let g = Genome::from_creature_genome(&CreatureGenome::default());
        let bytes = g.to_dna_bytes();
        let head: Vec<u8> = bytes.iter().take(24).copied().collect();
        assert_eq!(
            head,
            vec![
                b'D', b'N', b'A', b'1', // magic
                1, 0,                   // версия формата (u16 LE)
                1,                      // пол Male
                5, 0, 0, 0,             // 5 генов (S, M, Coat, Eye, Memory)
                0, 0,                   // kind Steering (u16)
                0, 0,                   // variant 0
                0, 0, 0, 0, 0, 0, 0, 0, // turn_on = 0
                0xFF,                   // первый байт turn_off = u64::MAX (бессрочно)
            ],
            "первые 24 байта формата DNA1 (дефолтный геном)"
        );
    }

    /// Raw не теряется: мутация не трогает данные, кроссовер наследует от A,
    /// `to_creature_genome` пропускает незнакомый класс, сам геном сохраняет.
    #[test]
    fn unknown_gene_preserved() {
        let mut g = Genome::from_creature_genome(&CreatureGenome::default());
        let raw = Gene {
            header: GeneHeader::new(GeneKind::Raw(999)),
            payload: GenePayload::Raw {
                data: vec![0.5, 1.5, -2.0],
            },
        };
        g.genes.push(raw.clone());
        let m = g.mutate(&mut XorShift::new(42));
        assert!(
            m.genes
                .iter()
                .any(|x| x.header.kind == GeneKind::Raw(999) && x.payload == raw.payload),
            "Raw переживает мутацию без изменений"
        );
        let c = g.crossover(&g, &mut XorShift::new(7));
        assert!(
            c.genes.iter().any(|x| x.header.kind == GeneKind::Raw(999)),
            "Raw наследуется от родителя A в кроссовере"
        );
        let old = c.to_creature_genome(0);
        assert_eq!(
            old.steering, g.to_creature_genome(0).steering,
            "to_creature_genome пропускает незнакомый класс"
        );
        assert!(
            g.genes.iter().any(|x| x.header.kind == GeneKind::Raw(999)),
            "геном-источник сохранил Raw"
        );
    }

    /// Онтогенез: ген с turn_on=100 не выражен в 50 тиков и выражен в 150;
    /// `to_creature_genome` видит окно (дефолтный метаболизм до включения,
    /// генный после).
    #[test]
    fn ontogeny_window() {
        let mut g = Genome::from_creature_genome(&CreatureGenome::default());
        g.genes.retain(|x| x.header.kind != GeneKind::Metabolism);
        g.genes.push(Gene {
            header: GeneHeader {
                kind: GeneKind::Metabolism,
                variant: 0,
                turn_on: 100,
                turn_off: u64::MAX,
                sex_link: SexLink::Ignore,
                mutable: true,
            },
            payload: GenePayload::Metabolism(MetabolismGene {
                cost_per_activity: 0.9,
                ..Default::default()
            }),
        });
        assert!(
            !g.expressed_at(50).any(|x| x.header.kind == GeneKind::Metabolism),
            "в 50 тиков поздний ген не выражен"
        );
        assert!(
            g.expressed_at(150).any(|x| x.header.kind == GeneKind::Metabolism),
            "в 150 тиков ген выражен"
        );
        let d = MetabolismGene::default();
        assert!(
            (g.to_creature_genome(50).metabolism.cost_per_activity - d.cost_per_activity).abs()
                < 1e-9,
            "до окна — дефолтный метаболизм"
        );
        assert!(
            (g.to_creature_genome(150).metabolism.cost_per_activity - 0.9).abs() < 1e-9,
            "в окне — генный метаболизм"
        );
    }

    /// Секс-линк: Male-ген у самки не выражен, у самца выражен; Ignore —
    /// у обоих.
    #[test]
    fn sex_link_semantics() {
        let mut g = Genome::from_creature_genome(&CreatureGenome::default());
        g.genes.retain(|x| x.header.kind != GeneKind::Metabolism);
        g.genes.push(Gene {
            header: GeneHeader {
                kind: GeneKind::Metabolism,
                variant: 0,
                turn_on: 0,
                turn_off: u64::MAX,
                sex_link: SexLink::Male,
                mutable: true,
            },
            payload: GenePayload::Metabolism(MetabolismGene {
                cost_per_activity: 0.9,
                ..Default::default()
            }),
        });
        g.sex = Sex::Female;
        assert!(
            (g.to_creature_genome(0).metabolism.cost_per_activity
                - MetabolismGene::default().cost_per_activity)
                .abs()
                < 1e-9,
            "Male-ген у самки не выражен"
        );
        g.sex = Sex::Male;
        assert!(
            (g.to_creature_genome(0).metabolism.cost_per_activity - 0.9).abs() < 1e-9,
            "Male-ген у самца выражен"
        );
        g.sex = Sex::Female;
        for gene in g.genes.iter_mut() {
            if gene.header.kind == GeneKind::Metabolism {
                gene.header.sex_link = SexLink::Ignore;
                gene.payload = GenePayload::Metabolism(MetabolismGene {
                    cost_per_activity: 0.5,
                    ..Default::default()
                });
            }
        }
        assert!(
            (g.to_creature_genome(0).metabolism.cost_per_activity - 0.5).abs() < 1e-9,
            "Ignore-ген выражен у обоих полов"
        );
    }

    /// Кроссовер смешивает расцветку: по-канальный выбор родителей — по
    /// диапазону сидов потомок содержит каналы и от A, и от B.
    #[test]
    fn crossover_mixes_coat() {
        let mk = |r: f32, g: f32, b: f32| {
            let mut x = Genome::from_creature_genome(&CreatureGenome::default());
            x.genes.retain(|y| y.header.kind != GeneKind::Coat);
            x.genes.push(Gene {
                header: GeneHeader::new(GeneKind::Coat),
                payload: GenePayload::Coat(CoatGene::new(r, g, b)),
            });
            x
        };
        let a = mk(0.1, 0.2, 0.3);
        let b = mk(0.9, 0.8, 0.7);
        let mut from_a = false;
        let mut from_b = false;
        for s in 0..64u64 {
            let c = a.crossover(&b, &mut XorShift::new(s));
            let coat = c.coat().expect("у потомка есть ген расцветки");
            from_a |=
                (coat.r - 0.1).abs() < 1e-6 || (coat.g - 0.2).abs() < 1e-6 || (coat.b - 0.3).abs() < 1e-6;
            from_b |=
                (coat.r - 0.9).abs() < 1e-6 || (coat.g - 0.8).abs() < 1e-6 || (coat.b - 0.7).abs() < 1e-6;
        }
        assert!(
            from_a && from_b,
            "потомок должен содержать каналы от обоих родителей"
        );
    }

    /// Адаптеры: from_creature_genome → to_creature_genome == оригинал
    /// (Steering/rules/instincts/metabolism/sex равны).
    #[test]
    fn adapter_equivalence() {
        let mut old = CreatureGenome::default();
        old.steering.bump_max = 6.5;
        old.steering.turn_rate_deg = 33.0;
        old.rule_genes = vec![
            RuleGene::spiker(0.1, 0.2, 0.3, 1.0),
            RuleGene::consolidator(0.7, 0.4),
        ];
        old.instincts = vec![Instinct {
            inputs: vec![(0, 4.0), (80, 8.0)],
            dopamine: 3.0,
        }];
        old.metabolism.heal_rate = 0.002;
        old.sex = Sex::Female;
        let back = Genome::from_creature_genome(&old).to_creature_genome(0);
        assert_eq!(back.steering, old.steering);
        assert_eq!(back.rule_genes, old.rule_genes);
        assert_eq!(back.instincts, old.instincts);
        assert_eq!(back.metabolism, old.metabolism);
        assert_eq!(back.sex, old.sex);
    }

    /// Детерминизм: mutate и crossover по одному сиду — один результат.
    #[test]
    fn determinism() {
        let g = sample_genome();
        let a = g.mutate(&mut XorShift::new(77));
        let b = g.mutate(&mut XorShift::new(77));
        assert_eq!(a, b, "мутация: один сид — один потомок");
        let c0 = g.crossover(&g, &mut XorShift::new(77));
        let c1 = g.crossover(&g, &mut XorShift::new(77));
        assert_eq!(c0, c1, "кроссовер: один сид — один потомок пары");
    }

    /// Мутация расцветки: каналы дрейфуют в границах 0..1 (не выходят).
    #[test]
    fn mutate_drifts_coat_in_bounds() {
        let g = Genome::from_creature_genome(&CreatureGenome::default());
        let coat0 = g.coat().expect("дикий тип несёт расцветку");
        assert!(
            coat0.r > 0.999 && coat0.g > 0.999 && coat0.b > 0.999,
            "дефолт — белый"
        );
        let mut changed = false;
        for _ in 0..16 {
            let c = g.mutate(&mut XorShift::new(1234)).coat().expect("расцветка есть");
            assert!(
                (0.0..=1.0).contains(&c.r) && (0.0..=1.0).contains(&c.g) && (0.0..=1.0).contains(&c.b),
                "каналы в 0..1: {c:?}"
            );
            changed |= c != coat0;
        }
        assert!(changed, "расцветка дрейфует мутацией");
    }

    /// Ошибки разбора: порча magic / версии / обрыв — громкие ошибки, не паника.
    #[test]
    fn from_dna_bytes_rejects_corruption() {
        let g = Genome::from_creature_genome(&CreatureGenome::default());
        let bytes = g.to_dna_bytes();
        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert_eq!(Genome::from_dna_bytes(&bad_magic), Err(DnaError::BadMagic));
        let mut bad_version = bytes.clone();
        bad_version[4] = 99;
        assert_eq!(
            Genome::from_dna_bytes(&bad_version),
            Err(DnaError::UnsupportedVersion(99))
        );
        assert_eq!(
            Genome::from_dna_bytes(&bytes[..bytes.len() - 1]),
            Err(DnaError::Truncated),
            "обрыв последнего байта — Truncated"
        );
    }
}
