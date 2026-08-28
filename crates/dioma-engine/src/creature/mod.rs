//! # Creature
//!
//! Игровое существо: обучаемый спайковый мозг плюс локальная физиология.
//!
//! Существо не читает карту и не ищет ягоды самостоятельно. Хост передаёт
//! [`Sense`], существо исполняет мозг и тело, затем возвращает [`Action`].
//! Координаты, столкновения, еда и социальное наблюдение принадлежат хосту
//! (`crate::sim::Arena`).
use crate::biology::metabolism::{chem, locus, Blood};
use crate::brain::grow::{grow, grow_gpu, stage_c_lite_layout};
use crate::brain::instincts::Instinct;
use crate::brain::neurons::{gains_from, Brain, PlasticityParams, BRAIN_STEPS_PER_TICK};
use crate::rng::XorShift;
use crate::Grown;

pub mod builder;
pub mod genome;
pub mod memory;
pub mod metrics;

pub use builder::CreatureBuilder;
pub use builder::BrainLayout;
pub use genome::{CreatureGenome, MetabolismGene, Sex};
pub use metrics::{BloodState, BrainStats, CreatureEvent, CreatureMetrics};

/// Регионы stage_c_lite_layout (см. brain::grow::stage_c_lite_layout).
/// Индексы регионов мозга в layout'е стадии C-lite (см. [`crate::brain::grow::stage_c_lite_layout`]).
/// Публичные: научные тулзы ([`crate::sci`]) и оверлеи группируют нейроны по регионам.
/// Ядро 0..6 — как в avoidance_layout; зрение и кора добавлены В КОНЕЦ
/// (append-only): старые индексы не сдвинуты.
pub const REGION_SL: usize = 0; // запах слева
pub const REGION_SR: usize = 1; // запах справа
pub const REGION_WL: usize = 2; // стена слева
pub const REGION_WR: usize = 3; // стена справа
pub const REGION_I: usize = 4; // тормозной интернейрон
pub const REGION_ML: usize = 5; // мотор слева (поворот влево)
pub const REGION_MR: usize = 6; // мотор справа (поворот вправо)
pub const REGION_VL: usize = 7; // зрение слева (глаз-бин)
pub const REGION_VR: usize = 8; // зрение справа (глаз-бин)
pub const REGION_CORTEX_E: usize = 9; // кора (рекуррентный хаб, возбуждающий)
pub const REGION_CORTEX_I: usize = 10; // тормозной контур коры

/// Базовый фоновый ток мозга (тонус нейронов).
const BASAL_EXT: f32 = 0.12;
/// Проходов сна при рождении: каждый инстинкт прожигается `BIRTH_BURN_PASSES`
/// раз (стимул + награда + консолидация) — «первый сон» новорождённого.
/// Несколько проходов нужны, чтобы путь успел набрать вес и закрепиться.
const BIRTH_BURN_PASSES: usize = 3;
/// Сигма мутации весов при пересадке проводки (`spawn_child`): потомок
/// наследует топологию родителя, а его сид роста мутирует веса — каждый
/// синапс получает детерминированную печать ребёнка (σ-дрейф от child_seed).
/// 5% на поколение: мягкий дрейф, не ломающий функциональную проводку.
const INHERIT_WEIGHT_SIGMA: f64 = 0.05;
/// Потолок пластичности СУЩЕСТВА (жизнь и рождение). Совпадает с масштабом
/// grown-весов: медиана ~2, хвосты десятки единиц. Движковый дефолт `w_max`
/// = 6 рассчитан на игрушечные/протокольные мозги — у grown-коннектома
/// первый же дофамин срезал бы хвосты (клэмп [0,6] на синапсах с
/// eligibility), и обучение не могло бы ПОТЕНЦИРОВАТЬ путь: охота давала
/// бы выученное ослабление вместо усиления. 64 — заведомо выше хвостов,
/// но не бесконечность: награда не может раздуть один синапс безгранично.
const PLASTICITY_W_MAX: f32 = 64.0;
/// Минимальный суммарный выученный прирост пути (ΣΔw пары регионов), при
/// котором путь сворачивается в инстинкт-ген потомку. Ниже — шум/дрейф
/// гомеостаза, не рефлекс.
const LEARNED_MIN_GAIN: f32 = 100.0;
/// Сколько сильнейших выученных путей передаётся за одно размножение
/// (защита от экспоненциального накопления инстинктов в геноме).
const LEARNED_TOP_N: usize = 3;
/// Масштаб «выученный прирост → дофамин инстинкта»: gain 500 → дофамин 5.
const LEARNED_DOPAMINE_SCALE: f32 = 100.0;
/// Ток ко-активации pre-региона при прожигании выученного инстинкта.
const LEARNED_PRE_DRIVE: f32 = 4.0;
/// Ток ко-активации post-региона при прожигании выученного инстинкта.
const LEARNED_POST_DRIVE: f32 = 8.0;
/// Прямые рефлекторные пары «сенсор → мотор» (регионы stage_c_lite_layout):
/// только они сворачиваются в инстинкты. Тормозные пути через I не
/// передаются — рефлекс это стимул→действие, а не тонкая настройка.
/// Зрения (VL/VR) здесь НЕТ намеренно: зрение учится через кору
/// обычным STDP и в инстинкт-гены не сворачивается («зрение учится,
/// а не прошивается — ради гибкой охоты», см. гейны в `assemble_impl`).
const REFLEX_PAIRS: [(usize, usize); 4] = [(0, 5), (1, 6), (2, 6), (3, 5)]; // SL→ML, SR→MR, WL→MR, WR→ML
/// Масштаб «запах [0..1] → ток сенсорного региона».
const SMELL_CURRENT: f32 = 4.0;
/// Масштаб «стена [0..1] → ток региона стены».
const WALL_CURRENT: f32 = 4.0;
/// Масштаб «зрение [0..1] → ток зрительного региона»: 0.7 × SMELL_CURRENT.
/// Зрение — дальнобойный, но слабый сигнал: подсказывает курс на дистанции,
/// где запаховый градиент ещё не различает стороны, но не заглушает обоняние
/// у рта. 0.7 подобрано на детур-тесте (см. отчёт): охота стала прямее,
/// но запах остался главным рулём у цели.
const VISION_CURRENT: f32 = 0.7 * SMELL_CURRENT;
/// Порог близости стены, с которого фиксируется облегчение (стена отодвинулась).
const RELIEF_MIN: f32 = 0.05;
/// Дофамин за облегчение (стена отодвинулась — побег успешен).
const RELIEF_DOPAMINE: f32 = 0.4;
/// Начальный гейн перекрёстных путей стены (WL→MR, WR→ML) — выше гоминга:
/// отворот от стены должен перебивать тропизм к еде, иначе существо давит
/// в стену, пока еда за ней. Пути пластичны: наказание/облегчение
/// достраивают их через STDP.
const WALL_AVOID_GAIN: f32 = 2.0;
/// Нормировка асимметрии моторных спайков → [-1..1] поворота.
/// (÷2: тик 50 мс — моторные суммы за тик удвоились (100 подшагов мозга
/// вместо 50); делим steer, чтобы диапазон и поворот за секунду остались
/// прежними: 0.25 × 0.6 рад/тик × 20 тик/с = 0.5 × 0.3 × 40.)
const STEER_SCALE: f32 = 0.25;
/// Моторный тон, при котором драйв = 1.0 (полный газ).
const TONIC_DRIVE_NORM: f32 = 0.5;
/// Порция еды за одно «съел» (хост списывает столько же у еды в мире).
pub const FOOD_PORTION: f32 = 60.0;
/// Дофамин за еду (фазический, в мозг для STDP).
const EAT_DOPAMINE: f32 = 1.0;
/// Зеркальный дофамин за «видел соседа за едой» (социальное обучение).
/// Слабее еды: подглядывание подкрепляет, но не так, как собственный обед.
const MIRROR_DOPAMINE: f32 = 0.5;
/// Негативный дофамин за удар о стену (наказание → LTD рефлекса,
/// который привёл к удару). Умеренный: серия ударов в стене должна
/// ослабить рефлекс (перестать скрести), но не убить охоту целиком.
const WALL_DOPAMINE: f32 = -0.5;
/// Сон: как сильно гомеостаз якорится на выученном за день (0..1).
const SLEEP_ANCHOR_ALPHA: f32 = 0.5;
/// Минимальная абсолютная дельта веса (от снимка рождения), которая
/// считается «выученным» и закрепляется во сне; ниже — шум, отсекается.
const SLEEP_STRONG_MIN: f32 = 0.5;
/// Насколько сильное выученное усиливается при закреплении во сне.
const SLEEP_REINFORCE: f32 = 0.15;
/// Доля слабого шума, отсекаемая во сне (вес тянется к снимку рождения).
const SLEEP_PRUNE: f32 = 0.7;

/// Набор сенсорных входов существа за один игровой тик.
///
/// Хост (`Arena` или другой симулятор) заполняет структуру до вызова
/// [`Creature::tick`]. Значения направленных сенсоров нормированы к `0..=1`;
/// `Creature` не читает карту напрямую.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sense {
    /// 0..1 — запах еды слева (относительно курса).
    pub smell_l: f32,
    /// 0..1 — запах еды справа.
    pub smell_r: f32,
    /// 0..1 — зрение: съедобная ягода в левой половине поля зрения
    /// (относительно курса, |пеленг| ≤ VISION_HALF_FOV). Дальность —
    /// VISION_RANGE (мировых юнитов); стены и комнаты непрозрачны, как
    /// для запаха. Заполняет хост ([`crate::sim::Arena::sense`]).
    pub vis_l: f32,
    /// 0..1 — зрение: съедобная ягода в правой половине поля зрения.
    pub vis_r: f32,
    /// 0..1 — близость стены слева (относительно курса).
    pub wall_l: f32,
    /// 0..1 — близость стены справа.
    pub wall_r: f32,
    /// true — еда в радиусе рта.
    pub food_near: bool,
    /// true — удар о стену на этом тике.
    pub wall: bool,
    /// Грубая клетка карты, где находится существо (для памяти места;
    /// 0 = контекст не задан). Заполняет хост.
    pub place: u32,
    /// Курс существа в радианах (0 = +x, против часовой; для памяти места).
    pub heading: f32,
    /// true — видел соседа за едой в этом/прошлом тике (зеркальный сигнал
    /// для социального обучения). Заполняет хост.
    pub see_eat: bool,
}

/// Решение существа, которое применяет игровой хост после `tick`.
///
/// `steer` задаёт направление поворота, `drive` — долю скорости, а `eat`
/// открывает рот для ближайшего съедобного объекта.
#[derive(Clone, Copy, Debug, Default)]
pub struct Action {
    /// Поворот: -1 = влево, +1 = вправо.
    pub steer: f32,
    /// Гейт скорости: 0..1 (доля `Creature::speed`).
    pub drive: f32,
    /// Открыт ли рот (съесть ближайшую еду).
    pub eat: bool,
}

/// Границы регионов, предвычисленные при сборке мозга.
#[derive(Clone, Copy)]
struct Bounds {
    sl: (usize, usize),
    sr: (usize, usize),
    wl: (usize, usize),
    wr: (usize, usize),
    vl: (usize, usize),
    vr: (usize, usize),
    ml: (usize, usize),
    mr: (usize, usize),
}

/// Существо: мозг + кровь + геном. Один вызов [`Creature::tick`] = один
/// мировой тик. Геном — то, что наследуется при размножении ([`Self::spawn_child`]).
pub struct Creature {
    pub brain: Brain,
    pub blood: Blood,
    /// Сид линии (детерминирует и геном, и рост коннектома).
    pub seed: u64,
    /// Родительская проводка, с которой существо родилось: топология
    /// (рёбра/регионы/типы) + базовые веса ДО жизни. Наследуется потомком
    /// при размножении (`spawn_child` пересаживает её, а не растит заново),
    /// и это эталон «генетических» весов: выученное за жизнь в нём не живёт.
    grown: Grown,
    /// Наследуемое: рост + инстинкты + гены метаболизма.
    pub genome: CreatureGenome,
    bounds: Bounds,
    /// Переиспользуемые буферы мозга (без аллокаций на тик / подшаг).
    /// `ext` — внешний ток (сенсоры) на тик; `spike_hist` — пачка векторов
    /// спайков на тик (батч-путь мозга).
    ext: Vec<f32>,
    spike_hist: Vec<Vec<i32>>,
    /// Эксперимент «сердце»: нейроны-пейсмейкеры (индексы) + тонический ток.
    /// `None` — выключено (поведение ровно прежнее). Включение — аддитивное:
    /// только добавка тока к выбранным нейронам (см. [`Self::enable_pacemaker`]).
    pacemaker: Option<(Vec<usize>, f32)>,
    /// Близость стены на прошлом тике — для сигнала облегчения (стена отошла).
    prev_wall: f32,
    /// Порций еды, съеденных за жизнь (мера успеха охоты — для отбора
    /// и для наследственности: см. [`Self::learned_instincts`]).
    food_eaten: u32,
    /// Память места: куда была еда в знакомых местах (гиппокампоподобный
    /// слой). Учится на «съел», ведёт к еде, когда запаха нет.
    hippo: memory::Hippocampus,
}

impl Creature {
    /// Собирает существо из сида: дефолтный геном, коннектом растёт из сида.
    pub fn new(seed: u64) -> Self {
        Self::from_genome(CreatureGenome::default(), seed)
    }

    /// То же, но с явными параметрами пластичности (протоколы и контроли:
    /// `lr = 0` отключает обучение — мозг-«камень» для сравнения).
    pub fn new_with_params(seed: u64, params: PlasticityParams) -> Self {
        let genome = CreatureGenome::default();
        let grown = grow(&stage_c_lite_layout(), &genome.steering, seed);
        Self::assemble(grown, genome, seed, params)
    }

    /// Существо с мозгом на GPU-динамике (pillars-wgpu, alpha-lif): та же
    /// наследственность, инстинкты и пластичность — мембрана считается на
    /// GPU. Без адаптера `Brain::new_gpu` тихо падает на CPU-динамику
    /// (см. `brain::neurons::Brain::new_gpu`).
    pub fn new_gpu(seed: u64) -> Self {
        let genome = CreatureGenome::default();
        let grown = grow(&stage_c_lite_layout(), &genome.steering, seed);
        Self::assemble_gpu(grown, genome, seed)
    }

    /// Существо целиком на GPU: коннектом растёт с диффузией морфогенов на
    /// wgpu (`grow_gpu`), мозг — на GPU-динамике. Без адаптера диффузия
    /// тихо падает на CPU (прометей сам выбирает бэкенд).
    pub fn new_gpu_grown(seed: u64) -> Self {
        let genome = CreatureGenome::default();
        let grown = grow_gpu(&stage_c_lite_layout(), &genome.steering, seed);
        Self::assemble_gpu(grown, genome, seed)
    }

    /// Существо для батч-пула арены (`Arena::new_pooled`): мембрана считается
    /// общим GPU-раннером ВСЕХ существ пула (один шаг на тик), пластичность —
    /// на CPU. Вызывает `assemble_pooled` (регистрирует коннектом в пуле).
    pub fn new_pooled(seed: u64, pool: &mut crate::brain::gpu_pool::BrainPool) -> Self {
        let genome = CreatureGenome::default();
        let grown = grow(&stage_c_lite_layout(), &genome.steering, seed);
        Self::assemble_pooled(grown, genome, seed, pool)
    }

    /// Существо из явного генома (наследственность): коннектом растёт из
    /// генома роста и сида, гены правил/инстинкты прицепляются к Grown,
    /// кровь строится по генам метаболизма. Мозг рождается «чистым» —
    /// выученных весов родителя здесь нет — но врождённые инстинкты генома
    /// прожигаются в него при рождении (первый сон, см. `assemble`).
    ///
    /// Жизненный потолок пластичности — `PLASTICITY_W_MAX` (в масштабе
    /// grown-весов), не движковый дефолт 6: иначе обучение не могло бы
    /// потенцировать пути, и эффект Болдуина был бы невозможен.
    pub fn from_genome(genome: CreatureGenome, seed: u64) -> Self {
        let grown = grow(&stage_c_lite_layout(), &genome.steering, seed);
        Self::from_genome_grown(genome, seed, grown)
    }

    /// Сборка из готового коннектома: гены правил и инстинкты генома
    /// прицепляются к grown, кровь строится по генам метаболизма. Это путь
    /// «пересадки проводки» — потомок собирается из grown родителя, а не
    /// из свежего роста.
    fn from_genome_grown(genome: CreatureGenome, seed: u64, mut grown: Grown) -> Self {
        grown.rule_genes = genome.rule_genes.clone();
        grown.instincts = genome.instincts.clone();
        Self::assemble(
            grown,
            genome,
            seed,
            PlasticityParams {
                w_max: PLASTICITY_W_MAX,
                ..Default::default()
            },
        )
    }

    /// Создаёт потомка из мутированного генома родителя и нового сида.
    ///
    /// Текущий обученный мозг не копируется напрямую; сильный опыт может
    /// перейти в наследуемый инстинкт через эффект Болдуина. **Проводка
    /// наследуется**: топология коннектома родителя (рёбра/регионы/типы)
    /// пересаживается ребёнку целиком — никакого роста заново. Сид роста
    /// мутирует: веса синапсов получают детерминированную печать ребёнка
    /// (σ-дрейф от child_seed), так что линия эволюционирует, а не клонируется.
    ///
    /// Выученные веса родителя НЕ наследуются: ребёнок берёт базовую
    /// проводку родителя (`self.grown` — эталон «генетических» весов ДО
    /// жизни), а не текущий мозг. Наследуются гены: рост, инстинкты, правила,
    /// метаболизм. Врождённые рефлексы (инстинкты) прожигаются в свежий
    /// мозг при рождении — ребёнок рождается с рефлексами родителей.
    ///
    /// Эффект Болдуина: сильные пути, выученные родителем за жизнь
    /// ([`Self::learned_instincts`]), сворачиваются в инстинкт-гены и
    /// дрейфуют как обычные гены — потомок рождается с врождённым
    /// рефлексом, который родитель нажил охотой. Ненаученный родитель
    /// (дельты весов ≈ 0) ничего не передаёт.
    /// Детерминированно: один сид — один потомок.
    pub fn spawn_child(&self, child_seed: u64) -> Self {
        let mut rng = XorShift::new(child_seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut genome = self.genome.mutate(&mut rng);
        genome.add_learned_instincts(self.learned_instincts(), &mut rng);
        // Пол — случайно при наследовании, как в C3 (первый ген пола читается
        // из случайного родителя): у нас «случайность» — детерминированный
        // хэш сида потомка. Пол не мутирует (`CreatureGenome::mutate` его
        // не трогает) — задаётся заново каждому ребёнку.
        genome.sex = Sex::from_seed(child_seed);
        // Пересадка проводки: топология родителя наследуется как есть,
        // сид роста мутирует — каждый синапс получает печать ребёнка.
        let mut grown = self.grown.clone();
        for w in grown.edges.iter_mut() {
            w.2 = (w.2 as f64 * (1.0 + rng.gauss() * INHERIT_WEIGHT_SIGMA)).max(0.1) as f32;
        }
        Self::from_genome_grown(genome, child_seed, grown)
    }

    /// Потомок ПАРЫ родителей (половое размножение): геном — кроссовер
    /// [`CreatureGenome::crossover`] обоих родителей + мутация сидом потомка.
    /// Проводка пересаживается от `self` (лучшего родителя пары — вызывает
    /// эволюция) с печатью сида потомка; эффект Болдуина — объединение
    /// выученных путей ОБОИХ родителей (дедупликация по паре регионов,
    /// потолок `LEARNED_TOP_N` на союз — у ребёнка не больше рефлексов,
    /// чем у одного родителя после свёртки). Пол — из сида потомка (C3).
    ///
    /// Детерминированно: тот же child_seed → тот же потомок пары (кроссовер
    /// и мутация посеяны одним rng).
    pub fn spawn_child_pair(&self, other: &Self, child_seed: u64) -> Self {
        let mut rng = XorShift::new(child_seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut genome = self.genome.crossover(other.genome(), &mut rng);
        genome = genome.mutate(&mut rng);
        // Эффект Болдуина от обоих родителей: сильные пути, нажитые каждым
        // за жизнь, сворачиваются в инстинкты — сначала объединение, потом
        // кап на союз (иначе два родителя по LEARNED_TOP_N удвоили бы геном).
        let mut learned = self.learned_instincts();
        learned.extend(other.learned_instincts());
        learned.sort_by(|x, y| y.dopamine.total_cmp(&x.dopamine));
        learned.dedup_by(|x, y| genome::instinct_pair_key(y) == genome::instinct_pair_key(x));
        learned.truncate(LEARNED_TOP_N);
        genome.add_learned_instincts(learned, &mut rng);
        // Пол — как в spawn_child: из сида потомка, а не от родителей.
        genome.sex = Sex::from_seed(child_seed);
        let mut grown = self.grown.clone();
        for w in grown.edges.iter_mut() {
            w.2 = (w.2 as f64 * (1.0 + rng.gauss() * INHERIT_WEIGHT_SIGMA)).max(0.1) as f32;
        }
        Self::from_genome_grown(genome, child_seed, grown)
    }

    /// Проводка, с которой существо родилось (топология + базовые веса).
    /// Это то, что пересаживается потомку при размножении.
    pub fn grown(&self) -> &Grown {
        &self.grown
    }

    /// Геном существа (для интроспекции/эволюции).
    pub fn genome(&self) -> &CreatureGenome {
        &self.genome
    }

    /// Сборка из выращенного коннектома и генома: гейны анатомии, мозг,
    /// кровь по генам метаболизма, границы регионов.
    ///
    /// Рождение = первый сон: врождённые инстинкты генома прожигаются в
    /// свежий мозг (`burn_into` — стимул + дофамин + консолидация + снимок
    /// памяти). Выученные веса родителя НЕ наследуются (обнуляются рождением),
    /// но врождённые рефлексы родителей ребёнок получает с первого тика.
    fn assemble(grown: Grown, genome: CreatureGenome, seed: u64, params: PlasticityParams) -> Self {
        Self::assemble_impl(grown, genome, seed, params, false)
    }

    /// Сборка с GPU-динамикой (мозг `Brain::new_gpu`); HAAP-сборок на GPU
    /// нет — новый мир существ по умолчанию LIF.
    fn assemble_gpu(grown: Grown, genome: CreatureGenome, seed: u64) -> Self {
        Self::assemble_impl(
            grown,
            genome,
            seed,
            PlasticityParams {
                w_max: PLASTICITY_W_MAX,
                ..Default::default()
            },
            true,
        )
    }

    fn assemble_impl(
        grown: Grown,
        genome: CreatureGenome,
        seed: u64,
        params: PlasticityParams,
        gpu: bool,
    ) -> Self {
        // Пол существа — из его сида (детерминированно, без ГСЧ). Единая
        // точка для всех конструкторов: `Creature::new(seed)` и потомки
        // `spawn_child` рождаются с полом `Sex::from_seed(сид)` — как в C3,
        // где пол выбирается случайно при наследовании.
        let mut genome = genome;
        genome.sex = Sex::from_seed(seed);
        // Стадия C-lite: исходящие рёбра коры выведены из мягкого гомеостаза
        // ядра (см. `PlasticityParams::homeo_exclude_pre`) — кора учится
        // дофамином, но не участвует в защите рефлексов от наказания.
        let mut params = params;
        params.homeo_exclude_pre = vec![REGION_CORTEX_E, REGION_CORTEX_I];
        // Гейны анатомии (стадия C-lite, 11 регионов). Ядро 0..6 — как раньше.
        // Зрение учится, а не прошивается — ради гибкой охоты: прямые
        // VL/VR→ML/MR аксоны схемы (stage_c_lite_layout) получают НУЛЕВОЙ гейн
        // и выпадают при сборке мозга; зрение входит в моторы ТОЛЬКО через
        // кору (VL/VR→CORTEX_E→ML/MR) обычным STDP — eligibility глобальная,
        // дофамин сам решает, какой «зрительный руль» нажить.
        // В REFLEX_PAIRS зрение не добавляем (инстинктом не рождается).
        //
        // Масштабы гейнов коры (перетюнены на стадии C-lite, см. отчёт):
        // - выход кора→моторы 0.1: у новорождённого необученные веса коры
        //   (400 нейронов × медиана 2) не должны ни глушить рефлексы, ни
        //   раздувать цель мягкого гомеостаза (target_in_sum) — иначе наказание
        //   стеной больше не перераспределяется на гоминг (гоминг гас).
        //   Слабый, но живой путь: STDP донашивает его за жизнь;
        // - рекуррентность CORTEX_E→CORTEX_E: кора почти полносвязна
        //   (фан-аут растёт квадратично с числом нейронов: морфоген собирает
        //   концы аксонов всех сом в яблочко, каждая пара замыкается).
        //   Петля с медианным весом 2 самовозбуждается тем раньше, чем больше
        //   нейронов: стадия B (90 сом, 7.9k синапсов) держала гейн 0.4,
        //   C-lite (400 сом, 154k синапсов) — 0.02 (0.4 дал бы судорогу всего
        //   хаба). Гейн выбирается по фактическому числу сом региона 9:
        //   эталонные stage_b-существа (бенчмарки) остаются би-в-бит прежними;
        // - кора→ядро (I) 0.2: тоже живёт вне цели гомеостаза ядра.
        let gains = assemble_gains(&grown);
        let brain = if gpu {
            Brain::new_gpu(&grown, gains, params)
        } else {
            Brain::new(&grown, gains, params)
        };
        Self::finish_assemble(grown, genome, seed, brain)
    }

    /// Сборка существа для батч-пула арены: та же голова, что `assemble_impl`,
    /// но мозг строится на `Dynamics::Pooled` (мембрана — общий GPU-раннер
    /// арены, пластичность — CPU). Дефолтный геном (без инстинктов) рождается
    /// «пустым» — прожигание в пуле не поддерживается (см. `Brain::new_pooled`).
    pub(crate) fn assemble_pooled(
        grown: Grown,
        genome: CreatureGenome,
        seed: u64,
        pool: &mut crate::brain::gpu_pool::BrainPool,
    ) -> Self {
        let mut genome = genome;
        genome.sex = Sex::from_seed(seed);
        let mut params = PlasticityParams {
            w_max: PLASTICITY_W_MAX,
            ..Default::default()
        };
        params.homeo_exclude_pre = vec![REGION_CORTEX_E, REGION_CORTEX_I];
        let gains = assemble_gains(&grown);
        let csr = crate::brain::neurons::build_connectome(&grown, &gains, false, 1.0, f32::INFINITY);
        // Регистрируем блок CSR в общем GPU-раннере (смещения хранит пул),
        // затем строим мозг на зеркале весов.
        pool.register(&csr);
        let brain = Brain::new_pooled(&grown, gains, params, csr);
        Self::finish_assemble(grown, genome, seed, brain)
    }

    /// Хвост сборки (общий для всех путей): кровь по генам метаболизма,
    /// границы регионов, структура существа и «первый сон» новорождённого.
    fn finish_assemble(grown: Grown, genome: CreatureGenome, seed: u64, brain: Brain) -> Self {
        let brain_n = brain.n;
        let mut blood = Blood::creature(false);
        genome.metabolism.apply_to(&mut blood);
        let bounds = Bounds {
            sl: brain.region_bounds(REGION_SL),
            sr: brain.region_bounds(REGION_SR),
            wl: brain.region_bounds(REGION_WL),
            wr: brain.region_bounds(REGION_WR),
            vl: brain.region_bounds(REGION_VL),
            vr: brain.region_bounds(REGION_VR),
            ml: brain.region_bounds(REGION_ML),
            mr: brain.region_bounds(REGION_MR),
        };
        let mut c = Creature {
            brain,
            blood,
            seed,
            genome,
            grown,
            bounds,
            ext: vec![BASAL_EXT; brain_n],
            spike_hist: Vec::with_capacity(BRAIN_STEPS_PER_TICK),
            pacemaker: None,
            prev_wall: 0.0,
            food_eaten: 0,
            hippo: memory::Hippocampus::default(),
        };
        // «Первый сон» новорождённого: врождённые рефлексы прожигаются в
        // свежий мозг (стимул + награда + консолидация). Потолок клэмпа
        // существа — `PLASTICITY_W_MAX`, так что прожиг работает в том же
        // масштабе, что и жизнь. Детерминированно (без ГСЧ) — сид линии
        // определяет всё; у дефолтного генома инстинктов нет, так что
        // обычные существа рождаются «пустыми» и учатся сами.
        for instinct in &c.genome.instincts {
            for _ in 0..BIRTH_BURN_PASSES {
                instinct.burn_into(&mut c.brain, BASAL_EXT);
            }
        }
        c
    }

    /// Пачка векторов спайков последнего тика: по одному вектору на подшаг
    /// мозга (`BRAIN_STEPS_PER_TICK` элементов), каждый — спайки нейронов
    /// подшага. Питает спайк-растр научных тулзов ([`crate::sci`]).
    pub fn spike_hist(&self) -> &[Vec<i32>] {
        &self.spike_hist
    }

    /// Эксперимент «сердце»: первые `n` нейронов тормозного региона I получают
    /// постоянный тонический ток `tonic` в дополнение к фоновому `BASAL_EXT`.
    ///
    /// LIF с током выше реобазы генерирует ритмические burst'ы сам по себе
    /// (заряд → порог → спайк → рефрактерность → сброс), как синоатриальный
    /// узел. Спайки пейсмейкера видно в научной станции как периодические
    /// полосы растра и ЧСС/ЭКГ. Вызывать ДО первого `tick`; детерминированно
    /// от сида. Тонический ток инкрементальный логикой сенсоров не трогается
    /// (регион I сенсорами не управляется).
    pub fn enable_pacemaker(&mut self, n: usize, tonic: f32) {
        let (start, end) = self.brain.region_bounds(REGION_I);
        let count = n.min(end - start);
        let indices: Vec<usize> = (start..start + count).collect();
        for &i in &indices {
            self.ext[i] += tonic;
        }
        self.pacemaker = Some((indices, tonic));
    }

    /// Нейроны-пейсмейкеры и их тонический ток (`None` — выключено).
    /// Научные тулзы считают ЧСС и подсвечивают эти нейроны в растре.
    pub fn pacemaker(&self) -> Option<(&[usize], f32)> {
        self.pacemaker.as_ref().map(|(i, t)| (i.as_slice(), *t))
    }

    /// Выключить пейсмейкер: ток больше не прикладывается, нейроны региона I
    /// возвращаются к обычному поведению. Повторный `enable_pacemaker`
    /// включит заново. (Тонический ток прикладывается каждый тик из
    /// `self.pacemaker`, поэтому выключение — просто `None`.)
    pub fn disable_pacemaker(&mut self) {
        self.pacemaker = None;
    }

    /// Исполняет один тик жизни: сенсоры → мозг → тело → действие.
    ///
    /// Вызов обновляет внутреннюю физиологию и пластичность. Хост отвечает
    /// за применение возвращённого движения и за отправку событий мира.
    pub fn tick(&mut self, sense: &Sense) -> Action {
        let relief = self.prepare_tick(sense);
        // 2. Мозг: подшаги тика, сумма спайков моторных пулов. Батч-путь:
        // все подшаги гонятся одним вызовом (GPU — без ожиданий на подшаг,
        // ридбек — только моторные суммы), пластичность — по каждому вектору
        // после (или на GPU, если обучение целиком на GPU).
        let spt = self.brain.steps_per_tick();
        let (ml, mr) = self.brain.step_batch_motor(
            &self.ext,
            spt,
            self.bounds.ml,
            self.bounds.mr,
            &mut self.spike_hist,
        );
        self.finish_tick(sense, relief, ml, mr)
    }

    /// Фаза 1 тика (общая для CPU и батч-пула): сенсоры мира +
    /// чувствительность тела → внешний ток мозга (`self.ext`). Возвращает
    /// relief — сигнал облегчения (стена отодвинулась за прошлый тик).
    ///
    /// Переиспользуемый `self.ext`: бафф на месте (нет аллокации на тик).
    /// Полный пересчёт на каждом тике (эталон 2550cae): попытка
    /// «инкрементального» обновления дельтами от prev_sense давала
    /// систематический дрейф — дельта `(s_t - s_{t-1})·g_t` вычитала
    /// старое значение, посчитанное с ПРЕЖНИМ гейном `g_{t-1}`, и ошибка
    /// `s·Δg` накапливалась, когда сенсорный гейн двигался (голод/урон/
    /// глюкоза). Мозг получал искажённый запах/стену → охота ломалась
    /// (детур-тест: 141 удар о стену, голод до нуля).
    pub fn prepare_tick(&mut self, sense: &Sense) -> f32 {
        // 0. Облегчение: стена отодвинулась за прошлый тик → поощрение побега.
        //    Дофамин кредитует именно тот отворот, который убрал стену из поля
        //    (eligibility прошлого тика ещё горячая).
        let prox = sense.wall_l.max(sense.wall_r);
        let relief = (self.prev_wall - prox).max(0.0);

        let gain = self.blood.sensor_gain();
        // Пейсмейкер берётся заранее: `ext` — mutable-займ `self.ext`, а
        // `self.pacemaker` — отдельное поле, disjoint borrow (без клона на тик).
        let pm = self.pacemaker.as_ref().map(|(i, t)| (i.as_slice(), *t));
        let ext = &mut self.ext;
        ext.fill(BASAL_EXT);
        let l_current = sense.smell_l * gain * SMELL_CURRENT;
        let r_current = sense.smell_r * gain * SMELL_CURRENT;
        for e in &mut ext[self.bounds.sl.0..self.bounds.sl.1] {
            *e += l_current;
        }
        for e in &mut ext[self.bounds.sr.0..self.bounds.sr.1] {
            *e += r_current;
        }
        let wl_current = sense.wall_l * gain * WALL_CURRENT;
        let wr_current = sense.wall_r * gain * WALL_CURRENT;
        for e in &mut ext[self.bounds.wl.0..self.bounds.wl.1] {
            *e += wl_current;
        }
        for e in &mut ext[self.bounds.wr.0..self.bounds.wr.1] {
            *e += wr_current;
        }
        // Зрение: как запах, но слабее (VISION_CURRENT = 0.7 × SMELL_CURRENT)
        // и со своим масштабом дальности — см. const'ы. Регионы VL/VR в конце
        // мозга (append-only), ток не трогает ядро рефлексов.
        let vl_current = sense.vis_l * gain * VISION_CURRENT;
        let vr_current = sense.vis_r * gain * VISION_CURRENT;
        for e in &mut ext[self.bounds.vl.0..self.bounds.vl.1] {
            *e += vl_current;
        }
        for e in &mut ext[self.bounds.vr.0..self.bounds.vr.1] {
            *e += vr_current;
        }
        // Пейсмейкер «сердце»: тонический ток прикладывается каждый тик
        // (ext пересчитывается целиком — иначе `fill` выше затёр бы ток,
        // и «сердце» билось бы один тик). Регион I сенсорами не управляется,
        // так что ток — чистый ритм-генератор, как синоатриальный узел.
        if let Some((idx, tonic)) = pm {
            for &i in idx {
                ext[i] += tonic;
            }
        }
        relief
    }

    /// Фаза 2 (батч-пул): проглотить спайки тика из ОБЩЕГО flat-буфера пула
    /// (мембрана всех существ посчитана одним GPU-шагом) и доиграть тик
    /// (кровь/награды/действие) — семантика ровно как в `tick`, но динамика
    /// пришла из пула. `relief` — из `prepare_tick` (фаза 1), `neuron_off` —
    /// смещение существа в общей сети (`BrainPool::slots()`).
    pub fn tick_pooled(
        &mut self,
        sense: &Sense,
        relief: f32,
        flat: &[u32],
        spt: usize,
        neuron_off: usize,
    ) -> Action {
        let (ml, mr) = self.brain.ingest_batched_flat(
            flat,
            spt,
            neuron_off,
            self.bounds.ml,
            self.bounds.mr,
        );
        self.finish_tick(sense, relief, ml, mr)
    }

    /// Текущий внешний ток (для пула: собрать ext всех существ перед шагом).
    pub fn pooled_ext(&self) -> &[f32] {
        &self.ext
    }

    /// Фаза 3 тика (общая): тело, награды, действие. Моторные суммы `(ml, mr)`
    /// приходят из мозга (свой раннер или пул).
    fn finish_tick(&mut self, sense: &Sense, relief: f32, ml: i32, mr: i32) -> Action {
        // 3. Тело: моторная активность, дефицит глюкозы и события — в кровь.
        self.blood.loci[locus::RATE_MOTOR] = (ml + mr) as f32;
        self.blood.loci[locus::GLUCOSE_DEFICIT] = (150.0 - self.blood.conc[chem::GLUCOSE]).max(0.0);
        if sense.wall {
            self.hit_wall();
        }
        let eat = sense.food_near && self.blood.wants_food();
        if eat {
            self.blood.ingest_food(FOOD_PORTION);
            self.food_eaten += 1;
            // Память места: еда была здесь, и я смотрел в этом направлении.
            if sense.place != 0 {
                self.hippo.remember(sense.place, sense.heading);
            }
        }
        self.blood.update();

        // 4. Награда за еду: обучение стоит глюкозы (метаболический бюджет).
        if eat && self.blood.pay_for_plasticity(EAT_DOPAMINE) {
            self.brain
                .schedule_reward(EAT_DOPAMINE * self.blood.plasticity_budget());
        }
        // Социальное обучение (зеркальный дофамин): видел соседа за едой —
        // то, что я делаю сейчас, ведёт к еде. Подкрепляет текущее поведение
        // (куда я рулю — туда еда), не требуя съесть самому. Слабее еды и
        // тоже стоит глюкозы — культура не бесплатна.
        if sense.see_eat
            && self.blood.wants_food()
            && self.blood.pay_for_plasticity(MIRROR_DOPAMINE)
        {
            self.brain
                .schedule_reward(MIRROR_DOPAMINE * self.blood.plasticity_budget());
        }
        // Облегчение: успешный отворот от стены — положительный сигнал
        // (негативное подкрепление: стена отошла = «так и делай»).
        if relief > RELIEF_MIN {
            self.brain.reward_now(relief * RELIEF_DOPAMINE);
        }
        self.prev_wall = sense.wall_l.max(sense.wall_r);

        // 5. Действия: асимметрия моторов → поворот; моторный тон → драйв.
        let mut steer =
            ((mr - ml) as f32 * STEER_SCALE * self.blood.turn_gain() as f32).clamp(-1.0, 1.0);
        // Память места: запаха нет (градиента нет), но место знакомо и голод
        // есть — руль тянет выровнять курс на запомненное направление еды
        // (таксис выключен, память ведёт). В незнакомом месте или с запахом
        // рулит обычный гоминг.
        if (sense.smell_l - sense.smell_r).abs() < 0.02
            && sense.place != 0
            && self.blood.wants_food()
        {
            if let Some((target, strength)) = self.hippo.recall(sense.place) {
                steer += memory::steer_toward(sense.heading, target) * strength;
            }
        }
        steer = steer.clamp(-1.0, 1.0);
        let drive = (self.blood.motor_tonic() / TONIC_DRIVE_NORM).clamp(0.0, 1.0);
        Action { steer, drive, eat }
    }

    // ─── Сон (ночная фаза — зовёт хост между днями) ───────────────────────

    /// Ночь: сон-консолидация. Днём мозг копит eligibility (обучение),
    /// во сне: окно обучения закрывается (eligibility обнуляется), гомеостаз
    /// якорится на выученном (`sleep_consolidate`), сильные пути дня
    /// закрепляются в долговременные веса, слабые (шум) отсекаются
    /// (`consolidate_sleep`), тело восстанавливается (`blood.sleep`:
    /// усталость/кортизол падают, глюкоза и урон — лучше). Эталон Болдуина
    /// (`memory_lt`) не трогается — выученное накапливается всю жизнь и
    /// сворачивается в инстинкт-гены при размножении.
    pub fn sleep(&mut self) {
        self.brain.clear_eligibility();
        self.brain.sleep_consolidate(SLEEP_ANCHOR_ALPHA);
        self.brain
            .consolidate_sleep(SLEEP_STRONG_MIN, SLEEP_REINFORCE, SLEEP_PRUNE);
        // Гиппокамп: слабые места забываются, воспоминания затухают.
        self.hippo.consolidate();
        self.blood.sleep();
    }

    // ─── События (хост сообщает после тика) ───────────────────────────────

    /// Удар о стену: боль в кровь + негативный дофамин в мозг. Хост зовёт в
    /// тот же тик, когда движение было заблокировано стеной — eligibility
    /// синапсов, только что гнавших существо в стену, ещё свежие, и STDP
    /// делает LTD: рефлекс слабеет (существо учится избегать). Наказание не
    /// платит глюкозой — боль жёсткий сигнал, а не метаболический расход.
    pub fn hit_wall(&mut self) {
        self.blood.bump_wall();
        self.brain.reward_now(WALL_DOPAMINE);
    }

    // ─── Наблюдение (для хоста и тестов) ────────────────────────────────

    /// Максимальная скорость тела за тик (учитывает усталость/повреждение).
    pub fn speed(&self) -> f32 {
        self.blood.speed() as f32
    }

    pub fn hunger(&self) -> f32 {
        self.blood.conc[chem::HUNGER]
    }

    pub fn glucose(&self) -> f32 {
        self.blood.conc[chem::GLUCOSE]
    }

    pub fn fatigue(&self) -> f32 {
        self.blood.conc[chem::FATIGUE]
    }

    /// Серотонин (настроение) — кровь.
    pub fn serotonin(&self) -> f32 {
        self.blood.conc[chem::SEROTONIN]
    }

    /// Кортизол (стресс).
    pub fn cortisol(&self) -> f32 {
        self.blood.conc[chem::CORTISOL]
    }

    /// Дофамин (REWARD-канал крови).
    pub fn dopamine_level(&self) -> f32 {
        self.blood.conc[chem::REWARD]
    }

    /// Хочет ли есть (рецептор рта).
    pub fn wants_food(&self) -> bool {
        self.blood.wants_food()
    }

    /// Порций еды, съеденных за жизнь (мера успеха охоты).
    pub fn food_eaten(&self) -> u32 {
        self.food_eaten
    }

    /// Пол существа (самец/самка — как в C3, значения 1/2). Пока без
    /// игровых эффектов: пол-связанные гены и размножение — транш 3.
    pub fn sex(&self) -> Sex {
        self.genome.sex
    }

    /// Сколько мест запомнено (память места — диагностика).
    pub fn places_remembered(&self) -> usize {
        self.hippo.places_remembered()
    }

    /// Свернуть выученные пути в инстинкт-гены (эффект Болдуина): сильные
    /// приросты весов относительно снимка рождения (`brain.learned_gain`)
    /// по прямым рефлекторным парам становятся инстинктами «ко-активация
    /// сенсор+мотор», которые потомку прожигаются при рождении. Возвращает
    /// до `LEARNED_TOP_N` сильнейших путей выше `LEARNED_MIN_GAIN`, дофамин
    /// пропорционален приросту. Ненаученный мозг (дельты ≈ 0) возвращает
    /// пустой список — выученное передаётся только если его реально нажили.
    pub fn learned_instincts(&self) -> Vec<Instinct> {
        let mut learned: Vec<(f32, usize, usize)> = REFLEX_PAIRS
            .iter()
            .map(|&(pr, po)| (self.brain.learned_gain(pr, po), pr, po))
            .filter(|&(gain, _, _)| gain >= LEARNED_MIN_GAIN)
            .collect();
        learned.sort_by(|a, b| b.0.total_cmp(&a.0));
        learned.truncate(LEARNED_TOP_N);
        learned
            .into_iter()
            .map(|(gain, pr, po)| {
                let (sb, se) = self.brain.region_bounds(pr);
                let (mb, me) = self.brain.region_bounds(po);
                let mut inputs = Vec::with_capacity((se - sb) + (me - mb));
                inputs.extend((sb..se).map(|i| (i, LEARNED_PRE_DRIVE)));
                inputs.extend((mb..me).map(|i| (i, LEARNED_POST_DRIVE)));
                let dopamine = (gain / LEARNED_DOPAMINE_SCALE).clamp(0.2, 8.0);
                Instinct { inputs, dopamine }
            })
            .collect()
    }

    /// Жив: мозг не уничтожен голодом.
    pub fn alive(&self) -> bool {
        self.blood.organ.damage < 1.0
    }
}

/// Гейны анатомии (стадия C-lite, 11 регионов). Ядро 0..6 — как раньше.
/// Зрение учится, а не прошивается — ради гибкой охоты: прямые
/// VL/VR→ML/MR аксоны схемы (stage_c_lite_layout) получают НУЛЕВОЙ гейн
/// и выпадают при сборке мозга; зрение входит в моторы ТОЛЬКО через
/// кору (VL/VR→CORTEX_E→ML/MR) обычным STDP — eligibility глобальная,
/// дофамин сам решает, какой «зрительный руль» нажить.
/// В REFLEX_PAIRS зрение не добавляем (инстинктом не рождается).
///
/// Масштабы гейнов коры (перетюнены на стадии C-lite, см. отчёт):
/// - выход кора→моторы 0.1: у новорождённого необученные веса коры
///   (400 нейронов × медиана 2) не должны ни глушить рефлексы, ни
///   раздувать цель мягкого гомеостаза (target_in_sum) — иначе наказание
///   стеной больше не перераспределяется на гоминг (гоминг гас).
///   Слабый, но живой путь: STDP донашивает его за жизнь;
/// - рекуррентность CORTEX_E→CORTEX_E: кора почти полносвязна
///   (фан-аут растёт квадратично с числом нейронов: морфоген собирает
///   концы аксонов всех сом в яблочко, каждая пара замыкается).
///   Петля с медианным весом 2 самовозбуждается тем раньше, чем больше
///   нейронов: стадия B (90 сом, 7.9k синапсов) держала гейн 0.4,
///   C-lite (400 сом, 154k синапсов) — 0.02 (0.4 дал бы судорогу всего
///   хаба). Гейн выбирается по фактическому числу сом региона 9:
///   эталонные stage_b-существа (бенчмарки) остаются би-в-бит прежними;
/// - кора→ядро (I) 0.2: тоже живёт вне цели гомеостаза ядра.
fn assemble_gains(grown: &Grown) -> Vec<Vec<f32>> {
    let cortex_n = grown.region_of.iter().filter(|&&r| r == REGION_CORTEX_E).count();
    let cortex_recur = if cortex_n >= 200 { 0.02 } else { 0.4 };
    gains_from(
        11,
        &[
            // ── Ядро рефлексов (как в avoidance_layout) ──
            (REGION_SL, REGION_ML, 1.0),
            (REGION_SR, REGION_MR, 1.0),
            (REGION_SL, REGION_I, 0.8),
            (REGION_SR, REGION_I, 0.8),
            (REGION_WL, REGION_MR, WALL_AVOID_GAIN),
            (REGION_WR, REGION_ML, WALL_AVOID_GAIN),
            (REGION_I, REGION_ML, 1.0),
            (REGION_I, REGION_MR, 1.0),
            // ── Зрение → кора (без прямых VL/VR→мотор) ──
            (REGION_VL, REGION_CORTEX_E, 1.0),
            (REGION_VR, REGION_CORTEX_E, 1.0),
            // ── Кора: рекуррентность, E/I-контур, слабый выход на моторы ──
            (REGION_CORTEX_E, REGION_CORTEX_E, cortex_recur),
            (REGION_CORTEX_E, REGION_CORTEX_I, 1.0),
            (REGION_CORTEX_I, REGION_CORTEX_E, 1.0),
            (REGION_CORTEX_E, REGION_ML, 0.1),
            (REGION_CORTEX_E, REGION_MR, 0.1),
            (REGION_CORTEX_I, REGION_ML, 0.1),
            (REGION_CORTEX_I, REGION_MR, 0.1),
            // Тормозной контур коры в ядро — слабый (вне цели гомеостаза).
            (REGION_CORTEX_E, REGION_I, 0.2),
            (REGION_CORTEX_I, REGION_I, 0.2),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::instincts::Instinct;

    #[test]
    fn pacemaker_fires_rhythmically() {
        let mut c = Creature::new(7);
        // Эксперимент «сердце»: без пейсмейкера регион I молчит, с током —
        // спайкает. Детерминированно от сида.
        assert!(c.pacemaker().is_none(), "по умолчанию пейсмейкер выключен");

        // Ток 0.06 — чуть выше реобазы (свип: 0.04 молчит, 0.06 → 86/200,
        // 0.08 → 200/200): LIF успевает разрядиться между спайками — ритм.
        c.enable_pacemaker(3, 0.06);
        let pm: Vec<usize> = c
            .pacemaker()
            .expect("пейсмейкер включён")
            .0
            .to_vec();
        assert!(!pm.is_empty());

        // Сенсоры молчат: если спайки есть — это пейсмейкер, а не драйв.
        let sense = Sense::default();
        let mut total = 0i64;
        let mut ticks_fired = 0u32;
        for _ in 0..200u64 {
            c.tick(&sense);
            let fired_this_tick: i64 = c
                .spike_hist()
                .iter()
                .map(|col| pm.iter().map(|&i| col[i] as i64).sum::<i64>())
                .sum();
            if fired_this_tick > 0 {
                ticks_fired += 1;
            }
            total += fired_this_tick;
        }
        assert!(total > 0, "пейсмейкер должен спайкать: {total} спайков за 200 тиков");
        assert!(
            ticks_fired > 10 && ticks_fired < 200,
            "ритм: пейсмейкер спайкает периодически, а не постоянно: {ticks_fired}/200"
        );
    }

    #[test]
    fn smell_turns_toward_food() {
        let mut c = Creature::new(7);
        // Запах слева → поворот влево (steer < 0), запах справа → вправо.
        let left = c.tick(&Sense {
            smell_l: 1.0,
            smell_r: 0.0,
            ..Default::default()
        });
        assert!(left.steer < 0.0, "запах слева: steer={}", left.steer);
        let right = c.tick(&Sense {
            smell_l: 0.0,
            smell_r: 1.0,
            ..Default::default()
        });
        assert!(right.steer > 0.0, "запах справа: steer={}", right.steer);
    }

    #[test]
    fn hungry_creature_eats_sated_does_not() {
        let mut hungry = Creature::new(11);
        hungry.blood.conc[chem::HUNGER] = 200.0;
        let a = hungry.tick(&Sense {
            food_near: true,
            ..Default::default()
        });
        assert!(a.eat, "голодный должен есть");
        assert!(hungry.blood.conc[chem::FOOD] > 0.0, "еда попала в желудок");

        let mut sated = Creature::new(11);
        let a = sated.tick(&Sense {
            food_near: true,
            ..Default::default()
        });
        assert!(!a.eat, "сытый не должен есть");
    }

    #[test]
    fn activity_accumulates_fatigue() {
        let mut c = Creature::new(13);
        c.blood.conc[chem::HUNGER] = 200.0;
        // Долго бежим вперёд (сильный драйв → RATE_MOTOR > 0).
        let f0 = c.fatigue();
        for _ in 0..300 {
            c.tick(&Sense {
                smell_l: 0.5,
                smell_r: 0.5,
                ..Default::default()
            });
        }
        assert!(
            c.fatigue() > f0 + 1.0,
            "активность должна копить утомление: {} → {}",
            f0,
            c.fatigue()
        );
    }

    /// Геттеры настроения читают канал крови: серотонин/кортизол/дофамин —
    /// именованный доступ для инспектора/игры. Еда отражается в REWARD-канале.
    #[test]
    fn mood_getters_report_blood_chems() {
        let mut c = Creature::new(7);
        // Геттеры = кровь (детерминированные начальные значения).
        assert_eq!(c.serotonin(), c.blood.conc[chem::SEROTONIN]);
        assert_eq!(c.cortisol(), c.blood.conc[chem::CORTISOL]);
        assert_eq!(c.dopamine_level(), c.blood.conc[chem::REWARD]);
        // Свежий организм: серотонин — фон, дофамин — ноль.
        assert!(c.serotonin() > 0.0, "фоновый серотонин: {}", c.serotonin());
        assert_eq!(c.dopamine_level(), 0.0, "дофамин до еды нулевой");

        // Еда → дофамин в кровь (канал REWARD через EVENT_ATE) — геттер видит.
        c.blood.conc[chem::HUNGER] = 200.0;
        c.tick(&Sense {
            food_near: true,
            ..Default::default()
        });
        assert!(
            c.dopamine_level() > 0.0,
            "еда должна дать дофамин: {}",
            c.dopamine_level()
        );
    }

    #[test]
    fn drive_positive_when_hungry() {
        let mut c = Creature::new(17);
        c.blood.conc[chem::HUNGER] = 200.0;
        let a = c.tick(&Sense::default());
        assert!(a.drive > 0.0, "голодный движется: drive={}", a.drive);
    }

    /// STDP реально обучается: серия ударов о стену (при одновременном запахе)
    /// ослабляет рефлекс запах→руль — steer к еде падает вдвое и больше (LTD
    /// через негативный дофамин). Контроль с `lr=0` не меняет рефлекс: гаснет
    /// именно пластичность, а не тревожность/утомление (боль усиливает turn_gain).
    #[test]
    fn wall_punishment_weakens_smell_steer_reflex() {
        let (before, after) = punish_reflex(13, PlasticityParams::default().lr);
        assert!(before > 0.8, "исходный рефлекс вправо: steer={before}");
        assert!(
            after < before * 0.5,
            "наказание стеной должно сильно ослабить рефлекс: {before} → {after}"
        );

        // Контроль без пластичности: рефлекс жив (его держит только LTD).
        let (_, ctrl_after) = punish_reflex(13, 0.0);
        assert!(
            ctrl_after > 0.8,
            "без обучения (lr=0) рефлекс не должен гаснуть: after={ctrl_after}"
        );
    }

    /// Направленное ощущение стены: стена слева отворачивает вправо, стена
    /// справа — влево (перекрёстная проводка WL→MR / WR→ML в avoidance_layout).
    #[test]
    fn wall_sensors_steer_away_from_wall() {
        for seed in [1u64, 7, 13, 42] {
            let mut c = Creature::new(seed);
            c.blood.conc[chem::HUNGER] = 200.0;
            let wall_l = avg_steer(
                &mut c,
                Sense {
                    wall_l: 1.0,
                    ..Default::default()
                },
            );
            let wall_r = avg_steer(
                &mut c,
                Sense {
                    wall_r: 1.0,
                    ..Default::default()
                },
            );
            assert!(
                wall_l > 0.2,
                "стена слева → поворот вправо (seed {seed}): steer={wall_l}"
            );
            assert!(
                wall_r < -0.2,
                "стена справа → поворот влево (seed {seed}): steer={wall_r}"
            );
        }
    }

    /// Ключевое свойство: направленные сенсоры стены защищают гоминг от
    /// наказания. Во время удара о стену активен и запах еды (манит в стену),
    /// и канал стены; LTD бьёт по обоим, но мягкий гомеостаз и перераспределение
    /// сохраняют запах→мотор. Без стены-сенсора тот же удар выжигает гоминг
    /// целиком (steer к еде → 0) — существо «забывает», как охотиться.
    #[test]
    fn wall_sense_protects_homing_from_punishment() {
        for seed in [1u64, 13, 42] {
            let mut with = Creature::new(seed);
            let mut without = Creature::new(seed);
            with.blood.conc[chem::HUNGER] = 200.0;
            without.blood.conc[chem::HUNGER] = 200.0;
            let punish_with = Sense {
                smell_r: 1.0,
                wall_l: 1.0,
                wall: true,
                ..Default::default()
            };
            let punish_without = Sense {
                smell_r: 1.0,
                wall: true,
                ..Default::default()
            };
            for _ in 0..400 {
                with.tick(&punish_with);
                without.tick(&punish_without);
            }
            let clean = Sense {
                smell_r: 1.0,
                ..Default::default()
            };
            let with_steer = avg_steer(&mut with, clean);
            let without_steer = avg_steer(&mut without, clean);
            assert!(
                with_steer > 0.5,
                "с сенсором стены гоминг переживает наказание (seed {seed}): {with_steer}"
            );
            assert!(
                without_steer < with_steer * 0.3,
                "без сенсора стены гоминг гаснет (seed {seed}): {with_steer} → {without_steer}"
            );
        }
    }

    /// Средний steer по 40 тикам (сенсор константный).
    fn avg_steer(c: &mut Creature, sense: Sense) -> f32 {
        let mut acc = 0.0f32;
        for _ in 0..40 {
            acc += c.tick(&sense).steer;
        }
        acc / 40.0
    }

    /// Сумма весов коннектома (для сравнения «выучено vs свежевыращено»).
    fn total_weight(c: &Creature) -> f32 {
        c.brain.csr_triplets().2.iter().sum()
    }

    /// Сумма БАЗОВЫХ весов проводки (grown — эталон до жизни).
    fn grown_weight(c: &Creature) -> f32 {
        c.grown.edges.iter().map(|e| e.2 as f64).sum::<f64>() as f32
    }

    /// Наследственность коннектома: потомок пересаживает топологию родителя
    /// (рёбра/регионы/типы — без роста заново), а сид роста мутирует веса —
    /// линия эволюционирует, а не клонируется и не растит проводку с нуля.
    #[test]
    fn spawn_child_replants_parent_wiring_with_seed_mutation() {
        let parent = Creature::new(7);
        let child = parent.spawn_child(99);
        assert_eq!(child.seed, 99, "сид потомка — его собственный");
        assert_ne!(
            child.genome.steering, parent.genome.steering,
            "геном роста должен мутировать"
        );
        // Топология наследуется: та же структура проводки (число рёбер,
        // регионы, типы), как если бы ребёнок вырос из той же схемы.
        assert_eq!(
            child.grown.edges.len(),
            parent.grown.edges.len(),
            "число синапсов наследуется"
        );
        assert_eq!(child.grown.region_of, parent.grown.region_of);
        assert_eq!(child.grown.nt, parent.grown.nt);
        // Веса дрейфуют от родительских (мутация сида роста) — не клон.
        assert!(
            (grown_weight(&child) - grown_weight(&parent)).abs() > 1.0,
            "сид роста мутирует веса: {} vs {}",
            grown_weight(&child),
            grown_weight(&parent)
        );
    }

    /// Размножение детерминировано: один сид потомка — один и тот же потомок
    /// (геном и коннектом бит-в-бит).
    #[test]
    fn spawn_child_is_deterministic_from_seed() {
        let parent = Creature::new(13);
        let a = parent.spawn_child(77);
        let b = parent.spawn_child(77);
        assert_eq!(a.genome, b.genome);
        assert_eq!(total_weight(&a), total_weight(&b));
        assert_eq!(
            a.blood.organ.cost_per_activity,
            b.blood.organ.cost_per_activity
        );
    }

    /// Пол потомка — как у C3: случайно при наследовании, но детерминированно
    /// (один сид потомка — один пол; по диапазону сидов встречаются оба пола).
    /// Пол не мутирует — у ребёнка он заново выбран из его сида.
    #[test]
    fn child_sex_is_random_per_seed_but_deterministic() {
        let parent = Creature::new(1);
        let mut males = 0;
        let mut seen: Vec<Sex> = Vec::new();
        for child_seed in 0..64u64 {
            let child = parent.spawn_child(child_seed);
            let expected = Sex::from_seed(child_seed);
            assert_eq!(
                child.sex(),
                expected,
                "пол потомка = пол из его сида (seed {child_seed})"
            );
            if child.sex() == Sex::Male {
                males += 1;
            }
            seen.push(child.sex());
        }
        assert!(
            males > 0 && males < 64,
            "по диапазону сидов должны быть оба пола: самцов {males}/64"
        );
        // Один сид — один пол (детерминированность).
        assert_eq!(
            parent.spawn_child(7).sex(),
            parent.spawn_child(7).sex(),
            "один сид потомка — один пол"
        );
        assert!(
            seen.iter().any(|&s| s == Sex::Male) && seen.iter().any(|&s| s == Sex::Female),
            "потомки бывают обоих полов"
        );
    }

    /// Наследование инстинктов и генов метаболизма: ребёнок получает их из
    /// генома родителя (с мутацией в границах), кровь собрана по генам.
    #[test]
    fn instincts_and_metabolism_inherited_with_mutation() {
        let mut genome = CreatureGenome::default();
        genome.instincts = vec![Instinct {
            inputs: vec![(3, 0.5)],
            dopamine: 2.5,
        }];
        genome.metabolism.cost_per_activity = 0.01;
        let parent = Creature::from_genome(genome, 5);
        assert_eq!(
            parent.blood.organ.cost_per_activity, 0.01,
            "кровь родителя собрана по его генам"
        );

        let child = parent.spawn_child(6);
        // Инстинкт унаследован (мутация дрейфует дофамин в [0.2, 8]).
        assert_eq!(child.genome.instincts.len(), 1, "инстинкт наследуется");
        let d = child.genome.instincts[0].dopamine;
        assert!(
            (0.2..=8.0).contains(&d),
            "дофамин инстинкта в границах: {d}"
        );
        assert!(
            (d - 2.5).abs() < 1.0,
            "дофамин дрейфует от родительского, а не прыгает: {d}"
        );
        // Метаболизм унаследован и применён к крови ребёнка.
        let cost = child.blood.organ.cost_per_activity;
        assert!(
            cost > 0.005,
            "стоимость активности унаследована от родителя (0.01 ± дрейф): {cost}"
        );
        assert!(
            (cost - 0.01).abs() < 0.005,
            "мутация метаболизма мягкая: {cost}"
        );
    }

    /// Рождение = первый сон: унаследованные инстинкты прожигаются в свежий
    /// мозг — ребёнок рождается с врождённым рефлексом родителя и реагирует
    /// сильнее с ПЕРВОГО тика (ещё до всякого обучения). Инстинкт усиливает
    /// существующий путь SL→ML (гоминг на запах слева): ко-активация всего
    /// SL и всего ML при рождении даёт eligibility всем синапсам пути,
    /// дофамин + консолидация делают его постоянным. Контроль без инстинкта
    /// на том же коннектоме и сиде отворачивает слабее. Потолок клэмпа при
    /// рождении выше рабочего w_max, иначе дофамин срезал бы grown-хвосты
    /// весов и ОСЛАБИЛ рефлекс (см. burn_into_with_cap).
    #[test]
    fn offspring_born_with_burned_in_reflex() {
        // Регионы avoidance_layout фиксированы: SL=0..20, SR=20..40,
        // WL=40..52, WR=52..64, I=64..80, ML=80..100, MR=100..120.
        let mut inputs = Vec::new();
        for i in 0..20 {
            inputs.push((i, 4.0)); // весь SL — «запах слева»
        }
        for i in 80..100 {
            inputs.push((i, 8.0)); // весь ML — «поворот влево»
        }
        for seed in [1u64, 7, 13, 42] {
            let mut genome = CreatureGenome::default();
            genome.instincts = vec![Instinct {
                inputs: inputs.clone(),
                dopamine: 2.0,
            }];
            let mut born = Creature::from_genome(genome.clone(), seed);
            let mut plain = Creature::from_genome(
                CreatureGenome {
                    instincts: Vec::new(),
                    ..genome
                },
                seed,
            );
            born.blood.conc[chem::HUNGER] = 200.0;
            plain.blood.conc[chem::HUNGER] = 200.0;
            // Слабый стимул — ниже насыщения моторов: рефлекс меряется по
            // асимметрии, а не по потолку.
            let smell_l = Sense {
                smell_l: 0.05,
                ..Default::default()
            };
            let born_steer = avg_steer(&mut born, smell_l);
            let plain_steer = avg_steer(&mut plain, smell_l);
            assert!(
                born_steer < -0.5,
                "прожитый рефлекс должен быть сильным (seed {seed}): steer={born_steer}"
            );
            assert!(
                born_steer < plain_steer - 0.1,
                "рождённый с инстинктом отворачивает заметно сильнее контроля \
                 (seed {seed}): {born_steer} vs {plain_steer}"
            );
        }
    }

    /// Эффект Болдуина: успешно охотившийся родитель передаёт потомку
    /// свёрнутый рефлекс. После охоты (запах слева + еда) путь SL→ML
    /// родителя реально потенцирован; `spawn_child` сворачивает его в
    /// инстинкт-ген, и потомок рождается с врождённым гомингом — с первого
    /// тика отворачивает к запаху. Потомок НЕНАУЧЕННОГО родителя (тот же
    /// сид) рождается без рефлекса: разница только в выученном, всё
    /// остальное (геном, сид потомка) идентично.
    #[test]
    fn hunting_parent_passes_learned_reflex_to_offspring() {
        let hunt = Sense {
            smell_l: 1.0,
            food_near: true,
            ..Default::default()
        };
        let mut trained = Creature::new(1);
        trained.blood.conc[chem::HUNGER] = 200.0;
        for _ in 0..300 {
            trained.tick(&hunt);
        }
        assert!(trained.food_eaten() > 0, "родитель должен охотиться");
        let naive = Creature::new(1);

        let learned = trained.learned_instincts();
        assert!(
            !learned.is_empty(),
            "охота должна оставить выученный путь (SL→ML)"
        );
        assert_eq!(
            naive.learned_instincts().len(),
            0,
            "ненаученный родитель ничего не передаёт"
        );

        let mut t_child = trained.spawn_child(99);
        let mut n_child = naive.spawn_child(99);
        assert_eq!(
            t_child.genome.instincts.len(),
            1,
            "потомок рождается с инстинктом, свёрнутым из выученного"
        );
        assert_eq!(
            n_child.genome.instincts.len(),
            0,
            "потомок наивного — без инстинктов"
        );
        t_child.blood.conc[chem::HUNGER] = 200.0;
        n_child.blood.conc[chem::HUNGER] = 200.0;

        // Оба потомка наследуют ОДНУ проводку родителя (один сид 1 + один
        // сид потомка 99) — различие только в прожитом инстинкте. Стимул
        // 0.05: ниже насыщения моторов, чтобы разница рефлекса была видна
        // (при 0.1 оба упираются в steer = -1).
        let smell = Sense {
            smell_l: 0.05,
            ..Default::default()
        };
        let t_steer = avg_steer(&mut t_child, smell);
        let n_steer = avg_steer(&mut n_child, smell);
        assert!(
            t_steer < -0.05,
            "потомок охотника рождён с рефлексом: steer={t_steer}"
        );
        assert!(
            t_steer < n_steer - 0.05,
            "рефлекс потомка охотника заметно сильнее наивного: {t_steer} vs {n_steer}"
        );
    }

    /// Свёртка выученного детерминирована: два потомка обученного родителя
    /// из одного сида — бит-в-бит одинаковые геномы (инстинкты совпадают).
    #[test]
    fn learned_reflex_inheritance_is_deterministic() {
        let hunt = Sense {
            smell_l: 1.0,
            food_near: true,
            ..Default::default()
        };
        let mut trained = Creature::new(13);
        trained.blood.conc[chem::HUNGER] = 200.0;
        for _ in 0..300 {
            trained.tick(&hunt);
        }
        let a = trained.spawn_child(77);
        let b = trained.spawn_child(77);
        assert_eq!(a.genome, b.genome);
        assert_eq!(
            total_weight(&a),
            total_weight(&b),
            "прожиг свёрнутого инстинкта детерминирован"
        );
    }

    /// Прожигание детерминировано: два рождения из одного генома и сида —
    /// бит-в-бит одинаковый мозг (веса совпадают).
    #[test]
    fn birth_burn_is_deterministic_from_seed() {
        let mut genome = CreatureGenome::default();
        genome.instincts = vec![Instinct {
            inputs: vec![(0, 4.0), (80, 8.0)],
            dopamine: 2.0,
        }];
        let a = Creature::from_genome(genome.clone(), 21);
        let b = Creature::from_genome(genome, 21);
        assert_eq!(total_weight(&a), total_weight(&b));
        assert_eq!(a.genome, b.genome);
    }

    /// Обнуление выученных весов: родитель учится (веса мозга меняются), но
    /// потомок пересаживает БАЗОВУЮ проводку родителя (`grown` — эталон до
    /// жизни) с печатью своего сида — выученное родителем не наследуется:
    /// потомок ближе к базе, чем к обученному мозгу родителя.
    #[test]
    fn learned_weights_are_reset_in_offspring() {
        let mut parent = Creature::new(7);
        parent.blood.conc[chem::HUNGER] = 200.0;
        let base = grown_weight(&parent); // базовые веса проводки родителя
        let w0 = total_weight(&parent);
        // Обучение: наказание стеной реально меняет веса мозга.
        let punish = Sense {
            smell_r: 1.0,
            wall_l: 1.0,
            wall: true,
            ..Default::default()
        };
        for _ in 0..200 {
            parent.tick(&punish);
        }
        let w1 = total_weight(&parent);
        assert!(
            (w1 - w0).abs() > 1.0,
            "обучение должно изменить веса родителя: {w0} → {w1}"
        );

        let child = parent.spawn_child(99);
        // Потомок наследует базовую проводку (не обученную): его веса ближе
        // к `base` (+ печать сида), чем к обученному мозгу родителя `w1`.
        assert!(
            (grown_weight(&child) - base).abs() < (w1 - base).abs() * 0.5,
            "потомок наследует базу, а не выученное: base={base} child={} w1={w1}",
            grown_weight(&child)
        );
        assert!(
            (total_weight(&child) - w1).abs() > 1.0,
            "потомок не копирует выученные веса родителя"
        );
    }

    /// Социальное обучение: существо видит соседа за едой (see_eat) и
    /// получает зеркальный дофамин — путь запах→руль потенцируется БЕЗ
    /// собственной еды. Культура распространяется стаей без генов.
    #[test]
    fn social_learning_reinforces_behavior_without_eating() {
        // Ученик: нюхает еду слева (рулит влево) и видит, как сосед ест.
        let mut learner = Creature::new(3);
        learner.blood.conc[chem::HUNGER] = 200.0;
        let watch = Sense {
            smell_l: 1.0,
            smell_r: 0.0,
            see_eat: true,
            ..Default::default()
        };
        for _ in 0..300 {
            learner.tick(&watch);
        }
        assert_eq!(learner.food_eaten(), 0, "ученик сам не ест");
        let g_learner = learner.brain.learned_gain(REGION_SL, REGION_ML);

        // Контроль: тот же прогон без зеркального сигнала.
        let mut lonely = Creature::new(3);
        lonely.blood.conc[chem::HUNGER] = 200.0;
        let plain = Sense {
            smell_l: 1.0,
            smell_r: 0.0,
            ..Default::default()
        };
        for _ in 0..300 {
            lonely.tick(&plain);
        }
        let g_lonely = lonely.brain.learned_gain(REGION_SL, REGION_ML);

        assert!(
            g_learner > 0.0,
            "зеркальный сигнал дал выученный прирост: {g_learner}"
        );
        assert!(
            g_learner > g_lonely + 10.0,
            "социальное обучение сильнее одиночного: {g_learner} vs {g_lonely}"
        );
    }

    /// Память места: существо запоминает, в каком направлении была еда
    /// в знакомом месте, и без запаха рулит туда по памяти (гоминг без
    /// таксиса). В незнакомом месте памяти нет — руля нет.
    #[test]
    fn place_memory_guides_homing_without_smell() {
        let mut c = Creature::new(7);
        c.blood.conc[chem::HUNGER] = 200.0;
        // Учим: в месте 5 еда была, когда смотрели на восток (heading 0).
        let meal = Sense {
            smell_l: 1.0,
            food_near: true,
            place: 5,
            heading: 0.0,
            ..Default::default()
        };
        for _ in 0..30 {
            assert!(c.tick(&meal).eat, "должен есть и запоминать");
        }
        assert_eq!(c.places_remembered(), 1, "место 5 запомнено");

        // Запаха нет, смотрим на север (π/2) в месте 5 → память тянет
        // на восток: поворот вправо (steer > 0).
        let mem = Sense {
            place: 5,
            heading: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        let steer_known = avg_steer(&mut c, mem);
        assert!(
            steer_known > 0.2,
            "память ведёт к еде на востоке: steer={steer_known}"
        );

        // Контроль: незнакомое место (99) без памяти — руля нет.
        let other = Sense {
            place: 99,
            heading: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        let steer_unknown = avg_steer(&mut c, other);
        assert!(
            steer_unknown.abs() < 0.1,
            "в незнакомом месте памяти нет: steer={steer_unknown}"
        );
        assert!(
            steer_known > steer_unknown + 0.2,
            "память должна заметно отличаться от пустоты: {steer_known} vs {steer_unknown}"
        );
    }

    /// Сон улучшает охоту: два существа охотятся одинаковое число тиков,
    /// но «спящий» после дня закрепляет выученное (сон: консолидация +
    /// якорь гомеостаза), а «бодрый» простаивает без сна. Спящий сохраняет
    /// больше выученного прироста пути запах→мотор (learned_gain) — его
    /// день не вымывается дрейфом, и охота опирается на закреплённый рефлекс.
    #[test]
    fn sleep_cycle_beats_awake_hunting() {
        let hunt = Sense {
            smell_l: 1.0,
            food_near: true,
            ..Default::default()
        };
        let night = Sense::default();
        let mut sleeper = Creature::new(5);
        let mut awake = Creature::new(5);
        sleeper.blood.conc[chem::HUNGER] = 200.0;
        awake.blood.conc[chem::HUNGER] = 200.0;
        for day in 0..6 {
            for _ in 0..50 {
                sleeper.tick(&hunt);
            }
            for _ in 0..50 {
                awake.tick(&hunt);
            }
            if day % 2 == 1 {
                sleeper.sleep();
                // Ночь без сна: дрейф гомеостаза вымывает выученное.
                for _ in 0..25 {
                    awake.tick(&night);
                }
            }
        }
        let g_s = sleeper.brain.learned_gain(REGION_SL, REGION_ML);
        let g_a = awake.brain.learned_gain(REGION_SL, REGION_ML);
        assert!(g_s > 0.0, "спящий сохранил выученный прирост пути: {g_s}");
        assert!(
            g_s > g_a,
            "сон закрепляет выученное лучше бодрствования: sleeper={g_s} awake={g_a}"
        );
    }

    /// Прогон: мерить steer с запахом справа, 300 раз ударить о стену с тем же
    /// запахом, мерить снова.
    fn punish_reflex(seed: u64, lr: f32) -> (f32, f32) {
        let mut c = Creature::new_with_params(
            seed,
            PlasticityParams {
                lr,
                ..Default::default()
            },
        );
        c.blood.conc[chem::HUNGER] = 200.0;
        let smell_r = Sense {
            smell_r: 1.0,
            ..Default::default()
        };
        let smell_r_wall = Sense {
            smell_r: 1.0,
            wall: true,
            ..Default::default()
        };
        let avg_steer = |c: &mut Creature| -> f32 {
            let mut s = 0.0f32;
            for _ in 0..60 {
                s += c.tick(&smell_r).steer;
            }
            s / 60.0
        };
        let before = avg_steer(&mut c);
        for _ in 0..300 {
            c.tick(&smell_r_wall);
        }
        let after = avg_steer(&mut c);
        (before, after)
    }
}
