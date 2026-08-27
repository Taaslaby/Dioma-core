# 🏗️ Architecture - Dioma Engine

Архитектурная документация Dioma Engine: дизайн, паттерны и внутреннее устройство.

## Обзор

Dioma Engine — это модульный движок для симуляции искусственных существ с обучаемыми спайковыми нейронными сетями, метаболизмом и эволюцией.

```
┌─────────────────────────────────────────────────────────┐
│                    Dioma Engine                         │
├─────────────────────────────────────────────────────────┤
│  Game Layer (dioma-render, executable)                  │
├─────────────────────────────────────────────────────────┤
│  Simulation Layer (sim, world)                          │
├─────────────────────────────────────────────────────────┤
│  Creature Layer (creature, genome, metrics)             │
├─────────────────────────────────────────────────────────┤
│  Subsystems Layer (engine, body, env)                   │
├─────────────────────────────────────────────────────────┤
│  Core Layer (rng, error, script)                        │
└─────────────────────────────────────────────────────────┘
```

---

## Модульная структура

### Core Layer

**Назначение**: Базовые утилиты и общие компоненты

```
src/
├── rng.rs          # Детерминированный RNG (XorShift)
├── error.rs        # Типы ошибок
└── script/         # Скриптовая VM для игровых объектов
    ├── compiler.rs
    ├── lexer.rs
    ├── machine.rs
    └── value.rs
```

**Ключевые компоненты**:
- `XorShift` - быстрый детерминированный PRNG
- `DiomaError` - унифицированная обработка ошибок
- `Machine` - стековая VM для скриптов

### Subsystems Layer

**Назначение**: Независимые подсистемы существа

#### Engine (Нейронная система)

```
src/engine/
├── neurons.rs      # Brain: спайковая динамика + пластичность
├── grow.rs         # Рост коннектома из генома
├── instincts.rs    # Врождённые рефлексы
├── behavior.rs     # Поведенческие паттерны
├── regulatory.rs   # Регуляторные сети
└── gpu.rs          # GPU ускорение
```

**Архитектура Brain**:
```rust
pub struct Brain {
    dynamics: Dynamics,      // Спайковая динамика (CPU/GPU)
    plast: PlastBrain,       // Пластичность (STDP)
    pre_of: Vec<usize>,      // Пресинаптические нейроны
    post_of: Vec<usize>,     // Постсинаптические нейроны
    region_of: Vec<usize>,   // Регионы нейронов
    thaw_at: Vec<usize>,     // Окна пластичности
    haap_cache: Vec<f32>,    // Кэш для HAAP динамики
}
```

**Паттерны**:
- **Strategy**: `Dynamics` (CPU/GPU реализации)
- **Template Method**: `step()` с подшагами
- **Cache**: `haap_cache` для оптимизации

#### Body (Метаболизм)

```
src/body/
├── metabolism.rs   # Blood: химия и органы
├── physiology.rs   # Физиологические процессы
├── organs.rs       # Органы и их функции
├── tissue.rs       # Ткани и регенерация
└── cell_chemistry.rs # Клеточная биохимия
```

**Архитектура Blood**:
```rust
pub struct Blood {
    conc: Vec<f32>,          // Концентрации веществ [0..255]
    loci: Vec<f32>,          // Локусы рецепторов
    decay_rates: Vec<f32>,   // Скорости распада (SIMD)
    organ: Organ,            // Состояние органов
    tick: usize,             // Внутренний счётчик
}
```

**Оптимизации**:
- **SIMD**: `wide::f32x8` для распадов
- **Batch**: Групповая обработка рецепторов
- **Cache-friendly**: Линейные массивы

#### Env (Окружение)

```
src/env/
├── map.rs          # Карта мира
├── climate.rs      # Климат и погода
├── world_clock.rs  # Временные циклы
└── math.rs         # Математические утилиты
```

### Creature Layer

**Назначение**: Интеграция подсистем в единое существо

```
src/creature/
├── mod.rs          # Creature: главный класс
├── builder.rs      # Builder Pattern
├── genome.rs       # Геном и наследование
├── memory.rs       # Память и обучение
└── metrics.rs      # Метрики и телеметрия
```

**Архитектура Creature**:
```rust
pub struct Creature {
    brain: Brain,            // Нейронная система
    blood: Blood,            // Метаболизм
    seed: u64,               // Детерминизм
    genome: CreatureGenome,  // Наследственность
    food_eaten: u32,         // Статистика
    sensor_cache: SensorCache, // Инкрементальные сенсоры
}
```

**Паттерны**:
- **Facade**: Упрощённый интерфейс к подсистемам
- **Builder**: Гибкое создание
- **Observer**: Metrics API

### Simulation Layer

**Назначение**: Игровая логика и мир

```
src/sim/
├── mod.rs          # Arena: симуляция популяции
├── evolution.rs    # Эволюционные алгоритмы
└── berry.rs        # Игровые объекты (еда)

src/world/
├── mod.rs          # World: игровой мир
├── agent.rs        # Агенты (скриптовые объекты)
└── sim.rs          # Симуляционный цикл
```

---

## Ключевые паттерны проектирования

### 1. Builder Pattern

**Проблема**: Множество конструкторов с разными параметрами

**Решение**:
```rust
pub struct CreatureBuilder {
    seed: u64,
    genome: Option<CreatureGenome>,
    plasticity_params: Option<PlasticityParams>,
    use_gpu: bool,
    gpu_grown: bool,
}

impl CreatureBuilder {
    pub fn new() -> Self { /* ... */ }
    pub fn seed(mut self, seed: u64) -> Self { /* ... */ }
    pub fn gpu(mut self, enabled: bool) -> Self { /* ... */ }
    pub fn build(self) -> Creature { /* ... */ }
}
```

**Преимущества**:
- Читаемость
- Гибкость
- Расширяемость

### 2. Strategy Pattern

**Проблема**: Разные реализации динамики (CPU/GPU)

**Решение**:
```rust
pub enum Dynamics {
    Cpu(CpuNetwork),
    Gpu(GpuNetwork),
}

impl Dynamics {
    pub fn step(&mut self, input: &[f32]) -> Vec<bool> {
        match self {
            Dynamics::Cpu(net) => net.step(input),
            Dynamics::Gpu(net) => net.step(input),
        }
    }
}
```

**Преимущества**:
- Прозрачное переключение
- Единый интерфейс
- Тестируемость

### 3. Facade Pattern

**Проблема**: Сложное взаимодействие подсистем

**Решение**:
```rust
impl Creature {
    pub fn tick(&mut self, sense: &Sense) -> Action {
        // Упрощённый интерфейс к сложной логике
        self.update_sensors_incremental(sense);
        self.brain.step(/* ... */);
        self.blood.tick();
        self.extract_action()
    }
}
```

**Преимущества**:
- Простота использования
- Инкапсуляция сложности
- Стабильный API

### 4. Object Pool

**Проблема**: Частые аллокации временных буферов

**Решение**:
```rust
pub struct Brain {
    haap_cache: Vec<f32>,    // Переиспользуемый буфер
    norm_sums: Vec<f32>,     // Переиспользуемый буфер
}

impl Brain {
    fn step(&mut self) {
        self.haap_cache.fill(0.0);  // Очистка вместо аллокации
        // Использование кэша
    }
}
```

**Преимущества**:
- Снижение аллокаций
- Предсказуемая производительность
- Меньше фрагментации памяти

### 5. Template Method

**Проблема**: Общий алгоритм с вариативными шагами

**Решение**:
```rust
impl Brain {
    pub fn step(&mut self, input: &[f32]) -> Vec<bool> {
        // Шаблон алгоритма
        self.prepare_input(input);
        let spikes = self.dynamics.step(&self.haap_cache);
        self.update_eligibility(&spikes);
        self.apply_pending_rewards();
        spikes
    }
}
```

**Преимущества**:
- Переиспользование кода
- Расширяемость
- Консистентность

---

## Потоки данных

### Tick Pipeline

```
┌─────────┐
│  Sense  │ (Сенсорный вход от мира)
└────┬────┘
     │
     v
┌─────────────────────┐
│ update_sensors()    │ (Инкрементальное обновление)
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│ Brain::step()       │ (10 подшагов спайковой динамики)
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│ Blood::tick()       │ (Метаболизм: распады, рецепторы)
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│ extract_action()    │ (Моторные нейроны → Action)
└─────────┬───────────┘
          │
          v
     ┌────────┐
     │ Action │ (Выход в мир)
     └────────┘
```

### Learning Pipeline

```
┌──────────────┐
│ Experience   │ (Взаимодействие с миром)
└──────┬───────┘
       │
       v
┌──────────────────────┐
│ Eligibility traces   │ (STDP: pre-post корреляции)
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│ Reward signal        │ (Дофамин: еда, боль, социальное)
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│ Weight update        │ (Δw = η·D·e)
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│ Normalization        │ (Стабилизация весов)
└──────┬───────────────┘
       │
       v
┌──────────────────────┐
│ Sleep consolidation  │ (Периодическое закрепление)
└──────────────────────┘
```

---

## Оптимизации производительности

### Memory Layout

**Проблема**: Cache misses при обходе структур

**Решение**: Structure of Arrays (SoA)
```rust
// Плохо: Array of Structures (AoS)
struct Synapse {
    pre: usize,
    post: usize,
    weight: f32,
    eligibility: f32,
}
let synapses: Vec<Synapse>;

// Хорошо: Structure of Arrays (SoA)
struct Brain {
    pre_of: Vec<usize>,
    post_of: Vec<usize>,
    weights: Vec<f32>,
    eligibility: Vec<f32>,
}
```

**Эффект**: Лучшая локальность данных, меньше cache misses

### SIMD Vectorization

**Проблема**: Скалярная обработка массивов

**Решение**: `wide::f32x8` для параллельной обработки
```rust
use wide::f32x8;

// Обработка 8 элементов за раз
for chunk in 0..(N / 8) {
    let base = chunk * 8;
    let vec_a = f32x8::new([/* 8 элементов */]);
    let vec_b = f32x8::new([/* 8 элементов */]);
    let result = vec_a * vec_b;
    // Запись результата
}
```

**Эффект**: 3-4x ускорение на поддерживаемых CPU

### Incremental Updates

**Проблема**: Полный пересчёт при малых изменениях

**Решение**: Кэширование и дифференциальные обновления
```rust
struct SensorCache {
    prev_values: [f32; N],
    dirty_flags: [bool; N],
}

fn update_incremental(&mut self, new_values: &[f32]) {
    for i in 0..N {
        if (new_values[i] - self.prev_values[i]).abs() > EPSILON {
            self.dirty_flags[i] = true;
            self.prev_values[i] = new_values[i];
        }
    }
    // Обновить только dirty регионы
}
```

**Эффект**: 10-15% ускорение при медленных изменениях

---

## Масштабируемость

### Параллелизм

**Уровень 1**: Batch обработка существ
```rust
use rayon::prelude::*;

creatures.par_iter_mut()
    .zip(senses.par_iter())
    .for_each(|(creature, sense)| {
        creature.tick(sense);
    });
```

**Уровень 2**: GPU ускорение
```rust
// Перенос динамики на GPU
let dynamics = Dynamics::Gpu(GpuNetwork::new(/* ... */));
```

**Уровень 3**: Распределённая симуляция
```rust
// Будущее: MPI/gRPC для кластеров
```

### Профилирование

**Инструменты**:
```bash
# CPU профилирование
cargo flamegraph --bench bench_all

# Memory профилирование
cargo valgrind --bench bench_all

# GPU профилирование
nsys profile ./target/release/dioma-engine
```

**Метрики**:
- Ticks per second (TPS)
- Memory usage per creature
- GPU utilization
- Cache hit rate

---

## Тестирование

### Уровни тестирования

**Unit тесты**: Отдельные компоненты
```rust
#[test]
fn brain_step_deterministic() {
    let brain1 = Brain::new(/* seed=42 */);
    let brain2 = Brain::new(/* seed=42 */);
    
    for _ in 0..100 {
        let out1 = brain1.step(&input);
        let out2 = brain2.step(&input);
        assert_eq!(out1, out2);
    }
}
```

**Integration тесты**: Взаимодействие подсистем
```rust
#[test]
fn creature_learning() {
    let mut creature = Creature::new(42);
    
    // Обучение: запах → поворот
    for _ in 0..1000 {
        let sense = Sense { smell_l: 1.0, ..default() };
        let action = creature.tick(&sense);
        // Проверка обучения
    }
}
```

**Benchmark тесты**: Производительность
```rust
#[bench]
fn bench_creature_tick(b: &mut Bencher) {
    let mut creature = Creature::new(42);
    let sense = Sense::default();
    
    b.iter(|| {
        creature.tick(&sense)
    });
}
```

### Детерминизм

**Гарантии**:
- Фиксированный seed → идентичное поведение
- Независимость от порядка выполнения
- Воспроизводимость на разных платформах

**Проверка**:
```rust
#[test]
fn determinism_across_runs() {
    let trace1 = run_simulation(seed=42, ticks=1000);
    let trace2 = run_simulation(seed=42, ticks=1000);
    assert_eq!(trace1, trace2);
}
```

---

## Расширяемость

### Добавление новых регионов мозга

```rust
// 1. Определить регион в layout
pub fn custom_layout() -> Layout {
    Layout {
        regions: vec![
            Region { name: "custom", size: 50, neuron_type: NeuronType::Lif },
            // ...
        ],
        // ...
    }
}

// 2. Добавить связи
pub fn custom_connections() -> Vec<Connection> {
    vec![
        Connection { from: "custom", to: "motor", weight: 1.0 },
        // ...
    ]
}

// 3. Использовать
let grown = grow(&custom_layout(), &genome, seed);
let creature = Creature::from_genome_grown(genome, seed, grown);
```

### Добавление новых химических веществ

```rust
// 1. Расширить chem модуль
pub mod chem {
    pub const CUSTOM: usize = 9;
    pub const N: usize = 10; // Было 9
}

// 2. Добавить в Blood
impl Blood {
    pub fn new() -> Self {
        let mut conc = vec![0.0; chem::N];
        conc[chem::CUSTOM] = 50.0; // Начальное значение
        // ...
    }
}

// 3. Добавить распад
let decay_rates = vec![
    // ...
    0.95, // CUSTOM decay rate
];
```

### Добавление новых метрик

```rust
// 1. Расширить CreatureMetrics
pub struct CreatureMetrics {
    // Существующие поля
    pub custom_metric: f32,
}

// 2. Реализовать вычисление
impl Creature {
    pub fn metrics(&self) -> CreatureMetrics {
        CreatureMetrics {
            // ...
            custom_metric: self.compute_custom_metric(),
        }
    }
}
```

---

## Зависимости

### Основные

```toml
[dependencies]
# Нейронная динамика
pillars = { path = "../SiMuse/crates/pillars" }
pillars-wgpu = { path = "../SiMuse/crates/pillars-wgpu" }

# SIMD оптимизации
wide = "0.7"

# Математика
glam = "0.24"

# Сериализация (опционально)
serde = { version = "1.0", optional = true }
```

### Опциональные

```toml
[dev-dependencies]
criterion = "0.5"  # Бенчмарки
proptest = "1.0"   # Property-based тестирование
```

---

## Roadmap

### Ближайшие улучшения

1. **Async API** - неблокирующие операции
2. **Сериализация** - сохранение/загрузка состояния
3. **Error Handling** - Result вместо паник
4. **Конфигурация** - TOML файлы для параметров

### Долгосрочные цели

1. **Распределённая симуляция** - кластеры GPU
2. **Визуализация** - встроенный 3D рендер
3. **Интерактивность** - real-time управление
4. **Экосистема** - хищники, симбиоз, паразиты

---

## Заключение

Dioma Engine построен на принципах:
- ✅ **Модульность** - независимые подсистемы
- ✅ **Производительность** - SIMD, GPU, кэширование
- ✅ **Расширяемость** - паттерны проектирования
- ✅ **Тестируемость** - детерминизм и изоляция
- ✅ **Документированность** - понятный код и API

**Архитектура готова к масштабированию и эволюции** 🚀
