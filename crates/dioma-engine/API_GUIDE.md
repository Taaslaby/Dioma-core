# 🎯 API Guide - Dioma Engine

Полное руководство по использованию API Dioma Engine для создания и управления искусственными существами.

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Builder Pattern](#builder-pattern)
3. [Жизненный цикл существа](#жизненный-цикл-существа)
4. [Сенсоры и действия](#сенсоры-и-действия)
5. [Метрики и мониторинг](#метрики-и-мониторинг)
6. [Геном и наследование](#геном-и-наследование)
7. [GPU ускорение](#gpu-ускорение)
8. [Продвинутые техники](#продвинутые-техники)

---

## Быстрый старт

### Минимальный пример

```rust
use dioma_engine::creature::{Creature, Sense};

fn main() {
    // Создание существа
    let mut creature = Creature::new(42);
    
    // Создание сенсорного входа
    let sense = Sense {
        smell_l: 0.5,    // Запах слева
        smell_r: 0.3,    // Запах справа
        wall_l: 0.0,     // Расстояние до стены слева
        wall_r: 0.1,     // Расстояние до стены справа
        food_near: false,
        wall: false,
        place: 0,
        heading: 0.0,
        see_eat: false,
    };
    
    // Выполнение одного тика
    let action = creature.tick(&sense);
    
    println!("Steer: {:.2}", action.steer);  // Поворот
    println!("Drive: {:.2}", action.drive);  // Скорость
    println!("Eat: {}", action.eat);         // Попытка съесть
}
```

### Установка

```toml
[dependencies]
dioma-engine = "0.2.0"
```

---

## Builder Pattern

### Основы

Builder Pattern предоставляет гибкий способ создания существ с различными конфигурациями:

```rust
use dioma_engine::creature::CreatureBuilder;

// Простое существо
let creature = CreatureBuilder::new()
    .seed(42)
    .build();
```

### Параметры пластичности

Контроль обучения мозга:

```rust
let creature = CreatureBuilder::new()
    .seed(123)
    .plasticity_lr(0.01)        // Learning rate
    .plasticity_w_max(32.0)     // Максимальный вес синапса
    .build();
```

**Рекомендации**:
- `lr = 0.0` - отключает обучение (мозг-"камень")
- `lr = 0.001-0.01` - медленное обучение
- `lr = 0.05-0.1` - быстрое обучение
- `w_max = 6.0` - дефолт (балансирует стабильность/гибкость)
- `w_max = 32.0` - высокая пластичность (эффект Болдуина)

### GPU ускорение

```rust
// GPU динамика (мозг на GPU)
let creature = CreatureBuilder::new()
    .seed(456)
    .gpu(true)
    .build();

// Полный GPU (рост + динамика)
let creature = CreatureBuilder::new()
    .seed(789)
    .gpu_grown(true)  // Автоматически включает gpu(true)
    .build();
```

**Требования**:
- Vulkan/Metal/DX12 совместимый GPU
- Без GPU автоматически fallback на CPU

### Кастомный геном

```rust
use dioma_engine::creature::CreatureGenome;

let mut genome = CreatureGenome::default();
genome.steering.diffusion_rate = 0.5;
genome.metabolism.glucose_cost = 0.15;

let creature = CreatureBuilder::new()
    .seed(999)
    .genome(genome)
    .build();
```

---

## Жизненный цикл существа

### Основной цикл

```rust
let mut creature = CreatureBuilder::new().seed(42).build();

for tick in 0..1000 {
    // 1. Получить сенсорный вход от окружения
    let sense = get_sense_from_world(&creature);
    
    // 2. Выполнить тик существа
    let action = creature.tick(&sense);
    
    // 3. Применить действие в мире
    apply_action_to_world(&creature, &action);
    
    // 4. Обработать события (еда, столкновения)
    if food_eaten {
        // Награда применяется автоматически внутри tick()
    }
}
```

### Сон и консолидация памяти

```rust
// Периодический сон для закрепления обучения
if tick % 1000 == 0 {
    creature.sleep(
        0.5,  // threshold: порог отсечения слабых весов
        0.2,  // boost: усиление сильных весов
        0.8,  // decay: затухание шума
    );
}
```

**Эффекты сна**:
- Закрепляет сильно выученные паттерны
- Отсекает слабый шум
- Предотвращает катастрофическое забывание

### Размножение

```rust
// Создание потомка с мутацией
let child = creature.spawn_child(
    new_seed,
    0.05,  // mutation_rate: вероятность мутации гена
);

// Потомок наследует:
// - Геном родителя (с мутациями)
// - Врождённые инстинкты
// - НЕ наследует выученные веса (эффект Вейсмана)
```

---

## Сенсоры и действия

### Структура Sense

```rust
pub struct Sense {
    /// Запах слева [0..1]
    pub smell_l: f32,
    /// Запах справа [0..1]
    pub smell_r: f32,
    /// Расстояние до стены слева [0..1]
    pub wall_l: f32,
    /// Расстояние до стены справа [0..1]
    pub wall_r: f32,
    /// Еда в радиусе рта
    pub food_near: bool,
    /// Удар о стену на этом тике
    pub wall: bool,
    /// Грубая клетка карты (для памяти места)
    pub place: u32,
    /// Курс в радианах (0 = +x, против часовой)
    pub heading: f32,
    /// Видел соседа за едой (социальное обучение)
    pub see_eat: bool,
}
```

**Рекомендации**:
- Нормализуйте значения в [0..1]
- `wall = true` вызывает боль и стресс
- `food_near = true` + `action.eat = true` = награда
- `see_eat = true` усиливает обучение (зеркальные нейроны)

### Структура Action

```rust
pub struct Action {
    /// Поворот: -1 = влево, +1 = вправо
    pub steer: f32,
    /// Гейт скорости: 0..1
    pub drive: f32,
    /// Открыт ли рот
    pub eat: bool,
}
```

**Интерпретация**:
```rust
let action = creature.tick(&sense);

// Применение в физике
let turn_angle = action.steer * MAX_TURN_RATE * dt;
let velocity = action.drive * creature.max_speed();

// Попытка съесть
if action.eat && food_in_range {
    consume_food();
}
```

---

## Метрики и мониторинг

### Базовые метрики

```rust
use dioma_engine::creature::CreatureMetrics;

let metrics = creature.metrics();

println!("Glucose: {:.1}", metrics.glucose_level);
println!("Hunger: {:.1}", metrics.hunger_level);
println!("Pain: {:.1}", metrics.pain_level);
println!("Brain damage: {:.2}%", metrics.brain_damage * 100.0);
println!("Learning rate: {:.4}", metrics.effective_learning_rate);
```

### Детальное состояние

```rust
// Полное состояние крови
let blood = creature.blood_state();
println!("Tick: {}", blood.tick);
println!("Chemicals: {:?}", blood.concentrations);

// Статистика мозга
let brain = creature.brain_stats();
println!("Neurons: {}", brain.neuron_count);
println!("Regions: {}", brain.region_count);
println!("W_max: {:.1}", brain.plasticity_w_max);
```

### Мониторинг в реальном времени

```rust
struct CreatureMonitor {
    glucose_history: Vec<f32>,
    damage_history: Vec<f32>,
}

impl CreatureMonitor {
    fn update(&mut self, creature: &Creature) {
        let m = creature.metrics();
        self.glucose_history.push(m.glucose_level);
        self.damage_history.push(m.brain_damage);
        
        // Предупреждения
        if m.glucose_level < 50.0 {
            warn!("Low glucose: {:.1}", m.glucose_level);
        }
        if m.brain_damage > 0.5 {
            error!("Critical brain damage: {:.2}%", m.brain_damage * 100.0);
        }
    }
}
```

---

## Геном и наследование

### Структура генома

```rust
pub struct CreatureGenome {
    /// Гены роста коннектома
    pub steering: GrowthGenes,
    /// Гены метаболизма
    pub metabolism: MetabolismGene,
    /// Пол (влияет на размножение)
    pub sex: Sex,
}
```

### Модификация генома

```rust
let mut genome = CreatureGenome::default();

// Изменение роста мозга
genome.steering.diffusion_rate = 0.7;  // Скорость диффузии морфогенов
genome.steering.prune_threshold = 0.1; // Порог обрезки слабых связей

// Изменение метаболизма
genome.metabolism.glucose_cost = 0.2;  // Стоимость жизни
genome.metabolism.decay_rates[0] = 0.99; // Скорость распада глюкозы

let creature = CreatureBuilder::new()
    .genome(genome)
    .seed(42)
    .build();
```

### Эволюция популяции

```rust
struct Population {
    creatures: Vec<Creature>,
}

impl Population {
    fn evolve_generation(&mut self) {
        // 1. Оценка приспособленности
        let fitness: Vec<f32> = self.creatures
            .iter()
            .map(|c| c.food_eaten as f32)
            .collect();
        
        // 2. Селекция (турнирная)
        let parents = tournament_selection(&self.creatures, &fitness, 0.7);
        
        // 3. Размножение с мутацией
        let mut next_gen = Vec::new();
        for (i, parent) in parents.iter().enumerate() {
            let child = parent.spawn_child(
                new_seed(i),
                0.05, // 5% мутация
            );
            next_gen.push(child);
        }
        
        self.creatures = next_gen;
    }
}
```

---

## GPU ускорение

### Когда использовать GPU

**Используйте GPU если**:
- ✅ Симуляция 100+ существ одновременно
- ✅ Большие нейронные сети (500+ нейронов)
- ✅ Доступен совместимый GPU

**Используйте CPU если**:
- ✅ Малое количество существ (<50)
- ✅ Отладка и профилирование
- ✅ Детерминированность критична

### Производительность

```rust
// Бенчмарк: 1000 существ, 100 тиков

// CPU (однопоточно)
let start = Instant::now();
for _ in 0..100 {
    for creature in &mut creatures {
        creature.tick(&sense);
    }
}
println!("CPU: {:?}", start.elapsed()); // ~5.2s

// GPU
let start = Instant::now();
for _ in 0..100 {
    gpu_batch_tick(&mut creatures, &senses);
}
println!("GPU: {:?}", start.elapsed()); // ~0.8s
```

**Прирост**: ~6.5x для больших популяций

### Fallback на CPU

```rust
// Автоматический fallback если GPU недоступен
let creature = CreatureBuilder::new()
    .gpu(true)  // Попытка использовать GPU
    .build();   // Тихо падает на CPU если нет GPU

// Проверка используемого бэкенда
if creature.is_gpu() {
    println!("Using GPU acceleration");
} else {
    println!("Using CPU (GPU unavailable)");
}
```

---

## Продвинутые техники

### Кастомные инстинкты

```rust
use dioma_engine::engine::instincts::Instinct;

// Создание врождённого рефлекса
let avoid_pain = Instinct::new(
    "avoid_pain",
    vec![/* стимулируемые нейроны */],
    10.0, // сила прожигания
);

// Применение при рождении
creature.burn_instinct(&avoid_pain);
```

### Социальное обучение

```rust
// Наблюдение за успешным соседом
if neighbor_eating && creature_sees_neighbor {
    let sense = Sense {
        see_eat: true,  // Активирует зеркальные нейроны
        ..sense
    };
    
    let action = creature.tick(&sense);
    // Обучение усиливается через see_eat сигнал
}
```

### Адаптивная пластичность

```rust
// Динамическое изменение learning rate
let base_lr = 0.01;
let stress_factor = creature.metrics().cortisol_level / 255.0;
let adaptive_lr = base_lr * (1.0 - stress_factor * 0.5);

// Пересоздание с новым lr (для следующего поколения)
let child = CreatureBuilder::new()
    .genome(parent.genome().clone())
    .plasticity_lr(adaptive_lr)
    .seed(new_seed)
    .build();
```

### Сохранение и загрузка

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct CreatureSnapshot {
    seed: u64,
    genome: CreatureGenome,
    food_eaten: u32,
    tick: usize,
}

// Сохранение
let snapshot = CreatureSnapshot {
    seed: creature.seed,
    genome: creature.genome().clone(),
    food_eaten: creature.food_eaten,
    tick: creature.blood.tick,
};
let json = serde_json::to_string(&snapshot)?;
std::fs::write("creature.json", json)?;

// Загрузка
let json = std::fs::read_to_string("creature.json")?;
let snapshot: CreatureSnapshot = serde_json::from_json(&json)?;
let creature = CreatureBuilder::new()
    .seed(snapshot.seed)
    .genome(snapshot.genome)
    .build();
```

---

## Лучшие практики

### Производительность

1. **Используйте release сборку**
   ```bash
   cargo build --release
   ```

2. **Batch обработка**
   ```rust
   // Плохо: по одному
   for creature in &mut creatures {
       creature.tick(&sense);
   }
   
   // Хорошо: batch на GPU
   gpu_batch_tick(&mut creatures, &senses);
   ```

3. **Переиспользуйте объекты**
   ```rust
   let mut sense = Sense::default();
   for tick in 0..1000 {
       update_sense(&mut sense, world);
       creature.tick(&sense);
   }
   ```

### Отладка

1. **Логирование метрик**
   ```rust
   if tick % 100 == 0 {
       let m = creature.metrics();
       info!("Tick {}: glucose={:.1}, damage={:.2}%", 
             tick, m.glucose_level, m.brain_damage * 100.0);
   }
   ```

2. **Детерминированность**
   ```rust
   // Фиксированный seed для воспроизводимости
   let creature = Creature::new(42);
   ```

3. **Визуализация**
   ```rust
   // Экспорт для внешней визуализации
   let brain_state = creature.brain.export_state();
   visualizer.render(brain_state);
   ```

---

## Примеры использования

### Простая симуляция

```rust
fn simple_simulation() {
    let mut creature = Creature::new(42);
    let mut world = World::new();
    
    for tick in 0..10000 {
        let sense = world.sense_for(&creature);
        let action = creature.tick(&sense);
        world.apply_action(&creature, &action);
        
        if tick % 1000 == 0 {
            println!("Tick {}: food eaten = {}", 
                     tick, creature.food_eaten);
        }
    }
}
```

### Эволюционный эксперимент

```rust
fn evolution_experiment() {
    let mut population = Population::new(100);
    
    for generation in 0..50 {
        // Симуляция жизни
        for _ in 0..1000 {
            population.tick();
        }
        
        // Эволюция
        population.evolve_generation();
        
        let avg_fitness = population.average_fitness();
        println!("Gen {}: avg fitness = {:.2}", 
                 generation, avg_fitness);
    }
}
```

---

## Заключение

Dioma Engine предоставляет:
- ✅ **Простой API** - легко начать
- ✅ **Гибкость** - Builder Pattern для любых конфигураций
- ✅ **Производительность** - 33-50% оптимизации + GPU
- ✅ **Observability** - Metrics API для мониторинга
- ✅ **Эволюция** - Геном и размножение из коробки

**Следующие шаги**:
1. Изучите [примеры](examples/)
2. Прочитайте [PERFORMANCE.md](PERFORMANCE.md)
3. Экспериментируйте с параметрами!
