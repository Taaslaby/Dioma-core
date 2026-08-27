# 🧬 Dioma Engine

**Высокопроизводительный движок для симуляции искусственных существ с обучаемыми спайковыми нейронными сетями, метаболизмом и эволюцией.**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/performance-33--50%25%20faster-brightgreen.svg)](PERFORMANCE.md)

---

## ✨ Особенности

### 🚀 Производительность
- **33-50% прирост** благодаря 4 критическим оптимизациям
- **SIMD ускорение** для метаболизма (15-20% прирост)
- **GPU поддержка** для массовых симуляций
- **Инкрементальные обновления** сенсоров (10-15% прирост)

### 🧠 Нейронная система
- Спайковые нейронные сети (LIF, HAAP модели)
- Дофаминовая пластичность (STDP)
- Врождённые инстинкты
- Консолидация памяти через сон

### 🩸 Метаболизм
- 9 химических веществ (глюкоза, дофамин, серотонин, кортизол и др.)
- Органы и повреждения
- Энергетический баланс
- Стресс и усталость

### 🧬 Эволюция
- Геномное наследование
- Мутации и селекция
- Эффект Болдуина
- Половое размножение

### 🎯 Удобный API
- **Builder Pattern** для гибкого создания существ
- **Metrics API** для мониторинга и отладки
- Детерминированность (фиксированный seed)
- Обратная совместимость

---

## 🚀 Быстрый старт

### Установка

```toml
[dependencies]
dioma-engine = "0.2.0"
```

### Минимальный пример

```rust
use dioma_engine::creature::{Creature, Sense};

fn main() {
    // Создание существа
    let mut creature = Creature::new(42);
    
    // Сенсорный вход
    let sense = Sense {
        smell_l: 0.5,
        smell_r: 0.3,
        wall_l: 0.0,
        wall_r: 0.1,
        food_near: false,
        wall: false,
        place: 0,
        heading: 0.0,
        see_eat: false,
    };
    
    // Выполнение тика
    let action = creature.tick(&sense);
    
    println!("Steer: {:.2}", action.steer);
    println!("Drive: {:.2}", action.drive);
    println!("Eat: {}", action.eat);
}
```

### Builder Pattern

```rust
use dioma_engine::creature::CreatureBuilder;

// Простое существо
let creature = CreatureBuilder::new()
    .seed(42)
    .build();

// С кастомными параметрами
let creature = CreatureBuilder::new()
    .seed(123)
    .plasticity_lr(0.01)
    .plasticity_w_max(32.0)
    .build();

// С GPU ускорением
let creature = CreatureBuilder::new()
    .seed(456)
    .gpu(true)
    .build();
```

### Metrics API

```rust
use dioma_engine::creature::CreatureMetrics;

let metrics = creature.metrics();

println!("Glucose: {:.1}", metrics.glucose_level);
println!("Hunger: {:.1}", metrics.hunger_level);
println!("Pain: {:.1}", metrics.pain_level);
println!("Brain damage: {:.2}%", metrics.brain_damage * 100.0);
println!("Learning rate: {:.4}", metrics.effective_learning_rate);
```

---

## 📊 Оптимизации производительности

| Оптимизация | Прирост | Область |
|-------------|---------|---------|
| HAAP кэш | 5-10% | Нейронная динамика |
| SIMD распады | 15-20% | Метаболизм |
| Инкрементальные сенсоры | 10-15% | Сенсорная обработка |
| Batch receptors | 3-5% | Рецепторы |

**Суммарный прирост**: 33-50% 🚀

Подробнее: [PERFORMANCE.md](PERFORMANCE.md)

---

## 📚 Документация

### Руководства
- **[API Guide](API_GUIDE.md)** - Полное руководство по API
- **[Performance](PERFORMANCE.md)** - Оптимизации и бенчмарки
- **[Architecture](ARCHITECTURE.md)** - Архитектура и дизайн
- **[Contributing](CONTRIBUTING.md)** - Руководство для контрибьюторов

### Примеры
- `examples/creature_builder.rs` - Использование Builder Pattern
- `examples/creature_metrics.rs` - Мониторинг метрик
- `examples/brain_viz.rs` - Визуализация мозга

### API документация
```bash
cargo doc --open
```

---

## 🎯 Примеры использования

### Эволюционный эксперимент

```rust
use dioma_engine::creature::CreatureBuilder;

fn evolution_experiment() {
    let mut population: Vec<_> = (0..100)
        .map(|i| CreatureBuilder::new().seed(i).build())
        .collect();
    
    for generation in 0..50 {
        // Симуляция жизни
        for _ in 0..1000 {
            for creature in &mut population {
                let sense = get_sense(&creature);
                creature.tick(&sense);
            }
        }
        
        // Селекция и размножение
        population = evolve_generation(population);
        
        let avg_fitness: f32 = population.iter()
            .map(|c| c.food_eaten as f32)
            .sum::<f32>() / population.len() as f32;
        
        println!("Gen {}: avg fitness = {:.2}", generation, avg_fitness);
    }
}
```

### Мониторинг в реальном времени

```rust
use dioma_engine::creature::{Creature, CreatureMetrics};

fn monitor_creature(creature: &Creature) {
    let m = creature.metrics();
    
    // Предупреждения
    if m.glucose_level < 50.0 {
        warn!("Low glucose: {:.1}", m.glucose_level);
    }
    
    if m.brain_damage > 0.5 {
        error!("Critical brain damage: {:.2}%", m.brain_damage * 100.0);
    }
    
    // Логирование
    info!("Tick {}: glucose={:.1}, hunger={:.1}, pain={:.1}",
          m.tick, m.glucose_level, m.hunger_level, m.pain_level);
}
```

---

## 🔬 Бенчмарки

### Creature::tick()

```
Baseline (до оптимизаций):  ~2.5 µs
Optimized (после):          ~1.7 µs
Прирост:                    ~47%
```

### Метаболизм (Blood::tick())

```
Scalar decay:   ~45 ns
SIMD decay:     ~28 ns
Прирост:        ~37.8%
```

### Сенсоры

```
Full update:        ~120 ns
Incremental update: ~85 ns
Прирост:            ~29.2%
```

Запуск бенчмарков:
```bash
cargo bench
```

---

## 🛠️ Разработка

### Требования

- Rust 1.70+
- Cargo
- GPU (опционально, для GPU ускорения)

### Сборка

```bash
# Debug
cargo build

# Release
cargo build --release

# Тесты
cargo test

# Бенчмарки
cargo bench
```

### Тестирование

```bash
# Все тесты
cargo test

# Конкретный модуль
cargo test creature::metrics

# С выводом
cargo test -- --nocapture

# Coverage
cargo tarpaulin --out Html
```

---

## 🤝 Вклад в проект

Мы приветствуем вклад в проект! См. [CONTRIBUTING.md](CONTRIBUTING.md) для деталей.

### Области для вклада

- 🚀 **Производительность** - новые оптимизации
- 🧠 **Функциональность** - новые типы нейронов, химические вещества
- 📚 **Документация** - примеры, туториалы, переводы
- 🧪 **Тестирование** - увеличение coverage, property-based тесты
- 🎨 **Визуализация** - инструменты для отладки и мониторинга

---

## 📈 Roadmap

### v0.3.0 (Ближайшее)
- [ ] Async API для неблокирующих операций
- [ ] Сериализация состояния (сохранение/загрузка)
- [ ] Error Handling (Result вместо паник)
- [ ] Конфигурация через TOML файлы

### v0.4.0 (Среднесрочное)
- [ ] Batch обработка для множества существ
- [ ] Улучшенная GPU поддержка
- [ ] Визуализация в реальном времени
- [ ] Профилирование и отладочные инструменты

### v1.0.0 (Долгосрочное)
- [ ] Распределённая симуляция (кластеры)
- [ ] Экосистема (хищники, симбиоз)
- [ ] Интерактивное управление
- [ ] Полная документация и туториалы

---

## 📜 Лицензия

MIT License - см. [LICENSE](LICENSE) для деталей.

---

## 🙏 Благодарности

- **Pillars** - спайковая нейронная динамика
- **wide** - SIMD оптимизации
- **glam** - математические операции
- Всем контрибьюторам проекта!

---

## 📞 Контакты

- **Issues**: [GitHub Issues](https://github.com/your-org/dioma-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/dioma-engine/discussions)
- **Email**: your-email@example.com

---

## 🌟 Звёзды

Если проект вам понравился, поставьте ⭐ на GitHub!

---

**Создано с ❤️ для исследований в области искусственной жизни и эволюционных алгоритмов.**