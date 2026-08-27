# 🤝 Contributing to Dioma Engine

Руководство для разработчиков, желающих внести вклад в проект.

## Содержание

1. [Начало работы](#начало-работы)
2. [Стандарты кода](#стандарты-кода)
3. [Процесс разработки](#процесс-разработки)
4. [Тестирование](#тестирование)
5. [Оптимизация](#оптимизация)
6. [Документация](#документация)
7. [Pull Request процесс](#pull-request-процесс)

---

## Начало работы

### Требования

- **Rust**: 1.70+ (stable)
- **Cargo**: последняя версия
- **Git**: для контроля версий
- **GPU** (опционально): для GPU-ускорения

### Клонирование репозитория

```bash
git clone https://github.com/your-org/dioma-engine.git
cd dioma-engine
```

### Сборка

```bash
# Debug сборка
cargo build

# Release сборка
cargo build --release

# Запуск тестов
cargo test

# Запуск бенчмарков
cargo bench
```

### Структура проекта

```
dioma-engine/
├── src/
│   ├── body/           # Метаболизм
│   ├── creature/       # Существа
│   ├── engine/         # Нейронная система
│   ├── env/            # Окружение
│   ├── script/         # Скриптовая VM
│   ├── sim/            # Симуляция
│   └── world/          # Игровой мир
├── examples/           # Примеры использования
├── benches/            # Бенчмарки
├── tests/              # Интеграционные тесты
└── docs/               # Документация
```

---

## Стандарты кода

### Форматирование

Используем стандартный `rustfmt`:

```bash
cargo fmt
```

**Конфигурация** (`.rustfmt.toml`):
```toml
max_width = 100
tab_spaces = 4
edition = "2021"
```

### Линтинг

Используем `clippy` с строгими правилами:

```bash
cargo clippy -- -D warnings
```

**Обязательные проверки**:
- Нет `unwrap()` в production коде
- Нет `panic!()` без веской причины
- Все публичные API документированы
- Нет неиспользуемого кода

### Именование

**Модули**: `snake_case`
```rust
mod neural_network;
mod creature_builder;
```

**Типы**: `PascalCase`
```rust
struct CreatureGenome;
enum NeuronType;
```

**Функции/переменные**: `snake_case`
```rust
fn update_sensors();
let glucose_level = 128.0;
```

**Константы**: `SCREAMING_SNAKE_CASE`
```rust
const MAX_NEURONS: usize = 1000;
const PLASTICITY_W_MAX: f32 = 64.0;
```

### Комментарии

**Документация API**:
```rust
/// Создаёт новое существо из сида.
///
/// # Аргументы
///
/// * `seed` - Детерминированный сид для RNG
///
/// # Примеры
///
/// ```
/// let creature = Creature::new(42);
/// ```
pub fn new(seed: u64) -> Self {
    // ...
}
```

**Внутренние комментарии**:
```rust
// ОПТИМИЗАЦИЯ: Используем SIMD для параллельной обработки
for chunk in 0..(N / 8) {
    // ...
}

// TODO: Добавить поддержку адаптивного learning rate
// FIXME: Исправить race condition в GPU коде
```

---

## Процесс разработки

### Workflow

1. **Создать ветку**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Разработка**
   - Пишем код
   - Добавляем тесты
   - Обновляем документацию

3. **Коммит**
   ```bash
   git add .
   git commit -m "feat: add incremental sensor updates"
   ```

4. **Push и PR**
   ```bash
   git push origin feature/my-feature
   ```

### Conventional Commits

Используем стандарт [Conventional Commits](https://www.conventionalcommits.org/):

**Типы**:
- `feat:` - новая функциональность
- `fix:` - исправление бага
- `perf:` - оптимизация производительности
- `refactor:` - рефакторинг без изменения функциональности
- `docs:` - изменения в документации
- `test:` - добавление/изменение тестов
- `chore:` - рутинные задачи (обновление зависимостей и т.д.)

**Примеры**:
```bash
feat: add Builder Pattern for Creature
fix: correct SIMD alignment in metabolism
perf: optimize HAAP cache allocation
docs: update API guide with metrics examples
test: add determinism tests for Brain
```

### Ветвление

**Основные ветки**:
- `main` - стабильная версия
- `develop` - разработка

**Feature ветки**:
- `feature/builder-pattern`
- `feature/metrics-api`
- `feature/gpu-acceleration`

**Bugfix ветки**:
- `fix/memory-leak`
- `fix/race-condition`

---

## Тестирование

### Unit тесты

Каждый модуль должен иметь тесты:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creature_creation() {
        let creature = Creature::new(42);
        assert_eq!(creature.seed, 42);
    }

    #[test]
    fn test_determinism() {
        let mut c1 = Creature::new(42);
        let mut c2 = Creature::new(42);
        
        let sense = Sense::default();
        for _ in 0..100 {
            let a1 = c1.tick(&sense);
            let a2 = c2.tick(&sense);
            assert_eq!(a1.steer, a2.steer);
        }
    }
}
```

### Integration тесты

В директории `tests/`:

```rust
// tests/creature_learning.rs
use dioma_engine::creature::*;

#[test]
fn creature_learns_to_avoid_walls() {
    let mut creature = Creature::new(42);
    
    // Обучение через 1000 тиков
    for _ in 0..1000 {
        let sense = Sense {
            wall_l: 0.8,
            ..Default::default()
        };
        creature.tick(&sense);
    }
    
    // Проверка обучения
    let action = creature.tick(&sense);
    assert!(action.steer > 0.0); // Поворот вправо
}
```

### Бенчмарки

В директории `benches/`:

```rust
// benches/creature_tick.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dioma_engine::creature::*;

fn bench_creature_tick(c: &mut Criterion) {
    let mut creature = Creature::new(42);
    let sense = Sense::default();
    
    c.bench_function("creature_tick", |b| {
        b.iter(|| {
            creature.tick(black_box(&sense))
        })
    });
}

criterion_group!(benches, bench_creature_tick);
criterion_main!(benches);
```

### Запуск тестов

```bash
# Все тесты
cargo test

# Конкретный модуль
cargo test creature::metrics

# С выводом
cargo test -- --nocapture

# Бенчмарки
cargo bench

# Coverage
cargo tarpaulin --out Html
```

---

## Оптимизация

### Профилирование

**CPU профилирование**:
```bash
# Flamegraph
cargo flamegraph --bench bench_all

# Perf (Linux)
perf record -g ./target/release/dioma-engine
perf report
```

**Memory профилирование**:
```bash
# Valgrind
cargo valgrind --bench bench_all

# Heaptrack
heaptrack ./target/release/dioma-engine
```

### Чеклист оптимизаций

Перед оптимизацией:
- [ ] Профилировать и найти узкое место
- [ ] Измерить baseline производительность
- [ ] Определить целевой прирост

После оптимизации:
- [ ] Измерить новую производительность
- [ ] Проверить корректность (тесты)
- [ ] Документировать изменения
- [ ] Обновить бенчмарки

### Примеры оптимизаций

**1. Избегать аллокаций**:
```rust
// Плохо
fn process(&self) -> Vec<f32> {
    let mut result = Vec::new();
    for i in 0..self.n {
        result.push(self.data[i] * 2.0);
    }
    result
}

// Хорошо
fn process(&self, output: &mut [f32]) {
    for i in 0..self.n {
        output[i] = self.data[i] * 2.0;
    }
}
```

**2. Использовать SIMD**:
```rust
use wide::f32x8;

// Обработка блоками по 8
for chunk in 0..(N / 8) {
    let base = chunk * 8;
    let vec = f32x8::new([/* 8 элементов */]);
    let result = vec * 2.0;
    // Запись результата
}
```

**3. Кэширование**:
```rust
struct Cache {
    prev_value: f32,
    dirty: bool,
}

fn update(&mut self, new_value: f32) {
    if (new_value - self.prev_value).abs() > EPSILON {
        self.dirty = true;
        self.prev_value = new_value;
    }
}
```

---

## Документация

### Типы документации

1. **API документация** - в коде (`///`)
2. **Руководства** - Markdown файлы
3. **Примеры** - в `examples/`
4. **Архитектура** - `ARCHITECTURE.md`

### Генерация документации

```bash
# Локальная документация
cargo doc --open

# С приватными элементами
cargo doc --document-private-items --open
```

### Стандарты документации

**Публичные API**:
```rust
/// Краткое описание (одна строка).
///
/// Детальное описание функциональности.
/// Может быть многострочным.
///
/// # Аргументы
///
/// * `param1` - Описание первого параметра
/// * `param2` - Описание второго параметра
///
/// # Возвращаемое значение
///
/// Описание возвращаемого значения.
///
/// # Примеры
///
/// ```
/// let result = function(42, "test");
/// assert_eq!(result, expected);
/// ```
///
/// # Паники
///
/// Функция паникует если `param1 < 0`.
///
/// # Ошибки
///
/// Возвращает `Err` если файл не найден.
pub fn function(param1: i32, param2: &str) -> Result<T, E> {
    // ...
}
```

**Модули**:
```rust
//! Модуль для работы с метаболизмом существ.
//!
//! Этот модуль предоставляет структуры и функции для симуляции
//! биохимических процессов в теле существа.
//!
//! # Примеры
//!
//! ```
//! use dioma_engine::body::Blood;
//!
//! let blood = Blood::new();
//! blood.tick();
//! ```
```

---

## Pull Request процесс

### Чеклист перед PR

- [ ] Код отформатирован (`cargo fmt`)
- [ ] Нет предупреждений clippy (`cargo clippy`)
- [ ] Все тесты проходят (`cargo test`)
- [ ] Добавлены новые тесты для новой функциональности
- [ ] Документация обновлена
- [ ] Бенчмарки обновлены (если применимо)
- [ ] CHANGELOG.md обновлён

### Шаблон PR

```markdown
## Описание

Краткое описание изменений.

## Мотивация

Почему эти изменения необходимы?

## Изменения

- Добавлено: ...
- Изменено: ...
- Удалено: ...

## Тестирование

Как тестировались изменения?

## Производительность

Влияние на производительность (если применимо):
- До: X ns
- После: Y ns
- Прирост: Z%

## Чеклист

- [ ] Тесты добавлены/обновлены
- [ ] Документация обновлена
- [ ] Бенчмарки обновлены
- [ ] CHANGELOG.md обновлён
```

### Review процесс

1. **Автоматические проверки**
   - CI/CD pipeline
   - Тесты
   - Линтинг
   - Coverage

2. **Code review**
   - Минимум 1 одобрение
   - Проверка стиля кода
   - Проверка тестов
   - Проверка документации

3. **Merge**
   - Squash and merge для feature веток
   - Rebase для bugfix веток

---

## Области для вклада

### Высокий приоритет

1. **Производительность**
   - Новые SIMD оптимизации
   - GPU ускорение
   - Параллелизм

2. **Функциональность**
   - Новые типы нейронов
   - Новые химические вещества
   - Новые инстинкты

3. **Документация**
   - Примеры использования
   - Туториалы
   - Переводы

### Средний приоритет

1. **Тестирование**
   - Увеличение coverage
   - Property-based тесты
   - Fuzzing

2. **Инструменты**
   - Визуализация
   - Профилирование
   - Отладка

### Низкий приоритет

1. **Рефакторинг**
   - Улучшение структуры кода
   - Удаление технического долга

2. **Экспериментальные фичи**
   - Новые алгоритмы обучения
   - Альтернативные архитектуры

---

## Коммуникация

### Каналы

- **GitHub Issues** - баги и feature requests
- **GitHub Discussions** - общие вопросы
- **Discord** - real-time общение (если есть)

### Создание Issue

**Bug report**:
```markdown
## Описание бага

Краткое описание проблемы.

## Шаги для воспроизведения

1. Создать существо с seed=42
2. Вызвать tick() 100 раз
3. Наблюдать неожиданное поведение

## Ожидаемое поведение

Что должно было произойти.

## Фактическое поведение

Что произошло на самом деле.

## Окружение

- OS: Windows 11
- Rust: 1.70.0
- Dioma Engine: 0.2.0
```

**Feature request**:
```markdown
## Описание фичи

Что вы хотите добавить?

## Мотивация

Почему это полезно?

## Предлагаемое решение

Как это можно реализовать?

## Альтернативы

Какие есть альтернативные подходы?
```

---

## Лицензия

Внося вклад в проект, вы соглашаетесь с тем, что ваш код будет лицензирован под той же лицензией, что и проект.

---

## Благодарности

Спасибо всем контрибьюторам за вклад в развитие Dioma Engine! 🎉

**Список контрибьюторов**: см. [CONTRIBUTORS.md](CONTRIBUTORS.md)

---

## Вопросы?

Если у вас есть вопросы, не стесняйтесь:
- Открыть Issue
- Спросить в Discussions
- Связаться с мейнтейнерами

**Удачи в разработке!** 🚀
