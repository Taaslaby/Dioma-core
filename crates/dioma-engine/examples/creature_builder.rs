//! Пример использования CreatureBuilder для создания существ с различными конфигурациями.

use dioma_engine::creature::{CreatureBuilder, Sense};

fn main() {
    println!("=== Примеры использования CreatureBuilder ===\n");

    // 1. Простое существо с дефолтными настройками
    println!("1. Создание простого существа:");
    let creature1 = CreatureBuilder::new()
        .seed(42)
        .build();
    println!("   Seed: {}", creature1.seed);
    println!("   ✓ Создано\n");

    // 2. Существо с кастомными параметрами пластичности
    println!("2. Существо с низким learning rate (медленное обучение):");
    let creature2 = CreatureBuilder::new()
        .seed(123)
        .plasticity_lr(0.001)
        .plasticity_w_max(32.0)
        .build();
    println!("   Seed: {}", creature2.seed);
    println!("   ✓ Создано\n");

    // 3. Существо на GPU (если доступно)
    println!("3. Существо с GPU-динамикой:");
    let creature3 = CreatureBuilder::new()
        .seed(456)
        .gpu(true)
        .build();
    println!("   Seed: {}", creature3.seed);
    println!("   ✓ Создано\n");

    // 4. Полностью GPU существо (рост + динамика)
    println!("4. Существо с GPU-ростом и GPU-динамикой:");
    let creature4 = CreatureBuilder::new()
        .seed(789)
        .gpu_grown(true)
        .build();
    println!("   Seed: {}", creature4.seed);
    println!("   ✓ Создано\n");

    // 5. Тестовый тик
    println!("5. Тестовый тик существа:");
    let mut creature = CreatureBuilder::new()
        .seed(999)
        .plasticity_lr(0.01)
        .build();
    
    let sense = Sense {
        smell_l: 0.5,
        smell_r: 0.3,
        vis_l: 0.0,
        vis_r: 0.0,
        wall_l: 0.0,
        wall_r: 0.1,
        food_near: false,
        wall: false,
        place: 0,
        heading: 0.0,
        see_eat: false,
    };
    
    let action = creature.tick(&sense);
    println!("   Steer: {:.3}", action.steer);
    println!("   Drive: {:.3}", action.drive);
    println!("   Eat: {}", action.eat);
    println!("   ✓ Тик выполнен\n");

    println!("=== Все примеры успешно выполнены ===");
}
