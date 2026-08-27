//! Пример использования Metrics API для мониторинга состояния существ.

use dioma_engine::creature::{CreatureBuilder, Sense};

fn main() {
    println!("=== Пример использования Metrics API ===\n");

    let mut creature = CreatureBuilder::new()
        .seed(42)
        .plasticity_lr(0.01)
        .build();

    println!("Начальное состояние:");
    print_metrics(&creature);

    // Симуляция нескольких тиков
    println!("\n--- Симуляция 10 тиков ---");
    for i in 1..=10 {
        let sense = Sense {
            smell_l: 0.5 + (i as f32 * 0.05),
            smell_r: 0.3,
            vis_l: 0.0,
            vis_r: 0.0,
            wall_l: 0.0,
            wall_r: 0.1,
            food_near: i == 5, // Еда на 5-м тике
            wall: i == 8,      // Удар о стену на 8-м тике
            place: 0,
            heading: 0.0,
            see_eat: false,
        };

        let action = creature.tick(&sense);
        
        if i == 5 || i == 8 || i == 10 {
            println!("\nТик {}:", i);
            println!("  Action: steer={:.2}, drive={:.2}, eat={}", 
                     action.steer, action.drive, action.eat);
            print_metrics(&creature);
        }
    }

    // Детальная информация
    println!("\n--- Детальная информация ---");
    let blood_state = creature.blood_state();
    println!("Blood tick: {}", blood_state.tick);
    println!("Chemicals count: {}", blood_state.concentrations.len());
    
    let brain_stats = creature.brain_stats();
    println!("\nBrain stats:");
    println!("  Neurons: {}", brain_stats.neuron_count);
    println!("  Regions: {}", brain_stats.region_count);
    println!("  W_max: {:.1}", brain_stats.plasticity_w_max);

    println!("\n=== Пример завершён ===");
}

fn print_metrics(creature: &dioma_engine::creature::Creature) {
    let m = creature.metrics();
    println!("  Glucose: {:.1}", m.glucose_level);
    println!("  Hunger: {:.1}", m.hunger_level);
    println!("  Pain: {:.1}", m.pain_level);
    println!("  Reward: {:.1}", m.reward_level);
    println!("  Fatigue: {:.1}", m.fatigue_level);
    println!("  Serotonin: {:.1}", m.serotonin_level);
    println!("  Cortisol: {:.1}", m.cortisol_level);
    println!("  Brain damage: {:.1}%", m.brain_damage * 100.0);
    println!("  Plasticity budget: {:.2}", m.plasticity_budget);
    println!("  Food eaten: {}", m.food_eaten);
}
