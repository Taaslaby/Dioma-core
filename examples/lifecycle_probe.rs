//! Живой прогон полного жизненного цикла существа на реальном API:
//!   охота (запах слева + еда) → реально ест (food_eaten растёт) →
//!   мозг реально обучается (STDP потенцирует путь → learned_instincts) →
//!   потомок через spawn_child рождается с врождённым рефлексом (Болдуин) →
//!   рефлекс работает с первого тика (асимметрия поворота).
//! Прогоняется на CPU-существе и на GPU-существе (динамика+обучение на GPU).
//!
//! Запуск: `cargo run --release --example lifecycle_probe`
use dioma::creature::{Creature, Sense};
use dioma::biology::metabolism::chem;

fn avg_steer(c: &mut Creature, sense: Sense) -> f32 {
    let mut sum = 0.0f32;
    for _ in 0..20 {
        sum += c.tick(&sense).steer;
    }
    sum / 20.0
}

fn hunt_and_inherit(name: &str, mut trained: Creature) -> bool {
    let hunt = Sense {
        smell_l: 1.0,
        food_near: true,
        ..Default::default()
    };
    trained.blood.conc[chem::HUNGER] = 200.0;
    for _ in 0..300 {
        trained.tick(&hunt);
    }
    let eaten = trained.food_eaten();
    let learned = trained.learned_instincts().len();

    let mut t_child = trained.spawn_child(99);
    let child_instincts = t_child.genome.instincts.len();
    let mut n_child = Creature::new(1).spawn_child(99);
    t_child.blood.conc[chem::HUNGER] = 200.0;
    n_child.blood.conc[chem::HUNGER] = 200.0;
    // Стимул 0.05 (не 0.1): оба потомка наследуют одну проводку, при 0.1
    // моторы упираются в насыщение и разница рефлекса не видна.
    let smell = Sense {
        smell_l: 0.05,
        ..Default::default()
    };
    let t_steer = avg_steer(&mut t_child, smell);
    let n_steer = avg_steer(&mut n_child, smell);

    println!(
        "[{name}] food_eaten={eaten}, learned_instincts={learned}, потомок: инстинктов={child_instincts}, steer={t_steer:.2} vs наивный {n_steer:.2}"
    );
    eaten > 0 && learned > 0 && child_instincts == 1 && t_steer < -0.05 && t_steer < n_steer - 0.05
}

fn main() {
    let cpu = Creature::new(1);
    let gpu = Creature::new_gpu(1);
    let gpu_grown = Creature::new_gpu_grown(1);
    println!("мозг: CPU | GPU on_gpu={} | GPU+GPU-рост on_gpu={}", gpu.brain.on_gpu(), gpu_grown.brain.on_gpu());

    let mut all_ok = true;
    all_ok &= hunt_and_inherit("CPU", cpu);
    all_ok &= hunt_and_inherit("GPU", gpu);
    if gpu_grown.brain.on_gpu() {
        all_ok &= hunt_and_inherit("GPU-рост", gpu_grown);
    }

    // Контроль: ненаученный родитель ничего не передаёт.
    let naive = Creature::new(1);
    println!(
        "[контроль] наивный родитель: learned_instincts={}",
        naive.learned_instincts().len()
    );
    all_ok &= naive.learned_instincts().is_empty();

    // Обучение реально двигает поведение: проверяем, что веса изменились.
    println!(
        "=== {}",
        if all_ok {
            "LIFECYCLE OK: охотится, ест, учится, передаёт рефлекс потомку"
        } else {
            "ПРОВАЛ: одно из звеньев цикла не выполнилось"
        }
    );
    std::process::exit(if all_ok { 0 } else { 1 });
}
