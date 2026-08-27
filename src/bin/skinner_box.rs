//! Ящик Скиннера — мозг учится свет→действие через R-STDP.
use dioma_engine::engine::neurons::{Brain, PlasticityParams};
use dioma_engine::rng::XorShift;

const N_SENSORS: usize = 4;
const N_MOTORS: usize = 4;

fn skinner_grown() -> dioma::Grown {
    // Диагональ: сенсор i → мотор i.
    let mut edges = Vec::new();
    for s in 0..N_SENSORS {
        edges.push((s, N_SENSORS + s, 3.0));
    }
    dioma::Grown {
        edges,
        region_of: vec![0usize; N_SENSORS]
            .into_iter()
            .chain(vec![1usize; N_MOTORS])
            .collect(),
        nt: vec![1.0f32; N_TOTAL],
        growth_s: 0.0,
        rule_genes: Vec::new(),
        instincts: Vec::new(),
    }
}

const N_TOTAL: usize = N_SENSORS + N_MOTORS;

fn main() {
    println!("ЯЩИК СКИННЕРА");

    let grown = skinner_grown();
    let gains = vec![
        vec![0.0f32, 1.0], // сенсоры → моторы
        vec![0.0f32, 0.0],
    ];
    let mut brain = Brain::new(
        &grown,
        gains,
        PlasticityParams {
            w_max: 6.0,
            lr: 0.3,
            ..Default::default()
        },
    );

    let n = brain.n;
    let steps_per_tick = brain.steps_per_tick();
    println!("{} нейронов · {} подшагов/тик", n, steps_per_tick);

    // ── Обучение ──
    let mut rng = XorShift::new(42);
    let mut correct_total = 0usize;

    for trial in 1..=50 {
        // Случайный целевой сенсор.
        let target_sensor = (rng.unit() * 4.0) as usize % N_SENSORS;

        // Стимул на целевой сенсор.
        let ext = {
            let mut e = vec![3.0f32; n];
            e[target_sensor] += 5.0;
            e
        };

        // Шагаем много подшагов — нейроны интегрируют.
        let mut up_fired = false;
        for _ in 0..steps_per_tick {
            let spikes = brain.step(&ext);
            if spikes[N_SENSORS + target_sensor] != 0 {
                up_fired = true;
            }
        }

        // Награда за активность правильного мотора.
        if up_fired {
            correct_total += 1;
            brain.reward_now(1.0);
        } else {
            brain.reward_now(-0.2);
        }

        if trial % 10 == 0 {
            println!(
                "  проба {:>2}: {}/{} правильных",
                trial, correct_total, trial
            );
        }
    }

    println!(
        "\nточность обучения: {:.0}%",
        correct_total as f64 / 50.0 * 100.0
    );
}
