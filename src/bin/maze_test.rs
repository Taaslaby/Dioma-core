//! ЛАБИРИНТ — тест памяти и решений для разных мозгов.
use dioma::engine::neurons::{Brain, PlasticityParams};
use dioma::heritage::sardna_to_grown;
use dioma::world::arena::{grow_agent, Genome};

const TICKS_PER_PHASE: usize = 100;

/// Простой T-выбор: маяк → тьма → поворот налево или направо?
fn run_maze(brain: &mut Brain, n: usize, n_trials: usize) -> (usize, usize) {
    let mut correct = 0usize;
    let wrong = 0usize;
    let half = n / 2;

    for trial in 0..n_trials {
        let correct_side = if trial % 2 == 0 { 0usize } else { 1usize }; // чередуем

        // Фаза 1: МАЯК — светим в сторону правильного рукава (50 тиков).
        let beacon = {
            let mut e = vec![0.5f32; n];
            let start = if correct_side == 0 { 0 } else { half };
            for cell in e.iter_mut().skip(start).take(half) { *cell += 3.0; }
            e
        };
        for _ in 0..TICKS_PER_PHASE { brain.step(&beacon); }

        // Фаза 2: ТЬМА — ноль входа. Смотрим куда тянет мозг.
        let dark = vec![0.12f32; n]; // только тоник
        let mut left_act = 0i64;
        let mut right_act = 0i64;
        for _ in 0..TICKS_PER_PHASE / 2 {
            let spikes = brain.step(&dark);
            left_act += spikes[..half].iter().map(|&s| s as i64).sum::<i64>();
            right_act += spikes[half..].iter().map(|&s| s as i64).sum::<i64>();
        }

        // Выбор: какая сторона активнее — туда и идём.
        let chose_left = left_act >= right_act;
        if chose_left == (correct_side == 0) {
            correct += 1;
            brain.reward_now(1.0);
        } else {
            brain.reward_now(-0.5);
        }
    }
    (correct, wrong)
}

fn main() {
    println!("ЛАБИРИНТ: маяк → тьма → выбор рукава");
    println!("20 проб, чередуем сторону\n");

    // ── Grown Agent ──
    let genome = Genome::wildtype(42);
    let grown = grow_agent(&genome);
    let gains = dioma::world::arena::arena_gains(&genome);
    let nn = grown.region_of.len();
    let mut b = Brain::new(&grown, gains, PlasticityParams::default());
    let (c, w) = run_maze(&mut b, nn, 20);
    println!("Grown Agent: {}✓ {}✗ ({:.0}%)", c, w, c as f64 / 20.0 * 100.0);

    // ── SarDNA Pro ──
    match sardna_to_grown("D:/Проекты/asm_engines/sardaanaCM/data/generated/SarDNA_Pro_best.taas") {
        Ok(g) => {
            let gains = vec![vec![0.6f32; 4]; 4];
            let cols = dioma::body::physiology::brain_columns(
                &g.region_of, dioma::body::physiology::cognitive_archetype, &Default::default());
            let params = PlasticityParams {
                homeo_every_ticks: 1,
                ..Default::default()
            };
            let mut b = Brain::new_haap(&g, gains, params, &cols);
            let n_haap = b.n;
            let half_haap = n_haap / 4; // четверть для каждой стороны
            let _ = half_haap;
            
            let (c, w) = run_maze_haap(&mut b, n_haap, 20);
            println!("SarDNA HAAP: {}✓ {}✗ ({:.0}%)", c, w, c as f64 / 20.0 * 100.0);
        }
        Err(e) => println!("SarDNA error: {e}"),
    }

    // ── Diff SNN ──
    match sardna_to_grown("D:/Проекты/asm_engines/sardaanaCM/data/generated/Diff_snn_best.taas") {
        Ok(g) => {
            let gains = vec![vec![0.6f32; 4]; 4];
            let nn = g.region_of.len();
            let mut b = Brain::new(&g, gains, PlasticityParams::default());
            let (c, w) = run_maze(&mut b, nn, 20);
            println!("Diff SNN: {}✓ {}✗ ({:.0}%)", c, w, c as f64 / 20.0 * 100.0);
        }
        Err(e) => println!("Diff error: {e}"),
    }
}

/// MAZE для HAAP-мозга.
fn run_maze_haap(brain: &mut Brain, n_haap: usize, n_trials: usize) -> (usize, usize) {
    let mut correct = 0usize;
    let wrong = 0usize;

    for trial in 0..n_trials {
        let correct_side = trial % 2;
        
        // Маяк.
        let beacon = {
            let mut e = vec![3.5f32; n_haap];
            if correct_side == 0 {
                for cell in e.iter_mut().take(n_haap / 2) { *cell += 3.0; }
            } else {
                for cell in e.iter_mut().skip(n_haap / 2).take(n_haap / 2) { *cell += 3.0; }
            }
            e
        };
        for _ in 0..50 { brain.step(&beacon); }

        // Тьма + выбор.
        let dark = vec![0.12f32; n_haap];
        let mut left_act = 0i64;
        let mut right_act = 0i64;
        for _ in 0..50 {
            let spikes = brain.step(&dark);
            left_act += spikes[..n_haap/2].iter().map(|&s| s as i64).sum::<i64>();
            right_act += spikes[n_haap/2..].iter().map(|&s| s as i64).sum::<i64>();
        }

        let chose_left = left_act >= right_act;
        if chose_left == (correct_side == 0) {
            correct += 1;
            brain.reward_now(1.0);
        } else {
            brain.reward_now(-0.5);
        }
    }
    (correct, wrong)
}
