//! -------------------------------------------------------------------------
//! Filename:    src/bin/sardana_skinner.rs
//! Class:       SemantikaEngine
//! Purpose:     Ящик Скиннера на сардаана-мозге (256 нейронов от трансформера)
//! Description: та же задача что skinner-box но мозг — sarDNA коннектом.
//!              Сенсоры → моторы через настоящую трансформерную проводку.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::brain::{Brain, PlasticityParams};
use dioma::heritage::sardna_to_grown;
use taas_format::TaasReader;

fn main() {
    println!("╔═══════════════════════════════════════╗");
    println!("║  SARDAANA ЯЩИК СКИННЕРА               ║");
    println!("╚═══════════════════════════════════════╝");

    // ── 1. Грузим сардаану ──
    let path = "D:/Проекты/asm_engines/sardaanaCM/data/generated/SarDNA_Pro_best.taas";
    let r = TaasReader::from_file(path).unwrap();
    let rp: Vec<i32> = r.get_i32("row_ptr").unwrap();
    let ci: Vec<i32> = r.get_i32("col_indices").unwrap();
    let vals_raw: Vec<f32> = r.get_f32("values").unwrap();
    let nt: Vec<f32> = r.get_f32("nt_types").unwrap();
    
    let n = rp.len().saturating_sub(1);
    let n_syn = ci.len();

    // Масштабируем веса.
    let w_max = vals_raw.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if w_max > 0.0 { 2.0 / w_max } else { 1.0 };
    let vals: Vec<f32> = vals_raw.iter().map(|&v| v * scale).collect();

    // Назначаем регионы по степени (как в heritage).
    let degrees: Vec<usize> = rp.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let mut sorted_deg = degrees.clone();
    sorted_deg.sort_unstable_by(|a, b| b.cmp(a));
    let motor_thresh = sorted_deg[n * 25 / 100];
    let region_of: Vec<usize> = degrees
        .iter().map(|&d| if d >= motor_thresh { 3 } else { 0 }).collect();

    // Строим ConnectomeData.
    let conn = ConnectomeData::new(rp.clone(), ci.clone(), vals.clone(), nt.clone(), vec![]);

    // ── 2. BrainRunner ──
    let mut brain = BrainRunner::new(conn, BrainConfig::lif_alpha(0.05)).unwrap();
    
    println!("сардаана: {n} нейронов · {n_syn} синапсов");
    println!("моторов (степень ≥{motor_thresh}): {}", 
        degrees.iter().filter(|&&d| d >= motor_thresh).count());

    // Определяем моторные и сенсорные нейроны по регионам.
    let motors: Vec<usize> = (0..n).filter(|&i| region_of[i] == 3).collect();
    let sensors: Vec<usize> = (0..n).filter(|&i| region_of[i] != 3).take(N_SENSORS_MAX).collect();
    const N_SENSORS_MAX: usize = 20;

    println!("моторных нейронов: {} · сенсорных: {}", motors.len(), sensors.len());

    // ── 3. ОБУЧЕНИЕ: свет → правильный мотор → награда ──
    // Выбираем 4 целевых мотора из списка.
    let targets: Vec<usize> = motors.iter().step_by(motors.len() / 4 + 1)
        .take(4).cloned().collect();
    
    println!("\nцелевые моторы: {:?}", targets);

    let mut correct = 0usize;
    let mut wrong = 0usize;
    const TRIALS: usize = 50;

    for trial in 1..=TRIALS {
        // Свет: стимул на случайный сенсор.
        let stim_sensor = sensors[trial % sensors.len()];
        
        let mut ext = vec![3.0f32; n]; // базовый драйв
        ext[stim_sensor] += 10.0; // яркая вспышка

        // Шаг мозга — много подшагов для интеграции.
        let mut winner_motor = usize::MAX;
        let mut winner_rate = 0i64;
        for _ in 0..brain_steps() {
            let buf = brain.run(1, &ext);
            for (mi, &m_idx) in motors.iter().enumerate() {
                let s = buf[m_idx] as i64;
                if s > winner_rate { winner_rate = s; winner_motor = mi; }
            }
        }

        // Награда если правильный мотор стрельнул.
        if Some(winner_motor) == targets.iter().position(|&t| t == motors[winner_motor.min(motors.len()-1)]) {
            correct += 1;
        }
        wrong += 1 - correct.min(1);

        if trial % 10 == 0 {
            println!("  проба {:>2}: победил мотор {} ({})", trial, winner_motor,
                if winner_motor == targets[trial % targets.len()] {"✓"} else {"—"});
        }
    }
}

fn brain_steps() -> usize { 50 }

// Заглушки чтобы компилировалось — уберём при рефакторинге.
fn _unused() {}
