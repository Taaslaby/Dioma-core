//! Экспортёр «золотого следа» биохимии для Unity-порта (com.dioma.engine/Bio).
//!
//! Гоняет `Blood::creature(false)` по детерминированному сценарию (моторная
//! активность, еда, удары о стену, сон, плата за пластичность) и пишет
//! по-тиковый JSON-трейс всех 9 химий + 9 локусов + урона органа. C#-порт
//! реплеит тот же сценарий и сверяет числа — биохимия целочисленно-детерминированная,
//! так что ожидается совпадение с точностью float.
//!
//! Запуск: `cargo run --release --bin export_bio -- <dir>`
//! (dir = dioma-engine/Tests/Editor/Data).

use std::fs;
use std::path::PathBuf;

use dioma_engine::biology::metabolism::{chem, locus, Blood};

const N_TICKS: usize = 600;

/// Событие тика сценария (экспортируется — C# реплеит из JSON).
struct Script {
    /// Спайки моторных пулов за тик (пишется в locus::RATE_MOTOR).
    rate_motor: f32,
    ate: bool,
    wall: bool,
    /// Плата за пластичность (дофамин), 0 = не платим.
    pay: f32,
    sleep: bool,
}

/// Сценарий: разгон моторики (утомление), еда каждые 60 тиков, удары о стену,
/// плата за обучение после еды, сон на 400-м тике.
fn scripted(t: usize) -> Script {
    Script {
        // Моторная активность: тихо → рывок → сатурация (эмиттер FATIGUE).
        rate_motor: match t {
            0..=99 => 4.0,     // ниже порога эмиттера (6.0) — усталости нет
            100..=299 => 120.0,
            300..=449 => 320.0, // охотничий рывок
            _ => 40.0,
        },
        ate: t % 60 == 30,
        wall: t == 150 || t == 151 || t == 400,
        pay: if t % 60 == 31 { 1.0 } else { 0.0 },
        sleep: t == 500,
    }
}

fn f32s(v: &[f32]) -> String {
    v.iter()
        .map(|x| format!("{}", x))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).expect("out dir");

    let mut blood = Blood::creature(false);
    // Голод новорождённого арены (ARENA_NEWBORN_HUNGER): с нулём тело сотни
    // тиков ждёт распада стартовой глюкозы и не успевает захотеть есть.
    blood.conc[chem::HUNGER] = 120.0;

    let mut rows = Vec::with_capacity(N_TICKS);
    for t in 0..N_TICKS {
        let s = scripted(t);

        // Хост пишет драйв-локусы ДО update() — как `Creature::finish_tick`.
        blood.loci[locus::RATE_MOTOR] = s.rate_motor;
        blood.loci[locus::GLUCOSE_DEFICIT] = (150.0 - blood.conc[chem::GLUCOSE]).max(0.0);
        if s.wall {
            blood.bump_wall();
        }
        if s.ate {
            blood.ingest_food(60.0); // FOOD_PORTION
        }
        blood.update();
        let paid = if s.pay > 0.0 {
            blood.pay_for_plasticity(s.pay)
        } else {
            false
        };
        if s.sleep {
            blood.sleep();
        }

        rows.push(format!(
            "{{\"rateMotor\":{},\"ate\":{},\"wall\":{},\"pay\":{},\"sleep\":{},\
             \"paid\":{},\"conc\":[{}],\"loci\":[{}],\"damage\":{},\
             \"sensorGain\":{},\"motorTonic\":{},\"speed\":{},\"turnGain\":{},\
             \"wantsFood\":{},\"budget\":{}}}",
            s.rate_motor,
            s.ate,
            s.wall,
            s.pay,
            s.sleep,
            paid,
            f32s(&blood.conc),
            f32s(&blood.loci),
            blood.organ.damage,
            blood.sensor_gain(),
            blood.motor_tonic(),
            blood.speed(),
            blood.turn_gain(),
            blood.wants_food(),
            blood.plasticity_budget(),
        ));

        if t % 100 == 0 || t + 1 == N_TICKS {
            println!(
                "tick {:3}: глюкоза {:6.2} голод {:6.2} усталость {:6.2} боль {:6.2} \
                 кортизол {:5.2} ACh {:5.2} | gain {:.2} tonic {:.3} speed {:.2}",
                t,
                blood.conc[chem::GLUCOSE],
                blood.conc[chem::HUNGER],
                blood.conc[chem::FATIGUE],
                blood.conc[chem::PAIN],
                blood.conc[chem::CORTISOL],
                blood.conc[chem::ACETYLCHOLINE],
                blood.sensor_gain(),
                blood.motor_tonic(),
                blood.speed(),
            );
        }
    }

    let json = format!("{{\"ticks\":[\n{}\n]}}\n", rows.join(",\n"));
    let path = out_dir.join("dioma_bio_golden.json");
    fs::write(&path, &json).expect("write bio golden");
    println!("след биохимии: {} ({} тиков)", path.display(), N_TICKS);
}
