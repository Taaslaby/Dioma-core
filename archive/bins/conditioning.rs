//! -------------------------------------------------------------------------
//! Filename:    src/bin/conditioning.rs
//! Class:       SemantikaEngine
//! Purpose:     классическое обусловливание: танцуй вверх-вниз при свете!
//! Description: ТЕСТ 1 — Аппетитивное обучение: свет → танцуй (верх/низ)
//!              → вкусняшка. Мозг учится ассоциировать свет с наградой.
//!              ТЕСТ 2 — Аверсивное обучение: шум в мозг при ошибке.
//!              Неправильное действие → болевой сигнал → избегание.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma_engine::rng::XorShift;
use pillars_connectome::{BrainConfig, BrainRunner, ConnectomeData};

fn main() {
    println!("╔═══════════════════════════════════════╗");
    println!("║  КЛАССИЧЕСКОЕ ОБУСЛОВЛИВАНИЕ          ║");
    println!("╚═══════════════════════════════════════╝");

    // ── Создаём крошечный мозг: 4 нейрона ──
    // Нейрон 0: MOTOR_UP (танцует вверх)
    // Нейрон 1: MOTOR_DOWN (танцует вниз)
    // Нейрон 2: LIGHT_SENSOR (видит лампочку)
    // Нейрон 3: SHOCK_SENSOR (чувствует боль)

    let n = 4;
    // Полносвязный: все ко всем.
    let mut rp = vec![0i32; n + 1];
    let mut ci = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        rp[i] = ci.len() as i32;
        for j in 0..n {
            if i != j {
                ci.push(j as i32);
                vals.push(0.8); // начальный вес
            }
        }
    }
    rp[n] = ci.len() as i32;
    let nt = vec![1.0f32; n];
    let conn = ConnectomeData::new(rp.clone(), ci.clone(), vals.clone(), nt, vec![]);
    let mut runner = BrainRunner::new(conn, BrainConfig::lif_alpha(0.05)).unwrap();

    println!("мозг: 4 нейрона (UP, DOWN, LIGHT, SHOCK)");
    println!("задача: когда СВЕТ → чередуй ВВЕРХ и ВНИЗ\n");

    // ── ТЕСТ 1: Обучение танцу со вкусняшкой ──
    println!("── ФАЗА ОБУЧЕНИЯ (30 проб) ──");
    let mut dance_score = 0usize;
    let _prev_dir = 0i32; // 0=нет, 1=вверх, -1=вниз

    for trial in 1..=30 {
        // Свет включается.
        let ext = vec![3.0f32, 3.0, 5.0, 0.0]; // LIGHT яркий

        let mut up_fired = false;
        let mut down_fired = false;
        let _up_before_down = false;

        // Проба: 100 шагов (5 мс).
        for step in 0..100 {
            let buf = runner.run(1, &ext);

            let up_spiked = buf[0] != 0;
            let down_spiked = buf[1] != 0;

            if step < 50 {
                // первая половина = ожидаем ВВЕРХ
                if up_spiked {
                    up_fired = true;
                }
            } else {
                // вторая половина = ожидаем ВНИЗ
                if down_spiked {
                    down_fired = true;
                }
            }
        }

        // Правильный танец: сначала вверх, потом вниз.
        let danced = up_fired && down_fired;
        if danced {
            dance_score += 1;
        }

        // Награда если танцевал.
        if danced {
            runner.run(1, &[0.0, 0.0, 0.0, 0.0]); // пустой шаг для тайминга
        }

        if trial % 10 == 0 {
            println!(
                "  проба {:>2}: танец={}",
                trial,
                if danced { "✓" } else { "✗" }
            );
        }
    }
    println!("  итог обучения: {}/30 танцев", dance_score);

    // ── ТЕСТ 2: Аверсивное обучение (шок при ошибке) ──
    println!("\n── ФАЗА НАКАЗАНИЯ (30 проб) ──");
    println!("правило: НЕ стреляй дважды подряд одним нейроном");
    let mut shocks = 0usize;
    let mut correct_avoids = 0usize;
    let mut last_winner = 0i32;

    for trial in 1..=30 {
        let ext = vec![3.0f32, 3.0, 5.0, 0.0];
        let mut winner_this_trial = 0i32;

        for _step in 0..50 {
            let buf = runner.run(1, &ext);
            if buf[0] != 0 {
                winner_this_trial = 0;
                break;
            }
            if buf[1] != 0 {
                winner_this_trial = 1;
                break;
            }
        }

        // ШОК: если тот же победил что и в прошлый раз — шум в мозг!
        if winner_this_trial == last_winner && last_winner >= 0 {
            // Шум: случайный ток на все нейроны.
            let mut rng = XorShift::new(trial * 7919);
            let noise_ext: Vec<f32> = (0..n).map(|_| rng.unit() as f32 * 40.0).collect();
            runner.run(1, &noise_ext);
            shocks += 1;
            println!(
                "  ⚡ ШОК! проба {} — повтор {}",
                trial,
                if winner_this_trial == 0 { "UP" } else { "DOWN" }
            );
        } else {
            correct_avoids += 1;
        }

        last_winner = winner_this_trial;
    }
    println!("  шоков получено: {shocks} · правильных избеганий: {correct_avoids}");

    // ── ИТОГ ──
    println!("\n╔ ИТОГ ═════════════════════════════╗");
    println!("║ Танцев выучено: {}/30", dance_score);
    println!("║ Шоков получено: {}/30", shocks);
    println!("║ Правильных избеганий: {}/30", correct_avoids);
    println!("╚════════════════════════════════════╝");
}
