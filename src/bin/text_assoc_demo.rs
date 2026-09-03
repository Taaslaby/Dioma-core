//! text_assoc_demo — магия: мозг учит слово «яблоко» = еда
//! Запуск: `cargo run --release --bin text_assoc_demo`
//! Без Unity, без скриптов — только STDP + дофамин + кора.
//!
//! Протокол:
//! 1. До обучения слово — пустой звук (steer ~0)
//! 2. 300 тиков: слово + еда → дофамин 1.0 → LTP TEXT→CORTEX_E→ML
//! 3. После: слово ОДНО даёт поворот и тягу как запах

use dioma_engine::biology::metabolism::chem;
use dioma_engine::creature::{Creature, Sense, REGION_CORTEX_E, REGION_ML};

fn avg_steer(c: &mut Creature, s: Sense) -> f32 {
    let mut acc = 0.0;
    for _ in 0..40 { acc += c.tick(&s).steer; }
    acc / 40.0
}
fn avg_drive(c: &mut Creature, s: Sense) -> f32 {
    let mut acc = 0.0;
    for _ in 0..40 { acc += c.tick(&s).drive; }
    acc / 40.0
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  DIOMA — мозг учит слово «яблоко» = еда                  ║");
    println!("║  STDP + дофамин + кора · 760 нейронов · без Unity        ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Слово как ток 0..1 в кору (TEXT_CURRENT 4.0). Кодируем: 1="яблоко", 0=тишина.
    let word_apple = Sense { text: 1.0, ..Default::default() };
    let word_stone = Sense { text: 0.3, ..Default::default() }; // другое слово — слабый ток
    let silence = Sense::default();

    // ── До ───────────────────────────────────────────────────────────
    let mut naive = Creature::new(42);
    println!("── До обучения ─────────────────────────────────────────────");
    println!("  слово 'яблоко' → steer {:.3}, drive {:.3}", avg_steer(&mut naive.clone(), word_apple), avg_drive(&mut naive.clone(), word_apple));
    println!("  тишина          → steer {:.3}, drive {:.3}\n", avg_steer(&mut naive, silence), avg_drive(&mut Creature::new(42), silence));

    // ── Обучение: слово + еда ────────────────────────────────────────
    println!("── Обучение: 300× «яблоко» + еда (дофамин) ────────────────");
    let mut learner = Creature::new(42);
    learner.blood.conc[chem::HUNGER] = 200.0;
    for _ in 0..300 {
        learner.tick(&Sense { text: 1.0, food_near: true, ..Default::default() });
    }
    let gain_good = learner.brain.learned_gain(REGION_CORTEX_E, REGION_ML);
    println!("  съедено: {}, learned_gain CORTEX_E→ML = {:.1}", learner.food_eaten(), gain_good);
    println!("  кровь: дофамин {:.1}, голод {:.0} → {:.0}\n", learner.dopamine_level(), 200.0, learner.hunger());

    // Контраст: слово «камень» + боль
    let mut stone_learner = Creature::new(42);
    stone_learner.blood.conc[chem::HUNGER] = 200.0;
    for _ in 0..300 {
        let s = Sense { text: 0.3, wall: true, ..Default::default() };
        stone_learner.tick(&s);
    }
    let gain_bad = stone_learner.brain.learned_gain(REGION_CORTEX_E, REGION_ML);
    println!("  слово 'камень' + удар → gain {:.1} (должен быть ≤0)\n", gain_bad);

    // ── После ─────────────────────────────────────────────────────────
    println!("── После обучения ──────────────────────────────────────────");
    let steer_apple = avg_steer(&mut learner, word_apple);
    let drive_apple = avg_drive(&mut learner, word_apple);
    let steer_stone = avg_steer(&mut stone_learner, word_stone);
    let steer_silence = avg_steer(&mut Creature::new(99), silence); // контроль - другой сид

    println!("  'яблоко' → steer {:.3} (был ~0), drive {:.3} (оживление)", steer_apple, drive_apple);
    println!("  'камень' → steer {:.3} (слабее/в другую сторону)", steer_stone);
    println!("  тишина   → steer {:.3}\n", steer_silence);

    // ── Наследство ────────────────────────────────────────────────────
    println!("── Наследство: ребёнок рождается зная слово ──────────────");
    let mut child = learner.spawn_child(99);
    child.blood.conc[chem::HUNGER] = 200.0;
    let child_steer = avg_steer(&mut child, Sense { text: 0.2, ..Default::default() });
    let mut naive_child = Creature::new(1).spawn_child(99);
    naive_child.blood.conc[chem::HUNGER] = 200.0;
    let naive_steer = avg_steer(&mut naive_child, Sense { text: 0.2, ..Default::default() });
    println!("  ребёнок знающего (слабый шёпот 0.2): steer {:.3}", child_steer);
    println!("  ребёнок наивного: steer {:.3}", naive_steer);
    println!("  → Болдуин: ген инстинкта TEXT→моторика прожжён\n");

    // ── Финал ─────────────────────────────────────────────────────────
    if steer_apple.abs() > 0.1 && gain_good > 100.0 && child_steer.abs() > naive_steer.abs() {
        println!("╔════════════════════════════════════════════════════════════╗");
        println!("║  ✨ СЛОВО СТАЛО ЕДОЙ. Мозг связал звук с наградой,        ║");
        println!("║  а ребёнок унаследовал его без единого укуса.            ║");
        println!("║  760 нейронов, 0 скриптов — только STDP и кровь.         ║");
        println!("╚════════════════════════════════════════════════════════════╝");
    } else {
        println!("(ожидался сильный эффект — проверь gains, возможно нужен другой сид)");
    }
}

// Clone для naive теста — Creature не Clone, делаем через seed
trait CloneNaive { fn clone(&self) -> Self; }
impl CloneNaive for Creature {
    fn clone(&self) -> Self { Creature::new(self.seed) }
}
