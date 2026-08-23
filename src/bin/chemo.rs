//! -------------------------------------------------------------------------
//! Filename:    src/bin/chemo.rs
//! Class:       DebugSuite
//! Purpose:     poc-chemo: blood v2 homeostasis demo
//! Description: creature chemistry vs always-hungry robot; fatigue, organ cost.
//! 
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use dioma::free_life::{run_life, SIM_TICKS};

fn main() {
    println!("poc-chemo: кровь v2 — стехиометрия, локусы, органы (порт Creatures 3)");
    println!(
        "жизнь = {:.0} с, еды в мире = {}, регровт = 10 с\n",
        SIM_TICKS as f64 * 0.025,
        dioma::free_life::N_FOOD
    );

    let t0 = std::time::Instant::now();
    let chem = run_life(false, 0xA_E11);
    let robot = run_life(true, 0xA_E11);

    let show = |tag: &str, r: &dioma::free_life::LifeReport| {
        println!(
            "{tag}: съедено = {:>3}, приёмов пищи = {:>2}, голодных тиков = {:>4}/{}, \
             min глюкоза = {:>4.0}, пик утомления = {:>5.1}, урон мозга = {:.2}{}",
            r.eaten,
            r.meals,
            r.hungry_ticks,
            SIM_TICKS,
            r.min_glucose,
            r.peak_fatigue,
            r.brain_damage,
            if r.dead { " — УМЕР" } else { "" }
        );
    };
    show("ХИМИЯ ", &chem);
    show("РОБОТ ", &robot);

    println!("\n═══ СВОДКА ═══");
    println!(
        "гомеостаз: химия съедает {} против {} у робота; утомление достигает {:.1} (моторная активность реально стоит сил)",
        chem.eaten, robot.eaten, chem.peak_fatigue
    );
    let verdict = if chem.eaten < robot.eaten
        && chem.min_glucose >= 0.0
        && chem.meals >= 3
        && chem.peak_fatigue > 10.0
    {
        "КРОВЬ v2 РАБОТАЕТ: стехиометрический гомеостаз + утомление от мышления"
    } else {
        "биохимия не сошлась — настраивать реакции"
    };
    println!("ВЕРДИКТ: {verdict}");
    println!("\nполное время: {:.2} с", t0.elapsed().as_secs_f64());
}
