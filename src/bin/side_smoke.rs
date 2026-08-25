use dioma::world::side::sim::SideWorld;
fn main() {
    let mut w = SideWorld::new(3, 2026_0826);
    let mut max_fall = 0.0f64;
    for t in 0..3000 {
        w.step_tick();
        for f in &w.fruits {
            if f.alive && f.ripen_until.is_none() {
                max_fall = max_fall.max(f.body.y);
            }
        }
        if t % 1000 == 0 {
            let meals: usize = w.creatures.iter().map(|c| c.meals).sum();
            let xs: Vec<String> = w.creatures.iter().map(|c| format!("#{},{:.0},комн{}", c.id, c.body.x, c.body.room)).collect();
            println!("t={t} еды={meals} тела=[{}]", xs.join(" "));
        }
    }
    let meals: usize = w.creatures.iter().map(|c| c.meals).sum();
    let bitter: usize = w.creatures.iter().map(|c| c.bitter_eaten).sum();
    println!("ИТОГ: съедено={meals} горьких={bitter} макс_высота_плодов={max_fall:.0}");
    assert!(max_fall > 200.0, "плоды должны падать на землю (y>200)");
}