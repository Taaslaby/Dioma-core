// ВРЕМЕННЫЙ зонд (удалить): оба триала детюра на 760н.
use dioma_engine::biology::metabolism::{chem, Blood};
use dioma_engine::env::map::WorldMap;
use dioma_engine::env::math::Vec2;
use dioma_engine::sim::Arena;

#[test]
fn probe_detour_760_2trials() {
    let mut map = WorldMap::default();
    map.add_room(0.0, 0.0, 20.0, 20.0);
    map.add_room(20.0, 10.0, 60.0, 20.0);
    let mut a = Arena::new(42, map);
    a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
    for trial in 0..2 {
        a.critters[0].creature.blood = Blood::creature(false);
        a.critters[0].creature.blood.conc[chem::HUNGER] = 255.0;
        for id in a.berries() {
            a.world.kill(id);
        }
        let berry = a.add_berry(Vec2::new(55.0, 15.0));
        let h0 = a.critters[0].wall_hits;
        let mut ate_at = None;
        for t in 1..=1200i32 {
            a.step();
            if let Some(ag) = a.world.agent(berry) {
                if ag.vars[1] > 0.0 {
                    ate_at = Some(t);
                    break;
                }
            }
            if t % 200 == 0 {
                let c = &a.critters[0];
                println!("триал{} t={t:>4} pos=({:>5.1},{:>5.1}) head={:.2} голод={:>4.0} глюк={:>4.0} устал={:>4.0} спит={} ударов={}",
                    trial + 1, c.pos.x, c.pos.y, c.heading, c.creature.hunger(),
                    c.creature.glucose(), c.creature.fatigue(), c.sleeping, c.wall_hits - h0);
            }
        }
        match ate_at {
            Some(t) => println!("триал{}: СЪЕЛ на тике {t}, ударов {}", trial + 1, a.critters[0].wall_hits - h0),
            None => {
                let c = &a.critters[0];
                println!("триал{}: НЕ СЪЕЛ. pos=({:.1},{:.1}) голод={:.0} глюк={:.0} устал={:.0} спит={} ударов={}",
                    trial + 1, c.pos.x, c.pos.y, c.heading, c.creature.hunger(),
                    c.creature.fatigue(), c.sleeping, c.wall_hits - h0);
            }
        }
        assert!(ate_at.is_some(), "триал {}: должно съесть", trial + 1);
        a.critters[0].pos = Vec2::new(5.0, 5.0);
        a.critters[0].heading = 0.0;
    }
}
