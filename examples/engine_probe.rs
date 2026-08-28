//! Проба: объезд L-стены на движке SiMuse — ест ли существо вообще.
use dioma::biology::metabolism::{chem, Blood};
use dioma::env::map::WorldMap;
use dioma::env::Vec2;
use dioma::sim::Arena;

fn main() {
    let mut map = WorldMap::default();
    map.add_room(0.0, 0.0, 20.0, 20.0);
    map.add_room(20.0, 10.0, 60.0, 20.0);
    let mut a = Arena::new(42, map);
    a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
    for trial in 0..2 {
        a.critters[0].creature.blood = Blood::creature(false);
        a.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        let berry = a.add_berry(Vec2::new(55.0, 15.0));
        let h0 = a.critters[0].wall_hits;
        let mut ate_at = None;
        for t in 0..3000u32 {
            a.step();
            if a
                .world
                .agent(berry)
                .map(|ag| ag.vars[1] > 0.0)
                .unwrap_or(false)
            {
                ate_at = Some(t);
                break;
            }
        }
        println!(
            "trial {trial}: ate_at={ate_at:?} wall_hits={} pos={:?} heading={:.2}",
            a.critters[0].wall_hits - h0,
            a.critters[0].pos,
            a.critters[0].heading
        );
        a.critters[0].pos = Vec2::new(5.0, 5.0);
        a.critters[0].heading = 0.0;
    }
}
