//! Валидация ручной репродукции тика 1: cpu1(полный) vs cpu2(ручной),
//! gpu1(полный) vs gpu2(ручной). Если репродукция верна — оба должны
//! совпасть; тогда источник расхождения локализуется точнее.
use dioma::creature::{Creature, Sense};

fn wd(a: &Creature, b: &Creature) -> f32 {
    let wa = a.brain.csr_values();
    let wb = b.brain.csr_values();
    wa.iter().zip(&wb).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn main() {
    let seed = 7u64;
    let s0 = Sense { smell_l: 0.6, smell_r: 0.6, vis_l: 0.0, vis_r: 0.0, wall_l: 0.9, wall_r: 0.0, food_near: true, wall: true, place: 0, heading: 0.0, see_eat: false };
    let s1 = Sense { smell_l: 0.1, smell_r: 0.1, vis_l: 0.0, vis_r: 0.0, wall_l: 0.0, wall_r: 0.0, food_near: true, wall: false, place: 0, heading: 0.0, see_eat: false };

    // Полные
    let mut cpu1 = Creature::new(seed);
    let mut gpu1 = Creature::new_gpu(seed);
    cpu1.tick(&s0);
    gpu1.tick(&s0);
    cpu1.tick(&s1);
    gpu1.tick(&s1);
    println!("полный: cpu vs gpu wmaxdiff={:.6}", wd(&cpu1, &gpu1));

    // Ручные (тик 0 полный, тик 1 — все фазы)
    let mut cpu2 = Creature::new(seed);
    let mut gpu2 = Creature::new_gpu(seed);
    cpu2.tick(&s0);
    gpu2.tick(&s0);
    let manual = |c: &mut Creature, g: &mut Creature| {
        let n = c.brain.n;
        let gc = c.blood.sensor_gain();
        let gg = g.blood.sensor_gain();
        let mk = |c: &Creature, gain: f32| -> Vec<f32> {
            let mut e = vec![0.12f32; n];
            let sl = c.brain.region_bounds(0);
            let sr = c.brain.region_bounds(1);
            let wl = c.brain.region_bounds(2);
            let wr = c.brain.region_bounds(3);
            for i in sl.0..sl.1 { e[i] += s1.smell_l * gain * 4.0; }
            for i in sr.0..sr.1 { e[i] += s1.smell_r * gain * 4.0; }
            for i in wl.0..wl.1 { e[i] += s1.wall_l * gain * 4.0; }
            for i in wr.0..wr.1 { e[i] += s1.wall_r * gain * 4.0; }
            e
        };
        let e_c = mk(c, gc);
        let e_g = mk(g, gg);
        let spt = c.brain.steps_per_tick();
        let mut sp = Vec::new();
        let mut sp_c = Vec::new();
        let mut sp_g = Vec::new();
        let (mut mlc, mut mrc, mut mlg, mut mrg) = (0i32, 0i32, 0i32, 0i32);
        let stride = n.div_ceil(64) * 64;
        for sub in 0..spt {
            c.brain.step_into(&e_c, &mut sp);
            sp_c.clone_from(&sp);
            mlc += sp[c.brain.region_bounds(5).0..c.brain.region_bounds(5).1].iter().sum::<i32>();
            mrc += sp[c.brain.region_bounds(6).0..c.brain.region_bounds(6).1].iter().sum::<i32>();
            g.brain.step_into(&e_g, &mut sp);
            sp_g.clone_from(&sp);
            mlg += sp[g.brain.region_bounds(5).0..g.brain.region_bounds(5).1].iter().sum::<i32>();
            mrg += sp[g.brain.region_bounds(6).0..g.brain.region_bounds(6).1].iter().sum::<i32>();
            let vc = c.brain.debug_cpu_v();
            let gc = c.brain.debug_cpu_g();
            let rc = c.brain.debug_cpu_refr();
            let st = g.brain.debug_gpu_states();
            let cin = c.brain.debug_cpu_in();
            let iin = g.brain.debug_gpu_i_in();
            let mut mdv = 0.0f32;
            let mut mvi = 0;
            let mut mdg = 0.0f32;
            let mut mgi = 0;
            let mut mdr = 0.0f32;
            let mut mri = 0;
            for i in 0..n {
                let dv = (vc[i] - st[0 * stride + i]).abs();
                let dg = (gc[i] - st[1 * stride + i]).abs();
                let dr = (rc[i] - st[2 * stride + i]).abs();
                if dv > mdv { mdv = dv; mvi = i; }
                if dg > mdg { mdg = dg; mgi = i; }
                if dr > mdr { mdr = dr; mri = i; }
            }
            println!("[ручной] подшаг {sub}: maxdv={mdv:.3e}(#{mvi}) maxdg={mdg:.3e}(#{mgi}) maxdr={mdr:.3e}(#{mri})");
            // Сравнение спайков этого подшага
            let mut spike_diff = 0;
            let mut first_sd = 0usize;
            for i in 0..n {
                if sp_c[i] != sp_g[i] {
                    spike_diff += 1;
                    if spike_diff == 1 { first_sd = i; }
                }
            }
            // Печатаем для подшагов 4..7
            if sub >= 4 && sub <= 6 {
                println!("   спайки: расхождений={spike_diff} первый=#{first_sd}");
                if mdg > 0.0 {
                    let i = mgi;
                    println!(
                        "  #{i}: v={:.9} g={:.9} refr={:.1} вход={:.9} | gpu v={:.9} g={:.9} refr={:.1} вход={:.9}",
                        vc[i], gc[i], rc[i], cin[i], st[0 * stride + i], st[1 * stride + i], st[2 * stride + i], iin[i]
                    );
                }
            }
        }
        c.blood.loci[dioma::biology::metabolism::locus::RATE_MOTOR] = (mlc + mrc) as f32;
        g.blood.loci[dioma::biology::metabolism::locus::RATE_MOTOR] = (mlg + mrg) as f32;
        c.blood.update();
        g.blood.update();
        // фаза 4: relief-награда (prev_wall=0.9 → relief=0.9)
        c.brain.reward_now(0.9 * 0.3);
        g.brain.reward_now(0.9 * 0.3);
    };
    manual(&mut cpu2, &mut gpu2);
    println!("ручной: cpu vs gpu wmaxdiff={:.6}", wd(&cpu2, &gpu2));
    println!("cpu1(полн) vs cpu2(руч): wmaxdiff={:.6}", wd(&cpu1, &cpu2));
    println!("gpu1(полн) vs gpu2(руч): wmaxdiff={:.6}", wd(&gpu1, &gpu2));
}