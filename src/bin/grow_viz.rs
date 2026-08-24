//! -------------------------------------------------------------------------
//! Filename:    src/bin/grow_viz.rs
//! Class:       SemantikaEngine
//! Purpose:     покадровый фильм роста мозга агента в прометеусе
//! Description: тот же сетап, что engine::grow::grow() для agent_layout,
//!              но шагаем порциями и рисуем вид сверху (x-y):
//!              сомы буквами региона, аксоны точками, конусы роста '@'.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::engine::grow::{agent_layout, Steering};
use dioma::rng::XorShift;
use prometheus_grow::{
    ArtificialGradient, Bounds, CellType, ChemoAttraction, Grid, Secretor, Substance, World,
};
use std::collections::HashMap;

fn main() {
    let layout = agent_layout();
    let st = Steering::default();

    // ── Сетап мира (копия grow(), без изменений логики) ──
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for r in &layout.regions {
        for k in 0..3 {
            lo[k] = lo[k].min(r.center[k] - r.radius);
            hi[k] = hi[k].max(r.center[k] + r.radius);
        }
    }
    let margin = 70.0;
    let origin = prometheus_grow::Vec3::new(lo[0] - margin, lo[1] - margin, lo[2] - margin);
    let span = ((hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]) + 2.0 * margin).max(50.0);
    let grid = Grid::new(
        (span / 5.0).ceil() as usize,
        (span / 5.0).ceil() as usize,
        (span / 5.0).ceil() as usize,
        5.0,
        origin,
    );
    let bounds = Bounds {
        min: origin,
        max: prometheus_grow::Vec3::new(origin.x + span, origin.y + span, origin.z + span),
    };
    let mut world = World::new(grid, 1.0, bounds, 7);
    world.synapse_dist = 0.001;
    world.max_syn_per_cell = usize::MAX;
    world.max_cells = layout.n().max(64).next_power_of_two();
    world.max_neurites = layout.n().next_power_of_two() * 16;

    let mut sources: HashMap<String, Vec<[f64; 3]>> = HashMap::new();
    for r in &layout.regions {
        if let Some(m) = &r.secrete {
            sources.entry(m.clone()).or_default().push(r.center);
        }
    }
    for (name, cents) in &sources {
        let cx = cents.iter().map(|c| c[0]).sum::<f64>() / cents.len() as f64;
        let cy = cents.iter().map(|c| c[1]).sum::<f64>() / cents.len() as f64;
        let bumps = vec![
            ArtificialGradient::Gaussian { axis: 0, max: st.bump_max, mean: cx, sigma: st.bump_sigma },
            ArtificialGradient::Gaussian { axis: 1, max: st.bump_max, mean: cy, sigma: st.bump_sigma },
        ];
        world.add_substance(Substance::new(name, 0.2, Steering::DEGRADATION, &world.grid, bumps));
    }

    let mut rng = XorShift::new(777); // тот же сид, что в демо-прогонах
    for r in &layout.regions {
        for _ in 0..r.count {
            let (u, v) = (rng.unit() * 2.0 - 1.0, rng.unit() * 2.0 * std::f64::consts::PI);
            let c = u.cos();
            let rad = r.radius * rng.unit().cbrt();
            let xy = (1.0 - u * u).max(0.0).sqrt() * rad;
            let p = prometheus_grow::Vec3::new(
                r.center[0] + xy * v.cos(),
                r.center[1] + xy * v.sin(),
                r.center[2] + c * rad,
            );
            let ty = if r.inhibitory { CellType::Inhibitory } else { CellType::Excitatory };
            let cell_id = world.add_cell(p, ty, 4.0);
            world.cells[cell_id].tag = r.tag.to_string();
            if let Some(m) = &r.secrete {
                world.add_soma_module(cell_id, Box::new(Secretor::new(m, 0.06)));
            }
            for a in &r.axons {
                let nb = world.new_neurite(cell_id, true, 0.5);
                let mut chemo = ChemoAttraction::new(&a.target);
                chemo.noise_weight = st.noise_weight;
                chemo.gradient_weight = st.gradient_weight;
                chemo.old_dir_weight = st.old_dir_weight;
                chemo.turn_rate_deg = Some(st.turn_rate_deg);
                chemo.stop_concentration = st.stop_concentration;
                chemo.max_len_um = Some(a.max_len);
                world.add_neurite_module(cell_id, nb, Box::new(chemo));
            }
        }
    }

    // ── Кадры роста ──
    const FRAMES: [usize; 5] = [0, 60, 150, 260, 400];
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  РОСТ МОЗГА АГЕНТА В ПРОМЕТЕУСЕ (вид сверху, x–y)      ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!("  сомы: s=SL r=SR i=(тормоза) l=ML m=MR · аксоны: · · конусы: @");

    let mut done = 0usize;
    for &frame in &FRAMES {
        while done < frame {
            world.step();
            done += 1;
        }
        draw(&world, frame);
    }

    // ── Узнавание и синапсы (финал) ──
    let mut allowed = HashMap::new();
    let mut prob = HashMap::new();
    for (a, b) in &layout.allowed_pairs {
        allowed.insert((a.clone(), b.clone()), true);
        prob.insert((a.clone(), b.clone()), 1.0);
    }
    use prometheus_grow::TagRecognition;
    let rec = TagRecognition { allowed, prob };
    world.form_synapses_tagged(st.synapse_dist, &rec);

    let g = world.to_graph();
    let tag_of_node: Vec<usize> = world
        .cells
        .iter()
        .map(|c| match c.tag.as_str() {
            "SL" => 0,
            "SR" => 1,
            "I" => 2,
            "ML" => 3,
            _ => 4,
        })
        .collect();
    let mut by_pair: HashMap<(usize, usize), usize> = HashMap::new();
    for e in &g.edges {
        *by_pair
            .entry((tag_of_node[e.pre], tag_of_node[e.post]))
            .or_default() += 1;
    }
    println!("\n  ══ СИНАПСЫ ПОСЛЕ УЗНАВАНИЯ ({} всего) ══", g.edges.len());
    for (ri, r) in layout.regions.iter().enumerate() {
        for (rj, rj_r) in layout.regions.iter().enumerate() {
            if let Some(&n) = by_pair.get(&(ri, rj)) {
                println!("    {} → {}: {:>4} синапсов", r.tag, rj_r.tag, n);
            }
        }
    }
}

/// Вид сверху: холст 66×33 по координатам [-95..95]².
fn draw(world: &World, step: usize) {
    const W: usize = 66;
    const H: usize = 33;
    const HALF: f64 = 95.0;
    let mut canvas = vec![vec![' '; W]; H];
    let put = |cv: &mut Vec<Vec<char>>, x: f64, y: f64, ch: char| {
        let cx = ((x + HALF) / (2.0 * HALF) * (W - 1) as f64).round() as isize;
        let cy = ((HALF - y) / (2.0 * HALF) * (H - 1) as f64).round() as isize;
        if cx >= 0 && cx < W as isize && cy >= 0 && cy < H as isize {
            cv[cy as usize][cx as usize] = ch;
        }
    };

    let mut tips = 0usize;
    let mut total_len = 0.0f64;
    for cell in &world.cells {
        // Аксоны: сегменты точками, конус — '@'.
        for n in &cell.neurites {
            for seg in &n.points {
                put(&mut canvas, seg.x, seg.y, '·');
            }
            total_len += n.length_um();
            if n.tip().x.is_finite() {
                put(&mut canvas, n.tip().x, n.tip().y, '@');
                tips += 1;
            }
        }
    }
    // Сомы поверх (последние).
    for cell in &world.cells {
        let glyph = match cell.tag.as_str() {
            "SL" => 's',
            "SR" => 'r',
            "I" => 'i',
            "ML" => 'l',
            "MR" => 'm',
            _ => '?',
        };
        put(&mut canvas, cell.soma.pos.x, cell.soma.pos.y, glyph);
    }

    println!("\n  ── шаг {step:>3} ── конусов: {tips:>3} · длина аксонов: {total_len:>6.0} мкм ──");
    for row in &canvas {
        let line: String = row.iter().collect();
        println!("   |{line}|");
    }
}
