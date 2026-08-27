//! Этап 3 — рост коннектома на GPU: честный замер.
//!
//! Оба пути строят ОДИН мир прометея (build_world) и гоняют одинаковое число
//! шагов роста; разница только в диффузии морфогенов:
//!   CPU — rayon f64, GPU — wgpu compute (f32) с persistent-буферами.
//!
//! Сценарии: avoidance (игровой), cognitive (больше), синтетический
//! масштабированный (крупная сетка — там GPU обязан выигрывать).
//! Плюс санити: существо из GPU-выращенного коннектома собирается и тикает.
//!
//! Запуск: `cargo run --release --example grow_gpu_probe`
use dioma::creature::{Creature, Sense};
use dioma::engine::grow::{self, Layout, Steering};
use prometheus_grow::DiffusionBackend;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

/// Считать рёбра по парам регионов — топологическая «отпечатка» коннектома.
fn pair_counts(
    edges: &[(usize, usize, f32)],
    region_of: &[usize],
) -> BTreeMap<(usize, usize), usize> {
    let mut m = BTreeMap::new();
    for &(pre, post, _) in edges {
        *m.entry((region_of[pre], region_of[post])).or_insert(0) += 1;
    }
    m
}

/// Синтетическая схема: регионы раздвинуты в `scale` раз — сетка крупнее,
/// там диффузия тяжелее и GPU должен пересечь CPU.
fn scaled(layout: &Layout, scale: f64) -> Layout {
    let regions = layout
        .regions
        .iter()
        .map(|r| grow::RegionSpec {
            tag: r.tag,
            center: [r.center[0] * scale, r.center[1] * scale, r.center[2] * scale],
            radius: r.radius,
            count: r.count,
            inhibitory: r.inhibitory,
            secrete: r.secrete.clone(),
            axons: r
                .axons
                .iter()
                .map(|a| grow::AxonSpec {
                    target: a.target.clone(),
                    max_len: a.max_len * scale,
                })
                .collect(),
        })
        .collect();
    Layout {
        regions,
        allowed_pairs: layout.allowed_pairs.clone(),
    }
}

/// Прикинуть размер сетки мира (как в build_world): span + margin, cell = 5.
fn grid_ncells(layout: &Layout) -> usize {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for r in &layout.regions {
        for k in 0..3 {
            lo[k] = lo[k].min(r.center[k] - r.radius);
            hi[k] = hi[k].max(r.center[k] + r.radius);
        }
    }
    let span = ((hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]) + 140.0).max(50.0);
    (span / 5.0).ceil() as usize
}

fn run_case(name: &str, layout: &Layout, st: &Steering, gpu: &DiffusionBackend) {
    // CPU (rayon, f64).
    let t0 = Instant::now();
    let c = grow::grow(layout, st, 777);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

    // GPU (тот же бэкенд-адаптер, клонированный handle).
    let t0 = Instant::now();
    let g = grow::grow_with_backend(layout, st, 777, gpu.clone());
    let gpu_ms = t0.elapsed().as_secs_f64() * 1e3;

    let pc = pair_counts(&c.edges, &c.region_of);
    let pg = pair_counts(&g.edges, &g.region_of);
    let union: BTreeSet<_> = pc.keys().chain(pg.keys()).cloned().collect();
    let common = union
        .iter()
        .filter(|k| pc.get(k).copied().unwrap_or(0) > 0 && pg.get(k).copied().unwrap_or(0) > 0)
        .count();
    let nc = grid_ncells(layout);

    println!(
        "== {name}: {} нейронов, сетка ~{nc}³ ({}M клеток), {} шагов",
        layout.n(),
        nc * nc * nc / 1_000_000,
        st.grow_steps
    );
    println!(
        "   CPU: {cpu_ms:8.1} мс, рёбер {:5}, пар {:2} | GPU: {gpu_ms:8.1} мс, рёбер {:5}, пар {:2} | GPU/CPU {:.2}×, общих пар {}/{}",
        c.edges.len(),
        pc.len(),
        g.edges.len(),
        pg.len(),
        gpu_ms / cpu_ms,
        common,
        union.len()
    );
}

fn main() {
    let st = Steering::default();
    let st_short = Steering { grow_steps: 200, ..st };
    let gpu = pollster::block_on(DiffusionBackend::new());
    let is_gpu = matches!(gpu, DiffusionBackend::Gpu(_));
    println!(
        "диффузионный бэкенд: {}",
        if is_gpu {
            "GPU (wgpu)"
        } else {
            "CPU (fallback — адаптер не найден)"
        }
    );

    // Прогрев GPU-пути: компиляция шейдеров, создание persistent-буферов.
    grow::grow_with_backend(&grow::avoidance_layout(), &st_short, 1, gpu.clone());

    run_case("avoidance", &grow::avoidance_layout(), &st_short, &gpu);
    run_case("cognitive", &grow::cognitive_layout(), &st_short, &gpu);
    run_case("avoidance×2", &scaled(&grow::avoidance_layout(), 2.0), &st_short, &gpu);
    run_case("avoidance×3", &scaled(&grow::avoidance_layout(), 3.0), &st_short, &gpu);

    // Санити: существо целиком на GPU — коннектом вырос на GPU, мозг на GPU.
    let mut c = Creature::new_gpu_grown(7);
    let sense = Sense {
        smell_l: 0.6,
        smell_r: 0.3,
        wall_l: 0.0,
        wall_r: 0.0,
        food_near: true,
        wall: false,
    };
    let mut steer_sum = 0.0f32;
    let mut drive_sum = 0.0f32;
    for _ in 0..100 {
        let a = c.tick(&sense);
        steer_sum += a.steer;
        drive_sum += a.drive;
    }
    println!(
        "санити: существо на GPU (мозг on_gpu={}, рёбер {}), 100 тиков → Σsteer={steer_sum:.2} Σdrive={drive_sum:.2}",
        c.brain.on_gpu(),
        c.brain.n_syn()
    );
}
