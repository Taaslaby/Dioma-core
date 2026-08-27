//! Временный диагностический тест: 6 потоков, каждый поднимает GPU и создаёт
//! атлас с одним спрайтом (add_rgba, как в atlas_grows_on_overflow).

mod common;

use common::gpu;
use dioma_render::SpriteAtlas;

fn probe() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 64);
    let opaque = vec![255u8; 16 * 16 * 4];
    atlas.add_rgba(&gpu.queue, "s", 16, 16, &opaque).expect("ok");
    assert_eq!(atlas.len(), 1);
}

#[test]
fn probe_a() {
    probe();
}

#[test]
fn probe_b() {
    probe();
}

#[test]
fn probe_c() {
    probe();
}

#[test]
fn probe_d() {
    probe();
}

#[test]
fn probe_e() {
    probe();
}

#[test]
fn probe_f() {
    probe();
}
