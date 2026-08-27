// adex.wgsl — AdExModel (zoo/neurons/adex.rs) — bit-identical to CPU fast_exp/vtrap
// stub — полный порт как lif.wgsl, c exp(-dt/tau) через fast_exp_poly
@group(0) @binding(0) var<storage, read_write> states: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> i_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> spikes: array<u32>;
struct Args { n: u32, stride: u32, dt: f32, _pad: f32 };
@group(0) @binding(4) var<uniform> args: Args;
@compute @workgroup_size(64)
fn adex_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= args.n) { return; }
    // TODO: port adex step exactly (vtrap, exp) — same as pillars/src/math/vtrap.rs
    spikes[i] = 0u;
}
