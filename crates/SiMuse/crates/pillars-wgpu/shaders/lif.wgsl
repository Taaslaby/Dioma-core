// lif.wgsl — WebGPU compute for LIFModel (port of zoo/neurons/lif.rs:42)
// SoA layout: states[col*stride + i], params[col*stride + i], stride = n.div_ceil(64)*64

struct Args {
    n: u32,
    stride: u32,
    dt: f32,
};

@group(0) @binding(0) var<storage, read_write> states: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> i_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(4) var<uniform> args: Args;

@compute @workgroup_size(64)
fn lif_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= args.n) { return; }
    let stride = args.stride;
    // SoA columns: state v = 0*stride + i, refr = 1*stride + i
    // param dt = 0, inv_tau_m = 1, v_rest = 2, v_thresh = 3, v_reset = 4, tau_ref = 5
    var v = states[0u * stride + i];
    var refr = states[1u * stride + i];
    let dt = params[0u * stride + i];
    let inv_tau_m = params[1u * stride + i];
    let v_rest = params[2u * stride + i];
    let v_thresh = params[3u * stride + i];
    let v_reset = params[4u * stride + i];
    let tau_ref = params[5u * stride + i];
    let i_syn = i_in[i];

    let is_refr = refr > 0.0;
    var v_new = v;
    if (!is_refr) {
        let dv = (v_rest - v + i_syn) * inv_tau_m * dt;
        v_new = v + dv;
    }
    var spike: u32 = 0u;
    if (!is_refr && v_new >= v_thresh) {
        spike = 1u;
        v_new = v_reset;
    }
    // refractory timer
    var refr_new = refr - dt;
    if (spike == 1u) {
        refr_new = tau_ref;
    }
    if (refr_new < 0.0) { refr_new = 0.0; }

    states[0u * stride + i] = select(v_new, v, is_refr && spike == 0u);
    states[1u * stride + i] = refr_new;
    spikes[i] = spike;
}
