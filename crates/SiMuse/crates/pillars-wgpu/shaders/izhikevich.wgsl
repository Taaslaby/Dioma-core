// izhikevich.wgsl — WebGPU compute for IzhikevichModel (zoo/neurons/izhikevich.rs)
// SoA: v, u, trace, refr, a, b, c, d are param columns; state v/u

struct Args { n: u32, stride: u32, dt: f32, _pad: f32 };

@group(0) @binding(0) var<storage, read_write> states: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read> i_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(4) var<uniform> args: Args;

@compute @workgroup_size(64)
fn izhikevich_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= args.n) { return; }
    let s = args.stride;
    var v = states[0u * s + i];
    var u = states[1u * s + i];
    let a = params[2u * s + i];
    let b = params[3u * s + i];
    // simplified Euler — полная версия с c/d/reset как в lif.wgsl
    let i_syn = i_in[i];
    let dt = args.dt;
    // dv = 0.04*v*v + 5*v + 140 - u + I
    let dv = (0.04 * v * v + 5.0 * v + 140.0 - u + i_syn) * dt;
    v = v + dv;
    u = u + a * (b * v - u) * dt;
    var sp: u32 = 0u;
    if (v >= 30.0) {
        // c, d in params[4], params[5]
        let c = params[4u * s + i];
        let d = params[5u * s + i];
        v = c;
        u = u + d;
        sp = 1u;
    }
    states[0u * s + i] = v;
    states[1u * s + i] = u;
    spikes[i] = sp;
}
