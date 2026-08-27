//! Кто фьюзит a*b+c в fma — rust-CPU или GPU-шейдер?
fn main() {
    let g_old: f32 = 37.563832759; // реконструкция из шага 8
    let syn: f32 = 1.0 - 0.5 / 3.0;
    let delayed: f32 = 45.078308105;
    let plain = g_old * syn + delayed;
    let fused = g_old.mul_add(syn, delayed);
    let manual_v: f32 = -48.9153;
    println!("plain = {plain:.9}");
    println!("fma   = {fused:.9}");
    println!("diff  = {:.9}", (plain - fused).abs());
    println!("CPU g = 76.382690430");
    println!("GPU g = 76.382682800");
    println!("v  = {manual_v}");
    // v: v_new = v + tf*(g_old - (v - v_rest)) — тоже подозрение
    let v: f32 = -49.5; let v_rest: f32 = -52.0; let tf: f32 = 0.025;
    let integrand = g_old - (v - v_rest);
    let vp = v + tf * integrand;
    let vf = v.mul_add(tf, tf * integrand - tf * (v - v_rest)).max(-1e9);
    println!("v plain={vp:.9}");
    println!("v fma-alt={vf:.9}");
}
