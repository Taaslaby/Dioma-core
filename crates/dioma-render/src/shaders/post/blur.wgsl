// dioma-render — пост-проход «размытие» (9-точечный box blur).
//
// Полноэкранный квад, uniform и сэмплер сцены — в post/post_common.wgsl
// (приклеивается в shaders.rs, см. `shaders::post_blur()`); здесь только
// фрагмент. Сэмплирует сцену 9 точками вокруг фрагмента; radius 0 →
// пассивное копирование.
//
// Uniform: vec4(u_time, vignette_strength, grain_strength, blur_radius) —
// используется только blur_radius (в пикселях).

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(1.0) / vec2<f32>(textureDimensions(scene_tex));
    let r = post.p.w * texel;
    var sum: vec4<f32> = vec4<f32>(0.0);
    var offs = array<vec2<f32>, 9>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0,  0.0), vec2<f32>(0.0,  0.0), vec2<f32>(1.0,  0.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>(0.0,  1.0), vec2<f32>(1.0,  1.0),
    );
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        sum = sum + textureSample(scene_tex, scene_sampler, in.uv + offs[i] * r);
    }
    return sum / 9.0;
}
