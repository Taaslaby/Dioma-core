// dioma-render — пост-проход «виньетка + плёночное зерно».
//
// Полноэкранный квад, uniform и сэмплер сцены — в post/post_common.wgsl
// (приклеивается в shaders.rs, см. `shaders::post_vignette_grain()`); здесь
// только фрагмент. Vignette 0.0 → проход вырождается в пассивное копирование.
//
// Uniform: vec4(u_time, vignette_strength, grain_strength, 0).

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(scene_tex, scene_sampler, in.uv);

    // Виньетка: плавное затемнение от центра (d=0) к углам (d≈0.707).
    let d = distance(in.uv, vec2<f32>(0.5));
    let vig = 1.0 - post.p.y * smoothstep(0.2, 0.72, d);

    // Зерно: шум по пикселю (встроенная позиция фрагмента), плывущий со временем.
    let seed = vec2<u32>(floor(in.clip_pos.xy + vec2<f32>(post.p.x * 13.0)));
    let g = (hash01_post(seed) - 0.5) * post.p.z;

    return vec4<f32>(c.rgb * vig + vec3<f32>(g), c.a);
}

// Детерминированный псевдослучайный шум по ячейке (для зерна). Пост-проходы
// не получают общий пролог мировых шейдеров, поэтому хэш локален.
fn hash01_post(v: vec2<u32>) -> f32 {
    var h: u32 = v.x * 1664525u + v.y * 1013904223u + 374761393u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= h >> 16u;
    return f32(h & 0xFFFFFFu) / 16777216.0;
}
