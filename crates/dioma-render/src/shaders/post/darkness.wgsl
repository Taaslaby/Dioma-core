// dioma-render — пост-проход «тьма/ночь»: умножение сцены на цвет.
//
// Ночное небо (тёмно-синий tint) затемняет мир, сохраняя оттенки. Вместе с
// аддитивными источниками света ([`crate::Renderer::render_lights`]) даёт
// цикл дня и ночи: свет рисуется ПОВЕРХ тьмы и «пробивает» её собой.
// Полноэкранный квад, uniform и сэмплер сцены — в post/post_common.wgsl
// (приклеивается в shaders.rs, см. `shaders::post_darkness()`).
//
// Uniform: vec4(tint.r, tint.g, tint.b, strength).
//   tint.rgb    — множитель цвета сцены (1,1,1 = без изменений);
//   strength    — сила эффекта 0..1 (смешение с исходной сценой);
//                 0 → проход вырождается в пассивное копирование.

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(scene_tex, scene_sampler, in.uv);
    let darkened = c.rgb * post.p.rgb;
    return vec4<f32>(mix(c.rgb, darkened, post.p.a), c.a);
}
