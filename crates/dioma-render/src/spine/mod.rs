//! Минимальная поддержка Spine 2D (формат 3.8) — скелетная анимация поверх
//! мешевого пайплайна ([`crate::MeshBatch`]).
//!
//! ⚠️ Лицензия/происхождение: набело написанная реализация чтения открытого
//! формата Spine JSON/Atlas (проверена на экспорте 3.8.75), не производная от
//! официальных рантаймов. Для продуктового использования Spine требуется
//! лицензия Esoteric Software.
//!
//! Поддержано: регион-attachment'ы (квады), иерархия костей с аффинным
//! наследованием, таймлайны rotate/translate/scale (linear/stepped/bezier),
//! цикл анимации. НЕ поддержано: mesh/weighted attachment'ы, скины кроме
//! default, констрейнты, бинарный .skel.
//!
//! ```text
//! let data = SpineData::load(&device, &queue, atlas_text, png, json)?;
//! let mut inst = data.instance("animation");
//! // кадр:
//! inst.update(dt);
//! mesh_batch.push(&inst.build_mesh(&data));   // renderer.render_meshes(..)
//! ```

use std::collections::HashMap;

use serde_json::Value;

use crate::scene::atlas::SpriteAtlas;
use crate::scene::mesh::{Mesh, MeshVertex};

// ── Atlas (.atlas) ──────────────────────────────────────────────────────────

/// Регион страницы атласа (px страницы, origin — верх-лево).
#[derive(Clone, Debug)]
pub struct AtlasRegion {
    pub name: String,
    pub x: u32,
    pub y: u32,
    /// Размер области В СТРАНИЦЕ (у повернутых — уже свапнут относительно orig).
    pub w: u32,
    pub h: u32,
    /// Регион упакован с поворотом на 90°.
    pub rotate: bool,
}

/// Разбирает текст `.atlas` (первая страница).
pub fn parse_atlas(text: &str) -> Result<(u32, u32, Vec<AtlasRegion>), String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    lines.next().ok_or("atlas: пуст")?; // имя PNG
    let mut size = (0u32, 0u32);
    let mut regions: Vec<AtlasRegion> = Vec::new();
    let mut current: Option<AtlasRegion> = None;
    for line in lines {
        if !line.contains(':') {
            if let Some(c) = current.take() {
                regions.push(c);
            }
            current = Some(AtlasRegion {
                name: line.to_string(),
                x: 0,
                y: 0,
                w: 0,
                h: 0,
                rotate: false,
            });
            continue;
        }
        let (key, val) = line.split_once(':').unwrap_or(("", ""));
        let vals: Vec<u32> = val.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        // «size» ДО первого имени региона — размер страницы.
        if key.trim() == "size" && vals.len() == 2 && current.is_none() {
            size = (vals[0], vals[1]);
            continue;
        }
        let Some(cur) = current.as_mut() else { continue };
        match (key.trim(), vals.len()) {
            ("rotate", _) => cur.rotate = val.trim() == "true",
            ("xy", 2) => {
                cur.x = vals[0];
                cur.y = vals[1];
            }
            ("size", 2) => {
                cur.w = vals[0];
                cur.h = vals[1];
            }
            _ => {}
        }
    }
    if let Some(c) = current.take() {
        regions.push(c);
    }
    if size.0 == 0 {
        return Err("atlas: нет size страницы".into());
    }
    Ok((size.0, size.1, regions))
}

// ── Скелет (JSON 3.8) ───────────────────────────────────────────────────────

/// Setup-описание кости.
#[derive(Clone, Debug)]
pub struct BoneDef {
    pub name: String,
    pub parent: Option<usize>,
    pub rotation: f32,
    pub x: f32,
    pub y: f32,
    pub scale_x: f32,
    pub scale_y: f32,
}

/// Регион-attachment: квад с локальным трансформом на кости слота.
#[derive(Clone, Copy, Debug)]
pub struct RegionAttachment {
    pub x: f32,
    pub y: f32,
    pub rotation: f32,
    pub width: f32,
    pub height: f32,
}

/// Слот: кость + attachment по умолчанию (порядок массива = draw order).
#[derive(Clone, Debug)]
pub struct SlotDef {
    pub name: String,
    pub bone: usize,
    pub attachment: String,
}

/// Ключ таймлайна с кривой интерполяции.
#[derive(Clone, Copy, Debug)]
pub struct Key {
    pub time: f32,
    pub value: f32,
    pub value2: f32,
    pub curve: Curve,
}

#[derive(Clone, Copy, Debug)]
pub enum Curve {
    Linear,
    Stepped,
    /// Безье (x1, y1, x2, y2) по времени.
    Bezier([f32; 4]),
}

/// Таймлайн кости: дельты к setup-позе (rotate — аддитив к углу).
#[derive(Clone, Debug, Default)]
pub struct BoneTimeline {
    pub rotate: Vec<Key>,
    pub translate: Vec<Key>,
    pub scale: Vec<Key>,
}

impl BoneTimeline {
    fn duration(&self) -> f32 {
        self.rotate
            .last()
            .or(self.translate.last())
            .or(self.scale.last())
            .map(|k| k.time)
            .unwrap_or(0.0)
    }
}

/// Анимация: таймлайны по индексам костей.
#[derive(Clone, Debug, Default)]
pub struct Animation {
    pub timelines: HashMap<usize, BoneTimeline>,
}

impl Animation {
    pub fn duration(&self) -> f32 {
        self.timelines.values().map(BoneTimeline::duration).fold(0.0, f32::max)
    }
}

/// Разобранный скелет.
#[derive(Clone, Debug)]
pub struct SpineSkeleton {
    pub bones: Vec<BoneDef>,
    pub slots: Vec<SlotDef>,
    /// Имя слота → регион-attachment (default-скин).
    pub attachments: HashMap<String, RegionAttachment>,
    pub animations: HashMap<String, Animation>,
    /// Центрирующий сдвиг из границ JSON (skeleton.x/y/width/height).
    pub offset: [f32; 2],
}

impl SpineSkeleton {
    pub fn from_str(json: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(json).map_err(|e| format!("skeleton JSON: {e}"))?;
        let e = |m: &str| format!("skeleton: {m}");

        let mut bones: Vec<BoneDef> = Vec::new();
        for b in v["bones"].as_array().ok_or_else(|| e("нет bones"))? {
            let name = b["name"].as_str().ok_or_else(|| e("кость без name"))?.to_string();
            // Родитель ищется среди уже добавленных (JSON-порядок — родители раньше).
            let parent = b["parent"]
                .as_str()
                .and_then(|p| bones.iter().position(|b| b.name == p));
            bones.push(BoneDef {
                name,
                parent,
                rotation: b["rotation"].as_f64().unwrap_or(0.0) as f32,
                x: b["x"].as_f64().unwrap_or(0.0) as f32,
                y: b["y"].as_f64().unwrap_or(0.0) as f32,
                scale_x: b["scaleX"].as_f64().unwrap_or(1.0) as f32,
                scale_y: b["scaleY"].as_f64().unwrap_or(1.0) as f32,
            });
        }

        let mut slots: Vec<SlotDef> = Vec::new();
        let mut attachments = HashMap::new();
        for s in v["slots"].as_array().ok_or_else(|| e("нет slots"))? {
            let name = s["name"].as_str().ok_or_else(|| e("слот без name"))?;
            let bone_name = s["bone"].as_str().ok_or_else(|| e("слот без bone"))?;
            let bone = bones
                .iter()
                .position(|b| b.name == bone_name)
                .ok_or_else(|| e("слот: кость не найдена"))?;
            let att = s["attachment"].as_str().unwrap_or("").to_string();
            slots.push(SlotDef { name: name.to_string(), bone, attachment: att.clone() });
        }
        if let Some(skin) = v["skins"][0]["attachments"].as_object() {
            for (slot_name, atts) in skin {
                if let Some((_, def)) = atts.as_object().and_then(|o| o.iter().next()) {
                    attachments.insert(
                        slot_name.clone(),
                        RegionAttachment {
                            x: def["x"].as_f64().unwrap_or(0.0) as f32,
                            y: def["y"].as_f64().unwrap_or(0.0) as f32,
                            rotation: def["rotation"].as_f64().unwrap_or(0.0) as f32,
                            width: def["width"].as_f64().unwrap_or(0.0) as f32,
                            height: def["height"].as_f64().unwrap_or(0.0) as f32,
                        },
                    );
                }
            }
        }

        let mut animations = HashMap::new();
        if let Some(anims) = v["animations"].as_object() {
            for (anim_name, anim) in anims {
                let mut timelines = HashMap::new();
                if let Some(bones_tracks) = anim["bones"].as_object() {
                    for (bone_name, tracks) in bones_tracks {
                        if let Some(idx) = bones.iter().position(|b| &b.name == bone_name) {
                            timelines.insert(
                                idx,
                                BoneTimeline {
                                    rotate: read_track1(&tracks["rotate"]),
                                    translate: read_track2(&tracks["translate"]),
                                    scale: read_track2(&tracks["scale"]),
                                },
                            );
                        }
                    }
                }
                animations.insert(anim_name.clone(), Animation { timelines });
            }
        }

        let ox = v["skeleton"]["x"].as_f64().unwrap_or(0.0) as f32;
        let oy = v["skeleton"]["y"].as_f64().unwrap_or(0.0) as f32;
        let w = v["skeleton"]["width"].as_f64().unwrap_or(0.0) as f32;
        let h = v["skeleton"]["height"].as_f64().unwrap_or(0.0) as f32;
        let offset = [-ox - w / 2.0, -(oy + h / 2.0)];

        Ok(Self { bones, slots, attachments, animations, offset })
    }
}

fn read_track1(v: &Value) -> Vec<Key> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .map(|k| Key {
                    time: k["time"].as_f64().unwrap_or(0.0) as f32,
                    value: k["angle"].as_f64().unwrap_or(0.0) as f32,
                    value2: 0.0,
                    curve: read_curve(k),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn read_track2(v: &Value) -> Vec<Key> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .map(|k| Key {
                    time: k["time"].as_f64().unwrap_or(0.0) as f32,
                    value: k["x"].as_f64().unwrap_or(0.0) as f32,
                    value2: k["y"].as_f64().unwrap_or(0.0) as f32,
                    curve: read_curve(k),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn read_curve(k: &Value) -> Curve {
    match &k["curve"] {
        Value::String(s) if s == "stepped" => Curve::Stepped,
        Value::Array(a) if a.len() == 4 => Curve::Bezier([
            a[0].as_f64().unwrap_or(0.0) as f32,
            a[1].as_f64().unwrap_or(0.0) as f32,
            a[2].as_f64().unwrap_or(1.0) as f32,
            a[3].as_f64().unwrap_or(1.0) as f32,
        ]),
        Value::Number(_) => {
            // Сокращение Spine: curve = x1, c3 = x3 → безье (x1, 0, x3, 1).
            let x1 = k["curve"].as_f64().unwrap_or(0.0) as f32;
            let x3 = k["c3"].as_f64().unwrap_or(1.0) as f32;
            Curve::Bezier([x1, 0.0, x3, 1.0])
        }
        _ => Curve::Linear,
    }
}

fn ease(curve: Curve, t: f32) -> f32 {
    match curve {
        Curve::Linear => t,
        Curve::Stepped => 0.0,
        Curve::Bezier([x1, y1, x2, y2]) => {
            // Безье по времени: ищем u, где bezier_x(u) = t; 10 итераций ~1e-3.
            let at = |u: f32| -> (f32, f32) {
                let om = 1.0 - u;
                (
                    3.0 * om * om * u * x1 + 3.0 * om * u * u * x2 + u * u * u,
                    3.0 * om * om * u * y1 + 3.0 * om * u * u * y2 + u * u * u,
                )
            };
            let (mut lo, mut hi) = (0.0f32, 1.0f32);
            let mut u = t;
            for _ in 0..10 {
                u = (lo + hi) * 0.5;
                if at(u).0 < t {
                    lo = u;
                } else {
                    hi = u;
                }
            }
            at(u).1.clamp(0.0, 1.0)
        }
    }
}

/// Сэмпл трека в момент `time`: (value, value2) на сегменте a→next.
fn sample(keys: &[Key], time: f32, out: &mut (f32, f32)) {
    let Some(first) = keys.first() else { return };
    let last = *keys.last().unwrap();
    if time <= first.time || keys.len() == 1 {
        *out = (first.value, first.value2);
        return;
    }
    if time >= last.time {
        *out = (last.value, last.value2);
        return;
    }
    // Сегмент [i-1, i): первый ключ с time > текущего.
    let i = keys.partition_point(|k| k.time <= time);
    let (a, b) = (keys[i - 1], keys[i]);
    let t01 = ((time - a.time) / (b.time - a.time).max(1e-6)).clamp(0.0, 1.0);
    let t = ease(b.curve, t01);
    out.0 = a.value + (b.value - a.value) * t;
    out.1 = a.value2 + (b.value2 - a.value2) * t;
}

// ── Рантайм ─────────────────────────────────────────────────────────────────

/// Аффин `[a, b, c, d, tx, ty]`: точка' = (a·x + c·y + tx, b·x + d·y + ty).
type Affine = [f32; 6];

fn compose(p: &Affine, l: &Affine) -> Affine {
    [
        p[0] * l[0] + p[2] * l[1],
        p[1] * l[0] + p[3] * l[1],
        p[0] * l[2] + p[2] * l[3],
        p[1] * l[2] + p[3] * l[3],
        p[0] * l[4] + p[2] * l[5] + p[4],
        p[1] * l[4] + p[3] * l[5] + p[5],
    ]
}

fn local_affine(rotation: f32, x: f32, y: f32, sx: f32, sy: f32) -> Affine {
    let (c, s) = (rotation.to_radians().cos(), rotation.to_radians().sin());
    [c * sx, s * sx, -s * sy, c * sy, x, y]
}

/// Загруженные данные: атлас (текстура+регионы) + скелет + UV-углы.
pub struct SpineData {
    pub atlas: SpriteAtlas,
    pub skeleton: SpineSkeleton,
    /// Имя attachment → UV-углы квада [TL, TR, BL, BR].
    corner_uvs: HashMap<String, [(f32, f32); 4]>,
}

impl SpineData {
    /// Загружает атлас и скелет (тексты файлов + байты PNG страницы).
    pub fn load(
        device: &std::sync::Arc<wgpu::Device>,
        queue: &wgpu::Queue,
        atlas_text: &str,
        png_bytes: &[u8],
        skeleton_json: &str,
    ) -> Result<Self, String> {
        let (page_w, page_h, regions) = parse_atlas(atlas_text)?;
        let img = image::load_from_memory(png_bytes)
            .map_err(|e| format!("spine PNG: {e}"))?
            .to_rgba8();
        if img.dimensions() != (page_w, page_h) {
            return Err(format!(
                "spine: PNG {}×{} ≠ атлас {}×{}",
                img.dimensions().0,
                img.dimensions().1,
                page_w,
                page_h
            ));
        }

        let mut atlas = SpriteAtlas::new(device, page_w.max(page_h).max(256).next_power_of_two());
        atlas
            .replace_rgba(
                queue,
                "spine_page",
                page_w,
                page_h,
                &img.into_raw(),
            )
            .map_err(|e| format!("spine: страница не влезла: {e}"))?;
        // Страница одна — регистрируем рег.ы как суб-регионы имени spine_page
        // нельзя (add/insert по другим именам работает: insert_region напрямую).
        let mut corner_uvs = HashMap::new();
        for r in &regions {
            let (u0, v0) = (r.x as f32 / page_w as f32, r.y as f32 / page_h as f32);
            let uvs = if r.rotate {
                // Пак повернут на 90°: спрайт верх-лево → пак правый-верх.
                let (u1, v1) = ((r.x + r.h) as f32 / page_w as f32, (r.y + r.w) as f32 / page_h as f32);
                [(u1, v0), (u0, v0), (u1, v1), (u0, v1)]
            } else {
                let (u1, v1) = ((r.x + r.w) as f32 / page_w as f32, (r.y + r.h) as f32 / page_h as f32);
                [(u0, v0), (u1, v0), (u0, v1), (u1, v1)]
            };
            corner_uvs.insert(r.name.clone(), uvs);
            let rect = if r.rotate {
                let (u1, v1) =
                    ((r.x + r.h) as f32 / page_w as f32, (r.y + r.w) as f32 / page_h as f32);
                [u0, v0, u1, v1]
            } else {
                let (u1, v1) =
                    ((r.x + r.w) as f32 / page_w as f32, (r.y + r.h) as f32 / page_h as f32);
                [u0, v0, u1, v1]
            };
            atlas.insert_region(&r.name, rect);
        }

        let skeleton = SpineSkeleton::from_str(skeleton_json)?;
        Ok(Self { atlas, skeleton, corner_uvs })
    }

    /// Создаёт проигрыватель анимации.
    pub fn instance(&self, animation: &str) -> SpineInstance {
        SpineInstance {
            skeleton: self.skeleton.clone(),
            corner_uvs: self.corner_uvs.clone(),
            animation: animation.to_string(),
            time: 0.0,
            offset: self.skeleton.offset,
            world: vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; self.skeleton.bones.len()],
        }
    }
}

/// Проигрыватель: скелет + время + текущая анимация.
#[derive(Clone)]
pub struct SpineInstance {
    pub skeleton: SpineSkeleton,
    corner_uvs: HashMap<String, [(f32, f32); 4]>,
    pub animation: String,
    pub time: f32,
    /// Экранный сдвиг модели (по умолчанию — центр границ в origin).
    pub offset: [f32; 2],
    world: Vec<Affine>,
}

impl SpineInstance {
    /// Длительность текущей анимации (сек).
    pub fn duration(&self) -> f32 {
        self.skeleton.animations.get(&self.animation).map(Animation::duration).unwrap_or(0.0)
    }

    /// Продвигает время (цикл) и позирует кости.
    pub fn update(&mut self, dt: f32) {
        let dur = self.duration();
        if dur > 0.0 {
            self.time = (self.time + dt) % dur;
        }
        self.pose();
    }

    fn pose(&mut self) {
        let anim = self
            .skeleton
            .animations
            .get(&self.animation)
            .cloned()
            .unwrap_or_default();
        self.world.clear();
        for (i, bone) in self.skeleton.bones.iter().enumerate() {
            let mut rot = bone.rotation;
            let mut xy = (bone.x, bone.y);
            let mut sc = (bone.scale_x, bone.scale_y);
            if let Some(tl) = anim.timelines.get(&i) {
                let mut out = (0.0, 0.0);
                sample(&tl.rotate, self.time, &mut out);
                rot += out.0;
                sample(&tl.translate, self.time, &mut out);
                xy = (bone.x + out.0, bone.y + out.1);
                sample(&tl.scale, self.time, &mut out);
                sc = (bone.scale_x * out.0, bone.scale_y * out.1);
            }
            let local = local_affine(rot, xy.0, xy.1, sc.0, sc.1);
            let world = match bone.parent {
                Some(p) => compose(&self.world[p], &local),
                None => local,
            };
            self.world.push(world);
        }
    }

    /// Квады слотов в draw order — готовый меш для `render_meshes`.
    pub fn build_mesh(&self) -> Mesh {
        let mut mesh = Mesh { vertices: Vec::new(), indices: Vec::new() };
        for slot in &self.skeleton.slots {
            let Some(att) = self.skeleton.attachments.get(&slot.attachment) else {
                continue;
            };
            let Some(uvs) = self.corner_uvs.get(&slot.attachment) else {
                continue;
            };
            let bone = self.world[slot.bone];
            let att_local = local_affine(att.rotation, att.x, att.y, 1.0, 1.0);
            let m = compose(&bone, &att_local);
            let (hw, hh) = (att.width / 2.0, att.height / 2.0);
            // Углы: TL, TR, BL, BR (x — вправо, y — вверх) + UV каждого.
            let corners = [
                (-hw, hh, uvs[0]),
                (hw, hh, uvs[1]),
                (-hw, -hh, uvs[2]),
                (hw, -hh, uvs[3]),
            ];
            let base = mesh.vertices.len() as u32;
            for (cx, cy, (u, v)) in corners {
                mesh.vertices.push(MeshVertex {
                    position: [
                        m[0] * cx + m[2] * cy + m[4] + self.offset[0],
                        m[1] * cx + m[3] * cy + m[5] + self.offset[1],
                    ],
                    uv: [u, v],
                    color: [1.0, 1.0, 1.0, 1.0],
                });
            }
            mesh.indices.extend([base, base + 1, base + 2, base + 2, base + 1, base + 3]);
        }
        mesh
    }
}
