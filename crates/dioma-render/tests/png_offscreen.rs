//! Тесты PNG-загрузки и авто-роста атласа: roundtrip save→load, обрезка
//! прозрачных полей, рост при переполнении, пакетная загрузка папки, ошибки
//! ввода (нет файла / мусорные байты).
//!
//! Файлы-фикстуры кладутся в `std::env::temp_dir()` с уникальными именами
//! (`dioma_render_test_` + имя теста) и удаляются в конце best-effort.

mod common;

use std::env::temp_dir;
use std::fs;
use std::path::PathBuf;

use common::gpu;
use dioma_render::atlas::save_png;
use dioma_render::SpriteAtlas;

/// Уникальный путь в temp_dir для фикстуры теста `test`.
fn temp_path(test: &str) -> PathBuf {
    temp_dir().join(format!("dioma_render_test_{test}_{}", std::process::id()))
}

/// Кодирует RGBA8-буфер в PNG (через `save_png`) и возвращает байты файла.
/// Файл читается и сразу удаляется; имя уникально на тест (`tag`).
fn png_bytes(tag: &str, rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    let path = temp_path(tag);
    save_png(path.to_str().unwrap(), w, h, rgba).expect("save_png ok");
    let bytes = fs::read(&path).expect("прочитать PNG с диска");
    let _ = fs::remove_file(&path);
    bytes
}

// ── 1. Roundtrip: save_png → add_png / add_png_bytes ────────────────────────

#[test]
fn png_save_load_roundtrip() {
    let Some(gpu) = gpu() else { return };
    // Полностью непрозрачный 8×8: красный блок 4×4 в углу, остальное белое —
    // обрезка не нужна (всё непрозрачно), roundtrip проверяет целостность.
    let mut rgba = vec![255u8; 8 * 8 * 4];
    for y in 0..4 {
        for x in 0..4 {
            let i = (y * 8 + x) * 4;
            rgba[i..i + 4].copy_from_slice(&[255, 0, 0, 255]);
        }
    }
    let path = temp_path("png_save_load_roundtrip");
    save_png(path.to_str().unwrap(), 8, 8, &rgba).expect("save_png ok");

    let mut atlas = SpriteAtlas::new(&gpu.device, 64);
    atlas
        .add_png(&gpu.queue, "saved", path.to_str().unwrap())
        .expect("add_png из файла");
    let info = atlas.region_info("saved").expect("регион 'saved' есть");
    assert_eq!(info.size_px, [8, 8], "непрозрачный PNG не обрезается");
    assert_eq!(info.offset, [0.0, 0.0], "без обрезки смещение (0,0)");
    let [u0, v0, u1, v1] = info.uv;
    assert!((0.0..=1.0).contains(&u0) && u0 < u1 && u1 <= 1.0, "uv по X: {u0}..{u1}");
    assert!((0.0..=1.0).contains(&v0) && v0 < v1 && v1 <= 1.0, "uv по Y: {v0}..{v1}");

    // Те же байты — через add_png_bytes: результат идентичен.
    let bytes = fs::read(&path).expect("прочитать PNG с диска");
    let mut atlas2 = SpriteAtlas::new(&gpu.device, 64);
    atlas2
        .add_png_bytes(&gpu.queue, "saved", &bytes)
        .expect("add_png_bytes из буфера");
    assert_eq!(atlas2.region_info("saved").unwrap().size_px, [8, 8]);
    assert_eq!(
        atlas2.region("saved").unwrap(),
        atlas.region("saved").unwrap(),
        "одинаковые байты → одинаковые UV"
    );

    let _ = fs::remove_file(&path);
}

// ── 2. Обрезка прозрачной рамки ─────────────────────────────────────────────

#[test]
fn png_trim_removes_transparent_border() {
    let Some(gpu) = gpu() else { return };
    // 16×16: прозрачная рамка 3px, внутри 10×10 непрозрачный зелёный.
    let mut rgba = vec![0u8; 16 * 16 * 4];
    for y in 3..13 {
        for x in 3..13 {
            let i = (y * 16 + x) * 4;
            rgba[i..i + 4].copy_from_slice(&[0, 200, 0, 255]);
        }
    }
    let bytes = png_bytes("png_trim_removes_transparent_border", &rgba, 16, 16);

    let mut atlas = SpriteAtlas::new(&gpu.device, 64);
    atlas.add_png_bytes(&gpu.queue, "green", &bytes).expect("add_png_bytes ok");
    let info = atlas.region_info("green").expect("регион 'green' есть");
    assert_eq!(info.size_px, [10, 10], "прозрачная рамка обрезается");
    assert_eq!(info.offset, [3.0, 3.0], "смещение = левый-верхний угол рамки");
}

// ── 3. Авто-рост атласа при переполнении ───────────────────────────────────

#[test]
fn atlas_grows_on_overflow() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 32);
    let opaque = vec![255u8; 16 * 16 * 4];
    for i in 0..4 {
        atlas.add_rgba(&gpu.queue, &format!("s{i}"), 16, 16, &opaque).expect("первые 4 влезают в 32²");
    }
    assert_eq!(atlas.len(), 4);
    assert_eq!(atlas.size(), (32, 32));

    // Пятый 16×16 в атлас 32² не влезает → авто-рост: сторона удваивается.
    atlas.add_rgba(&gpu.queue, "s4", 16, 16, &opaque).expect("пятый: авто-рост");
    assert_eq!(atlas.len(), 5);
    assert_eq!(atlas.width(), 64);
    assert_eq!(atlas.height(), 64);
    assert_eq!(atlas.size(), (64, 64));

    // Все 5 регионов на месте и занимают разные UV-прямоугольники.
    let mut uvs: Vec<[f32; 4]> = (0..5).map(|i| atlas.region(&format!("s{i}")).unwrap()).collect();
    uvs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    uvs.dedup();
    assert_eq!(uvs.len(), 5, "у всех регионов разные UV");

    // Шестой тоже успешно добавляется.
    atlas.add_rgba(&gpu.queue, "s5", 16, 16, &opaque).expect("шестой");
    assert_eq!(atlas.len(), 6);
}

// ── 4. Пакетная загрузка папки ──────────────────────────────────────────────

#[test]
fn load_folder_batch() {
    let Some(gpu) = gpu() else { return };
    let dir = temp_dir().join(format!("dioma_render_test_load_folder_batch_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir); // остатки прошлого запуска — best-effort
    fs::create_dir_all(&dir).expect("создать temp-папку");

    let a = vec![255u8; 4 * 4 * 4];
    let b = vec![200u8; 6 * 6 * 4];
    save_png(dir.join("a.png").to_str().unwrap(), 4, 4, &a).expect("a.png");
    save_png(dir.join("b.png").to_str().unwrap(), 6, 6, &b).expect("b.png");

    let mut atlas = SpriteAtlas::new(&gpu.device, 64);
    let n = atlas.load_folder(&gpu.queue, dir.to_str().unwrap()).expect("load_folder ok");
    assert_eq!(n, 2, "загружено два PNG");
    let ai = atlas.region_info("a").expect("регион 'a'");
    let bi = atlas.region_info("b").expect("регион 'b'");
    assert_eq!(ai.size_px, [4, 4], "имя региона = имя файла без .png");
    assert_eq!(bi.size_px, [6, 6]);

    let _ = fs::remove_dir_all(&dir);
}

// ── 5. Ошибки ввода ─────────────────────────────────────────────────────────

#[test]
fn add_png_missing_file_errors() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 64);
    let path = temp_dir().join("dioma_render_test_add_png_missing_file_errors.png");
    let err = atlas.add_png(&gpu.queue, "x", path.to_str().unwrap());
    assert!(err.is_err(), "нет файла → ошибка");
}

#[test]
fn png_bad_data_errors() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 64);
    let garbage = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09";
    let err = atlas.add_png_bytes(&gpu.queue, "x", garbage);
    assert!(err.is_err(), "мусорные байты → ошибка декодирования");
}

/// Скачанные CC0-спрайты (assets/sprites) грузятся через load_folder;
/// атлас растёт под крупный bunny.png (248×340 > стартовые 256).
#[test]
fn assets_folder_loads() {
    let Some(gpu) = gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 256);
    let n = atlas
        .load_folder(&gpu.queue, "assets/sprites")
        .expect("папка assets/sprites существует в крейте");
    assert!(n >= 8, "ожидаем ≥8 спрайтов, загружено {n}");
    assert!(atlas.region("bunny").is_some(), "bunny.png зарегистрирован по имени файла");
    assert!(atlas.region("apple").is_some(), "apple.png зарегистрирован");
    // Атлас вырос: bunny 248×340 не влезает в 256².
    assert!(atlas.width() >= 512, "атлас вырос под bunny: {}", atlas.width());
}
