//! Экспортёр эталонного `.taas` v2 для Unity-порта.
//!
//! Пишет файл СТАНДАРТНЫМ писателем `taas-format` (тот самый крейт, что
//! использует движок) с секциями всех типов и известными данными. C#-читатель
//! обязан разобрать его и получить те же числа: это доказывает, что порт
//! формата совместим, а не «работает сам с собой».
//!
//! Секции без сжатия: C#-порт не тянет zstd (в Unity его нет без внешней
//! зависимости), и для сохранений это не нужно — веса 32k синапсов это 128 КБ.
//!
//! Запуск: `cargo run --release --bin export_taas -- <dir>`
//! (dir = dioma-engine/Tests/Editor/Data).

use std::fs;
use std::path::PathBuf;

use taas_format::{TaasReader, TaasWriter};

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    fs::create_dir_all(&out_dir).expect("out dir");

    // Данные подобраны так, чтобы поймать типовые ошибки порта: порядок
    // байтов, знаковые/беззнаковые, денормали, спецзначения, юникод.
    let floats: Vec<f32> = vec![
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.125,
        3.402_823_5e38,   // близко к f32::MAX
        1.175_494_4e-38,  // близко к min positive normal
        core::f32::consts::PI,
        -2.718_281_8,
        1e-7,
    ];
    let ints: Vec<i32> = vec![0, 1, -1, 127, -128, 32767, -32768, 2_147_483_647, -2_147_483_648, 42];
    let longs: Vec<i64> = vec![0, -1, 9_223_372_036_854_775_807, -9_223_372_036_854_775_808, 1234567890123];
    let bytes: Vec<u8> = (0..=255u8).collect();

    let mut w = TaasWriter::new();
    w.game("dioma");
    w.add_metadata("purpose", "паритет формата");
    w.add_metadata("connectome", "dioma_c");
    w.add_metadata("synapses", "32583");
    w.f32("brain.weights", &floats);
    w.i32("regions", &ints);
    w.i64("counters", &longs);
    w.u8("palette", &bytes);
    w.str("note", "существо №1 — «Кит», кириллица + emoji 🐋");
    w.json(
        "genome",
        &serde_json::json!({
            "metabolism": {
                "costPerActivity": 0.0015,
                "starveDamage": 0.004,
                "costPerDopamine": 0.4,
                "healRate": 0.0001
            },
            "instincts": [
                { "dopamine": 2.5, "inputs": [[0, 4.0], [56, 8.0]] }
            ],
            "life": { "foodEaten": 17, "wallHits": 3, "nights": 5 }
        }),
    );

    let path = out_dir.join("dioma_reference.taas");
    let bytes_out = w.to_bytes().expect("assemble taas");
    fs::write(&path, &bytes_out).expect("write taas");

    // Самопроверка: эталонный писатель и читатель должны сойтись.
    let r = TaasReader::from_bytes(&bytes_out).expect("read back");
    println!(
        "эталон: {} ({} байт, секций {})",
        path.display(),
        bytes_out.len(),
        r.manifest.sections.len()
    );
    for sec in &r.manifest.sections {
        println!(
            "  {:<14} {:<5} count={:<5} size={:<6} crc={:08x}",
            sec.name,
            sec.kind,
            sec.count,
            sec.size,
            sec.crc32.unwrap_or(0)
        );
    }

    // Ожидаемые значения — в JSON рядом: C#-тест сверяет числа, а не только
    // «файл прочитался». i64 пишем СТРОКАМИ: значения на границах диапазона
    // (±2^63) не проходят через double без потери точности, а именно они и
    // ловят ошибки порядка байтов.
    let expect = format!(
        "{{\"floats\":[{}],\"ints\":[{}],\"longs\":[{}],\"byteCount\":{},\"note\":{},\
         \"game\":\"dioma\",\"sections\":{}}}\n",
        floats
            .iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<_>>()
            .join(","),
        ints.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
        longs
            .iter()
            .map(|v| format!("\"{v}\""))
            .collect::<Vec<_>>()
            .join(","),
        bytes.len(),
        serde_json::to_string("существо №1 — «Кит», кириллица + emoji 🐋").unwrap(),
        r.manifest.sections.len(),
    );
    let expect_path = out_dir.join("dioma_reference_taas_expect.json");
    fs::write(&expect_path, &expect).expect("write expect");
    println!("ожидания: {}", expect_path.display());
}
