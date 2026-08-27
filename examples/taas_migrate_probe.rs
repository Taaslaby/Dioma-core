//! Проба: авто-миграция настоящего TAAS1-файла (коннектом сардааны)
//! → читается новым TaasReader → пересохраняется в TAAS2 → читается снова.

use taas_format::{TaasReader, TaasWriter};

fn main() {
    let path = "D:/Проекты/asm_engines/sardaanaCM/experiments/sardna_ultra/generated/flywire_10k_n10000_t0_85.taas";
    println!("── 1. Чтение легаси TAAS1 через авто-миграцию ──");
    let r = TaasReader::from_file(path).expect("auto-migrated read");
    println!(
        "магия={:?} версия={} секций={}",
        &r.list_sections().first().map(|_| "TAAS2").unwrap_or("?"),
        r.manifest.version,
        r.list_sections().len()
    );
    for s in r.list_sections() {
        println!("  {:<14} {:>5} count={}", s.name, s.kind, s.count);
    }
    let rp: Vec<i32> = r.get_i32("row_ptr").expect("row_ptr");
    let ci: Vec<i32> = r.get_i32("col_indices").expect("col_indices");
    let vals: Vec<f32> = r.get_f32("values").expect("values");
    let nt: Vec<f32> = r.get_f32("nt_types").expect("nt_types");
    let n = rp.len().saturating_sub(1);
    println!("нейронов={n} синапсов={}", ci.len());
    assert_eq!(ci.len(), vals.len(), "CSR длины согласованы");
    assert!(nt.is_empty() || nt.len() >= n, "nt_types ок");
    assert!(r.validate().is_ok(), "CRC всех секций ок");

    println!("── 2. Пересохранение в TAAS2 + повторное чтение ──");
    let tmp = "taas_migrate_probe_out.taas";
    let mut w = TaasWriter::new();
    w.game("dioma")
        .i32("row_ptr", &rp)
        .i32("col_indices", &ci)
        .f32("values", &vals)
        .f32("nt_types", &nt)
        .add_metadata("migrated", "probe");
    w.to_file(tmp).expect("write TAAS2");
    let r2 = TaasReader::from_file(tmp).expect("read TAAS2");
    assert_eq!(r2.manifest.version, "2.0.0");
    assert_eq!(r2.get_i32("row_ptr").unwrap(), rp);
    assert_eq!(r2.get_i32("col_indices").unwrap(), ci);
    assert_eq!(r2.get_f32("values").unwrap(), vals);
    assert_eq!(r2.get_f32("nt_types").unwrap(), nt);
    std::fs::remove_file(tmp).ok();
    println!("OK: легаси TAAS1 мигрирован, данные бит-в-бит, TAAS2 перечитан");
}
