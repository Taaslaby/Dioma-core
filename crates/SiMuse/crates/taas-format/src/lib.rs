//! # `.taas` v2 — игровой формат снапшотов и реплея
//!
//! Движок больше не научный: формат переработан под сохранения/реплей игры.
//! Вместо тензоров с shape — **именованные типизированные секции**: игре
//! достаточно знать имя секции, читатель сам разберёт kind и count.
//!
//! ## Бинарный формат `.taas`
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │  MAGIC: b"TAAS2" (5 bytes)           │
//! │  HEADER_LEN: u64 LE (8 bytes)        │
//! │  HEADER: UTF-8 JSON (HEADER_LEN)     │
//! │  ├── version: "2.0.0"               │
//! │  ├── created: "2026-08-27T..."      │
//! │  ├── game: "dioma" (опционально)    │
//! │  ├── sections: [                    │
//! │  │    { name, kind, count, offset,   │
//! │  │      size, crc32, compression,    │
//! │  │      uncompressed_size }          │
//! │  └── metadata: { ... }              │
//! │  SECTION_DATA: [...]                 │
//! └──────────────────────────────────────┘
//! ```
//!
//! Каждая секция CRC32-подписана (по распакованным данным) и может быть
//! сжата zstd независимо от остальных. Старые научные файлы `TAAS1`
//! авто-мигрируются при чтении (`TaasReader`).
//!
//! ## Реплей (`REPL2`)
//!
//! Append-only поток событий `{tick, kind, data}` — см. [`replay`]. Пишется
//! по ходу симуляции, читается последовательно; каждый кадр самодостаточен
//! (CRC), битый кадр можно пропустить без потери остального лога.

pub mod inspect;
pub mod migration;
pub mod reader;
pub mod replay;
pub mod types;
pub mod writer;

pub use inspect::inspect_taas;
pub use migration::{migrate_bytes, needs_migration};
pub use reader::TaasReader;
pub use replay::{ReplayEvent, ReplayReader, ReplayWriter, REPLAY_MAGIC, REPLAY_VERSION};
pub use types::{SectionEntry, SectionKind, TaasManifest, TAAS1_MAGIC, TAAS_MAGIC, TAAS_VERSION};
pub use writer::TaasWriter;

// ─── Сборка файла: MAGIC + header_len + JSON-манифест + payload ──────

pub(crate) fn assemble_bytes(manifest: &TaasManifest, payload: &[u8]) -> Result<Vec<u8>, String> {
    let header =
        serde_json::to_vec(manifest).map_err(|e| format!("manifest JSON: {e}"))?;
    let mut out = Vec::with_capacity(13 + header.len() + payload.len());
    out.extend_from_slice(TAAS_MAGIC);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(payload);
    Ok(out)
}

// ─── Fallback timestamp (без chrono-зависимости) ────────────

pub(crate) fn chrono_or_fallback() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let secs_rem = time_secs % 60;
    let year = 1970 + (days as f64 / 365.25) as u64;
    format!("{year}-01-01T{hours:02}:{mins:02}:{secs_rem:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::ReplayWriter;
    use crate::types::SectionKind;

    #[test]
    fn test_roundtrip_all_kinds() {
        let mut w = TaasWriter::new();
        w.game("dioma")
            .add_metadata("world", "meadow")
            .f32("weights", &[1.0, 2.0, 3.0])
            .i32("ids", &[10, 20])
            .i64("big", &[1i64 << 40])
            .u64("stamps", &[42])
            .u8("mask", &[0, 255, 7])
            .str("name", "creature-7")
            .json("genome", &serde_json::json!({"mut": 0.01, "tags": ["a"]}))
            .blob("raw", &[9, 8, 7]);

        let bytes = w.to_bytes().unwrap();
        let r = TaasReader::from_bytes(&bytes).unwrap();

        assert_eq!(r.manifest.version, TAAS_VERSION);
        assert_eq!(r.manifest.game.as_deref(), Some("dioma"));
        assert_eq!(r.metadata().get("world").unwrap(), "meadow");
        assert_eq!(r.get_f32("weights").unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(r.get_i32("ids").unwrap(), &[10, 20]);
        assert_eq!(r.get_i64("big").unwrap(), &[1i64 << 40]);
        assert_eq!(r.get_u64("stamps").unwrap(), &[42]);
        assert_eq!(r.get_u8("mask").unwrap(), &[0, 255, 7]);
        assert_eq!(r.get_str("name").unwrap(), "creature-7");
        let g: serde_json::Value = r.get_json("genome").unwrap();
        assert_eq!(g["mut"], 0.01);
        assert_eq!(r.get_blob("raw").unwrap(), &[9, 8, 7]);
        assert!(r.has_section("weights"));
        assert!(!r.has_section("nope"));
        assert_eq!(r.list_sections().len(), 8);
        assert!(r.validate().is_ok());
    }

    #[test]
    fn test_wrong_kind_rejected() {
        let mut w = TaasWriter::new();
        w.i32("x", &[1]);
        let r = TaasReader::from_bytes(&w.to_bytes().unwrap()).unwrap();
        assert!(r.get_f32("x").is_err());
    }

    #[test]
    fn test_corruption_detected() {
        let mut w = TaasWriter::new();
        w.f32("data", &[1.0; 64]);
        let mut bytes = w.to_bytes().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let r = TaasReader::from_bytes(&bytes).unwrap();
        // Манифест читается, но секция падает по CRC.
        assert!(r.get_f32("data").is_err());
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_invalid_magic() {
        assert!(TaasReader::from_bytes(b"NOTTAAS\x00\x00\x00").is_err());
    }

    #[test]
    fn test_zstd_roundtrip() {
        let big: Vec<f32> = (0..4096).map(|i| i as f32).collect();
        let mut w = TaasWriter::new();
        w.f32("raw", &big); // без сжатия
        w.set_compression(3);
        w.f32("compressed", &big);
        w.disable_compression();
        w.f32("raw2", &big);

        let bytes = w.to_bytes().unwrap();
        let r = TaasReader::from_bytes(&bytes).unwrap();
        assert!(r.section("raw").unwrap().compression.is_none());
        let comp = r.section("compressed").unwrap();
        assert_eq!(comp.compression.as_deref(), Some("zstd"));
        assert_eq!(
            comp.uncompressed_size,
            Some((big.len() * 4) as u64)
        );
        assert!(r.section("raw2").unwrap().compression.is_none());
        assert_eq!(r.get_f32("raw").unwrap(), big);
        assert_eq!(r.get_f32("compressed").unwrap(), big);
        assert_eq!(r.get_f32("raw2").unwrap(), big);
        assert!(r.validate().is_ok());
    }

    #[test]
    fn test_auto_migration_from_taas1() {
        // TAAS1-файл читается прозрачно через from_bytes.
        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(TAAS1_MAGIC);
        let f: Vec<f32> = vec![7.0, 8.0];
        let data: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
        let man = serde_json::json!({
            "version": "0.2.0",
            "created": "2026-01-01T00:00:00Z",
            "components": [{
                "name": "weights",
                "component_type": "Test",
                "dtype": "f32",
                "shape": [2],
                "offset": 0,
                "size": data.len()
            }],
            "metadata": {"model": "legacy"}
        });
        let header = serde_json::to_vec(&man).unwrap();
        raw.extend_from_slice(&(header.len() as u64).to_le_bytes());
        raw.extend_from_slice(&header);
        raw.extend_from_slice(&data);

        let r = TaasReader::from_bytes(&raw).unwrap();
        assert_eq!(r.manifest.version, TAAS_VERSION);
        assert_eq!(r.get_f32("weights").unwrap(), &[7.0, 8.0]);
        assert_eq!(r.metadata().get("model").unwrap(), "legacy");
        assert_eq!(r.metadata().get("migrated_from").unwrap(), "0.2.0");
    }

    #[test]
    fn test_inspect_output() {
        let mut w = TaasWriter::new();
        w.game("dioma").f32("test", &[1.0, 2.0, 3.0]).add_metadata("world", "meadow");
        let tmp = "test_inspect2.taas";
        w.to_file(tmp).unwrap();
        let out = inspect_taas(tmp).unwrap();
        assert!(out.contains("test"));
        assert!(out.contains("dioma"));
        assert!(out.contains("meadow"));
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_file_roundtrip() {
        let tmp = "test_file2.taas";
        let mut w = TaasWriter::new();
        w.f32("x", &[1.5, -2.5]);
        w.to_file(tmp).unwrap();
        let r = TaasReader::from_file(tmp).unwrap();
        assert_eq!(r.get_f32("x").unwrap(), &[1.5, -2.5]);
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_section_kind_mapping() {
        assert_eq!(SectionKind::F32.as_str(), "f32");
        assert_eq!(SectionKind::parse("i64"), Some(SectionKind::I64));
        assert_eq!(SectionKind::parse("nope"), None);
        assert_eq!(SectionKind::F32.elem_size(), Some(4));
        assert_eq!(SectionKind::Json.elem_size(), None);
    }

    /// Снапшот «существа» поверх секций: brain (f32 веса), genome (json),
    /// метаданные — и бит-в-бит рестор.
    #[test]
    fn test_creature_save_restore() {
        let weights: Vec<f32> = (0..120).map(|i| (i as f32) * 0.01).collect();
        let mut w = TaasWriter::new();
        w.game("dioma")
            .u64("tick", &[1234])
            .f32("brain_weights", &weights)
            .f32("brain_state", &vec![0.5; 120])
            .json("genome", &serde_json::json!({"grow_seed": 777, "mut": 0.05}))
            .u8("instincts", &[1, 0, 1]);
        let bytes = w.to_bytes().unwrap();

        let r = TaasReader::from_bytes(&bytes).unwrap();
        assert_eq!(r.get_u64("tick").unwrap(), &[1234]);
        assert_eq!(r.get_f32("brain_weights").unwrap(), weights);
        assert_eq!(r.get_f32("brain_state").unwrap(), vec![0.5; 120]);
        let g: serde_json::Value = r.get_json("genome").unwrap();
        assert_eq!(g["grow_seed"], 777);
        assert_eq!(r.get_u8("instincts").unwrap(), &[1, 0, 1]);
    }

    /// Соседство снапшота и реплея в одном прогоне.
    #[test]
    fn test_snapshot_and_replay_together() {
        let mut snap = TaasWriter::new();
        snap.game("dioma").u64("tick", &[10]).f32("w", &[1.0; 8]);
        let snap_bytes = snap.to_bytes().unwrap();
        let r = TaasReader::from_bytes(&snap_bytes).unwrap();
        assert!(r.validate().is_ok());

        let mut rep = ReplayWriter::new();
        rep.push(0, "spawn", &serde_json::json!({"x": 0.0})).unwrap();
        rep.push(10, "snapshot", &serde_json::json!({"file": "quicksave.taas"})).unwrap();
        let rep_bytes = rep.into_bytes().unwrap();
        let mut rr = crate::replay::ReplayReader::from_bytes(&rep_bytes).unwrap();
        let evs = rr.collect().unwrap();
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[1].kind, "snapshot");
    }
}
