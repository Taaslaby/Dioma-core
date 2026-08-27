//! Миграция старого научного формата `.taas` v1 (TAAS1) на игровой v2 (TAAS2).
//!
//! TAAS1 хранил «компоненты» (тензоры + скаляры с shape/dtype). TAAS2 — это
//! именованные типизированные секции. Миграция маппит каждый компонент в
//! секцию с тем же именем и dtype, пересчитывает CRC и пересобирает файл.
//! Игра всегда работает только с v2 — легаси читается прозрачно.

use crate::types::*;
use crate::writer::TaasWriter;
use std::collections::HashMap;

/// Структуры старого манифеста (v0.1.0 / v0.2.0, магия TAAS1).
#[derive(serde::Deserialize)]
struct OldManifest {
    version: String,
    #[serde(default)]
    #[allow(dead_code)]
    created: String,
    #[serde(default)]
    components: Vec<OldComponent>,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

#[derive(serde::Deserialize)]
struct OldComponent {
    name: String,
    #[serde(default)]
    #[allow(dead_code)]
    component_type: String,
    dtype: String,
    #[serde(default)]
    #[allow(dead_code)]
    shape: Vec<usize>,
    offset: u64,
    size: u64,
    #[serde(default)]
    crc32: Option<u32>,
    #[serde(default)]
    compression: Option<String>,
    #[serde(default)]
    uncompressed_size: Option<u64>,
}

fn kind_from_dtype(dtype: &str) -> Option<SectionKind> {
    match dtype {
        "f32" => Some(SectionKind::F32),
        "i32" => Some(SectionKind::I32),
        "i64" => Some(SectionKind::I64),
        "u64" => Some(SectionKind::U64),
        "u8" => Some(SectionKind::U8),
        "str" => Some(SectionKind::Str),
        "json" => Some(SectionKind::Json),
        "blob" => Some(SectionKind::Blob),
        _ => None,
    }
}

/// Мигрировать сырые байты в актуальный формат.
///
/// - TAAS2 — возвращает байты как есть;
/// - TAAS1 — пересобирает в TAAS2;
/// - что-то ещё — ошибка.
pub fn migrate_bytes(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if bytes.starts_with(TAAS_MAGIC) {
        return Ok(bytes.to_vec());
    }
    if !bytes.starts_with(TAAS1_MAGIC) {
        return Err(format!(
            "Not a .taas file: magic {:?}",
            &bytes[..bytes.len().min(5)]
        ));
    }
    migrate_taas1(bytes)
}

/// Нужна ли миграция файлу с данной магией.
pub fn needs_migration(magic: &[u8]) -> bool {
    magic.starts_with(TAAS1_MAGIC)
}

fn migrate_taas1(bytes: &[u8]) -> Result<Vec<u8>, String> {
    // Парсим заголовок TAAS1: MAGIC(5) + header_len(8) + JSON.
    let mut pos = 5usize;
    if bytes.len() < 5 + 8 {
        return Err("TAAS1 file too short".to_string());
    }
    let header_len = u64::from_le_bytes(
        bytes[pos..pos + 8]
            .try_into()
            .map_err(|_| "TAAS1 header length truncated".to_string())?,
    ) as usize;
    pos += 8;
    if pos + header_len > bytes.len() {
        return Err("TAAS1 header truncated".to_string());
    }
    let header_str = std::str::from_utf8(&bytes[pos..pos + header_len])
        .map_err(|e| format!("TAAS1 header is not UTF-8: {e}"))?;
    let old: OldManifest =
        serde_json::from_str(header_str).map_err(|e| format!("TAAS1 manifest JSON: {e}"))?;
    pos += header_len;
    let data = &bytes[pos..];

    let mut writer = TaasWriter::new();
    writer.add_metadata("migrated_from", &old.version);
    for (k, v) in &old.metadata {
        writer.add_metadata(k, v);
    }

    for comp in &old.components {
        let start = comp.offset as usize;
        let end = start + comp.size as usize;
        if end > data.len() {
            return Err(format!(
                "TAAS1 component '{}' (offset={}, size={}) out of bounds (data len={})",
                comp.name,
                comp.offset,
                comp.size,
                data.len()
            ));
        }
        let stored = &data[start..end];
        // Распаковываем, если сжато.
        let raw = match comp.compression.as_deref() {
            Some("zstd") => {
                let mut dec = zstd::Decoder::new(stored)
                    .map_err(|e| format!("TAAS1 '{}' zstd: {e}", comp.name))?;
                let mut buf = Vec::with_capacity(comp.uncompressed_size.unwrap_or(0) as usize);
                std::io::Read::read_to_end(&mut dec, &mut buf)
                    .map_err(|e| format!("TAAS1 '{}' zstd: {e}", comp.name))?;
                buf
            }
            _ => stored.to_vec(),
        };
        if let Some(expected) = comp.crc32 {
            let actual = crate::writer::compute_crc32(&raw);
            if actual != expected {
                return Err(format!(
                    "TAAS1 component '{}' CRC32 mismatch: expected {:08x}, got {:08x}",
                    comp.name, expected, actual
                ));
            }
        }
        let kind = kind_from_dtype(&comp.dtype).ok_or_else(|| {
            format!("TAAS1 component '{}' has unknown dtype '{}'", comp.name, comp.dtype)
        })?;
        let count = match kind.elem_size() {
            Some(es) => raw.len() / es,
            None => raw.len(),
        };
        writer.push(&comp.name, kind, count, &raw);
    }

    writer.to_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::TaasReader;

    /// Собрать TAAS1-файл вручную (старый манифест + сырые данные) и
    /// проверить, что миграция даёт v2 с теми же данными.
    fn build_taas1(components: &[(&str, &str, u64, &[u8])], meta: &[(&str, &str)]) -> Vec<u8> {
        let mut offset = 0u64;
        let comps: Vec<serde_json::Value> = components
            .iter()
            .map(|(name, dtype, count, data)| {
                let c = serde_json::json!({
                    "name": name,
                    "component_type": "mig",
                    "dtype": dtype,
                    "shape": [*count as usize],
                    "offset": offset,
                    "size": data.len(),
                });
                offset += data.len() as u64;
                c
            })
            .collect();
        let mut meta_obj = serde_json::Map::new();
        for (k, v) in meta {
            meta_obj.insert(k.to_string(), serde_json::json!(v));
        }
        let man = serde_json::json!({
            "version": "0.2.0",
            "created": "2026-01-01T00:00:00Z",
            "components": comps,
            "metadata": meta_obj,
        });
        let header = serde_json::to_vec(&man).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(TAAS1_MAGIC);
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        for (_, _, _, data) in components {
            out.extend_from_slice(data);
        }
        out
    }

    #[test]
    fn test_migrate_taas1_roundtrip() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ids_raw: Vec<u8> = [1i32, 2].iter().flat_map(|&v| v.to_le_bytes()).collect();
        let old = build_taas1(
            &[("weights", "f32", 4, &raw), ("ids", "i32", 2, &ids_raw)],
            &[("model", "mig-test")],
        );
        let migrated = migrate_bytes(&old).unwrap();
        assert!(migrated.starts_with(TAAS_MAGIC), "magic is TAAS2");

        let reader = TaasReader::from_bytes(&migrated).unwrap();
        assert_eq!(reader.manifest.version, TAAS_VERSION);
        assert_eq!(reader.get_f32("weights").unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(reader.get_i32("ids").unwrap(), &[1, 2]);
        assert_eq!(reader.metadata().get("model").unwrap(), "mig-test");
        assert_eq!(reader.metadata().get("migrated_from").unwrap(), "0.2.0");
        // CRC теперь есть у каждой секции.
        assert!(reader.section("weights").unwrap().crc32.is_some());
    }

    #[test]
    fn test_migrate_corrupt_crc_rejected() {
        // Старый компонент заявляет CRC, противоречащий данным, — миграция
        // обязана отвергнуть файл, а не тащить порчу дальше.
        let data = 1.0f32.to_le_bytes();
        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(TAAS1_MAGIC);
        let man = serde_json::json!({
            "version": "0.2.0",
            "created": "2026-01-01T00:00:00Z",
            "components": [{
                "name": "w",
                "component_type": "mig",
                "dtype": "f32",
                "shape": [1],
                "offset": 0,
                "size": data.len(),
                "crc32": 0xDEAD_BEEFu64
            }],
            "metadata": {}
        });
        let header = serde_json::to_vec(&man).unwrap();
        raw.extend_from_slice(&(header.len() as u64).to_le_bytes());
        raw.extend_from_slice(&header);
        raw.extend_from_slice(&data);

        assert!(migrate_bytes(&raw).is_err());
    }

    #[test]
    fn test_migrate_current_is_noop() {
        let mut w = TaasWriter::new();
        w.f32("x", &[1.0]);
        let bytes = w.to_bytes().unwrap();
        assert_eq!(migrate_bytes(&bytes).unwrap(), bytes);
    }
}
