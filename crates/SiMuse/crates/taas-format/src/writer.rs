//! Писатель `.taas` v2: секции по именам, типизированные, с CRC и zstd.

use crate::types::*;
use std::collections::BTreeMap;

/// Помощник: CRC32 байтов.
pub(crate) fn compute_crc32(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

pub struct TaasWriter {
    sections: Vec<SectionEntry>,
    payload: Vec<u8>,
    metadata: BTreeMap<String, String>,
    compression: Option<i32>,
    game: Option<String>,
    created: Option<String>,
}

impl Default for TaasWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TaasWriter {
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
            payload: Vec::new(),
            metadata: BTreeMap::new(),
            compression: None,
            game: None,
            created: None,
        }
    }

    /// Пометить файл игрой/сценой (полезно при загрузке: какой мир открывать).
    pub fn game(&mut self, game: &str) -> &mut Self {
        self.game = Some(game.to_string());
        self
    }

    /// Зафиксировать метку времени вместо часов машины. Нужно эталонным
    /// файлам: иначе один и тот же снапшот даёт разные байты при каждом
    /// экспорте и его нельзя сверять с закоммиченным.
    pub fn created(&mut self, timestamp: &str) -> &mut Self {
        self.created = Some(timestamp.to_string());
        self
    }

    /// Включить zstd для последующих секций. level 1..22, дефолт 3.
    pub fn set_compression(&mut self, level: i32) -> &mut Self {
        self.compression = Some(level.clamp(1, 22));
        self
    }

    /// Отключить сжатие для последующих секций.
    pub fn disable_compression(&mut self) -> &mut Self {
        self.compression = None;
        self
    }

    pub fn add_metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Низкоуровневая запись секции: имя, kind, число элементов, сырые байты.
    pub fn push(&mut self, name: &str, kind: SectionKind, count: usize, data: &[u8]) -> &mut Self {
        let crc32 = Some(compute_crc32(data));
        let (stored, compression, uncompressed_size) = match self.compression {
            Some(level) => match zstd::bulk::compress(data, level) {
                Ok(c) => (c, Some("zstd".to_string()), Some(data.len() as u64)),
                Err(_) => (data.to_vec(), None, None),
            },
            None => (data.to_vec(), None, None),
        };
        let entry = SectionEntry {
            name: name.to_string(),
            kind: kind.as_str().to_string(),
            count,
            offset: self.payload.len() as u64,
            size: stored.len() as u64,
            crc32,
            compression,
            uncompressed_size,
        };
        self.payload.extend_from_slice(&stored);
        self.sections.push(entry);
        self
    }

    pub fn f32(&mut self, name: &str, data: &[f32]) -> &mut Self {
        let bytes = bytemuck::cast_slice(data);
        self.push(name, SectionKind::F32, data.len(), bytes)
    }

    pub fn i32(&mut self, name: &str, data: &[i32]) -> &mut Self {
        let bytes = bytemuck::cast_slice(data);
        self.push(name, SectionKind::I32, data.len(), bytes)
    }

    pub fn i64(&mut self, name: &str, data: &[i64]) -> &mut Self {
        let bytes = bytemuck::cast_slice(data);
        self.push(name, SectionKind::I64, data.len(), bytes)
    }

    pub fn u64(&mut self, name: &str, data: &[u64]) -> &mut Self {
        let bytes = bytemuck::cast_slice(data);
        self.push(name, SectionKind::U64, data.len(), bytes)
    }

    pub fn u8(&mut self, name: &str, data: &[u8]) -> &mut Self {
        self.push(name, SectionKind::U8, data.len(), data)
    }

    pub fn str(&mut self, name: &str, data: &str) -> &mut Self {
        self.push(name, SectionKind::Str, data.len(), data.as_bytes())
    }

    pub fn json<T: serde::Serialize>(&mut self, name: &str, value: &T) -> &mut Self {
        let s = serde_json::to_vec(value).unwrap_or_default();
        self.push(name, SectionKind::Json, s.len(), &s)
    }

    pub fn blob(&mut self, name: &str, data: &[u8]) -> &mut Self {
        self.push(name, SectionKind::Blob, data.len(), data)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let manifest = TaasManifest {
            version: TAAS_VERSION.to_string(),
            created: self.created.clone().unwrap_or_else(crate::chrono_or_fallback),
            game: self.game.clone(),
            sections: self.sections.clone(),
            metadata: self.metadata.clone(),
        };
        crate::assemble_bytes(&manifest, &self.payload)
    }

    pub fn to_file(&self, path: &str) -> Result<(), String> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| format!("write {path}: {e}"))
    }
}
