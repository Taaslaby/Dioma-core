//! Читатель `.taas` v2: именованные типизированные секции.
//!
//! Старые файлы TAAS1 (научный формат) автоматически мигрируются на v2
//! при чтении — игре не нужно знать про легаси.

use crate::types::*;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

enum DataSource {
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl DataSource {
    fn as_slice(&self) -> &[u8] {
        match self {
            DataSource::Owned(v) => v.as_slice(),
            DataSource::Mapped(m) => m.as_ref(),
        }
    }
}

pub struct TaasReader {
    pub manifest: TaasManifest,
    source: DataSource,
    data_offset: usize,
}

impl TaasReader {
    /// Прочитать из байтов. Если магия — TAAS1 (старый научный формат),
    /// файл сначала мигрируется на v2.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let bytes = if bytes.starts_with(TAAS1_MAGIC) {
            crate::migration::migrate_bytes(bytes)?
        } else {
            bytes.to_vec()
        };
        let (manifest, data_offset) = Self::parse_header(&bytes)?;
        Self::validate_offsets(&manifest, &bytes, data_offset)?;
        Ok(Self {
            manifest,
            source: DataSource::Owned(bytes),
            data_offset,
        })
    }

    /// mmap-чтение с диска. Старые TAAS1 тоже мигрируются.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let file =
            std::fs::File::open(path.as_ref()).map_err(|e| format!("Cannot open file: {e}"))?;
        let mmap =
            unsafe { memmap2::Mmap::map(&file).map_err(|e| format!("Cannot mmap file: {e}"))? };
        let bytes = mmap.as_ref();
        if bytes.starts_with(TAAS1_MAGIC) {
            // Миграция требует пересборки — читаем в память и идём через from_bytes.
            let owned = bytes.to_vec();
            return Self::from_bytes(&owned);
        }
        let (manifest, data_offset) = Self::parse_header(bytes)?;
        Self::validate_offsets(&manifest, bytes, data_offset)?;
        Ok(Self {
            manifest,
            source: DataSource::Mapped(mmap),
            data_offset,
        })
    }

    fn parse_header(bytes: &[u8]) -> Result<(TaasManifest, usize), String> {
        let mut pos = 0usize;
        if bytes.len() < 5 + 8 {
            return Err("File too short".to_string());
        }
        if &bytes[0..5] != TAAS_MAGIC {
            return Err(format!(
                "Invalid magic: expected TAAS2, got {:?}",
                &bytes[0..5]
            ));
        }
        pos += 5;
        let header_len = u64::from_le_bytes(
            bytes[pos..pos + 8]
                .try_into()
                .map_err(|_| "Header length field truncated".to_string())?,
        ) as usize;
        pos += 8;
        if pos + header_len > bytes.len() {
            return Err("Header truncated".to_string());
        }
        let header_str = std::str::from_utf8(&bytes[pos..pos + header_len])
            .map_err(|e| format!("Invalid UTF-8 in header: {e}"))?;
        let manifest: TaasManifest =
            serde_json::from_str(header_str).map_err(|e| format!("Invalid manifest JSON: {e}"))?;
        pos += header_len;
        Ok((manifest, pos))
    }

    fn validate_offsets(
        manifest: &TaasManifest,
        bytes: &[u8],
        data_offset: usize,
    ) -> Result<(), String> {
        for sec in &manifest.sections {
            let start = data_offset + sec.offset as usize;
            let end = start + sec.size as usize;
            if end > bytes.len() {
                return Err(format!(
                    "Section '{}' (offset={}, size={}) exceeds data (len={})",
                    sec.name,
                    sec.offset,
                    sec.size,
                    bytes.len() - data_offset
                ));
            }
        }
        Ok(())
    }

    fn data_bytes(&self) -> &[u8] {
        &self.source.as_slice()[self.data_offset..]
    }

    fn read_bytes(&self, entry: &SectionEntry) -> Result<Vec<u8>, String> {
        let data = self.data_bytes();
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        if end > data.len() {
            return Err(format!("Section '{}' data out of bounds", entry.name));
        }
        let slice = &data[start..end];

        // Decompress if needed.
        let decompressed = match entry.compression.as_deref() {
            Some("zstd") => {
                let mut decoder = zstd::Decoder::new(slice).map_err(|e| {
                    format!("Section '{}' zstd decompression failed: {e}", entry.name)
                })?;
                let mut buf = Vec::with_capacity(entry.uncompressed_size.unwrap_or(0) as usize);
                decoder.read_to_end(&mut buf).map_err(|e| {
                    format!("Section '{}' zstd decompression failed: {e}", entry.name)
                })?;
                buf
            }
            _ => slice.to_vec(),
        };

        // Verify CRC32 on (decompressed) data.
        if let Some(expected_crc) = entry.crc32 {
            let actual_crc = crate::writer::compute_crc32(&decompressed);
            if actual_crc != expected_crc {
                return Err(format!(
                    "Section '{}' CRC32 mismatch: expected {:08x}, got {:08x}",
                    entry.name, expected_crc, actual_crc
                ));
            }
        }

        Ok(decompressed)
    }

    fn check_kind(&self, name: &str, want: SectionKind) -> Result<&SectionEntry, String> {
        let entry = self.section(name)?;
        if entry.kind != want.as_str() {
            return Err(format!(
                "Section '{name}' is not {}, got {}",
                want.as_str(),
                entry.kind
            ));
        }
        Ok(entry)
    }

    /// Сырые (распакованные) байты секции с проверкой CRC.
    pub fn get_blob(&self, name: &str) -> Result<Vec<u8>, String> {
        let entry = self.section(name)?;
        self.read_bytes(entry)
    }

    fn check_len(name: &str, entry: &SectionEntry, src: &[u8], es: usize) -> Result<(), String> {
        let expect = entry.count * es;
        if src.len() != expect {
            return Err(format!(
                "Section '{name}' {}: data len {} != count {} * {es}",
                entry.kind,
                src.len(),
                entry.count
            ));
        }
        Ok(())
    }

    pub fn get_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let entry = self.check_kind(name, SectionKind::F32)?;
        let src = self.read_bytes(entry)?;
        Self::check_len(name, entry, &src, 4)?;
        Ok(src
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn get_i32(&self, name: &str) -> Result<Vec<i32>, String> {
        let entry = self.check_kind(name, SectionKind::I32)?;
        let src = self.read_bytes(entry)?;
        Self::check_len(name, entry, &src, 4)?;
        Ok(src
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn get_i64(&self, name: &str) -> Result<Vec<i64>, String> {
        let entry = self.check_kind(name, SectionKind::I64)?;
        let src = self.read_bytes(entry)?;
        Self::check_len(name, entry, &src, 8)?;
        Ok(src
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn get_u64(&self, name: &str) -> Result<Vec<u64>, String> {
        let entry = self.check_kind(name, SectionKind::U64)?;
        let src = self.read_bytes(entry)?;
        Self::check_len(name, entry, &src, 8)?;
        Ok(src
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn get_u8(&self, name: &str) -> Result<Vec<u8>, String> {
        let entry = self.check_kind(name, SectionKind::U8)?;
        self.read_bytes(entry)
    }

    pub fn get_str(&self, name: &str) -> Result<String, String> {
        let entry = self.check_kind(name, SectionKind::Str)?;
        let src = self.read_bytes(entry)?;
        String::from_utf8(src).map_err(|e| format!("Section '{name}' is not valid UTF-8: {e}"))
    }

    pub fn get_json<T: serde::de::DeserializeOwned>(&self, name: &str) -> Result<T, String> {
        let entry = self.check_kind(name, SectionKind::Json)?;
        let src = self.read_bytes(entry)?;
        serde_json::from_slice(&src).map_err(|e| format!("Section '{name}' bad JSON: {e}"))
    }

    pub fn has_section(&self, name: &str) -> bool {
        self.manifest.sections.iter().any(|s| s.name == name)
    }

    pub fn section(&self, name: &str) -> Result<&SectionEntry, String> {
        self.manifest
            .sections
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| format!("Section '{name}' not found in manifest"))
    }

    pub fn list_sections(&self) -> &[SectionEntry] {
        &self.manifest.sections
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.manifest.metadata
    }

    pub fn total_size(&self) -> usize {
        self.source.as_slice().len()
    }

    /// Прочитать и проверить CRC всех секций.
    pub fn validate(&self) -> Result<(), String> {
        for sec in &self.manifest.sections {
            self.read_bytes(sec)?;
        }
        Ok(())
    }
}
