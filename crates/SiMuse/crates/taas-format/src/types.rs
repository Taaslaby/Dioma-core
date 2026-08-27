//! Типы формата `.taas` v2 — игровой снапшот: именованные типизированные
//! секции вместо научных тензоров. Секция = имя + kind + данные; читатель
//! достаёт по имени, игре не нужно знать shape наугад.

use std::collections::HashMap;

pub const TAAS_MAGIC: &[u8; 5] = b"TAAS2";
/// Магия старого научного формата — его читатель авто-мигрирует на v2.
pub const TAAS1_MAGIC: &[u8; 5] = b"TAAS1";
pub const TAAS_VERSION: &str = "2.0.0";

/// Тип данных секции. Числовые kind'ы имеют фиксированный размер элемента;
/// str/json/blob — сырые байты (count = длина в байтах).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    F32,
    I32,
    I64,
    U64,
    U8,
    Str,
    Json,
    Blob,
}

impl SectionKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::U8 => "u8",
            Self::Str => "str",
            Self::Json => "json",
            Self::Blob => "blob",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Self::F32),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "u64" => Some(Self::U64),
            "u8" => Some(Self::U8),
            "str" => Some(Self::Str),
            "json" => Some(Self::Json),
            "blob" => Some(Self::Blob),
            _ => None,
        }
    }

    /// Размер элемента в байтах для числовых kind'ов; None для str/json/blob.
    pub fn elem_size(&self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::I32 => Some(4),
            Self::I64 => Some(8),
            Self::U64 => Some(8),
            Self::U8 => Some(1),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SectionEntry {
    pub name: String,
    pub kind: String,
    /// Для числовых kind'ов — число элементов; для str/json/blob — длина в байтах.
    pub count: usize,
    pub offset: u64,
    /// Хранимый (возможно сжатый) размер в байтах.
    pub size: u64,
    /// CRC32 распакованных данных (опционально; старые архивы без него).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crc32: Option<u32>,
    /// Алгоритм сжатия ("zstd"); None = без сжатия.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression: Option<String>,
    /// Исходный (несжатый) размер — только при сжатии.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncompressed_size: Option<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaasManifest {
    pub version: String,
    pub created: String,
    /// Какая игра/сцена писала файл ("dioma", "arena", ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub game: Option<String>,
    pub sections: Vec<SectionEntry>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}
