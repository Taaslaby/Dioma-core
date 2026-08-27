//! Стриминговый лог реплея — append-only запись событий мира по тикам.
//!
//! Формат:
//! ```text
//! MAGIC "REPL2" (5) + VERSION u32 LE
//! FRAME*  — каждый кадр:
//!   LEN u32 LE      — длина payload
//!   CRC u32 LE      — CRC32(payload)
//!   PAYLOAD         — JSON: {"tick": u64, "kind": str, "data": {...}}
//! ```
//!
//! Реплей пишется потоково (мир тикает — события дописываются), читается
//! последовательно. Кадр самодостаточен: битый кадр можно пропустить, не
//! теряя остальной лог.

use serde::Serialize;
use std::io::{Read, Write};
use std::path::Path;

pub const REPLAY_MAGIC: &[u8; 5] = b"REPL2";
pub const REPLAY_VERSION: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplayEvent {
    pub tick: u64,
    pub kind: String,
    #[serde(default)]
    pub data: serde_json::Value,
}

fn frame_bytes(event: &ReplayEvent) -> Result<Vec<u8>, String> {
    let payload = serde_json::to_vec(event).map_err(|e| format!("event JSON: {e}"))?;
    let crc = crate::writer::compute_crc32(&payload);
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

fn parse_frame(buf: &[u8], index: usize) -> Result<(ReplayEvent, usize), String> {
    if buf.len() < 8 {
        return Err(format!("frame {index}: truncated header"));
    }
    let len = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let crc = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    if 8 + len > buf.len() {
        return Err(format!("frame {index}: truncated payload (len {len})"));
    }
    let payload = &buf[8..8 + len];
    let actual = crate::writer::compute_crc32(payload);
    if actual != crc {
        return Err(format!(
            "frame {index}: CRC32 mismatch (expected {crc:08x}, got {actual:08x})"
        ));
    }
    let event: ReplayEvent =
        serde_json::from_slice(payload).map_err(|e| format!("frame {index}: JSON: {e}"))?;
    Ok((event, 8 + len))
}

/// Писатель реплея. Пишет в Vec (память) или в файл (поток).
pub struct ReplayWriter {
    bytes: Vec<u8>,
    file: Option<std::fs::File>,
    events: u64,
}

impl ReplayWriter {
    pub fn new() -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(REPLAY_MAGIC);
        bytes.extend_from_slice(&REPLAY_VERSION.to_le_bytes());
        Self {
            bytes,
            file: None,
            events: 0,
        }
    }

    pub fn to_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let mut w = Self::new();
        let file = std::fs::File::create(path.as_ref())
            .map_err(|e| format!("create replay: {e}"))?;
        let mut f = std::io::BufWriter::new(file);
        f.write_all(&w.bytes)
            .map_err(|e| format!("write replay header: {e}"))?;
        w.file = Some(
            f.into_inner()
                .map_err(|e| format!("flush replay header: {e}"))?,
        );
        Ok(w)
    }

    /// Добавить событие. `data` — любое сериализуемое значение.
    pub fn push<T: Serialize>(&mut self, tick: u64, kind: &str, data: &T) -> Result<(), String> {
        let data = serde_json::to_value(data).map_err(|e| format!("event data: {e}"))?;
        let ev = ReplayEvent {
            tick,
            kind: kind.to_string(),
            data,
        };
        let frame = frame_bytes(&ev)?;
        if let Some(f) = &mut self.file {
            f.write_all(&frame)
                .map_err(|e| format!("write replay frame: {e}"))?;
        } else {
            self.bytes.extend_from_slice(&frame);
        }
        self.events += 1;
        Ok(())
    }

    /// Сбросить буфер в файл (для потокового писателя).
    pub fn flush(&mut self) -> Result<(), String> {
        if let Some(f) = &mut self.file {
            f.flush().map_err(|e| format!("flush replay: {e}"))
        } else {
            Ok(())
        }
    }

    pub fn num_events(&self) -> u64 {
        self.events
    }

    /// Полные байты реплея (только для памяти-режима).
    pub fn into_bytes(self) -> Result<Vec<u8>, String> {
        if self.file.is_some() {
            return Err("replay is file-backed; use flush() instead".to_string());
        }
        Ok(self.bytes)
    }
}

impl Default for ReplayWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Читатель реплея: последовательный обход кадров.
pub struct ReplayReader {
    bytes: Vec<u8>,
    pos: usize,
    version: u32,
}

impl ReplayReader {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if !bytes.starts_with(REPLAY_MAGIC) {
            return Err(format!("not a replay: magic {:?}", &bytes[..bytes.len().min(5)]));
        }
        let version = u32::from_le_bytes(bytes[5..9].try_into().unwrap());
        Ok(Self {
            bytes: bytes.to_vec(),
            pos: 9,
            version,
        })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let mut buf = Vec::new();
        let mut f = std::fs::File::open(path.as_ref())
            .map_err(|e| format!("open replay: {e}"))?;
        f.read_to_end(&mut buf).map_err(|e| format!("read replay: {e}"))?;
        Self::from_bytes(&buf)
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    /// Следующий кадр; `None` — конец лога.
    pub fn next_event(&mut self) -> Result<Option<ReplayEvent>, String> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let (ev, consumed) = parse_frame(&self.bytes[self.pos..], self.frame_index())?;
        self.pos += consumed;
        Ok(Some(ev))
    }

    fn frame_index(&self) -> usize {
        // Приблизительный индекс кадра для сообщений об ошибке.
        self.pos / 16
    }

    /// Собрать все события (удобно для коротких логов).
    pub fn collect(&mut self) -> Result<Vec<ReplayEvent>, String> {
        let mut out = Vec::new();
        while let Some(ev) = self.next_event()? {
            out.push(ev);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_roundtrip() {
        let mut w = ReplayWriter::new();
        w.push(0, "spawn", &serde_json::json!({"x": 1.0, "y": 2.0})).unwrap();
        w.push(10, "eat", &serde_json::json!({"food": 3})).unwrap();
        w.push(10, "die", &serde_json::json!({"reason": "hunger"})).unwrap();
        let bytes = w.into_bytes().unwrap();

        let mut r = ReplayReader::from_bytes(&bytes).unwrap();
        assert_eq!(r.version(), REPLAY_VERSION);
        let events = r.collect().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].tick, 0);
        assert_eq!(events[0].kind, "spawn");
        assert_eq!(events[0].data["x"], 1.0);
        assert_eq!(events[1].tick, 10);
        assert_eq!(events[1].kind, "eat");
        assert_eq!(events[2].kind, "die");
        assert_eq!(events[2].data["reason"], "hunger");
    }

    #[test]
    fn test_replay_corruption_detected() {
        let mut w = ReplayWriter::new();
        w.push(0, "spawn", &serde_json::json!({"x": 1})).unwrap();
        w.push(1, "step", &serde_json::json!({"n": 2})).unwrap();
        let mut bytes = w.into_bytes().unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x40;
        let mut r = ReplayReader::from_bytes(&bytes).unwrap();
        // Первый или второй кадр должен отдаться с ошибкой CRC — но не паникой.
        let mut saw_error = false;
        loop {
            match r.next_event() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(_) => {
                    saw_error = true;
                    break;
                }
            }
        }
        assert!(saw_error, "corrupted replay must error, not silently pass");
    }

    #[test]
    fn test_replay_bad_magic() {
        assert!(ReplayReader::from_bytes(b"XXXX\x00").is_err());
    }

    #[test]
    fn test_replay_file_roundtrip() {
        let path = "test_replay_file.rep";
        let mut w = ReplayWriter::to_file(path).unwrap();
        w.push(5, "step", &serde_json::json!({"t": 5})).unwrap();
        w.flush().unwrap();
        drop(w);

        let mut r = ReplayReader::from_file(path).unwrap();
        let evs = r.collect().unwrap();
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].tick, 5);
        std::fs::remove_file(path).ok();
    }
}
