//! Человекочитаемый дамп `.taas` v2: манифест + секции + метаданные.

use crate::reader::TaasReader;

pub fn inspect_taas(path: &str) -> Result<String, String> {
    let reader = TaasReader::from_file(path)?;
    let mut lines = Vec::new();

    lines.push(format!(".taas archive: {path}"));
    lines.push(format!("   Version: {}", reader.manifest.version));
    lines.push(format!("   Created: {}", reader.manifest.created));
    if let Some(game) = &reader.manifest.game {
        lines.push(format!("   Game:    {game}"));
    }
    lines.push(format!("   Sections: {}", reader.manifest.sections.len()));
    lines.push(String::new());

    for sec in &reader.manifest.sections {
        let count_str = match sec.compression.as_deref() {
            Some("zstd") => format!(
                "{} (uncompressed {})",
                human_size(sec.size),
                human_size(sec.uncompressed_size.unwrap_or(0))
            ),
            _ => human_size(sec.size),
        };
        let crc = match sec.crc32 {
            Some(c) => format!("{c:08x}"),
            None => "-".to_string(),
        };
        lines.push(format!(
            "   {:<24} {:<5} {:>10}  crc={}  offset={}",
            sec.name, sec.kind, count_str, crc, sec.offset
        ));
        lines.push(format!(
            "   {:>29} count={}",
            "",
            sec.count
        ));
    }

    if !reader.manifest.metadata.is_empty() {
        lines.push(String::new());
        lines.push("   Metadata:".to_string());
        for (k, v) in &reader.manifest.metadata {
            lines.push(format!("      {:<20} = {v}", k));
        }
    }

    lines.push(String::new());
    lines.push(format!("   Total size: {}", human_size(reader.total_size() as u64)));

    Ok(lines.join("\n"))
}

fn human_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}
