//! -------------------------------------------------------------------------
//! Filename:    src/heritage.rs
//! Class:       DebugSuite
//! Purpose:     save/load creatures — heredity of learned minds
//! Description: рост детерминирован сидом, поэтому в файл пишется только
//!              ВЫУЧЕННОЕ: значения CSR коры (пластичность) и W2 разума
//!              (политика). Топология, layout и физиология восстанавливаются
//!              из сида. Наследник = пересоздать по родительскому сиду +
//!              перелить веса с мутацией.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use crate::organ_life::{OrganSim, SIM_TICKS};
use crate::Grown;
use taas_format::{TaasReader, TaasWriter};

/// Загрузить сардаана-коннектом как Grown — готовый для Brain/OrganSim.
///
/// sarDNA трансформер генерирует CSR (row_ptr, col_indices, values, nt).
/// Конвертируем в Grown: рёбра из тройков, регионы по степени (хабы=моторы).
/// Веса масштабируются в рабочий диапазон диомы (±2).
pub fn sardna_to_grown(
    path: impl AsRef<std::path::Path>,
) -> Result<Grown, String> {
    let r = TaasReader::from_file(path)?;

    let rp: Vec<i32> = r.get_i32("row_ptr")?;
    let ci: Vec<i32> = r.get_i32("col_indices")?;
    let raw_vals: Vec<f32> = r.get_f32("values")?;
    let nt_raw: Vec<f32> = r.get_f32("nt_types")?;

    let n = rp.len().saturating_sub(1);

    // Масштабируем веса: сардана даёт ±5, диома работает на ±2.
    let w_max = raw_vals.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if w_max > 0.0 { 2.0 / w_max } else { 1.0 };

    // Степени для назначения регионов.
    let degrees: Vec<usize> = rp.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let mut sorted_deg = degrees.clone();
    sorted_deg.sort_unstable_by(|a, b| b.cmp(a));
    let motor_thresh = sorted_deg[n * 20 / 100]; // топ-20% = моторы

    // Рёбра из CSR.
    let mut edges = Vec::with_capacity(ci.len());
    for i in 0..n {
        for j in rp[i] as usize..rp[i + 1] as usize {
            edges.push((
                i,
                ci[j] as usize,
                raw_vals[j] * scale,
            ));
        }
    }

    // Регионы: хабы (высокая степень) = регион 3 (моторы), остальные = регион 0.
    let region_of: Vec<usize> = degrees
        .iter()
        .map(|&d| if d >= motor_thresh { 3 } else { 0 })
        .collect();

    Ok(Grown {
        edges,
        region_of,
        nt: nt_raw.iter().map(|&v| v.clamp(-1.0, 1.0)).collect(),
        growth_s: 0.0,
        rule_genes: Vec::new(),
    })
}

/// Мутация CSR-весов наследника (относительная, на синапс).
pub const MUTATE_CSR_SIGMA: f32 = 0.05;
/// Мутация W2 наследника (абсолютная, на вес).
pub const MUTATE_W2_SIGMA: f32 = 0.02;
/// Мутация генов правил наследника (абсолютная, на параметр).
pub const MUTATE_RULE_SIGMA: f32 = 0.08;

/// Гены правил → колонки .taas (rg_template + rg_p0..p7).
fn write_rule_genes(w: &mut TaasWriter, genes: &[crate::pythia::RuleGene]) {
    if genes.is_empty() {
        return;
    }
    let tmpl: Vec<f32> = genes.iter().map(|g| g.template as f32).collect();
    w.add_f32("rg_template", "f32", &tmpl);
    for j in 0..8 {
        let col: Vec<f32> = genes.iter().map(|g| g.p[j]).collect();
        w.add_f32(&format!("rg_p{j}"), "f32", &col);
    }
}

/// Колонки .taas → гены правил (None если их нет в файле).
fn read_rule_genes(r: &TaasReader) -> Result<Option<Vec<crate::pythia::RuleGene>>, String> {
    let tmpl = match r.get_f32("rg_template") {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let mut cols: Vec<[f32; 8]> = Vec::new();
    for _ in &tmpl {
        cols.push([0.0; 8]);
    }
    for j in 0..8 {
        let c = r.get_f32(&format!("rg_p{j}"))?;
        for (i, slot) in cols.iter_mut().enumerate() {
            slot[j] = *c.get(i).unwrap_or(&0.0);
        }
    }
    let genes = tmpl
        .iter()
        .zip(&cols)
        .map(|(&t, p)| crate::pythia::RuleGene { template: t as u8, p: *p })
        .collect();
    Ok(Some(genes))
}

/// Сохранить существо: сид + выученные веса + счётчики.
pub fn save_creature(path: impl AsRef<std::path::Path>, sim: &OrganSim) -> Result<(), String> {
    let mut w = TaasWriter::new();
    w.add_metadata("kind", "dioma-organ-creature");
    w.add_metadata("format", "1");
    w.add_metadata("seed", &sim.seed.to_string());
    w.add_metadata("eaten", &sim.eaten_total().to_string());

    let csr = sim.organs.cortex.csr_values();
    w.add_f32("cortex_csr", "f32", &csr);

    if let Some(mind) = sim.mind.as_ref() {
        w.add_f32("w2", "f32", &mind.weight_heat());
    }

    write_rule_genes(&mut w, &sim.rule_genes);

    w.save_to_file(path)
}

/// Метаданные загруженного существа.
#[derive(Debug, Clone)]
pub struct CreatureMeta {
    pub seed: u64,
    pub eaten: usize,
    pub had_mind: bool,
}

/// Влить наследство в свежее существо (созданное тем же сидом):
/// переливает выученные CSR-веса и W2 разума.
pub fn load_creature_into(
    path: impl AsRef<std::path::Path>,
    sim: &mut OrganSim,
) -> Result<CreatureMeta, String> {
    let r = TaasReader::from_file(path)?;
    let seed: u64 = r
        .metadata()
        .get("seed")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "нет метаданных seed".to_string())?;
    let eaten: usize = r
        .metadata()
        .get("eaten")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let had_mind = r.get_f32("w2").is_ok();

    let csr = r.get_f32("cortex_csr")?;
    sim.organs
        .cortex
        .set_csr_values(&csr)
        .map_err(|e| format!("cortex CSR: {e}"))?;

    if let (Some(mind), Ok(w2)) = (&mut sim.mind, r.get_f32("w2")) {
        mind.set_weights(&w2);
    }

    if let Some(genes) = read_rule_genes(&r)? {
        sim.rule_genes = genes;
    }

    Ok(CreatureMeta { seed, eaten, had_mind })
}

/// Наследник: тело от сида родителя + унаследованные веса с мутацией +
/// собственный юный разум, растущий из родительской политики.
pub fn spawn_child(parent: &OrganSim, child_seed: u64, mutate: bool) -> OrganSim {
    let mut child = OrganSim::new(false, child_seed).with_mind(child_seed);

    // Выученная кора родителя → ребёнку (± мутация).
    let parent_csr = parent.organs.cortex.csr_values();
    let mut csr = child.organs.cortex.csr_values();
    let n = parent_csr.len().min(csr.len());
    for i in 0..n {
        csr[i] = if mutate {
            parent_csr[i] * (1.0 + MUTATE_CSR_SIGMA * gauss(child_seed, i as u64))
        } else {
            parent_csr[i]
        };
    }
    // Тот же сид → та же топология CSR → длины гарантированно совпадают.
    child.organs
        .cortex
        .set_csr_values(&csr)
        .expect("child CSR: same genome must give same topology");

    // Политика родителя → стартовые веса разума ребёнка (± мутация).
    if let (Some(pm), Some(cm)) = (&parent.mind, &mut child.mind) {
        let pw = pm.weight_heat();
        cm.set_weights_mutated(&pw, if mutate { MUTATE_W2_SIGMA } else { 0.0 }, child_seed ^ 0xC41D);
    }

    // Гены правил родителя → ребёнку (± мутация): динамика регионов
    // наследуется наравне с весами.
    if mutate && !parent.rule_genes.is_empty() {
        child.rule_genes = parent
            .rule_genes
            .iter()
            .enumerate()
            .map(|(ri, g)| {
                let mut rng = crate::rng::XorShift::new(child_seed ^ (0x9E37_79B9 ^ ri as u64));
                g.mutate(&mut rng, MUTATE_RULE_SIGMA)
            })
            .collect();
    } else {
        child.rule_genes = parent.rule_genes.clone();
    }

    child
}

/// Прогнать жизнь и вернуть тик первой еды (usize::MAX — не съел).
pub fn ticks_to_first_meal(sim: &mut OrganSim) -> usize {
    for _ in 0..SIM_TICKS {
        let obs = sim.step_tick();
        if obs.ate {
            return obs.tick;
        }
        if !obs.alive {
            return usize::MAX;
        }
    }
    usize::MAX
}

fn gauss(seed: u64, i: u64) -> f32 {
    // Детерминированный шум: хэш пары (seed, i) → [-1, 1).
    let x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(i.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let hi = ((x >> 40) & 0xFFFF) as f32 / 65536.0;
    hi * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pythia::{templates, RuleGene};

    #[test]
    fn rule_genes_roundtrip_through_taas() {
        let genes = vec![
            RuleGene::spiker(0.7, 0.5, 0.42, 2.0),
            RuleGene::default(),
            RuleGene::consolidator(0.9, 0.6),
        ];
        let path = std::env::temp_dir().join("dioma_rg_test.taas");
        let mut w = TaasWriter::new();
        w.add_f32("cortex_csr", "f32", &[1.0]);
        write_rule_genes(&mut w, &genes);
        w.save_to_file(&path).unwrap();

        let r = TaasReader::from_file(&path).unwrap();
        let back = read_rule_genes(&r).unwrap().expect("гены должны быть");
        assert_eq!(back.len(), genes.len());
        for (a, b) in back.iter().zip(&genes) {
            assert_eq!(a.template, b.template);
            assert!((a.p[2] - b.p[2]).abs() < 1e-6);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn no_genes_in_file_means_none() {
        let path = std::env::temp_dir().join("dioma_rg_none.taas");
        let mut w = TaasWriter::new();
        w.add_f32("cortex_csr", "f32", &[1.0]);
        w.save_to_file(&path).unwrap();
        let r = TaasReader::from_file(&path).unwrap();
        assert!(read_rule_genes(&r).unwrap().is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn child_mutates_rule_genes_only_params() {
        // Родитель без OrganSim: мутация — чистая функция гена+сида,
        // проверяем через RuleGene::mutate с тем же выводом, что spawn_child.
        let parent_genes = [RuleGene::spiker(0.7, 0.5, 0.5, 2.0); 5];
        let child_seed = 777u64;
        let ri = 3usize;
        let mut rng =
            crate::rng::XorShift::new(child_seed ^ (0x9E37_79B9 ^ ri as u64));
        let mutated = parent_genes[ri].mutate(&mut rng, MUTATE_RULE_SIGMA);
        assert_eq!(mutated.template, templates::SPIKER);
        assert_ne!(mutated.p, parent_genes[ri].p);
        // Все параметры в допустимом диапазоне.
        assert!(mutated.p.iter().all(|&v| (0.0..=2.0).contains(&v)));
    }
}
