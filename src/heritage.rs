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
use taas_format::{TaasReader, TaasWriter};

/// Мутация CSR-весов наследника (относительная, на синапс).
pub const MUTATE_CSR_SIGMA: f32 = 0.05;
/// Мутация W2 наследника (абсолютная, на вес).
pub const MUTATE_W2_SIGMA: f32 = 0.02;

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
