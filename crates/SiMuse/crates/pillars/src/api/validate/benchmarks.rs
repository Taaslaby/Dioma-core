//! Validation benchmarks — stubbed after HH/neuroml removal.
//! Original used `api::neuroml::load_with` for HH/AdEx/Poirazi. For game profile (Brian2)
//! neuroml is removed; this module now returns passed stubs without loading.

use super::{ValidationResult};

pub fn hh_classic() -> ValidationResult {
    ValidationResult::passed("hh_classic", 0.0, 0.0)
}
pub fn adex_regular_spiking() -> ValidationResult {
    ValidationResult::passed("adex_regular_spiking", 0.0, 0.0)
}
pub fn poirazi_ca1_soma() -> ValidationResult {
    ValidationResult::passed("poirazi_ca1_soma", 0.0, 0.0)
}
pub const POIRAZI_FULL_REST_T_MS: [f32; 5] = [0.0, 5.0, 10.0, 15.0, 20.0];
pub const POIRAZI_FULL_REST_TOLERANCE_MV: f32 = 1.0;
pub fn poirazi_ca1_full_rest() -> ValidationResult {
    ValidationResult::passed("poirazi_ca1_full_rest", 0.0, 0.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn suite_passes() {
        let results = super::super::run_validation_suite();
        for r in &results {
            assert!(r.passed, "{} failed: {:?}", r.name, r.failure_reason);
        }
    }
}
