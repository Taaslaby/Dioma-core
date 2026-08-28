//! Local tissue chemistry between cell secretion and systemic blood.

use crate::biology::metabolism::Blood;
use crate::biology::biovm::{BioEvent, BioEventKind, SubstanceId};

/// Canonical mapping from neutral BioVM substance ids to Blood channels.
///
/// The engine layer speaks abstract [`SubstanceId`]s; only the body layer
/// knows which chemical channel each substance feeds.
#[derive(Clone, Debug, Default)]
pub struct SubstanceMap {
    channels: Vec<Option<usize>>,
}

impl SubstanceMap {
    pub fn new(n_substances: usize) -> Self {
        Self {
            channels: vec![None; n_substances],
        }
    }

    pub fn bind(&mut self, substance: SubstanceId, chemical: usize) {
        let index = substance.0 as usize;
        if index >= self.channels.len() {
            self.channels.resize(index + 1, None);
        }
        self.channels[index] = Some(chemical);
    }

    pub fn channel(&self, substance: SubstanceId) -> Option<usize> {
        self.channels.get(substance.0 as usize).copied().flatten()
    }
}

/// Well-mixed local chemical compartment.
///
/// Cell events add an amount of substance to the tissue. Transport moves it
/// into blood down a concentration gradient while preserving total amount
/// across the two configured volumes.
#[derive(Clone, Debug)]
pub struct TissueChemistry {
    /// Concentration per substance in the local tissue compartment.
    concentration: Vec<f32>,
    /// Tissue volume in arbitrary but consistent simulation units.
    volume: f32,
    /// Exchange rate between tissue and blood.
    permeability: f32,
    /// Fraction remaining after one unit of simulation time.
    decay: Vec<f32>,
}

impl TissueChemistry {
    pub fn new(n_substances: usize, volume: f32, permeability: f32) -> Self {
        assert!(volume > 0.0);
        assert!(permeability >= 0.0);
        Self {
            concentration: vec![0.0; n_substances],
            volume,
            permeability,
            decay: vec![1.0; n_substances],
        }
    }

    /// Set the fraction remaining after one unit of simulation time.
    pub fn set_decay_factor(&mut self, substance: SubstanceId, factor: f32) {
        if let Some(decay) = self.decay.get_mut(substance.0 as usize) {
            *decay = factor.clamp(0.0, 1.0);
        }
    }

    /// Apply local first-order decay for one time step.
    pub fn decay(&mut self, dt: f32) {
        assert!(dt >= 0.0);
        for (concentration, &factor) in self.concentration.iter_mut().zip(&self.decay) {
            *concentration *= factor.powf(dt);
        }
    }

    pub fn concentration(&self, substance: SubstanceId) -> f32 {
        self.concentration
            .get(substance.0 as usize)
            .copied()
            .unwrap_or(0.0)
    }

    /// Add secretion events to the local tissue compartment.
    pub fn absorb_events(&mut self, events: &[BioEvent]) {
        for event in events {
            if event.kind != BioEventKind::Secrete {
                continue;
            }
            let Some(substance) = event.substance else {
                continue;
            };
            if let Some(concentration) = self.concentration.get_mut(substance.0 as usize) {
                *concentration += event.value.max(0.0) / self.volume;
            }
        }
    }

    /// Move substances into blood for one time step.
    ///
    /// `Blood::conc` is treated as concentration in a compartment of
    /// `blood_volume`. The returned value is the total transported amount.
    pub fn transport_to_blood(
        &mut self,
        blood: &mut Blood,
        blood_volume: f32,
        dt: f32,
        map: &SubstanceMap,
    ) -> f32 {
        assert!(blood_volume > 0.0);
        assert!(dt >= 0.0);
        let mut transported = 0.0;

        for (index, tissue_concentration) in self.concentration.iter_mut().enumerate() {
            let substance = SubstanceId(index as u16);
            let Some(chemical) = map.channel(substance) else {
                continue;
            };
            let Some(blood_concentration) = blood.conc.get_mut(chemical) else {
                continue;
            };

            let gradient = (*tissue_concentration - *blood_concentration).max(0.0);
            let available = *tissue_concentration * self.volume;
            let blood_capacity = (255.0 - *blood_concentration).max(0.0) * blood_volume;
            let amount = (self.permeability * gradient * dt * self.volume)
                .min(available)
                .min(blood_capacity);
            *tissue_concentration -= amount / self.volume;
            *blood_concentration += amount / blood_volume;
            transported += amount;
        }

        transported
    }

    /// Apply decay and exchange each mapped substance in both directions.
    ///
    /// The returned value is the total absolute amount exchanged. Unlike
    /// `transport_to_blood`, this method also permits blood to replenish a
    /// depleted tissue compartment.
    pub fn step_exchange(
        &mut self,
        blood: &mut Blood,
        blood_volume: f32,
        dt: f32,
        map: &SubstanceMap,
    ) -> f32 {
        assert!(blood_volume > 0.0);
        assert!(dt >= 0.0);
        self.decay(dt);
        let mut exchanged = 0.0;

        for (index, tissue_concentration) in self.concentration.iter_mut().enumerate() {
            let substance = SubstanceId(index as u16);
            let Some(chemical) = map.channel(substance) else {
                continue;
            };
            let Some(blood_concentration) = blood.conc.get_mut(chemical) else {
                continue;
            };
            let gradient = *tissue_concentration - *blood_concentration;
            let amount = (self.permeability * gradient.abs() * dt * self.volume)
                .min(if gradient >= 0.0 {
                    *tissue_concentration * self.volume
                } else {
                    *blood_concentration * blood_volume
                })
                .min(if gradient >= 0.0 {
                    (255.0 - *blood_concentration).max(0.0) * blood_volume
                } else {
                    f32::INFINITY
                });

            if gradient >= 0.0 {
                *tissue_concentration -= amount / self.volume;
                *blood_concentration += amount / blood_volume;
            } else {
                *tissue_concentration += amount / self.volume;
                *blood_concentration -= amount / blood_volume;
            }
            exchanged += amount;
        }
        exchanged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cortisol_map() -> SubstanceMap {
        let mut map = SubstanceMap::new(2);
        map.bind(SubstanceId(0), crate::biology::metabolism::chem::CORTISOL);
        map
    }

    #[test]
    fn transport_preserves_amount_across_compartments() {
        let substance = SubstanceId(0);
        let mut tissue = TissueChemistry::new(1, 2.0, 0.25);
        tissue.concentration[0] = 4.0;
        let mut blood = Blood::creature(true);
        blood.conc[crate::biology::metabolism::chem::CORTISOL] = 0.0;
        let before = tissue.concentration(substance) * 2.0;

        let moved = tissue.transport_to_blood(&mut blood, 4.0, 1.0, &cortisol_map());
        let after = tissue.concentration(substance) * 2.0
            + blood.conc[crate::biology::metabolism::chem::CORTISOL] * 4.0;

        assert!((moved - 2.0).abs() < 1e-6);
        assert!((before - after).abs() < 1e-6);
    }

    #[test]
    fn saturated_blood_leaves_excess_in_tissue() {
        let mut tissue = TissueChemistry::new(1, 1.0, 1.0);
        tissue.concentration[0] = 10.0;
        let mut blood = Blood::creature(true);
        blood.conc[crate::biology::metabolism::chem::CORTISOL] = 255.0;

        let moved = tissue.transport_to_blood(&mut blood, 1.0, 1.0, &cortisol_map());

        assert_eq!(moved, 0.0);
        assert_eq!(tissue.concentration(SubstanceId(0)), 10.0);
    }

    #[test]
    fn exchange_replenishes_tissue_from_blood() {
        let mut tissue = TissueChemistry::new(1, 1.0, 0.5);
        let mut blood = Blood::creature(true);
        blood.conc[crate::biology::metabolism::chem::CORTISOL] = 4.0;

        let moved = tissue.step_exchange(&mut blood, 2.0, 1.0, &cortisol_map());

        assert!((moved - 2.0).abs() < 1e-6);
        assert!((tissue.concentration(SubstanceId(0)) - 2.0).abs() < 1e-6);
        assert!((blood.conc[crate::biology::metabolism::chem::CORTISOL] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn tissue_decay_is_applied_before_exchange() {
        let mut tissue = TissueChemistry::new(1, 1.0, 0.0);
        tissue.concentration[0] = 8.0;
        tissue.set_decay_factor(SubstanceId(0), 0.25);

        tissue.decay(1.0);

        assert!((tissue.concentration(SubstanceId(0)) - 2.0).abs() < 1e-6);
    }
}
