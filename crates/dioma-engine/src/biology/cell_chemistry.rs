//! Orchestration between programmable cells, tissue and blood.
//!
//! Canonical flow: `biology::biocell` runs cell programs and emits
//! [`BioEvent`]s; this module owns how those events enter organism
//! chemistry (tissue compartment, then blood).

use crate::biology::metabolism::{chem, Blood};
use crate::biology::tissue::{SubstanceMap, TissueChemistry};
use crate::biology::biocell::BioCellRuntime;
use crate::biology::biovm::{BioEvent, BioEventKind, SubstanceId};

/// Starting substance bindings for creature physiology.
///
/// Extend as new secretory cell types appear; ids are part of the save
/// format once snapshots land, so treat them as stable.
pub fn creature_substance_map() -> SubstanceMap {
    let mut map = SubstanceMap::new(8);
    map.bind(SubstanceId(1), chem::SEROTONIN);
    map.bind(SubstanceId(2), chem::CORTISOL);
    map.bind(SubstanceId(3), chem::ACETYLCHOLINE);
    map
}

/// Run cells, deposit secretion in tissue, then transport it into blood.
pub fn step_into_blood(
    cells: &mut BioCellRuntime,
    tissue: &mut TissueChemistry,
    blood: &mut Blood,
    blood_volume: f32,
    dt: f32,
    map: &SubstanceMap,
) -> Vec<BioEvent> {
    let events = cells.step();
    tissue.absorb_events(&events);
    tissue.transport_to_blood(blood, blood_volume, dt, map);
    events
}

/// Run cells, absorb secretion into tissue, then exchange tissue and blood.
pub fn step_exchange(
    cells: &mut BioCellRuntime,
    tissue: &mut TissueChemistry,
    blood: &mut Blood,
    blood_volume: f32,
    dt: f32,
    map: &SubstanceMap,
) -> Vec<BioEvent> {
    let events = cells.step();
    tissue.absorb_events(&events);
    tissue.step_exchange(blood, blood_volume, dt, map);
    events
}

/// Direct injection retained for protocols that intentionally omit tissue.
pub fn inject_events_into_blood(blood: &mut Blood, events: &[BioEvent], map: &SubstanceMap) {
    for event in events {
        if event.kind != BioEventKind::Secrete {
            continue;
        }
        let Some(substance) = event.substance else {
            continue;
        };
        if let Some(chemical) = map.channel(substance) {
            blood.conc[chemical] = (blood.conc[chemical] + event.value).clamp(0.0, 255.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secretion_flows_cells_tissue_blood_before_biochemistry() {
        let mut cells = BioCellRuntime::secretory(2, 0.5, 2.0, SubstanceId(1));
        cells.set_signal(&[0.8, 1.0]);
        let mut tissue = TissueChemistry::new(2, 1.0, 0.5);
        let mut blood = Blood::creature(true);
        blood.conc[chem::SEROTONIN] = 0.0;
        let before = blood.conc[chem::SEROTONIN];
        let map = creature_substance_map();

        let events = step_into_blood(&mut cells, &mut tissue, &mut blood, 2.0, 1.0, &map);

        assert_eq!(events.len(), 2);
        assert!(blood.conc[chem::SEROTONIN] > before);
        assert!(tissue.concentration(SubstanceId(1)) > 0.0);
        blood.update();
        assert!(blood.conc[chem::SEROTONIN] > 0.0);
    }

    #[test]
    fn unbound_substances_are_ignored_on_direct_injection() {
        let mut blood = Blood::creature(true);
        blood.conc[chem::SEROTONIN] = 10.0;
        let map = SubstanceMap::new(2); // nothing bound

        inject_events_into_blood(
            &mut blood,
            &[BioEvent {
                cell: 0,
                kind: BioEventKind::Secrete,
                substance: Some(SubstanceId(1)),
                value: 5.0,
            }],
            &map,
        );

        assert!((blood.conc[chem::SEROTONIN] - 10.0).abs() < 1e-6);
    }
}
