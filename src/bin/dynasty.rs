//! -------------------------------------------------------------------------
//! Filename:    src/bin/dynasty.rs
//! Class:       SemantikaEngine
//! Purpose:     династия: характер (ген правила) наследуется и дрейфует
//! Description: фиксированная анатомия (agent_layout), меняются только
//!              гены правил. Поколение = мутация родительского гена.
//!              Подпись характера — минимальная сила стимула, при которой
//!              существо ещё решает, куда двигаться.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use dioma::brain::{gains_from, Brain, PlasticityParams};
use dioma::grow;
use dioma::pythia::{compete, PythiaUnit, RuleGene};
use dioma::rng::XorShift;

const SL: usize = 0;
const SR: usize = 1;
/// Регион-носитель «гена решения» в этом протоколе (ML).
const DECISION_GENE_REGION: usize = 3;

fn main() {
    println!("╔═══════════════════════════════════════════╗");
    println!("║  ДИНАСТИЯ: характер по наследству          ║");
    println!("╚═══════════════════════════════════════════╝");

    let grown = grow::grow(&grow::agent_layout(), &grow::Steering::default(), 777);
    let gains = gains_from(
        5,
        &[
            (SL, 3, 1.0),
            (SR, 4, 1.0),
            (SL, 2, 0.8),
            (SR, 2, 0.8),
            (2, 3, 1.0),
            (2, 4, 1.0),
        ],
    );
    let params = PlasticityParams::default();

    // Родоначальник: решительность средняя (порог 0.5).
    let mut gene = RuleGene::spiker(0.7, 0.5, 0.50, 2.0);
    let mut lineage_seed = 1000u64;

    println!(
        "  пок │ порог гена │ подпись (мин амп реш.) │ точн@2 │ точн@9"
    );
    for gen in 0..=3 {
        let mut brain = Brain::new(&grown, gains.clone(), params.clone());
        let signature = min_deciding_amplitude(&gene, &mut brain);
        let acc_low = accuracy_at(&gene, &mut brain, 2);
        let acc_high = accuracy_at(&gene, &mut brain, 9);

        println!(
            "  {gen:>3} │ {thr:>10.3} │ {sig:>22} │ {al:>4}/20 │ {ah:>4}/20",
            thr = gene.p[2],
            sig = format!("{signature:?}"),
            al = acc_low,
            ah = acc_high
        );

        // Наследник: ген родителя ± шум (та же формула, что spawn_child).
        lineage_seed = lineage_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut rng = XorShift::new(lineage_seed ^ (0x9E37_79B9 ^ DECISION_GENE_REGION as u64));
        gene = gene.mutate(&mut rng, dioma::heritage::MUTATE_RULE_SIGMA);
    }
    println!();
    println!("  Анатомия одна у всех — различие создают только гены правил.");
    println!("  Порог дрейфует поколениями: линия становится смелее или");
    println!("  осторожнее без всякого обучения — чистая наследственность.");
}

/// Минимальная амплитуда (из решётки), где существо решает верно ≥16/20.
fn min_deciding_amplitude(gene: &RuleGene, brain: &mut Brain) -> Option<usize> {
    [1usize, 2, 3, 4, 5, 7, 9]
        .into_iter()
        .find(|&amp| accuracy_at(gene, brain, amp) >= 16)
}

/// Точность выбора стороны на заданной амплитуде (движок решения — юнит
/// из 2 клеток с геном `gene`, входы — сенсорные всплески /350).
fn accuracy_at(gene: &RuleGene, brain: &mut Brain, amp: usize) -> u32 {
    let grown_n = brain.n;
    let mut rng = XorShift::new(20260824 + amp as u64);
    let spt = brain.steps_per_tick();
    let mut ok = 0u32;
    for trial in 0..20 {
        let true_left = trial % 2 == 0;
        let (sens, other) = if true_left { (SL, SR) } else { (SR, SL) };
        let mut ext = vec![0.12f32; grown_n];
        let (a, b) = brain.region_bounds(sens);
        for x in ext.iter_mut().take(b).skip(a) {
            *x = amp as f32 * 0.35 + (rng.unit() - 0.5) as f32 * 0.15;
        }
        let (oa, ob) = brain.region_bounds(other);
        for x in ext.iter_mut().take(ob).skip(oa) {
            *x = 0.12;
        }
        let mut sl = 0i64;
        let mut sr = 0i64;
        for _ in 0..spt {
            let sp = brain.step(&ext);
            sl += brain.region_spikes(&sp, SL) as i64;
            sr += brain.region_spikes(&sp, SR) as i64;
        }
        let scores = [sl as f32 / 350.0, sr as f32 / 350.0];
        let choice = decide(gene, &scores);
        let expect = if true_left { 0 } else { 1 };
        if choice == Some(expect) {
            ok += 1;
        }
    }
    ok
}

fn decide(gene: &RuleGene, scores: &[f32; 2]) -> Option<usize> {
    let mut u = PythiaUnit::from_gene(gene, 2, 5150);
    u.set_inputs(scores);
    let mut outs = [0.0f32; 2];
    for _ in 0..12 {
        let o = u.step();
        outs[0] += o[0];
        outs[1] += o[1];
    }
    compete(&outs, 0.05, 0.05)
}
