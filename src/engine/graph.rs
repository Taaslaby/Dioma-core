//! -------------------------------------------------------------------------
//! Filename:    src/plasticity.rs
//! Class:       neuworks
//! Purpose:     тропическая перестройка связей в течение жизни
//! Description: трофическое поле (EMA спайков) ведёт связи: слабые синапсы
//!              неактивных путей рассасываются, новые прорастают между
//!              активными клетками. Граф меняется ЖИВОЙ, а не только при
//!              рождении. Плюс синк весов Brain → Grown для пересборки.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
use crate::rng::XorShift;
use crate::Grown;
use std::collections::HashSet;

/// Настройки перестройки.
#[derive(Clone, Copy, Debug)]
pub struct RewireOpts {
    /// Сколько перемещений за один вызов.
    pub moves_per_call: usize,
    /// Синапсы слабее этого веса — кандидаты на рассасывание.
    pub weak_w: f32,
    /// Вес нового ростка.
    pub new_w: f32,
    /// Кап исходящих связей клетки (не превращать хабы в монстров).
    pub cap_out: usize,
    /// Кап входящих связей клетки.
    pub cap_in: usize,
}

impl Default for RewireOpts {
    fn default() -> Self {
        Self {
            moves_per_call: 6,
            weak_w: 0.35,
            new_w: 0.5,
            cap_out: 32,
            cap_in: 32,
        }
    }
}

/// Трофическое поле: EMA спайковой активности каждой клетки.
pub struct TropicField {
    ema: Vec<f32>,
    tick: usize,
    every: usize,
}

impl TropicField {
    /// `every` — период перестройки в тиках.
    pub fn new(n: usize, every: usize) -> Self {
        Self {
            ema: vec![0.0; n],
            tick: 0,
            every: every.max(1),
        }
    }

    /// Учёт спайков тика; true — пора перестраивать.
    pub fn observe(&mut self, spikes: &[i32]) -> bool {
        for (e, &s) in self.ema.iter_mut().zip(spikes) {
            *e += 0.05 * (s.min(3) as f32 - *e);
        }
        self.tick += 1;
        self.tick.is_multiple_of(self.every)
    }

    pub fn field(&self) -> &[f32] {
        &self.ema
    }
}

fn percentile(v: &[f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    let idx = ((s.len() as f32 - 1.0) * q)
        .round()
        .clamp(0.0, s.len() as f32 - 1.0) as usize;
    s[idx]
}

/// Отчёт о перестройке.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RewireReport {
    pub moved: usize,
}

/// Журнал переносов: индексы удалённых рёбер и пророщенные ростки.
pub struct RewireJournal {
    pub removals: Vec<usize>,
    pub sprouts: Vec<(usize, usize, f32)>,
}

/// Перестроить граф: слабые синапсы неактивных путей рассасываются, новые
/// ростки прорастают между активными клетками (трофический сигнал ≥ 75-го
/// процентили). Слабые = |w| ≤ weak_w И у обеих сторон останутся связи.
pub fn tropic_rewire(
    grown: &mut Grown,
    troph: &[f32],
    rng: &mut XorShift,
    opts: &RewireOpts,
) -> (RewireReport, RewireJournal) {
    let n = grown.nt.len();
    let thr = percentile(troph, 0.75).max(0.01);

    let mut out_deg = vec![0usize; n];
    let mut in_deg = vec![0usize; n];
    let mut pairs: HashSet<(usize, usize)> = HashSet::new();
    for &(pre, post, _) in &grown.edges {
        out_deg[pre] += 1;
        in_deg[post] += 1;
        pairs.insert((pre, post));
    }

    let mut weak: Vec<usize> = grown
        .edges
        .iter()
        .enumerate()
        .filter(|&(_, &(pre, post, w))| {
            w.abs() <= opts.weak_w
                && out_deg[pre] >= 2
                && in_deg[post] >= 2
                && troph[pre] < thr * 0.5
        })
        .map(|(ei, _)| ei)
        .collect();

    let active: Vec<usize> = (0..n).filter(|&i| troph[i] >= thr).collect();
    let mut journal = RewireJournal {
        removals: Vec::new(),
        sprouts: Vec::new(),
    };
    if active.len() < 2 {
        return (RewireReport { moved: 0 }, journal);
    }

    for _ in 0..opts.moves_per_call {
        if weak.is_empty() {
            break;
        }
        let pick = (rng.unit() * weak.len() as f64) as usize % weak.len();
        let ei = weak.swap_remove(pick);
        let (pre, post, _) = grown.edges[ei];

        // Степени могли просесть уже в этом вызове (другое слабое ребро с тем
        // же pre/post удалено раньше): перепроверяем актуально, иначе нейрон
        // останется без единой связи вопреки контракту «у обеих сторон останется
        // связь» (и в release дошло бы до wraparound usize).
        if out_deg[pre] < 2 || in_deg[post] < 2 {
            continue;
        }

        let a = active[(rng.unit() * active.len() as f64) as usize % active.len()];
        let b = active[(rng.unit() * active.len() as f64) as usize % active.len()];
        if a == b
            || pairs.contains(&(a, b))
            || out_deg[a] >= opts.cap_out
            || in_deg[b] >= opts.cap_in
        {
            continue;
        }

        out_deg[pre] -= 1;
        in_deg[post] -= 1;
        out_deg[a] += 1;
        in_deg[b] += 1;
        pairs.remove(&(pre, post));
        pairs.insert((a, b));
        journal.removals.push(ei);
        journal.sprouts.push((a, b, opts.new_w));
    }

    // Однопроходное применение: помечаем удалённые и пересобираем список.
    let remove_set: HashSet<usize> = journal.removals.iter().copied().collect();
    let sprouts = journal.sprouts.clone();
    let mut kept: Vec<(usize, usize, f32)> = Vec::with_capacity(grown.edges.len());
    for (ei, edge) in grown.edges.drain(..).enumerate() {
        if !remove_set.contains(&ei) {
            kept.push(edge);
        }
    }
    kept.extend(sprouts);
    grown.edges = kept;

    let report = RewireReport {
        moved: journal.removals.len(),
    };
    (report, journal)
}

/// Синк весов Brain → Grown: раскладывает значения CSR обратно по рёбрам
/// grown (порядок обхода идентичен построению CSR в `Brain::with_model`).
/// Веса делятся на гейн пары регионов (Brain строит w·gain).
pub fn sync_weights_from_brain(
    grown: &mut Grown,
    triplets: (&[usize], &[usize], &[f32]),
    gains: &[Vec<f32>],
) -> Result<(), String> {
    let (pre_of, post_of, vals) = triplets;
    let n = grown.nt.len();
    let mut per_pre: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ei, &(pre, post, _)) in grown.edges.iter().enumerate() {
        let g = gains[grown.region_of[pre]][grown.region_of[post]];
        if g > 0.0 {
            per_pre[pre].push(ei);
        }
    }
    let mut k = 0usize;
    for edges_of_pre in per_pre.iter() {
        for &ei in edges_of_pre {
            if k >= vals.len() {
                return Err(format!("CSR короче графа: k={k} > {}", vals.len()));
            }
            let (gp, gq, _) = grown.edges[ei];
            if gp != pre_of[k] || gq != post_of[k] {
                return Err(format!(
                    "расхождение порядка: ребро ({gp},{gq}) vs CSR ({},{})",
                    pre_of[k], post_of[k]
                ));
            }
            let gain = gains[grown.region_of[gp]][grown.region_of[gq]].max(1e-6);
            grown.edges[ei].2 = vals[k] / gain;
            k += 1;
        }
    }
    if k != vals.len() {
        return Err(format!("CSR длиннее графа: {k} vs {}", vals.len()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::XorShift;

    fn toy_grown() -> Grown {
        // 8 клеток, 2 региона по 4; цепочки S→I→M плюс слабые боковые.
        let edges = vec![
            (0, 4, 1.5),
            (1, 4, 1.2),
            (4, 6, 1.8),
            (5, 7, 1.4),
            (0, 1, 0.10), // слабая
            (1, 5, 0.20), // слабая
            (2, 6, 0.05), // слабая, тихий pre
            (3, 7, 1.0),
        ];
        Grown {
            edges,
            region_of: vec![0, 0, 0, 0, 1, 1, 1, 1],
            nt: vec![1.0; 8],
            growth_s: 0.0,
            rule_genes: Vec::new(),
            instincts: Vec::new(),
        }
    }

    #[test]
    fn rewire_moves_weak_edges_only() {
        let mut g = toy_grown();
        let strong_before = g.edges.iter().filter(|&&(_, _, w)| w > 0.5).count();
        // Трофический сигнал: активны клетки 0..4, тихие — остальные.
        let troph = vec![3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        let mut rng = XorShift::new(9);
        let (rep, j) = tropic_rewire(&mut g, &troph, &mut rng, &RewireOpts::default());
        let strong_after = g.edges.iter().filter(|&&(_, _, w)| w > 0.5).count();
        assert_eq!(
            strong_after, strong_before,
            "сильные связи не должны страдать"
        );
        assert!(
            j.sprouts.iter().all(|&(a, _, _)| troph[a] >= 3.0),
            "ростки — от активных"
        );
        assert!(rep.moved <= 6);
    }

    #[test]
    fn sync_roundtrip_preserves_weights() {
        let g0 = toy_grown();
        let gains = crate::engine::neurons::gains_from(2, &[(0, 1, 1.0)]);
        let mut brain = crate::engine::neurons::Brain::new(&g0, gains.clone(), Default::default());
        // Немного «поучились»: удвоим все веса.
        let (_, _, vals) = brain.csr_triplets();
        let doubled: Vec<f32> = vals.iter().map(|&v| v * 2.0).collect();
        brain.set_csr_values(&doubled).unwrap();

        let mut g = g0.clone();
        let t = brain.csr_triplets();
        sync_weights_from_brain(&mut g, (&t.0, &t.1, &t.2), &gains).unwrap();
        // Пересобранный мозг должен иметь те же веса.
        let brain2 = crate::engine::neurons::Brain::new(&g, gains, Default::default());
        let (_, _, v2) = brain2.csr_triplets();
        for (a, b) in doubled.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-4, "вес потерян: {a} vs {b}");
        }
    }

    #[test]
    fn quiet_field_blocks_rewire() {
        let mut g = toy_grown();
        let troph = vec![0.0; 8]; // всё тихо — активных нет
        let mut rng = XorShift::new(5);
        let (rep, _) = tropic_rewire(&mut g, &troph, &mut rng, &RewireOpts::default());
        assert_eq!(rep.moved, 0);
    }

    #[test]
    fn rewire_does_not_starve_shared_source() {
        // Клетка 0 имеет шесть слабых исходящих связей — все кандидаты на
        // рассасывание с общим источником. Без перепроверки актуальной степени
        // внутри цикла за один вызов удалились бы все шесть, и out_deg[0]
        // ушёл бы в 0 (в release — wraparound usize).
        let edges = vec![
            (0, 6, 0.10),
            (0, 7, 0.10),
            (0, 8, 0.10),
            (0, 9, 0.10),
            (0, 10, 0.10),
            (0, 11, 0.10),
            (1, 6, 1.0),
            (2, 7, 1.0),
            (3, 8, 1.0),
            (4, 9, 1.0),
            (5, 10, 1.0),
            (1, 11, 1.0),
        ];
        let make_grown = || Grown {
            edges: edges.clone(),
            region_of: vec![0; 12],
            nt: vec![1.0; 12],
            growth_s: 0.0,
            rule_genes: Vec::new(),
            instincts: Vec::new(),
        };
        // Трофика: активны 1..=5, тихие — 0 и посты 6..=11.
        let troph = vec![0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for seed in 0..64u64 {
            let mut g = make_grown();
            let mut rng = XorShift::new(seed);
            let _ = tropic_rewire(&mut g, &troph, &mut rng, &RewireOpts::default());

            let out0 = g.edges.iter().filter(|&&(pre, _, _)| pre == 0).count();
            assert!(
                out0 >= 1,
                "seed {seed}: источник 0 остался без исходящих связей"
            );
        }
    }
}
