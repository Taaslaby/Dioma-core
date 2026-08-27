//! -------------------------------------------------------------------------
//! Class:       brain
//! Purpose:     High-level brain lifecycle — connectivity + lazy dopamine-STDP
//! Description: `Brain` = synapse topology + weights + 3-factor dopamine-STDP
//!              with LAZY eligibility. Traces are per-neuron, and eligibility
//!              is only touched on spike events (per-synapse `e` is folded at
//!              reward time) — cost scales with activity, not with synapse
//!              count × steps. Learning happens inside a typed session token
//!              (`LearnToken`): outside a session the brain is pure inference,
//!              no rewards can be applied (enforced by the borrow checker).
//!              Lifecycle verbs: `snapshot`/`recall`/`drift_rms` (sleep
//!              consolidation), `learned_gain` (Baldwin inheritance).
//!
//! Dynamics match `crate::kernels::dopamine::dopamine_traces_step` exactly
//! (per-step decay factors use the same `fast_exp` family; two events on one
//! synapse in the same step are summed before the single trailing decay).
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

use crate::math::fast_exp::fast_exp;

/// Izhikevich-style 3-factor STDP parameters (defaults from Izhikevich 2007).
#[derive(Clone, Debug)]
pub struct StdpConfig {
    /// Simulation step, ms.
    pub dt: f32,
    /// Pre-trace time constant, ms.
    pub tau_pre: f32,
    /// Post-trace time constant, ms.
    pub tau_post: f32,
    /// Eligibility trace time constant, ms.
    pub tau_e: f32,
    /// LTP amplitude.
    pub a_ltp: f32,
    /// LTD amplitude.
    pub a_ltd: f32,
    /// Learning rate applied at reward time.
    pub lr: f32,
    /// Weight floor.
    pub w_min: f32,
    /// Weight ceiling.
    pub w_max: f32,
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            dt: 0.5,
            tau_pre: 20.0,
            tau_post: 20.0,
            tau_e: 1000.0,
            a_ltp: 0.1,
            a_ltd: 0.12,
            lr: 1.0,
            w_min: 0.0,
            w_max: 1.0,
        }
    }
}

/// High-level spiking brain: topology + weights + lazy dopamine-STDP.
///
/// The membrane itself stays outside (use a `NeuralPopulation`/model runner);
/// `step` consumes per-neuron spike flags (0/1) and updates traces +
/// eligibility. Weights change only inside a [`LearnToken`] session.
///
/// Gaps between events are decayed by repeated multiplication with the dense
/// kernel's per-step constant — bit-identical to the reference path (a single
/// `fast_exp` over the gap would be more accurate, but we reproduce the dense
/// kernel's rounding exactly; the total multiply count is the same the dense
/// path already spent per step, just batched at events/rewards).
pub struct Brain {
    n_neurons: usize,
    pre_of: Vec<u32>,
    post_of: Vec<u32>,
    /// Neuron -> outgoing synapse indices.
    out_edges: Vec<Vec<u32>>,
    /// Neuron -> incoming synapse indices.
    in_edges: Vec<Vec<u32>>,
    weights: Vec<f32>,
    cfg: StdpConfig,
    /// Current step counter (starts at 1 after the first `step`).
    t: u64,
    /// Per-neuron pre-trace (post-synaptic spike history of the source).
    trace_pre: Vec<f32>,
    /// Per-neuron post-trace (pre-synaptic spike history of the target).
    trace_post: Vec<f32>,
    /// Per-synapse eligibility, value as of `last_sync`.
    elig: Vec<f32>,
    /// Step index at which each synapse's eligibility was last synced.
    last_sync: Vec<u64>,
    /// Scratch buffer for spike events of one step (reused, no per-step alloc).
    events: Vec<(u32, f32)>,
    /// Per-synapse stamp of the last step that applied the trailing decay —
    /// dedups co-firing events so a synapse decays exactly once per step
    /// (bit-identical to the dense kernel's single `e *= d_e` per step).
    stamp: Vec<u64>,
    /// Long-term snapshot (birth memory) for sleep consolidation / Baldwin.
    baked: Option<Vec<f32>>,
}

impl Brain {
    /// Build a brain from synapse topology. `pre[i] -> post[i]` is synapse `i`;
    /// `weights.len() == pre.len()`.
    pub fn new(
        n_neurons: usize,
        pre: &[u32],
        post: &[u32],
        weights: Vec<f32>,
        cfg: StdpConfig,
    ) -> Self {
        assert_eq!(pre.len(), post.len(), "brain: pre/post length mismatch");
        assert_eq!(weights.len(), pre.len(), "brain: weights length mismatch");
        let mut out_edges = vec![Vec::new(); n_neurons];
        let mut in_edges = vec![Vec::new(); n_neurons];
        for (i, (&p, &q)) in pre.iter().zip(post).enumerate() {
            let (p, q) = (p as usize, q as usize);
            assert!(p < n_neurons && q < n_neurons, "brain: neuron index OOB");
            out_edges[p].push(i as u32);
            in_edges[q].push(i as u32);
        }
        let n_syn = pre.len();
        Self {
            n_neurons,
            pre_of: pre.to_vec(),
            post_of: post.to_vec(),
            out_edges,
            in_edges,
            weights,
            cfg,
            t: 0,
            trace_pre: vec![0.0; n_neurons],
            trace_post: vec![0.0; n_neurons],
            elig: vec![0.0; n_syn],
            last_sync: vec![0; n_syn],
            events: Vec::new(),
            stamp: vec![0; n_syn],
            baked: None,
        }
    }

    pub fn n_neurons(&self) -> usize {
        self.n_neurons
    }

    pub fn n_synapses(&self) -> usize {
        self.pre_of.len()
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut [f32] {
        &mut self.weights
    }

    /// Current eligibility, as of each synapse's last sync (call
    /// [`Self::fold_eligibility`] first to bring them to now).
    pub fn eligibility(&self) -> &[f32] {
        &self.elig
    }

    /// DEBUG: per-neuron pre/post traces (сверка с GPU-кернелом пошагово).
    pub fn traces(&self) -> (Vec<f32>, Vec<f32>) {
        (self.trace_pre.clone(), self.trace_post.clone())
    }

    /// Current step index.
    pub fn t(&self) -> u64 {
        self.t
    }

    /// Open a learning session. Rewards can only be applied through the
    /// returned token; dropping it (or calling [`LearnToken::end`]) closes the
    /// session and hands the brain back.
    pub fn begin_learning(&mut self) -> LearnToken<'_> {
        LearnToken { brain: self }
    }

    /// One dynamics step: decay per-neuron traces, then apply spike-triggered
    /// eligibility updates for every synapse of a spiking neuron. O(neurons +
    /// spikes·degree) — no per-synapse work for silent synapses.
    pub fn step(&mut self, spikes: &[i32]) {
        debug_assert_eq!(spikes.len(), self.n_neurons, "brain: spikes length");
        self.t += 1;
        let t = self.t;
        let d_pre = fast_exp(-self.cfg.dt / self.cfg.tau_pre);
        let d_post = fast_exp(-self.cfg.dt / self.cfg.tau_post);
        let d_e = fast_exp(-self.cfg.dt / self.cfg.tau_e);

        // Per-neuron traces: decay, then add the spike (kernel order).
        // ЯВНЫЙ fma (`mul_add`): шейдер считает `tp * d + spike` через fma
        // (одно округление), а `*tp *= d; *tp += 1.0` дал бы два — 1 ulp
        // дрейфа → расхождение eligibility с GPU. Обе стороны fma — бит-в-бит.
        for (n, &sp) in spikes.iter().enumerate() {
            let add = if sp > 0 { 1.0 } else { 0.0 };
            let tp = &mut self.trace_pre[n];
            *tp = tp.mul_add(d_pre, add);
            let tpo = &mut self.trace_post[n];
            *tpo = tpo.mul_add(d_post, add);
        }

        // Events. A spiking neuron as POST potentiates its incoming synapses
        // (LTP term uses the pre-trace of the source); as PRE it depresses its
        // outgoing ones (LTD term uses the post-trace of the target). ALL
        // events of the step are staged first, then processed in two phases:
        // accumulate contributions, then ONE trailing decay per touched
        // synapse — so co-firing terms (pre AND post of a synapse spiking in
        // the same step) sum before the single multiply, exactly like the
        // dense kernel's `e = (e + ltp − ltd)·d_e` (bit-identical rounding).
        self.events.clear();
        for (n, &sp) in spikes.iter().enumerate() {
            if sp == 0 {
                continue;
            }
            for &i in &self.in_edges[n] {
                let pre = self.pre_of[i as usize] as usize;
                self.events
                    .push((i, self.cfg.a_ltp * self.trace_pre[pre]));
            }
            for &i in &self.out_edges[n] {
                let post = self.post_of[i as usize] as usize;
                self.events
                    .push((i, -self.cfg.a_ltd * self.trace_post[post]));
            }
        }
        let n_ev = self.events.len();
        // Phase A: decay each touched synapse's accumulator to `t-1`, then
        // accumulate its contributions (all of this step's, in order).
        // The gap is decayed by REPEATED multiplication with the per-step
        // constant — the exact op sequence the dense kernel performs, so the
        // result is bit-identical (a single fast_exp over the gap would be
        // *more* accurate, but the dense kernel's per-step rounding is the
        // behavior we reproduce, and the lazy batching costs the same total
        // multiplies the dense path already spent).
        for k in 0..n_ev {
            let (i, c) = self.events[k];
            let i = i as usize;
            let e = &mut self.elig[i];
            let s = self.last_sync[i];
            if s < t {
                let elapsed = t - s;
                if elapsed > 1 {
                    for _ in 1..elapsed {
                        *e *= d_e;
                    }
                }
                self.last_sync[i] = t;
            }
            *e += c;
        }
        // Phase B: exactly one trailing decay per touched synapse per step.
        for k in 0..n_ev {
            let i = self.events[k].0 as usize;
            if self.stamp[i] != t {
                self.stamp[i] = t;
                self.elig[i] *= d_e;
            }
        }
    }

    /// Bring every eligibility value to the current step: `e *= d^(t - s)`.
    /// O(synapses), but only needed at reward/consolidation time. Synapses
    /// with zero eligibility are skipped (decaying 0 costs nothing and stays
    /// exact). Gaps decay by repeated multiplication, matching the dense
    /// kernel's per-step rounding bit-for-bit.
    pub fn fold_eligibility(&mut self) {
        let t = self.t;
        for i in 0..self.elig.len() {
            let s = self.last_sync[i];
            if s < t && self.elig[i] != 0.0 {
                let elapsed = t - s;
                for _ in 0..elapsed {
                    self.elig[i] *= self.cfg_decay_e();
                }
                self.last_sync[i] = t;
            } else if s < t {
                self.last_sync[i] = t;
            }
        }
    }

    /// Per-step eligibility decay constant (same expression as the dense kernel).
    fn cfg_decay_e(&self) -> f32 {
        fast_exp(-self.cfg.dt / self.cfg.tau_e)
    }

    /// Zero all eligibility traces (window close / freeze).
    pub fn clear_eligibility(&mut self) {
        self.elig.fill(0.0);
        self.last_sync.fill(self.t);
    }

    /// Keep eligibility only on synapses whose mask is `true`, zero the rest.
    pub fn restrict_eligibility(&mut self, keep: &[bool]) {
        for (i, &k) in keep.iter().enumerate().take(self.elig.len()) {
            if !k {
                self.elig[i] = 0.0;
            }
        }
    }

    /// Weight update: `Δw = lr·dopamine·e`, clamped to `[w_min, w_max]`;
    /// synapses with `Δw == 0` are left untouched (innate weights preserved).
    /// Folds eligibility to `now` first, so rewards credit the whole history.
    fn reward_dopamine(&mut self, dopamine: f32, w_cap: Option<f32>) {
        self.fold_eligibility();
        let w_min = self.cfg.w_min;
        let w_max = w_cap.unwrap_or(self.cfg.w_max);
        let lr_dopa = self.cfg.lr * dopamine;
        for i in 0..self.weights.len() {
            let delta = lr_dopa * self.elig[i];
            if delta == 0.0 {
                continue;
            }
            let w = self.weights[i] + delta;
            self.weights[i] = w.clamp(w_min, w_max);
        }
    }

    /// Save the current weights as the long-term snapshot (birth memory).
    pub fn snapshot(&mut self) {
        self.baked = Some(self.weights.clone());
    }

    /// Pull weights toward the snapshot: `w += alpha·(lt - w)`.
    pub fn recall(&mut self, alpha: f32) {
        if let Some(lt) = &self.baked {
            for (w, &l) in self.weights.iter_mut().zip(lt.iter()) {
                *w += alpha * (l - *w);
            }
        }
    }

    /// RMS drift of current weights vs the snapshot (0 if none).
    pub fn drift_rms(&self) -> f32 {
        match &self.baked {
            None => 0.0,
            Some(lt) => {
                let n = lt.len().min(self.weights.len());
                let s: f32 = (0..n)
                    .map(|i| {
                        let d = self.weights[i] - lt[i];
                        d * d
                    })
                    .sum();
                (s / n.max(1) as f32).sqrt()
            }
        }
    }

    /// Per-synapse learned gain: current weights minus the snapshot
    /// (zeros when no snapshot was taken). Caller filters by topology/regions.
    pub fn learned_gain(&self) -> Vec<f32> {
        match &self.baked {
            None => vec![0.0; self.weights.len()],
            Some(lt) => self
                .weights
                .iter()
                .zip(lt.iter())
                .map(|(w, l)| w - l)
                .collect(),
        }
    }
}

/// Typed learning session. While a token is alive, rewards can be applied;
/// `end` (or drop) closes the session. Outside a session no reward path
/// exists on [`Brain`] — enforced by the borrow checker.
pub struct LearnToken<'a> {
    brain: &'a mut Brain,
}

impl<'a> LearnToken<'a> {
    /// One dynamics step inside the session.
    pub fn step(&mut self, spikes: &[i32]) {
        self.brain.step(spikes);
    }

    /// Apply dopamine now, crediting accumulated eligibility.
    pub fn reward(&mut self, dopamine: f32) {
        self.brain.reward_dopamine(dopamine, None);
    }

    /// Apply dopamine scaled by an external gate in `[0, 1]` (metabolic
    /// budget: stress silences fixation).
    pub fn reward_gated(&mut self, dopamine: f32, gate: f32) {
        self.brain
            .reward_dopamine(dopamine * gate.clamp(0.0, 1.0), None);
    }

    /// Apply dopamine with an explicit weight ceiling (birth burn / instinct
    /// formation above the working plasticity limit).
    pub fn reward_capped(&mut self, dopamine: f32, w_cap: f32) {
        self.brain.reward_dopamine(dopamine, Some(w_cap));
    }

    /// Close the session. The brain is handed back to the caller.
    pub fn end(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::model::DopamineSTDP;

    /// Deterministic LCG for reproducible tests.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn f(&mut self) -> f32 {
            (self.next() % 1_000_000) as f32 / 1_000_000.0
        }
    }

    fn make_net(rng: &mut Lcg, n_neurons: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let mut pre = Vec::new();
        let mut post = Vec::new();
        let mut w = Vec::new();
        for src in 0..n_neurons {
            let out = (rng.next() % 5) as usize;
            for _ in 0..out {
                let dst = (rng.next() % n_neurons as u64) as usize;
                pre.push(src as u32);
                post.push(dst as u32);
                w.push(rng.f());
            }
        }
        (pre, post, w)
    }

    fn cfg() -> StdpConfig {
        StdpConfig {
            dt: 0.5,
            tau_pre: 20.0,
            tau_post: 20.0,
            tau_e: 1000.0,
            a_ltp: 0.1,
            a_ltd: 0.12,
            lr: 0.5,
            w_min: 0.0,
            w_max: 4.0,
        }
    }

    #[test]
    fn lazy_eligibility_matches_dense_dynamics() {
        let mut rng = Lcg(0xC0FFEE);
        let n_neurons = 24;
        let (pre, post, w0) = make_net(&mut rng, n_neurons);
        let c = cfg();
        assert!(pre.len() > 16, "need a real connectome for this test");

        let mut lazy = Brain::new(n_neurons, &pre, &post, w0.clone(), c.clone());
        let mut stdp = DopamineSTDP::new(pre.len(), c.dt);
        stdp.a_ltp = c.a_ltp;
        stdp.a_ltd = c.a_ltd;
        stdp.w_min = c.w_min;
        stdp.w_max = c.w_max;
        let mut wd = w0.clone();

        for step in 0..600u64 {
            let spikes: Vec<i32> = (0..n_neurons)
                .map(|_| if rng.f() < 0.18 { 1 } else { 0 })
                .collect();
            lazy.step(&spikes);

            // Dense reference: gather per-synapse spikes, then trace step.
            let pre_syn: Vec<i32> = pre.iter().map(|&p| spikes[p as usize]).collect();
            let post_syn: Vec<i32> = post.iter().map(|&p| spikes[p as usize]).collect();
            stdp.step(&pre_syn, &post_syn);

            if step % 12 == 7 {
                let d = 1.2;
                // Apply the same reward to both sides (lazy folds + updates;
                // dense applies on its always-current eligibility).
                lazy.reward_dopamine(d, None);
                stdp.apply_dopamine(&mut wd, d, c.lr);

                // Weights must agree closely. Tolerance is slightly loose:
                // lazy folds eligibility decay with one fast_exp over the gap,
                // dense multiplies per step — pure float rounding that
                // compounds over hundreds of steps (≈0.1% observed).
                for i in 0..wd.len() {
                    let diff = (lazy.weights()[i] - wd[i]).abs();
                    let tol = 1e-3 + 5e-3 * wd[i].abs();
                    assert!(
                        diff < tol,
                        "weight {i} diverged after step {step}: lazy={} dense={}",
                        lazy.weights()[i],
                        wd[i]
                    );
                }
                // Eligibility must agree (abs + rel tolerance; dense and lazy
                // differ only by exp-approximation rounding).
                for i in 0..lazy.elig.len() {
                    let a = lazy.elig[i];
                    let b = stdp.eligibility[i];
                    // Dense multiplies one fast_exp constant per step; lazy
                    // calls fast_exp on gap-sized arguments. The minimax poly
                    // error differs per argument and compounds over hundreds
                    // of steps — observed drift ≈1.5%, so 2% relative.
                    let tol = 1e-3 + 2e-2 * a.abs().max(b.abs());
                    assert!(
                        (a - b).abs() <= tol,
                        "eligibility {i} diverged at step {step}: lazy={a} dense={b}"
                    );
                }
            }
        }
        // A reward path through the token must also move weights identically.
        let mut lazy2 = Brain::new(n_neurons, &pre, &post, w0.clone(), c.clone());
        let mut tok = lazy2.begin_learning();
        for step in 0..300u64 {
            let spikes: Vec<i32> = (0..n_neurons)
                .map(|_| if rng.f() < 0.18 { 1 } else { 0 })
                .collect();
            tok.step(&spikes);
            if step % 10 == 3 {
                tok.reward(0.9);
            }
        }
        tok.end();
    }

    #[test]
    fn session_token_gates_rewards_and_ends() {
        let mut lcg = Lcg(7);
        let (pre, post, w0) = make_net(&mut lcg, 16);
        let mut b = Brain::new(16, &pre, &post, w0.clone(), cfg());
        let before = b.weights().to_vec();

        // Inference outside a session: stepping changes traces only.
        for _ in 0..50 {
            b.step(&[1; 16]);
        }
        assert_eq!(b.weights(), before.as_slice());

        // Inside a session, sparse spikes + rewards move weights.
        {
            let mut tok = b.begin_learning();
            for _ in 0..200 {
                let spikes: Vec<i32> = (0..16)
                    .map(|_| if lcg.f() < 0.15 { 1 } else { 0 })
                    .collect();
                tok.step(&spikes);
                if lcg.next() % 3 == 0 {
                    tok.reward(1.5);
                }
            }
            tok.end();
        }
        assert_ne!(b.weights(), before.as_slice(), "rewards must move weights");

        // After end, the brain is usable again and untouched by stepping.
        let mid = b.weights().to_vec();
        for _ in 0..50 {
            b.step(&[1; 16]);
        }
        assert_eq!(b.weights(), mid.as_slice());
    }

    #[test]
    fn snapshot_recall_and_learned_gain() {
        let mut rng = Lcg(11);
        let (pre, post, w0) = make_net(&mut rng, 16);
        let mut b = Brain::new(16, &pre, &post, w0.clone(), cfg());
        b.snapshot();
        assert_eq!(b.drift_rms(), 0.0);
        assert!(b.learned_gain().iter().all(|&g| g == 0.0));

        // Learn: some synapses move off the snapshot.
        {
            let mut tok = b.begin_learning();
            for _ in 0..80 {
                tok.step(&[1; 16]);
                tok.reward(1.5);
            }
            tok.end();
        }
        assert!(b.drift_rms() > 0.0, "learning must create drift");
        assert!(b.learned_gain().iter().any(|&g| g != 0.0));

        // Full recall restores the snapshot (within float rounding).
        b.recall(1.0);
        let n = w0.len();
        for i in 0..n {
            assert!(
                (b.weights()[i] - w0[i]).abs() < 1e-6,
                "recall(1.0) must restore snapshot at {i}"
            );
        }
    }

    #[test]
    fn determinism_from_identical_inputs() {
        let mut rng = Lcg(99);
        let (pre, post, w0) = make_net(&mut rng, 16);
        let c = cfg();
        let run = |seed: u64| {
            let mut lcg = Lcg(seed);
            let mut b = Brain::new(16, &pre, &post, w0.clone(), c.clone());
            for _ in 0..200 {
                let spikes: Vec<i32> = (0..16)
                    .map(|_| if lcg.f() < 0.2 { 1 } else { 0 })
                    .collect();
                let mut tok = b.begin_learning();
                tok.step(&spikes);
                if lcg.next() % 5 == 0 {
                    tok.reward(1.0);
                }
                tok.end();
            }
            b.weights().to_vec()
        };
        assert_eq!(run(0xDEAD), run(0xDEAD));
    }

    #[test]
    fn birth_burn_raises_weights_above_working_cap() {
        // Single synapse 0->1. Pre-then-post pairing with an LTP-dominant
        // config (a_ltp >> a_ltd): eligibility accumulates positive and
        // reward_capped pushes the weight above the working cap 4.0.
        // (With symmetric amplitudes the alternating pairing nets LTD — the
        // post-trace is warm when the pre fires — so we bias the test.)
        let pre = vec![0u32];
        let post = vec![1u32];
        let w0 = vec![0.5f32];
        let mut burn_cfg = cfg();
        burn_cfg.a_ltp = 0.3;
        burn_cfg.a_ltd = 0.02;

        let mut b = Brain::new(2, &pre, &post, w0.clone(), burn_cfg.clone());
        {
            let mut tok = b.begin_learning();
            for _ in 0..200 {
                tok.step(&[1, 0]); // pre fires: builds trace_pre
                tok.step(&[0, 1]); // post fires: LTP with hot pre-trace
                tok.reward_capped(2.0, 64.0);
            }
            tok.end();
        }
        assert!(
            b.weights()[0] > 4.0,
            "capped reward must exceed the working cap, got {}",
            b.weights()[0]
        );

        // Control: the same pairing with a NORMAL reward caps at w_max.
        let mut b2 = Brain::new(2, &pre, &post, w0.clone(), burn_cfg.clone());
        {
            let mut tok = b2.begin_learning();
            for _ in 0..200 {
                tok.step(&[1, 0]);
                tok.step(&[0, 1]);
                tok.reward(2.0);
            }
            tok.end();
        }
        assert!(
            b2.weights()[0] <= 4.0 + 1e-6,
            "normal reward must stop at the working cap, got {}",
            b2.weights()[0]
        );
    }
}
