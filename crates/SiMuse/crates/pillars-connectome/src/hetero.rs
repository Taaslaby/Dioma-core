//! # HeteroRunner вЂ” РЅР°СЃС‚РѕСЏС‰РёР№ РіРµС‚РµСЂРѕРіРµРЅРЅС‹Р№ РјРѕР·Рі
//!
//! РћР”РРќ РѕР±С‰РёР№ РєРѕРЅРЅРµРєС‚РѕРј (CSR + nt_types), РІ РєРѕС‚РѕСЂРѕРј СЂР°Р·РЅС‹Рµ РіСЂСѓРїРїС‹ РЅРµР№СЂРѕРЅРѕРІ
//! Р¶РёРІСѓС‚ РЅР° Р РђР—РќР«РҐ РјРѕРґРµР»СЏС… РґРёРЅР°РјРёРєРё РѕРґРЅРѕРІСЂРµРјРµРЅРЅРѕ. Р’ РѕС‚Р»РёС‡РёРµ РѕС‚
//! [`BrainRunner`](crate::BrainRunner) (РѕРґРЅР° РјРѕРґРµР»СЊ РЅР° РІСЃСЋ СЃРµС‚СЊ), Р·РґРµСЃСЊ:
//!
//! ```text
//!         в”Њв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ  РћР‘Р©РР™ CSR-РєРѕРЅРЅРµРєС‚РѕРј  в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”ђ
//!         в”‚                                                       в”‚
//!   РіСЂСѓРїРїР° Alpha  в”Ђв”ЂСЃРїР°Р№РєРёв”Ђв”Ђв–є  РѕР±С‰РёР№ queue в”Ђв”Ђв–є CSR propagate в”Ђв”Ђв–є i_in
//!   РіСЂСѓРїРїР° HH     в”Ђв”ЂСЃРїР°Р№РєРёв”Ђв”Ђв–є        в–І                              в”‚
//!   РіСЂСѓРїРїР° HAAP   в”Ђв”ЂСЃРїР°Р№РєРёв”Ђв”Ђв–є        в”‚                              в–ј
//!   РіСЂСѓРїРїР° Izh    в”Ђв”ЂСЃРїР°Р№РєРёв”Ђв”Ђв–є   (РІСЃРµ РјРѕРґРµР»Рё)            i_in[РіР»РѕР±Р°Р»СЊРЅС‹Рµ]
//!   РіСЂСѓРїРїР° Rate   в”Ђв”ЂСЃРїР°Р№РєРёв”Ђв”Ђв–є                              в”‚  (СЃР»РµРґ. С€Р°Рі)
//!                                                          в–ј
//!                                         РєР°Р¶РґР°СЏ РіСЂСѓРїРїР° Р±РµСЂС‘С‚ РЎР’РћРЃ РёР· i_in
//! ```
//!
//! РљР°Р¶РґС‹Р№ С€Р°Рі:
//! 1. РєР°Р¶РґР°СЏ РіСЂСѓРїРїР° СЃС‚РµРїР°РµС‚ РЎР’РћР® РґРёРЅР°РјРёРєСѓ (dynamics-only, Р±РµР· РІРЅСѓС‚СЂРµРЅРЅРµР№
//!    propagation) СЃРѕ СЃРІРѕРёРј РІС…РѕРґРѕРј: `i_in[РіР»РѕР±Р°Р»СЊРЅС‹Рµ РёРЅРґРµРєСЃС‹] + external`;
//! 2. Р»РѕРєР°Р»СЊРЅС‹Рµ СЃРїР°Р№РєРё СЂР°Р·Р»РµС‚Р°СЋС‚СЃСЏ РІ РѕР±С‰РёР№ РіР»РѕР±Р°Р»СЊРЅС‹Р№ РІРµРєС‚РѕСЂ СЃРїР°Р№РєРѕРІ;
//! 3. РѕР±С‰РёР№ queue в†’ **РѕРґРЅР°** CSR-propagation С‡РµСЂРµР· Р’Р•РЎР¬ РєРѕРЅРЅРµРєС‚РѕРј в†’ `i_in`
//!    РґР»СЏ СЃР»РµРґСѓСЋС‰РµРіРѕ С€Р°РіР°.
//!
//! РўР°Рє СЃРїР°Р№Рє LIF-РЅРµР№СЂРѕРЅР° СЂРµР°Р»СЊРЅРѕ РґРѕСЃС‚РёРіР°РµС‚ HH/HAAP/Izh/Rate-РЅРµР№СЂРѕРЅР° Рё
//! РЅР°РѕР±РѕСЂРѕС‚ вЂ” С‡РµСЂРµР· РѕРґРЅРё Рё С‚Рµ Р¶Рµ СЃРёРЅР°РїСЃС‹. Р­С‚Рѕ Рё РµСЃС‚СЊ В«РЅР°СЃС‚РѕСЏС‰РёР№
//! РіРµС‚РµСЂРѕРіРµРЅРЅС‹Р№ РјРѕР·РіВ», РєРѕС‚РѕСЂРѕРіРѕ СЂР°РЅСЊС€Рµ РЅРµ Р±С‹Р»Рѕ (mono-runner РїСЂРѕРіРѕРЅСЏР» РІСЃСЋ
//! СЃРµС‚СЊ С‡РµСЂРµР· РѕРґРЅСѓ РјРѕРґРµР»СЊ).
//!
//! ## Р•РґРёРЅС‹Р№ С‚Р°РєС‚
//!
//! Р’СЃРµ РіСЂСѓРїРїС‹ РґРµР»СЏС‚ РћР”РРќ `dt_ms` (РѕР±С‰Р°СЏ РІСЂРµРјРµРЅРЅР°МЃСЏ Р»РёРЅРёСЏ). Р•СЃР»Рё РІ РјРѕР·РіСѓ РµСЃС‚СЊ
//! РіСЂСѓРїРїР° СЃ Р¶С‘СЃС‚РєРёРј С€Р°РіРѕРј (HAAP/HH С‚СЂРµР±СѓСЋС‚ ~0.05 РјСЃ), РіР»РѕР±Р°Р»СЊРЅС‹Р№ `dt_ms`
//! РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РЅРµ РєСЂСѓРїРЅРµРµ вЂ” РёРЅР°С‡Рµ СЌС‚Рё РјРѕРґРµР»Рё С‡РёСЃР»РµРЅРЅРѕ РЅРµСЃС‚Р°Р±РёР»СЊРЅС‹. Р“СЂСѓРїРїС‹,
//! РєРѕС‚РѕСЂС‹Рј РјРµР»РєРёР№ С€Р°Рі РЅРµ РЅСѓР¶РµРЅ (alpha/Izh/Rate), РїСЂРѕСЃС‚Рѕ СЃС‚РµРїР°СЋС‚ С‡Р°С‰Рµ.
//!
//! ## РЎРѕРіР»Р°С€РµРЅРёРµ Рѕ РІС…РѕРґРµ
//!
//! - `external_I` Рё Poisson вЂ” В«РґСЂР°Р№РІВ» (РІ РјР°СЃС€С‚Р°Р±Рµ LIF-С‚РѕРєР°);
//! - `i_in` (СЂРµР·СѓР»СЊС‚Р°С‚ propagation) вЂ” СЂРµРєСѓСЂСЂРµРЅС‚РЅС‹Р№ РІС…РѕРґ (weight Г— nt, СѓР¶Рµ СЃРѕ
//!   Р·РЅР°РєРѕРј С‚РѕСЂРјРѕР·РЅРѕРіРѕ РїСЂРµСЃРёРЅР°РїСЃР°), СЃ Р·Р°РґРµСЂР¶РєРѕР№ 1 С€Р°Рі (РєР°Рє РІ mono HH/HAAP);
//! - alpha-РіСЂСѓРїРїР°: РґСЂР°Р№РІ РёРґС‘С‚ **РїСЂСЏРјРѕ РІ `v`** (РєР°Рє РІ `run_lif_alpha`),
//!   СЂРµРєСѓСЂСЂРµРЅС‚РЅС‹Р№ вЂ” РІ `i_syn` (РїСЂРѕРІРѕРґРёРјРѕСЃС‚СЊ);
//! - HH/HAAP/Izh/Rate: `РІС…РѕРґ = РјР°СЃС€С‚Р°Р±В·(РґСЂР°Р№РІ + СЂРµРєСѓСЂСЂРµРЅС‚РЅС‹Р№)`.

use crate::data::ConnectomeData;
use crate::propagate::{push_spikes_to_queue, record_spikes};
use crate::runner::{alpha_lif_population, NeuronConfig, PoissonGroup};
use crate::sim::{
    haap_population, step_izhikevich_pillars,
};
use crate::types::SpikeQueue;
use pillars::core::NeuralPopulation;
use pillars::kernels::csr::ValidatedCsrGraph;
use pillars::zoo::{
    AlphaLifModel, AlphaLifModelStateIdx, HAAPModel, TransformerModel,
    TransformerModelParamIdx, TransformerModelStateIdx,
};
use rand::{Rng, SeedableRng};

/// РћРґРЅР° РјРѕРґРµР»СЊ РґРёРЅР°РјРёРєРё РґР»СЏ РѕРґРЅРѕР№ РіСЂСѓРїРїС‹ РЅРµР№СЂРѕРЅРѕРІ.
///
/// РљР°Р¶РґР°СЏ РіСЂСѓРїРїР° РІР»Р°РґРµРµС‚ СЃРІРѕРёРј СЃРѕСЃС‚РѕСЏРЅРёРµРј (РїРѕРїСѓР»СЏС†РёСЏ/РјР°СЃСЃРёРІС‹) Рё СЃРІРѕРёРјРё
/// РіР»РѕР±Р°Р»СЊРЅС‹РјРё РёРЅРґРµРєСЃР°РјРё РЅРµР№СЂРѕРЅРѕРІ РІ РѕР±С‰РµРј РєРѕРЅРЅРµРєС‚РѕРјРµ.
enum GroupEngine {
    /// LIF + alpha-СЃРёРЅР°РїСЃ (Shiu et al. 2024). Р”СЂР°Р№РІ в†’ `v`, СЂРµРєСѓСЂСЂРµРЅС‚ в†’ `i_syn`.
    ///
    /// РђРєСЃРѕРЅР°Р»СЊРЅР°СЏ Р·Р°РґРµСЂР¶РєР°: РєР°Рє РІ mono `run_lif_alpha`, СЂРµРєСѓСЂСЂРµРЅС‚ РёРґС‘С‚
    /// С‡РµСЂРµР· РєРѕР»СЊС†Рѕ РёР· `delay_steps+1` СЃР»РѕС‚РѕРІ (t_delayв†’С€Р°РіРё). Р‘РµР· СЌС‚РѕРіРѕ
    /// hetero РѕС‚РґР°С‘С‚ СЂРµРєСѓСЂСЂРµРЅС‚ Р·Р° 1 С€Р°Рі РІРјРµСЃС‚Рѕ ~4 вЂ” РЅР° РґРёРЅР°РјРёС‡РµСЃРєРѕРј РІС…РѕРґРµ
    /// РїРµС‚Р»Рё СЃС…РѕРґСЏС‚СЃСЏ РёРЅР°С‡Рµ Рё РјРѕРіСѓС‚ РєРѕР»Р»Р°РїСЃРёСЂРѕРІР°С‚СЊ РІ С‚РёС€РёРЅСѓ.
    Alpha {
        indices: Vec<usize>,
        pop: NeuralPopulation<AlphaLifModel>,
        /// Р’С…РѕРґРЅРѕР№ РјР°СЃС€С‚Р°Р± РґСЂР°Р№РІР° (1.0 = РєР°Рє РІ mono LifAlpha).
        drive_scale: f32,
        /// РњР°СЃС€С‚Р°Р± СЂРµРєСѓСЂСЂРµРЅС‚РЅРѕРіРѕ РІС…РѕРґР° (СЃРёРЅР°РїС‚РёС‡РµСЃРєРёРµ РІРµСЃР° РѕР±С‰РёРµ РґР»СЏ РІСЃРµС…
        /// РјРѕРґРµР»РµР№; С‡СѓРІСЃС‚РІРёС‚РµР»СЊРЅРѕСЃС‚СЊ Рє РЅРёРј Сѓ РєР°Р¶РґРѕР№ РјРѕРґРµР»Рё СЃРІРѕСЏ).
        rec_scale: f32,
        /// Р§РёСЃР»Рѕ С€Р°РіРѕРІ Р·Р°РґРµСЂР¶РєРё Р°РєСЃРѕРЅР°Р»СЊРЅРѕРіРѕ РєРѕР»СЊС†Р°.
        delay_steps: usize,
        /// РљРѕР»СЊС†Рѕ Р·Р°РґРµСЂР¶РєРё: (delay_steps+1) Г— k (k = indices.len()).
        delay_ring: Vec<f32>,
        /// РџРѕР·РёС†РёСЏ Р·Р°РїРёСЃРё РІ РєРѕР»СЊС†Рѕ.
        delay_pos: usize,
    },
    /// Hodgkin-Huxley вЂ” СѓРґР°Р»С‘РЅ (РѕРїРµСЂР°РЅРґС‹ vtrap/fast_exp РѕСЃС‚Р°Р»РёСЃСЊ РґР»СЏ DIY).
    /// РљРѕРЅС„РёРі HH С‚РµРїРµСЂСЊ РѕС‚РєР»РѕРЅСЏРµС‚СЃСЏ РІ build_engine.
    HhStub {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
    },
    /// HAAP (M-current + h-current), current-based.
    Haap {
        indices: Vec<usize>,
        pop: NeuralPopulation<HAAPModel>,
        /// РџРµСЂСЃРёСЃС‚РµРЅС‚РЅС‹Р№ СЃРёРЅР°РїС‚РёС‡РµСЃРєРёР№ С‚РѕРє (Р·Р°С‚СѓС…Р°РµС‚ СЃ П„_syn).
        syn: Vec<f32>,
        decay: f32,
        drive_scale: f32,
        rec_scale: f32,
    },
    /// Izhikevich (regular spiking).
    Izh {
        indices: Vec<usize>,
        v: Vec<f32>,
        u: Vec<f32>,
        trace: Vec<f32>,
        refr: Vec<f32>,
        a: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
        d: Vec<f32>,
        dt_ms: f32,
        tau_ref: f32,
        drive_scale: f32,
        rec_scale: f32,
    },
    /// Rate model (continuous firing rates).
    Rate {
        indices: Vec<usize>,
        v: Vec<f32>,
        tau: f32,
        dt_ms: f32,
        drive_scale: f32,
        rec_scale: f32,
    },
    /// SurgeN вЂ” РЅРµР№СЂРѕРЅ-С‚СЂР°РЅСЃС„РѕСЂРјРµСЂ (pillars zoo `TransformerModel`):
    /// temporal self-attention РЅР°Рґ 8 СЃР»РѕС‚Р°РјРё РїР°РјСЏС‚Рё, leaky-РёРЅС‚РµРіСЂР°С‚РѕСЂ в†’ СЃРїР°Р№Рє.
    /// Р’С…РѕРґ = РјР°СЃС€С‚Р°Р±В·(РґСЂР°Р№РІ + СЂРµРєСѓСЂСЂРµРЅС‚), РєР°Рє Сѓ HH/Izh; Р·Р°РґРµСЂР¶РєР° 1 С€Р°Рі.
    Transformer {
        indices: Vec<usize>,
        pop: NeuralPopulation<TransformerModel>,
        drive_scale: f32,
        rec_scale: f32,
    },
}

/// One NeuralPopulation-backed group exported for pillars-cuda
/// `CompiledNetwork::new`: (source, kernel_name, n, ns, np, states_flat, params_flat).
pub type GpuPopExport = (
    &'static str,
    &'static str,
    usize,
    usize,
    usize,
    Vec<f32>,
    Vec<f32>,
);

impl GroupEngine {
    fn indices(&self) -> &[usize] {
        match self {
            GroupEngine::Alpha { indices, .. }
            | GroupEngine::HhStub { indices, .. }
            | GroupEngine::Haap { indices, .. }
            | GroupEngine::Izh { indices, .. }
            | GroupEngine::Rate { indices, .. }
            | GroupEngine::Transformer { indices, .. } => indices,
        }
    }

    /// Dynamics-only step: С‡РёС‚Р°РµС‚ `drive` (external+poisson, РіР»РѕР±Р°Р»СЊРЅС‹Р№) Рё
    /// `rec` (СЂРµРєСѓСЂСЂРµРЅС‚РЅС‹Р№ РІС…РѕРґ, РіР»РѕР±Р°Р»СЊРЅС‹Р№), РїРёС€РµС‚ СЃРїР°Р№РєРё РІ `spikes_out`
    /// (РіР»РѕР±Р°Р»СЊРЅС‹Р№ РІРµРєС‚РѕСЂ, РїРѕ РіР»РѕР±Р°Р»СЊРЅС‹Рј РёРЅРґРµРєСЃР°Рј РіСЂСѓРїРїС‹).
    fn step(&mut self, drive: &[f32], rec: &[f32], spikes_out: &mut [i32]) {
        let _ = rec;
        let _ = spikes_out;
        self.step_inner(drive, rec, spikes_out);
    }

    /// РћС‚РґР°С‚СЊ С‚РµРєСѓС‰РёР№ СЃР»РѕР№ СЂРµРєСѓСЂСЂРµРЅС‚Р° РІ РіСЂСѓРїРїСѓ (РїРѕСЃР»Рµ propagation).
    /// Alpha-РіСЂСѓРїРїС‹ СЃРєР»Р°РґС‹РІР°СЋС‚ РµРіРѕ РІ СЃРІРѕС‘ РєРѕР»СЊС†Рѕ Р·Р°РґРµСЂР¶РєРё; РѕСЃС‚Р°Р»СЊРЅС‹Рµ вЂ” no-op
    /// (РѕРЅРё Рё С‚Р°Рє С‡РёС‚Р°СЋС‚ `rec` РЅР° СЃР»РµРґСѓСЋС‰РµРј С€Р°РіРµ).
    fn deliver(&mut self, rec: &[f32]) {
        if let GroupEngine::Alpha {
            indices,
            delay_steps,
            delay_ring,
            delay_pos,
            ..
        } = self
        {
            let k = indices.len();
            let size = *delay_steps + 1;
            let slot = &mut delay_ring[*delay_pos * k.max(1)..(*delay_pos + 1) * k.max(1)];
            for (j, &g) in indices.iter().enumerate() {
                slot[j] = rec.get(g).copied().unwrap_or(0.0);
            }
            *delay_pos = (*delay_pos + 1) % size;
        }
    }

    fn step_inner(&mut self, drive: &[f32], rec: &[f32], spikes_out: &mut [i32]) {
        match self {
            GroupEngine::Alpha {
                indices,
                pop,
                drive_scale,
                rec_scale,
                delay_steps,
                delay_ring,
                delay_pos,
            } => {
                let n = indices.len();
                let mut v_slice = vec![0.0f32; n];
                let mut i_syn = vec![0.0f32; n];
                let _ = delay_steps;
                let read_pos = *delay_pos;
                for (k, &g) in indices.iter().enumerate() {
                    // Р”СЂР°Р№РІ в†’ v (РєР°Рє РІ run_lif_alpha), РјР°СЃС€С‚Р°Р±РёСЂРѕРІР°РЅРЅС‹Р№.
                    v_slice[k] = drive.get(g).copied().unwrap_or(0.0) * *drive_scale;
                    // Р РµРєСѓСЂСЂРµРЅС‚ в†’ i_syn: С‡РёС‚Р°РµРј Р—РђР”Р•Р Р–РђРќРќР«Р™ СЃР»РѕС‚ РєРѕР»СЊС†Р°
                    // (РєР°Рє mono run_lif_alpha, Р° РЅРµ 1-С€Р°РіРѕРІС‹Р№ rec).
                    i_syn[k] = delay_ring[read_pos * n + k] * *rec_scale;
                }
                // РџРµСЂРµРЅРѕСЃРёРј РЅР°РєРѕРїР»РµРЅРЅС‹Р№ v РїРѕРїСѓР»СЏС†РёРё + СЌС‚РѕС‚ С€Р°Рі РґСЂР°Р№РІР°.
                {
                    let v_col = pop.state_slice_mut(AlphaLifModelStateIdx::v.into());
                    for (k, &g) in indices.iter().enumerate() {
                        // v РїРѕРїСѓР»СЏС†РёРё Р¶РёРІС‘С‚ РІ Р»РѕРєР°Р»СЊРЅС‹С… РёРЅРґРµРєСЃР°С… РіСЂСѓРїРїС‹.
                        let _ = g;
                        v_col[k] += v_slice[k];
                    }
                }
                let mut spikes = vec![0i32; n];
                pop.step(&i_syn, &mut spikes);
                for (k, &g) in indices.iter().enumerate() {
                    if spikes[k] != 0 {
                        spikes_out[g] = 1;
                    }
                }
            }
            GroupEngine::HhStub { .. } => {
                // HH удалён — группа не степается (build_engine уже отклоняет создание)
            }
            GroupEngine::Haap {
                indices,
                pop,
                syn,
                decay,
                drive_scale,
                rec_scale,
            } => {
                let n = indices.len();
                let mut i_in = vec![0.0f32; n];
                for (k, &g) in indices.iter().enumerate() {
                    // РџРµСЂСЃРёСЃС‚РµРЅС‚РЅС‹Р№ СЃРёРЅР°РїС‚РёС‡РµСЃРєРёР№ С‚РѕРє: decay + РЅРѕРІС‹Р№ СЂРµРєСѓСЂСЂРµРЅС‚.
                    syn[k] = syn[k] * *decay + rec.get(g).copied().unwrap_or(0.0) * *rec_scale;
                    i_in[k] = drive.get(g).copied().unwrap_or(0.0) * *drive_scale + syn[k];
                }
                let mut spikes = vec![0i32; n];
                pop.step(&i_in, &mut spikes);
                for (k, &g) in indices.iter().enumerate() {
                    if spikes[k] != 0 {
                        spikes_out[g] = 1;
                    }
                }
            }
            GroupEngine::Izh {
                indices,
                v,
                u,
                trace,
                refr,
                a,
                b,
                c,
                d,
                dt_ms,
                tau_ref,
                drive_scale,
                rec_scale,
            } => {
                // SIMD-С€Р°Рі СЂР°Р±РѕС‚Р°РµС‚ СЃ РїР°РґРґР»РµРЅРЅС‹РјРё РјР°СЃСЃРёРІР°РјРё; РїРёС€РµРј РґСЂР°Р№РІ С‚РѕР»СЊРєРѕ
                // РІ РїРµСЂРІС‹Рµ n (СЂРµР°Р»СЊРЅС‹Рµ РЅРµР№СЂРѕРЅС‹), С…РІРѕСЃС‚ вЂ” РЅСѓР»Рё.
                let kp = v.len();
                let mut ii = vec![0.0f32; kp];
                for (k, &g) in indices.iter().enumerate() {
                    ii[k] = drive.get(g).copied().unwrap_or(0.0) * *drive_scale
                        + rec.get(g).copied().unwrap_or(0.0) * *rec_scale;
                }
                let mut spikes = vec![0i32; kp];
                step_izhikevich_pillars(
                    v,
                    u,
                    trace,
                    a,
                    b,
                    &mut ii,
                    *dt_ms,
                    0.99,
                    &mut spikes,
                    c,
                    d,
                    refr,
                    *tau_ref,
                );
                for (k, &g) in indices.iter().enumerate() {
                    if spikes[k] != 0 {
                        spikes_out[g] = 1;
                    }
                }
            }
            GroupEngine::Rate {
                indices,
                v,
                tau,
                dt_ms,
                drive_scale,
                rec_scale,
            } => {
                let dt = *dt_ms / 1000.0;
                let decay = (-dt / tau.max(1e-6)).exp();
                let inv_decay = 1.0 - decay;
                for (k, &g) in indices.iter().enumerate() {
                    let input = (drive.get(g).copied().unwrap_or(0.0) * *drive_scale
                        + rec.get(g).copied().unwrap_or(0.0) * *rec_scale)
                        .clamp(0.0, 1.0);
                    v[k] = v[k] * decay + input * inv_decay;
                    if v[k] > 0.5 {
                        spikes_out[g] = 1;
                    }
                }
            }
            GroupEngine::Transformer {
                indices,
                pop,
                drive_scale,
                rec_scale,
            } => {
                // Р’С…РѕРґ = РјР°СЃС€С‚Р°Р±В·(РґСЂР°Р№РІ + СЂРµРєСѓСЂСЂРµРЅС‚), РєР°Рє Сѓ HH/Izh. РњРѕРґРµР»СЊ СЃР°РјР°
                // РЅРѕСЂРјР°Р»РёР·СѓРµС‚ query Рё РґРµСЂР¶РёС‚ РїР°РјСЏС‚СЊ РїРѕСЃР»РµРґРЅРёС… 8 РІС…РѕРґРѕРІ.
                let mut i_in = vec![0.0f32; indices.len()];
                for (k, &g) in indices.iter().enumerate() {
                    i_in[k] = drive.get(g).copied().unwrap_or(0.0) * *drive_scale
                        + rec.get(g).copied().unwrap_or(0.0) * *rec_scale;
                }
                let mut spikes = vec![0i32; indices.len()];
                pop.step(&i_in, &mut spikes);
                for (k, &g) in indices.iter().enumerate() {
                    if spikes[k] != 0 {
                        spikes_out[g] = 1;
                    }
                }
            }
        }
    }
}

/// РћРїРёСЃР°РЅРёРµ РѕРґРЅРѕР№ РіСЂСѓРїРїС‹ РЅРµР№СЂРѕРЅРѕРІ РґР»СЏ [`HeteroRunner`].
#[derive(Debug, Clone)]
pub struct HeteroGroup {
    /// РњРѕРґРµР»СЊ Рё РїР°СЂР°РјРµС‚СЂС‹ РіСЂСѓРїРїС‹.
    pub config: NeuronConfig,
    /// Р“Р»РѕР±Р°Р»СЊРЅС‹Рµ РёРЅРґРµРєСЃС‹ РЅРµР№СЂРѕРЅРѕРІ СЌС‚РѕР№ РіСЂСѓРїРїС‹ РІ РѕР±С‰РµРј РєРѕРЅРЅРµРєС‚РѕРјРµ.
    pub indices: Vec<usize>,
    /// РњР°СЃС€С‚Р°Р± РІС…РѕРґРЅРѕРіРѕ РґСЂР°Р№РІР° (external + Poisson) СЌС‚РѕР№ РіСЂСѓРїРїС‹.
    /// LIF-alpha: 1.0; HAAP/HH: ~8.0 (СЃРј. kailo HAAP_INPUT_SCALE).
    pub drive_scale: f32,
    /// РњР°СЃС€С‚Р°Р± СЂРµРєСѓСЂСЂРµРЅС‚РЅРѕРіРѕ РІС…РѕРґР° (СЃРёРЅР°РїС‚РёС‡РµСЃРєРёРµ РІРµСЃР° РѕР±С‰РёРµ).
    /// LIF-alpha: 1.0; HAAP: 0.25 (СЃРј. kailo HAAP_SYN_SCALE).
    pub rec_scale: f32,
}

/// РќР°СЃС‚РѕСЏС‰РёР№ РіРµС‚РµСЂРѕРіРµРЅРЅС‹Р№ РјРѕР·Рі: РѕРґРёРЅ РєРѕРЅРЅРµРєС‚РѕРј, РјРЅРѕРіРѕ РјРѕРґРµР»РµР№ СЃСЂР°Р·Сѓ.
///
/// РљРѕРЅСЃС‚СЂСѓРєС‚РѕСЂ [`HeteroRunner::new`] РїСЂРёРЅРёРјР°РµС‚ РїРѕР»РЅС‹Р№ РєРѕРЅРЅРµРєС‚РѕРј Рё СЃРїРёСЃРѕРє
/// РіСЂСѓРїРї. Р“СЂСѓРїРїС‹ РїРѕРєСЂС‹РІР°СЋС‚ Р’РЎР• РЅРµР№СЂРѕРЅС‹ (РєР°Р¶РґС‹Р№ СЂРѕРІРЅРѕ РѕРґРёРЅ СЂР°Р·) вЂ” РёРЅР°С‡Рµ
/// РЅРµРїРѕРєСЂС‹С‚С‹Рµ РЅРµР№СЂРѕРЅС‹ РјРѕР»С‡Р°С‚ (РёС… СЃРѕСЃС‚РѕСЏРЅРёРµ РЅРµ СЃС‚РµРїР°РµС‚СЃСЏ).
///
/// # РЎРЅР°РїС€РѕС‚С‹ (save/load)
///
/// [`HeteroRunner::snapshot`] РІРѕР·РІСЂР°С‰Р°РµС‚ РїРѕР»РЅС‹Р№ СЃР»РµРїРѕРє СЃРѕСЃС‚РѕСЏРЅРёСЏ
/// (РєРѕРЅРЅРµРєС‚РѕРј СЃ С‚РµРєСѓС‰РёРјРё РІРµСЃР°РјРё, РІСЃРµ РіСЂСѓРїРїС‹, СЃС‡С‘С‚С‡РёРє С€Р°РіРѕРІ), Р°
/// [`HeteroRunner::restore`] РІРѕСЃСЃС‚Р°РЅР°РІР»РёРІР°РµС‚ РµРіРѕ РЅР° РјРµСЃС‚Рµ. РЎРЅР°РїС€РѕС‚ вЂ” СЌС‚Рѕ
/// С‡РёСЃС‚С‹Рµ РґР°РЅРЅС‹Рµ (`Vec`), РёС… РјРѕР¶РЅРѕ СЃРµСЂРёР°Р»РёР·РѕРІР°С‚СЊ С‡РµСЂРµР· РєСЂРµР№С‚ `taas-format`
/// (`save_hetero_snapshot`/`load_hetero_snapshot`).
///
/// вљ пёЏ RNG РџСѓР°СЃСЃРѕРЅРѕРІСЃРєРёС… РіСЂСѓРїРї РЅРµ СЃРµСЂРёР°Р»РёР·СѓРµС‚СЃСЏ (Сѓ `StdRng` РЅРµС‚ API С‡С‚РµРЅРёСЏ
/// СЃРѕСЃС‚РѕСЏРЅРёСЏ) вЂ” РїРѕСЃР»Рµ `restore` СЃС‚РѕС…Р°СЃС‚РёРєР° РЅР°С‡РёРЅР°РµС‚СЃСЏ СЃ РЅР°С‡Р°Р»СЊРЅРѕРіРѕ СЃРёРґР°, Р°
/// РЅРµ РїСЂРѕРґРѕР»Р¶Р°РµС‚СЃСЏ СЃ С‚РѕС‡РєРё СЃРЅР°РїС€РѕС‚Р°. Р”Р»СЏ РґРµС‚РµСЂРјРёРЅРёСЂРѕРІР°РЅРЅРѕР№ СЃРёРјСѓР»СЏС†РёРё Р±РµР·
/// РџСѓР°СЃСЃРѕРЅР° СЃРЅР°РїС€РѕС‚/restore вЂ” Р±РёС‚-РІ-Р±РёС‚.
pub struct HeteroRunner {
    /// RNG РџСѓР°СЃСЃРѕРЅРѕРІСЃРєРёС… РіСЂСѓРїРї: СЃРѕР·РґР°С‘С‚СЃСЏ РёР· С„РёРєСЃРёСЂРѕРІР°РЅРЅРѕРіРѕ СЃРёРґР° (СЃРј.
    /// `StdRng::from_seed` РІ `new`). РЎРёРґ С…СЂР°РЅРёС‚СЃСЏ РІ СЃРЅР°РїС€РѕС‚Рµ Рё
    /// РІРѕСЃСЃС‚Р°РЅР°РІР»РёРІР°РµС‚СЃСЏ РїСЂРё `restore`.
    pub(crate) poisson_seed: [u8; 32],

    /// РћР±С‰РёР№ РєРѕРЅРЅРµРєС‚РѕРј (РїСѓР±Р»РёС‡РЅС‹Р№ вЂ” РїР»Р°СЃС‚РёС‡РЅРѕСЃС‚СЊ kailo С‡РёС‚Р°РµС‚ values).
    pub connectome: ConnectomeData,
    csr_graph: ValidatedCsrGraph,
    /// РљРѕРЅС„РёРіРё РіСЂСѓРїРї (РґР»СЏ reset/РїРµСЂРµСЃС‚СЂРѕРµРЅРёСЏ engines).
    specs: Vec<HeteroGroup>,
    groups: Vec<GroupEngine>,
    // РћР±С‰РёРµ Р±СѓС„РµСЂС‹ (РіР»РѕР±Р°Р»СЊРЅС‹Рµ, СЂР°Р·РјРµСЂ n).
    drive: Vec<f32>,  // external + poisson РЅР° СЌС‚РѕС‚ С€Р°Рі
    rec: Vec<f32>,    // СЂРµРєСѓСЂСЂРµРЅС‚РЅС‹Р№ РІС…РѕРґ (СЂРµР·СѓР»СЊС‚Р°С‚ propagation)
    spikes: Vec<i32>, // РіР»РѕР±Р°Р»СЊРЅС‹Рµ СЃРїР°Р№РєРё С€Р°РіР°
    queue: SpikeQueue,
    spike_buffer: Vec<i32>,
    step_counter: usize,
    n: usize,
    dt_ms: f32,
    poisson_groups: Vec<PoissonGroup>,
    rng: rand::rngs::StdRng,
}

impl HeteroRunner {
    /// РЎРѕР·РґР°С‚СЊ РіРµС‚РµСЂРѕРіРµРЅРЅС‹Р№ РјРѕР·Рі.
    ///
    /// # РћС€РёР±РєРё
    /// - РїСѓСЃС‚РѕР№ РєРѕРЅРЅРµРєС‚РѕРј;
    /// - РїСѓСЃС‚РѕР№ СЃРїРёСЃРѕРє РіСЂСѓРїРї;
    /// - РіСЂСѓРїРїР° РїРѕРєСЂС‹РІР°РµС‚ РЅРµР№СЂРѕРЅ Р·Р° РїСЂРµРґРµР»Р°РјРё РєРѕРЅРЅРµРєС‚РѕРјР°;
    /// - РЅРµР№СЂРѕРЅ РїРѕРєСЂС‹С‚ РґРІР°Р¶РґС‹ (РіСЂСѓРїРїС‹ РїРµСЂРµСЃРµРєР°СЋС‚СЃСЏ).
    pub fn new(
        connectome: ConnectomeData,
        groups: Vec<HeteroGroup>,
        dt_ms: f32,
    ) -> Result<Self, String> {
        let n = connectome.n_neurons;
        if n == 0 {
            return Err("HeteroRunner: connectome has 0 neurons".into());
        }
        if groups.is_empty() {
            return Err("HeteroRunner: no groups".into());
        }
        let csr_graph =
            ValidatedCsrGraph::from_slices(&connectome.row_ptr, &connectome.col_indices, n);

        // РџСЂРѕРІРµСЂРєР° РїРѕРєСЂС‹С‚РёСЏ: РєР°Р¶РґС‹Р№ РЅРµР№СЂРѕРЅ СЂРѕРІРЅРѕ РѕРґРёРЅ СЂР°Р·.
        let mut covered = vec![false; n];
        let mut engines = Vec::with_capacity(groups.len());
        let mut specs = Vec::with_capacity(groups.len());
        for g in groups {
            if g.indices.is_empty() {
                continue; // РїСѓСЃС‚С‹Рµ РіСЂСѓРїРїС‹ РїСЂРѕРїСѓСЃРєР°РµРј
            }
            for &idx in &g.indices {
                if idx >= n {
                    return Err(format!("HeteroRunner: group index {idx} >= n={n}"));
                }
                if covered[idx] {
                    return Err(format!("HeteroRunner: neuron {idx} covered twice"));
                }
                covered[idx] = true;
            }
            let eng = build_engine(&g, dt_ms)?;
            specs.push(g.clone());
            engines.push(eng);
        }
        if engines.is_empty() {
            return Err("HeteroRunner: no non-empty groups".into());
        }

        let poisson_seed = [0u8; 32];
        let rng = rand::rngs::StdRng::from_seed(poisson_seed);
        let queue_cap = n.max(4).next_power_of_two();
        Ok(Self {
            connectome,
            csr_graph,
            specs,
            groups: engines,
            drive: vec![0.0f32; n],
            rec: vec![0.0f32; n],
            spikes: vec![0i32; n],
            queue: SpikeQueue::new(queue_cap),
            spike_buffer: vec![0i32; n * 64],
            step_counter: 0,
            n,
            dt_ms,
            poisson_groups: Vec::new(),
            rng,
            poisson_seed,
        })
    }

    /// Р”РѕР±Р°РІРёС‚СЊ Poisson-РіСЂСѓРїРїСѓ (РіР»РѕР±Р°Р»СЊРЅС‹Рµ РёРЅРґРµРєСЃС‹).
    pub fn add_poisson_group(&mut self, indices: &[usize], rate_hz: f32, weight: f32) {
        let indices = indices.iter().filter(|&&i| i < self.n).copied().collect();
        self.poisson_groups.push(PoissonGroup {
            indices,
            rate_hz,
            weight,
        });
    }

    /// РћС‡РёСЃС‚РёС‚СЊ Poisson-РіСЂСѓРїРїС‹.
    pub fn clear_poisson_groups(&mut self) {
        self.poisson_groups.clear();
    }

    /// Р§РёСЃР»Рѕ РЅРµР№СЂРѕРЅРѕРІ.
    pub fn n_neurons(&self) -> usize {
        self.n
    }

    /// РћР±С‰РёР№ С€Р°Рі РёРЅС‚РµРіСЂР°С†РёРё (РјСЃ).
    pub fn dt_ms(&self) -> f32 {
        self.dt_ms
    }

    /// Р’СЃРµРіРѕ С€Р°РіРѕРІ РїСЂРѕСЃРёРјСѓР»РёСЂРѕРІР°РЅРѕ.
    pub fn step_counter(&self) -> usize {
        self.step_counter
    }

    /// Р“Р»РѕР±Р°Р»СЊРЅС‹Рµ РёРЅРґРµРєСЃС‹ РєР°Р¶РґРѕР№ РіСЂСѓРїРїС‹ (РґР»СЏ РґРёР°РіРЅРѕСЃС‚РёРєРё/observatory).
    pub fn group_indices(&self) -> Vec<&[usize]> {
        self.groups.iter().map(|g| g.indices()).collect()
    }

    /// Р­РєСЃРїРѕСЂС‚ РґР°РЅРЅС‹С… РіСЂСѓРїРї РґР»СЏ pillars-cuda `CompiledNetwork::new`.
    ///
    /// Р’РѕР·РІСЂР°С‰Р°РµС‚ Vec<(source, kernel_name, n, ns, np, states, params)> РґР»СЏ
    /// РєР°Р¶РґРѕР№ РЅРµРїСѓСЃС‚РѕР№ РіСЂСѓРїРїС‹ вЂ” СЂРѕРІРЅРѕ С‚Рѕ, С‡С‚Рѕ РЅСѓР¶РЅРѕ РґР»СЏ `NetworkPop`.
    /// Р­РєСЃРїРѕСЂС‚ РґР°РЅРЅС‹С… РіСЂСѓРїРї РґР»СЏ pillars-cuda `CompiledNetwork::new`.
    ///
    /// Returns Vec of (source, kernel_name, n, ns, np, states_flat, params_flat)
    /// for each NeuralPopulation-backed group (Alpha/Hh/Haap/Transformer).
    /// Izh and Rate use raw vectors and are skipped (CPU fallback).
    pub fn export_pops_for_gpu(&self) -> Vec<GpuPopExport> {
        use pillars::zoo::neurons::*;
        use pillars::zoo::Model;
        macro_rules! export_pop {
            ($pop:expr, $indices:expr, $source:expr, $kernel:expr, $M:ty) => {{
                let n = $indices.len();
                let stride = n.div_ceil(8) * 8;
                let ns = <$M as Model>::N_STATE;
                let np = <$M as Model>::N_PARAMS;
                let mut states = Vec::with_capacity(ns * stride);
                for c in 0..ns {
                    let col = $pop.state_slice(c);
                    states.extend_from_slice(col);
                    states.resize(states.len() + (stride - col.len()), 0.0);
                }
                let mut params = Vec::with_capacity(np * stride);
                for c in 0..np {
                    let col = $pop.param_slice(c);
                    params.extend_from_slice(col);
                    params.resize(params.len() + (stride - col.len()), 0.0);
                }
                ($source, $kernel, n, ns, np, states, params)
            }};
        }
        self.groups
            .iter()
            .filter_map(|g| match g {
                GroupEngine::Alpha { pop, indices, .. } => Some(export_pop!(
                    pop,
                    indices,
                    LIF_CUDA_SOURCE,
                    "lif_kernel",
                    AlphaLifModel
                )),
                GroupEngine::HhStub { .. } => None, // HH удалён
                GroupEngine::Haap { pop, indices, .. } => Some(export_pop!(
                    pop,
                    indices,
                    HAAP_CUDA_SOURCE,
                    "haap_kernel",
                    HAAPModel
                )),
                GroupEngine::Transformer { pop, indices, .. } => Some(export_pop!(
                    pop,
                    indices,
                    TRANSFORMER_CUDA_SOURCE,
                    "transformer_kernel",
                    TransformerModel
                )),
                _ => None, // Izh/Rate use raw vectors вЂ” no NeuralPopulation to export
            })
            .collect()
    }

    /// Р—Р°РїСѓСЃС‚РёС‚СЊ `n_steps` С€Р°РіРѕРІ. `external_I` вЂ” РіР»РѕР±Р°Р»СЊРЅС‹Р№ РґСЂР°Р№РІ `[n]`.
    ///
    /// Р’РѕР·РІСЂР°С‰Р°РµС‚ spike buffer `[n_steps Г— n]` (0/1).
    pub fn run(&mut self, n_steps: usize, external_I: &[f32]) -> &[i32] {
        let needed = n_steps * self.n;
        if self.spike_buffer.len() < needed {
            self.spike_buffer.resize(needed, 0);
        }
        self.spike_buffer[..needed].fill(0);

        let n = self.n;
        for step in 0..n_steps {
            // 1. Р”СЂР°Р№РІ = external + poisson (РіР»РѕР±Р°Р»СЊРЅС‹Р№).
            self.drive[..n].fill(0.0);
            for (i, &e) in external_I.iter().enumerate().take(n) {
                self.drive[i] += e;
            }
            inject_poisson_into(
                &self.poisson_groups,
                &mut self.rng,
                &mut self.drive,
                self.dt_ms,
            );

            // 2. РљР°Р¶РґР°СЏ РіСЂСѓРїРїР° СЃС‚РµРїР°РµС‚ РЎР’РћР® РґРёРЅР°РјРёРєСѓ (dynamics-only).
            self.spikes[..n].fill(0);
            for g in self.groups.iter_mut() {
                g.step(&self.drive, &self.rec, &mut self.spikes);
            }

            // 3. РћР±С‰РёР№ queue + record.
            push_spikes_to_queue(&self.spikes, n, &mut self.queue);
            record_spikes(&self.spikes, n, &mut self.spike_buffer, step);

            // 4. РћР”РќРђ propagation С‡РµСЂРµР· Р’Р•РЎР¬ РєРѕРЅРЅРµРєС‚РѕРј в†’ rec (СЃР»РµРґ. С€Р°Рі).
            self.rec[..n].fill(0.0);
            crate::propagate::csr_propagate_from_queue_with_topology(
                self.csr_graph.as_topology(),
                &self.connectome.values,
                &self.connectome.nt_types,
                &mut self.queue,
                &mut self.rec,
                1.0,
            );

            // 5. Р Р°Р·РґР°С‚СЊ rec РІ РіСЂСѓРїРїС‹ (alpha в†’ РєРѕР»СЊС†Рѕ Р·Р°РґРµСЂР¶РєРё).
            for g in self.groups.iter_mut() {
                g.deliver(&self.rec);
            }
        }

        self.step_counter += n_steps;
        &self.spike_buffer[..needed]
    }

    /// РЎР±СЂРѕСЃРёС‚СЊ СЃРѕСЃС‚РѕСЏРЅРёРµ РІСЃРµС… РіСЂСѓРїРї (РїРѕРїСѓР»СЏС†РёРё/РјР°СЃСЃРёРІС‹ в†’ РґРµС„РѕР»С‚).
    pub fn reset(&mut self) {
        let specs = self.specs.clone();
        let dt = self.dt_ms;
        let mut engines = Vec::with_capacity(specs.len());
        for spec in &specs {
            match build_engine(spec, dt) {
                Ok(eng) => engines.push(eng),
                Err(_) => continue,
            }
        }
        self.groups = engines;
        self.spike_buffer.fill(0);
        self.step_counter = 0;
        self.rec.fill(0.0);
        self.drive.fill(0.0);
        self.spikes.fill(0);
    }
}

/// РџРѕСЃС‚СЂРѕРёС‚СЊ engine РіСЂСѓРїРїС‹ РїРѕ РєРѕРЅС„РёРіСѓ.
fn build_engine(g: &HeteroGroup, dt_ms: f32) -> Result<GroupEngine, String> {
    let k = g.indices.len();
    let scale = if g.drive_scale > 0.0 {
        g.drive_scale
    } else {
        1.0
    };
    let rscale = if g.rec_scale > 0.0 { g.rec_scale } else { 1.0 };
    let engine = match &g.config {
        NeuronConfig::LifAlpha {
            t_delay,
            tau_ref,
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            tau_syn,
            ..
        } => {
            let pop = alpha_lif_population(
                k, dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, *tau_syn,
            );
            // РђРєСЃРѕРЅР°Р»СЊРЅР°СЏ Р·Р°РґРµСЂР¶РєР° (РєР°Рє mono run_lif_alpha): РєРѕР»СЊС†Рѕ РёР·
            // (delay_steps+1) СЃР»РѕС‚РѕРІ РїРѕ k РЅРµР№СЂРѕРЅРѕРІ. РЁРёСЂРёРЅР° вЂ” СЂРѕРІРЅРѕ РЅР° РѕРґРёРЅ
            // С€Р°Рі Р±РѕР»СЊС€Рµ Р·Р°РґРµСЂР¶РєРё, С‡С‚РѕР±С‹ С‡РёС‚Р°РµРјС‹Р№ СЃР»РѕС‚ Р±С‹Р» РёР· РїСЂРѕС€Р»РѕРіРѕ.
            let delay_steps = (*t_delay / dt_ms.max(1e-6)).round().max(0.0) as usize;
            let size = delay_steps + 1;
            GroupEngine::Alpha {
                indices: g.indices.clone(),
                pop,
                drive_scale: scale,
                rec_scale: rscale,
                delay_steps,
                delay_ring: vec![0.0; size * k.max(1)],
                delay_pos: 0,
            }
        }
        NeuronConfig::Lif {
            tau_ref,
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            ..
        }
        | NeuronConfig::LifSynaptic {
            tau_ref,
            tau_m,
            v_rest,
            v_thresh,
            v_reset,
            ..
        } => {
            // Lif/LifSynaptic вЂ” РєР°Рє alpha (conductance-РїРѕРґРѕР±РЅС‹Р№ РІС…РѕРґ) РґР»СЏ РїСЂРѕСЃС‚РѕС‚С‹.
            let pop = alpha_lif_population(
                k, dt_ms, *tau_ref, *tau_m, *v_rest, *v_thresh, *v_reset, 5.0,
            );
            GroupEngine::Alpha {
                indices: g.indices.clone(),
                pop,
                drive_scale: scale,
                rec_scale: rscale,
                delay_steps: 0,
                delay_ring: vec![0.0; k.max(1)],
                delay_pos: 0,
            }
        }
        NeuronConfig::Hh { .. } | NeuronConfig::HhSynaptic { .. } => {
            return Err(
                "HeteroRunner: HH удалён (операнды vtrap/fast_exp остались — собери свою модель через #[derive(NeuronModel)])"
                    .into(),
            );
        }
        NeuronConfig::Haap { .. } => {
            let pop = haap_population(k, dt_ms);
            let decay = 0.0; // РјРіРЅРѕРІРµРЅРЅС‹Р№ С‚РѕРє (РєР°Рє mono run_haap СЃ П„_syn=0)
            GroupEngine::Haap {
                indices: g.indices.clone(),
                pop,
                syn: vec![0.0; k],
                decay,
                drive_scale: scale,
                rec_scale: rscale,
            }
        }
        NeuronConfig::HaapSynaptic { tau_syn, .. } => {
            let pop = haap_population(k, dt_ms);
            let decay = if *tau_syn > 0.0 {
                (-dt_ms / tau_syn).exp()
            } else {
                0.0
            };
            GroupEngine::Haap {
                indices: g.indices.clone(),
                pop,
                syn: vec![0.0; k],
                decay,
                drive_scale: scale,
                rec_scale: rscale,
            }
        }
        NeuronConfig::Izh {
            dt_ms: idt,
            tau_ref,
            a,
            b,
            c,
            d,
        }
        | NeuronConfig::IzhSynaptic {
            dt_ms: idt,
            tau_ref,
            a,
            b,
            c,
            d,
        } => {
            let _ = idt; // РёСЃРїРѕР»СЊР·СѓРµРј РіР»РѕР±Р°Р»СЊРЅС‹Р№ dt_ms
                         // SIMD-СЏРґСЂРѕ Izh С‚СЂРµР±СѓРµС‚ РєСЂР°С‚РЅС‹Рµ 8 РґР»РёРЅС‹ вЂ” РїР°РґРґРёРј, РєР°Рє mono runner.
            let kp = k.div_ceil(8) * 8;
            GroupEngine::Izh {
                indices: g.indices.clone(),
                v: vec![-65.0; kp],
                u: vec![-13.0; kp],
                trace: vec![0.0; kp],
                refr: vec![0.0; kp],
                a: vec![*a; kp],
                b: vec![*b; kp],
                c: vec![*c; kp],
                d: vec![*d; kp],
                dt_ms,
                tau_ref: *tau_ref,
                drive_scale: scale,
                rec_scale: rscale,
            }
        }
        NeuronConfig::IzhDual { .. } | NeuronConfig::IzhEprop { .. } => {
            return Err("HeteroRunner: IzhDual/IzhEprop РЅРµ РїРѕРґРґРµСЂР¶Р°РЅС‹ (РёСЃРїРѕР»СЊР·СѓР№С‚Рµ Izh)".into());
        }
        NeuronConfig::Rate { tau, .. } => GroupEngine::Rate {
            indices: g.indices.clone(),
            v: vec![0.0; k],
            tau: *tau,
            dt_ms,
            drive_scale: scale,
            rec_scale: rscale,
        },
        NeuronConfig::Transformer {
            dt_ms: tdt,
            decay,
            gain,
            inv_tau_acc,
            thresh,
            k_scale,
            eta_k,
            eta_v,
            w_base,
            w_span,
            v_base,
            v_span,
        } => {
            let _ = tdt; // РјРѕРґРµР»СЊ РёСЃРїРѕР»СЊР·СѓРµС‚ ctx.dt() (РіР»РѕР±Р°Р»СЊРЅС‹Р№ dt_ms)
            let pop = transformer_population(
                k,
                dt_ms,
                *decay,
                *gain,
                *inv_tau_acc,
                *thresh,
                *k_scale,
                *eta_k,
                *eta_v,
                *w_base,
                *w_span,
                *v_base,
                *v_span,
            );
            GroupEngine::Transformer {
                indices: g.indices.clone(),
                pop,
                drive_scale: scale,
                rec_scale: rscale,
            }
        }
        NeuronConfig::Gco { .. } => {
            return Err("HeteroRunner: Gco РЅРµ РїРѕРґРґРµСЂР¶Р°РЅ (РёСЃРїРѕР»СЊР·СѓР№С‚Рµ Rate)".into());
        }
    };
    Ok(engine)
}

/// РЎРѕР±СЂР°С‚СЊ РїРѕРїСѓР»СЏС†РёСЋ SurgeN-С‚СЂР°РЅСЃС„РѕСЂРјРµСЂРѕРІ (pillars zoo `TransformerModel`).
/// Р’РµСЃР° QKV СЃС‚Р°СЂС‚СѓСЋС‚ РЅР° Р±Р°Р·Рµ (СЂР°РІРЅРѕРјРµСЂРЅРѕРµ РІРЅРёРјР°РЅРёРµ); РѕР±СѓС‡РµРЅРёРµ вЂ” РїРѕ eta.
fn transformer_population(
    k: usize,
    dt_ms: f32,
    decay: f32,
    gain: f32,
    inv_tau_acc: f32,
    thresh: f32,
    k_scale: f32,
    eta_k: f32,
    eta_v: f32,
    w_base: f32,
    w_span: f32,
    v_base: f32,
    v_span: f32,
) -> NeuralPopulation<TransformerModel> {
    let mut pop = NeuralPopulation::<TransformerModel>::new(k);
    pop.fill_param(TransformerModelParamIdx::dt.into(), dt_ms);
    pop.fill_param(TransformerModelParamIdx::decay.into(), decay);
    pop.fill_param(TransformerModelParamIdx::gain.into(), gain);
    pop.fill_param(TransformerModelParamIdx::inv_tau_acc.into(), inv_tau_acc);
    pop.fill_param(TransformerModelParamIdx::thresh.into(), thresh);
    pop.fill_param(TransformerModelParamIdx::k_scale.into(), k_scale);
    pop.fill_param(TransformerModelParamIdx::eta_k.into(), eta_k);
    pop.fill_param(TransformerModelParamIdx::eta_v.into(), eta_v);
    pop.fill_param(TransformerModelParamIdx::w_base.into(), w_base);
    pop.fill_param(TransformerModelParamIdx::w_span.into(), w_span);
    pop.fill_param(TransformerModelParamIdx::v_base.into(), v_base);
    pop.fill_param(TransformerModelParamIdx::v_span.into(), v_span);
    // РЎС‚Р°СЂС‚: СЂР°РІРЅРѕРјРµСЂРЅРѕРµ РІРЅРёРјР°РЅРёРµ (РІСЃРµ РєР»СЋС‡Рё/Р·РЅР°С‡РµРЅРёСЏ РЅР° Р±Р°Р·Рµ).
    for i in 0..8 {
        pop.fill_state(TransformerModelStateIdx::w0 as usize + i, w_base);
        pop.fill_state(TransformerModelStateIdx::v0 as usize + i, v_base);
    }
    pop
}

/// Inject Poisson-group events into an arbitrary buffer (РѕР±С‰Р°СЏ Р»РѕРіРёРєР°).
fn inject_poisson_into(
    groups: &[PoissonGroup],
    rng: &mut rand::rngs::StdRng,
    buf: &mut [f32],
    dt_ms: f32,
) {
    for pg in groups {
        if pg.rate_hz <= 0.0 || pg.weight == 0.0 {
            continue;
        }
        let p = pg.rate_hz * dt_ms / 1000.0;
        if p <= 0.0 {
            continue;
        }
        if p >= 1.0 {
            for &idx in &pg.indices {
                if idx < buf.len() {
                    buf[idx] += pg.weight;
                }
            }
        } else {
            for &idx in &pg.indices {
                if idx < buf.len() && rng.gen::<f32>() < p {
                    buf[idx] += pg.weight;
                }
            }
        }
    }
}

// в”Ђв”Ђв”Ђ РЎРЅР°РїС€РѕС‚С‹ СЃРѕСЃС‚РѕСЏРЅРёСЏ РіРµС‚РµСЂРѕРіРµРЅРЅРѕРіРѕ РјРѕР·РіР° в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// РџРѕР»РЅРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ РѕРґРЅРѕР№ РіСЂСѓРїРїС‹ РіРµС‚РµСЂРѕРіРµРЅРЅРѕРіРѕ РјРѕР·РіР° (С‡РёСЃС‚С‹Рµ РґР°РЅРЅС‹Рµ).
///
/// РџР°СЂР°РјРµС‚СЂС‹/СЃРѕСЃС‚РѕСЏРЅРёСЏ РјРѕРґРµР»РµР№ С…СЂР°РЅСЏС‚СЃСЏ РІ SoA-РєРѕР»РѕРЅРєР°С… (РєР°Рє
/// `NeuralPopulation`), РґР»РёРЅР° РєРѕР»РѕРЅРѕРє вЂ” SIMD-РїР°РґРґРёРЅРі (РєСЂР°С‚РЅР° 8). Р”Р»СЏ
/// Izh/Rate вЂ” РѕС‚РґРµР»СЊРЅС‹Рµ РјР°СЃСЃРёРІС‹. РЎРЅР°РїС€РѕС‚ РЅРµ СЃРµСЂРёР°Р»РёР·СѓРµРј СЃР°Рј РїРѕ СЃРµР±Рµ;
/// СЃРµСЂРёР°Р»РёР·Р°С†РёСЏ вЂ” РІ РєСЂРµР№С‚Рµ `taas-format`.
#[derive(Debug, Clone)]
pub enum HeteroGroupState {
    /// LIF + alpha-СЃРёРЅР°РїСЃ (Shiu et al. 2024).
    Alpha {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        /// N_PARAMS РєРѕР»РѕРЅРєРё (dt, time_factor, v_rest, v_thresh, v_reset,
        /// refr_steps, syn_decay).
        params: Vec<Vec<f32>>,
        /// N_STATE РєРѕР»РѕРЅРєРё (v, g, refr_cnt).
        state: Vec<Vec<f32>>,
        delay_ring: Vec<f32>,
        /// Р§РёСЃР»Рѕ СЃР»РѕС‚РѕРІ Р·Р°РґРµСЂР¶РєРё (steps_delay + 1).
        delay_size: usize,
        delay_pos: usize,
    },
    /// Hodgkin-Huxley (standard РёР»Рё pacemaking).
    Hh {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        params: Vec<Vec<f32>>,
        state: Vec<Vec<f32>>,
    },
    /// HAAP (M-С‚РѕРє + h-С‚РѕРє).
    Haap {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        params: Vec<Vec<f32>>,
        state: Vec<Vec<f32>>,
        /// РџРµСЂСЃРёСЃС‚РµРЅС‚РЅС‹Р№ СЃРёРЅР°РїС‚РёС‡РµСЃРєРёР№ С‚РѕРє.
        syn: Vec<f32>,
        decay: f32,
    },
    /// Izhikevich (regular spiking).
    Izh {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        v: Vec<f32>,
        u: Vec<f32>,
        trace: Vec<f32>,
        refr: Vec<f32>,
        a: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
        d: Vec<f32>,
        dt_ms: f32,
        tau_ref: f32,
    },
    /// Rate model.
    Rate {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        v: Vec<f32>,
        tau: f32,
        dt_ms: f32,
    },
    /// SurgeN вЂ” РЅРµР№СЂРѕРЅ-С‚СЂР°РЅСЃС„РѕСЂРјРµСЂ (pillars zoo `TransformerModel`).
    Transformer {
        indices: Vec<usize>,
        drive_scale: f32,
        rec_scale: f32,
        params: Vec<Vec<f32>>,
        state: Vec<Vec<f32>>,
    },
}

impl HeteroGroupState {
    /// РРЅРґРµРєСЃС‹ РЅРµР№СЂРѕРЅРѕРІ РіСЂСѓРїРїС‹ (РіР»РѕР±Р°Р»СЊРЅС‹Рµ).
    pub fn indices(&self) -> &[usize] {
        match self {
            HeteroGroupState::Alpha { indices, .. }
            | HeteroGroupState::Hh { indices, .. }
            | HeteroGroupState::Haap { indices, .. }
            | HeteroGroupState::Izh { indices, .. }
            | HeteroGroupState::Rate { indices, .. }
            | HeteroGroupState::Transformer { indices, .. } => indices,
        }
    }

    /// РРјСЏ С‚РёРїР° РіСЂСѓРїРїС‹ (РґР»СЏ РґРёР°РіРЅРѕСЃС‚РёРєРё/С„РѕСЂРјР°С‚Р°).
    pub fn kind(&self) -> &'static str {
        match self {
            HeteroGroupState::Alpha { .. } => "alpha",
            HeteroGroupState::Hh { .. } => "hh",
            HeteroGroupState::Haap { .. } => "haap",
            HeteroGroupState::Izh { .. } => "izh",
            HeteroGroupState::Rate { .. } => "rate",
            HeteroGroupState::Transformer { .. } => "transformer",
        }
    }
}

/// РџРѕР»РЅС‹Р№ СЃР»РµРїРѕРє СЃРѕСЃС‚РѕСЏРЅРёСЏ [HeteroRunner].
#[derive(Debug, Clone)]
pub struct HeteroSnapshot {
    pub connectome: ConnectomeData,
    pub step_counter: usize,
    pub dt_ms: f32,
    /// РЎРёРґ RNG РџСѓР°СЃСЃРѕРЅРѕРІСЃРєРёС… РіСЂСѓРїРї (СЃРј. РґРѕРє [`HeteroRunner`]).
    pub poisson_seed: [u8; 32],
    pub groups: Vec<HeteroGroupState>,
}

/// РљРѕР»РѕРЅРєРё РїР°СЂР°РјРµС‚СЂРѕРІ/СЃРѕСЃС‚РѕСЏРЅРёСЏ РјРѕРґРµР»Рё (SoA, РєР°Рє `NeuralPopulation`).
fn pop_columns<M: pillars::zoo::Model + 'static>(
    pop: &pillars::NeuralPopulation<M>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let params = (0..M::N_PARAMS)
        .map(|c| pop.param_slice(c).to_vec())
        .collect();
    let state = (0..M::N_STATE)
        .map(|c| pop.state_slice(c).to_vec())
        .collect();
    (params, state)
}

/// РџРµСЂРµСЃРѕР±СЂР°С‚СЊ РїРѕРїСѓР»СЏС†РёСЋ РёР· СЃРЅР°РїС€РѕС‚-РєРѕР»РѕРЅРѕРє.
fn pop_from_columns<M: pillars::zoo::Model + 'static>(
    params: &[Vec<f32>],
    state: &[Vec<f32>],
    n: usize,
    kind: &str,
) -> Result<pillars::NeuralPopulation<M>, String> {
    if params.len() != M::N_PARAMS || state.len() != M::N_STATE {
        return Err(format!(
            "hetero restore: {kind}: РєРѕР»РѕРЅРѕРє {}/{} != {}/{}",
            params.len(),
            state.len(),
            M::N_PARAMS,
            M::N_STATE
        ));
    }
    let mut pop = pillars::NeuralPopulation::<M>::new(n);
    for (c, col) in params.iter().enumerate() {
        let sl = pop.param_slice_mut(c);
        if col.len() > sl.len() {
            return Err(format!("hetero restore: {kind} p_{c} РґР»РёРЅР° {}", col.len()));
        }
        sl[..col.len()].copy_from_slice(col);
    }
    for (c, col) in state.iter().enumerate() {
        let sl = pop.state_slice_mut(c);
        if col.len() > sl.len() {
            return Err(format!("hetero restore: {kind} s_{c} РґР»РёРЅР° {}", col.len()));
        }
        sl[..col.len()].copy_from_slice(col);
    }
    Ok(pop)
}

impl HeteroRunner {
    /// РџРѕР»РЅС‹Р№ СЃР»РµРїРѕРє С‚РµРєСѓС‰РµРіРѕ СЃРѕСЃС‚РѕСЏРЅРёСЏ (РґР»СЏ save/load С‡РµСЂРµР· taas-format).
    ///
    /// Р’РµСЃР° `connectome.values` вЂ” С‚РµРєСѓС‰РёРµ (РїРѕСЃР»Рµ РїР»Р°СЃС‚РёС‡РЅРѕСЃС‚Рё), С‚Р°Рє С‡С‚Рѕ
    /// restore РІРѕР·РІСЂР°С‰Р°РµС‚ СЂРѕРІРЅРѕ С‚Рѕ Р¶Рµ СЃРѕСЃС‚РѕСЏРЅРёРµ, РІРєР»СЋС‡Р°СЏ РїР»Р°СЃС‚РёРєСѓ.
    pub fn snapshot(&self) -> HeteroSnapshot {
        let groups = self
            .groups
            .iter()
            .map(|g| match g {
                GroupEngine::Alpha {
                    indices,
                    pop,
                    drive_scale,
                    rec_scale,
                    delay_steps,
                    delay_ring,
                    delay_pos,
                } => {
                    let (params, state) = pop_columns(pop);
                    HeteroGroupState::Alpha {
                        indices: indices.clone(),
                        drive_scale: *drive_scale,
                        rec_scale: *rec_scale,
                        params,
                        state,
                        delay_ring: delay_ring.clone(),
                        delay_size: delay_steps + 1,
                        delay_pos: *delay_pos,
                    }
                }
                GroupEngine::HhStub {
                    indices,
                    drive_scale,
                    rec_scale,
                } => HeteroGroupState::Hh {
                    indices: indices.clone(),
                    drive_scale: *drive_scale,
                    rec_scale: *rec_scale,
                    params: vec![],
                    state: vec![],
                },
                GroupEngine::Haap {
                    indices,
                    pop,
                    syn,
                    decay,
                    drive_scale,
                    rec_scale,
                } => {
                    let (params, state) = pop_columns(pop);
                    HeteroGroupState::Haap {
                        indices: indices.clone(),
                        drive_scale: *drive_scale,
                        rec_scale: *rec_scale,
                        params,
                        state,
                        syn: syn.clone(),
                        decay: *decay,
                    }
                }
                GroupEngine::Izh {
                    indices,
                    v,
                    u,
                    trace,
                    refr,
                    a,
                    b,
                    c,
                    d,
                    dt_ms,
                    tau_ref,
                    drive_scale,
                    rec_scale,
                } => HeteroGroupState::Izh {
                    indices: indices.clone(),
                    drive_scale: *drive_scale,
                    rec_scale: *rec_scale,
                    v: v.clone(),
                    u: u.clone(),
                    trace: trace.clone(),
                    refr: refr.clone(),
                    a: a.clone(),
                    b: b.clone(),
                    c: c.clone(),
                    d: d.clone(),
                    dt_ms: *dt_ms,
                    tau_ref: *tau_ref,
                },
                GroupEngine::Rate {
                    indices,
                    v,
                    tau,
                    dt_ms,
                    drive_scale,
                    rec_scale,
                } => HeteroGroupState::Rate {
                    indices: indices.clone(),
                    drive_scale: *drive_scale,
                    rec_scale: *rec_scale,
                    v: v.clone(),
                    tau: *tau,
                    dt_ms: *dt_ms,
                },
                GroupEngine::Transformer {
                    indices,
                    pop,
                    drive_scale,
                    rec_scale,
                } => {
                    let (params, state) = pop_columns(pop);
                    HeteroGroupState::Transformer {
                        indices: indices.clone(),
                        drive_scale: *drive_scale,
                        rec_scale: *rec_scale,
                        params,
                        state,
                    }
                }
            })
            .collect();
        HeteroSnapshot {
            connectome: self.connectome.clone(),
            step_counter: self.step_counter,
            dt_ms: self.dt_ms,
            poisson_seed: self.poisson_seed,
            groups,
        }
    }

    /// Р’РѕСЃСЃС‚Р°РЅРѕРІРёС‚СЊ СЃРѕСЃС‚РѕСЏРЅРёРµ СЂРѕРІРЅРѕ РёР· СЃРЅР°РїС€РѕС‚Р°.
    ///
    /// РџРµСЂРµСЃРѕР±РёСЂР°РµС‚ РіСЂСѓРїРїС‹ РёР· СЃРЅР°РїС€РѕС‚Р° (РїР°СЂР°РјРµС‚СЂС‹+СЃРѕСЃС‚РѕСЏРЅРёРµ), Р·Р°РјРµРЅСЏРµС‚
    /// РєРѕРЅРЅРµРєС‚РѕРј (РІ.С‡. РІРµСЃР° РїРѕСЃР»Рµ РїР»Р°СЃС‚РёС‡РЅРѕСЃС‚Рё) Рё СЃР±СЂР°СЃС‹РІР°РµС‚ РѕС‡РµСЂРµРґРё.
    /// РџСѓР°СЃСЃРѕРЅ РїРѕСЃР»Рµ `restore` СЃС‚Р°СЂС‚СѓРµС‚ СЃ `poisson_seed` (СЃРј. РґРѕРє РјРѕРґСѓР»СЏ).
    pub fn restore(&mut self, snap: &HeteroSnapshot) -> Result<(), String> {
        let n = snap.connectome.n_neurons;
        if n != self.n {
            return Err(format!("hetero restore: n_neurons {} != {}", n, self.n));
        }
        if snap.dt_ms != self.dt_ms {
            return Err(format!(
                "hetero restore: dt_ms {} != {}",
                snap.dt_ms, self.dt_ms
            ));
        }

        // Р’Р°Р»РёРґРёСЂСѓРµРј РїРѕРєСЂС‹С‚РёРµ РіСЂСѓРїРїР°РјРё РєР°Рє РІ `new`.
        let mut covered = vec![false; n];
        let mut engines = Vec::with_capacity(snap.groups.len());
        let mut specs = Vec::with_capacity(snap.groups.len());
        for gs in &snap.groups {
            if gs.indices().is_empty() {
                continue;
            }
            for &idx in gs.indices() {
                if idx >= n {
                    return Err(format!("hetero restore: index {idx} >= n={n}"));
                }
                if covered[idx] {
                    return Err(format!("hetero restore: neuron {idx} covered twice"));
                }
                covered[idx] = true;
            }
            let (eng, spec) = engine_from_state(gs, snap.dt_ms)?;
            specs.push(spec);
            engines.push(eng);
        }

        // РљР°РєРёРµ-С‚Рѕ РЅРµР№СЂРѕРЅС‹ РЅРµ РїРѕРєСЂС‹С‚С‹ вЂ” РѕС‚РєР»РѕРЅСЏРµРј (РєР°Рє РІ new).
        if let Some(idx) = covered.iter().position(|&c| !c) {
            return Err(format!("hetero restore: neuron {idx} is not covered"));
        }

        self.connectome = snap.connectome.clone();
        self.csr_graph = ValidatedCsrGraph::from_slices(
            &self.connectome.row_ptr,
            &self.connectome.col_indices,
            n,
        );
        self.specs = specs;
        self.groups = engines;
        self.step_counter = snap.step_counter;
        self.poisson_seed = snap.poisson_seed;
        self.rng = rand::rngs::StdRng::from_seed(snap.poisson_seed);
        self.queue.clear();
        self.rec.fill(0.0);
        self.drive.fill(0.0);
        self.spikes.fill(0);
        self.spike_buffer.fill(0);
        Ok(())
    }
}

/// РџРµСЂРµСЃРѕР±СЂР°С‚СЊ РѕРґРёРЅ GroupEngine РёР· СЃРЅР°РїС€РѕС‚-СЃРѕСЃС‚РѕСЏРЅРёСЏ РіСЂСѓРїРїС‹.
fn engine_from_state(
    gs: &HeteroGroupState,
    dt_ms: f32,
) -> Result<(GroupEngine, HeteroGroup), String> {
    let k = gs.indices().len();
    let spec = HeteroGroup {
        config: match gs {
            // РљРѕРЅС„РёРі РЅСѓР¶РµРЅ С‚РѕР»СЊРєРѕ РґР»СЏ specs; СЃР°Рј engine СЃС‚СЂРѕРёРј РїСЂСЏРјС‹РјРё РїРѕР»СЏРјРё.
            HeteroGroupState::Alpha { .. } => NeuronConfig::LifAlpha {
                dt_ms,
                tau_ref: 2.2,
                tau_m: 20.0,
                v_rest: -52.0,
                v_thresh: -45.0,
                v_reset: -52.0,
                tau_syn: 5.0,
                t_delay: 1.8,
            },
            HeteroGroupState::Hh { .. } => NeuronConfig::Hh {
                dt_ms,
                use_pacemaking: false,
            },
            HeteroGroupState::Haap { .. } => NeuronConfig::HaapSynaptic {
                dt_ms,
                tau_syn: 12.0,
            },
            HeteroGroupState::Izh { .. } => NeuronConfig::Izh {
                dt_ms,
                tau_ref: 2.0,
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 8.0,
            },
            HeteroGroupState::Rate { .. } => NeuronConfig::Rate { dt_ms, tau: 0.0 },
            HeteroGroupState::Transformer { .. } => NeuronConfig::Transformer {
                dt_ms,
                decay: 0.95,
                gain: 1.0,
                inv_tau_acc: 0.5,
                thresh: 0.35,
                k_scale: 0.5,
                eta_k: 0.0,
                eta_v: 0.0,
                w_base: 1.0,
                w_span: 2.0,
                v_base: 1.0,
                v_span: 2.0,
            },
        },
        indices: gs.indices().to_vec(),
        drive_scale: gs_drive(gs),
        rec_scale: gs_rec(gs),
    };
    let eng = match gs {
        HeteroGroupState::Alpha {
            indices,
            drive_scale,
            rec_scale,
            params,
            state,
            delay_ring,
            delay_size,
            delay_pos,
        } => {
            let pop = pop_from_columns::<pillars::zoo::AlphaLifModel>(params, state, k, "alpha")?;
            GroupEngine::Alpha {
                indices: indices.clone(),
                pop,
                drive_scale: *drive_scale,
                rec_scale: *rec_scale,
                delay_steps: delay_size.saturating_sub(1),
                delay_ring: delay_ring.clone(),
                delay_pos: *delay_pos % (*delay_size).max(1),
            }
        }
        HeteroGroupState::Hh { .. } => {
            return Err(
                "hetero restore: HH удалён — снапшот HH не поддерживается (используй LIF/Izh)".into(),
            );
        }
        HeteroGroupState::Haap {
            indices,
            drive_scale,
            rec_scale,
            params,
            state,
            syn,
            decay,
        } => {
            let pop = pop_from_columns::<pillars::zoo::HAAPModel>(params, state, k, "haap")?;
            GroupEngine::Haap {
                indices: indices.clone(),
                pop,
                syn: syn.clone(),
                decay: *decay,
                drive_scale: *drive_scale,
                rec_scale: *rec_scale,
            }
        }
        HeteroGroupState::Izh {
            indices,
            drive_scale,
            rec_scale,
            v,
            u,
            trace,
            refr,
            a,
            b,
            c,
            d,
            dt_ms,
            tau_ref,
        } => GroupEngine::Izh {
            indices: indices.clone(),
            v: v.clone(),
            u: u.clone(),
            trace: trace.clone(),
            refr: refr.clone(),
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
            d: d.clone(),
            dt_ms: *dt_ms,
            tau_ref: *tau_ref,
            drive_scale: *drive_scale,
            rec_scale: *rec_scale,
        },
        HeteroGroupState::Rate {
            indices,
            drive_scale,
            rec_scale,
            v,
            tau,
            dt_ms,
        } => GroupEngine::Rate {
            indices: indices.clone(),
            v: v.clone(),
            tau: *tau,
            dt_ms: *dt_ms,
            drive_scale: *drive_scale,
            rec_scale: *rec_scale,
        },
        HeteroGroupState::Transformer {
            indices,
            drive_scale,
            rec_scale,
            params,
            state,
        } => {
            let pop = pop_from_columns::<pillars::zoo::TransformerModel>(
                params,
                state,
                k,
                "transformer",
            )?;
            GroupEngine::Transformer {
                indices: indices.clone(),
                pop,
                drive_scale: *drive_scale,
                rec_scale: *rec_scale,
            }
        }
    };
    Ok((eng, spec))
}

fn gs_drive(gs: &HeteroGroupState) -> f32 {
    match gs {
        HeteroGroupState::Alpha { drive_scale, .. }
        | HeteroGroupState::Hh { drive_scale, .. }
        | HeteroGroupState::Haap { drive_scale, .. }
        | HeteroGroupState::Izh { drive_scale, .. }
        | HeteroGroupState::Rate { drive_scale, .. }
        | HeteroGroupState::Transformer { drive_scale, .. } => *drive_scale,
    }
}

fn gs_rec(gs: &HeteroGroupState) -> f32 {
    match gs {
        HeteroGroupState::Alpha { rec_scale, .. }
        | HeteroGroupState::Hh { rec_scale, .. }
        | HeteroGroupState::Haap { rec_scale, .. }
        | HeteroGroupState::Izh { rec_scale, .. }
        | HeteroGroupState::Rate { rec_scale, .. }
        | HeteroGroupState::Transformer { rec_scale, .. } => *rec_scale,
    }
}

// в”Ђв”Ђв”Ђ РўРµСЃС‚С‹ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConnectomeData;

    /// 4 РЅРµР№СЂРѕРЅР°: 0в†’1, 1в†’2, 2в†’3, 3в†’0 (РєРѕР»СЊС†Рѕ). Р’РµСЃР° 0.8/0.5/0.3/0.7.
    fn ring() -> ConnectomeData {
        ConnectomeData::new(
            vec![0i32, 1, 2, 3, 4],
            vec![1, 2, 3, 0],
            vec![0.8f32, 0.5, 0.3, 0.7],
            vec![1.0, 1.0, -1.0, 1.0],
            vec![("a", vec![0, 1]), ("b", vec![2, 3])],
        )
    }

    /// Р“РµС‚РµСЂРѕРіРµРЅРЅС‹Р№ РјРѕР·Рі: С†РµРїРѕС‡РєР° 0(alpha)в†’1(Izh)в†’2(alpha) СЃ СЃРёР»СЊРЅС‹РјРё РІРµСЃР°РјРё
    /// + Rate-РЅРµР№СЂРѕРЅ 3, Р¶РёРІСѓС‰РёР№ СЂСЏРґРѕРј. РЎРїР°Р№Рє alpha-0 РїСЂРѕР±РёРІР°РµС‚ Izh-1, С‚РѕС‚ вЂ”
    /// alpha-2: СЃРёРіРЅР°Р» РїРµСЂРµСЃРµРєР°РµС‚ РіСЂР°РЅРёС†С‹ РјРѕРґРµР»РµР№ РІ РѕРґРЅРѕРј РєРѕРЅРЅРµРєС‚РѕРјРµ.
    #[test]
    fn heterogeneous_mixed_models_share_connectome() {
        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2, 2],
            vec![1, 2],
            vec![40.0f32, 40.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![],
        );
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 2],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::izh(0.5),
                indices: vec![1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::rate(0.5, 0.001),
                indices: vec![3],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let mut runner = HeteroRunner::new(conn, groups, 0.5).unwrap();
        // Rate-3 РїРѕР»СѓС‡Р°РµС‚ СЃРѕР±СЃС‚РІРµРЅРЅС‹Р№ РґСЂР°Р№РІ (РєР°Рє СЃРµРЅСЃРѕСЂРЅС‹Р№ РєР°РЅР°Р» РїРѕРїСѓР»СЏС†РёРё).
        let ext = vec![30.0f32, 0.0, 0.0, 1.0];
        let spikes = runner.run(300, &ext).to_vec();

        let s0: i32 = spikes.iter().step_by(4).copied().sum();
        let s1: i32 = spikes.iter().skip(1).step_by(4).copied().sum();
        let s2: i32 = spikes.iter().skip(2).step_by(4).copied().sum();
        let s3: i32 = spikes.iter().skip(3).step_by(4).copied().sum();
        assert!(s0 > 0, "alpha 0 should fire, got {s0}");
        assert!(s1 > 0, "Izh 1 should fire from alpha 0, got {s1}");
        assert!(
            s2 > 0,
            "alpha 2 must fire from Izh 1 (cross-model), got {s2}"
        );
        assert!(s3 > 0, "Rate 3 should fire from its own drive, got {s3}");
    }

    /// РЎРїР°Р№Рє РёР· alpha-РіСЂСѓРїРїС‹ СЂРµР°Р»СЊРЅРѕ РґРѕС…РѕРґРёС‚ РґРѕ HH-РіСЂСѓРїРїС‹ (С‚РѕРЅРєРёР№ С€Р°Рі dt,
    /// HH С‡РёСЃР»РµРЅРЅРѕ РЅРµСЃС‚Р°Р±РёР»РµРЅ РїСЂРё 0.5): С†РµРїРѕС‡РєР° 0(alpha)в†’1(HH)в†’2(alpha).
    /// РЎРїР°Р№Рє РёР· alpha-РіСЂСѓРїРїС‹ СЂРµР°Р»СЊРЅРѕ РґРѕС…РѕРґРёС‚ РґРѕ Transformer-РіСЂСѓРїРїС‹ Рё РёР· РЅРµС‘
    /// вЂ” РІ СЃР»РµРґСѓСЋС‰РёР№ alpha: С†РµРїРѕС‡РєР° 0(alpha)в†’1(transformer)в†’2(alpha).
    /// РўСЂР°РЅСЃС„РѕСЂРјРµСЂ РґРµСЂР¶РёС‚ РїР°РјСЏС‚СЊ РІС…РѕРґРѕРІ Рё СЃРїР°Р№РєР°РµС‚ РЅР° СѓСЃС‚РѕР№С‡РёРІРѕРј СЃРёРіРЅР°Р»Рµ.
    #[test]
    fn cross_model_propagation_alpha_transformer_alpha() {
        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2],
            vec![1, 2],
            vec![40.0f32, 40.0],
            vec![1.0, 1.0, 1.0],
            vec![],
        );
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 2],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::transformer(0.5),
                indices: vec![1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let mut runner = HeteroRunner::new(conn, groups, 0.5).unwrap();
        let ext = vec![30.0f32, 0.0, 0.0];
        let spikes = runner.run(3000, &ext).to_vec();

        let s0: i32 = spikes.iter().step_by(3).copied().sum();
        let s1: i32 = spikes.iter().skip(1).step_by(3).copied().sum();
        let s2: i32 = spikes.iter().skip(2).step_by(3).copied().sum();
        assert!(s0 > 0, "drive neuron 0 should fire, got {s0}");
        assert!(
            s1 > 0,
            "transformer neuron 1 should fire from alpha drive, got {s1}"
        );
        assert!(
            s2 > 0,
            "alpha neuron 2 must fire from transformer neuron 1 (cross-model), got {s2}"
        );
    }

    /// Round-trip СЃРЅР°РїС€РѕС‚Р°: РїСЂРѕРіРѕРЅ СЃ `restore` РЅР° РїРѕР»РїСѓС‚Рё Р±РёС‚-РІ-Р±РёС‚
    /// СЃРѕРІРїР°РґР°РµС‚ СЃ РЅРµРїСЂРµСЂС‹РІРЅС‹Рј РїСЂРѕРіРѕРЅРѕРј (Р±РµР· РџСѓР°СЃСЃРѕРЅР°). РЎРѕСЃС‚РѕСЏРЅРёСЏ РІСЃРµС…
    /// РіСЂСѓРїРї (incl. delay ring Рё РїРµСЂСЃРёСЃС‚РµРЅС‚РЅС‹С… С‚РѕРєРѕРІ) РІРѕСЃСЃС‚Р°РЅР°РІР»РёРІР°СЋС‚СЃСЏ.
    #[test]
    fn snapshot_restore_roundtrip_bitwise() {
        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2, 2],
            vec![1, 2],
            vec![40.0f32, 40.0],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![],
        );
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 2],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::izh(0.5),
                indices: vec![1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::rate(0.5, 0.001),
                indices: vec![3],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let ext = vec![30.0f32, 0.0, 0.0, 1.0];

        // РљРѕРЅС‚СЂРѕР»СЊ: РЅРµРїСЂРµСЂС‹РІРЅС‹Р№ РїСЂРѕРіРѕРЅ 300 С€Р°РіРѕРІ.
        let mut a = HeteroRunner::new(conn.clone(), groups.clone(), 0.5).unwrap();
        let full = a.run(300, &ext).to_vec();

        // РЎРЅР°РїС€РѕС‚ РЅР° РїРѕР»РїСѓС‚Рё: 200 С€Р°РіРѕРІ в†’ СЃРЅРёРјРѕРє в†’ restore в†’ РµС‰С‘ 100.
        let mut b = HeteroRunner::new(conn, groups, 0.5).unwrap();
        let first = b.run(200, &ext).to_vec();
        let snap = b.snapshot();
        // РџСЂРѕРІРµСЂСЏРµРј, С‡С‚Рѕ РІРµСЃР° РїРѕСЃР»Рµ РїР»Р°СЃС‚РёС‡РЅРѕСЃС‚Рё/СЃРёРјСѓР»СЏС†РёРё = РєР°Рє РІ РєРѕРЅС‚СЂРѕР»Рµ.
        assert_eq!(snap.connectome.values, a.connectome.values);
        assert_eq!(snap.step_counter, 200);
        b.restore(&snap).unwrap();
        let second = b.run(100, &ext).to_vec();

        // РЎРєР»РµР№РєР° = РЅРµРїСЂРµСЂС‹РІРЅС‹Р№ РїСЂРѕРіРѕРЅ.
        assert_eq!(first.len() + second.len(), full.len());
        for i in 0..first.len() {
            assert_eq!(first[i], full[i], "first segment mismatch at {i}");
        }
        for (j, &s) in second.iter().enumerate() {
            assert_eq!(s, full[200 * 4 + j], "second segment mismatch at {j}");
        }
        assert_eq!(b.step_counter(), 300);
    }

    /// РџСѓСЃС‚РѕРµ РїРѕРєСЂС‹С‚РёРµ РЅРµР№СЂРѕРЅР° РІ СЃРЅР°РїС€РѕС‚Рµ вЂ” РѕС€РёР±РєР° restore.
    #[test]
    fn restore_rejects_uncovered() {
        let conn = ring();
        let groups = vec![HeteroGroup {
            config: crate::BrainConfig::lif_alpha(0.5),
            indices: vec![0, 1, 2, 3],
            drive_scale: 1.0,
            rec_scale: 1.0,
        }];
        let mut r = HeteroRunner::new(conn, groups, 0.5).unwrap();
        r.run(10, &[1.0; 4]);
        let snap = r.snapshot();
        // Р›РѕРјР°РµРј РїРѕРєСЂС‹С‚РёРµ: СѓР±РёСЂР°РµРј РЅРµР№СЂРѕРЅ 3 РёР· РіСЂСѓРїРїС‹.
        let mut bad = snap.clone();
        if let HeteroGroupState::Alpha { indices, .. } = &mut bad.groups[0] {
            indices.pop();
        }
        assert!(r.restore(&bad).is_err());
    }

    /// РџСѓСЃС‚РѕРµ РїРѕРєСЂС‹С‚РёРµ РЅРµР№СЂРѕРЅР° вЂ” РѕС€РёР±РєР°.
    #[test]
    fn overlapping_groups_rejected() {
        let conn = ring();
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::rate(0.5, 10.0),
                indices: vec![1, 2],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        assert!(HeteroRunner::new(conn, groups, 0.5).is_err());
    }

    /// РџСѓСЃС‚РѕР№ РєРѕРЅРЅРµРєС‚РѕРј вЂ” РѕС€РёР±РєР°.
    #[test]
    fn empty_connectome_rejected() {
        let conn = ConnectomeData::new(vec![0i32], vec![], vec![], vec![], vec![]);
        let groups = vec![HeteroGroup {
            config: crate::BrainConfig::lif_alpha(0.5),
            indices: vec![0],
            drive_scale: 1.0,
            rec_scale: 1.0,
        }];
        assert!(HeteroRunner::new(conn, groups, 0.5).is_err());
    }

    /// Poisson-СЃС‚РёРјСѓР»СЏС†РёСЏ РІ РіРµС‚РµСЂРѕРіРµРЅРЅРѕРј РјРѕР·РіРµ.
    #[test]
    fn poisson_drives_hetero_brain() {
        let conn = ring();
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.5),
                indices: vec![0, 1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::rate(0.5, 10.0),
                indices: vec![2, 3],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let mut runner = HeteroRunner::new(conn, groups, 0.5).unwrap();
        runner.add_poisson_group(&[0, 1], 500.0, 50.0);
        let spikes = runner.run(100, &[0.0; 4]).to_vec();
        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "poisson-only should fire neurons, got {total}");
    }

    /// HAAP-РіСЂСѓРїРїР° СЃ РїРµСЂСЃРёСЃС‚РµРЅС‚РЅС‹Рј СЃРёРЅР°РїС‚РёС‡РµСЃРєРёРј С‚РѕРєРѕРј (П„_syn > 0).
    #[test]
    fn haap_group_runs_with_synaptic_decay() {
        // 0(alpha) в†’ 1(haap), РІРµСЃ 40 (HAAP-РїРѕСЂРѕРі ~40 РјР’ РѕС‚ РїРѕРєРѕСЏ в€’65).
        let conn = ConnectomeData::new(
            vec![0i32, 1, 1],
            vec![1],
            vec![40.0f32],
            vec![1.0, 1.0, 1.0],
            vec![],
        );
        let groups = vec![
            HeteroGroup {
                config: crate::BrainConfig::lif_alpha(0.05),
                indices: vec![0],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
            HeteroGroup {
                config: crate::BrainConfig::haap_synaptic(0.05, 12.0),
                indices: vec![1],
                drive_scale: 1.0,
                rec_scale: 1.0,
            },
        ];
        let mut runner = HeteroRunner::new(conn, groups, 0.05).unwrap();
        let ext = vec![30.0f32, 0.0];
        let spikes = runner.run(2000, &ext).to_vec();
        let s0: i32 = spikes.iter().step_by(2).copied().sum();
        let s1: i32 = spikes.iter().skip(1).step_by(2).copied().sum();
        assert!(s0 > 0, "alpha drive should fire, got {s0}");
        assert!(s1 > 0, "HAAP should fire from alpha input, got {s1}");
    }
}

