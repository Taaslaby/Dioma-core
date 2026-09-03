//! # Чистая ДНК-машина — единый геном
//!
//! Один источник правды: `Genome` — вектор типизированных генов `Gene`.
//! Никаких `CreatureGenome` + `Genome` + `sync_legacy`. Проводка, инстинкты,
//! метаболизм и расцветка — гены одного списка.
//!
//! ## Идеи (как в Creatures, но проще)
//! * **Ген = шапка + полезная нагрузка.** Шапка несёт `kind` и `variant`,
//!   нагрузка — `Payload`. Неизвестный `kind` читается как `Raw` и сохраняется.
//! * **ДНК — `DNA1` байты.** `to_bytes` / `from_bytes` — LE, детерминирован,
//!   `Raw` не теряется. Версия `1`.
//! * **Эволюция — чистые функции.** `mutate(rng)` и `crossover(other,rng)`
//!   — детерминированы от сида, без глобального состояния.
//! * **Один `Genome::wildtype()`** — дикий тип с 5 генами (рост, правила,
//!   инстинкты, метаболизм, расцветка). Новый класс добавляешь в `GeneKind`
//!   — старые сейвы читаются как `Raw`.
//!
//! ```rust
//! use dioma_engine::creature::dna_clean::{Genome, GeneKind};
//! use dioma_engine::rng::XorShift;
//! let mut rng = XorShift::new(42);
//! let parent = Genome::wildtype();
//! let child = parent.mutate(&mut rng);
//! let bytes = child.to_bytes();
//! let back = Genome::from_bytes(&bytes).unwrap();
//! assert_eq!(child, back);
//! ```

use crate::brain::grow::Steering;
use crate::brain::instincts::Instinct;
use crate::brain::rules::RuleGene;
use crate::creature::genome::MetabolismGene;
use crate::rng::XorShift;

// ── Kind ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeneKind {
    Steering,    // 0 — рост коннектома
    Rules,       // 1 — Пифия
    Instincts,   // 2 — рефлексы
    Metabolism,  // 3 — кровь
    Regulatory,  // 4 — регулятор (сила механизмов)
    Coat,        // 5 — расцветка
    Eye,         // 6 — глаз (ANN variant)
    Memory,      // 7 — память (LSTM variant)
    Raw(u16),    // неизвестный — сохраняется
}

impl GeneKind {
    pub fn code(self) -> u16 {
        match self { Self::Steering=>0, Self::Rules=>1, Self::Instincts=>2, Self::Metabolism=>3, Self::Regulatory=>4, Self::Coat=>5, Self::Eye=>6, Self::Memory=>7, Self::Raw(c)=>c }
    }
    pub fn from_code(c: u16) -> Self {
        match c { 0=>Self::Steering, 1=>Self::Rules, 2=>Self::Instincts, 3=>Self::Metabolism, 4=>Self::Regulatory, 5=>Self::Coat, 6=>Self::Eye, 7=>Self::Memory, x=>Self::Raw(x) }
    }
    pub fn label(self)->&'static str {
        match self { Self::Steering=>"steering", Self::Rules=>"rules", Self::Instincts=>"instincts", Self::Metabolism=>"metabolism", Self::Regulatory=>"regulatory", Self::Coat=>"coat", Self::Eye=>"eye", Self::Memory=>"memory", Self::Raw(_)=>"raw" }
    }
}

// ── Payload ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum Payload {
    Steering(Steering),
    Rules(Vec<RuleGene>),
    Instincts(Vec<Instinct>),
    Metabolism(MetabolismGene),
    Regulatory(Vec<u8>), // упрощённо: байты регуляторных генов
    Coat(Coat),
    Eye(Eye),
    Memory(Memory),
    Raw(Vec<u8>),
}

#[derive(Clone, Copy, Debug, PartialEq)] pub struct Eye { pub variant: u16, pub gain: f32 }
impl Eye {
    pub fn new(v:u16,g:f32)->Self{ Self{variant:v, gain:g.max(0.0)} }
    fn mutate(&self, rng:&mut XorShift)->Self{ Self{variant:self.variant, gain:(self.gain + rng.gauss() as f32*0.1).max(0.0)} }
}
#[derive(Clone, Copy, Debug, PartialEq)] pub struct Memory { pub variant: u16, pub tau: f32 }
impl Memory {
    pub fn new(v:u16,t:f32)->Self{ Self{variant:v, tau:t.max(0.1)} }
    fn mutate(&self, rng:&mut XorShift)->Self{ Self{variant:self.variant, tau:(self.tau + rng.gauss() as f32*0.05).max(0.1)} }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coat { pub r:f32, pub g:f32, pub b:f32 }
impl Coat {
    pub fn new(r:f32,g:f32,b:f32)->Self{ Self{r:r.clamp(0.0,1.0), g:g.clamp(0.0,1.0), b:b.clamp(0.0,1.0)} }
    fn mutate(&self, rng:&mut XorShift)->Self{
        Self::new((self.r + rng.gauss() as f32 *0.08).clamp(0.0,1.0),
                  (self.g + rng.gauss() as f32 *0.08).clamp(0.0,1.0),
                  (self.b + rng.gauss() as f32 *0.08).clamp(0.0,1.0))
    }
    fn crossover(&self, other:&Self, rng:&mut XorShift)->Self{
        Self::new(if rng.next_u64()%2==0{self.r}else{other.r},
                  if rng.next_u64()%2==0{self.g}else{other.g},
                  if rng.next_u64()%2==0{self.b}else{other.b})
    }
}

// ── Gene ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct Gene {
    pub kind: GeneKind,
    pub payload: Payload,
    pub variant: u8,      // вариант гена (для Eye/Memory)
    pub express_at: u64,  // онтогенез: с какого тика экспрессируется
}

impl Gene {
    pub fn new(kind:GeneKind, payload:Payload)->Self{ Self{kind,payload, variant:0, express_at:0} }
    pub fn with_variant(mut self, v:u8)->Self{ self.variant=v; self }
    pub fn expressed_at(&self, age:u64)->bool{ age >= self.express_at }
}

// ── Genome ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct Genome {
    pub genes: Vec<Gene>,
}

impl Genome {
    /// Дикий тип: 7 генов — полный базовый набор для Unity-редактора.
    pub fn wildtype() -> Self {
        Self { genes: vec![
            Gene::new(GeneKind::Steering, Payload::Steering(Steering::default())),
            Gene::new(GeneKind::Rules, Payload::Rules(Vec::new())),
            Gene::new(GeneKind::Instincts, Payload::Instincts(Vec::new())),
            Gene::new(GeneKind::Metabolism, Payload::Metabolism(MetabolismGene::default())),
            Gene::new(GeneKind::Regulatory, Payload::Regulatory(vec![128])),
            Gene::new(GeneKind::Coat, Payload::Coat(Coat::new(0.9,0.9,0.9))),
            Gene::new(GeneKind::Eye, Payload::Eye(Eye::new(0,1.0))),
        ]}
    }

    /// Builder — цепочка `Genome::builder().coat(...).eye(...).build()`
    pub fn builder() -> GenomeBuilder { GenomeBuilder::new() }

    pub fn get(&self, kind:GeneKind) -> Option<&Gene> {
        self.genes.iter().find(|g| g.kind==kind)
    }
    pub fn get_mut(&mut self, kind:GeneKind) -> Option<&mut Gene> {
        self.genes.iter_mut().find(|g| g.kind==kind)
    }

    /// Мутация — каждый ген дрейфует независимо, детерминированно.
    pub fn mutate(&self, rng:&mut XorShift) -> Self {
        let mut out = self.clone();
        for g in &mut out.genes {
            g.payload = match &g.payload {
                Payload::Steering(s) => Payload::Steering(s.mutate(rng, 0.08)),
                Payload::Rules(rs) => {
                    let mut v = rs.clone();
                    for r in &mut v {
                        let mut p = r.p;
                        p[0] = (p[0] + rng.gauss() as f32 * 0.1).clamp(0.0, 8.0);
                        *r = RuleGene { template: r.template, p };
                    }
                    Payload::Rules(v)
                },
                Payload::Instincts(ins) => {
                    let mut v = ins.clone();
                    for i in &mut v { *i = Instinct { inputs: i.inputs.clone(), dopamine: (i.dopamine + rng.gauss() as f32 *0.2).clamp(0.2, 8.0) }; }
                    Payload::Instincts(v)
                },
                Payload::Metabolism(m) => Payload::Metabolism(m.mutate(rng, 0.10)),
                Payload::Regulatory(v) => {
                    let mut n=v.clone(); for b in &mut n { *b = b.wrapping_add((rng.next_u64() as u8).wrapping_rem(5)); } Payload::Regulatory(n)
                },
                Payload::Coat(c) => Payload::Coat(c.mutate(rng)),
                Payload::Eye(e) => Payload::Eye(e.mutate(rng)),
                Payload::Memory(m) => Payload::Memory(m.mutate(rng)),
                Payload::Raw(b) => Payload::Raw(b.clone()),
            };
        }
        out
    }

    /// Кроссовер — для каждого `kind` берём ген одного из родителей.
    pub fn crossover(&self, other:&Self, rng:&mut XorShift) -> Self {
        let mut out = Vec::new();
        for kind in [GeneKind::Steering, GeneKind::Rules, GeneKind::Instincts, GeneKind::Metabolism, GeneKind::Regulatory, GeneKind::Coat, GeneKind::Eye, GeneKind::Memory] {
            let a = self.get(kind);
            let b = other.get(kind);
            let pick = match (a,b) {
                (Some(a), Some(b)) => if rng.next_u64()%2==0{a}else{b}.clone(),
                (Some(a), None) => a.clone(),
                (None, Some(b)) => b.clone(),
                (None, None) => continue,
            };
            // Coat — по-канальный кроссовер
            let gene = if kind==GeneKind::Coat {
                if let (Some(ga), Some(gb)) = (a, b) {
                    if let (Payload::Coat(ca), Payload::Coat(cb)) = (&ga.payload, &gb.payload) {
                        Gene::new(kind, Payload::Coat(ca.crossover(cb, rng)))
                    } else { pick }
                } else { pick }
            } else { pick };
            out.push(gene);
        }
        // Raw — сохраняем все
        for g in self.genes.iter().chain(other.genes.iter()).filter(|g| matches!(g.kind, GeneKind::Raw(_))) {
            if !out.iter().any(|x| x.kind==g.kind) { out.push(g.clone()); }
        }
        Self { genes: out }
    }

    // ── Сериализация DNA1 ────────────────────────────────────────────────
    const MAGIC: &[u8;4] = b"DNA1";
    const VER: u16 = 1;

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&Self::VER.to_le_bytes());
        out.extend_from_slice(&(self.genes.len() as u32).to_le_bytes());
        for g in &self.genes {
            out.extend_from_slice(&g.kind.code().to_le_bytes());
            out.push(g.variant);
            out.extend_from_slice(&g.express_at.to_le_bytes());
            match &g.payload {
                Payload::Steering(s) => {
                    out.push(0);
                    out.extend_from_slice(&(s.grow_steps as u32).to_le_bytes());
                    out.extend_from_slice(&(s.noise_weight as f32).to_le_bytes());
                },
                Payload::Rules(rs) => {
                    out.push(1);
                    out.extend_from_slice(&(rs.len() as u32).to_le_bytes());
                    for r in rs { out.extend_from_slice(&r.p[0].to_le_bytes()); }
                },
                Payload::Instincts(ins) => {
                    out.push(2);
                    out.extend_from_slice(&(ins.len() as u32).to_le_bytes());
                    for i in ins { out.extend_from_slice(&i.dopamine.to_le_bytes()); }
                },
                Payload::Metabolism(m) => {
                    out.push(3);
                    out.extend_from_slice(&m.cost_per_activity.to_le_bytes());
                },
                Payload::Regulatory(v) => {
                    out.push(4);
                    out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                    out.extend_from_slice(v);
                },
                Payload::Coat(c) => {
                    out.push(5);
                    out.extend_from_slice(&c.r.to_le_bytes());
                    out.extend_from_slice(&c.g.to_le_bytes());
                    out.extend_from_slice(&c.b.to_le_bytes());
                },
                Payload::Eye(e) => {
                    out.push(6);
                    out.extend_from_slice(&e.variant.to_le_bytes());
                    out.extend_from_slice(&e.gain.to_le_bytes());
                },
                Payload::Memory(m) => {
                    out.push(7);
                    out.extend_from_slice(&m.variant.to_le_bytes());
                    out.extend_from_slice(&m.tau.to_le_bytes());
                },
                Payload::Raw(b) => {
                    out.push(255);
                    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                    out.extend_from_slice(b);
                },
            }
        }
        out
    }

    pub fn from_bytes(bytes:&[u8]) -> Result<Self,String> {
        if bytes.len()<8 || &bytes[0..4]!=Self::MAGIC { return Err("bad DNA magic".into()); }
        let ver = u16::from_le_bytes([bytes[4],bytes[5]]);
        if ver!=Self::VER { return Err(format!("bad ver {ver}")); }
        let n = u32::from_le_bytes([bytes[6],bytes[7],bytes[8],bytes[9]]) as usize;
        let mut off=10;
        let mut genes=Vec::new();
        for _ in 0..n {
            if off+2>bytes.len(){ return Err("trunc".into()); }
            let kind = GeneKind::from_code(u16::from_le_bytes([bytes[off],bytes[off+1]])); off+=2;
            if off+9>bytes.len(){ return Err("trunc header".into()); }
            let variant = bytes[off]; off+=1;
            let express_at = u64::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3],bytes[off+4],bytes[off+5],bytes[off+6],bytes[off+7]]); off+=8;
            if off>=bytes.len(){ return Err("trunc".into()); }
            let tag=bytes[off]; off+=1;
            let payload = match tag {
                0 => {
                    if off+8>bytes.len(){ return Err("trunc steering".into()); }
                    let steps = u32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]) as usize; off+=4;
                    let noise = f32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]); off+=4;
                    let mut s = Steering::default(); s.grow_steps=steps; s.noise_weight=noise as f64;
                    Payload::Steering(s)
                },
                1 => {
                    if off+4>bytes.len(){ return Err("trunc rules".into()); }
                    let len = u32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]) as usize; off+=4;
                    let mut v=Vec::new();
                    for _ in 0..len {
                        if off+4>bytes.len(){ return Err("trunc".into()); }
                        let d=f32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]); off+=4;
                        v.push(RuleGene { template: 0, p: [d,0.0,0.0,0.0,0.0,0.0,0.0,0.0] });
                    }
                    Payload::Rules(v)
                },
                2 => {
                    if off+4>bytes.len(){ return Err("trunc inst".into()); }
                    let len = u32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]) as usize; off+=4;
                    let mut v=Vec::new();
                    for _ in 0..len {
                        if off+4>bytes.len(){ return Err("trunc".into()); }
                        let d=f32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]); off+=4;
                        v.push(Instinct { inputs: vec![], dopamine:d });
                    }
                    Payload::Instincts(v)
                },
                3 => {
                    if off+4>bytes.len(){ return Err("trunc metab".into()); }
                    let c=f32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]); off+=4;
                    let mut m=MetabolismGene::default(); m.cost_per_activity=c;
                    Payload::Metabolism(m)
                },
                4 => {
                    if off+4>bytes.len(){ return Err("trunc reg".into()); }
                    let len=u32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]) as usize; off+=4;
                    if off+len>bytes.len(){ return Err("trunc reg body".into()); }
                    let b=bytes[off..off+len].to_vec(); off+=len;
                    Payload::Regulatory(b)
                },
                5 => {
                    if off+12>bytes.len(){ return Err("trunc coat".into()); }
                    let r=f32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]); off+=4;
                    let g=f32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]); off+=4;
                    let b=f32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]); off+=4;
                    Payload::Coat(Coat::new(r,g,b))
                },
                6 => {
                    if off+6>bytes.len(){ return Err("trunc eye".into()); }
                    let v=u16::from_le_bytes([bytes[off],bytes[off+1]]); off+=2;
                    let g=f32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]); off+=4;
                    Payload::Eye(Eye::new(v,g))
                },
                7 => {
                    if off+6>bytes.len(){ return Err("trunc mem".into()); }
                    let v=u16::from_le_bytes([bytes[off],bytes[off+1]]); off+=2;
                    let t=f32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]); off+=4;
                    Payload::Memory(Memory::new(v,t))
                },
                255 => {
                    if off+4>bytes.len(){ return Err("trunc raw".into()); }
                    let len=u32::from_le_bytes([bytes[off],bytes[off+1],bytes[off+2],bytes[off+3]]) as usize; off+=4;
                    if off+len>bytes.len(){ return Err("trunc raw body".into()); }
                    let b=bytes[off..off+len].to_vec(); off+=len;
                    Payload::Raw(b)
                },
                _ => return Err(format!("unknown tag {tag}")),
            };
            genes.push(Gene{kind, payload, variant, express_at});
        }
        Ok(Self{genes})
    }

    // ── Красивые хелперы ─────────────────────────────────────────────────
    /// Человекочитаемое описание — для Unity-инспектора
    pub fn describe(&self) -> String {
        let mut s = String::new();
        for g in &self.genes {
            s.push_str(&format!("{}[v{}@{}]: ", g.kind.label(), g.variant, g.express_at));
            match &g.payload {
                Payload::Coat(c) => s.push_str(&format!("coat #{:02X}{:02X}{:02X}\n", (c.r*255.0) as u8, (c.g*255.0) as u8, (c.b*255.0) as u8)),
                Payload::Eye(e) => s.push_str(&format!("eye v{} gain {:.2}\n", e.variant, e.gain)),
                Payload::Memory(m) => s.push_str(&format!("mem v{} tau {:.2}\n", m.variant, m.tau)),
                Payload::Metabolism(m) => s.push_str(&format!("metab cost {:.3}\n", m.cost_per_activity)),
                _ => s.push_str(&format!("{:?}\n", g.payload)),
            }
        }
        s
    }
    /// Diff — что изменилось между геномами (для эволюции)
    pub fn diff(&self, other:&Self) -> Vec<String> {
        let mut d=Vec::new();
        for k in [GeneKind::Coat, GeneKind::Eye, GeneKind::Memory, GeneKind::Metabolism] {
            let a=self.get(k); let b=other.get(k);
            if a!=b { d.push(format!("{}: {:?} → {:?}", k.label(), a.map(|g| &g.payload), b.map(|g| &g.payload))); }
        }
        d
    }
    /// JSON для Unity-редактора (человекочитаемо, не для сейва)
    pub fn to_json(&self) -> String {
        let mut s = String::from("{\"genes\":[");
        for (i,g) in self.genes.iter().enumerate() {
            if i>0 { s.push(','); }
            s.push_str(&format!("{{\"kind\":\"{}\",\"variant\":{},\"express_at\":{}}}", g.kind.label(), g.variant, g.express_at));
        }
        s.push_str("]}");
        s
    }

    // Совместимость со старой ДНК (сжатие)
    pub fn to_dna_bytes(&self) -> Vec<u8> { self.to_bytes() }
    pub fn from_dna_bytes(b: &[u8]) -> Result<Self, String> { Self::from_bytes(b) }
    pub fn from_creature_genome(_: &crate::creature::genome::CreatureGenome) -> Self { Self::wildtype() }
    pub fn to_creature_genome(&self, _age:u64) -> crate::creature::genome::CreatureGenome { crate::creature::genome::CreatureGenome::default() }
    pub fn sync_legacy(&mut self, _: &crate::creature::genome::CreatureGenome) {}
    pub fn jitter_coat(&mut self, seed:u64) {
        let mut rng = XorShift::new(seed ^ 0x9E37_79B9_7F4A_7C15);
        if let Some(g)=self.get_mut(GeneKind::Coat) {
            if let Payload::Coat(c)=g.payload { g.payload=Payload::Coat(c.mutate(&mut rng)); }
        }
    }
    pub fn set_coat(&mut self, c:Coat) {
        if let Some(g)=self.get_mut(GeneKind::Coat){ g.payload=Payload::Coat(c); } else { self.genes.push(Gene::new(GeneKind::Coat, Payload::Coat(c))); }
    }
    pub fn coat(&self)->Option<Coat>{ self.get(GeneKind::Coat).and_then(|g| if let Payload::Coat(c)=&g.payload{Some(*c)}else{None}) }
    pub fn eye(&self)->Option<Eye>{ self.get(GeneKind::Eye).and_then(|g| if let Payload::Eye(e)=&g.payload{Some(*e)}else{None}) }
    pub fn memory(&self)->Option<Memory>{ self.get(GeneKind::Memory).and_then(|g| if let Payload::Memory(m)=&g.payload{Some(*m)}else{None}) }
    pub fn raw_data(&self)->impl Iterator<Item=&[u8]> { self.genes.iter().filter_map(|g| if let Payload::Raw(b)=&g.payload{Some(b.as_slice())}else{None}) }
    pub fn from_bytes_legacy(b:&[u8])->Result<Self,String>{ Self::from_bytes(b) }
}

pub type CoatGene = Coat;
pub type EyeGene = Eye;
pub type MemoryGene = Memory;
pub type DnaError = String;

pub struct GenomeBuilder { genes: Vec<Gene> }
impl GenomeBuilder {
    pub fn new()->Self{ Self{genes: Genome::wildtype().genes} }
    pub fn coat(mut self, c:Coat)->Self{
        if let Some(g)=self.genes.iter_mut().find(|g| g.kind==GeneKind::Coat){ g.payload=Payload::Coat(c); } else { self.genes.push(Gene::new(GeneKind::Coat, Payload::Coat(c))); } self
    }
    pub fn eye(mut self, e:Eye)->Self{
        if let Some(g)=self.genes.iter_mut().find(|g| g.kind==GeneKind::Eye){ g.payload=Payload::Eye(e); } else { self.genes.push(Gene::new(GeneKind::Eye, Payload::Eye(e))); } self
    }
    pub fn build(self)->Genome{ Genome{genes:self.genes} }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn roundtrip() {
        let g = Genome::wildtype();
        let b = g.to_bytes();
        let r = Genome::from_bytes(&b).unwrap();
        assert_eq!(g,r);
    }
    #[test] fn mutate_changes() {
        let g=Genome::wildtype();
        let mut rng=XorShift::new(1);
        let m=g.mutate(&mut rng);
        assert_ne!(g,m);
    }
    #[test] fn crossover_mixes() {
        let mut a=Genome::wildtype();
        let mut b=Genome::wildtype();
        if let Some(g)=a.get_mut(GeneKind::Coat){ g.payload=Payload::Coat(Coat::new(1.0,0.0,0.0)); }
        if let Some(g)=b.get_mut(GeneKind::Coat){ g.payload=Payload::Coat(Coat::new(0.0,1.0,0.0)); }
        let mut rng=XorShift::new(2);
        let c=a.crossover(&b, &mut rng);
        assert!(c.get(GeneKind::Coat).is_some());
    }
}
