//! -------------------------------------------------------------------------
//! Filename:    src/rng.rs
//! Class:       randomizer
//! Purpose:     deterministic XorShift RNG
//! Description: unit() + gauss(); shared by all modules, seeded per protocol.
//!
//! author: Atlasov anton
//! -------------------------------------------------------------------------

/// Детерминированный хэш индекса в [0, 1) — разброс драйва/ролей по нейронам
/// без состояния (пейсмекерная разреженность, сенсорный шум).
pub fn hash01(i: usize) -> f64 {
    let x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    ((x >> 33) as f64) / (u32::MAX as f64)
}

pub struct XorShift(pub u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Сумма трёх равномерных ≈ гауссиана (для мутаций достаточно).
    pub fn gauss(&mut self) -> f64 {
        self.unit() + self.unit() + self.unit() - 1.5
    }
}
