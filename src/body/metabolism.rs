//! -------------------------------------------------------------------------
//! Filename:    src/metabolism.rs
//! Class:       bioEngine
//! Purpose:
//! Description: stoichiometric reactions (limiting reagent), named loci,
//!
//! receptors/emitters (RE_REDUCE/RE_DIGITAL, EM_REMOVE analogs),
//! brain organ: spikes cost glucose, starvation damages and slows.
//! author: Atlasov anton
//! -------------------------------------------------------------------------

/// Химические вещества.
use std::collections::VecDeque;
pub mod chem {
    pub const GLUCOSE: usize = 0; // питание
    pub const FOOD: usize = 1; // содержимое желудка (еда → глюкоза)
    pub const HUNGER: usize = 2; // драйв (синтезируется всегда, сгорает с глюкозой)
    pub const PAIN: usize = 3; // ноцицепция (удары о стены)
    pub const REWARD: usize = 4; // дофамин (событие еды)
    pub const FATIGUE: usize = 5; // утомление (от моторной активности!)
    pub const N: usize = 6;
}

/// Локусы — именованные точки «чего угодно» существа.
pub mod locus {
    pub const SENSOR_GAIN: usize = 0; // рецепторы пишут, тело читает
    pub const MOTOR_TONIC: usize = 1;
    pub const SPEED: usize = 2;
    pub const TURN_GAIN: usize = 3;
    pub const GAIN_GLOBAL: usize = 4; // множитель от повреждения органов
    pub const RATE_MOTOR: usize = 5; // спайки моторных пулов за тик (тело пишет)
    pub const EVENT_ATE: usize = 6; // событие еды (тело пишет, эмиттер гасит)
    pub const EVENT_WALL: usize = 7; // удар о стену
    pub const GLUCOSE_DEFICIT: usize = 8; // тело пишет: max(0, 150 − глюкоза)
    pub const N: usize = 9;

    /// Значения по умолчанию (рецепторы складываются поверх).
    pub fn defaults() -> Vec<f32> {
        let mut v = vec![0.0f32; N];
        v[SENSOR_GAIN] = 3.5;
        v[MOTOR_TONIC] = 0.12;
        v[SPEED] = 2.0;
        v[TURN_GAIN] = 0.06;
        v[GAIN_GLOBAL] = 1.0;
        v
    }
}

/// Реакция: `propR1·R1 + propR2·R2 --rate--> propP1·P1 + propP2·P2`.
/// Реагент 0 = «нет» (синтез из ничего: rate = моль/тик).
#[derive(Clone, Debug)]
pub struct Reaction {
    pub prop_r1: f32,
    pub r1: usize,
    pub prop_r2: f32,
    pub r2: usize,
    pub rate: f32,
    pub prop_p1: f32,
    pub p1: usize,
    pub prop_p2: f32,
    pub p2: usize,
}

/// Рецептор: химия → локус (аддитивно поверх дефолта).
#[derive(Clone, Debug)]
pub struct Receptor {
    pub chem: usize,
    pub threshold: f32,
    pub gain: f32,
    /// true — вычитать (RE_REDUCE), false — прибавлять.
    pub reduce: bool,
    /// true — цифровой отклик (фикс. гейн при любом сигнале > порога).
    pub digital: bool,
    pub locus: usize,
}

/// Эмиттер: локус → химия.
#[derive(Clone, Debug)]
pub struct Emitter {
    pub locus: usize,
    pub threshold: f32,
    pub gain: f32,
    /// true — гасить локус после отклика (событийные всплески).
    pub remove: bool,
    /// true — плоский гейн при любом сигнале > порога (их EM_DIGITAL);
    /// false — аналоговый: гейн × сигнал.
    pub digital: bool,
    pub chem: usize,
}

/// Орган «мозг»: ест глюкозу за активность, голодает — повреждается.
#[derive(Clone, Debug)]
pub struct BrainOrgan {
    /// Глюкозы на единицу активности (спайк моторов за тик).
    pub cost_per_activity: f32,
    /// Урон за тик при пустой глюкозе.
    pub starve_damage: f32,
    /// 0..1 — накопленное повреждение (глушит GAIN_GLOBAL).
    pub damage: f32,
}

/// Один тик крови для отладки: события, химии, локусы, урон органа.
#[derive(Clone, Debug)]
pub struct TraceTick {
    pub tick: usize,
    pub ate: bool,
    pub wall: bool,
    pub chemicals: Vec<f32>,
    pub loci: Vec<f32>,
    pub damage: f32,
}

pub struct Blood {
    /// Счётчик тиков крови (для trace и окон).
    pub tick: usize,
    /// Trace-режим: кольцевой буфер последних тиков (None = выключен).
    trace: Option<(VecDeque<TraceTick>, usize)>,
    pub conc: Vec<f32>,
    decay: Vec<f32>,
    pub reactions: Vec<Reaction>,
    pub receptors: Vec<Receptor>,
    pub emitters: Vec<Emitter>,
    /// Локусы: события/активность пишутся телом до update(),
    /// параметры читаются после.
    pub loci: Vec<f32>,
    pub organ: BrainOrgan,
    /// Принудительные концентрации (контроль «робота»).
    pub pins: Vec<(usize, f32)>,
}

impl Blood {
    /// Биохимия существа : гомеостаз голода через
    /// стехиометрию, утомление от моторики, дофамин от еды, боль от стен.
    pub fn creature(robot: bool) -> Self {
        let reactions = vec![
            // Пищеварение: 1 FOOD → 1 GLUCOSE.
            Reaction {
                prop_r1: 1.0,
                r1: chem::FOOD,
                prop_r2: 0.0,
                r2: usize::MAX,
                rate: 0.10,
                prop_p1: 1.0,
                p1: chem::GLUCOSE,
                prop_p2: 0.0,
                p2: usize::MAX,
            },
            // Ферментативное утоление: 1 HUNGER + 1 GLUCOSE → 1 GLUCOSE
            // (глюкоза — фермент: нужна, но conserve'd).
            Reaction {
                prop_r1: 1.0,
                r1: chem::HUNGER,
                prop_r2: 1.0,
                r2: chem::GLUCOSE,
                rate: 0.010,
                prop_p1: 1.0,
                p1: chem::GLUCOSE,
                prop_p2: 0.0,
                p2: usize::MAX,
            },
        ];
        let receptors = vec![
            // Голод обостряет обоняние и моторный тон.
            Receptor {
                chem: chem::HUNGER,
                threshold: 10.0,
                gain: 4.0,
                reduce: false,
                digital: false,
                locus: locus::SENSOR_GAIN,
            },
            Receptor {
                chem: chem::HUNGER,
                threshold: 10.0,
                gain: 0.10,
                reduce: false,
                digital: false,
                locus: locus::MOTOR_TONIC,
            },
            // Боль — тревожность поворота.
            Receptor {
                chem: chem::PAIN,
                threshold: 5.0,
                gain: 0.04,
                reduce: false,
                digital: true,
                locus: locus::TURN_GAIN,
            },
            // Утомление и сытость замедляют тело.
            Receptor {
                chem: chem::FATIGUE,
                threshold: 20.0,
                gain: 1.2,
                reduce: true,
                digital: false,
                locus: locus::SPEED,
            },
            Receptor {
                chem: chem::GLUCOSE,
                threshold: 150.0,
                gain: 0.4,
                reduce: true,
                digital: false,
                locus: locus::SPEED,
            },
            // Повреждение мозга глушит всё.
            Receptor {
                chem: chem::HUNGER,
                threshold: 240.0,
                gain: 1.5,
                reduce: false,
                digital: true,
                locus: locus::MOTOR_TONIC,
            },
        ];
        let emitters = vec![
            // Событие еды → дофамин.
            Emitter {
                locus: locus::EVENT_ATE,
                threshold: 0.5,
                gain: 60.0,
                remove: true,
                digital: true,
                chem: chem::REWARD,
            },
            // Удар о стену → боль.
            Emitter {
                locus: locus::EVENT_WALL,
                threshold: 0.5,
                gain: 40.0,
                remove: true,
                digital: true,
                chem: chem::PAIN,
            },
            // Моторная активность → утомление (мозг→химия→тело, замкнуто).
            Emitter {
                locus: locus::RATE_MOTOR,
                threshold: 25.0,
                gain: 0.05,
                remove: false,
                digital: false,
                chem: chem::FATIGUE,
            },
            // Дефицит глюкозы → голод (драйв из локуса, как у Creatures).
            Emitter {
                locus: locus::GLUCOSE_DEFICIT,
                threshold: 15.0,
                gain: 0.006,
                remove: false,
                digital: false,
                chem: chem::HUNGER,
            },
        ];
        let mut conc = vec![0.0f32; chem::N];
        conc[chem::GLUCOSE] = 128.0;
        let decay = {
            let mut d = vec![0.999f32; chem::N];
            d[chem::FOOD] = 0.999;
            d[chem::PAIN] = 0.90;
            d[chem::REWARD] = 0.94;
            d[chem::HUNGER] = 0.995;
            d[chem::FATIGUE] = 0.995;
            d
        };
        let mut pins = Vec::new();
        if robot {
            pins.push((chem::HUNGER, 255.0));
        }
        Self {
            tick: 0,
            trace: None,
            conc,
            decay,
            reactions,
            receptors,
            emitters,
            loci: locus::defaults(),
            organ: BrainOrgan {
                cost_per_activity: 0.0015,
                starve_damage: 0.002,
                damage: 0.0,
            },
            pins,
        }
    }

    /// Включить trace-режим (кольцевой буфер на `cap` тиков).
    pub fn enable_trace(&mut self, cap: usize) {
        self.trace = Some((VecDeque::new(), cap));
    }

    /// Последние тики трейса (если режим включен).
    pub fn trace(&self) -> Option<&VecDeque<TraceTick>> {
        self.trace.as_ref().map(|(q, _)| q)
    }

    /// Один тик крови. Порядок: реакции → распады → эмиттеры → рецепторы →
    /// орган (цена активности, урон голода) → пины → запись трейса.
    pub fn update(&mut self) {
        let (ev_ate, ev_wall) = (
            self.loci[locus::EVENT_ATE] > 0.5,
            self.loci[locus::EVENT_WALL] > 0.5,
        );
        // 1. Реакции (стехиометрия с лимитирующим реагентом).
        for r in &self.reactions {
            let avail1 = if r.r1 == usize::MAX {
                f32::INFINITY
            } else {
                self.conc[r.r1] / r.prop_r1.max(1e-9)
            };
            let avail2 = if r.r2 == usize::MAX {
                f32::INFINITY
            } else {
                self.conc[r.r2] / r.prop_r2.max(1e-9)
            };
            let moles = if r.r1 == usize::MAX && r.r2 == usize::MAX {
                r.rate // синтез из ничего: константа
            } else {
                r.rate * avail1.min(avail2)
            };
            if moles <= 0.0 {
                continue;
            }
            if r.r1 != usize::MAX {
                self.conc[r.r1] -= moles * r.prop_r1;
            }
            if r.r2 != usize::MAX {
                self.conc[r.r2] -= moles * r.prop_r2;
            }
            if r.p1 != usize::MAX {
                self.conc[r.p1] += moles * r.prop_p1;
            }
            if r.p2 != usize::MAX {
                self.conc[r.p2] += moles * r.prop_p2;
            }
        }

        // 2. Распады (полураспады).
        for (c, d) in self.conc.iter_mut().zip(&self.decay) {
            *c = (*c * d).clamp(0.0, 255.0);
        }

        // 3. Эмиттеры: локус → химия.
        for e in &self.emitters {
            let v = self.loci[e.locus];
            if v > e.threshold {
                let amount = if e.digital { e.gain } else { e.gain * v };
                self.conc[e.chem] = (self.conc[e.chem] + amount).min(255.0);
            }
        }
        // Погасить событийные локусы (EM_REMOVE).
        for e in &self.emitters {
            if e.remove && self.loci[e.locus] > e.threshold {
                self.loci[e.locus] = 0.0;
            }
        }

        // 4. Рецепторы: химия → локус (аддитивно поверх дефолтов).
        self.loci = locus::defaults();
        for r in &self.receptors {
            let signal = (self.conc[r.chem] - r.threshold).max(0.0);
            let effect = if r.digital {
                if signal > 0.0 {
                    r.gain
                } else {
                    0.0
                }
            } else {
                r.gain * signal / 255.0
            };
            self.loci[r.locus] += if r.reduce { -effect } else { effect };
        }

        // 5. Орган «мозг»: активность стоит глюкозы; голод калечит.
        let activity = self.loci[locus::RATE_MOTOR];
        let cost = self.organ.cost_per_activity * activity;
        if self.conc[chem::GLUCOSE] >= cost {
            self.conc[chem::GLUCOSE] -= cost;
        } else {
            self.conc[chem::GLUCOSE] = 0.0;
            self.organ.damage = (self.organ.damage + self.organ.starve_damage).min(1.0);
        }
        // Повреждение глушит глобальный гейн производительности.
        self.loci[locus::GAIN_GLOBAL] *= 1.0 - self.organ.damage;

        // 6. Пины (контроль «робота»).
        for &(c, v) in &self.pins {
            self.conc[c] = v;
        }

        // 7. Trace-режим.
        self.tick += 1;
        if let Some((q, cap)) = &mut self.trace {
            q.push_back(TraceTick {
                tick: self.tick,
                ate: ev_ate,
                wall: ev_wall,
                chemicals: self.conc.clone(),
                loci: self.loci.clone(),
                damage: self.organ.damage,
            });
            while q.len() > *cap {
                q.pop_front();
            }
        }
    }

    // ─── События (тело пишет до update) ─────────────────────────────────────
    pub fn ingest_food(&mut self, amount: f32) {
        self.conc[chem::FOOD] = (self.conc[chem::FOOD] + amount).min(255.0);
        self.loci[locus::EVENT_ATE] = 1.0;
    }
    pub fn bump_wall(&mut self) {
        self.loci[locus::EVENT_WALL] = 1.0;
    }

    // ─── Параметры (тело читает после update) ───────────────────────────────
    pub fn sensor_gain(&self) -> f32 {
        self.loci[locus::SENSOR_GAIN].max(0.1) * self.loci[locus::GAIN_GLOBAL]
    }
    pub fn motor_tonic(&self) -> f32 {
        (self.loci[locus::MOTOR_TONIC] + 0.15 * self.conc[chem::PAIN] / 255.0)
            * self.loci[locus::GAIN_GLOBAL]
    }
    pub fn speed(&self) -> f64 {
        self.loci[locus::SPEED].max(0.3) as f64 * self.loci[locus::GAIN_GLOBAL] as f64
    }
    pub fn turn_gain(&self) -> f64 {
        self.loci[locus::TURN_GAIN].max(0.01) as f64 * self.loci[locus::GAIN_GLOBAL] as f64
    }

    /// Еда интересна только голодному (рецептор «рта»).
    pub fn wants_food(&self) -> bool {
        self.conc[chem::HUNGER] > 40.0
    }
}
