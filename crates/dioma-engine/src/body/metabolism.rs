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
use wide::f32x8;

pub mod chem {
    pub const GLUCOSE: usize = 0; // питание
    pub const FOOD: usize = 1; // содержимое желудка (еда → глюкоза)
    pub const HUNGER: usize = 2; // драйв (синтезируется всегда, сгорает с глюкозой)
    pub const PAIN: usize = 3; // ноцицепция (удары о стены)
    pub const REWARD: usize = 4; // дофамин (событие еды)
    pub const FATIGUE: usize = 5; // утомление (от моторной активности!)
    /// Серотонин-подобный: медленный фон благополучия. Растёт от сытости,
    /// тает часами. Высокий = сытый спокойный мир; низкий = тревожность.
    pub const SEROTONIN: usize = 6;
    /// Кортизол-подобный: медленный стресс. Качели голода и боль его гонят;
    /// высокий подавляет пластичность (обучение дороже) и обостряет боль.
    pub const CORTISOL: usize = 7;
    /// Ацетилхолин-подобный: посредник внимания. Пищеварение выделяет,
    /// обучение расходует — мост «поел → мозг готов учиться».
    pub const ACETYLCHOLINE: usize = 8;
    pub const N: usize = 9;
}

/// Тайм-константы распада (доля, остающаяся за тик 50 мс).
///
/// (²: тик 50 мс) — каждый множитель возведён в квадрат: два 25-мс тика
/// теперь укладываются в один 50-мс, поэтому за секунду (20 тактов) остаётся
/// ровно та же доля, что при тике 25 мс (40 тактов): d₅₀²⁰ = (d₂₅²)²⁰ = d₂₅⁴⁰.
pub mod decay {
    use super::chem;
    /// Гормональные химии: полураспад ~ минуты — память тела о недавней жизни.
    pub const HORMONE_SLOW: f32 = 0.99900025; // 0.9995²
    pub fn table() -> Vec<f32> {
        let mut d = vec![0.998001f32; chem::N]; // 0.999²
        d[chem::PAIN] = 0.81; // 0.90²
        d[chem::REWARD] = 0.8836; // 0.94²
        d[chem::HUNGER] = 0.990025; // 0.995²
        d[chem::FATIGUE] = 0.990025; // 0.995²
        d[chem::SEROTONIN] = HORMONE_SLOW;
        d[chem::CORTISOL] = HORMONE_SLOW;
        d[chem::ACETYLCHOLINE] = 0.996004; // 0.998²
        d
    }
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
    /// Глюкоза на единицу дофамина при фиксации веса (метаболический бюджет
    /// обучения): пластичность не бесплатна — интеллект конкурирует с телом.
    pub cost_per_dopamine: f32,
    /// Восстановление повреждения за тик при сытости (глюкоза > 150):
    /// голодание калечит, изобилие лечит — градиент отбора вместо
    /// храповика «пир/вымирание».
    pub heal_rate: f32,
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
            // Пищеварение: 1 FOOD → 1 GLUCOSE + 0.3 ACh (внимание к миру:
            // поел — мозг получил посредник для обучения).
            // (×2: тик 50 мс) — скорость реакции за такт удвоена: за секунду
            // та же порция FOOD превращается в то же количество глюкозы.
            Reaction {
                prop_r1: 1.0,
                r1: chem::FOOD,
                prop_r2: 0.0,
                r2: usize::MAX,
                rate: 0.20,
                prop_p1: 1.0,
                p1: chem::GLUCOSE,
                prop_p2: 0.3,
                p2: chem::ACETYLCHOLINE,
            },
            // Ферментативное утоление: 1 HUNGER + 1 GLUCOSE → 1 GLUCOSE
            // (глюкоза — фермент: нужна, но conserve'd).
            // (×2: тик 50 мс)
            Reaction {
                prop_r1: 1.0,
                r1: chem::HUNGER,
                prop_r2: 1.0,
                r2: chem::GLUCOSE,
                rate: 0.020,
                prop_p1: 1.0,
                p1: chem::GLUCOSE,
                prop_p2: 0.0,
                p2: usize::MAX,
            },
            // Синтез серотонина из сытости: 1 GLUCOSE → 1 SEROTONIN.
            // Медленный каскад: сытость сегодня = спокойствие на часы.
            // rate мал намеренно: глюкоза не должна выгорать от одной сытости
            // (этой реакцией полураспад глюкозы — десятки секунд, не тики).
            // (×2: тик 50 мс)
            Reaction {
                prop_r1: 1.0,
                r1: chem::GLUCOSE,
                prop_r2: 0.0,
                r2: usize::MAX,
                rate: 0.0008,
                prop_p1: 1.0,
                p1: chem::SEROTONIN,
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
            // Паническая гиперактивность при почти смертельном голоде:
            // крайний голод добавляет фиксированный моторный тон (мечется,
            // а не глушится). Повреждение глушится отдельно — GAIN_GLOBAL в шаге 5.
            Receptor {
                chem: chem::HUNGER,
                threshold: 240.0,
                gain: 1.5,
                reduce: false,
                digital: true,
                locus: locus::MOTOR_TONIC,
            },
            // Серотонин успокаивает: снижает тревожность поворота.
            Receptor {
                chem: chem::SEROTONIN,
                threshold: 10.0,
                gain: 0.03,
                reduce: true,
                digital: false,
                locus: locus::TURN_GAIN,
            },
            // Кортизол обостряет обоняние (стрессовая бдительность),
            // но глушит моторный тон (замороженность). Качели голода
            // гоняют кортизол — темперамент существа меняется минутами.
            Receptor {
                chem: chem::CORTISOL,
                threshold: 20.0,
                gain: 1.5,
                reduce: false,
                digital: false,
                locus: locus::SENSOR_GAIN,
            },
            Receptor {
                chem: chem::CORTISOL,
                threshold: 60.0,
                gain: 0.5,
                reduce: true,
                digital: false,
                locus: locus::MOTOR_TONIC,
            },
        ];
        let emitters = vec![
            // Событие еды → дофамин. Цифровой всплеск: НЕ ×2 — это импульс
            // события, а не скорость; распад (0.8836/тик) уже пересчитан под
            // тик 50 мс, и за секунду импульс гаснет так же, как при 25 мс.
            Emitter {
                locus: locus::EVENT_ATE,
                threshold: 0.5,
                gain: 60.0,
                remove: true,
                digital: true,
                chem: chem::REWARD,
            },
            // Удар о стену → боль. Тоже событийный импульс — НЕ ×2 (см. выше).
            Emitter {
                locus: locus::EVENT_WALL,
                threshold: 0.5,
                gain: 40.0,
                remove: true,
                digital: true,
                chem: chem::PAIN,
            },
            // Моторная активность → утомление (мозг→химия→тело, замкнуто).
            // Порог под реальный динамик: моторы дают ~5–20 спайков за тик,
            // а не десятки LIF-единиц. Раньше порог 25 делал утомление
            // мёртвым механизмом (fatigue всегда 0).
            // Гейн НЕ ×2: RATE_MOTOR (спайки моторов за тик) сам удваивается
            // вместе с BRAIN_STEPS_PER_TICK (100 подшагов вместо 50) — за
            // секунду утомление копится с той же скоростью, что при 25 мс.
            Emitter {
                locus: locus::RATE_MOTOR,
                threshold: 6.0,
                gain: 0.20,
                remove: false,
                digital: false,
                chem: chem::FATIGUE,
            },
            // Дефицит глюкозы → голод (драйв из локуса, как у Creatures).
            // (×2: тик 50 мс) — непрерывная продукция гормона: за такт вдвое
            // больше, за секунду столько же.
            Emitter {
                locus: locus::GLUCOSE_DEFICIT,
                threshold: 15.0,
                gain: 0.012,
                remove: false,
                digital: false,
                chem: chem::HUNGER,
            },
            // Дефицит глюкозы → стресс-гормон (медленный): хронический голод
            // копит кортизол, и существо становится пугливым на минуты.
            // (×2: тик 50 мс)
            Emitter {
                locus: locus::GLUCOSE_DEFICIT,
                threshold: 40.0,
                gain: 0.008,
                remove: false,
                digital: false,
                chem: chem::CORTISOL,
            },
            // Боль тоже вливает стресс (острый эпизод поверх хроники).
            // Событийный импульс — НЕ ×2 (см. REWARD выше).
            Emitter {
                locus: locus::EVENT_WALL,
                threshold: 0.5,
                gain: 8.0,
                remove: true,
                digital: true,
                chem: chem::CORTISOL,
            },
        ];
        let mut conc = vec![0.0f32; chem::N];
        conc[chem::GLUCOSE] = 128.0;
        conc[chem::SEROTONIN] = 30.0;
        let decay = decay::table();
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
                // Цена за единицу активности НЕ ×2: активность (спайки моторов
                // за тик) сама удваивается от 100 подшагов мозга вместо 50 —
                // расход глюкозы за секунду остаётся тем же, что при 25 мс
                // (0.0015 × 2×R × 20 = 0.0015 × R × 40).
                cost_per_activity: 0.0015,
                // (×2: тик 50 мс) — урон за тик при пустой глюкозе удвоен
                // (не связан с активностью мозга).
                starve_damage: 0.004,
                damage: 0.0,
                // Цена за событие дофамина — НЕ ×2 (не скорость).
                cost_per_dopamine: 0.4,
                // (×2: тик 50 мс)
                heal_rate: 0.0001,
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
    /// ВАЖНО: реакции внутри тика последовательны — каждая видит концентрации
    /// после предыдущей; итог зависит от порядка в `reactions`, а не симметричен.
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

        // 2. Распады (полураспады) с SIMD оптимизацией.
        // Обрабатываем первые 8 элементов через f32x8, последний скалярно.
        debug_assert_eq!(self.conc.len(), chem::N);
        debug_assert_eq!(self.decay.len(), chem::N);
        
        if self.conc.len() >= 8 {
            // SIMD путь для первых 8 элементов
            let conc_simd = f32x8::new([
                self.conc[0], self.conc[1], self.conc[2], self.conc[3],
                self.conc[4], self.conc[5], self.conc[6], self.conc[7],
            ]);
            let decay_simd = f32x8::new([
                self.decay[0], self.decay[1], self.decay[2], self.decay[3],
                self.decay[4], self.decay[5], self.decay[6], self.decay[7],
            ]);
            let result = (conc_simd * decay_simd).max(f32x8::splat(0.0)).min(f32x8::splat(255.0));
            let arr = result.to_array();
            self.conc[0..8].copy_from_slice(&arr);
            
            // Скалярная обработка оставшихся элементов
            for i in 8..self.conc.len() {
                self.conc[i] = (self.conc[i] * self.decay[i]).clamp(0.0, 255.0);
            }
        } else {
            // Fallback для маленьких массивов
            for (c, d) in self.conc.iter_mut().zip(&self.decay) {
                *c = (*c * d).clamp(0.0, 255.0);
            }
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
        // Входные локусы (события и драйвы: EVENT_*, RATE_MOTOR,
        // GLUCOSE_DEFICIT) пишутся ТЕЛОМ до update() — сброс дефолтов их
        // ЗАТИРАЕТ, и урон/мощность в шаге 5 видят нули (существа не
        // умирали и не голодали). Сохраняем их до пересчёта рецепторов.
        let drive_loci = [
            (locus::EVENT_ATE, self.loci[locus::EVENT_ATE]),
            (locus::EVENT_WALL, self.loci[locus::EVENT_WALL]),
            (locus::RATE_MOTOR, self.loci[locus::RATE_MOTOR]),
            (locus::GLUCOSE_DEFICIT, self.loci[locus::GLUCOSE_DEFICIT]),
        ];
        self.loci = locus::defaults();
        for &(l, v) in &drive_loci {
            self.loci[l] = v;
        }
        // Batch обновление рецепторов: группируем эффекты по локусам
        let mut locus_effects: [f32; locus::N] = [0.0; locus::N];
        
        for r in &self.receptors {
            let signal = (self.conc[r.chem] - r.threshold).max(0.0);
            let effect = if r.digital {
                if signal > 0.0 { r.gain } else { 0.0 }
            } else {
                r.gain * signal / 255.0
            };
            locus_effects[r.locus] += if r.reduce { -effect } else { effect };
        }
        
        // Применяем накопленные эффекты одним проходом
        for (locus_idx, &effect) in locus_effects.iter().enumerate() {
            if effect != 0.0 {
                self.loci[locus_idx] += effect;
            }
        }

        // 5. Орган «мозг»: активность стоит глюкозы; голод калечит,
        // изобилие лечит (градиент отбора вместо храповика).
        let activity = self.loci[locus::RATE_MOTOR];
        let cost = self.organ.cost_per_activity * activity;
        if self.conc[chem::GLUCOSE] >= cost {
            self.conc[chem::GLUCOSE] -= cost;
        } else {
            self.conc[chem::GLUCOSE] = 0.0;
            self.organ.damage = (self.organ.damage + self.organ.starve_damage).min(1.0);
        }
        if self.conc[chem::GLUCOSE] > 150.0 && self.organ.damage > 0.0 {
            self.organ.damage = (self.organ.damage - self.organ.heal_rate).max(0.0);
            // Заживление тоже ест глюкозу — восстановление не бесплатно.
            // (×2: тик 50 мс)
            self.conc[chem::GLUCOSE] -= 0.02;
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

    /// Сон: тело восстанавливается — усталость и кортизол падают (стресс
    /// сходит), глюкоза чуть пополняется (покой), повреждения заживают.
    /// Зовёт хост в ночной фазе, вместе с консолидацией мозга.
    pub fn sleep(&mut self) {
        self.conc[chem::FATIGUE] = (self.conc[chem::FATIGUE] * 0.3).min(255.0);
        self.conc[chem::CORTISOL] = (self.conc[chem::CORTISOL] * 0.5).min(255.0);
        self.conc[chem::GLUCOSE] = (self.conc[chem::GLUCOSE] + 15.0).min(255.0);
        self.organ.damage = (self.organ.damage - 0.02).max(0.0);
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

    /// Метаболический множитель пластичности [0..1]: кортизол дорогой
    /// (стресс → обучение подавлено), ацетилхолин дешёвый (внимание после
    /// еды → окно обучаемости). Мозг читает перед reward_now.
    pub fn plasticity_budget(&self) -> f32 {
        let stress = (self.conc[chem::CORTISOL] / 120.0).min(1.0);
        let attention = (self.conc[chem::ACETYLCHOLINE] / 60.0).min(1.0);
        (1.0 - 0.6 * stress) * (0.7 + 0.3 * attention)
    }

    /// Списать метаболическую цену фиксации веса: сколько глюкозы стоит
    /// этот дофамин. false — мозг обанкротился, награда не прошла.
    pub fn pay_for_plasticity(&mut self, dopamine: f32) -> bool {
        let cost = self.organ.cost_per_dopamine * dopamine;
        if self.conc[chem::GLUCOSE] >= cost {
            self.conc[chem::GLUCOSE] -= cost;
            true
        } else {
            // Частичная выплата: слабый сигнал обучения при истощении.
            let paid = self.conc[chem::GLUCOSE];
            self.conc[chem::GLUCOSE] = 0.0;
            paid > 0.0
        }
    }
}

#[cfg(test)]
mod hormone_tests {
    use super::*;

    /// Гормоны — память тела: после серии ударов кортизол держится минуты,
    /// а не тики (в отличие от боли с decay 0.90).
    #[test]
    fn cortisol_outlives_pain_by_orders_of_magnitude() {
        let mut b = Blood::creature(false);
        b.conc[chem::PAIN] = 100.0;
        b.conc[chem::CORTISOL] = 100.0;
        for _ in 0..200 {
            b.update();
        }
        assert!(
            b.conc[chem::PAIN] < 1.0,
            "боль должна угаснуть: {}",
            b.conc[chem::PAIN]
        );
        assert!(
            b.conc[chem::CORTISOL] > 50.0,
            "кортизол — медленная память: {}",
            b.conc[chem::CORTISOL]
        );
    }

    /// Каскад: пищеварение выделяет ацетилхолин (после еды окно внимания).
    #[test]
    fn digestion_produces_acetylcholine() {
        let mut b = Blood::creature(false);
        let ach0 = b.conc[chem::ACETYLCHOLINE];
        b.ingest_food(90.0);
        for _ in 0..50 {
            b.update();
        }
        assert!(
            b.conc[chem::ACETYLCHOLINE] > ach0 + 1.0,
            "еда должна дать посредник внимания: {}",
            b.conc[chem::ACETYLCHOLINE]
        );
    }

    /// Метаболический бюджет: стресс подавляет пластичность, внимание помогает.
    #[test]
    fn plasticity_budget_respects_hormones() {
        let mut calm = Blood::creature(false);
        calm.conc[chem::SEROTONIN] = 80.0;
        let mut stressed = Blood::creature(false);
        stressed.conc[chem::CORTISOL] = 150.0;
        let mut attentive = Blood::creature(false);
        attentive.conc[chem::ACETYLCHOLINE] = 120.0;

        let base = calm.plasticity_budget();
        assert!(
            stressed.plasticity_budget() < base * 0.6,
            "стресс должен глушить обучение"
        );
        assert!(
            attentive.plasticity_budget() > base,
            "внимание должно помогать обучению"
        );
    }

    /// Обучение стоит глюкозы; при банкротстве платится остаток.
    #[test]
    fn plasticity_costs_glucose_and_can_bankrupt() {
        let mut b = Blood::creature(false);
        b.conc[chem::GLUCOSE] = 10.0;
        assert!(b.pay_for_plasticity(5.0), "достаточно денег — платим");
        assert!((b.conc[chem::GLUCOSE] - (10.0 - 0.4 * 5.0)).abs() < 1e-4);

        b.conc[chem::GLUCOSE] = 0.5;
        assert!(
            b.pay_for_plasticity(5.0),
            "частичная выплата — глюкоза была"
        );
        assert_eq!(b.conc[chem::GLUCOSE], 0.0, "банкрот: всё списано");
    }

    /// Входные локусы (события/драйвы) не должны затираться рецепторами:
    /// тело пишет их ДО update(), урон и драйв обязаны дойти до органа.
    #[test]
    fn drive_loci_survive_receptor_reset() {
        let mut b = Blood::creature(false);
        b.loci[locus::RATE_MOTOR] = 100.0;
        b.loci[locus::GLUCOSE_DEFICIT] = 50.0;
        b.loci[locus::EVENT_WALL] = 1.0;
        b.update();
        // Урон: моторная активность стоит глюкозу; дефицит → голод.
        assert!(
            b.loci[locus::RATE_MOTOR] > 0.0,
            "RATE_MOTOR затёрт рецепторами: {}",
            b.loci[locus::RATE_MOTOR]
        );
        assert!(
            b.loci[locus::GLUCOSE_DEFICIT] > 0.0,
            "GLUCOSE_DEFICIT затёрт: {}",
            b.loci[locus::GLUCOSE_DEFICIT]
        );
        assert!(
            b.loci[locus::EVENT_WALL] == 0.0,
            "EVENT_WALL должен гаснуть после эмиссии"
        );
    }

    /// Заживление: изобилие лечит повреждение, голодание калечит.
    #[test]
    fn abundance_heals_starvation_damage() {
        let mut b = Blood::creature(false);
        b.organ.damage = 0.2;
        // Держим глюкозу в зоне изобилия: еда поступает быстрее расхода.
        for _ in 0..2000 {
            b.conc[chem::GLUCOSE] = 200.0; // искусственная капельница
            b.update();
        }
        assert!(
            b.organ.damage < 0.15,
            "изобилие должно лечить: {}",
            b.organ.damage
        );
    }

    /// Регресс: синтез серотонина не должен выжигать глюкозу. Раньше
    /// rate=0.02 сжигал ~2% запаса за тик (полураспад ~33 тика ≈ 0.8 с)
    /// и существо голодало за доли секунды просто от сытости. Теперь
    /// глюкоза тратится медленно, а серотонин всё ещё копится.
    /// (÷2: тик 50 мс — 250 тиков = ~12.5 с, как 500 тиков при 25 мс.)
    #[test]
    fn serotonin_synthesis_does_not_drain_glucose() {
        let mut b = Blood::creature(false);
        b.loci[locus::RATE_MOTOR] = 0.0; // без моторной активности
        let g0 = b.conc[chem::GLUCOSE];
        let s0 = b.conc[chem::SEROTONIN];

        for _ in 0..250 {
            b.update();
        }

        assert!(
            b.conc[chem::GLUCOSE] > g0 * 0.4,
            "сытость не должна выжигать глюкозу: {} → {}",
            g0,
            b.conc[chem::GLUCOSE]
        );
        // Серотонин — медленный фон: за 250 тиков (~12.5 с) заметно растёт,
        // но не взлетает (равновесие ~0.8×глюкоза, тает с HORMONE_SLOW).
        assert!(
            b.conc[chem::SEROTONIN] > s0 + 5.0,
            "сытость должна копить серотонин: {} → {}",
            s0,
            b.conc[chem::SEROTONIN]
        );
    }
}
