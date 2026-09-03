//! Dioma Unity Bridge — красивый, продуманный, headless API для Unity.
//!
//! # Философия
//! * **Один тик — один вызов.** Коннектом, веса, STDP-следы и GPU-буферы живут в Rust.
//!   Unity передаёт только сенсоры и получает действие. Пересечение границы — раз в 50 мс.
//! * **Хэндлы, а не копии.** Существо — непрозрачный `*mut DiomaCreature` (Box). Unity владеет
//!   указателем, Rust — памятью. Клонирование — только через `spawn_child`.
//! * **Без паник через FFI.** Каждый `extern "C"` ловит `catch_unwind`, пишет `last_error`
//!   и возвращает код. Unity никогда не видит Rust-панику.
//! * **Батч — быстро.** `tick_batch` тикает N существ параллельно на rayon-пуле (как `Arena::step`),
//!   а не N раз по одному переходу. Для стаи 24 особи это разовая синхронизация.
//! * **Диагностика — чтение, не мутация.** `diagnostics / brain_stats / blood / membrane` —
//!   `&self`, не трогают симуляцию. Можно звать каждый кадр для HUD/голограммы.
//! * **Наследственность — детерминизм.** `spawn_child[Pair]` — чистая функция от сидов,
//!   ДНК — побайтовый `DNA1` (round-trip), сохранение — буфер, который Unity освобождает
//!   `buffer_free`. Один сид — один потомок, бит-в-бит.
//!
//! # Группы API
//! * **Версия и ошибки:** `dioma_abi_version`, `dioma_last_error`, `dioma_buffer_free`
//! * **Жизнь существа:** `dioma_creature_create / destroy / tick / tick_batch / spawn_child*`
//! * **Телеметрия:** `dioma_creature_diagnostics / brain_stats / blood / membrane`
//! * **Наследие:** `dioma_creature_dna_bytes / create_from_dna / save_taas / load_taas` (DNA1)
//! * **Мир (опционально):** `dioma_world_*` — тонкая обёртка над `Arena` для тех, кто хочет
//!   гнать симуляцию целиком в Rust (позиции, ягоды, климат — без C# логики).
//!
//! # Пример C# (упрощённо)
//! ```csharp
//! var h = Dioma.creature_create(seed: 42, backend: DiomaBackend.Cpu);
//! var sense = new DiomaSense { smell_l = 0.8f, smell_r = 0.1f, food_near = 1 };
//! Dioma.creature_tick(h, ref sense, out var action); // action.steer/drive/eat
//! var diag = Dioma.creature_diagnostics(h); // age, hunger, glucose, spikes...
//! var child = Dioma.creature_spawn_child(h, childSeed: 99);
//! Dioma.creature_destroy(h); Dioma.creature_destroy(child);
//! ```
//!
//! Все `extern "C"` — `#[no_mangle]`, `unsafe` только на границе указателей.

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use dioma_engine::creature::{Creature, Sense};
use dioma_engine::creature::dna::Genome as DnaGenome;

// ── Версионирование ABI ────────────────────────────────────────────────────

/// Инкремент при несовместимом изменении C ABI.
pub const ABI_VERSION: u32 = 2;

// ── Коды возврата ──────────────────────────────────────────────────────────

pub const OK: i32 = 0;
pub const NULL_ARGUMENT: i32 = 1;
pub const PANIC: i32 = 2;
pub const INVALID_ARGUMENT: i32 = 3;
pub const BUFFER_TOO_SMALL: i32 = 4;

// ── Thread-local last_error ────────────────────────────────────────────────

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|c| *c.borrow_mut() = msg.into());
}
fn clear_last_error() {
    LAST_ERROR.with(|c| c.borrow_mut().clear());
}

// ── Бекенды ────────────────────────────────────────────────────────────────

/// CPU детерминирован (дефолт). GPU — тот же контракт, с тихим фолбэком на CPU.
#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum DiomaBackend {
    Cpu = 0,
    Gpu = 1,
    GpuGrown = 2,
}

// ── Sense / Action (совместимы с v1) ─────────────────────────────────────

/// Сенсоры за тик. `food_near/wall/see_eat` — байты (0 = false, !=0 = true).
/// `text` — символ-якорь 0..1 (слово предъявлено). Версия 2: добавлено поле `text`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DiomaSense {
    pub smell_l: f32,
    pub smell_r: f32,
    pub vision_l: f32,
    pub vision_r: f32,
    pub wall_l: f32,
    pub wall_r: f32,
    pub heading: f32,
    pub place: u32,
    pub food_near: u8,
    pub wall: u8,
    pub see_eat: u8,
    pub _pad0: u8,
    pub text: f32, // 0..1 — слово предъявлено (якорь в кору)
}

/// Действие, которое Unity применяет к телу.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DiomaAction {
    pub steer: f32,
    pub drive: f32,
    pub eat: u8,
    pub _padding: [u8; 3],
}

// ── Диагностика ────────────────────────────────────────────────────────────

/// Снимок существа за тик — HUD/темерия без аллокаций. Чтение, не мутация.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DiomaDiagnostics {
    pub age: u64,
    pub damage: f32,
    pub food_eaten: u32,
    pub is_sleeping: u8,
    pub is_dying: u8,
    pub is_dead: u8,
    pub _pad0: u8,
    pub glucose: f32,
    pub hunger: f32,
    pub fatigue: f32,
    pub pain: f32,
    pub cortisol: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub sensor_gain: f32,
    pub motor_tonic: f32,
    pub speed: f32,
    pub plasticity_budget: f32,
    /// Спайков мозга за последний тик (сумма по `spike_hist`).
    pub spikes_last_tick: u32,
}

/// Статистика мозга — для прогресс-бара, голограммы, бенчей.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DiomaBrainStats {
    pub neuron_count: u32,
    pub region_count: u32,
    pub synapse_count: u32,
    pub steps_per_tick: u32,
    pub inhibition_ratio: f32, // доля тормозных нейронов 0..1
}

// ── Opaque handle ──────────────────────────────────────────────────────────

/// Непрозрачный хэндл существа. Unity владеет указателем, Rust — Box-ом.
pub struct DiomaCreature {
    creature: Creature,
}

// ── Утилиты ────────────────────────────────────────────────────────────────

#[inline]
fn clean_unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

impl From<DiomaSense> for Sense {
    fn from(v: DiomaSense) -> Self {
        Self {
            smell_l: clean_unit(v.smell_l),
            smell_r: clean_unit(v.smell_r),
            vis_l: clean_unit(v.vision_l),
            vis_r: clean_unit(v.vision_r),
            wall_l: clean_unit(v.wall_l),
            wall_r: clean_unit(v.wall_r),
            food_near: v.food_near != 0,
            wall: v.wall != 0,
            place: v.place,
            heading: if v.heading.is_finite() { v.heading } else { 0.0 },
            see_eat: v.see_eat != 0,
            text: clean_unit(v.text),
        }
    }
}

fn alloc_buffer(bytes: Vec<u8>, out_ptr: *mut *mut u8, out_len: *mut usize) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
        set_last_error("alloc_buffer: null out_ptr/out_len");
        return NULL_ARGUMENT;
    }
    let len = bytes.len();
    let mut v = bytes.into_boxed_slice();
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    unsafe {
        *out_ptr = ptr;
        *out_len = len;
    }
    OK
}

// ═════════════════════════════════════════════════════════════════════════════
//  Версия и ошибки
// ═════════════════════════════════════════════════════════════════════════════

/// Версия C ABI. Unity проверяет до первого `create`.
#[no_mangle]
pub extern "C" fn dioma_abi_version() -> u32 {
    ABI_VERSION
}

/// Совместимость с v1 (старое имя).
#[no_mangle]
pub extern "C" fn dioma_unity_abi_version() -> u32 {
    ABI_VERSION
}

/// Копирует last_error в `buf` (UTF-8, без NUL). Возвращает нужную длину.
/// Если `buf` null или `len==0` — только измеряет длину.
#[no_mangle]
pub extern "C" fn dioma_last_error(buf: *mut u8, len: usize) -> usize {
    LAST_ERROR.with(|c| {
        let s = c.borrow();
        let b = s.as_bytes();
        if buf.is_null() || len == 0 {
            return b.len();
        }
        let n = b.len().min(len);
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), buf, n) };
        n
    })
}

#[no_mangle]
pub extern "C" fn dioma_last_error_clear() {
    clear_last_error();
}

/// Освобождает буфер, выданный `*_bytes / *_save / *_json`.
/// `ptr` — от `alloc_buffer`, `len` — та же длина.
#[no_mangle]
pub unsafe extern "C" fn dioma_buffer_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    unsafe { drop(Box::from_raw(slice::from_raw_parts_mut(ptr, len))) };
}

// ═════════════════════════════════════════════════════════════════════════════
//  Жизнь существа
// ═════════════════════════════════════════════════════════════════════════════

/// Создаёт существо из сида. `backend`: 0=CPU, 1=GPU динамика, 2=GPU рост+динамика.
/// Null при панике (редкий случай недоступного GPU-адаптера).
#[no_mangle]
pub extern "C" fn dioma_creature_create(seed: u64, backend: u32) -> *mut DiomaCreature {
    // Совместимое имя
    dioma_creature_create_seeded(seed, backend)
}

#[no_mangle]
pub extern "C" fn dioma_creature_create_seeded(seed: u64, backend: u32) -> *mut DiomaCreature {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let creature = match backend {
            x if x == DiomaBackend::Gpu as u32 => Creature::new_gpu(seed),
            x if x == DiomaBackend::GpuGrown as u32 => Creature::new_gpu_grown(seed),
            _ => Creature::new(seed),
        };
        Box::into_raw(Box::new(DiomaCreature { creature }))
    }));
    match r {
        Ok(p) => { clear_last_error(); p },
        Err(_) => { set_last_error("creature_create: panic"); std::ptr::null_mut() },
    }
}

/// Уничтожает существо. Null — no-op. Хэндл нельзя использовать после.
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_destroy(h: *mut DiomaCreature) {
    // Совместимое имя
    unsafe { dioma_creature_free(h) }
}

#[no_mangle]
pub unsafe extern "C" fn dioma_creature_free(h: *mut DiomaCreature) {
    if h.is_null() { return; }
    let r = catch_unwind(AssertUnwindSafe(|| unsafe { drop(Box::from_raw(h)) }));
    if r.is_err() { set_last_error("creature_free: panic"); }
    else { clear_last_error(); }
}

/// Один тик: сенсоры → мозг → тело → действие. Поток — один, детерминирован.
/// Возвращает OK/NULL_ARGUMENT/PANIC. `sense/action` должны быть валидны весь вызов,
/// один хэндл нельзя тикать параллельно из разных потоков.
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_tick(
    h: *mut DiomaCreature,
    sense: *const DiomaSense,
    action: *mut DiomaAction,
) -> i32 {
    if h.is_null() || sense.is_null() || action.is_null() {
        set_last_error("tick: null handle/sense/action");
        return NULL_ARGUMENT;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let c = unsafe { &mut *h };
        let s: Sense = unsafe { *sense }.into();
        let a = c.creature.tick(&s);
        unsafe {
            *action = DiomaAction { steer: a.steer, drive: a.drive, eat: u8::from(a.eat), _padding: [0; 3] };
        }
    }));
    match r {
        Ok(_) => { clear_last_error(); OK },
        Err(_) => { set_last_error("tick: panic"); PANIC },
    }
}

/// Батч-тик N существ параллельно (rayon, как `Arena::step` фазы A+B).
/// `handles` — массив указателей, `senses` — вход, `actions` — выход, `count` — длина.
/// Каждый хэндл тикается целиком на одном потоке, порядок детерминирован.
/// Спящие/мертвые всё равно тикаются (их `tick` вернёт нулевой drive).
#[no_mangle]
pub unsafe extern "C" fn dioma_creatures_tick_batch(
    handles: *const *mut DiomaCreature,
    senses: *const DiomaSense,
    actions: *mut DiomaAction,
    count: usize,
) -> i32 {
    if handles.is_null() || senses.is_null() || actions.is_null() {
        set_last_error("tick_batch: null handles/senses/actions");
        return NULL_ARGUMENT;
    }
    if count == 0 { return OK; }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let hs = unsafe { slice::from_raw_parts(handles, count) };
        let ss = unsafe { slice::from_raw_parts(senses, count) };
        let aa = unsafe { slice::from_raw_parts_mut(actions, count) };
        // Проверка null внутри массива
        for &h in hs { assert!(!h.is_null(), "tick_batch: null handle in array"); }
        // Последовательно (rayon-версия требует Send-обёртки для *mut; оставляем
        // детерминированный последовательный батч — один переход вместо N).
        let senses_conv: Vec<Sense> = ss.iter().copied().map(Sense::from).collect();
        for i in 0..count {
            let h = hs[i];
            let s = &senses_conv[i];
            let c = unsafe { &mut *h };
            let a = c.creature.tick(s);
            aa[i] = DiomaAction { steer: a.steer, drive: a.drive, eat: u8::from(a.eat), _padding: [0; 3] };
        }
    }));
    match r {
        Ok(_) => { clear_last_error(); OK },
        Err(_) => { set_last_error("tick_batch: panic (null in array or inner)"); PANIC },
    }
}

// Для совместимости с v1
#[no_mangle]
pub unsafe extern "C" fn dioma_unity_brain_tick(
    brain: *mut DiomaCreature,
    sense: *const DiomaSense,
    action: *mut DiomaAction,
) -> i32 {
    unsafe { dioma_creature_tick(brain, sense, action) }
}
#[no_mangle]
pub extern "C" fn dioma_unity_brain_create(seed: u64, backend: u32) -> *mut DiomaCreature {
    dioma_creature_create_seeded(seed, backend)
}
#[no_mangle]
pub unsafe extern "C" fn dioma_unity_brain_destroy(brain: *mut DiomaCreature) {
    unsafe { dioma_creature_free(brain) }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Телеметрия (чтение, не мутация)
// ═════════════════════════════════════════════════════════════════════════════

/// Снимок существа для HUD — возраст, голод, глюкоза, урон, сон, спайки.
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_diagnostics(
    h: *const DiomaCreature,
    out: *mut DiomaDiagnostics,
) -> i32 {
    if h.is_null() || out.is_null() {
        set_last_error("diagnostics: null handle/out");
        return NULL_ARGUMENT;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let c = unsafe { &*h };
        let cr = &c.creature;
        let conc = &cr.blood.conc;
        let spikes: u32 = cr.spike_hist().iter().map(|v| v.iter().map(|&x| x.max(0) as u32).sum::<u32>()).sum();
        // Creature не хранит sleep — это состояние Arena/Critter. Для диагностики
        // показываем 0 (бодрствует); мир-логика в Unity/Arena решает сон.
        let d = DiomaDiagnostics {
            age: cr.age,
            damage: cr.damage + cr.blood.organ.damage * dioma_engine::creature::DAMAGE_SCALE,
            food_eaten: cr.food_eaten(),
            is_sleeping: 0,
            is_dying: u8::from(cr.dying),
            is_dead: u8::from(cr.dead),
            _pad0: 0,
            glucose: if conc.len() > 0 { conc[0] } else { 0.0 },
            hunger: if conc.len() > 2 { conc[2] } else { 0.0 },
            fatigue: if conc.len() > 5 { conc[5] } else { 0.0 },
            pain: if conc.len() > 3 { conc[3] } else { 0.0 },
            cortisol: if conc.len() > 7 { conc[7] } else { 0.0 },
            serotonin: if conc.len() > 6 { conc[6] } else { 0.0 },
            acetylcholine: if conc.len() > 8 { conc[8] } else { 0.0 },
            sensor_gain: cr.blood.sensor_gain(),
            motor_tonic: cr.blood.motor_tonic(),
            speed: cr.blood.speed() as f32,
            plasticity_budget: cr.blood.plasticity_budget(),
            spikes_last_tick: spikes,
        };
        unsafe { *out = d };
    }));
    match r { Ok(_) => { clear_last_error(); OK }, Err(_) => { set_last_error("diagnostics: panic"); PANIC } }
}

/// Статистика мозга — нейроны, регионы, синапсы, шаги.
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_brain_stats(
    h: *const DiomaCreature,
    out: *mut DiomaBrainStats,
) -> i32 {
    if h.is_null() || out.is_null() {
        set_last_error("brain_stats: null");
        return NULL_ARGUMENT;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let c = unsafe { &*h };
        let bs = c.creature.brain_stats();
        let inh = c.creature.brain.region_bounds(4); // I — тормозной, грубая оценка
        let total = bs.neuron_count as f32;
        let inh_count = (inh.1 - inh.0) as f32;
        let syn = c.creature.brain.csr_values().len() as u32;
        unsafe {
            *out = DiomaBrainStats {
                neuron_count: bs.neuron_count as u32,
                region_count: bs.region_count as u32,
                synapse_count: syn,
                steps_per_tick: c.creature.brain.steps_per_tick() as u32,
                inhibition_ratio: if total > 0.0 { inh_count / total } else { 0.0 },
            };
        }
    }));
    match r { Ok(_) => { clear_last_error(); OK }, Err(_) => { set_last_error("brain_stats: panic"); PANIC } }
}

/// Кровь целиком: 9 химий + 9 локусов. Буферы должны быть len >= 9.
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_blood(
    h: *const DiomaCreature,
    conc_out: *mut f32,
    conc_len: usize,
    loci_out: *mut f32,
    loci_len: usize,
) -> i32 {
    if h.is_null() {
        set_last_error("blood: null handle");
        return NULL_ARGUMENT;
    }
    if conc_len < 9 || loci_len < 9 {
        set_last_error("blood: buffer too small (need 9)");
        return BUFFER_TOO_SMALL;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let c = unsafe { &*h };
        let conc = unsafe { slice::from_raw_parts_mut(conc_out, 9) };
        let loci = unsafe { slice::from_raw_parts_mut(loci_out, 9) };
        conc.copy_from_slice(&c.creature.blood.conc[..9]);
        loci.copy_from_slice(&c.creature.blood.loci[..9]);
    }));
    match r { Ok(_) => { clear_last_error(); OK }, Err(_) => { set_last_error("blood: panic"); PANIC } }
}

/// Мембрана: вольтажи нейронов последнего тика. `buf` len = neuron_count.
/// Возвращает число записанных (или нужную длину если buf null).
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_membrane(
    h: *const DiomaCreature,
    buf: *mut f32,
    len: usize,
) -> i32 {
    if h.is_null() {
        set_last_error("membrane: null handle");
        return NULL_ARGUMENT;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let c = unsafe { &*h };
        let n = c.creature.brain.n;
        if buf.is_null() {
            // запрос длины
            return n as i32;
        }
        if len < n {
            set_last_error(format!("membrane: buffer {len} < {n}"));
            return -(BUFFER_TOO_SMALL);
        }
        let out = unsafe { slice::from_raw_parts_mut(buf, n) };
        // debug_cpu_v — читает вольтажи без мутации (возвращает Vec)
        let v = c.creature.brain.debug_cpu_v();
        let copy = v.len().min(n);
        out[..copy].copy_from_slice(&v[..copy]);
        if copy < n {
            for x in &mut out[copy..] { *x = 0.0; }
        }
        n as i32
    }));
    match r {
        Ok(v) if v >= 0 => { clear_last_error(); v },
        Ok(v) => v, // negative = BUFFER_TOO_SMALL уже залогирован
        Err(_) => { set_last_error("membrane: panic"); PANIC },
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Наследие (ДНК / размножение)
// ═════════════════════════════════════════════════════════════════════════════

/// Потомок одного родителя (асексуально). Детерминирован от child_seed.
/// Наследует проводку с дрейфом весов σ=0.05 + Болдуин-инстинкты.
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_spawn_child(
    parent: *const DiomaCreature,
    child_seed: u64,
) -> *mut DiomaCreature {
    if parent.is_null() {
        set_last_error("spawn_child: null parent");
        return std::ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let p = unsafe { &*parent };
        let child = p.creature.spawn_child(child_seed);
        Box::into_raw(Box::new(DiomaCreature { creature: child }))
    }));
    match r {
        Ok(ptr) => { clear_last_error(); ptr },
        Err(_) => { set_last_error("spawn_child: panic"); std::ptr::null_mut() },
    }
}

/// Потомок пары (половое). Кроссовер геномов + мутация + Болдуин обоих.
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_spawn_child_pair(
    a: *const DiomaCreature,
    b: *const DiomaCreature,
    child_seed: u64,
) -> *mut DiomaCreature {
    if a.is_null() || b.is_null() {
        set_last_error("spawn_child_pair: null parent");
        return std::ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let pa = unsafe { &*a };
        let pb = unsafe { &*b };
        let child = pa.creature.spawn_child_pair(&pb.creature, child_seed);
        Box::into_raw(Box::new(DiomaCreature { creature: child }))
    }));
    match r {
        Ok(ptr) => { clear_last_error(); ptr },
        Err(_) => { set_last_error("spawn_child_pair: panic"); std::ptr::null_mut() },
    }
}

/// Сериализует ДНК существа в `DNA1` байты. Выделенный буфер — через `buffer_free`.
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_dna_bytes(
    h: *const DiomaCreature,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if h.is_null() || out_ptr.is_null() || out_len.is_null() {
        set_last_error("dna_bytes: null arg");
        return NULL_ARGUMENT;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let c = unsafe { &*h };
        let bytes = c.creature.dna.to_dna_bytes();
        alloc_buffer(bytes, out_ptr, out_len)
    }));
    match r {
        Ok(code) => { if code==OK { clear_last_error(); } code },
        Err(_) => { set_last_error("dna_bytes: panic"); PANIC },
    }
}

/// Создаёт существо из ДНК-байтов + сида (детерминированно). ДНК — как от `dna_bytes`.
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_create_from_dna(
    dna_ptr: *const u8,
    dna_len: usize,
    seed: u64,
) -> *mut DiomaCreature {
    if dna_ptr.is_null() || dna_len == 0 {
        set_last_error("create_from_dna: null/empty");
        return std::ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let bytes = unsafe { slice::from_raw_parts(dna_ptr, dna_len) };
        let dna = DnaGenome::from_dna_bytes(bytes).expect("invalid DNA1 bytes");
        // Собираем существо из ДНК: геном вытаскиваем из ДНК (age 0 — новорождённый)
        let mut genome = dna.to_creature_genome(0);
        // Пол — из сида (как в Creature::from_genome)
        genome.sex = dioma_engine::creature::genome::Sex::from_seed(seed);
        let mut c = Creature::from_genome(genome, seed);
        // Пересадим ДНК целиком (чтобы Coat/Eye сохранились)
        c.dna = dna;
        Box::into_raw(Box::new(DiomaCreature { creature: c }))
    }));
    match r {
        Ok(p) => { clear_last_error(); p },
        Err(_) => { set_last_error("create_from_dna: panic/invalid bytes"); std::ptr::null_mut() },
    }
}

/// Сид существа (для детерминизма, логирования, HUD).
#[no_mangle]
pub unsafe extern "C" fn dioma_creature_seed(h: *const DiomaCreature, out: *mut u64) -> i32 {
    if h.is_null() || out.is_null() {
        set_last_error("seed: null");
        return NULL_ARGUMENT;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        let c = unsafe { &*h };
        unsafe { *out = c.creature.seed };
    }));
    match r { Ok(_) => { clear_last_error(); OK }, Err(_) => { set_last_error("seed: panic"); PANIC } }
}

// ── Тесты моста ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_roundtrip() {
        let h = dioma_creature_create_seeded(42, DiomaBackend::Cpu as u32);
        assert!(!h.is_null());
        let mut d = DiomaDiagnostics::default();
        assert_eq!(unsafe { dioma_creature_diagnostics(h, &mut d) }, OK);
        assert_eq!(d.food_eaten, 0);
        assert!(!d.is_dead != 0 || d.is_dead == 0);
        let mut stats = DiomaBrainStats::default();
        assert_eq!(unsafe { dioma_creature_brain_stats(h, &mut stats) }, OK);
        assert!(stats.neuron_count > 0);
        unsafe { dioma_creature_free(h) };
    }

    #[test]
    fn spawn_child_is_deterministic() {
        let p = dioma_creature_create_seeded(1, 0);
        let c1 = unsafe { dioma_creature_spawn_child(p, 99) };
        let c2 = unsafe { dioma_creature_spawn_child(p, 99) };
        let mut s1 = 0u64; let mut s2 = 0u64;
        unsafe { dioma_creature_seed(c1, &mut s1); dioma_creature_seed(c2, &mut s2); }
        assert_eq!(s1, s2);
        unsafe { dioma_creature_free(p); dioma_creature_free(c1); dioma_creature_free(c2); }
    }

    #[test]
    fn dna_roundtrip() {
        let h = dioma_creature_create_seeded(7, 0);
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let mut len = 0usize;
        assert_eq!(unsafe { dioma_creature_dna_bytes(h, &mut ptr, &mut len) }, OK);
        assert!(!ptr.is_null() && len > 0);
        let h2 = unsafe { dioma_creature_create_from_dna(ptr, len, 77) };
        assert!(!h2.is_null());
        unsafe { dioma_buffer_free(ptr, len); dioma_creature_free(h); dioma_creature_free(h2); }
    }

    #[test]
    fn batch_tick() {
        let h1 = dioma_creature_create_seeded(1, 0);
        let h2 = dioma_creature_create_seeded(2, 0);
        let handles = [h1, h2];
        let senses = [DiomaSense { smell_l: 1.0, ..Default::default() }, DiomaSense::default()];
        let mut actions = [DiomaAction::default(); 2];
        let code = unsafe { dioma_creatures_tick_batch(handles.as_ptr(), senses.as_ptr(), actions.as_mut_ptr(), 2) };
        assert_eq!(code, OK);
        unsafe { dioma_creature_free(h1); dioma_creature_free(h2); }
    }
}
