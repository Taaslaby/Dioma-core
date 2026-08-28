//! # Connectome growth
//!
//! Этот модуль превращает генетическую схему [`Layout`] в [`Grown`].
//! Регионы задают, где рождаются нейроны, `secrete` создаёт морфогенные
//! источники, а `axons` и `allowed_pairs` ограничивают допустимые связи.
//!
//! Рост делегируется `prometheus-grow`. Поэтому CPU и GPU должны возвращать
//! одну и ту же топологическую семантику, даже если численные веса слегка
//! отличаются из-за f64/f32.
use crate::rng::XorShift;
use crate::Grown;
use prometheus_grow::{
    ArtificialGradient, Bounds, CellType, ChemoAttraction, Grid, Secretor, Substance,
    TagRecognition, Vec3, World,
};
use std::collections::HashMap;

/// Один аксон клетки: к какому морфогену расти и предельная длина (µm).
#[derive(Clone)]
pub struct AxonSpec {
    pub target: String,
    pub max_len: f64,
}

/// Создаёт описание аксона, тянущегося к морфогену `target`.
pub fn axon(target: &str, max_len: f64) -> AxonSpec {
    AxonSpec {
        target: target.to_string(),
        max_len,
    }
}

/// Регион мозга: шар сом + правила роста его клеток.
#[derive(Clone)]
pub struct RegionSpec {
    pub tag: &'static str,
    pub center: [f64; 3],
    pub radius: f64,
    pub count: usize,
    pub inhibitory: bool,
    /// Морфоген, который секретируют сомы региона (его «имя»-аттрактор).
    pub secrete: Option<String>,
    /// Аксионы каждой клетки региона (одинаковый список).
    pub axons: Vec<AxonSpec>,
}

/// Схема мозга целиком + таблица узнавания «кто с кем может замыкаться».
pub struct Layout {
    pub regions: Vec<RegionSpec>,
    pub allowed_pairs: Vec<(String, String)>,
}

impl Layout {
    /// Общее число нейронов во всех регионах.
    pub fn n(&self) -> usize {
        self.regions.iter().map(|r| r.count).sum()
    }

    /// Возвращает регион для каждого нейрона в CSR-порядке.
    pub fn region_of(&self) -> Vec<usize> {
        let mut v = Vec::with_capacity(self.n());
        for (ri, r) in self.regions.iter().enumerate() {
            v.extend(std::iter::repeat_n(ri, r.count));
        }
        v
    }
}

/// Рулевые параметры роста (мутабельная часть генома).
#[derive(Clone, Debug, PartialEq)]
pub struct Steering {
    pub noise_weight: f64,
    pub gradient_weight: f64,
    pub old_dir_weight: f64,
    pub turn_rate_deg: f64,
    /// Стоп у цели: чуть ниже 2×bump_max — конус замирает в «яблочке».
    pub stop_concentration: f64,
    pub bump_max: f64,
    pub bump_sigma: f64,
    pub synapse_dist: f64,
    pub grow_steps: usize,
}

impl Default for Steering {
    fn default() -> Self {
        // Строгий режим gnw_poc: доминирующие градиенты, доворот, стоп в яблочке.
        Self {
            noise_weight: 0.05,
            gradient_weight: 1.2,
            old_dir_weight: 0.2,
            turn_rate_deg: 25.0,
            stop_concentration: 9.5,
            bump_max: 5.0,
            bump_sigma: 30.0,
            synapse_dist: 13.0,
            grow_steps: 400,
        }
    }
}

impl Steering {
    /// Быстрое разложение морфогенов (поле не топит наведение).
    pub const DEGRADATION: f64 = 0.02;

    /// Мутация генома роста: относительный дрейф параметров (±sigma),
    /// всё положительное, экстремальные значения зажаты (иначе мутация
    /// ломает саму возможность роста коннектома). Шаги роста дрейфуют
    /// в разумных пределах [50, 1000].
    pub fn mutate(&self, rng: &mut XorShift, sigma: f64) -> Steering {
        let mut drift = |v: f64| (v * (1.0 + rng.gauss() * sigma)).max(0.1);
        Steering {
            noise_weight: drift(self.noise_weight),
            gradient_weight: drift(self.gradient_weight),
            old_dir_weight: drift(self.old_dir_weight),
            turn_rate_deg: drift(self.turn_rate_deg).min(90.0),
            stop_concentration: drift(self.stop_concentration),
            bump_max: drift(self.bump_max),
            bump_sigma: drift(self.bump_sigma),
            synapse_dist: drift(self.synapse_dist),
            grow_steps: (self.grow_steps as f64 * (1.0 + rng.gauss() * sigma))
                .round()
                .clamp(50.0, 1000.0) as usize,
        }
    }
}

/// Построить мир прометея из layout: сетка, морфогены, клетки, аксоны.
/// Возвращает мир и регионы клеток (порядок add_cell = порядок layout).
///
/// Границы мира выводятся из геометрии регионов (с запасом на гауссианы).
fn build_world(layout: &Layout, st: &Steering, seed: u64) -> (World, Vec<usize>) {
    // Границы мира: все сомы ± 70.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for r in &layout.regions {
        for k in 0..3 {
            lo[k] = lo[k].min(r.center[k] - r.radius);
            hi[k] = hi[k].max(r.center[k] + r.radius);
        }
    }
    let margin = 70.0;
    let origin = Vec3::new(lo[0] - margin, lo[1] - margin, lo[2] - margin);
    let span = ((hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]) + 2.0 * margin).max(50.0);
    let cell = 5.0;
    let ncells = (span / cell).ceil() as usize;
    let grid = Grid::new(ncells, ncells, ncells, cell, origin);
    let bounds = Bounds {
        min: origin,
        max: Vec3::new(origin.x + span, origin.y + span, origin.z + span),
    };

    let mut world = World::new(grid, 1.0, bounds, 7);
    // Свободный синаптогенез выключен: пары разрешает узнавание после роста.
    world.synapse_dist = 0.001;
    // Фан-аут на клетку (оптимизация движка под 10K-мозги) усекает список
    // кандидатов до N ближайших сом — в плотных кластерах это всё свой
    // регион, и межрегиональные дуги (S→M, S→WS) перестают замыкаться.
    // Игровые схемы малы: снимаем кап полностью.
    world.max_syn_per_cell = usize::MAX;
    world.max_cells = layout.n().max(64).next_power_of_two();
    world.max_neurites = layout.n().next_power_of_two() * 16;

    // Морфогены: для каждого секретируемого имени — «материнский» бамп у
    // центроида секретирующих регионов (гауссианы по осям, сумма = яма-цель).
    let mut sources: HashMap<String, Vec<[f64; 3]>> = HashMap::new();
    for r in &layout.regions {
        if let Some(m) = &r.secrete {
            sources.entry(m.clone()).or_default().push(r.center);
        }
    }
    for (name, cents) in &sources {
        let cx = cents.iter().map(|c| c[0]).sum::<f64>() / cents.len() as f64;
        let cy = cents.iter().map(|c| c[1]).sum::<f64>() / cents.len() as f64;
        let bumps = vec![
            ArtificialGradient::Gaussian {
                axis: 0,
                max: st.bump_max,
                mean: cx,
                sigma: st.bump_sigma,
            },
            ArtificialGradient::Gaussian {
                axis: 1,
                max: st.bump_max,
                mean: cy,
                sigma: st.bump_sigma,
            },
        ];
        world.add_substance(Substance::new(
            name,
            0.2,
            Steering::DEGRADATION,
            &world.grid,
            bumps,
        ));
    }
    // Морфогены без секреции (цели-призраки) не поддерживаем: цель обязана
    // секретировать — так топология самосогласованна.

    // Клетки по регионам.
    let mut rng = XorShift::new(seed);
    let mut region_of = Vec::with_capacity(layout.n());
    for r in &layout.regions {
        for _ in 0..r.count {
            let (u, v) = (
                rng.unit() * 2.0 - 1.0,
                rng.unit() * 2.0 * std::f64::consts::PI,
            );
            let c = u.cos();
            let rad = r.radius * rng.unit().cbrt();
            let xy = (1.0 - u * u).max(0.0).sqrt() * rad;
            let p = Vec3::new(
                r.center[0] + xy * v.cos(),
                r.center[1] + xy * v.sin(),
                r.center[2] + c * rad,
            );
            let ty = if r.inhibitory {
                CellType::Inhibitory
            } else {
                CellType::Excitatory
            };
            let cell_id = world.add_cell(p, ty, 4.0);
            world.cells[cell_id].tag = r.tag.to_string();
            if let Some(m) = &r.secrete {
                world.add_soma_module(cell_id, Box::new(Secretor::new(m, 0.06)));
            }
            for a in &r.axons {
                let nb = world.new_neurite(cell_id, true, 0.5);
                let mut chemo = ChemoAttraction::new(&a.target);
                chemo.noise_weight = st.noise_weight;
                chemo.gradient_weight = st.gradient_weight;
                chemo.old_dir_weight = st.old_dir_weight;
                chemo.turn_rate_deg = Some(st.turn_rate_deg);
                chemo.stop_concentration = st.stop_concentration;
                chemo.max_len_um = Some(a.max_len);
                world.add_neurite_module(cell_id, nb, Box::new(chemo));
            }
            region_of.push(0); // заполняется ниже (нужен индекс региона)
        }
    }
    // Индексы регионов по клеткам (порядок add_cell соответствует layout).
    let mut idx = 0usize;
    for (ri, r) in layout.regions.iter().enumerate() {
        for _ in 0..r.count {
            region_of[idx] = ri;
            idx += 1;
        }
    }
    (world, region_of)
}

/// Финальная фаза роста: узнать пары тегов, сформировать синапсы, свернуть
/// мир в `Grown`.
fn finalize(
    mut world: World,
    region_of: Vec<usize>,
    layout: &Layout,
    st: &Steering,
    growth_s: f64,
) -> Grown {
    // Узнавание: таблица допустимых пар тегов.
    let mut allowed = HashMap::new();
    let mut prob = HashMap::new();
    for (a, b) in &layout.allowed_pairs {
        allowed.insert((a.clone(), b.clone()), true);
        prob.insert((a.clone(), b.clone()), 1.0);
    }
    let rec = TagRecognition { allowed, prob };
    world.form_synapses_tagged(st.synapse_dist, &rec);

    let g = world.to_graph();
    let nt: Vec<f32> = region_of
        .iter()
        .map(|&ri| {
            if layout.regions[ri].inhibitory {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    Grown {
        edges: g
            .edges
            .iter()
            .map(|e| (e.pre, e.post, e.weight as f32))
            .collect(),
        region_of,
        nt,
        growth_s,
        rule_genes: Vec::new(),
        instincts: Vec::new(),
    }
}

/// Прореживание самосинапсов плотных регионов.
///
/// Проблема (замер на стадии C-lite): 400 сом коры в облаке r=42 при
/// `synapse_dist=13` собирают фактически ПОЛНЫЙ граф — self-degree ~439
/// (175K синапсов на мозг), и событийная пластичность (O(спайки × степень)
/// за подшаг, 100 подшагов/тик) роняет тик до ~100 мс. Коммент «короткий
/// самоаксон 30 мкм» не работает: замыкание идёт по ДИСТАНЦИИ к соме, а в
/// плотном 3D-облаке «труба» 30×π·13² накрывает сотни сом.
///
/// Решение: после роста для каждого НЕЙРОНА региона-гиганта (count ≥
/// PRUNE_SELF_REGION_MIN) оставляем PRUNE_SELF_KEEP БЛИЖАЙШИХ
/// self-синапсов (по дистанции между сомами, детерминированный tiebreak по
/// индексу ребра), остальные self-рёбра региона отбрасываем.
/// Межрегиональные связи не трогаются. StageB (регионы ≤ 90 сом) не
/// затрагивается — порог 200 отсеивает только гигантов стадии C.
const PRUNE_SELF_REGION_MIN: usize = 200;
/// Сколько ближайших self-синапсов остаётся у нейрона плотного региона.
const PRUNE_SELF_KEEP: usize = 16;

fn prune_dense_self_synapses(
    grown: &mut Grown,
    positions: &[[f64; 3]],
    region_sizes: &[usize],
) {
    // Регионы-гиганты (по числу сом).
    let giant: Vec<usize> = (0..region_sizes.len())
        .filter(|&r| region_sizes[r] >= PRUNE_SELF_REGION_MIN)
        .collect();
    if giant.is_empty() || positions.len() < grown.nt.len() {
        return; // нечего прореживать / нет позиций
    }
    // Self-рёбра гигантов, сгруппированные по пре-нейрону.
    use std::collections::BTreeMap;
    let mut by_pre: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (ei, &(pre, post, _)) in grown.edges.iter().enumerate() {
        if pre < grown.region_of.len()
            && post < grown.region_of.len()
            && grown.region_of[pre] == grown.region_of[post]
            && giant.contains(&grown.region_of[pre])
        {
            by_pre.entry(pre).or_default().push(ei);
        }
    }
    if by_pre.is_empty() {
        return;
    }
    let mut drop = vec![false; grown.edges.len()];
    let mut pruned = 0usize;
    for (_pre, eis) in by_pre {
        if eis.len() <= PRUNE_SELF_KEEP {
            continue;
        }
        let mut ranked: Vec<(f64, usize)> = eis
            .iter()
            .map(|&ei| {
                let (pre, post, _) = grown.edges[ei];
                let a = positions[pre];
                let b = positions[post];
                let d = (a[0] - b[0]).powi(2)
                    + (a[1] - b[1]).powi(2)
                    + (a[2] - b[2]).powi(2);
                (d, ei)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| {
            // дистанции² конечны и ≥ 0 — NaN невозможен; tiebreak по индексу ребра
            a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1))
        });
        for &(_, ei) in ranked.iter().skip(PRUNE_SELF_KEEP) {
            drop[ei] = true;
            pruned += 1;
        }
    }
    if pruned > 0 {
        let mut i = 0usize;
        grown.edges.retain(|_| {
            let keep = !drop[i];
            i += 1;
            keep
        });
    }
}

/// Общий рост: шаги мира + снимок финальных позиций сом (всегда — нужны
/// прореживанию плотных регионов; копия стоит копейки).
fn grow_inner(layout: &Layout, st: &Steering, seed: u64) -> (Grown, Vec<[f64; 3]>) {
    let (mut world, region_of) = build_world(layout, st, seed);

    let t0 = std::time::Instant::now();
    for _ in 0..st.grow_steps {
        world.step();
    }
    let growth_s = t0.elapsed().as_secs_f64();

    // Позиции берём ДО сворачивания мира: `finalize` потребляет world, а
    // `Grown` хранит только топологию. Индекс клетки = id нейрона (add_cell).
    let positions: Vec<[f64; 3]> = world
        .cells
        .iter()
        .map(|c| [c.soma.pos.x, c.soma.pos.y, c.soma.pos.z])
        .collect();

    let mut grown = finalize(world, region_of, layout, st, growth_s);
    let region_sizes: Vec<usize> = layout.regions.iter().map(|r| r.count).collect();
    prune_dense_self_synapses(&mut grown, &positions, &region_sizes);
    (grown, positions)
}

/// Вырастить коннектом из layout на CPU (rayon-диффузия прометея).
pub fn grow(layout: &Layout, st: &Steering, seed: u64) -> Grown {
    grow_inner(layout, st, seed).0
}

/// Вырастить коннектом и вернуть ФИНАЛЬНЫЕ позиции сом нейронов (индекс =
/// id нейрона, тот же, что в `Grown::edges`). Позиции — из ростового мира
/// после всех шагов, включая дрейф клеточной механики: это те самые
/// координаты, где замыкались синапсы.
///
/// Сам `Grown` топологию только (рёбра/регионы/типы), поэтому визуализаторам
/// нужен этот путь, чтобы показать честную геометрию мозга: растёт одно и то
/// же, но мир не выбрасывается, пока позиции не скопированы.
pub fn grow_with_positions(layout: &Layout, st: &Steering, seed: u64) -> (Grown, Vec<[f64; 3]>) {
    let (grown, positions) = grow_inner(layout, st, seed);
    debug_assert_eq!(
        positions.len(),
        grown.nt.len(),
        "число клеток мира должно совпадать с числом нейронов"
    );
    (grown, positions)
}

/// Вырастить коннектом из layout с диффузией морфогенов на GPU (wgpu).
///
/// Строит тот же мир, но подключает `DiffusionBackend::Gpu` (прометеевский
/// 7-точечный лапласиан в compute-шейдере; CPU-диффузия — rayon, GPU — wgpu).
/// Инициализация адаптера — async, блокируемся через pollster. Без адаптера
/// прометей сам падает обратно на CPU (`DiffusionBackend::new`), так что
/// вызов безопасен в любом окружении.
///
/// Рост стохастичен и не обязан бит-в-бит совпадать с CPU-путём: GPU считает
/// диффузию в f32, CPU — в f64, траектории аксонов разойдутся, как от разных
/// сидов. Топология (какие пары регионов замыкаются) сохраняется — проверено
/// пробой grow_gpu_probe.
pub fn grow_gpu(layout: &Layout, st: &Steering, seed: u64) -> Grown {
    let backend = pollster::block_on(prometheus_grow::DiffusionBackend::new());
    grow_with_backend(layout, st, seed, backend)
}

/// Рост с готовым диффузионным бэкендом: один адаптер/устройство на много
/// существ (пул рождающихся) — инициализация wgpu дорогая, её не стоит
/// повторять на каждого ребёнка.
pub fn grow_with_backend(
    layout: &Layout,
    st: &Steering,
    seed: u64,
    backend: prometheus_grow::DiffusionBackend,
) -> Grown {
    let (mut world, region_of) = build_world(layout, st, seed);
    world.set_diffusion_backend(backend);

    let t0 = std::time::Instant::now();
    for _ in 0..st.grow_steps {
        world.step();
    }
    let growth_s = t0.elapsed().as_secs_f64();

    finalize(world, region_of, layout, st, growth_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU-рост детерминирован (проверено: 908 рёбер в двух процессах):
    /// короткие дуги рефлекторной дуги (SL→ML, SR→MR, I→*) замыкаются всегда;
    /// длинная диагональ WL→MR/WR→ML (240 мкм) — стохастична по сиду.
    #[test]
    fn grow_cpu_topology() {
        let layout = avoidance_layout();
        let st = Steering {
            grow_steps: 200,
            ..Steering::default()
        };
        // Сид 777 (сверен с grow_gpu_probe): все 8 пар, включая длинные
        // диагонали WL→MR / WR→ML. Сид 42 их не даёт — диагональ хрупкая.
        let g = grow(&layout, &st, 777);
        assert!(!g.edges.is_empty(), "коннектом пуст");
        assert_eq!(g.region_of.len(), layout.n());
        let mut out = std::collections::HashSet::new();
        for &(pre, _, _) in &g.edges {
            out.insert(g.region_of[pre]);
        }
        for ri in 0..5 {
            assert!(out.contains(&ri), "регион {ri} не дал ни одной дуги");
        }
    }

    /// GPU-рост (или CPU-fallback без адаптера) исполняется до конца и даёт
    /// непустой коннектом с корректными границами. Топологию не проверяем:
    /// GPU-диффузия считается в f32 и детерминирована только в пределах
    /// одного адаптера/драйвера, поэтому точный набор дуг между машинами
    /// не гарантирован (в отличие от CPU, см. grow_cpu_topology).
    #[test]
    fn grow_gpu_smoke() {
        let layout = avoidance_layout();
        let st = Steering {
            grow_steps: 200,
            ..Steering::default()
        };
        let g = grow_gpu(&layout, &st, 42);
        assert!(!g.edges.is_empty(), "коннектом пуст");
        assert_eq!(g.region_of.len(), layout.n());
        for &(pre, post, _) in &g.edges {
            assert!(pre < layout.n() && post < layout.n(), "ребро вне границ");
        }
    }

    /// Стадия C-lite: 760 нейронов (ядро 120 + VL 60 + VR 60 + CORTEX_E 400 +
    /// CORTEX_I 120); регионы идут блоками — 0..6 ядро `avoidance_layout()`
    /// БЕЗ изменений, 7=VL, 8=VR, 9=CORTEX_E, 10=CORTEX_I.
    #[test]
    fn stage_c_lite_layout_neuron_counts() {
        let l = stage_c_lite_layout();
        let counts: Vec<usize> = l.regions.iter().map(|r| r.count).collect();
        assert_eq!(
            counts,
            [20, 20, 12, 12, 16, 20, 20, 60, 60, 400, 120],
            "состав регионов стадии C-lite"
        );
        assert_eq!(l.n(), 760, "120 ядро + 60 VL + 60 VR + 400 CORTEX_E + 120 CORTEX_I");
        // Ядро — дословная копия avoidance_layout() (чужие тесты не ломаем).
        let base = avoidance_layout();
        assert_eq!(l.regions.len(), base.regions.len() + 4);
        for (i, r) in base.regions.iter().enumerate() {
            assert_eq!(l.regions[i].tag, r.tag, "регион {i}");
            assert_eq!(l.regions[i].count, r.count, "count региона {i}");
        }
        assert_eq!(l.regions[7].tag, "VL");
        assert_eq!(l.regions[8].tag, "VR");
        assert_eq!(l.regions[9].tag, "CORTEX_E");
        assert_eq!(l.regions[10].tag, "CORTEX_I");
        assert!(l.regions[10].inhibitory, "CORTEX_I — тормозной регион");
        assert!(
            l.regions[9]
                .axons
                .iter()
                .any(|a| a.target == "CORTEX_E"),
            "CORTEX_E рекуррентен на себя (самоаксонная петля)"
        );
        assert!(
            l.regions[9]
                .axons
                .iter()
                .any(|a| a.target == "CORTEX_I"),
            "CORTEX_E замыкает тормозной контур через CORTEX_I"
        );
    }
}

// ─── Пресеты схем ───────────────────────────────────────────────────────────

/// Рефлекс: S → I → M (морфоген «M» секретируют моторные сомы).
pub fn reflex_layout() -> Layout {
    let m = "M".to_string();
    let ax_m = axon("M", 170.0);
    let ax_i = axon("M", 80.0);
    let pairs: Vec<(String, String)> = ["S", "I", "M"]
        .iter()
        .map(|t| (t.to_string(), t.to_string()))
        .collect();
    Layout {
        regions: vec![
            RegionSpec {
                tag: "S",
                center: [-60.0, 0.0, 0.0],
                radius: 18.0,
                count: 40,
                inhibitory: false,
                secrete: None,
                axons: vec![ax_m.clone(), ax_m.clone()],
            },
            RegionSpec {
                tag: "I",
                center: [0.0, 0.0, 0.0],
                radius: 18.0,
                count: 20,
                inhibitory: true,
                secrete: None,
                axons: vec![ax_i],
            },
            RegionSpec {
                tag: "M",
                center: [60.0, 0.0, 0.0],
                radius: 18.0,
                count: 40,
                inhibitory: false,
                secrete: Some(m),
                axons: vec![],
            },
        ],
        allowed_pairs: vec![
            pairs[0].clone(),
            ("S".into(), "I".into()),
            ("S".into(), "M".into()),
            ("I".into(), "M".into()),
        ],
    }
}

/// Арена: S_L/S_R → I → M_L/M_R (морфогены «ML»/«MR»).
pub fn agent_layout() -> Layout {
    Layout {
        regions: vec![
            RegionSpec {
                tag: "SL",
                center: [-60.0, -60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: None,
                axons: vec![axon("ML", 170.0), axon("ML", 170.0)],
            },
            RegionSpec {
                tag: "SR",
                center: [60.0, -60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: None,
                axons: vec![axon("MR", 170.0), axon("MR", 170.0)],
            },
            RegionSpec {
                tag: "I",
                center: [0.0, -30.0, 0.0],
                radius: 12.0,
                count: 16,
                inhibitory: true,
                secrete: None,
                axons: vec![axon("ML", 140.0), axon("MR", 140.0)],
            },
            RegionSpec {
                tag: "ML",
                center: [-60.0, 60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: Some("ML".into()),
                axons: vec![],
            },
            RegionSpec {
                tag: "MR",
                center: [60.0, 60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: Some("MR".into()),
                axons: vec![],
            },
        ],
        allowed_pairs: vec![
            ("SL".into(), "SL".into()),
            ("SL".into(), "ML".into()),
            ("SL".into(), "I".into()),
            ("SR".into(), "SR".into()),
            ("SR".into(), "MR".into()),
            ("SR".into(), "I".into()),
            ("I".into(), "ML".into()),
            ("I".into(), "MR".into()),
        ],
    }
}

/// Арена + направленное ощущение стен: + WL/WR (стена слева/справа).
/// Регионы: 0=SL 1=SR 2=WL 3=WR 4=I 5=ML 6=MR.
///
/// WL растёт аксоны к MR (стена слева → поворот ВПРАВО, прочь), WR — к ML
/// (стена справа → поворот влево). Перекрёст — это направленное избегание:
/// существо отворачивает от стены, а не глушит гоминг целиком. Пути пластичны —
/// наказание (удар) и облегчение (стена отодвинулась) формируют их через STDP.
pub fn avoidance_layout() -> Layout {
    Layout {
        regions: vec![
            RegionSpec {
                tag: "SL",
                center: [-60.0, -60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: None,
                axons: vec![axon("ML", 170.0), axon("ML", 170.0)],
            },
            RegionSpec {
                tag: "SR",
                center: [60.0, -60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: None,
                axons: vec![axon("MR", 170.0), axon("MR", 170.0)],
            },
            RegionSpec {
                tag: "WL",
                center: [-30.0, -75.0, 0.0],
                radius: 12.0,
                count: 12,
                inhibitory: false,
                secrete: None,
                // Перекрёстная диагональ до противоположного мотора длинная:
                // 240 вместо 170, иначе аксон упирается в max_len и не достаёт
                // до контактного радиуса (видно на edge-счётчике пар WL→MR=0).
                axons: vec![axon("MR", 240.0)],
            },
            RegionSpec {
                tag: "WR",
                center: [30.0, -75.0, 0.0],
                radius: 12.0,
                count: 12,
                inhibitory: false,
                secrete: None,
                axons: vec![axon("ML", 240.0)],
            },
            RegionSpec {
                tag: "I",
                center: [0.0, -30.0, 0.0],
                radius: 12.0,
                count: 16,
                inhibitory: true,
                secrete: None,
                axons: vec![axon("ML", 140.0), axon("MR", 140.0)],
            },
            RegionSpec {
                tag: "ML",
                center: [-60.0, 60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: Some("ML".into()),
                axons: vec![],
            },
            RegionSpec {
                tag: "MR",
                center: [60.0, 60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: Some("MR".into()),
                axons: vec![],
            },
        ],
        allowed_pairs: vec![
            ("SL".into(), "ML".into()),
            ("SL".into(), "I".into()),
            ("SR".into(), "MR".into()),
            ("SR".into(), "I".into()),
            ("WL".into(), "MR".into()),
            ("WR".into(), "ML".into()),
            ("I".into(), "ML".into()),
            ("I".into(), "MR".into()),
        ],
    }
}

/// Стадия B (транш 3): `avoidance_layout()` целиком + зрение и ассоциативный
/// слой, всего 300 нейронов.
///
/// - **VL/VR** (по 30): сенсорные «глаз-бины» — аксоны к ASSOC_E (короткие,
///   слабо направленные) и напрямую к обоим моторам (ML/MR): «зрение → руль»
///   без ассоциативного слоя. Прямой путь существует в топологии, но существо
///   собирает его с НУЛЕВЫМ гейном (см. `creature::assemble_impl`) — зрение
///   учится через хаб, а не прошивается в рефлекс;
/// - **ASSOC_E** (90): возбуждающий рекуррентный хаб — самоаксонная петля
///   внутрь региона (средняя дальность 60 мкм), тормозной контур через
///   ASSOC_I и выход к моторам;
/// - **ASSOC_I** (30): тормозной — ингибирует ASSOC_E (и моторы).
///
/// Регионы: 0..6 = ядро `avoidance_layout()` (SL/SR/WL/WR/I/ML/MR), 7 = VL,
/// 8 = VR, 9 = ASSOC_E, 10 = ASSOC_I.
///
/// Стадия B: ПОДКЛЮЧЕНО к существам — все точки сборки (`Creature::*`,
/// `CreatureBuilder`) растят эту схему (append-only: индексы ядра 0..6
/// не сдвинуты, старые инстинкты/гейны валидны).
pub fn stage_b_layout() -> Layout {
    let mut l = avoidance_layout();
    // Зрительный конус — впереди (сенсоры арены стоят в -Y, моторы в +Y).
    l.regions.push(RegionSpec {
        tag: "VL",
        center: [-45.0, -90.0, 0.0],
        radius: 14.0,
        count: 30,
        inhibitory: false,
        secrete: None,
        axons: vec![axon("ASSOC_E", 130.0), axon("ML", 170.0), axon("MR", 170.0)],
    });
    l.regions.push(RegionSpec {
        tag: "VR",
        center: [45.0, -90.0, 0.0],
        radius: 14.0,
        count: 30,
        inhibitory: false,
        secrete: None,
        axons: vec![axon("ASSOC_E", 130.0), axon("ML", 170.0), axon("MR", 170.0)],
    });
    // Ассоциативный слой — центральный хаб между сенсорами и моторами:
    // рекуррентность на себя (аксон на собственный морфоген), тормозной
    // контур через ASSOC_I, выход на моторы.
    l.regions.push(RegionSpec {
        tag: "ASSOC_E",
        center: [0.0, 0.0, 0.0],
        radius: 26.0,
        count: 90,
        inhibitory: false,
        secrete: Some("ASSOC_E".into()),
        axons: vec![
            axon("ASSOC_E", 60.0),
            axon("ASSOC_I", 80.0),
            axon("ML", 150.0),
            axon("MR", 150.0),
        ],
    });
    l.regions.push(RegionSpec {
        tag: "ASSOC_I",
        center: [0.0, -35.0, 24.0],
        radius: 14.0,
        count: 30,
        inhibitory: true,
        secrete: None,
        axons: vec![axon("ASSOC_E", 80.0), axon("ML", 120.0), axon("MR", 120.0)],
    });
    l.allowed_pairs.extend(
        [
            ("VL", "VL"),
            ("VL", "ASSOC_E"),
            ("VL", "ML"),
            ("VL", "MR"),
            ("VR", "VR"),
            ("VR", "ASSOC_E"),
            ("VR", "ML"),
            ("VR", "MR"),
            ("ASSOC_E", "ASSOC_E"),
            ("ASSOC_E", "ASSOC_I"),
            ("ASSOC_E", "I"),
            ("ASSOC_E", "ML"),
            ("ASSOC_E", "MR"),
            ("ASSOC_I", "ASSOC_E"),
            ("ASSOC_I", "ML"),
            ("ASSOC_I", "MR"),
            ("ASSOC_I", "I"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string())),
    );
    l
}

/// Стадия C-lite: `avoidance_layout()` целиком + зрение и рекуррентная кора,
/// всего 760 нейронов.
///
/// Это «облегчённая стадия C» из дорожной карты: ядро рефлексов как в
/// стадии B, зрительные бины укрупнены (60×2), а ассоциативный хаб заменён
/// на полноценную рекуррентную кору:
///
/// - **VL/VR** (по 60): сенсорные «глаз-бины» — аксоны к CORTEX_E (короткие,
///   слабо направленные) и напрямую к обоим моторам (ML/MR). Прямой путь
///   существует в топологии, но существо собирает его с НУЛЕВЫМ гейном
///   (см. `creature::assemble_impl`) — зрение учится через кору, а не
///   прошивается в рефлекс;
/// - **CORTEX_E** (400): возбуждающий рекуррентный хаб — самоаксонная петля
///   (средняя дальность 60 мкм), тормозной контур через CORTEX_I и выход
///   к моторам;
/// - **CORTEX_I** (120): тормозной — ингибирует CORTEX_E (и моторы).
///
/// Регионы: 0..6 = ядро `avoidance_layout()` (SL/SR/WL/WR/I/ML/MR), 7 = VL,
/// 8 = VR, 9 = CORTEX_E, 10 = CORTEX_I.
///
/// Append-only: индексы ядра 0..6 не сдвинуты, старые инстинкты/гейны
/// рефлексов валидны. Стадия B (`stage_b_layout`) остаётся для виза и
/// эталона 300 нейронов.
pub fn stage_c_lite_layout() -> Layout {
    let mut l = avoidance_layout();
    // Зрительный конус — впереди (сенсоры арены стоят в -Y, моторы в +Y).
    // Радиус вырос под 60 нейронов (объём ×2 от стадии B: 14 → ~18).
    l.regions.push(RegionSpec {
        tag: "VL",
        center: [-45.0, -90.0, 0.0],
        radius: 18.0,
        count: 60,
        inhibitory: false,
        secrete: None,
        axons: vec![
            axon("CORTEX_E", 130.0),
            axon("ML", 170.0),
            axon("MR", 170.0),
        ],
    });
    l.regions.push(RegionSpec {
        tag: "VR",
        center: [45.0, -90.0, 0.0],
        radius: 18.0,
        count: 60,
        inhibitory: false,
        secrete: None,
        axons: vec![
            axon("CORTEX_E", 130.0),
            axon("ML", 170.0),
            axon("MR", 170.0),
        ],
    });
    // Кора — центральный хаб между сенсорами и моторами: рекуррентность
    // на себя (аксон на собственный морфоген), тормозной контур через
    // CORTEX_I, выход на моторы. Радиус 42 ≈ масштаб стадии B (26) × ∛(400/90).
    // Самоаксон КОРОТКИЙ (30 мкм): с полным охватом (60 мкм) морфоген
    // собирал бы все 400×400 пар (154k синапсов — квадратичный фан-аут),
    // и CPU-путь не тянул бы ни одной особи. 30 мкм при шаге сомы ~9 мкм
    // даёт разреженную рекуррентность (~15 соседей на нейрон) — хаб
    // остаётся живой петлёй, но цена синапсов линейная.
    l.regions.push(RegionSpec {
        tag: "CORTEX_E",
        center: [0.0, 0.0, 0.0],
        radius: 42.0,
        count: 400,
        inhibitory: false,
        secrete: Some("CORTEX_E".into()),
        axons: vec![
            axon("CORTEX_E", 30.0),
            axon("CORTEX_I", 80.0),
            axon("ML", 150.0),
            axon("MR", 150.0),
        ],
    });
    l.regions.push(RegionSpec {
        tag: "CORTEX_I",
        center: [0.0, -40.0, 30.0],
        radius: 22.0,
        count: 120,
        inhibitory: true,
        secrete: None,
        axons: vec![
            axon("CORTEX_E", 80.0),
            axon("ML", 120.0),
            axon("MR", 120.0),
        ],
    });
    l.allowed_pairs.extend(
        [
            ("VL", "VL"),
            ("VL", "CORTEX_E"),
            ("VL", "ML"),
            ("VL", "MR"),
            ("VR", "VR"),
            ("VR", "CORTEX_E"),
            ("VR", "ML"),
            ("VR", "MR"),
            ("CORTEX_E", "CORTEX_E"),
            ("CORTEX_E", "CORTEX_I"),
            ("CORTEX_E", "I"),
            ("CORTEX_E", "ML"),
            ("CORTEX_E", "MR"),
            ("CORTEX_I", "CORTEX_E"),
            ("CORTEX_I", "ML"),
            ("CORTEX_I", "MR"),
            ("CORTEX_I", "I"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string())),
    );
    l
}

/// Когнитивный агент: + WS_L/WS_R (workspace) и WM_L/WM_R (память).
/// Регионы: 0=SL 1=SR 2=I 3=ML 4=MR 5=WSL 6=WSR 7=WML 8=WMR.
pub fn cognitive_layout() -> Layout {
    Layout {
        regions: vec![
            RegionSpec {
                tag: "SL",
                center: [-60.0, -60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: None,
                axons: vec![axon("ML", 170.0), axon("ML", 170.0), axon("WSL", 60.0)],
            },
            RegionSpec {
                tag: "SR",
                center: [60.0, -60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: None,
                axons: vec![axon("MR", 170.0), axon("MR", 170.0), axon("WSR", 60.0)],
            },
            RegionSpec {
                tag: "I",
                center: [0.0, -30.0, 0.0],
                radius: 12.0,
                count: 16,
                inhibitory: true,
                secrete: None,
                axons: vec![axon("ML", 140.0), axon("MR", 140.0)],
            },
            RegionSpec {
                tag: "ML",
                center: [-60.0, 60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: Some("ML".into()),
                axons: vec![],
            },
            RegionSpec {
                tag: "MR",
                center: [60.0, 60.0, 0.0],
                radius: 15.0,
                count: 20,
                inhibitory: false,
                secrete: Some("MR".into()),
                axons: vec![],
            },
            RegionSpec {
                tag: "WSL",
                center: [-60.0, -30.0, 0.0],
                radius: 10.0,
                count: 16,
                inhibitory: false,
                secrete: Some("WSL".into()),
                axons: vec![axon("WML", 90.0), axon("ML", 170.0)],
            },
            RegionSpec {
                tag: "WSR",
                center: [60.0, -30.0, 0.0],
                radius: 10.0,
                count: 16,
                inhibitory: false,
                secrete: Some("WSR".into()),
                axons: vec![axon("WMR", 90.0), axon("MR", 170.0)],
            },
            RegionSpec {
                tag: "WML",
                center: [-100.0, 30.0, 0.0],
                radius: 12.0,
                count: 20,
                inhibitory: false,
                secrete: Some("WML".into()),
                axons: vec![axon("WML", 40.0), axon("ML", 170.0)],
            },
            RegionSpec {
                tag: "WMR",
                center: [100.0, 30.0, 0.0],
                radius: 12.0,
                count: 20,
                inhibitory: false,
                secrete: Some("WMR".into()),
                axons: vec![axon("WMR", 40.0), axon("MR", 170.0)],
            },
        ],
        allowed_pairs: vec![
            ("SL".into(), "SL".into()),
            ("SL".into(), "ML".into()),
            ("SL".into(), "WSL".into()),
            ("SR".into(), "SR".into()),
            ("SR".into(), "MR".into()),
            ("SR".into(), "WSR".into()),
            ("I".into(), "ML".into()),
            ("I".into(), "MR".into()),
            ("I".into(), "WSL".into()),
            ("I".into(), "WSR".into()),
            ("WSL".into(), "WML".into()),
            ("WSL".into(), "ML".into()),
            ("WSL".into(), "WSL".into()),
            ("WSR".into(), "WMR".into()),
            ("WSR".into(), "MR".into()),
            ("WSR".into(), "WSR".into()),
            ("WML".into(), "WML".into()),
            ("WML".into(), "ML".into()),
            ("WML".into(), "WSL".into()),
            ("WMR".into(), "WMR".into()),
            ("WMR".into(), "MR".into()),
            ("WMR".into(), "WSR".into()),
        ],
    }
}
