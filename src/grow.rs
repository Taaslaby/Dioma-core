//! -------------------------------------------------------------------------
//! Filename:    src/grow.rs
//! Class:       bioEngine_dna
//! Purpose:     universal connectome growth
//! Description: Layout (regions + morphogen secretion + axon specs + tag recognition)
//!
//! → prometheus-grow world. Maternal gaussian bumps guide cones from step 0;
//! tag table sets partner specificity. Presets: reflex / agent / cognitive.
//! author: Atlasov anton
//! -------------------------------------------------------------------------
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
    pub fn n(&self) -> usize {
        self.regions.iter().map(|r| r.count).sum()
    }

    pub fn region_of(&self) -> Vec<usize> {
        let mut v = Vec::with_capacity(self.n());
        for (ri, r) in self.regions.iter().enumerate() {
            v.extend(std::iter::repeat_n(ri, r.count));
        }
        v
    }
}

/// Рулевые параметры роста (мутабельная часть генома).
#[derive(Clone, Debug)]
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
}

/// Вырастить коннектом из layout.
///
/// Границы мира выводятся из геометрии регионов (с запасом на гауссианы).
pub fn grow(layout: &Layout, st: &Steering, seed: u64) -> Grown {
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

    let t0 = std::time::Instant::now();
    for _ in 0..st.grow_steps {
        world.step();
    }

    // Узнавание: таблица допустимых пар тегов.
    let mut allowed = HashMap::new();
    let mut prob = HashMap::new();
    for (a, b) in &layout.allowed_pairs {
        allowed.insert((a.clone(), b.clone()), true);
        prob.insert((a.clone(), b.clone()), 1.0);
    }
    let rec = TagRecognition { allowed, prob };
    world.form_synapses_tagged(st.synapse_dist, &rec);
    let growth_s = t0.elapsed().as_secs_f64();

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
