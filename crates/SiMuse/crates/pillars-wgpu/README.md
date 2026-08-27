# pillars-wgpu — WebGPU compute backend для SNN

Порт `pillars-cuda` на wgpu (Vulkan / Metal / DX12 / WebGPU): тот же
CSR-коннектом + кольцевой буфер задержек, но без привязки к NVIDIA.

## Статус: рабочий движок, проверен бит-в-бит

| Гейт | Что проверяет |
|---|---|
| `wgpu_smoke` | Прямой LIF-шаг: 200 шагов × 64 нейрона, спайки 697 = 697 |
| `wgsl_compile_check` | Все 13 автоген-шейдеров `#[dual_neuron]` компилируются |
| `network_parity` | Полная сеть: 2 популяции, кольцо+задержки, STDP, инъекция, states, reset |
| `cargo test -p pillars-wgpu` | `gpu_network_parity` — тот же parity + dopamine-обучение как регресс-гейт (без GPU — skip) |

## API

```rust
// популяция: автоген-шейдер #[dual_neuron] (SoA: states[ns*stride], params[np*stride])
let pop = WgpuPop {
    wgsl: LIF_WGSL,          // pillars::zoo::neurons::LIF_WGSL_SOURCE
    entry: "lif_kernel_wgsl",// имя кернела в шейдере
    n: 64, ns: 2, np: 6,
    states: vec![...], params: vec![...],
};

let mut net = WgpuNetwork::new(ctx, &[pop], &row_ptr, &col_idx, &weights, &delays, dt)?;

net.step()?;                       // copy → pops → zero → scatter (как cuda launch_network_step)
net.inject_current(0, &[i], &[v])?; // сенсорный ввод (приходит на след. шаге)
net.stdp_step(decay, lr_ltp, lr_ltd, w_min, w_max)?; // trace + веса, бит-в-бит cuda
// R-STDP (dopamine-обучение, бит-в-бит pillars::api::learn::brain):
net.eligibility_step(d_pre, d_post, d_e, a_ltp, a_ltd)?; // traces + per-synapse elig
net.reward(lr, dopamine, w_min, w_cap)?;                  // Δw = lr·dopamine·elig, клэмп
net.download_spikes(&mut out)?;    // флаги спайков (n_total)
net.download_states(0, &mut st)?;  // состояния популяции (ns*stride)
net.weights()?; net.set_weights(&w)?; // чтение/запись весов (наследственность, Болдуин)
net.reset(&pops)?;                 // сброс к начальным состояниям + очистка ring/trace
net.n_pops(); net.pop_n(i); net.n_total(); net.step_counter();
```

Семантика каждого метода — 1:1 с `pillars-cuda::CompiledNetwork`
(`step`, `stdp_step`, `inject_current`, `download_states`, `reset`, `upload/download_weights`).

## Встраивание в игру (диома)

Один инстанс сети = **батч мозгов**: каждая популяция — группа существ одного
типа нейрона, `i_in`/`spikes` — offset-окна (как `offset*4` в CUDA).
- сенсоры → `inject_current(pop, indices, values)`
- действия ← спайки выходной популяции (`download_spikes` + декод)
- обучение → `stdp_step` (или `set_weights` для Болдуина/рождения)
- новое поколение → `reset(&pops)` (веса при этом сохраняются — только для
  полного сброса; для ребёнка пишем веса через `set_weights`)

Масштаб (RTX 4060 Ti, release): 120k нейронов ≈ 1000 существ — **13–18× быстрее CPU**.

## Гетерогенность и тихие эпизоды

- **Гетерогенные существа** = разные популяции: каждая своя модель/размер/параметры,
  `i_in`/`spikes` — offset-окна. Существа одной топологии (общий геном-коннектом)
  батчатся в **одну популяцию**; гетерогенность весов — через `set_weights`.
- **Действия** ← спайки выходного региона: `download_spikes` даёт весь `n_total`,
  диома фильтрует по `region_bounds`.
- **R-STDP бит-в-бит** с портом pillars (`eligibility_step`/`reward`):
  - traces с двумя τ, per-synapse eligibility,
  - `reward` пропускает `Δw == 0` (врождённые пути не трогаются),
  - decay тихих шагов точный (нет «висящих» следов между наградами).

## Заметки по реализации

- WGSL-кернелы автогенерятся макросом `#[dual_neuron]` как `*_WGSL_SOURCE`
  (транспайлер CUDA-тела: `f32`-декларации, `select()` вместо тернарника,
  `f`-суффиксы, `(*S)[i]` на указателях).
- Шейдеры-помощники: `shaders/csr_scatter.wgsl` (copy/zero/scatter/inject),
  `shaders/dopamine.wgsl` (trace_update/csr_stdp).
- naga выбрасывает неиспользуемые биндинги из авто-лейаута — не объявляй
  лишние storage-переменные в кернеле.
- Storage-буферы для offset-окон выравниваются по 256 байт (NVIDIA).
