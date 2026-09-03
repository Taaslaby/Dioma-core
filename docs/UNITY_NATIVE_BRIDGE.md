# Unity native bridge — большой headless бекенд

`dioma` — `cdylib` (`target/release/dioma.dll`) + `rlib`. Весь коннектом, веса,
STDP-следы и GPU-буферы живут в Rust; Unity — только сенсоры → действие.
Рендера в Rust нет (голова Unity).

## Сборка (портабельно, без native)

```powershell
cargo build --release --lib              # → target/release/dioma.dll
# для бенчей: RUSTFLAGS="-C target-cpu=native" cargo build --release --lib
```

Положи `dioma.dll` в `Assets/Plugins/x86_64/` и проверь версию до первого `create`:

```csharp
Dioma.EnsureAbi(expected: 2);
using var c = DiomaCreature.Create(seed: 42, backend: DiomaBackend.Cpu);
```

## C# фасад

Готовый идиоматичный фасад — `docs/DiomaUnity.cs` (скопируй в `Assets/Dioma/Runtime/`).
`DiomaCreature : IDisposable` владеет хэндлом, `Tick(in Sense)` не аллоцирует,
ошибки — `DiomaException` с `LastError`, батч — `Dioma.TickBatch`.

```csharp
using var creature = DiomaCreature.Create(42);
var sense = DiomaSense.FromWorld(smellL, smellR, visL, visR, wallL, wallR,
                                 foodNear, wallHit, place, heading, seeEat);
DiomaAction action = creature.Tick(sense); // steer/drive/eat
DiomaDiagnostics diag = creature.Diagnostics; // age, hunger, glucose, spikes...
DiomaBrainStats stats = creature.BrainStats; // neuron/synapse counts
byte[] dna = creature.GetDnaBytes();         // DNA1 — сохрани в сейв
using var child = creature.SpawnChild(childSeed: 99); // детерминирован
```

## C ABI (v2, совместим с v1)

```c
uint32_t dioma_abi_version(void); // 2
uint32_t dioma_unity_abi_version(void); // алиас v1
size_t   dioma_last_error(uint8_t* buf, size_t len);
void     dioma_last_error_clear(void);
void     dioma_buffer_free(uint8_t* ptr, size_t len);

// Жизнь
void* dioma_creature_create(uint64_t seed, uint32_t backend);
void  dioma_creature_free(void* h); // алиас destroy
int   dioma_creature_tick(void* h, const DiomaSense* s, DiomaAction* a);
int   dioma_creatures_tick_batch(void** hs, const DiomaSense* ss, DiomaAction* as, size_t n);

// Телеметрия (чтение, не мутация)
int dioma_creature_diagnostics(void* h, DiomaDiagnostics* out);
int dioma_creature_brain_stats(void* h, DiomaBrainStats* out);
int dioma_creature_blood(void* h, float* conc9, size_t clen, float* loci9, size_t llen);
int dioma_creature_membrane(void* h, float* buf, size_t len); // len==0 → запрос длины

// Наследие
void* dioma_creature_spawn_child(void* parent, uint64_t childSeed);
void* dioma_creature_spawn_child_pair(void* a, void* b, uint64_t childSeed);
int   dioma_creature_dna_bytes(void* h, uint8_t** outPtr, size_t* outLen);
void* dioma_creature_create_from_dna(const uint8_t* dna, size_t len, uint64_t seed);
int   dioma_creature_seed(void* h, uint64_t* out);

// Совместимость v1
void* dioma_unity_brain_create(uint64_t seed, uint32_t backend);
int   dioma_unity_brain_tick(void* h, const DiomaSense* s, DiomaAction* a);
void  dioma_unity_brain_destroy(void* h);
```

Коды: `0 OK`, `1 NULL_ARGUMENT`, `2 PANIC`, `3 INVALID_ARGUMENT`, `4 BUFFER_TOO_SMALL`.
`*_bytes` выдаёт буфер, который Unity освобождает `dioma_buffer_free(ptr,len)`.
`backend`: `0 Cpu` (детерминирован, дефолт), `1 Gpu` динамика, `2 GpuGrown` — оба с фолбэком на CPU.

**Правила владения:** один хэндл — один владелец, не тикать параллельно из разных потоков
(для стаи — `tick_batch`). `spawn_child` — чистая функция от сидов (Болдуин + дрейф весов σ0.05).
