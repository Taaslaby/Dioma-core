// csr_scatter.wgsl — детерминированный per-post spike scatter + ring buffer.
//
// Скеттер идёт ПО ПОСТ-НЕЙРОНУ по in-edge CSR (rev_ptr/rev_src/rev_w/rev_d):
// каждый поток аккумулирует ЛОКАЛЬНО вклады всех заспайкавших источников,
// затем один раз пишет ring. Это устраняет гонку пре→пост версии, где
// несколько потоков (по пре) делали `ring[idx] += w` в одну ячейку — на GPU
// потеря вкладов (RAM-race) до ~90%.
//
// Порядок аккумуляции — по возрастанию источника (rev-массивы построены
// обходом src по возрастанию), что бит-в-бит совпадает с порядком
// `for pre in alpha_fired { for k in row(pre) { slot[post] += ... } }`
// в CPU `run_lif_alpha` (alpha_fired тоже по возрастанию).

struct ScatterArgs { n: u32, slots: u32, cur_slot: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read> rev_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> rev_src: array<u32>;
@group(0) @binding(2) var<storage, read> rev_edge: array<u32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read> rev_d: array<u32>;
@group(0) @binding(5) var<storage, read> spikes: array<u32>;
@group(0) @binding(6) var<storage, read_write> ring: array<f32>;
@group(0) @binding(7) var<uniform> args: ScatterArgs;
@group(0) @binding(8) var<storage, read> signs: array<f32>;

@compute @workgroup_size(64)
fn scatter_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let post = gid.x;
    if (post >= args.n) { return; }
    // zero-фаза влита в скеттер: поток поста обнуляет свою ячейку текущего
    // слота (отдельный zero-диспатч экономит лаунч; гонки нет — каждая
    // ячейка ring[cur_slot] трогается ровно одним потоком).
    ring[args.cur_slot * args.n + post] = 0.0;
    let start = rev_ptr[post];
    let end = rev_ptr[post + 1u];
    // Бакеты по аксональной задержке: обычно один (задержки одинаковы),
    // но общий случай корректен. Инициализация нулями — по спецификации
    // WGSL (var без инициализатора).
    var acc: array<f32, 32>;
    for (var k = start; k < end; k = k + 1u) {
        let src = rev_src[k];
        if (spikes[src] == 0u) { continue; }
        var d = rev_d[k];
        if (d < 1u) { d = 1u; }
        // Знак торможения пресинапса (nt): как в run_lif_alpha, где
        // `slot[post] += vals[k] * sign` — иначе тормозные синапсы (веса
        // положительные, знак в nt) возбуждают пост-нейрон.
        // Вес читаем из ЖИВОГО weights_b через ремап in-edge → out-edge
        // (rev_edge): set_weights синхронизирует его на каждом шаге — никакой
        // мёртвой копии. ЯВНЫЙ fma — зеркало CPU-скеттера `vals.mul_add(sign,
        // slot)`: компилятор шейдера волен сворачивать mul+add в fma, а rust
        // без fast-math — нет → 1 ulp дрейфа i_in → ложные спайки у порога.
        acc[d % 32u] = fma(weights[rev_edge[k]], signs[src], acc[d % 32u]);
    }
    for (var d = 0u; d < 32u; d = d + 1u) {
        if (acc[d] != 0.0) {
            let tgt = (args.cur_slot + d) % args.slots;
            ring[tgt * args.n + post] += acc[d];
        }
    }
}

@group(0) @binding(0) var<storage, read> ring_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_in: array<f32>;
@group(0) @binding(2) var<uniform> copyArgs: ScatterArgs;

@compute @workgroup_size(64)
fn copy_slot_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= copyArgs.n) { return; }
    i_in[i] = ring_in[copyArgs.cur_slot * copyArgs.n + i];
}

// zero_slot_kernel пишет ring — свой биндинг с read_write (layout отличается от copy).
// Внимание: неиспользуемые биндинги naga выбрасывает из авто-лейаута — только ring + args.
@group(0) @binding(0) var<storage, read_write> ring_zero: array<f32>;
@group(0) @binding(1) var<uniform> zeroArgs: ScatterArgs;

@compute @workgroup_size(64)
fn zero_slot_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= zeroArgs.n) { return; }
    ring_zero[zeroArgs.cur_slot * zeroArgs.n + i] = 0.0;
}

// ── inject: сенсорный ввод — ring[cur_slot * n + idx] += value (прибытие на след. шаге) ──
struct InjectArgs { n: u32, count: u32, cur_slot: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> ring_inj: array<f32>;
@group(0) @binding(1) var<storage, read> inj_idx: array<u32>;
@group(0) @binding(2) var<storage, read> inj_val: array<f32>;
@group(0) @binding(3) var<uniform> iargs: InjectArgs;

@compute @workgroup_size(64)
fn inject_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= iargs.count) { return; }
    let idx = inj_idx[i];
    ring_inj[iargs.cur_slot * iargs.n + idx] += inj_val[i];
}

// ── v_jump: внешний ток как СКАЧОК НАПРЯЖЕНИЯ (v += value) — семантика
//    run_lif_alpha (external_I прямо в v), а не ring→i_in проводимости.
//    Индексы — ЛОКАЛЬНЫЕ для популяции (без i_slice): states буфер pop-локальный.
struct VJumpArgs { n: u32, count: u32, stride: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> vstates: array<f32>;
@group(0) @binding(1) var<storage, read> vj_idx: array<u32>;
@group(0) @binding(2) var<storage, read> vj_val: array<f32>;
@group(0) @binding(3) var<uniform> vjargs: VJumpArgs;

@compute @workgroup_size(64)
fn v_jump_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if (k >= vjargs.count) { return; }
    let i = vj_idx[k];
    if (i >= vjargs.n) { return; }
    vstates[0 * vjargs.stride + i] += vj_val[k];
}
