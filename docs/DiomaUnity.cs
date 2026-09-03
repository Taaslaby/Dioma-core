// DiomaUnity.cs — красивый, идиоматичный C# фасад над dioma.dll
// Клади в Unity: Assets/Dioma/Runtime/DiomaUnity.cs (asmdef без зависимостей)
// Требует: dioma.dll рядом (Plugins/x86_64/dioma.dll), .NET Standard 2.1
//
// Философия фасада:
// • `DiomaCreature : IDisposable` — владеет хэндлом, `using var c = DiomaCreature.Create(42)`.
//   Никаких `IntPtr` в геймплее — только `Sense` → `Action`.
// • `Sense`/`Action` — `struct` без аллокаций, `Tick` принимает `in`/`out`.
// • `Diagnostics`/`BrainStats` — снимок за тик для HUD/голограммы (чтение, не мутация).
// • `SpawnChild` — детерминизм: один сид — один потомок.
// • `Dna` — побайтовый `DNA1`, round-trip через `GetDnaBytes()` + `CreateFromDna()`.
// • Ошибки — исключения `DiomaException` с `LastError`, а не коды в геймплее.
// • Батч — `Dioma.TickBatch(creatures, senses, actions)` — один переход вместо N.

using System;
using System.Runtime.InteropServices;

namespace Dioma
{
    // ── Низкоуровневые типы (layout = Rust repr(C)) ─────────────────────────

    public enum DiomaBackend : uint { Cpu = 0, Gpu = 1, GpuGrown = 2 }

    [StructLayout(LayoutKind.Sequential)]
    public struct DiomaSense
    {
        public float smell_l, smell_r;
        public float vision_l, vision_r;
        public float wall_l, wall_r;
        public float heading;
        public uint place;
        public byte food_near, wall, see_eat, _pad;
        public static DiomaSense Default => new DiomaSense();

        // Хелперы для Unity-мира: заполни из NaturalFoodField / RayVisionSensor
        public static DiomaSense FromWorld(float smellL, float smellR, float visL, float visR,
            float wallL, float wallR, bool foodNear, bool wallHit, uint place, float heading, bool seeEat)
            => new DiomaSense
            {
                smell_l = smellL, smell_r = smellR,
                vision_l = visL, vision_r = visR,
                wall_l = wallL, wall_r = wallR,
                heading = heading, place = place,
                food_near = (byte)(foodNear ? 1 : 0), wall = (byte)(wallHit ? 1 : 0),
                see_eat = (byte)(seeEat ? 1 : 0)
            };
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DiomaAction
    {
        public float steer; // -1..1
        public float drive; // 0..1
        public byte eat;    // 0/1
        private byte _p0, _p1, _p2;
        public bool WantsEat => eat != 0;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DiomaDiagnostics
    {
        public ulong age;
        public float damage;
        public uint food_eaten;
        public byte is_sleeping, is_dying, is_dead, _pad0;
        public float glucose, hunger, fatigue, pain, cortisol, serotonin, acetylcholine;
        public float sensor_gain, motor_tonic, speed, plasticity_budget;
        public uint spikes_last_tick;
        public bool IsSleeping => is_sleeping != 0;
        public bool IsDying => is_dying != 0;
        public bool IsDead => is_dead != 0;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DiomaBrainStats
    {
        public uint neuron_count, region_count, synapse_count, steps_per_tick;
        public float inhibition_ratio;
    }

    // ── Исключение ───────────────────────────────────────────────────────────

    public sealed class DiomaException : Exception
    {
        public int Code { get; }
        public DiomaException(int code, string msg) : base($"dioma [{code}]: {msg}") { Code = code; }
    }

    // ── Нативный импорт (один DllImport на функцию) ─────────────────────────

    internal static class Native
    {
        const string Dll = "dioma";
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern uint dioma_abi_version();
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern UIntPtr dioma_last_error(byte[] buf, UIntPtr len);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern void dioma_last_error_clear();
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern void dioma_buffer_free(IntPtr ptr, UIntPtr len);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr dioma_creature_create(ulong seed, uint backend);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern void dioma_creature_free(IntPtr h);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int dioma_creature_tick(IntPtr h, in DiomaSense sense, out DiomaAction action);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int dioma_creatures_tick_batch(IntPtr[] handles, DiomaSense[] senses, DiomaAction[] actions, UIntPtr count);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int dioma_creature_diagnostics(IntPtr h, out DiomaDiagnostics diag);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int dioma_creature_brain_stats(IntPtr h, out DiomaBrainStats stats);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int dioma_creature_blood(IntPtr h, [Out] float[] conc9, UIntPtr concLen, [Out] float[] loci9, UIntPtr lociLen);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int dioma_creature_membrane(IntPtr h, [Out] float[] buf, UIntPtr len);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr dioma_creature_spawn_child(IntPtr parent, ulong childSeed);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr dioma_creature_spawn_child_pair(IntPtr a, IntPtr b, ulong childSeed);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int dioma_creature_dna_bytes(IntPtr h, out IntPtr outPtr, out UIntPtr outLen);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr dioma_creature_create_from_dna(IntPtr dnaPtr, UIntPtr dnaLen, ulong seed);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int dioma_creature_seed(IntPtr h, out ulong seed);

        // Совместимость v1
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern uint dioma_unity_abi_version();
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern IntPtr dioma_unity_brain_create(ulong seed, uint backend);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int dioma_unity_brain_tick(IntPtr h, in DiomaSense s, out DiomaAction a);
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern void dioma_unity_brain_destroy(IntPtr h);
    }

    // ── Удобный фасад ────────────────────────────────────────────────────────

    public static class Dioma
    {
        public const int OK = 0, NullArg = 1, Panic = 2, InvalidArg = 3, BufferTooSmall = 4;

        public static uint AbiVersion => Native.dioma_abi_version();

        public static void EnsureAbi(uint expected = 2)
        {
            var v = AbiVersion;
            if (v != expected) throw new DiomaException(-1, $"ABI mismatch: dll={v} expected={expected}");
        }

        public static string LastError()
        {
            var len = Native.dioma_last_error(null, UIntPtr.Zero);
            if (len == UIntPtr.Zero) return string.Empty;
            var buf = new byte[(int)len];
            Native.dioma_last_error(buf, len);
            return System.Text.Encoding.UTF8.GetString(buf);
        }

        static void Check(int code)
        {
            if (code == OK) return;
            var msg = LastError();
            if (string.IsNullOrEmpty(msg)) msg = code == NullArg ? "null argument" : code == BufferTooSmall ? "buffer too small" : "panic";
            throw new DiomaException(code, msg);
        }

        // Батч — один переход для стаи (детерминирован, быстрее N одиночных тиков)
        public static void TickBatch(DiomaCreature[] creatures, DiomaSense[] senses, DiomaAction[] actions)
        {
            if (creatures.Length != senses.Length || senses.Length != actions.Length)
                throw new ArgumentException("TickBatch: lengths must match");
            var handles = new IntPtr[creatures.Length];
            for (int i = 0; i < handles.Length; i++) handles[i] = creatures[i].DangerousHandle;
            Check(Native.dioma_creatures_tick_batch(handles, senses, actions, (UIntPtr)handles.Length));
        }
    }

    // ── Владение существом ───────────────────────────────────────────────────

    public sealed class DiomaCreature : IDisposable
    {
        IntPtr _h;
        bool _owned;
        public IntPtr DangerousHandle => _h;
        public bool IsAlive => _h != IntPtr.Zero;

        DiomaCreature(IntPtr h, bool owned) { _h = h; _owned = owned; }

        public static DiomaCreature Create(ulong seed, DiomaBackend backend = DiomaBackend.Cpu)
        {
            var h = Native.dioma_creature_create(seed, (uint)backend);
            if (h == IntPtr.Zero) throw new DiomaException(Dioma.Panic, Dioma.LastError());
            return new DiomaCreature(h, true);
        }

        public static DiomaCreature CreateFromDna(byte[] dna, ulong seed)
        {
            var p = Marshal.AllocHGlobal(dna.Length);
            try
            {
                Marshal.Copy(dna, 0, p, dna.Length);
                var h = Native.dioma_creature_create_from_dna(p, (UIntPtr)dna.Length, seed);
                if (h == IntPtr.Zero) throw new DiomaException(Dioma.Panic, Dioma.LastError());
                return new DiomaCreature(h, true);
            }
            finally { Marshal.FreeHGlobal(p); }
        }

        // Не владеет — для `TickBatch` без копий (редко нужно)
        internal static DiomaCreature Attach(IntPtr h) => new DiomaCreature(h, false);

        public DiomaAction Tick(in DiomaSense sense)
        {
            if (_h == IntPtr.Zero) throw new ObjectDisposedException(nameof(DiomaCreature));
            var code = Native.dioma_creature_tick(_h, in sense, out var a);
            if (code != Dioma.OK) throw new DiomaException(code, Dioma.LastError());
            return a;
        }

        public DiomaDiagnostics Diagnostics
        {
            get
            {
                if (_h == IntPtr.Zero) throw new ObjectDisposedException(nameof(DiomaCreature));
                var code = Native.dioma_creature_diagnostics(_h, out var d);
                if (code != Dioma.OK) throw new DiomaException(code, Dioma.LastError());
                return d;
            }
        }

        public DiomaBrainStats BrainStats
        {
            get
            {
                if (_h == IntPtr.Zero) throw new ObjectDisposedException(nameof(DiomaCreature));
                var code = Native.dioma_creature_brain_stats(_h, out var s);
                if (code != Dioma.OK) throw new DiomaException(code, Dioma.LastError());
                return s;
            }
        }

        public void GetBlood(Span<float> conc9, Span<float> loci9)
        {
            if (_h == IntPtr.Zero) throw new ObjectDisposedException(nameof(DiomaCreature));
            if (conc9.Length < 9 || loci9.Length < 9) throw new ArgumentException("GetBlood: need 9");
            var ca = conc9.ToArray(); var la = loci9.ToArray();
            var code = Native.dioma_creature_blood(_h, ca, (UIntPtr)9, la, (UIntPtr)9);
            if (code != Dioma.OK) throw new DiomaException(code, Dioma.LastError());
            ca.CopyTo(conc9); la.CopyTo(loci9);
        }

        public float[] GetMembrane()
        {
            if (_h == IntPtr.Zero) throw new ObjectDisposedException(nameof(DiomaCreature));
            var stats = BrainStats;
            var buf = new float[stats.neuron_count];
            var code = Native.dioma_creature_membrane(_h, buf, (UIntPtr)buf.Length);
            if (code < 0) throw new DiomaException(code, Dioma.LastError());
            return buf;
        }

        public ulong Seed
        {
            get
            {
                Native.dioma_creature_seed(_h, out var s);
                return s;
            }
        }

        public DiomaCreature SpawnChild(ulong childSeed)
        {
            var h = Native.dioma_creature_spawn_child(_h, childSeed);
            if (h == IntPtr.Zero) throw new DiomaException(Dioma.Panic, Dioma.LastError());
            return new DiomaCreature(h, true);
        }

        public DiomaCreature SpawnChildWith(DiomaCreature other, ulong childSeed)
        {
            var h = Native.dioma_creature_spawn_child_pair(_h, other._h, childSeed);
            if (h == IntPtr.Zero) throw new DiomaException(Dioma.Panic, Dioma.LastError());
            return new DiomaCreature(h, true);
        }

        public byte[] GetDnaBytes()
        {
            var code = Native.dioma_creature_dna_bytes(_h, out var ptr, out var len);
            if (code != Dioma.OK) throw new DiomaException(code, Dioma.LastError());
            var bytes = new byte[(int)len];
            Marshal.Copy(ptr, bytes, 0, (int)len);
            Native.dioma_buffer_free(ptr, len);
            return bytes;
        }

        public void Dispose()
        {
            if (_h != IntPtr.Zero && _owned)
            {
                Native.dioma_creature_free(_h);
                _h = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }
        ~DiomaCreature() { if (_h != IntPtr.Zero && _owned) Native.dioma_creature_free(_h); }
    }
}
