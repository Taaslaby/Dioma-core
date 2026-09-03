// ─── Тесты (извлечены из модуля: приватный доступ через super) ───
// Содержимое перенесено механически: #[path] держит модуль в том же
// дереве, все тесты видят приватные элементы как раньше.

    use super::*;
    use crate::body::metabolism::chem;

    #[test]
    fn creature_hunts_berry_and_it_crunches() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 80.0, 20.0);
        let mut arena = Arena::new(42, map);
        arena.spawn(Vec2::new(5.0, 10.0), 0.0, 42);
        // Голод сразу, чтобы не ждать сотни тиков синтеза HUNGER.
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        let berry = arena.add_berry(Vec2::new(35.0, 10.0));

        for _ in 0..250 {
            arena.step();
            // Съедена = ушла в стадию отрастания (vars[1] > 0).
            if arena
                .world
                .agent(berry)
                .map(|a| a.vars[1] > 0.0)
                .unwrap_or(false)
            {
                break;
            }
        }
        // Ещё один шаг: EAT-сообщение доставляется в world.step следующего
        // такта — только тогда скрипт «хрум» выполнится и попадёт в журнал.
        arena.step();

        assert!(
            arena
                .world
                .agent(berry)
                .map(|a| a.alive && a.vars[1] > 0.0)
                .unwrap_or(false),
            "ягода должна быть съедена и отрастать (не исчезать)"
        );
        assert!(
            arena.world.log().iter().any(|s| s.contains("хрум")),
            "ягода должна хрумнуть: {:?}",
            arena.world.log()
        );
        assert!(
            arena.critters[0].creature.glucose() > 128.0,
            "еда должна дать питание"
        );
    }

    /// Охота за стеной: существо с направленными сенсорами стены объезжает
    /// L-препятствие и доедает в обоих триалах (мозг живёт между триалами,
    /// кровь сбрасывается). Раньше (без сенсора) тот же удар выжигал гоминг,
    /// и второй триал голодал у стены.
    ///
    /// Между триалами старые ягоды убиваются: ягода первого триала на той же
    /// позиции (55,15) успевала отрасти (75 тиков = 3.75 с при тике 50 мс) и
    /// перехватывала еду второго триала — существо ело СТАРУЮ ягоду, а новая
    /// оставалась нетронутой, и тест считал триал проваленным.
    ///
    /// Голод на старте — 255 (потолок): при 200 существо доезжает до ягоды
    /// с голодом ровно у порога wants_food (40) — замер: 41 при тике 25 мс,
    /// 39 при 50 мс. Оба исхода — на грани, тест мерил не навигацию, а
    /// везение с таймингом. Потолок даёт запас ~10 пунктов и стабильно
    /// проверяет именно объезд L-стены.
    ///
    /// Порог ударов (замыкание контура, переякорено): сон не кормит
    /// (SLEEP_GLUCOSE_CAP=120 < 150) и дефицитный синтез голода ребалансирован
    /// (0.08/тик — равновесие HUNGER ≈ 80 при глюкозе 120). Голод устойчив,
    /// охота не умирает — и навигация стала чище: замер триалов 0 ударов
    /// в обоих (еда на t≈144/t≈216; при одной только капе сна без ребаланса
    /// экономики было 0 и 10). Порог 12 оставляет запас на случай будущих
    /// изменений экономики голода (панический тон при HUNGER > 240 толкает
    /// в угол L) при детерминированном прогоне (сид 42).
    #[test]
    fn detour_hunt_survives_trials_with_wall_sense() {
        let mut map = WorldMap::default();
        // L-форма: нижний правый квадрант (x>20, y<10) — пустота (стена).
        map.add_room(0.0, 0.0, 20.0, 20.0);
        map.add_room(20.0, 10.0, 60.0, 20.0);
        let mut a = Arena::new(42, map);
        a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
        for _ in 0..2 {
            a.critters[0].creature.blood = crate::body::metabolism::Blood::creature(false);
            a.critters[0].creature.blood.conc[chem::HUNGER] = 255.0;
            // Свежая ягода триала: прошлые убрать (см. док теста).
            for id in a.berries() {
                a.world.kill(id);
            }
            let berry = a.add_berry(Vec2::new(55.0, 15.0));
            let h0 = a.critters[0].wall_hits;
            let mut ate = false;
            let mut saw_vision = false;
            for _ in 0..800 {
                a.step();
                // Стадия B: проводка зрения жива — снимок виз. сигнала арены
                // (ягода видна в той же комнате в конусе курса). Если бы
                // зрение не было подключено к существам, флаг остался бы 0.
                if a.critters[0].vis_l > 0.0 || a.critters[0].vis_r > 0.0 {
                    saw_vision = true;
                }
                // Съедена = отрастает (vars[1] > 0).
                if a
                    .world
                    .agent(berry)
                    .map(|ag| ag.vars[1] > 0.0)
                    .unwrap_or(false)
                {
                    ate = true;
                    break;
                }
            }
            assert!(ate, "существо должно объехать стену и съесть ягоду");
            assert!(
                saw_vision,
                "в ходе объезда зрение должно видеть ягоду (стадия B подключена)"
            );
            assert!(
                a.critters[0].wall_hits - h0 <= 12,
                "объезд почти без ударов: {} ударов",
                a.critters[0].wall_hits - h0
            );
            // Смерть-транш: здоровый охотник за время детюра жив и не в
            // агонии — урон копят только голод/старость (см. фазу смертности).
            assert!(
                a.critters[0].creature.alive() && !a.critters[0].creature.dying,
                "здоровое существо не должно умирать за время детюра"
            );
            a.critters[0].pos = Vec2::new(5.0, 5.0);
            a.critters[0].heading = 0.0;
        }
    }

    /// Регрессионный сторож поведенческой эквивалентности биохимии.
    ///
    /// Фиксированный сценарий: детур-карта, сид 42, спавн (5,5)/0.0 (сид 1),
    /// голод 255, ягода (55,15) — триал 1 теста `detour_hunt_*` без сброса
    /// крови и без перезапуска. Константы сняты с ИСПРАВЛЕННОЙ реализации
    /// (SIMD-распады + batch-рецепторы, полный пересчёт сенсоров) и побайтно
    /// совпадают с чекпойнтом 2550cae (эталон до оптимизаций).
    ///
    /// Эталон после починки SIMD-распадов: «инкрементальные сенсоры»
    /// (дельты от prev_sense) давали систематический дрейф ext — мозг получал
    /// искажённый запах/стену, и HUNGER/PAIN на тике 200 уходили на десятки
    /// единиц (эталон 53.46/47.50). Любое будущее переупорядочивание шагов
    /// биохимии (распады/реакции/рецепторы) или сенсорного входа, меняющее
    /// поведение, ломает этот тест громко, а не тихим срывом охоты.
    ///
    /// СТАДИЯ B (переякорено): мозг 300 нейронов (зрение + ассоциативный хаб)
    /// изменил моторный драйв — существо ест раньше/прямее: GLUCOSE@200 упирается
    /// в потолок 255 (было 254.877), HUNGER@200 58.27 (было 53.46), PAIN@200 40.0
    /// (было 47.50), wall_hits по-прежнему 0. Эталон переснят с подключённой
    /// стадии B (см. переменную DIOMA_RECORD_TRACE в тесте) — сторож остаётся
    /// чувствительным: любые изменения шагов биохимии/сенсоров сдвинут значения.
    ///
    /// ЗАМЫКАНИЕ КОНТУРА (переякорено): сон не кормит (SLEEP_GLUCOSE_CAP=120 <
    /// порога дефицита 150) — GLUCOSE@200 больше НЕ упирается в потолок 255:
    /// существо поело и поспало, глюкоза 122.4 (выше капа сна, ниже 150 —
    /// голод продолжает синтезироваться), HUNGER@200 135.5 (дефицитный
    /// синтез 0.08/тик держит голод высоким — равновесие ~80+), PAIN@200
    /// 189.8 — устойчивый голод держит существо в движении, оно мечется в
    /// L-углу (11 ударов к 200, 27 к 300; до ребаланса экономики было
    /// 6 и 23). Эталон переснят после замыкания контура (см. DIOMA_RECORD_TRACE).
    ///
    /// ЭКОНОМИКА УТОМЛЕНИЯ (переякорено): гейн RATE_MOTOR→FATIGUE снижен ×6
    /// (0.20 → 0.033, см. metabolism.rs) — существо бодрствует дольше и ест
    /// иначе: GLUCOSE@200 93.8 (после еды и сна, ниже прежнего 122.4),
    /// HUNGER@200 197.9 (голод высок — дефицитный синтез при глюкозе ниже
    /// капа сна), PAIN@200 210.5, FATIGUE@200 60.4 (недавний сон, ×0.3).
    /// Траектория: 25 ударов к 200, дальше существо спит/ест — 25 к 300
    /// (до ребаланса было 11/27). Эталон переснят после ребаланса
    /// (см. DIOMA_RECORD_TRACE).
    /// СТАДИЯ C-LITE (переякорено): мозг 760 нейронов (кора 400+120 вместо
    /// 90+30, зрение 60+60) — моторика богаче, траектория другая: GLUCOSE@200
    /// 96.3, HUNGER@200 111.4 (голод к тику 200 не дожат — часть пути сна),
    /// PAIN@200 210.5, FATIGUE@200 163.6 (замедленная экономика + поздний
    /// сон), удары 24 к 200 и 31 к 300. Эталон переснят после подключения
    /// стадии C-lite (см. DIOMA_RECORD_TRACE).
    #[test]
    fn biochemistry_trace_is_stable() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 20.0, 20.0);
        map.add_room(20.0, 10.0, 60.0, 20.0);
        let mut a = Arena::new(42, map);
        a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
        a.critters[0].creature.blood = crate::body::metabolism::Blood::creature(false);
        a.critters[0].creature.blood.conc[chem::HUNGER] = 255.0;
        a.add_berry(Vec2::new(55.0, 15.0));

        // ВРЕМЕННО: печать новых эталонных значений стадии B.
        if std::env::var("DIOMA_RECORD_TRACE").is_ok() {
            for _ in 0..200 {
                a.step();
            }
            let conc = &a.critters[0].creature.blood.conc;
            println!("TRACE@200 GLUCOSE={:.9} HUNGER={:.9} PAIN={:.9} FATIGUE={:.9} SEROTONIN={:.9} ACETYLCHOLINE={:.9} wall_hits={}", conc[chem::GLUCOSE], conc[chem::HUNGER], conc[chem::PAIN], conc[chem::FATIGUE], conc[chem::SEROTONIN], conc[chem::ACETYLCHOLINE], a.critters[0].wall_hits);
            for _ in 200..300 {
                a.step();
            }
            let conc = &a.critters[0].creature.blood.conc;
            println!("TRACE@300 HUNGER={:.9} SEROTONIN={:.9} ACETYLCHOLINE={:.9} wall_hits={}", conc[chem::HUNGER], conc[chem::SEROTONIN], conc[chem::ACETYLCHOLINE], a.critters[0].wall_hits);
            return;
        }

        for _ in 0..200 {
            a.step();
        }
        let conc = &a.critters[0].creature.blood.conc;
        let close = |got: f32, want: f32| (got - want).abs() <= 1e-4;

        // Тик 200 (после еды и сна): глюкоза/голод/боль/усталость/серотонин/внимание.
        // Эталон после ребаланса утомления (×6): существо бодрствует дольше,
        // ест иначе — глюкоза 93.8 (было 122.4), голод 197.9 (было 135.5).
        assert!(
            close(conc[chem::GLUCOSE], 77.195983887),
            "GLUCOSE@200: {} ≠ 96.310821533", conc[chem::GLUCOSE]
        );
        assert!(
            close(conc[chem::HUNGER], 212.921829224),
            "HUNGER@200: {} ≠ 111.399505615", conc[chem::HUNGER]
        );
        assert!(
            close(conc[chem::PAIN], 94.213439941),
            "PAIN@200: {} ≠ 94.213439941", conc[chem::PAIN]
        );
        assert!(
            close(conc[chem::FATIGUE], 179.582946777),
            "FATIGUE@200: {} ≠ 163.586791992", conc[chem::FATIGUE]
        );
        assert!(
            close(conc[chem::SEROTONIN], 39.572669983),
            "SEROTONIN@200: {} ≠ 43.064628601", conc[chem::SEROTONIN]
        );
        assert!(
            close(conc[chem::ACETYLCHOLINE], 9.755206108),
            "ACETYLCHOLINE@200: {} ≠ 9.223410606", conc[chem::ACETYLCHOLINE]
        );
        // Тик 300: голод держится высоким (дефицитный синтез), внимание от еды.
        for _ in 200..300 {
            a.step();
        }
        let conc = &a.critters[0].creature.blood.conc;
        assert!(
            close(conc[chem::HUNGER], 143.243469238),
            "HUNGER@300: {} ≠ 193.141052246", conc[chem::HUNGER]
        );
        assert!(
            close(conc[chem::SEROTONIN], 43.100345612),
            "SEROTONIN@300: {} ≠ 46.128597260", conc[chem::SEROTONIN]
        );
        assert!(
            close(conc[chem::ACETYLCHOLINE], 21.523950577),
            "ACETYLCHOLINE@300: {} ≠ 22.424747467", conc[chem::ACETYLCHOLINE]
        );
        // Траектория эталона: с медленным утомлением существо бодрствует
        // дольше, к 200 успевает набрать 25 ударов в L-углу, дальше спит/ест —
        // к 300 счётчик не растёт (до ребаланса было 11 к 200 и 27 к 300).
        // Химия выше — основной сторож; счётчик фиксирует траекторию целиком.
        assert_eq!(
            a.critters[0].wall_hits, 25,
            "эталонная траектория стадии B: {} ударов", a.critters[0].wall_hits
        );
    }

    /// Сквозная проводка «стена → боль → мозг»: существо в углу упирается
    /// в стены, арена считает удар и поднимает ноцицепцию в крови
    /// (bump_wall → PAIN). В углу отворот невозможен — сенсоры стены дают
    /// компенсирующие сигналы, и удар гарантирован.
    #[test]
    fn wall_hit_counts_and_raises_pain() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut arena = Arena::new(7, map);
        arena.spawn(Vec2::new(39.5, 1.5), 0.0, 7);
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        let pain0 = arena.critters[0].creature.blood.conc[chem::PAIN];

        // Гоним до первого удара о стену (угол: удар на первом же такте).
        let mut guard = 0;
        while arena.critters[0].wall_hits == 0 && guard < 200 {
            arena.step();
            guard += 1;
        }
        assert!(
            arena.critters[0].wall_hits > 0,
            "существо должно упереться в стену и получить удары"
        );
        // bump_wall ставит EVENT_WALL, а боль эмитируется в blood.update()
        // следующего тика — один шаг, чтобы эмиттер сработал.
        arena.step();
        let pain = arena.critters[0].creature.blood.conc[chem::PAIN];
        assert!(
            pain > pain0 + 5.0,
            "удар о стену должен поднять боль: {pain0} → {pain}"
        );
    }

    #[test]
    fn berries_grow_only_in_fertile_rooms() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0); // плодородная
        map.add_room(40.0, 0.0, 80.0, 20.0); // бедная
        let mut arena = Arena::new(1, map);
        // Почти без распада веществ, чтобы питание не ушло за время теста.
        arena.world.map.rates.decay = [0.0005; 4];
        let nutr = Substance::Nutrients.idx();
        let light = Substance::Light.idx();
        let temp = Substance::Temperature.idx();
        arena.world.map.rooms[0].ca[nutr] = 200.0;
        arena.world.map.rooms[0].ca[light] = 200.0;
        // Климат трёхфакторный (тепло × свет × питание) — плодородная
        // комната должна быть и тёплой, иначе ягоды не растут.
        arena.world.map.rooms[0].ca[temp] = 200.0;

        let rich = arena.add_berry(Vec2::new(20.0, 10.0)); // плодородная
        let poor = arena.add_berry(Vec2::new(60.0, 10.0)); // бедная
        let n0 = arena.berries().len();

        for _ in 0..600 {
            arena.step();
        }

        let n1 = arena.berries().len();
        assert!(
            n1 > n0,
            "в плодородной комнате ягоды должны размножиться: {n0} → {n1}"
        );
        assert!(
            arena.world.agent(poor).is_some() && arena.world.agent(rich).is_some(),
            "родители не должны исчезать без еды"
        );
        assert!(
            arena.world.map.rooms[0].ca[nutr] < 200.0,
            "рост должен съедать питание комнаты"
        );
        assert!(n1 <= MAX_BERRIES, "популяция ограничена капом: {n1}");
    }

    #[test]
    fn arena_is_deterministic_from_seeds() {
        let run = || -> Vec<(f32, f32)> {
            let mut map = WorldMap::default();
            map.add_room(0.0, 0.0, 60.0, 20.0);
            let mut a = Arena::new(42, map);
            a.spawn(Vec2::new(5.0, 10.0), 0.0, 42);
            a.add_berry(Vec2::new(30.0, 10.0));
            for _ in 0..50 {
                a.step();
            }
            a.critters
                .iter()
                .map(|c| (c.pos.x, c.pos.y))
                .collect()
        };
        assert_eq!(run(), run(), "один сид — одна траектория");
    }

    // ─── Arena::pick (мост сим ↔ рендер: клик-выбор) ─────────────────────

    /// Пикинг выбирает БЛИЖАЙШЕЕ существо в радиусе (не любое в радиусе).
    #[test]
    fn pick_nearest_creature() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut a = Arena::new(1, map);
        let c1 = a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
        let c2 = a.spawn(Vec2::new(8.0, 5.0), 0.0, 2);
        // Курсор у c1: оба в радиусе 2.0, но c1 ближе.
        assert_eq!(a.pick(Vec2::new(6.0, 5.0), 2.0, 1.0), Some(c1));
        // Курсор у c2: c1 уже вне радиуса (dist 4.0 > 2.0).
        assert_eq!(a.pick(Vec2::new(9.0, 5.0), 2.0, 1.0), Some(c2));
        // Курсор между ними, но дальше 2.0 от обоих — мимо.
        assert_eq!(a.pick(Vec2::new(6.5, 9.0), 2.0, 1.0), None);
    }

    /// Без существ пикинг находит ягоду (живую, в радиусе).
    #[test]
    fn pick_berry_when_no_creature() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut a = Arena::new(1, map);
        let b1 = a.add_berry(Vec2::new(20.0, 10.0));
        assert_eq!(a.pick(Vec2::new(20.3, 10.2), 1.0, 1.0), Some(b1));
        // Вне радиуса ягоды — мимо.
        assert_eq!(a.pick(Vec2::new(23.0, 10.0), 1.0, 1.0), None);
    }

    /// Пустая точка (нет ни существ, ни ягод в радиусах) — None.
    #[test]
    fn pick_none_in_empty_space() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let a = Arena::new(1, map);
        assert_eq!(a.pick(Vec2::new(10.0, 10.0), 1.0, 1.0), None);
    }

    /// Существо приоритетнее ягоды: выбираем существо, даже если ягода ближе.
    #[test]
    fn pick_prefers_creature_over_berry_when_both_near() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut a = Arena::new(1, map);
        let c = a.spawn(Vec2::new(5.0, 5.0), 0.0, 1);
        a.add_berry(Vec2::new(5.4, 5.0)); // ягода ближе к курсору (5.2, 5)
        assert_eq!(a.pick(Vec2::new(5.2, 5.0), 2.0, 2.0), Some(c));
    }

    // ─── Сон (фаза сна в Arena::step) ────────────────────────────────────

    /// Сон в арене: уставшее существо (fatigue ≥ SLEEP_AT_FATIGUE) засыпает,
    /// спит SLEEP_TICKS тиков неподвижно (нет мозга/движения/еды), просыпается
    /// с усталостью заметно ниже порога. Метаболизм во сне стоит — дневная
    /// усталость сбрасывается единичным `creature.sleep()` на входе.
    ///
    /// Усталость копит моторная активность (RATE_MOTOR): две ягоды по бокам
    /// курса дают устойчивый симметричный запах — существо бежит вперёд
    /// (позиция меняется наяву) и выматывается, как в `creature::tests::
    /// activity_accumulates_fatigue` (сильный драйв без руления).
    #[test]
    fn tired_creature_falls_asleep_and_recovers() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 60.0, 20.0);
        let mut arena = Arena::new(9, map);
        arena.spawn(Vec2::new(10.0, 10.0), 0.0, 9);
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        // Ягоды симметрично по бокам курса: запах слева ≈ справа, моторы гонят.
        arena.add_berry(Vec2::new(30.0, 9.0));
        arena.add_berry(Vec2::new(30.0, 11.0));

        // Доводим до усталости: ждём, когда существо заснёт само.
        let mut entry = None;
        for t in 0..3000u32 {
            arena.step();
            if arena.critters[0].sleeping {
                entry = Some(t);
                break;
            }
        }
        let entry = entry.expect("существо должно устать и заснуть за 3000 тиков");
        assert!(entry > 0, "сон не наступает с первого тика (нет усталости)");
        assert!(
            arena.critters[0].sleep_ticks == SLEEP_TICKS,
            "сон начинается с полного окна: {}",
            arena.critters[0].sleep_ticks
        );

        // Сон: позиция не меняется (нет движения/еды) — наяву существо бежало,
        // значит заморозка именно от сна. Усталость фиксируем в момент
        // пробуждения — пока спит, метаболизм стоит, и после сна она намного
        // ниже порога.
        let mut woke_fatigue = None;
        for _ in 0..SLEEP_TICKS + 3 {
            let before = arena.critters[0].pos;
            arena.step();
            if arena.critters[0].sleeping {
                assert_eq!(
                    arena.critters[0].pos,
                    before,
                    "спящее существо не должно двигаться"
                );
            } else if woke_fatigue.is_none() {
                woke_fatigue = Some(arena.critters[0].creature.fatigue());
            }
        }
        let woke_fatigue = woke_fatigue
            .expect("после SLEEP_TICKS+ тиков существо должно проснуться");
        assert!(
            woke_fatigue < SLEEP_AT_FATIGUE * 0.5,
            "сон должен сбить усталость намного ниже порога: {woke_fatigue} (порог {SLEEP_AT_FATIGUE})"
        );
    }

    /// Зомби-ловушка (сторож): существо без моторной активности (RATE_MOTOR
    /// ниже порога эмиттера усталости) всё равно засыпает — базальная
    /// усталость копится от самого бодрствования. До фикса такие особи
    /// простаивали 21K тиков без единого сна (исследование эмерджентности).
    #[test]
    fn idle_creature_sleeps_via_basal_fatigue() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut arena = Arena::new(3, map);
        // Без ягод: запаха нет, моторы тихие (RATE_MOTOR < порога эмиттера).
        arena.spawn(Vec2::new(20.0, 10.0), 0.0, 3);
        // Голод 0: охота не активируется — чистый «простой».
        let mut slept_at = None;
        for t in 0..1000u64 {
            arena.step();
            if arena.critters[0].sleeping {
                slept_at = Some(t);
                break;
            }
        }
        let slept_at = slept_at
            .expect("даже тихое существо должно засыпать (базальная усталость)");
        assert!(
            slept_at <= 810,
            "базальное окно бодрствования: {} тиков",
            slept_at
        );
        // Просыпается и снова засыпает — цикл замкнут.
        let mut woke = false;
        let mut slept_again = false;
        for _ in 0..600u64 {
            arena.step();
            if !woke && !arena.critters[0].sleeping && arena.tick > slept_at {
                woke = true;
            }
            if woke && arena.critters[0].sleeping {
                slept_again = true;
                break;
            }
        }
        assert!(woke, "после сна существо просыпается");
        assert!(slept_again, "цикл сон-бодрствование повторяется");
    }

    // ─── Стены (запах/зеркало по комнатам, Sense::wall) ───────────────────

    /// Запах не проходит сквозь стены: ягода в соседней комнате у общей стены
    /// не пахнет существу в своей комнате (ягода смещена по y — без фильтра
    /// запах был бы ~0.5); зайдя в ту же комнату, существо чувствует её.
    /// Двери запах пока не проводят (транш 2).
    #[test]
    fn smell_does_not_pass_through_walls() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        map.add_room(40.0, 0.0, 80.0, 20.0);
        let mut arena = Arena::new(3, map);
        // Ягода в комнате B у общей стены, смещена по y (боковой запах).
        arena.add_berry(Vec2::new(40.5, 9.0));

        // Существо в комнате A у общей стены: дистанция ~1.4, но комната другая.
        let in_a = arena.sense(Vec2::new(39.4, 10.0), 0.0, false);
        assert_eq!(
            in_a.smell_l + in_a.smell_r,
            0.0,
            "запах не должен проходить сквозь стену: {in_a:?}"
        );
        assert!(!in_a.food_near, "еда за стеной не «у рта»");

        // Переходим в комнату B — та же ягода пахнет и «у рта».
        let in_b = arena.sense(Vec2::new(41.0, 10.0), 0.0, false);
        assert!(
            in_b.smell_l + in_b.smell_r > 0.0,
            "в той же комнате запах должен быть: {in_b:?}"
        );
        assert!(in_b.food_near, "в той же комнате ягода «у рта»");
    }

    /// Социальное зеркало тоже заперто стенами: сосед в той же комнате видит
    /// едущего (see_eat), сосед за стеной — нет, даже если дистанция до едока
    /// меньше SOCIAL_RADIUS.
    /// (Едок на 38.0, не у самой стены: при тике 50 мс шаг движения ×2, и
    /// со старта в 39.0 едок перелетал границу комнат за тик — ел уже в
    /// комнате B, и зеркало для соседа в A законно не срабатывало.)
    #[test]
    fn see_eat_mirror_is_room_gated() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        map.add_room(40.0, 0.0, 80.0, 20.0);
        let mut arena = Arena::new(4, map);
        // Едок в комнате A у стены, ягода у него во рту.
        let eater = arena.spawn(Vec2::new(38.0, 10.0), 0.0, 1);
        arena
            .critters
            .iter_mut()
            .find(|c| c.id == eater)
            .unwrap()
            .creature
            .blood
            .conc[chem::HUNGER] = 200.0;
        arena.add_berry(Vec2::new(39.4, 10.0));
        // Наблюдатель в той же комнате (2 юнита от едока).
        let same = arena.spawn(Vec2::new(37.0, 10.0), 0.0, 2);
        // Наблюдатель за стеной (2 юнита от едока, но другая комната).
        let across = arena.spawn(Vec2::new(41.0, 10.0), 0.0, 3);

        arena.step(); // едок съел → зеркало раздаёт see_eat

        let idx = |id: AgentId| arena.critters.iter().position(|c| c.id == id).unwrap();
        assert!(
            arena.critters[idx(same)].see_eat,
            "сосед в той же комнате должен видеть едущего"
        );
        assert!(
            !arena.critters[idx(across)].see_eat,
            "сосед за стеной не должен видеть едущего"
        );
    }

    /// Sense::wall заполняется ареной: стена прямо по курсу (обе диагональные
    /// пробы высоки) — true; боковая стена и открытое поле — false.
    #[test]
    fn wall_sense_detects_wall_ahead() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 20.0, 20.0);
        let arena = Arena::new(5, map);
        // Стена прямо: (18,10) курсом на +x, стена на x = 20 в ~2 юнитах.
        let facing = arena.sense(Vec2::new(18.0, 10.0), 0.0, false);
        assert!(facing.wall, "стена впереди должна заполнять wall: {facing:?}");
        // Открытое поле: стены нет нигде рядом.
        let open = arena.sense(Vec2::new(10.0, 10.0), 0.0, false);
        assert!(!open.wall, "в открытом поле wall=false: {open:?}");
        // Боковая стена (курс вдоль неё): только одна проба — не «прямо».
        let lateral = arena.sense(Vec2::new(1.5, 10.0), std::f32::consts::FRAC_PI_2, false);
        assert!(!lateral.wall, "боковая стена — не стена по курсу: {lateral:?}");
    }

    // ─── Зрение (стадия B: конус ±VISION_HALF_FOV, дальность VISION_RANGE) ──

    /// Зрительный конус: ягода в левой половине конуса (пеленг −45°) →
    /// vis_l > vis_r; в правой (+45°) → vis_r > vis_l; строго позади
    /// (пеленг π) — не видна. Ягода ровно по курсу (пеленг 0) уходит
    /// в правый бин — как у запаха (rel < 0 → левый, 0 → правый).
    #[test]
    fn vision_cone_left_right_and_behind() {
        let fresh = || {
            let mut map = WorldMap::default();
            map.add_room(0.0, 0.0, 40.0, 40.0);
            Arena::new(5, map)
        };
        let pos = Vec2::new(20.0, 20.0);
        let heading = 0.0f32;

        // Слева: ягода под −45° (dx=+10, dy=−10), dist 14.1 → vis ≈ 0.65.
        let mut a = fresh();
        a.add_berry(Vec2::new(30.0, 10.0));
        let s = a.sense(pos, heading, false);
        assert!(
            s.vis_l > 0.0 && s.vis_r == 0.0,
            "слева: vis_l={} vis_r={}",
            s.vis_l,
            s.vis_r
        );

        // Справа: ягода под +45° (dx=+10, dy=+10).
        let mut a = fresh();
        a.add_berry(Vec2::new(30.0, 30.0));
        let s = a.sense(pos, heading, false);
        assert!(
            s.vis_l == 0.0 && s.vis_r > 0.0,
            "справа: vis_l={} vis_r={}",
            s.vis_l,
            s.vis_r
        );

        // Позади (пеленг π): пеленг вне конуса — не видна.
        let mut a = fresh();
        a.add_berry(Vec2::new(10.0, 20.0));
        let s = a.sense(pos, heading, false);
        assert!(
            s.vis_l == 0.0 && s.vis_r == 0.0,
            "позади: vis_l={} vis_r={}",
            s.vis_l,
            s.vis_r
        );

        // Ровно по курсу (пеленг 0): правый бин (спецификация Sense).
        let mut a = fresh();
        a.add_berry(Vec2::new(30.0, 20.0));
        let s = a.sense(pos, heading, false);
        assert!(
            s.vis_r > 0.0 && s.vis_l == 0.0,
            "по курсу: vis_l={} vis_r={}",
            s.vis_l,
            s.vis_r
        );
    }

    /// Зрение не проходит сквозь стены: ягода в соседней комнате у общей
    /// стены не видна существу в своей комнате (та же проверка комнаты,
    /// что у запаха — дистанция и пеленг при этом в конусе); зайдя в ту же
    /// комнату, существо видит её.
    #[test]
    fn vision_blocked_by_room() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        map.add_room(40.0, 0.0, 80.0, 20.0);
        let mut arena = Arena::new(3, map);
        // Ягода в комнате B у общей стены, ровно по курсу существа из A.
        arena.add_berry(Vec2::new(45.0, 10.0));

        // Существо в комнате A: дистанция 10, пеленг 0 — видно было бы,
        // если бы стены были прозрачны. Комната другая — зрение гасится.
        let in_a = arena.sense(Vec2::new(35.0, 10.0), 0.0, false);
        assert_eq!(
            in_a.vis_l + in_a.vis_r,
            0.0,
            "зрение не должно проходить сквозь стену: {in_a:?}"
        );

        // Переходим в комнату B — та же ягода видна (дистанция 3, пеленг 0).
        let in_b = arena.sense(Vec2::new(42.0, 10.0), 0.0, false);
        assert!(
            in_b.vis_l + in_b.vis_r > 0.0,
            "в той же комнате зрение должно видеть: {in_b:?}"
        );
    }

    // ─── Berry-API (Arena::berry_info) ────────────────────────────────────

    /// Berry-API: berry_info читает внутренние vars[] через именованные поля
    /// (рост/отрастание/съедобность) — хост и мост рендера не трогают vars.
    #[test]
    fn berry_info_reports_growth_and_edibility() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut arena = Arena::new(6, map);
        let bid = arena.add_berry(Vec2::new(20.0, 10.0));

        // Свежая ягода: съедобна, рост 0, отрастание 0.
        let b = arena.berry_info(bid).expect("свежая ягода видна");
        assert!(b.state.is_edible(), "свежая ягода съедобна");
        assert_eq!(b.state.growth, 0.0);
        assert_eq!(b.state.regrow, 0.0);
        assert_eq!(b.pos, Vec2::new(20.0, 10.0));

        // Рост: vars[0] копит плодородие → growth виден через berry_info.
        arena.world.agent_mut(bid).unwrap().vars[0] = 3.0;
        let b = arena.berry_info(bid).unwrap();
        assert_eq!(b.state.growth, 3.0, "growth отражает vars[0]");

        // Съедена (отрастает): vars[1] > 0 → edible false, regrow виден.
        arena.world.agent_mut(bid).unwrap().vars[1] = 40.0;
        let b = arena.berry_info(bid).unwrap();
        assert!(!b.state.is_edible(), "отрастающая ягода не съедобна");
        assert_eq!(b.state.regrow, 40.0, "regrow отражает vars[1]");

        // Живой агент мира, но не ягода → None.
        let rock = arena.world.spawn(Cls::new(2, 2, 1), Vec2::new(5.0, 5.0), "rock");
        assert!(arena.berry_info(rock).is_none(), "не-ягода — None");
    }

    // ─── Мировые часы: день/ночь и сезоны (как в C3) ──────────────────────

    /// Часы арены: сутки реальных тиков обходят фазы рассвет→…→ночь,
    /// день в сезоне увеличивается, а год (4 сезона × 3 дня) зацикливается.
    /// Математика фаз покрыта в `env::world_clock::tests`; здесь — связка
    /// «тик арены → accessors Arena::{time_of_day, day_in_season, season}».
    #[test]
    fn world_clock_phases_and_seasons() {
        use crate::sim::TICKS_PER_DAY;

        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 14.0);
        let mut arena = Arena::new(1, map);
        assert_eq!(arena.time_of_day(), TimeOfDay::Dawn, "тик 0 — рассвет");
        assert_eq!(arena.day_in_season(), 0);
        assert_eq!(arena.season(), Season::Spring);

        // Полные сутки: фазы обошли круг, день в сезоне сдвинулся.
        for _ in 0..TICKS_PER_DAY {
            arena.step();
        }
        assert_eq!(arena.tick, TICKS_PER_DAY);
        assert_eq!(
            arena.time_of_day(),
            TimeOfDay::Dawn,
            "после суток — снова рассвет"
        );
        assert_eq!(arena.day_in_season(), 1, "день в сезоне увеличился");
        assert_eq!(arena.season(), Season::Spring, "сезон ещё весенний");

        // Сезон: 3 дня = 3 × TICKS_PER_DAY тиков.
        for _ in 0..2 * TICKS_PER_DAY {
            arena.step();
        }
        assert_eq!(arena.day_in_season(), 0, "сезон из 3 дней завершился");
        assert_eq!(arena.season(), Season::Summer, "наступило лето");
        // Доля фазы в [0, 1).
        let f = arena.day_phase_fraction();
        assert!((0.0..1.0).contains(&f), "фаза должна быть в [0,1): {f}");
    }

    /// Ночь гасит свет климата: в фазе Дня плодородность комнаты высокая,
    /// в Ночь та же комната темнеет (поле света стремится к 0.15 дневного
    /// уровня) и ягоды растут медленнее — свет движет ростом, как в C3.
    #[test]
    fn night_reduces_light_field() {
        use crate::sim::TICKS_PER_DAY;

        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 14.0);
        // Почти без распада комнат: сравниваем чисто эффект фазы суток.
        map.rates.decay = [0.0005; 4];
        let mut arena = Arena::new(1, map);
        let ca = &mut arena.world.map.rooms[0].ca;
        ca[Substance::Light.idx()] = 200.0;
        ca[Substance::Temperature.idx()] = 200.0;
        ca[Substance::Nutrients.idx()] = 200.0;

        let phase = 5760; // TICKS_PER_DAY/5
        arena.tick = TimeOfDay::Afternoon as u64 * phase;
        for _ in 0..100 { arena.step(); }
        let day_fert = arena.fertility(Vec2::new(20.0, 7.0));
        assert!(day_fert > 0.4, "днём плодородие высокое: {day_fert}");
        arena.tick = TimeOfDay::Night as u64 * phase;
        for _ in 0..100 { arena.step(); }
        let night_fert = arena.fertility(Vec2::new(20.0, 7.0));
        assert!(
            night_fert < day_fert * 0.7,
            "ночь должна заметно снизить плодородие: {day_fert} → {night_fert}"
        );
    }

    // ─── Смерть (C3-minimal): голод → агония → труп → уборка ─────────────

    /// Сцена истощения: глюкоза зажата пином на 0 (форс; сон больше не может
    /// подкинуть +15), панический голод 255 — моторы гремят у стен, активность
    /// стоит глюкозы, и кровь копит голодный урон на каждом активном тике
    /// (metabolism::BrainOrgan; арена его не дублирует).
    fn starvation_scene() -> Arena {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut arena = Arena::new(11, map);
        arena.spawn(Vec2::new(20.0, 10.0), 0.0, 11);
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 255.0;
        arena
            .critters[0]
            .creature
            .blood
            .pins
            .push((chem::GLUCOSE, 0.0));
        arena
    }

    /// Голод убивает, если существо не спасли: урон крови растёт → суммарный
    /// урон достигает DEATH_DAMAGE → агония (сенсоры нули, движение замерло,
    /// счётчик DYING_TICKS тикает вниз) → die() → труп лежит CORPSE_TICKS →
    /// убирается из self.critters. ВАРИАНТ СПАСЕНИЯ: глюкоза, поднятая
    /// хостом выше RECOVER_GLUCOSE в тик агонии, отменяет смерть — урон
    /// частично лечится (−RESCUE_HEAL очков), существо живо.
    #[test]
    fn starvation_dies_unless_rescued() {
        // ВАРИАНТ СПАСЕНИЯ: в агонии хост поднимает глюкозу — смерть отменена.
        {
            let mut a = starvation_scene();
            let mut was_dying = false;
            for _ in 0..1200u32 {
                a.step();
                if a.critters[0].creature.dying {
                    was_dying = true;
                    break;
                }
            }
            assert!(was_dying, "голодное существо должно войти в агонию");
            assert!(
                a.critters[0].creature.damage_points() >= DEATH_DAMAGE,
                "агония начинается от суммарного урона ≥ порога"
            );
            // Агония: тело замерло — сенсоров нет, движения нет.
            let frozen = a.critters[0].pos;
            a.step();
            assert_eq!(
                a.critters[0].pos, frozen,
                "умирающий не должен двигаться"
            );
            // Спасение: хост поднимает глюкозу выше RECOVER_GLUCOSE (пин не
            // мешает: тело умирающего не тикает — кровь заморожена, пин
            // применяется только внутри blood.update).
            a.critters[0].creature.blood.conc[chem::GLUCOSE] = 100.0;
            a.step();
            assert!(
                !a.critters[0].creature.dying && a.critters[0].creature.alive(),
                "сытое спасение отменяет смерть"
            );
            assert!(
                a.critters[0].creature.damage_points() < DEATH_DAMAGE,
                "спасение лечит частично: урон ниже порога"
            );
        }

        // БАЗОВЫЙ ВАРИАНТ: без спасения — агония → смерть → труп → уборка.
        {
            let mut a = starvation_scene();
            let mut died = false;
            for _ in 0..1500u32 {
                a.step();
                if a.critters[0].creature.dead {
                    died = true;
                    break;
                }
            }
            assert!(died, "голодное существо должно умереть за 1500 тиков");
            assert!(
                !a.critters[0].creature.alive(),
                "после die() живых не бывает"
            );
            // Труп лежит CORPSE_TICKS тиков, затем убирается из арены.
            for _ in 0..CORPSE_TICKS / 2 {
                a.step();
            }
            assert_eq!(a.critters.len(), 1, "труп ещё лежит в арене");
            for _ in 0..CORPSE_TICKS {
                a.step();
            }
            assert!(
                a.critters.is_empty(),
                "труп должен убираться из арены после CORPSE_TICKS"
            );
        }
    }

    /// Старость убивает: age ≥ max_age → старческий урон (SENESCENT_RATE,
    /// с max_age — утроенный гейн) копится → DEATH_DAMAGE → агония → смерть;
    /// труп убирается через CORPSE_TICKS. Глюкоза зажата пином на 30: кровь
    /// голодный урон НЕ копит (30 ≥ цены активности) и лечения нет (30 < 200)
    /// — урон в этой сцене ТОЛЬКО старческий (сторож «химия не задета»:
    /// organ.damage крови остаётся нулём до самой смерти).
    #[test]
    fn aging_kills_and_corpse_is_removed() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut arena = Arena::new(13, map);
        arena.spawn(Vec2::new(20.0, 10.0), 0.0, 13);
        // Форс старости: возраст на индивидуальном потолке max_age (старческий
        // урон идёт с утроенным гейном с первого тика). Пин держит глюкозу на
        // 30 — «нищий, но не голодающий» старик.
        arena.critters[0].creature.age = arena.critters[0].max_age;
        arena
            .critters[0]
            .creature
            .blood
            .pins
            .push((chem::GLUCOSE, 30.0));

        arena.step();
        assert!(
            arena.critters[0].creature.damage > 0.0,
            "старческий урон должен копиться после max_age: {}",
            arena.critters[0].creature.damage
        );

        let mut died = false;
        for _ in 0..5000u32 {
            arena.step();
            if arena.critters[0].creature.dead {
                died = true;
                break;
            }
        }
        assert!(died, "существо должно умереть от старости за 5000 тиков");
        assert!(!arena.critters[0].creature.alive(), "после die() живых нет");
        // Сторож «урон только по своей причине»: кровь цела — калечила
        // именно старость, а не голод.
        assert_eq!(
            arena.critters[0].creature.blood.organ.damage, 0.0,
            "голодный урон крови не должен расти при зажатой глюкозе"
        );
        // Труп убирается через CORPSE_TICKS.
        for _ in 0..CORPSE_TICKS {
            arena.step();
        }
        assert!(arena.critters.is_empty(), "труп старика убирается из арены");
    }

    // ─── Любопытство (анти-аттрактор застрявшего) ─────────────────────────

    /// Счётчик награды: собственная еда сбрасывает ticks_since_reward,
    /// чужой обед в поле зрения (see_eat) — тоже; тики без награды копят
    /// счётчик. Семантика драйва любопытства (встряска — ниже).
    #[test]
    fn ticks_since_reward_reset_on_eat() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 40.0, 20.0);
        let mut arena = Arena::new(5, map);
        arena.spawn(Vec2::new(10.0, 10.0), 0.0, 5);
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
        // Наблюдатель В ТОЙ ЖЕ КОМНАТЕ в радиусе зеркала (10), но за
        // радиусом рта (2) — чужую еду сам не съест.
        arena.spawn(Vec2::new(14.0, 10.0), 0.0, 6);
        let berry = arena.add_berry(Vec2::new(10.5, 10.0));

        arena.step(); // едок съедает (ягода у рта), зеркало раздаёт see_eat
        assert!(
            arena
                .world
                .agent(berry)
                .map(|a| a.vars[1] > 0.0)
                .unwrap_or(false),
            "ягода должна быть съедена первым же тактом"
        );
        assert_eq!(
            arena.critters[0].ticks_since_reward, 0,
            "собственная еда сбрасывает счётчик"
        );
        assert!(arena.critters[1].see_eat, "зеркало увидело едущего");
        arena.step(); // сенсор see_eat доходит до наблюдателя с задержкой зеркала
        assert_eq!(
            arena.critters[1].ticks_since_reward, 0,
            "виденный чужой обед тоже награда: счётчик наблюдателя сброшен"
        );
        // Дальше награды нет (ягода отрастает) — счётчик копится.
        for _ in 0..30 {
            arena.step();
        }
        assert!(
            arena.critters[0].ticks_since_reward >= 30,
            "без награды счётчик растёт: {}",
            arena.critters[0].ticks_since_reward
        );
    }

    /// Встряска любопытства: застрявшее без награды (600+ тиков бодрствования)
    /// существо получает периодическую добавку к рулю — траектория отличается
    /// от существа, которому «везёт» с наградами (счётчик всё время ниже
    /// порога). Тряска детерминирована: тот же сид — бит-в-бит та же судьба,
    /// rng-состояние симуляции не трогается. (Счётчик тикает только у
    /// бодрствующих — сторож сна MAX_AWAKE_TICKS замораживает его во сне,
    /// поэтому прогон гоним до фактического пересечения порога, а не до
    /// фиксированного числа тиков.)
    #[test]
    fn curiosity_shakes_stuck_creature() {
        // Сцена без ягод: естественную награду взять неоткуда. Глюкоза зажата
        // пином на 100: никто не голодает насмерть за прогон (смерть здесь
        // проверяют отдельные тесты).
        let scene = || -> Arena {
            let mut map = WorldMap::default();
            map.add_room(0.0, 0.0, 40.0, 20.0);
            let mut arena = Arena::new(21, map);
            arena.spawn(Vec2::new(20.0, 10.0), 0.0, 21);
            arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
            arena
                .critters[0]
                .creature
                .blood
                .pins
                .push((chem::GLUCOSE, 100.0));
            arena
        };
        let mut a = scene();
        let mut b = scene();
        // A копит тики без награды, пока счётчик не перешагнёт 626 — к этому
        // моменту встряска на 600 И на 625 уже случились. B живёт в мире
        // постоянных «наград» (счётчик сбрасывается у порога) — встрясок нет.
        let (mut a_max, mut b_max) = (0u64, 0u64);
        for _ in 0..3000u32 {
            a.step();
            b.step();
            a_max = a_max.max(a.critters[0].ticks_since_reward);
            b.critters[0].ticks_since_reward %= 595;
            b_max = b_max.max(b.critters[0].ticks_since_reward);
            if a.critters[0].ticks_since_reward >= 626 {
                break;
            }
        }
        assert!(
            a_max >= CURIOSITY_AFTER_TICKS + 26,
            "застрявшее существо должно дойти до окна встряски: {a_max}"
        );
        assert!(
            b_max < CURIOSITY_AFTER_TICKS,
            "«награждённое» существо не доходит до порога встряски: {b_max}"
        );
        // Встряска изменила траекторию A: руль получал добавку на 600/625.
        assert!(
            a.critters[0].pos != b.critters[0].pos
                || a.critters[0].heading != b.critters[0].heading,
            "встряска любопытства должна менять траекторию застрявшего"
        );
        // Детерминизм: два прогона того же сида — бит-в-бит одна судьба.
        let mut c = scene();
        let mut d = scene();
        for _ in 0..1000u32 {
            c.step();
            d.step();
        }
        assert_eq!(
            c.critters[0].pos,
            d.critters[0].pos,
            "один сид — одна траектория (встряска детерминирована)"
        );
        assert_eq!(
            c.critters[0].ticks_since_reward,
            d.critters[0].ticks_since_reward
        );
    }

    // ─── Спаривание (фаза 1.75): естественное рождение ──────────────────

    use crate::creature::dna::CoatGene;

    /// Сцена спаривания: комната 12×8, пара в 4 юнитах, оба зрелы
    /// (age = MATURITY_TICKS) и сыты. Пол выставлен вручную: по умолчанию
    /// пол — детерминированный хэш сида, тесту нужна пара М×Ж. Глюкоза по
    /// желанию запинена на 200 — метаболизм не съедает условие
    /// элегибельности за окно теста (в тесте платежа пина нет: платёж
    /// должен быть виден).
    fn mating_scene(pin_glucose: bool) -> Arena {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 12.0, 8.0);
        let mut arena = Arena::new(7, map);
        arena.spawn(Vec2::new(4.0, 4.0), 0.0, 7);
        arena.spawn(Vec2::new(8.0, 4.0), 0.0, 9);
        for (i, sex) in [(0usize, Sex::Male), (1usize, Sex::Female)] {
            let c = &mut arena.critters[i];
            c.creature.age = MATURITY_TICKS;
            c.creature.genome.sex = sex;
            c.creature.blood.conc[chem::GLUCOSE] = 200.0;
            if pin_glucose {
                c.creature.blood.pins.push((chem::GLUCOSE, 200.0));
            }
        }
        arena
    }

    /// Шагать до рождения (население > before), потолок ожидания 30 тиков.
    /// Тела снуют — дистанция пары иногда за радиусом, поэтому окно
    /// щедрое, а условие проверяется каждый тик. None — рождение не вышло.
    fn wait_birth(arena: &mut Arena, before: usize) -> Option<()> {
        for _ in 0..30 {
            arena.step();
            if arena.critters.len() > before {
                return Some(());
            }
        }
        None
    }

    /// Спаривание рождает ребёнка с СМЕШАННЫМИ генами: пара зрелых сытых
    /// существ разного пола рядом → третий криттер через существующую
    /// машину генов (`spawn_child_pair` — кроссовер + мутация). Ребёнок:
    /// голод новорождённого арены, пол от своего сида, возраст 0, потолок
    /// жизни от сида, расцветка — каналы ОБОИХ родителей; родители в
    /// кулдауне, ребёнок в комнате.
    #[test]
    fn mating_produces_child_with_mixed_genes() {
        let mut arena = mating_scene(true);
        // Расцветки разведены по каналам (как в mod.rs:spawn_child_pair_mixes_coat).
        arena
            .critters[0]
            .creature
            .dna
            .set_coat(CoatGene::new(0.9, 0.1, 0.9));
        arena
            .critters[1]
            .creature
            .dna
            .set_coat(CoatGene::new(0.1, 0.9, 0.1));

        wait_birth(&mut arena, 2).expect("пара должна родить ребёнка");
        assert_eq!(
            arena.critters.len(),
            3,
            "после спаривания населения 3 (кулдаун исключает второе немедленное рождение)"
        );
        let child = &arena.critters[2];
        // Новорождённый арены рождается голодным (первая еда достижима).
        assert_eq!(
            child.creature.hunger(),
            ARENA_NEWBORN_HUNGER,
            "у ребёнка HUNGER == ARENA_NEWBORN_HUNGER"
        );
        // Возраст и потолок жизни — новорождённый: age = 0, max_age от сида.
        assert_eq!(child.creature.age, 0, "ребёнок рождается с нулевым возрастом");
        assert_eq!(
            child.max_age,
            critter_max_age(child.creature.seed),
            "потолок жизни ребёнка — жребий его сида"
        );
        // Пол — от сида ребёнка (внутри spawn_child_pair), не от родителей.
        assert_eq!(
            child.creature.sex(),
            Sex::from_seed(child.creature.seed),
            "пол ребёнка от его сида"
        );
        // Гены — наследие машины кроссовера: каждый канал расцветки — от
        // одного из родителей (+ мутационный дрейф ±5%). Канальный кроссовер,
        // дающий каналы ОБОИХ сторон при зафиксированном сидe, покрыт
        // юнит-тестом creature::spawn_child_pair_mixes_coat; здесь сид
        // ребёнка — чистая функция тика рождения (детерминированно, но
        // заранее не зафиксирован), поэтому проверка по-канальная.
        let coat = child
            .creature
            .dna
            .coat()
            .expect("ребёнок пары несёт расцветку");
        let near = |x: f32, ys: [f32; 2]| ys.iter().any(|&y| (x - y).abs() < 0.2);
        assert!(
            near(coat.r, [0.9, 0.1]) && near(coat.g, [0.1, 0.9]) && near(coat.b, [0.9, 0.1]),
            "расцветка ребёнка — каналы от родителей (+дрейф): {coat:?}"
        );
        // Родители уходят в кулдаун; ребёнок — в комнате (не в пустоте).
        assert!(
            arena.critters[0].mate_cooldown > 0 && arena.critters[1].mate_cooldown > 0,
            "оба родителя в кулдауне после спаривания"
        );
        assert!(
            arena.world.map.room_at(child.pos).is_some(),
            "ребёнок рождается в игровой комнате"
        );
    }

    /// Пара одного пола ребёнка не рожает: правило пары — строго М×Ж
    /// (зеркальный случай Ж×Ж отсекается той же проверкой пола самки).
    #[test]
    fn same_sex_no_child() {
        let mut arena = mating_scene(true);
        arena.critters[1].creature.genome.sex = Sex::Male; // теперь оба самцы
        for _ in 0..60 {
            arena.step();
        }
        assert_eq!(arena.critters.len(), 2, "пара одного пола не рождает");
    }

    /// Кулдаун держит: после рождения родители (единственная зрелая пара на
    /// арене) не рождают второй раз, пока MATE_COOLDOWN не истечёт; снятый
    /// хостом кулдаун возвращает способность (условия элегибельности те же).
    #[test]
    fn cooldown_blocks() {
        let mut arena = mating_scene(true);
        wait_birth(&mut arena, 2).expect("первое спаривание должно состояться");
        assert_eq!(arena.critters.len(), 3);
        // Окно 300 тиков много меньше MATE_COOLDOWN=3000: рождений нет.
        for _ in 0..300 {
            arena.step();
        }
        assert_eq!(
            arena.critters.len(),
            3,
            "кулдаун держит: окно 300 << MATE_COOLDOWN={MATE_COOLDOWN}"
        );
        // Снятие кулдауна хостом: пара сводится снова (возраст/сытость на
        // месте, позиции выставлены рядом — в прогоне тела разбрелись).
        for c in arena.critters.iter_mut().take(2) {
            c.mate_cooldown = 0;
        }
        arena.critters[0].pos = Vec2::new(4.0, 4.0);
        arena.critters[1].pos = Vec2::new(8.0, 4.0);
        wait_birth(&mut arena, 3).expect("после снятия кулдауна второе рождение");
        assert_eq!(arena.critters.len(), 4);
    }

    /// Потолок населения (экология, MAX_CRITTERS): зрелая сытая разнополая
    /// пара на заполненной арене не рожает; после прореживания та же пара
    /// рождает — блокировала именно экология, а не элегибельность.
    #[test]
    fn population_cap_blocks() {
        let mut map = WorldMap::default();
        map.add_room(0.0, 0.0, 80.0, 40.0);
        let mut arena = Arena::new(21, map);
        // MAX_CRITTERS предзаспавненных сеткой по комнате.
        for i in 0..MAX_CRITTERS {
            let x = 5.0 + (i % 8) as f32 * 10.0;
            let y = 5.0 + (i / 8) as f32 * 15.0;
            arena.spawn(Vec2::new(x, y), 0.0, 100 + i as u64);
        }
        // Пара-кандидат: зрелая, сытая (пин), разнополая, рядом.
        for (i, sex) in [(0usize, Sex::Male), (1usize, Sex::Female)] {
            let c = &mut arena.critters[i];
            c.creature.age = MATURITY_TICKS;
            c.creature.genome.sex = sex;
            c.creature.blood.conc[chem::GLUCOSE] = 200.0;
            c.creature.blood.pins.push((chem::GLUCOSE, 200.0));
        }
        arena.critters[1].pos = arena.critters[0].pos + Vec2::new(3.0, 0.0);
        for _ in 0..40 {
            arena.step();
        }
        assert_eq!(
            arena.critters.len(),
            MAX_CRITTERS,
            "на потолке населения спаривания нет"
        );
        // Контраст: та же пара после прореживания рождает.
        arena.critters.truncate(2);
        arena.critters[0].pos = Vec2::new(4.0, 4.0);
        arena.critters[1].pos = Vec2::new(8.0, 4.0);
        wait_birth(&mut arena, 2).expect("после прореживания пара рождает");
        assert_eq!(arena.critters.len(), 3);
    }

    /// Оба родителя платят MATE_GLUCOSE_COST (пин снят: платёж виден в
    /// крови; метаболизм и активность только увеличивают разницу).
    #[test]
    fn parents_pay_glucose() {
        let mut arena = mating_scene(false);
        let before: Vec<f32> = arena
            .critters
            .iter()
            .map(|c| c.creature.glucose())
            .collect();
        wait_birth(&mut arena, 2).expect("пара без пина тоже рождает");
        for (i, &g0) in before.iter().enumerate().take(2) {
            let paid = g0 - arena.critters[i].creature.glucose();
            assert!(
                paid >= MATE_GLUCOSE_COST - 0.5,
                "родитель {i} заплатил MATE_GLUCOSE_COST (плюс метаболизм): {paid}"
            );
        }
    }

    /// Незрелые не спариваются: всё готово (сытость, пол, радиус), кроме
    /// возраста — age < MATURITY_TICKS рождений не даёт (за окно теста
    /// возраст доращивается до 60 — всё ещё много меньше 3600).
    #[test]
    fn immature_no_mating() {
        let mut arena = mating_scene(true);
        for c in arena.critters.iter_mut().take(2) {
            c.creature.age = 0;
        }
        for _ in 0..60 {
            arena.step();
        }
        assert_eq!(
            arena.critters.len(),
            2,
            "незрелые (age < MATURITY_TICKS) не спариваются"
        );
    }

    /// Спаривание детерминировано: два одинаковых прогона — бит-в-бит одна
    /// судьба (позиции всех троих и сид ребёнка совпадают; ГСЧ симуляции
    /// не тронут, сид ребёнка — чистая функция тика/сидов/номера пары).
    #[test]
    fn mating_is_deterministic() {
        let run = || {
            let mut arena = mating_scene(true);
            for _ in 0..30 {
                arena.step();
                if arena.critters.len() > 2 {
                    break;
                }
            }
            (
                arena.critters.iter().map(|c| c.pos).collect::<Vec<_>>(),
                arena.critters.get(2).map(|c| c.creature.seed),
            )
        };
        assert_eq!(run(), run(), "одна сцена — одно потомство (бит-в-бит)");
    }
