// ─── Тесты (извлечены из модуля: приватный доступ через super) ───
// Содержимое перенесено механически: #[path] держит модуль в том же
// дереве, все тесты видят приватные элементы как раньше.

    use super::*;
    use crate::brain::instincts::Instinct;
    use crate::creature::dna::CoatGene;

    #[test]
    fn pacemaker_fires_rhythmically() {
        let mut c = Creature::new(7);
        // Эксперимент «сердце»: без пейсмейкера регион I молчит, с током —
        // спайкает. Детерминированно от сида.
        assert!(c.pacemaker().is_none(), "по умолчанию пейсмейкер выключен");

        // Ток 0.06 — чуть выше реобазы (свип: 0.04 молчит, 0.06 → 86/200,
        // 0.08 → 200/200): LIF успевает разрядиться между спайками — ритм.
        c.enable_pacemaker(3, 0.06);
        let pm: Vec<usize> = c
            .pacemaker()
            .expect("пейсмейкер включён")
            .0
            .to_vec();
        assert!(!pm.is_empty());

        // Сенсоры молчат: если спайки есть — это пейсмейкер, а не драйв.
        let sense = Sense::default();
        let mut total = 0i64;
        let mut ticks_fired = 0u32;
        for _ in 0..200u64 {
            c.tick(&sense);
            let fired_this_tick: i64 = c
                .spike_hist()
                .iter()
                .map(|col| pm.iter().map(|&i| col[i] as i64).sum::<i64>())
                .sum();
            if fired_this_tick > 0 {
                ticks_fired += 1;
            }
            total += fired_this_tick;
        }
        assert!(total > 0, "пейсмейкер должен спайкать: {total} спайков за 200 тиков");
        assert!(
            ticks_fired > 10 && ticks_fired < 200,
            "ритм: пейсмейкер спайкает периодически, а не постоянно: {ticks_fired}/200"
        );
    }

    #[test]
    fn smell_turns_toward_food() {
        let mut c = Creature::new(7);
        // Запах слева → поворот влево (steer < 0), запах справа → вправо.
        let left = c.tick(&Sense {
            smell_l: 1.0,
            smell_r: 0.0,
            ..Default::default()
        });
        assert!(left.steer < 0.0, "запах слева: steer={}", left.steer);
        let right = c.tick(&Sense {
            smell_l: 0.0,
            smell_r: 1.0,
            ..Default::default()
        });
        assert!(right.steer > 0.0, "запах справа: steer={}", right.steer);
    }

    #[test]
    fn hungry_creature_eats_sated_does_not() {
        let mut hungry = Creature::new(11);
        hungry.blood.conc[chem::HUNGER] = 200.0;
        let a = hungry.tick(&Sense {
            food_near: true,
            ..Default::default()
        });
        assert!(a.eat, "голодный должен есть");
        assert!(hungry.blood.conc[chem::FOOD] > 0.0, "еда попала в желудок");

        let mut sated = Creature::new(11);
        let a = sated.tick(&Sense {
            food_near: true,
            ..Default::default()
        });
        assert!(!a.eat, "сытый не должен есть");
    }

    #[test]
    fn activity_accumulates_fatigue() {
        let mut c = Creature::new(13);
        c.blood.conc[chem::HUNGER] = 200.0;
        // Долго бежим вперёд (сильный драйв → RATE_MOTOR > 0).
        let f0 = c.fatigue();
        for _ in 0..300 {
            c.tick(&Sense {
                smell_l: 0.5,
                smell_r: 0.5,
                ..Default::default()
            });
        }
        assert!(
            c.fatigue() > f0 + 1.0,
            "активность должна копить утомление: {} → {}",
            f0,
            c.fatigue()
        );
    }

    /// Геттеры настроения читают канал крови: серотонин/кортизол/дофамин —
    /// именованный доступ для инспектора/игры. Еда отражается в REWARD-канале.
    #[test]
    fn mood_getters_report_blood_chems() {
        let mut c = Creature::new(7);
        // Геттеры = кровь (детерминированные начальные значения).
        assert_eq!(c.serotonin(), c.blood.conc[chem::SEROTONIN]);
        assert_eq!(c.cortisol(), c.blood.conc[chem::CORTISOL]);
        assert_eq!(c.dopamine_level(), c.blood.conc[chem::REWARD]);
        // Свежий организм: серотонин — фон, дофамин — ноль.
        assert!(c.serotonin() > 0.0, "фоновый серотонин: {}", c.serotonin());
        assert_eq!(c.dopamine_level(), 0.0, "дофамин до еды нулевой");

        // Еда → дофамин в кровь (канал REWARD через EVENT_ATE) — геттер видит.
        c.blood.conc[chem::HUNGER] = 200.0;
        c.tick(&Sense {
            food_near: true,
            ..Default::default()
        });
        assert!(
            c.dopamine_level() > 0.0,
            "еда должна дать дофамин: {}",
            c.dopamine_level()
        );
    }

    #[test]
    fn drive_positive_when_hungry() {
        let mut c = Creature::new(17);
        c.blood.conc[chem::HUNGER] = 200.0;
        let a = c.tick(&Sense::default());
        assert!(a.drive > 0.0, "голодный движется: drive={}", a.drive);
    }

    /// STDP реально обучается: серия ударов о стену (при одновременном запахе)
    /// ослабляет рефлекс запах→руль — steer к еде падает вдвое и больше (LTD
    /// через негативный дофамин). Контроль с `lr=0` не меняет рефлекс: гаснет
    /// именно пластичность, а не тревожность/утомление (боль усиливает turn_gain).
    #[test]
    fn wall_punishment_weakens_smell_steer_reflex() {
        let (before, after) = punish_reflex(13, PlasticityParams::default().lr);
        assert!(before > 0.8, "исходный рефлекс вправо: steer={before}");
        assert!(
            after < before * 0.5,
            "наказание стеной должно сильно ослабить рефлекс: {before} → {after}"
        );

        // Контроль без пластичности: рефлекс жив (его держит только LTD).
        let (_, ctrl_after) = punish_reflex(13, 0.0);
        assert!(
            ctrl_after > 0.8,
            "без обучения (lr=0) рефлекс не должен гаснуть: after={ctrl_after}"
        );
    }

    /// Направленное ощущение стены: стена слева отворачивает вправо, стена
    /// справа — влево (перекрёстная проводка WL→MR / WR→ML в avoidance_layout).
    #[test]
    fn wall_sensors_steer_away_from_wall() {
        for seed in [1u64, 7, 13, 42] {
            let mut c = Creature::new(seed);
            c.blood.conc[chem::HUNGER] = 200.0;
            let wall_l = avg_steer(
                &mut c,
                Sense {
                    wall_l: 1.0,
                    ..Default::default()
                },
            );
            let wall_r = avg_steer(
                &mut c,
                Sense {
                    wall_r: 1.0,
                    ..Default::default()
                },
            );
            assert!(
                wall_l > 0.2,
                "стена слева → поворот вправо (seed {seed}): steer={wall_l}"
            );
            assert!(
                wall_r < -0.2,
                "стена справа → поворот влево (seed {seed}): steer={wall_r}"
            );
        }
    }

    /// Ключевое свойство: направленные сенсоры стены защищают гоминг от
    /// наказания. Во время удара о стену активен и запах еды (манит в стену),
    /// и канал стены; LTD бьёт по обоим, но мягкий гомеостаз и перераспределение
    /// сохраняют запах→мотор. Без стены-сенсора тот же удар выжигает гоминг
    /// целиком (steer к еде → 0) — существо «забывает», как охотиться.
    #[test]
    fn wall_sense_protects_homing_from_punishment() {
        for seed in [1u64, 13, 42] {
            let mut with = Creature::new(seed);
            let mut without = Creature::new(seed);
            with.blood.conc[chem::HUNGER] = 200.0;
            without.blood.conc[chem::HUNGER] = 200.0;
            let punish_with = Sense {
                smell_r: 1.0,
                wall_l: 1.0,
                wall: true,
                ..Default::default()
            };
            let punish_without = Sense {
                smell_r: 1.0,
                wall: true,
                ..Default::default()
            };
            for _ in 0..400 {
                with.tick(&punish_with);
                without.tick(&punish_without);
            }
            let clean = Sense {
                smell_r: 1.0,
                ..Default::default()
            };
            let with_steer = avg_steer(&mut with, clean);
            let without_steer = avg_steer(&mut without, clean);
            assert!(
                with_steer > 0.5,
                "с сенсором стены гоминг переживает наказание (seed {seed}): {with_steer}"
            );
            assert!(
                without_steer < with_steer * 0.3,
                "без сенсора стены гоминг гаснет (seed {seed}): {with_steer} → {without_steer}"
            );
        }
    }

    /// Средний steer по 40 тикам (сенсор константный).
    fn avg_steer(c: &mut Creature, sense: Sense) -> f32 {
        let mut acc = 0.0f32;
        for _ in 0..40 {
            acc += c.tick(&sense).steer;
        }
        acc / 40.0
    }

    /// Сумма весов коннектома (для сравнения «выучено vs свежевыращено»).
    fn total_weight(c: &Creature) -> f32 {
        c.brain.csr_triplets().2.iter().sum()
    }

    /// Сумма БАЗОВЫХ весов проводки (grown — эталон до жизни).
    fn grown_weight(c: &Creature) -> f32 {
        c.grown.edges.iter().map(|e| e.2 as f64).sum::<f64>() as f32
    }

    /// Наследственность коннектома: потомок пересаживает топологию родителя
    /// (рёбра/регионы/типы — без роста заново), а сид роста мутирует веса —
    /// линия эволюционирует, а не клонируется и не растит проводку с нуля.
    #[test]
    fn spawn_child_replants_parent_wiring_with_seed_mutation() {
        let parent = Creature::new(7);
        let child = parent.spawn_child(99);
        assert_eq!(child.seed, 99, "сид потомка — его собственный");
        assert_ne!(
            child.genome.steering(), parent.genome.steering(),
            "геном роста должен мутировать"
        );
        // Топология наследуется: та же структура проводки (число рёбер,
        // регионы, типы), как если бы ребёнок вырос из той же схемы.
        assert_eq!(
            child.grown.edges.len(),
            parent.grown.edges.len(),
            "число синапсов наследуется"
        );
        assert_eq!(child.grown.region_of, parent.grown.region_of);
        assert_eq!(child.grown.nt, parent.grown.nt);
        // Веса дрейфуют от родительских (мутация сида роста) — не клон.
        assert!(
            (grown_weight(&child) - grown_weight(&parent)).abs() > 1.0,
            "сид роста мутирует веса: {} vs {}",
            grown_weight(&child),
            grown_weight(&parent)
        );
    }

    /// Размножение детерминировано: один сид потомка — один и тот же потомок
    /// (геном и коннектом бит-в-бит).
    #[test]
    fn spawn_child_is_deterministic_from_seed() {
        let parent = Creature::new(13);
        let a = parent.spawn_child(77);
        let b = parent.spawn_child(77);
        assert_eq!(
            a.genome.to_creature_genome(0),
            b.genome.to_creature_genome(0)
        );
        assert_eq!(total_weight(&a), total_weight(&b));
        assert_eq!(
            a.blood.organ.cost_per_activity,
            b.blood.organ.cost_per_activity
        );
    }

    /// Пол потомка — как у C3: случайно при наследовании, но детерминированно
    /// (один сид потомка — один пол; по диапазону сидов встречаются оба пола).
    /// Пол не мутирует — у ребёнка он заново выбран из его сида.
    #[test]
    fn child_sex_is_random_per_seed_but_deterministic() {
        let parent = Creature::new(1);
        let mut males = 0;
        let mut seen: Vec<Sex> = Vec::new();
        for child_seed in 0..64u64 {
            let child = parent.spawn_child(child_seed);
            let expected = Sex::from_seed(child_seed);
            assert_eq!(
                child.sex(),
                expected,
                "пол потомка = пол из его сида (seed {child_seed})"
            );
            if child.sex() == Sex::Male {
                males += 1;
            }
            seen.push(child.sex());
        }
        assert!(
            males > 0 && males < 64,
            "по диапазону сидов должны быть оба пола: самцов {males}/64"
        );
        // Один сид — один пол (детерминированность).
        assert_eq!(
            parent.spawn_child(7).sex(),
            parent.spawn_child(7).sex(),
            "один сид потомка — один пол"
        );
        assert!(
            seen.iter().any(|&s| s == Sex::Male) && seen.iter().any(|&s| s == Sex::Female),
            "потомки бывают обоих полов"
        );
    }

    /// Наследование инстинктов и генов метаболизма: ребёнок получает их из
    /// генома родителя (с мутацией в границах), кровь собрана по генам.
    #[test]
    fn instincts_and_metabolism_inherited_with_mutation() {
        let mut genome = CreatureGenome::default();
        genome.instincts = vec![Instinct {
            inputs: vec![(3, 0.5)],
            dopamine: 2.5,
        }];
        genome.metabolism.cost_per_activity = 0.01;
        let parent = Creature::from_genome(Genome::from_creature_genome(&genome), 5);
        assert_eq!(
            parent.blood.organ.cost_per_activity, 0.01,
            "кровь родителя собрана по его генам"
        );

        let child = parent.spawn_child(6);
        // Инстинкт унаследован (мутация дрейфует дофамин в [0.2, 8]).
        assert_eq!(child.genome.instincts().len(), 1, "инстинкт наследуется");
        let d = child.genome.instincts()[0].dopamine;
        assert!(
            (0.2..=8.0).contains(&d),
            "дофамин инстинкта в границах: {d}"
        );
        assert!(
            (d - 2.5).abs() < 1.0,
            "дофамин дрейфует от родительского, а не прыгает: {d}"
        );
        // Метаболизм унаследован и применён к крови ребёнка.
        let cost = child.blood.organ.cost_per_activity;
        assert!(
            cost > 0.005,
            "стоимость активности унаследована от родителя (0.01 ± дрейф): {cost}"
        );
        assert!(
            (cost - 0.01).abs() < 0.005,
            "мутация метаболизма мягкая: {cost}"
        );
    }

    /// Рождение = первый сон: унаследованные инстинкты прожигаются в свежий
    /// мозг — ребёнок рождается с врождённым рефлексом родителя и реагирует
    /// сильнее с ПЕРВОГО тика (ещё до всякого обучения). Инстинкт усиливает
    /// существующий путь SL→ML (гоминг на запах слева): ко-активация всего
    /// SL и всего ML при рождении даёт eligibility всем синапсам пути,
    /// дофамин + консолидация делают его постоянным. Контроль без инстинкта
    /// на том же коннектоме и сиде отворачивает слабее. Потолок клэмпа при
    /// рождении выше рабочего w_max, иначе дофамин срезал бы grown-хвосты
    /// весов и ОСЛАБИЛ рефлекс (см. burn_into_with_cap).
    #[test]
    fn offspring_born_with_burned_in_reflex() {
        // Регионы avoidance_layout фиксированы: SL=0..20, SR=20..40,
        // WL=40..52, WR=52..64, I=64..80, ML=80..100, MR=100..120.
        let mut inputs = Vec::new();
        for i in 0..20 {
            inputs.push((i, 4.0)); // весь SL — «запах слева»
        }
        for i in 80..100 {
            inputs.push((i, 8.0)); // весь ML — «поворот влево»
        }
        for seed in [1u64, 7, 13, 42] {
            let mut genome = CreatureGenome::default();
            genome.instincts = vec![Instinct {
                inputs: inputs.clone(),
                dopamine: 2.0,
            }];
            let mut born = Creature::from_genome(Genome::from_creature_genome(&genome.clone()), seed);
            let mut plain = Creature::from_genome(
                Genome::from_creature_genome(&CreatureGenome {
                    instincts: Vec::new(),
                    ..genome
                }),
                seed,
            );
            born.blood.conc[chem::HUNGER] = 200.0;
            plain.blood.conc[chem::HUNGER] = 200.0;
            // Слабый стимул — ниже насыщения моторов: рефлекс меряется по
            // асимметрии, а не по потолку.
            let smell_l = Sense {
                smell_l: 0.05,
                ..Default::default()
            };
            let born_steer = avg_steer(&mut born, smell_l);
            let plain_steer = avg_steer(&mut plain, smell_l);
            assert!(
                born_steer < -0.5,
                "прожитый рефлекс должен быть сильным (seed {seed}): steer={born_steer}"
            );
            assert!(
                born_steer < plain_steer - 0.1,
                "рождённый с инстинктом отворачивает заметно сильнее контроля \
                 (seed {seed}): {born_steer} vs {plain_steer}"
            );
        }
    }

    /// Эффект Болдуина: успешно охотившийся родитель передаёт потомку
    /// свёрнутый рефлекс. После охоты (запах слева + еда) путь SL→ML
    /// родителя реально потенцирован; `spawn_child` сворачивает его в
    /// инстинкт-ген, и потомок рождается с врождённым гомингом — с первого
    /// тика отворачивает к запаху. Потомок НЕНАУЧЕННОГО родителя (тот же
    /// сид) рождается без рефлекса: разница только в выученном, всё
    /// остальное (геном, сид потомка) идентично.
    #[test]
    fn hunting_parent_passes_learned_reflex_to_offspring() {
        let hunt = Sense {
            smell_l: 1.0,
            food_near: true,
            ..Default::default()
        };
        let mut trained = Creature::new(1);
        trained.blood.conc[chem::HUNGER] = 200.0;
        for _ in 0..300 {
            trained.tick(&hunt);
        }
        assert!(trained.food_eaten() > 0, "родитель должен охотиться");
        let naive = Creature::new(1);

        let learned = trained.learned_instincts();
        assert!(
            !learned.is_empty(),
            "охота должна оставить выученный путь (SL→ML)"
        );
        assert_eq!(
            naive.learned_instincts().len(),
            0,
            "ненаученный родитель ничего не передаёт"
        );

        let mut t_child = trained.spawn_child(99);
        let mut n_child = naive.spawn_child(99);
        assert_eq!(
            t_child.genome.instincts().len(),
            1,
            "потомок рождается с инстинктом, свёрнутым из выученного"
        );
        assert_eq!(
            n_child.genome.instincts().len(),
            0,
            "потомок наивного — без инстинктов"
        );
        t_child.blood.conc[chem::HUNGER] = 200.0;
        n_child.blood.conc[chem::HUNGER] = 200.0;

        // Оба потомка наследуют ОДНУ проводку родителя (один сид 1 + один
        // сид потомка 99) — различие только в прожитом инстинкте. Стимул
        // 0.05: ниже насыщения моторов, чтобы разница рефлекса была видна
        // (при 0.1 оба упираются в steer = -1).
        let smell = Sense {
            smell_l: 0.05,
            ..Default::default()
        };
        let t_steer = avg_steer(&mut t_child, smell);
        let n_steer = avg_steer(&mut n_child, smell);
        assert!(
            t_steer < -0.05,
            "потомок охотника рождён с рефлексом: steer={t_steer}"
        );
        assert!(
            t_steer < n_steer - 0.05,
            "рефлекс потомка охотника заметно сильнее наивного: {t_steer} vs {n_steer}"
        );
    }

    /// Свёртка выученного детерминирована: два потомка обученного родителя
    /// из одного сида — бит-в-бит одинаковые геномы (инстинкты совпадают).
    #[test]
    fn learned_reflex_inheritance_is_deterministic() {
        let hunt = Sense {
            smell_l: 1.0,
            food_near: true,
            ..Default::default()
        };
        let mut trained = Creature::new(13);
        trained.blood.conc[chem::HUNGER] = 200.0;
        for _ in 0..300 {
            trained.tick(&hunt);
        }
        let a = trained.spawn_child(77);
        let b = trained.spawn_child(77);
        assert_eq!(a.genome, b.genome);
        assert_eq!(
            total_weight(&a),
            total_weight(&b),
            "прожиг свёрнутого инстинкта детерминирован"
        );
    }

    /// Прожигание детерминировано: два рождения из одного генома и сида —
    /// бит-в-бит одинаковый мозг (веса совпадают).
    #[test]
    fn birth_burn_is_deterministic_from_seed() {
        let mut genome = CreatureGenome::default();
        genome.instincts = vec![Instinct {
            inputs: vec![(0, 4.0), (80, 8.0)],
            dopamine: 2.0,
        }];
        let a = Creature::from_genome(Genome::from_creature_genome(&genome.clone()), 21);
        let b = Creature::from_genome(Genome::from_creature_genome(&genome), 21);
        assert_eq!(total_weight(&a), total_weight(&b));
        assert_eq!(a.genome, b.genome);
    }

    /// Обнуление выученных весов: родитель учится (веса мозга меняются), но
    /// потомок пересаживает БАЗОВУЮ проводку родителя (`grown` — эталон до
    /// жизни) с печатью своего сида — выученное родителем не наследуется:
    /// потомок ближе к базе, чем к обученному мозгу родителя.
    #[test]
    fn learned_weights_are_reset_in_offspring() {
        let mut parent = Creature::new(7);
        parent.blood.conc[chem::HUNGER] = 200.0;
        let base = grown_weight(&parent); // базовые веса проводки родителя
        let w0 = total_weight(&parent);
        // Обучение: наказание стеной реально меняет веса мозга.
        let punish = Sense {
            smell_r: 1.0,
            wall_l: 1.0,
            wall: true,
            ..Default::default()
        };
        for _ in 0..200 {
            parent.tick(&punish);
        }
        let w1 = total_weight(&parent);
        assert!(
            (w1 - w0).abs() > 1.0,
            "обучение должно изменить веса родителя: {w0} → {w1}"
        );

        let child = parent.spawn_child(99);
        // Потомок наследует базовую проводку (не обученную): его веса ближе
        // к `base` (+ печать сида), чем к обученному мозгу родителя `w1`.
        assert!(
            (grown_weight(&child) - base).abs() < (w1 - base).abs() * 0.5,
            "потомок наследует базу, а не выученное: base={base} child={} w1={w1}",
            grown_weight(&child)
        );
        assert!(
            (total_weight(&child) - w1).abs() > 1.0,
            "потомок не копирует выученные веса родителя"
        );
    }

    /// Расширяемый геном наследуется: потомок получает расцветку родителя
    /// (мутация машины дрейфует каналы, не сбрасывая к дефолту), а зеркала
    /// legacy-классов остаются в синхроне с легаси-геномом (инвариант
    /// параллельного пути).
    #[test]
    fn spawn_child_carries_coat_gene() {
        let mut parent = Creature::new(7);
        parent.genome.set_coat(CoatGene::new(0.9, 0.4, 0.1));
        let child = parent.spawn_child(99);
        let coat = child.genome.coat().expect("потомок несёт ген расцветки");
        assert!(
            (coat.r - 0.9).abs() < 0.15
                && (coat.g - 0.4).abs() < 0.15
                && (coat.b - 0.1).abs() < 0.15,
            "расцветка наследуется с мягким дрейфом: {coat:?}"
        );
        assert!(
            coat != CoatGene::default(),
            "потомок не сбрасывается к дефолту (белый): {coat:?}"
        );
        // Инвариант зеркал: dna-адаптер восстанавливает легаси-геном.
        let mirrored = child.genome.to_creature_genome(0);
        assert_eq!(mirrored.steering, *child.genome.steering());
        assert_eq!(mirrored.rule_genes, child.genome.rule_genes());
        assert_eq!(mirrored.instincts, child.genome.instincts());
        assert_eq!(mirrored.metabolism, *child.genome.metabolism());
    }

    /// Потомок ПАРЫ: по-канальный кроссовер расцветки — ребёнок несёт каналы
    /// обоих родителей (+ мутационный дрейф), зеркала синхронны.
    #[test]
    fn spawn_child_pair_mixes_coat() {
        let mut a = Creature::new(1);
        a.genome.set_coat(CoatGene::new(0.9, 0.1, 0.9));
        let mut b = Creature::new(2);
        b.genome.set_coat(CoatGene::new(0.1, 0.9, 0.1));
        let child = a.spawn_child_pair(&b, 42);
        let coat = child.genome.coat().expect("потомок пары несёт расцветку");
        let from_a = (coat.r - 0.9).abs() < 0.2
            || (coat.g - 0.1).abs() < 0.2
            || (coat.b - 0.9).abs() < 0.2;
        let from_b = (coat.r - 0.1).abs() < 0.2
            || (coat.g - 0.9).abs() < 0.2
            || (coat.b - 0.1).abs() < 0.2;
        assert!(
            from_a && from_b,
            "потомок пары содержит каналы обоих родителей: {coat:?}"
        );
        assert_eq!(
            child.genome.to_creature_genome(0).steering,
            child.genome.steering().clone(),
            "зеркала legacy-классов синхронны после кроссовера"
        );
    }

    /// Социальное обучение: существо видит соседа за едой (see_eat) и
    /// получает зеркальный дофамин — путь запах→руль потенцируется БЕЗ
    /// собственной еды. Культура распространяется стаей без генов.
    #[test]
    fn social_learning_reinforces_behavior_without_eating() {
        // Ученик: нюхает еду слева (рулит влево) и видит, как сосед ест.
        let mut learner = Creature::new(3);
        learner.blood.conc[chem::HUNGER] = 200.0;
        let watch = Sense {
            smell_l: 1.0,
            smell_r: 0.0,
            see_eat: true,
            ..Default::default()
        };
        for _ in 0..300 {
            learner.tick(&watch);
        }
        assert_eq!(learner.food_eaten(), 0, "ученик сам не ест");
        let g_learner = learner.brain.learned_gain(REGION_SL, REGION_ML);

        // Контроль: тот же прогон без зеркального сигнала.
        let mut lonely = Creature::new(3);
        lonely.blood.conc[chem::HUNGER] = 200.0;
        let plain = Sense {
            smell_l: 1.0,
            smell_r: 0.0,
            ..Default::default()
        };
        for _ in 0..300 {
            lonely.tick(&plain);
        }
        let g_lonely = lonely.brain.learned_gain(REGION_SL, REGION_ML);

        assert!(
            g_learner > 0.0,
            "зеркальный сигнал дал выученный прирост: {g_learner}"
        );
        assert!(
            g_learner > g_lonely + 10.0,
            "социальное обучение сильнее одиночного: {g_learner} vs {g_lonely}"
        );
    }

    /// Память места: существо запоминает, в каком направлении была еда
    /// в знакомом месте, и без запаха рулит туда по памяти (гоминг без
    /// таксиса). В незнакомом месте памяти нет — руля нет.
    #[test]
    fn place_memory_guides_homing_without_smell() {
        let mut c = Creature::new(7);
        c.blood.conc[chem::HUNGER] = 200.0;
        // Учим: в месте 5 еда была, когда смотрели на восток (heading 0).
        let meal = Sense {
            smell_l: 1.0,
            food_near: true,
            place: 5,
            heading: 0.0,
            ..Default::default()
        };
        for _ in 0..30 {
            assert!(c.tick(&meal).eat, "должен есть и запоминать");
        }
        assert_eq!(c.places_remembered(), 1, "место 5 запомнено");

        // Запаха нет, смотрим на север (π/2) в месте 5 → память тянет
        // на восток: поворот вправо (steer > 0).
        let mem = Sense {
            place: 5,
            heading: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        let steer_known = avg_steer(&mut c, mem);
        assert!(
            steer_known > 0.2,
            "память ведёт к еде на востоке: steer={steer_known}"
        );

        // Контроль: незнакомое место (99) без памяти — руля нет.
        let other = Sense {
            place: 99,
            heading: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        let steer_unknown = avg_steer(&mut c, other);
        assert!(
            steer_unknown.abs() < 0.1,
            "в незнакомом месте памяти нет: steer={steer_unknown}"
        );
        assert!(
            steer_known > steer_unknown + 0.2,
            "память должна заметно отличаться от пустоты: {steer_known} vs {steer_unknown}"
        );
    }

    /// Сон улучшает охоту: два существа охотятся одинаковое число тиков,
    /// но «спящий» после дня закрепляет выученное (сон: консолидация +
    /// якорь гомеостаза), а «бодрый» простаивает без сна. Спящий сохраняет
    /// больше выученного прироста пути запах→мотор (learned_gain) — его
    /// день не вымывается дрейфом, и охота опирается на закреплённый рефлекс.
    #[test]
    fn sleep_cycle_beats_awake_hunting() {
        let hunt = Sense {
            smell_l: 1.0,
            food_near: true,
            ..Default::default()
        };
        let night = Sense::default();
        let mut sleeper = Creature::new(5);
        let mut awake = Creature::new(5);
        sleeper.blood.conc[chem::HUNGER] = 200.0;
        awake.blood.conc[chem::HUNGER] = 200.0;
        for day in 0..6 {
            for _ in 0..50 {
                sleeper.tick(&hunt);
            }
            for _ in 0..50 {
                awake.tick(&hunt);
            }
            if day % 2 == 1 {
                sleeper.sleep();
                // Ночь без сна: дрейф гомеостаза вымывает выученное.
                for _ in 0..25 {
                    awake.tick(&night);
                }
            }
        }
        let g_s = sleeper.brain.learned_gain(REGION_SL, REGION_ML);
        let g_a = awake.brain.learned_gain(REGION_SL, REGION_ML);
        assert!(g_s > 0.0, "спящий сохранил выученный прирост пути: {g_s}");
        assert!(
            g_s > g_a,
            "сон закрепляет выученное лучше бодрствования: sleeper={g_s} awake={g_a}"
        );
    }

    /// Прогон: мерить steer с запахом справа, 300 раз ударить о стену с тем же
    /// запахом, мерить снова.
    fn punish_reflex(seed: u64, lr: f32) -> (f32, f32) {
        let mut c = Creature::new_with_params(
            seed,
            PlasticityParams {
                lr,
                ..Default::default()
            },
        );
        c.blood.conc[chem::HUNGER] = 200.0;
        let smell_r = Sense {
            smell_r: 1.0,
            ..Default::default()
        };
        let smell_r_wall = Sense {
            smell_r: 1.0,
            wall: true,
            ..Default::default()
        };
        let avg_steer = |c: &mut Creature| -> f32 {
            let mut s = 0.0f32;
            for _ in 0..60 {
                s += c.tick(&smell_r).steer;
            }
            s / 60.0
        };
        let before = avg_steer(&mut c);
        for _ in 0..300 {
            c.tick(&smell_r_wall);
        }
        let after = avg_steer(&mut c);
        (before, after)
    }
