// ─── Тесты (извлечены из модуля: приватный доступ через super) ───
// Содержимое перенесено механически: #[path] держит модуль в том же
// дереве, все тесты видят приватные элементы как раньше.

    use super::*;

    fn toy_grown() -> Grown {
        let edges = vec![
            (0, 3, 1.5),
            (1, 4, 1.2),
            (2, 5, 1.8),
            (0, 4, 0.9),
            (1, 3, 0.7),
        ];
        Grown {
            edges,
            region_of: vec![0, 0, 0, 1, 1, 1],
            nt: vec![1.0; 6],
            growth_s: 0.0,
            rule_genes: Vec::new(),
            instincts: Vec::new(),
        }
    }

    fn homeo_params() -> PlasticityParams {
        PlasticityParams {
            homeo_every_ticks: 1,
            ..Default::default()
        }
    }

    #[test]
    fn sleep_consolidation_anchors_learning() {
        let grown = toy_grown();
        let gains = gains_from(2, &[(0, 1, 1.0)]);
        let mean_w = |b: &Brain| {
            let (_, _, v) = b.csr_triplets();
            v.iter().sum::<f32>() / v.len() as f32
        };

        // «Обучение»: удвоили все веса.
        let mut learned = Brain::new(&grown, gains.clone(), homeo_params());
        let m0 = mean_w(&learned);
        let (_, _, vals) = learned.csr_triplets();
        let doubled: Vec<f32> = vals.iter().map(|&v| v * 2.0).collect();
        learned.set_csr_values(&doubled).unwrap();

        // A: закрепил во сне — гомеостаз держит новую норму.
        let mut sleeper = Brain::new(&grown, gains.clone(), homeo_params());
        sleeper.set_csr_values(&doubled).unwrap();
        sleeper.sleep_consolidate(1.0);
        let zero = vec![0.0f32; 6];
        for _ in 0..160 {
            sleeper.step(&zero);
        }
        let ma = mean_w(&sleeper);

        // B: не спал — мягкая нормализация тянет обратно к старой норме.
        let mut awake = Brain::new(&grown, gains, homeo_params());
        awake.set_csr_values(&doubled).unwrap();
        for _ in 0..160 {
            awake.step(&zero);
        }
        let mb = mean_w(&awake);

        assert!(
            ma > m0 * 1.6,
            "консолидированный должен держать выученное: m0={m0} ma={ma}"
        );
        assert!(
            mb < ma - m0 * 0.2,
            "без сна выученное вымывается: ma={ma} mb={mb}"
        );
    }

    /// Сон-консолидация весов: сильное выученное закрепляется и усиливается,
    /// слабый шум отсекается. Эталон рождения (`memory_lt`) не трогается.
    #[test]
    fn sleep_consolidation_reinforces_strong_prunes_weak() {
        let grown = toy_grown();
        let gains = gains_from(2, &[(0, 1, 1.0)]);
        let mut b = Brain::new(&grown, gains, homeo_params());
        let lt = b.dynamics.values().to_vec(); // снимок рождения

        // 0-й синапс сильно выучен (+2), остальные — шум (+0.05).
        let mut vals = lt.clone();
        vals[0] += 2.0;
        for v in vals.iter_mut().skip(1) {
            *v += 0.05;
        }
        b.set_csr_values(&vals).unwrap();

        b.consolidate_sleep(0.5, 0.2, 0.8);
        let out = b.dynamics.values().to_vec();

        let d0 = out[0] - lt[0];
        assert!(
            d0 > 2.0 + 0.2,
            "сильное выученное усиливается при закреплении: d0={d0}"
        );
        for i in 1..lt.len() {
            let d = (out[i] - lt[i]).abs();
            assert!(d < 0.02, "слабый шум отсекается во сне: синапс {i} d={d}");
        }
        // Эталон Болдуина не сдвинут: выученное продолжает накапливаться.
        assert_eq!(b.memory_lt, lt);
    }
