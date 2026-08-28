//! Интеграционные offscreen-тесты моста «сим ↔ рендер» (этап 1):
//! снимок арены, презентация в спрайты, маппер ввода. Тесты с GPU-устройством
//! пропускаются, если адаптера нет (см. tests/common/mod.rs).

mod common;

use dioma::biology::metabolism::chem;
use dioma::creature::Creature;
use dioma::env::map::{Rect, WorldMap};
use dioma::env::Vec2;
use dioma::sim::Arena;
use dioma::view::{GameCommand, InputMapper, Presenter, SimView};
use dioma_render::{Camera, SpriteAtlas};
use winit::keyboard::KeyCode;

/// Маленькая арена для тестов: комната 10×10, одно существо, одна ягода.
/// `hungry` — сразу поднять голод выше порога `wants_food` (как в тестах движка).
fn test_arena(hungry: bool) -> Arena {
    let mut map = WorldMap::default();
    map.add_room(0.0, 0.0, 10.0, 10.0);
    let mut arena = Arena::new(42, map);
    arena.spawn_creature(Vec2::new(3.0, 3.0), 0.5, Creature::new(7));
    if hungry {
        arena.critters[0].creature.blood.conc[chem::HUNGER] = 200.0;
    }
    arena.add_berry(Vec2::new(7.0, 7.0));
    arena
}

/// Обратная сторона [`Camera::screen_to_world`]: мировая точка → экранные
/// пиксели (Y вниз) — для «клика» по известной мировой позиции.
fn world_to_screen(cam: &Camera, w: [f32; 2]) -> [f32; 2] {
    [
        cam.viewport[0] * 0.5 + (w[0] - cam.center[0]) * cam.zoom,
        cam.viewport[1] * 0.5 - (w[1] - cam.center[1]) * cam.zoom,
    ]
}

/// `SimView::capture` копирует поля арены: существо (позиция/курс/статы),
/// ягода (позиция/фаза), комнаты (прямоугольники).
#[test]
fn snapshot_captures_arena() {
    let arena = test_arena(false);
    let view = SimView::capture(&arena);

    // Существо: id, позиция, курс, живо, статы нормированы 0..1.
    assert_eq!(view.critters.len(), 1);
    let c = &view.critters[0];
    assert_eq!(c.id, arena.critters[0].id);
    assert_eq!(c.pos, Vec2::new(3.0, 3.0));
    assert_eq!(c.heading, 0.5);
    assert!(c.alive);
    assert_eq!(c.food_eaten, 0);
    // Пол существа — из его сида (детерминированно, как в C3 при наследовании).
    assert_eq!(
        c.sex,
        dioma_engine::creature::Sex::from_seed(7),
        "пол существа = пол из сида 7"
    );
    // Свежее существо: глюкоза 128/255 ≈ 0.5; голод 120/255 ≈ 0.47 —
    // новорождённый арены голоден с рождения (ARENA_NEWBORN_HUNGER=120,
    // см. sim::spawn_creature — замкнутый круг новичка); усталость ≈ 0.
    assert!((0.4..0.6).contains(&c.glucose), "глюкоза {} вне 0.4..0.6", c.glucose);
    assert!(
        (0.4..0.6).contains(&c.hunger),
        "голод новорождённого {} вне 0.4..0.6 (≈120/255)",
        c.hunger
    );
    assert!(c.fatigue < 0.1, "усталость свежего существа {} должна быть ~0", c.fatigue);

    // Мировые часы: тик 0 — рассвет, день 1 весны, фаза в начале.
    assert_eq!(view.time_of_day, dioma::view::TimeOfDay::Dawn);
    assert_eq!(view.day_in_season, 0);
    assert_eq!(view.season, dioma::view::Season::Spring);
    assert_eq!(view.phase_fraction, 0.0);

    // Ягода: позиция и фаза 1.0 (свежая: vars[1] == 0 → спелая).
    assert_eq!(view.berries.len(), 1);
    let b = &view.berries[0];
    assert_eq!(b.id, arena.world.agent(b.id).unwrap().id);
    assert_eq!(b.pos, Vec2::new(7.0, 7.0));
    assert_eq!(b.phase, 1.0, "свежая ягода должна быть спелой (фаза 1.0)");

    // Комнаты: прямоугольник скопирован.
    assert_eq!(view.rooms.len(), 1);
    assert_eq!(view.rooms[0].rect, Rect::new(0.0, 0.0, 10.0, 10.0));
    assert_eq!(view.rooms[0].idx, 0);
}

/// `Presenter::build` порождает спрайты: тайлы пола/стен, тень+тело+лицо на
/// каждое существо, ягоду. Голодное существо получает лицо «голодное»
/// (UV региона отличается от спокойного) и тёплый тинт тела.
#[test]
fn presenter_builds_scene() {
    let Some(gpu) = common::gpu() else { return };
    let mut atlas = SpriteAtlas::new(&gpu.device, 256);
    let presenter = Presenter::new(&gpu.queue, &mut atlas);

    let arena = test_arena(true); // существо голодное (wants_food)
    let view = SimView::capture(&arena);
    let scene = presenter.build(&view);

    // Комната 10×10 → тайлы есть (пол + стены).
    assert!(scene.tiles.len() > 0, "пол+стены должны дать тайлы");
    // По одному спрайту на существо и ягоду.
    assert_eq!(scene.bodies.len(), 1, "тело существа");
    assert_eq!(scene.shadows.len(), 1, "тень существа");
    assert_eq!(scene.faces.len(), 1, "лицо существа");
    assert_eq!(scene.berries.len(), 1, "ягода");

    // Глубина тела — `-pos.y` (y-сортировка «выше = дальше»).
    assert_eq!(scene.bodies[0].depth, -3.0);

    // Голодное существо: лицо «голодное», а не «спокойное».
    let hungry_uv = atlas.region("view_face_hungry").unwrap();
    let calm_uv = atlas.region("view_face_calm").unwrap();
    assert_eq!(scene.faces[0].uv_rect, hungry_uv, "голодное → лицо «голодное»");
    assert_ne!(scene.faces[0].uv_rect, calm_uv, "лицо должно отличаться от спокойного");
    // Тинт тела теплеет (не зелёный «сытый»).
    assert_ne!(scene.bodies[0].tint, [0.42, 0.92, 0.52, 1.0], "тинт тела должен теплеть");

    // Спокойное существо — контроль: лицо «спокойное». Новорождённый арены
    // голоден с рождения (ARENA_NEWBORN_HUNGER=120 > wants_food 40) — контроль
    // «сытого» лица получаем явным обнулением голода.
    let mut calm_arena = test_arena(false);
    calm_arena.critters[0].creature.blood.conc[chem::HUNGER] = 0.0;
    let scene_calm = presenter.build(&SimView::capture(&calm_arena));
    assert_eq!(scene_calm.faces[0].uv_rect, calm_uv, "сытое → лицо «спокойное»");
}

/// `InputMapper`: курсор на существе → `Select(Some(id))`, на пустоте →
/// `Select(None)`, Shift+клик → `SpawnCritter`, Space → `TogglePause`.
#[test]
fn input_mapper_selects_critter() {
    let arena = test_arena(false);
    let cid = arena.critters[0].id;

    let mut cam = Camera::new([100.0, 100.0]);
    cam.center = [5.0, 5.0];
    cam.zoom = 4.0;
    let mut input = InputMapper::new();

    // Клик ровно по существу (круглый обмен world↔screen при таких числах точен).
    input.on_cursor(world_to_screen(&cam, [3.0, 3.0]), &cam);
    assert_eq!(
        input.on_click(false, &arena),
        Some(GameCommand::Select(Some(cid))),
        "клик по существу выделяет его"
    );

    // Клик по пустоте (далеко от всего) → Select(None).
    input.on_cursor(world_to_screen(&cam, [-5.0, -5.0]), &cam);
    assert_eq!(input.on_click(false, &arena), Some(GameCommand::Select(None)));

    // Shift+клик — спавн существа в точке курсора.
    input.on_cursor(world_to_screen(&cam, [3.0, 3.0]), &cam);
    assert_eq!(
        input.on_click(true, &arena),
        Some(GameCommand::SpawnCritter(Vec2::new(3.0, 3.0)))
    );

    // Клавиши: Space — пауза.
    assert_eq!(input.on_key(KeyCode::Space), Some(GameCommand::TogglePause));
    // Скорости 1/2/4.
    assert_eq!(input.on_key(KeyCode::Digit1), Some(GameCommand::Speed(1)));
    assert_eq!(input.on_key(KeyCode::Digit2), Some(GameCommand::Speed(2)));
    assert_eq!(input.on_key(KeyCode::Digit3), Some(GameCommand::Speed(4)));
    // Escape — снять выделение.
    assert_eq!(input.on_key(KeyCode::Escape), Some(GameCommand::Select(None)));
    // Неизвестная клавиша — None.
    assert_eq!(input.on_key(KeyCode::KeyQ), None);
}
