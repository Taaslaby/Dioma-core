//! Спрайт-атлас: одна wgpu-текстура + процедурные генераторы пикселей (CPU)
//! + загрузка PNG-ассетов с обрезкой прозрачных полей и авто-ростом.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

/// Предельный размер стороны атласа: дальше атлас не растёт, добавление
/// возвращает ошибку «атлас переполнен».
const MAX_ATLAS_SIZE: u32 = 8192;

/// Гуттер (рамка) вокруг региона при включённых мипмапах: texels, по которым
/// усредняются ТОЛЬКО цвета самого региона (края реплицируются в рамку),
/// а не соседей по атласу.
const MIP_GUTTER: u32 = 4;

/// Максимум мип-уровней атласа (включая нулевой): 4 уровня — до 1/16 стороны,
/// дальше мерцание уже не мешает, а подтекание цветов растёт.
const MIP_MAX_LEVELS: u32 = 4;

/// Информация о регионе атласа: UV-прямоугольник, смещение обрезки и размер
/// в пикселях. Смещение > 0 означает, что прозрачная рамка исходного
/// изображения была обрезана при загрузке (см. [`SpriteAtlas::add_png`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionInfo {
    /// Нормализованный UV-прямоугольник `[u0, v0, u1, v1]` (как у [`SpriteAtlas::region`]).
    pub uv: [f32; 4],
    /// Смещение левого верхнего угла региона внутри исходного изображения
    /// в пикселях (0,0 — без обрезки; >0 — прозрачная рамка срезана).
    pub offset: [f32; 2],
    /// Размер региона в пикселях (после обрезки).
    pub size_px: [u32; 2],
}

/// Атлас спрайтов: текстура `RGBA8UnormSrgb` (линейная фильтрация) + UV-регионы.
///
/// Спрайты укладываются рядами слева-направо сверху-вниз. Когда спрайт не
/// влезает, атлас растёт автоматически: сторона удваивается (максимум 8192),
/// все регионы переупаковываются в порядке добавления (детерминированный
/// результат), данные перезаливаются на GPU, сэмплер и bind group
/// пересоздаются. Атлас и рендерер должны быть созданы на одном
/// wgpu-устройстве: bind group атласа ссылается на bind group layout с тем же
/// описанием, что и у пайплайна рендерера — wgpu дедуплицирует одинаковые
/// bind group layout'ы, поэтому они совместимы.
pub struct SpriteAtlas {
    /// Клон устройства: нужен для создания новой текстуры при росте атласа.
    /// `None` — атлас, собранный вокруг готовой текстуры ([`Self::from_texture`]),
    /// такие атласы не растут.
    device: Option<Arc<wgpu::Device>>,
    texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    /// Число мип-уровней текстуры атласа. 1 — без мипмапов (по умолчанию);
    /// включается до первого добавления региона через [`Self::enable_mipmaps`].
    mip_levels: u32,
    next_x: u32,
    next_y: u32,
    row_height: u32,
    regions: HashMap<String, [f32; 4]>,
    /// Исходные RGBA8-пиксели регионов (для переупаковки при росте атласа).
    pixels: HashMap<String, (u32, u32, Vec<u8>)>,
    /// Смещение региона в исходном изображении (см. [`RegionInfo::offset`]).
    offsets: HashMap<String, [f32; 2]>,
    /// Порядок добавления регионов — по нему идёт переупаковка, чтобы рост
    /// был детерминированным (HashMap итерируется непредсказуемо).
    order: Vec<String>,
}

impl SpriteAtlas {
    /// Квадратный атлас `size`×`size` пикселей.
    ///
    /// Устройство передаётся через `Arc<wgpu::Device>` (wgpu 22 `Device` не
    /// клонируется): атлас хранит его клон, чтобы при авто-росте создавать
    /// новую текстуру — `wgpu::Device` дешёвый (внутри — общий контекст).
    pub fn new(device: &Arc<wgpu::Device>, size: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dioma-render sprite atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let (sampler, bind_group) = make_sampler_and_bind_group(device, &texture);
        Self {
            device: Some(device.clone()),
            texture,
            sampler,
            bind_group,
            width: size,
            height: size,
            mip_levels: 1,
            next_x: 0,
            next_y: 0,
            row_height: 0,
            regions: HashMap::new(),
            pixels: HashMap::new(),
            offsets: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Добавляет спрайт `name` из плотно упакованных RGBA8-пикселей (`w × h × 4`).
    ///
    /// Если спрайт не влезает в текущий размер — атлас растёт автоматически,
    /// детерминированно, в порядке добавления: сторона удваивается (максимум
    /// 8192), все существующие регионы переупаковываются из сохранённых
    /// пикселей, затем укладывается новый. При превышении лимита возвращается
    /// ошибка «атлас переполнен».
    pub fn add_rgba(
        &mut self,
        queue: &wgpu::Queue,
        name: &str,
        w: u32,
        h: u32,
        pixels: &[u8],
    ) -> Result<(), String> {
        self.add_rgba_impl(queue, name, w, h, pixels, [0.0, 0.0])
    }

    /// Перезаписывает пиксели СУЩЕСТВУЮЩЕГО региона того же размера
    /// (тема UI перекрашивает панель/кнопку без переукладки атласа).
    /// Если региона нет — работает как [`Self::add_rgba`].
    pub fn replace_rgba(
        &mut self,
        queue: &wgpu::Queue,
        name: &str,
        w: u32,
        h: u32,
        pixels: &[u8],
    ) -> Result<(), String> {
        let Some(info) = self.region_info(name) else {
            return self.add_rgba(queue, name, w, h, pixels);
        };
        if pixels.len() != (w as usize) * (h as usize) * 4 {
            return Err(format!(
                "размер данных {} ≠ {w}×{h}×4",
                pixels.len()
            ));
        }
        if info.size_px != [w, h] {
            return Err(format!(
                "регион '{name}' имеет размер {:?}, замена {w}×{h} невозможна",
                info.size_px
            ));
        }
        // Пиксельный левый-верхний угол региона в текстуре из нормализованных uv.
        let x = (info.uv[0] * self.width as f32).round() as u32;
        let y = (info.uv[1] * self.height as f32).round() as u32;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        Ok(())
    }

    /// Внутренняя реализация добавления: как [`Self::add_rgba`], но с явным
    /// смещением обрезки (для PNG-регионов, у которых прозрачная рамка срезана).
    fn add_rgba_impl(
        &mut self,
        queue: &wgpu::Queue,
        name: &str,
        w: u32,
        h: u32,
        pixels: &[u8],
        offset: [f32; 2],
    ) -> Result<(), String> {
        if self.regions.contains_key(name) {
            return Err(format!("спрайт '{name}' уже есть в атласе"));
        }
        if pixels.len() != (w as usize) * (h as usize) * 4 {
            return Err(format!(
                "спрайт '{name}': размер данных {} ≠ {w}×{h}×4",
                pixels.len()
            ));
        }

        // Укладываем; если не влезает — атлас растёт (удвоение стороны, до
        // 8192) и переупаковывает все регионы, пока место не появится.
        let (x, y) = loop {
            if let Some(p) = self.place(w, h) {
                break p;
            }
            let next_size = (self.width.max(self.height) * 2).max(2);
            if next_size > MAX_ATLAS_SIZE {
                return Err(format!("атлас переполнен: регион '{name}' {w}×{h}"));
            }
            self.grow(queue, next_size)?;
        };
        let p = self.pad_texels();

        upload_region(
            queue,
            &self.texture,
            (x - p, y - p),
            w,
            h,
            pixels,
            p,
            self.mip_levels,
        );

        let u0 = x as f32 / self.width as f32;
        let v0 = y as f32 / self.height as f32;
        let u1 = (x + w) as f32 / self.width as f32;
        let v1 = (y + h) as f32 / self.height as f32;
        self.regions.insert(name.to_string(), [u0, v0, u1, v1]);
        self.pixels
            .insert(name.to_string(), (w, h, pixels.to_vec()));
        self.offsets.insert(name.to_string(), offset);
        self.order.push(name.to_string());
        Ok(())
    }

    /// Загружает PNG-файл `path` и добавляет спрайт `name` (укладка и авто-рост
    /// как у [`Self::add_rgba`]). Прозрачные поля по краям обрезаются; смещение
    /// обрезки доступно через [`Self::region_info`]. Ошибки чтения/декодирования
    /// возвращаются русскими сообщениями с именем и путём.
    pub fn add_png(&mut self, queue: &wgpu::Queue, name: &str, path: &str) -> Result<(), String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("PNG '{name}': не удалось прочитать файл '{path}': {e}"))?;
        self.add_png_bytes(queue, name, &data)
    }

    /// То же, что [`Self::add_png`], но из буфера в памяти.
    ///
    /// Поддерживаются 8-битные gray/rgb/rgba/palette (декодирование с
    /// `Transformations::EXPAND`: палитра → RGB, tRNS → альфа-канал); любой
    /// другой бит-глубины PNG отклоняется. Прозрачные поля обрезаются,
    /// смещение доступно через [`Self::region_info`].
    pub fn add_png_bytes(
        &mut self,
        queue: &wgpu::Queue,
        name: &str,
        data: &[u8],
    ) -> Result<(), String> {
        let decoder = png::Decoder::new(Cursor::new(data));
        let mut reader = decoder
            .read_info()
            .map_err(|e| format!("PNG '{name}': не удалось разобрать заголовок: {e}"))?;

        // Только 8-битный вход; остальные глубины отклоняем заранее.
        let info = reader.info();
        if info.bit_depth != png::BitDepth::Eight {
            return Err(format!(
                "PNG '{name}': поддерживается только 8-битный PNG (gray/rgb/rgba/palette)"
            ));
        }
        let width = info.width;
        let height = info.height;

        // EXPAND выдаёт буфер в ИСХОДНОМ числе каналов (gray → 1, rgb → 3,
        // rgba → 4, палитра → rgb → 3) — приводим к RGBA8 вручную ниже.
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let out = reader
            .next_frame(&mut buf)
            .map_err(|e| format!("PNG '{name}': не удалось декодировать: {e}"))?;
        let rgba = to_rgba8(out.color_type, &buf);

        let (w, h, offset, cropped) = trim_rgba(width, height, &rgba);
        self.add_rgba_impl(queue, name, w, h, &cropped, offset)
    }

    /// Нормализованный UV-прямоугольник спрайта `[u0, v0, u1, v1]`.
    pub fn region(&self, name: &str) -> Option<[f32; 4]> {
        self.regions.get(name).copied()
    }

    /// Информация о регионе: UV, смещение обрезки и размер в пикселях.
    /// Неизвестное имя (или регион без сохранённых пикселей, например глифы
    /// шрифта) → `None`.
    pub fn region_info(&self, name: &str) -> Option<RegionInfo> {
        let uv = self.regions.get(name).copied()?;
        let (w, h, _) = self.pixels.get(name)?;
        Some(RegionInfo {
            uv,
            offset: self.offsets.get(name).copied().unwrap_or([0.0, 0.0]),
            size_px: [*w, *h],
        })
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// Bind group атласа (текстура + сэмплер) для шейдерной группы 1.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Текущая ширина атласа в пикселях (растёт при переполнении).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Текущая высота атласа в пикселях.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Текущий размер атласа `(ширина, высота)` в пикселях.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn len(&self) -> usize {
        self.regions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// Включает мипмапы для атласа — НЕ РИСКУЕТ ПОСЛЕ первого добавления
    /// региона (ошибка), и только если атлас может расти (`device` сохранён).
    ///
    /// Пересоздаёт текстуру с цепочкой уровней (до [`MIP_MAX_LEVELS`]) и
    /// включает гуттер [`MIP_GUTTER`] texels вокруг каждого региона: на уровнях
    /// выше нулевого усредняются только собственные пиксели региона, цвета
    /// соседей не подтекают (края реплицируются в гуттер). UV-прямоугольники
    /// ([`Self::region`]) остаются интерьерными — визуально картинка не меняется;
    /// нога от гуттера на укладке даёт чуть меньшую полезную площадь, чем раньше,
    /// поэтому большой регион может упереться в лимит размера чуть раньше.
    ///
    /// Вызов повторно — no-op.
    pub fn enable_mipmaps(&mut self) -> Result<(), String> {
        if !self.regions.is_empty() {
            return Err("мипмапы включаются ДО первого добавления региона (существующие пиксели переупаковывать некорректно)".to_string());
        }
        let Some(device) = &self.device else {
            return Err(
                "атлас не может расти: устройство не сохранено (атлас собран из готовой текстуры)"
                    .to_string(),
            );
        };
        if self.mip_levels > 1 {
            return Ok(()); // уже включены
        }
        let levels = level_count(self.width);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dioma-render sprite atlas"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let (sampler, bind_group) = make_sampler_and_bind_group(device, &texture);
        self.texture = texture;
        self.sampler = sampler;
        self.bind_group = bind_group;
        self.mip_levels = levels;
        Ok(())
    }

    /// Сколько texels гуттера резервировать вокруг региона (0 без мипмапов).
    fn pad_texels(&self) -> u32 {
        if self.mip_levels > 1 {
            MIP_GUTTER
        } else {
            0
        }
    }

    /// Собирает атлас вокруг УЖЕ созданной текстуры `w`×`h` (регионы пусты).
    ///
    /// Нужен [`crate::debug::DebugFont`], который создаёт текстуру с данными
    /// напрямую (`create_texture_with_data`, без очереди) и регистрирует
    /// регионы вручную через [`Self::insert_region`].
    pub(crate) fn from_texture(
        device: &wgpu::Device,
        texture: wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Self {
        let (sampler, bind_group) = make_sampler_and_bind_group(device, &texture);
        Self {
            // Устройство не сохраняется: атлас вокруг готовой текстуры (глифы
            // шрифта) не содержит пикселей регионов и не может расти.
            device: None,
            texture,
            sampler,
            bind_group,
            width,
            height,
            mip_levels: 1,
            next_x: 0,
            next_y: 0,
            row_height: 0,
            regions: HashMap::new(),
            pixels: HashMap::new(),
            offsets: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Регистрирует UV-регион `name` без копирования пикселей (для атласов,
    /// созданных через [`Self::from_texture`], где данные уже на GPU).
    pub(crate) fn insert_region(&mut self, name: &str, uv: [f32; 4]) {
        self.regions.insert(name.to_string(), uv);
    }

    /// Простая shelf-укладка: ряды слева-направо, высота ряда — максимум по ряду.
    ///
    /// Резервирует место под регион ВМЕСТЕ с гуттером (`w+2p × h+2p`) и
    /// возвращает координаты ИНТЕРЬЕРНОГО угла региона; соседние регионы
    /// гарантированно отделены `2p`-texel-ной пустотой.
    fn place(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
        let p = self.pad_texels();
        let fw = w + 2 * p;
        let fh = h + 2 * p;
        // Регион с гуттером крупнее атласа не влезет ни при каком раскладе
        // (и не должен «уезжать» за границы текстуры при записи).
        if fw > self.width || fh > self.height {
            return None;
        }
        if self.next_x + fw > self.width {
            self.next_x = 0;
            self.next_y += self.row_height;
            self.row_height = 0;
        }
        if self.next_y + fh > self.height {
            return None;
        }
        let pos = (self.next_x + p, self.next_y + p);
        self.next_x += fw;
        self.row_height = self.row_height.max(fh);
        Some(pos)
    }

    /// Пакетная загрузка ассетов: все `*.png` из папки `dir`.
    ///
    /// Файлы сортируются по имени (детерминированный порядок загрузки); имя
    /// региона = имя файла без расширения `.png`. Дубликаты имён (например,
    /// `a.png` и `a.PNG`) — ошибка. Возвращает число загруженных спрайтов.
    pub fn load_folder(&mut self, queue: &wgpu::Queue, dir: &str) -> Result<usize, String> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("не удалось прочитать папку '{dir}': {e}"))?;
        let mut files: Vec<(String, String)> = Vec::new(); // (имя региона, путь)
        for entry in entries {
            let entry =
                entry.map_err(|e| format!("не удалось прочитать элемент папки '{dir}': {e}"))?;
            let path = entry.path();
            let is_png = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("png"));
            if !is_png {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            files.push((stem.to_string(), path.to_string_lossy().into_owned()));
        }
        files.sort_by(|a, b| a.0.cmp(&b.0));
        for pair in files.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(format!(
                    "папка '{dir}': дубликат имени региона '{}'",
                    pair[0].0
                ));
            }
        }
        let mut added = 0;
        for (stem, path) in &files {
            self.add_png(queue, stem, path)?;
            added += 1;
        }
        Ok(added)
    }
}

/// Пересоздаёт текстуру атласа `new_size`×`new_size` и заново укладывает все
/// регионы в порядке добавления ([`SpriteAtlas`]::`order`) из сохранённых
/// пикселей — детерминированный результат независимо от итерации HashMap.
/// Сэмплер и bind group пересоздаются для новой текстуры. Регионы без
/// сохранённых пикселей (зарегистрированные вручную через
/// [`SpriteAtlas::insert_region`]) при росте не переупаковываются.
impl SpriteAtlas {
    /// Пересоздаёт текстуру атласа `new_size`×`new_size` и заново укладывает
    /// все регионы в порядке добавления ([`SpriteAtlas`]::`order`) из
    /// сохранённых пикселей — детерминированный результат независимо от
    /// итерации HashMap. Сэмплер и bind group пересоздаются для новой
    /// текстуры. Регионы без сохранённых пикселей (зарегистрированные вручную
    /// через [`SpriteAtlas::insert_region`]) при росте не переупаковываются.
    fn grow(&mut self, queue: &wgpu::Queue, new_size: u32) -> Result<(), String> {
        let Some(device) = &self.device else {
            return Err(
                "атлас не может расти: устройство не сохранено (атлас собран из готовой текстуры)"
                    .to_string(),
            );
        };
        // Мип-цепь вычисляется от нового размера — только если она включена.
        let levels = if self.mip_levels > 1 {
            level_count(new_size)
        } else {
            1
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dioma-render sprite atlas"),
            size: wgpu::Extent3d {
                width: new_size,
                height: new_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let (sampler, bind_group) = make_sampler_and_bind_group(device, &texture);

        self.texture = texture;
        self.sampler = sampler;
        self.bind_group = bind_group;
        self.width = new_size;
        self.height = new_size;
        self.next_x = 0;
        self.next_y = 0;
        self.row_height = 0;

        let order: Vec<String> = self.order.clone();
        self.regions.clear();
        for name in order {
            let Some((w, h, px)) = self.pixels.get(&name).cloned() else {
                continue;
            };
            let p = self.pad_texels();
            let (x, y) = self.place(w, h).expect("после роста регион обязан влезть");
            upload_region(
                queue,
                &self.texture,
                (x - p, y - p),
                w,
                h,
                &px,
                p,
                self.mip_levels,
            );
            let uv = [
                x as f32 / self.width as f32,
                y as f32 / self.height as f32,
                (x + w) as f32 / self.width as f32,
                (y + h) as f32 / self.height as f32,
            ];
            self.regions.insert(name, uv);
        }
        Ok(())
    }
}

/// Пишет RGBA8-блок `w`×`h` на уровень `mip` текстуры атласа со смещением
/// `(x, y)`; буфер обязан быть ровно `w·h·4` байт.
///
/// `write_texture` требует bytes_per_row, кратный 256 (COPY_BYTES_PER_ROW_ALIGNMENT) —
/// подкладываем строки до ближайшего кратного, если нужно.
#[allow(clippy::too_many_arguments)] // 8 параметров: адресуемый блок + данные
fn write_block(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    pixels: &[u8],
) {
    debug_assert_eq!(pixels.len(), (w as usize) * (h as usize) * 4);
    let row_pitch = align_256(w * 4);
    let extent = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let copy_view = wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: mip,
        origin: wgpu::Origin3d { x, y, z: 0 },
        aspect: wgpu::TextureAspect::All,
    };
    if row_pitch == w * 4 {
        queue.write_texture(
            copy_view,
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_pitch),
                rows_per_image: Some(h),
            },
            extent,
        );
    } else {
        let mut padded = vec![0u8; row_pitch as usize * h as usize];
        for row in 0..h {
            let src =
                &pixels[(row as usize) * (w as usize) * 4..(row as usize + 1) * (w as usize) * 4];
            let dst = &mut padded[(row as usize) * row_pitch as usize
                ..(row as usize) * row_pitch as usize + (w as usize) * 4];
            dst.copy_from_slice(src);
        }
        queue.write_texture(
            copy_view,
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_pitch),
                rows_per_image: Some(h),
            },
            extent,
        );
    }
}

/// Заливает регион интерьерными пикселями (`w × h`) в ячейку с верхним левым
/// углом `(cx, cy)` — сюда входит и гуттер `pad` texels по каждой стороне:
/// уровень 0 пишется как регион+гуттер (края реплицируются), уровни выше
/// получаются box-filter усреднением предыдущего уровня (тоже со своим
/// подтянутым рамками краем). Без гуттера (`pad = 0`) блок пишется 1:1.
#[allow(clippy::too_many_arguments)] // 8 параметров: ячейка + данные + конфиг мипов
fn upload_region(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    cell: (u32, u32),
    w: u32,
    h: u32,
    pixels: &[u8],
    pad: u32,
    levels: u32,
) {
    debug_assert_eq!(pixels.len(), (w as usize) * (h as usize) * 4);
    let level0 = if pad == 0 {
        pixels.to_vec()
    } else {
        replicate_pad(w, h, pad, pixels)
    };
    let pw = w + 2 * pad;
    let ph = h + 2 * pad;
    write_block(queue, texture, 0, cell.0, cell.1, pw, ph, &level0);

    let mut cur = level0;
    let mut cw = pw;
    let mut ch = ph;
    for lvl in 1..levels as usize {
        let next = downsample(cw, ch, &cur);
        let nw = (cw / 2).max(1);
        let nh = (ch / 2).max(1);
        // Подпрямоугольник этой ячейки на уровне l — целочисленный сдвиг ячейки.
        write_block(
            queue,
            texture,
            lvl as u32,
            cell.0 >> lvl,
            cell.1 >> lvl,
            nw,
            nh,
            &next,
        );
        cur = next;
        cw = nw;
        ch = nh;
    }
}

/// Расширяет RGBA8-изображение `w×h` рамкой из реплицированных краёв шириной
/// `pad`: по контуру копируются ближайшие пиксели региона — усреднение мипов
/// не подтягивает цвета соседей из атласа.
fn replicate_pad(w: u32, h: u32, pad: u32, pixels: &[u8]) -> Vec<u8> {
    let pw = (w + 2 * pad) as usize;
    let ph = (h + 2 * pad) as usize;
    let p = pad as i64;
    let wi = w as i64;
    let hi = h as i64;
    let mut out = vec![0u8; pw * ph * 4];
    for py in 0..ph as i64 {
        for px in 0..pw as i64 {
            // Кламп индексов к области региона.
            let sx = (px - p).clamp(0, wi - 1);
            let sy = (py - p).clamp(0, hi - 1);
            let src = ((sy * wi + sx) * 4) as usize;
            let dst = ((py * pw as i64 + px) * 4) as usize;
            out[dst..dst + 4].copy_from_slice(&pixels[src..src + 4]);
        }
    }
    out
}

/// Box-filter усреднение RGBA8-изображения `w×h` вдвое (нечётные края клампятся:
/// крайний ткмль суммируется один раз). Возвращает изображение `(w/2 max 1, h/2 max 1)`.
fn downsample(w: u32, h: u32, data: &[u8]) -> Vec<u8> {
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for dy in 0..nh {
        for dx in 0..nw {
            let mut acc = [0u32; 4];
            for oy in 0..2u32 {
                for ox in 0..2u32 {
                    let sx = (dx * 2 + ox).min(w - 1);
                    let sy = (dy * 2 + oy).min(h - 1);
                    let src = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += data[src + c] as u32;
                    }
                }
            }
            let dst = ((dy * nw + dx) * 4) as usize;
            for c in 0..4 {
                out[dst + c] = (acc[c] / 4) as u8;
            }
        }
    }
    out
}

/// Число мип-уровней для стороны pow2 размера `size`: log2(size)+1,
/// ограниченное [`MIP_MAX_LEVELS`] снизу единицей.
fn level_count(size: u32) -> u32 {
    if size == 0 {
        return 1;
    }
    ((31 - size.leading_zeros()) + 1).clamp(1, MIP_MAX_LEVELS)
}

/// Приводит декодированный кадр к RGBA8: добавляет недостающие каналы
/// (rgb → альфа 255, gray → серый + альфа 255, gray+alpha → серый×3 + альфа).
/// С `Transformations::EXPAND` кадр приходит в исходном числе каналов —
/// здесь он нормализуется до 4.
fn to_rgba8(color_type: png::ColorType, data: &[u8]) -> Vec<u8> {
    use png::ColorType::*;
    match color_type {
        Rgba => data.to_vec(),
        Rgb => data
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        GrayscaleAlpha => data
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[0], c[0], c[1]])
            .collect(),
        Grayscale | Indexed => data.iter().flat_map(|&g| [g, g, g, 255]).collect(),
    }
}

/// Обрезает полностью прозрачную рамку по краям `rgba` (изображение `w`×`h`).
///
/// Возвращает `(новая ширина, новая высота, смещение [left, top] в пикселях,
/// обрезанные пиксели)`. Если изображение полностью прозрачно — возвращает
/// `(1, 1, [0, 0], 1×1 прозрачный пиксель)`.
fn trim_rgba(w: u32, h: u32, rgba: &[u8]) -> (u32, u32, [f32; 2], Vec<u8>) {
    let mut left = w;
    let mut top = h;
    let mut right = 0u32;
    let mut bottom = 0u32;
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            if rgba[((y * w + x) * 4 + 3) as usize] > 0 {
                any = true;
                left = left.min(x);
                top = top.min(y);
                right = right.max(x);
                bottom = bottom.max(y);
            }
        }
    }
    if !any {
        return (1, 1, [0.0, 0.0], vec![0u8; 4]);
    }
    let nw = right - left + 1;
    let nh = bottom - top + 1;
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        let src = ((top + y) * w + left) as usize * 4;
        let dst = &mut out[(y * nw * 4) as usize..((y + 1) * nw * 4) as usize];
        dst.copy_from_slice(&rgba[src..src + (nw * 4) as usize]);
    }
    (nw, nh, [left as f32, top as f32], out)
}

/// Сохраняет RGBA8-буфер в PNG-файл (8-бит, RGBA). Полезно для фикстур,
/// скриншотов и экспорта спрайтов.
pub fn save_png(path: &str, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    if rgba.len() != (width as usize) * (height as usize) * 4 {
        return Err(format!(
            "PNG '{path}': размер данных {} ≠ {width}×{height}×4",
            rgba.len()
        ));
    }
    let file = std::fs::File::create(path)
        .map_err(|e| format!("PNG '{path}': не удалось создать файл: {e}"))?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("PNG '{path}': ошибка заголовка: {e}"))?;
    writer
        .write_image_data(rgba)
        .map_err(|e| format!("PNG '{path}': ошибка записи: {e}"))?;
    Ok(())
}

/// Общие сэмплер и bind group атласа (текстура + сэмплер) для шейдерной
/// группы 1. Используется и [`SpriteAtlas::new`], и [`SpriteAtlas::from_texture`].
fn make_sampler_and_bind_group(
    device: &wgpu::Device,
    texture: &wgpu::Texture,
) -> (wgpu::Sampler, wgpu::BindGroup) {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("dioma-render atlas sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let layout = atlas_bind_group_layout(device);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dioma-render atlas bind group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    (sampler, bind_group)
}

/// Bind group layout для группы 1 (атлас: текстура + сэмплер).
///
/// Используется и атласом, и рендерером; одинаковые описания wgpu
/// дедуплицирует в один объект, поэтому bind group атласа совместим
/// с пайплайном рендерера.
pub(crate) fn atlas_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dioma-render atlas bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

// ── Процедурные генераторы (чистый CPU, без GPU) ────────────────────────────

/// Антиалиашенный круг. Возвращает `(ширина, высота, RGBA8-пиксели)`.
///
/// Размер текстуры — `2·radius + 2` пикселя (запас на сглаженный край);
/// альфа спадает по расстоянию от края (плавный falloff ~1px).
pub fn circle_rgba(radius_px: u32, color: [u8; 4]) -> (u32, u32, Vec<u8>) {
    let size = radius_px * 2 + 2;
    let mut pixels = vec![0u8; (size * size * 4) as usize];
    let center = (radius_px as f32 + 1.0, radius_px as f32 + 1.0);
    let r = radius_px as f32;
    for y in 0..size {
        for x in 0..size {
            let d = dist(x, y, center);
            // 1px плавный переход: альфа 1 внутри r-0.5, 0 снаружи r+0.5.
            let cov = (r - d + 0.5).clamp(0.0, 1.0);
            if cov > 0.0 {
                let i = ((y * size + x) * 4) as usize;
                pixels[i] = color[0];
                pixels[i + 1] = color[1];
                pixels[i + 2] = color[2];
                pixels[i + 3] = (color[3] as f32 * cov).round() as u8;
            }
        }
    }
    (size, size, pixels)
}

/// Кольцо с толщиной `thickness_px`. Размер текстуры — `2·radius + 2`.
pub fn ring_rgba(radius_px: u32, thickness_px: f32, color: [u8; 4]) -> (u32, u32, Vec<u8>) {
    let size = radius_px * 2 + 2;
    let mut pixels = vec![0u8; (size * size * 4) as usize];
    let center = (radius_px as f32 + 1.0, radius_px as f32 + 1.0);
    let r_out = radius_px as f32;
    let r_in = (r_out - thickness_px).max(0.0);
    for y in 0..size {
        for x in 0..size {
            let d = dist(x, y, center);
            let outer = (r_out - d + 0.5).clamp(0.0, 1.0); // внутри внешнего края
            let inner = (r_in - d + 0.5).clamp(0.0, 1.0); // внутри внутреннего края
            let cov = outer * (1.0 - inner);
            if cov > 0.0 {
                let i = ((y * size + x) * 4) as usize;
                pixels[i] = color[0];
                pixels[i + 1] = color[1];
                pixels[i + 2] = color[2];
                pixels[i + 3] = (color[3] as f32 * cov).round() as u8;
            }
        }
    }
    (size, size, pixels)
}

/// Ягода 64×64 с бликом, позиция которого зависит от `phase`.
///
/// `phase` вращает блик вокруг центра (как свет на глянцевой поверхности);
/// тело ягоды слегка затемняется к краю радиальным градиентом.
pub fn berry_rgba(phase: f32, color: [u8; 4]) -> (u32, u32, Vec<u8>) {
    const SIZE: u32 = 64;
    let mut pixels = vec![0u8; (SIZE * SIZE * 4) as usize];
    let center = (SIZE as f32 * 0.5, SIZE as f32 * 0.5);
    let body_r = 28.0;
    let highlight = (center.0 + 14.0 * phase.cos(), center.1 + 14.0 * phase.sin());
    let highlight_r = 6.0;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let d = dist(x, y, center);
            let body_cov = (body_r - d + 0.5).clamp(0.0, 1.0);
            if body_cov <= 0.0 {
                continue;
            }
            let shade = 0.72 + 0.28 * (1.0 - d / body_r).clamp(0.0, 1.0);
            let i = ((y * SIZE + x) * 4) as usize;
            pixels[i] = (color[0] as f32 * shade) as u8;
            pixels[i + 1] = (color[1] as f32 * shade) as u8;
            pixels[i + 2] = (color[2] as f32 * shade) as u8;
            pixels[i + 3] = (color[3] as f32 * body_cov) as u8;

            // Мягкий белый блик поверх тела.
            let dh = dist(x, y, highlight);
            let hl = (highlight_r - dh + 0.5).clamp(0.0, 1.0);
            if hl > 0.0 {
                for ch in 0..3 {
                    let c = pixels[i + ch] as f32;
                    pixels[i + ch] = (c + (255.0 - c) * hl) as u8;
                }
            }
        }
    }
    (SIZE, SIZE, pixels)
}

/// Горизонтальная полоса состояния (status bar): заливка слева, прогресс 0..1.
///
/// 1px рамка по краю — цветом `background`. Возвращает `(ширина, высота, RGBA8)`.
pub fn bar_rgba(
    width: u32,
    height: u32,
    fill: [u8; 4],
    background: [u8; 4],
    progress: f32,
) -> (u32, u32, Vec<u8>) {
    let w = width.max(2);
    let h = height.max(2);
    let p = progress.clamp(0.0, 1.0);
    // Граница заливки внутри рамки: от x=1 до 1 + (w−2)·p.
    let fill_edge = 1.0 + (w - 2) as f32 * p;
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let border = x == 0 || y == 0 || x == w - 1 || y == h - 1;
            let c = if border || (x as f32) >= fill_edge {
                background
            } else {
                fill
            };
            let i = ((y * w + x) * 4) as usize;
            pixels[i..i + 4].copy_from_slice(&c);
        }
    }
    (w, h, pixels)
}

/// Мягкий круг с плавным квадратичным falloff (мягче, чем [`circle_rgba`]).
///
/// Для свечений, туманов и текстур частиц, если понадобятся текстурированные
/// варианты. Возвращает `(ширина, высота, RGBA8)`.
pub fn soft_circle_rgba(radius_px: u32, color: [u8; 4]) -> (u32, u32, Vec<u8>) {
    let size = radius_px * 2 + 4;
    let mut pixels = vec![0u8; (size * size * 4) as usize];
    let center = (radius_px as f32 + 2.0, radius_px as f32 + 2.0);
    let r = radius_px as f32;
    for y in 0..size {
        for x in 0..size {
            let d = dist(x, y, center);
            if d >= r {
                continue;
            }
            // Квадратичный falloff от центра к краю (полная прозрачность на r).
            let cov = 1.0 - d / r;
            let a = cov * cov;
            let i = ((y * size + x) * 4) as usize;
            pixels[i] = color[0];
            pixels[i + 1] = color[1];
            pixels[i + 2] = color[2];
            pixels[i + 3] = (color[3] as f32 * a).round() as u8;
        }
    }
    (size, size, pixels)
}

/// Толстая линия `(x0, y0)→(x1, y1)` в прозрачном буфере `w`×`h`.
///
/// Псевдо-антиалиас: покрытие пикселя считается по расстоянию от его центра
/// до отрезка — плавный переход шириной ~1px вокруг краёв линии
/// (`clamp(thickness/2 − dist + 0.5, 0, 1)`), так что наклонные линии
/// выглядят мягче ступенек. Возвращает `(ширина, высота, RGBA8)`.
pub fn line_rgba(
    w: u32,
    h: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    thickness: f32,
    color: [u8; 4],
) -> (u32, u32, Vec<u8>) {
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    let half = thickness.max(0.0) * 0.5;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len2 = dx * dx + dy * dy;
    for y in 0..h {
        for x in 0..w {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let d = if len2 <= f32::EPSILON {
                // Вырожденный «отрезок» — точка.
                ((px - x0).powi(2) + (py - y0).powi(2)).sqrt()
            } else {
                // Проекция точки на отрезок, клампнутая к его концам.
                let t = (((px - x0) * dx + (py - y0) * dy) / len2).clamp(0.0, 1.0);
                let cx = x0 + t * dx;
                let cy = y0 + t * dy;
                ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
            };
            let cov = (half - d + 0.5).clamp(0.0, 1.0);
            if cov > 0.0 {
                let i = ((y * w + x) * 4) as usize;
                pixels[i] = color[0];
                pixels[i + 1] = color[1];
                pixels[i + 2] = color[2];
                pixels[i + 3] = (color[3] as f32 * cov).round() as u8;
            }
        }
    }
    (w, h, pixels)
}

/// Расстояние от центра пикселя `(x, y)` до точки `c` (в пиксельных единицах).
fn dist(x: u32, y: u32, c: (f32, f32)) -> f32 {
    let dx = x as f32 + 0.5 - c.0;
    let dy = y as f32 + 0.5 - c.1;
    (dx * dx + dy * dy).sqrt()
}

/// Округление вверх до кратного 256 (COPY_BYTES_PER_ROW_ALIGNMENT).
fn align_256(n: u32) -> u32 {
    ((n + 255) / 256) * 256
}
