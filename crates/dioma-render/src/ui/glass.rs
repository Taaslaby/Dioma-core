//! Задел под «стекло» UI: текстура под размытую сцену.

use crate::ui::UiSystem;

/// Offscreen-текстура, в которую `PostProcess::apply(PostFx::Blur { .. })`
/// пишет размытую сцену для стеклянных панелей ([`UiSystem::glass_panel`] +
/// `Renderer::render_ui_frost` + `Renderer::set_ui_backdrop`).
///
/// Порядок в кадре:
/// ```text
/// renderer.render(.., post.scene_view(), ..);              // сцена
/// renderer.render_lights(.., post.scene_view(), ..);
/// post.apply(queue, enc, backdrop.view(), Blur{..}, t);    // блюр сцены
/// renderer.set_ui_backdrop(&device, backdrop.view());
/// renderer.render_ui_frost(queue, enc, &view, atlas, ui.frost_batch(), vp);
/// post.apply(queue, enc, &view, PostFx::None, t);          // сцена на экран
/// renderer.render_ui(queue, enc, &view, atlas, ui.batch(), vp);
/// ```
pub struct GlassBackdrop {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    size: [u32; 2],
}

impl GlassBackdrop {
    /// Создаёт текстуру `w×h` (RENDER_ATTACHMENT | TEXTURE_BINDING).
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, w: u32, h: u32) -> Self {
        let mut this = Self {
            texture: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("dioma-render glass backdrop"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                ..Self::descriptor(format)
            }),
            view: wgpu::TextureView::default(),
            format,
            size: [1, 1],
        };
        this.resize(device, w, h);
        this
    }

    fn descriptor(format: wgpu::TextureFormat) -> wgpu::TextureDescriptor<'static> {
        wgpu::TextureDescriptor {
            label: Some("dioma-render glass backdrop"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }
    }

    /// Пересоздаёт текстуру под новый размер (Resized).
    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let (w, h) = (w.max(1), h.max(1));
        if self.size == [w, h] {
            return;
        }
        let mut desc = Self::descriptor(self.format);
        desc.size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        self.texture = device.create_texture(&desc);
        self.view = self.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.size = [w, h];
    }

    /// View для `PostProcess::apply(.., backdrop.view(), PostFx::Blur { .. }, ..)`.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}
