//! Проверка компиляции ВСЕХ автоген WGSL-шейдеров (после фиксов транспайлера).
//! Каждый, кто не парсится, — недоделанный фикс в dual.rs.
use pillars_wgpu::WgpuContext;

fn main() {
    let ctx = match WgpuContext::new_blocking() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("NO_ADAPTER: {e}");
            return;
        }
    };
    println!("ADAPTER: {} ({:?})", ctx.adapter_name, ctx.device_type);

    // имя кернела: {fn_suffix}_kernel_wgsl (см. dual.rs: fn_suffix = тип без "Model", to_lowercase)
    let models: &[(&str, &str, &str)] = &[
        ("ADEX", "adex_kernel_wgsl", pillars_wgpu::ADEX_WGSL_SOURCE),
        ("ALPHA_LIF", "alphalif_kernel_wgsl", pillars_wgpu::ALPHA_LIF_WGSL_SOURCE),
        ("LIF", "lif_kernel_wgsl", pillars_wgpu::LIF_WGSL_SOURCE),
        ("IZHIKEVICH", "izhikevich_kernel_wgsl", pillars_wgpu::IZHIKEVICH_WGSL_SOURCE),
        ("IZHIKEVICH_2007", "izhikevich2007_kernel_wgsl", pillars_wgpu::IZHIKEVICH_2007_WGSL_SOURCE),
        ("LIFAT", "lifat_kernel_wgsl", pillars_wgpu::LIFAT_WGSL_SOURCE),
        ("RATE", "rate_kernel_wgsl", pillars_wgpu::RATE_WGSL_SOURCE),
        ("SAW", "saw_kernel_wgsl", pillars_wgpu::SAW_WGSL_SOURCE),
        ("HAAP", "haap_kernel_wgsl", pillars_wgpu::HAAP_WGSL_SOURCE),
        ("HR", "hr_kernel_wgsl", pillars_wgpu::HR_WGSL_SOURCE),
        ("FHN", "fhn_kernel_wgsl", pillars_wgpu::FHN_WGSL_SOURCE),
        ("GLIF3", "glif3_kernel_wgsl", pillars_wgpu::GLIF3_WGSL_SOURCE),
        ("GLIF4", "glif4_kernel_wgsl", pillars_wgpu::GLIF4_WGSL_SOURCE),
    ];

    let mut ok = 0;
    let mut bad = 0;
    for (name, entry, src) in models {
        // wgpu 22: ошибки валидации — через error scope, не Result
        ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl((*src).into()),
        });
        let _ = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &shader,
            entry_point: &entry,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let err = pollster::block_on(ctx.device.pop_error_scope());
        match err {
            None => {
                ok += 1;
                println!("OK   {name} ({entry})");
            }
            Some(e) => {
                bad += 1;
                let msg = e.to_string();
                let line = msg
                    .lines()
                    .find(|l| l.trim_start().starts_with('┌') || l.contains("wgsl:"))
                    .unwrap_or(&msg);
                println!("FAIL {name}: {line}");
            }
        }
    }
    println!("RESULT: {ok} OK, {bad} FAIL");
}
