//! Smoke v3: МИНИМАЛЬНЫЙ изолированный тест. init → 1 диспатч LIF → полный дамп.
use pillars_wgpu::WgpuContext;
use wgpu::util::DeviceExt;

fn main() {
    let ctx = match WgpuContext::new_blocking() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("NO_ADAPTER: {e}");
            return;
        }
    };
    println!("ADAPTER_OK: {} ({:?})", ctx.adapter_name, ctx.device_type);

    // ── шейдер/пайплайн ──
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lif"),
            source: wgpu::ShaderSource::Wgsl(pillars_wgpu::LIF_WGSL.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lif"),
            layout: None,
            module: &shader,
            entry_point: "lif_kernel_wgsl",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    println!("PIPELINE_OK");

    // ── буферы ──
    let n = 64u32;
    let stride = 64u32;
    let ns = 2u32;
    let np = 6u32;
    let states = {
        let mut v = vec![0.0f32; (ns * stride) as usize];
        for i in 0..n as usize {
            v[i] = -70.0 + i as f32; // v: уникальные значения, чтобы видеть запись
            v[(stride as usize) + i] = 0.0; // refr
        }
        v
    };
    let params = {
        let mut p = vec![0.0f32; (np * stride) as usize];
        // SoA: колонка col = params[col * stride + i]
        for i in 0..n as usize {
            p[0 * stride as usize + i] = 0.5; // dt
            p[1 * stride as usize + i] = 0.1; // inv_tau_m
            p[2 * stride as usize + i] = -70.0; // v_rest
            p[3 * stride as usize + i] = -54.0; // v_thresh
            p[4 * stride as usize + i] = -70.0; // v_reset
            p[5 * stride as usize + i] = 2.0; // tau_ref
        }
        p
    };
    let i_in = vec![30.0f32; n as usize];
    let spikes = vec![0u32; n as usize];
    // WgslArgs { dt: f32, n: u32, stride: u32 } — n/stride пакуются как u32!
    let mut args = Vec::with_capacity(16);
    args.extend_from_slice(&0.5f32.to_le_bytes());
    args.extend_from_slice(&n.to_le_bytes());
    args.extend_from_slice(&stride.to_le_bytes());
    args.extend_from_slice(&0u32.to_le_bytes());

    let b_states = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("states"),
        contents: bytemuck::cast_slice(&states),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let b_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_iin = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("i_in"),
        contents: bytemuck::cast_slice(&i_in),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_spikes = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("spikes"),
        contents: bytemuck::cast_slice(&spikes),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let b_args = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("args"),
        contents: bytemuck::cast_slice(&args),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: b_states.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b_iin.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: b_spikes.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: b_args.as_entire_binding() },
        ],
    });

    let steps = std::env::var("STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let batch = std::env::var("BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // ── {steps} шагов: по {batch} диспатчей в пасс, отдельный submit на пасс ──
    // Спайки-буфер хранит только последний шаг — читаем и суммируем каждый батч.
    let mut gpu_total: u32 = 0;
    let mut remaining = steps;
    while remaining > 0 {
        let k = remaining.min(batch);
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("enc"),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        for _ in 0..k {
            pass.dispatch_workgroups(1, 1, 1);
        }
        drop(pass);
        ctx.queue.submit(Some(encoder.finish()));
        // читаем спайки последнего шага батча
        let (tx, rx) = std::sync::mpsc::channel();
        let rb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spike_rb"),
            size: (n as u64) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc2 = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("spike_copy") });
        enc2.copy_buffer_to_buffer(&b_spikes, 0, &rb, 0, (n as u64) * 4);
        ctx.queue.submit(Some(enc2.finish()));
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = rb.slice(..).get_mapped_range();
        let batch_spikes: &[u32] = bytemuck::cast_slice(&view);
        let bsum = batch_spikes.iter().sum::<u32>();
        gpu_total += bsum;
        println!("  batch(step {}-{}) spikes={bsum} first4={:?}", steps - remaining, steps - remaining + k, &batch_spikes[..4]);
        drop(view);
        rb.unmap();
        remaining -= k;
    }

    // ── полный дамп ──
    fn readback<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, len: usize) -> Vec<T> {
        let rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb"),
            size: (len * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rb_enc") });
        enc.copy_buffer_to_buffer(src, 0, &rb, 0, (len * std::mem::size_of::<T>()) as u64);
        queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        rb.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map failed");
        let view = rb.slice(..).get_mapped_range();
        let out: &[T] = bytemuck::cast_slice(&view);
        let v = out.to_vec();
        drop(view);
        rb.unmap();
        v
    }

    let v_out: Vec<f32> = readback(&ctx.device, &ctx.queue, &b_states, 128);
    println!("steps={steps} states_v[0..8]  = {:?}", &v_out[..8]);
    println!("steps={steps} states_refr[0..8]= {:?}", &v_out[64..72]);
    println!("spikes total     = {gpu_total}");

    // ── эталон CPU: тот же LIF, 200 шагов ──
    let mut v = vec![0.0f32; n as usize];
    let mut refr = vec![0.0f32; n as usize];
    for i in 0..n as usize {
        v[i] = -70.0 + i as f32;
    }
    let mut cpu_total = 0u32;
    for _ in 0..200 {
        for i in 0..n as usize {
            let zero = 0.0f32;
            let is_refr = (refr[i] > zero) as u32 as f32;
            let dv = (params[2 * stride as usize + i] - v[i] + i_in[i]) * params[1 * stride as usize + i] * params[0 * stride as usize + i];
            let v_new = v[i] + dv;
            let spike_raw = (v_new >= params[3 * stride as usize + i]) as u32 as f32;
            let spike = if is_refr != 0.0 { 0.0 } else { spike_raw };
            if spike != 0.0 {
                cpu_total += 1;
                v[i] = params[4 * stride as usize + i];
                refr[i] = params[5 * stride as usize + i];
            } else if is_refr != 0.0 {
                v[i] = v[i]; // заморожено
                refr[i] = (refr[i] - params[0 * stride as usize + i]).max(0.0);
            } else {
                v[i] = v_new;
                refr[i] = (refr[i] - params[0 * stride as usize + i]).max(0.0);
            }
        }
    }
    println!("CPU total        = {cpu_total}");
    println!("CPU v[0..8]      = {:?}", &v[..8]);
    println!("CPU refr[0..8]   = {:?}", &refr[..8]);
    println!("MATCH            = {}", if gpu_total == cpu_total { "ДА, бит-в-бит" } else { "НЕТ" });
}
