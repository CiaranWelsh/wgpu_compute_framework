// examples/vadf_stream.rs
use std::borrow::Cow;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use rand::{rngs::StdRng, Rng, SeedableRng};
use wgpu::util::DeviceExt;
use wgpu_compute_framework::{GpuContext, GpuBuffer}; // from your crate

// ---------------------- Data types ----------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
struct Hit {
    x: u32,
    y: u32,
    tot: u32, // units: 40MHz ticks (25 ns); we just accumulate whatever you feed us
    toa: u32, // units: 640MHz ticks; monotonic across stream
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
struct Params {
    t0_toa: u32,
    dwell_ticks: u32,
    n_scan: u32,
    det_w: u32,
    det_h: u32,
    cx: f32,
    cy: f32,
    r1_sq: f32,
    r2_sq: f32,
}

// ---------------------- Simulation ----------------------

struct SimCfg {
    scan_w: u32,
    scan_h: u32,
    dwell_ticks: u32, // in ToA ticks (640MHz units); integer-safe
    det_w: u32,
    det_h: u32,
    mean_hits_per_pos: u32,
    chunk_positions: u32, // approx "line length" in positions per chunk
    seed: u64,
}

// Generate a monotonically increasing stream of hits sorted by ToA.
// We emit them in chunks of positions (mimics line-by-line).
fn simulate_stream(cfg: &SimCfg) -> impl Iterator<Item = Vec<Hit>> + '_ {
    let total_positions = cfg.scan_w * cfg.scan_h;
    let mut pos = 0u32;
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    std::iter::from_fn(move || {
        if pos >= total_positions {
            return None;
        }
        let start_pos = pos;
        let end_pos = (pos + cfg.chunk_positions).min(total_positions);
        let mut chunk = Vec::new();

        for idx in start_pos..end_pos {
            let toa_start = cfg.dwell_ticks.saturating_mul(idx); // t0 assumed 0 for simplicity; set below
            // Poisson-ish variability without pulling in a full Poisson sampler
            let hits_here = (cfg.mean_hits_per_pos as i32
                + rng.gen_range(-((cfg.mean_hits_per_pos as i32) / 2)..=((cfg.mean_hits_per_pos as i32) / 2)))
                .max(0) as u32;

            for _ in 0..hits_here {
                // Spread ToA uniformly over the dwell window for the position
                let toa = toa_start + rng.gen_range(0..cfg.dwell_ticks.max(1));

                // Random pixel within detector
                let x = rng.gen_range(0..cfg.det_w);
                let y = rng.gen_range(0..cfg.det_h);

                // ToT with a mild center bias (toy model)
                let tot = 1 + ((cfg.det_w as i32 / 2 - x as i32).abs()
                    + (cfg.det_h as i32 / 2 - y as i32).abs()) as u32 % 8;

                chunk.push(Hit { x, y, tot, toa });
            }
        }

        // Sort by ToA to respect your pre-sorted invariant
        chunk.sort_unstable_by_key(|h| h.toa);

        pos = end_pos;
        Some(chunk)
    })
}

// ---------------------- GPU vADF (streaming) ----------------------

fn create_pipeline_and_layout(
    device: &wgpu::Device,
    shader_src: &str,
) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vADF Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_src)),
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vADF BGL"),
        entries: &[
            // hits
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::ReadOnlyStorage,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // out_sums (atomics)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // params (uniform)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vADF Pipeline Layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vADF Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (layout, pipeline)
}

fn dispatch_chunk(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    pipeline: &wgpu::ComputePipeline,
    hits: &[Hit],
    out_sums_buf: &wgpu::Buffer,
    params_buf: &wgpu::Buffer,
) {
    // Upload chunk to a transient storage buffer
    let hits_size = (hits.len() * std::mem::size_of::<Hit>()) as wgpu::BufferAddress;
    let hits_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hits chunk"),
        contents: bytemuck::cast_slice(hits),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vADF BG"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: hits_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_sums_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("vADF encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vADF pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // 1D dispatch: one thread per hit, rounded up by workgroup size (256)
        let wg_size = 256u32;
        let n_wg = ((hits.len() as u32) + wg_size - 1) / wg_size;
        if n_wg > 0 {
            cpass.dispatch_workgroups(n_wg, 1, 1);
        }
    }
    queue.submit(Some(encoder.finish()));
    // Block until done (simple & predictable)
    device.poll(wgpu::Maintain::Wait);
}

fn main() {
    env_logger::init();

    // ---- Config ----
    let sim = SimCfg {
        scan_w: 128,
        scan_h: 128,
        dwell_ticks: 800,   // e.g. 800 ticks @ 640MHz ≈ 1.25 us if 1 tick = 1.5625 ns
        det_w: 256,
        det_h: 256,
        mean_hits_per_pos: 64,
        chunk_positions: 128, // simulate "line" granularity
        seed: 42,
    };

    let n_scan = sim.scan_w * sim.scan_h;

    // ---- GPU context ----
    // Your framework: headless init + simple, synchronous behavior
    let context = GpuContext::new_blocking().expect("No suitable GPU adapter"); // :contentReference[oaicite:3]{index=3}
    let device = &context.device;
    let queue = &context.queue;

    // ---- Load shader ----
    let shader_src = include_str!("vadf.wgsl");
    let (bgl, pipeline) = create_pipeline_and_layout(device, shader_src);

    // ---- Persistent output buffer (atomics need STORAGE | COPY_SRC | COPY_DST) ----
    let out_bytes = (n_scan as usize * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
    let out_sums_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vADF out_sums"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Zero it once
    {
        // A tiny zero staging buffer
        let zeros = vec![0u8; out_bytes as usize];
        let zero_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zeros"),
            contents: &zeros,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero out") });
        enc.copy_buffer_to_buffer(&zero_buf, 0, &out_sums_buf, 0, out_bytes);
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // ---- Uniforms ----
    let params = Params {
        t0_toa: 0,                         // stream ToAs start at 0 in our simulator
        dwell_ticks: sim.dwell_ticks,
        n_scan,
        det_w: sim.det_w,
        det_h: sim.det_h,
        cx: (sim.det_w as f32) * 0.5,
        cy: (sim.det_h as f32) * 0.5,
        r1_sq: 20.0 * 20.0,
        r2_sq: 60.0 * 60.0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // ---- Stream & dispatch ----
    let start = Instant::now();
    let mut total_hits: u64 = 0;
    let mut total_dispatches: u32 = 0;

    for chunk in simulate_stream(&sim) {
        total_hits += chunk.len() as u64;
        dispatch_chunk(device, queue, &bgl, &pipeline, &chunk, &out_sums_buf, &params_buf);
        total_dispatches += 1;
    }
    let gpu_time = start.elapsed();

    // ---- Read back results ----
    let readback = GpuBuffer::<u32>::readback_blocking(device, queue, &out_sums_buf, n_scan as usize)
        .expect("readback failed"); // uses your typed-buffer helper to map/copy

    // ---- Report ----
    println!("vADF (sum of ToT) length = {}", readback.len());
    println!("Total hits processed: {}", total_hits);
    println!("Dispatches (chunks): {}", total_dispatches);
    println!("GPU wall time: {:?}  (throughput ~{:.2} Mhits/s)",
             gpu_time,
             (total_hits as f64) / (gpu_time.as_secs_f64() * 1e6)
    );

    // Example: print first few scan positions
    for i in 0..std::cmp::min(10, readback.len()) {
        println!("pos[{i}] = {}", readback[i]);
    }
}
