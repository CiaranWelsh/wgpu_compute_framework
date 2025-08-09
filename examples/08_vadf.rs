//! Example 8: Virtual Annular Dark-Field (vADF) imaging using wGPU.
//!
//! Uses the existing `wgpu_compute_framework` helpers — no wgpu boilerplate here.
//! We simulate streaming-like 4D-STEM hits, compute vADF on the GPU, and
//! verify one-for-one equality with a CPU reference.

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use wgpu_compute_framework::context::GpuContext;
use wgpu_compute_framework::compute::run_compute_single_input_custom_output;

/// One detector hit; `dx,dy` are integer offsets from detector centre.
/// `tot` is Time-over-Threshold (arbitrary units), `scan_id` is the scan position id.
///
/// We keep this plain and `#[repr(C)]` so it can be copied to a storage buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Hit {
    dx: i32,
    dy: i32,
    tot: u32,
    scan_id: u32,
}

fn main() {
    env_logger::init();
    let ctx = GpuContext::new_blocking().expect("GpuContext");

    // Scan + detector configuration (tweak to taste).
    let scan_rows: u32 = 64;
    let scan_cols: u32 = 64;
    let num_scan_positions = (scan_rows * scan_cols) as usize;

    let detector_size: u32 = 256;
    let centre = (detector_size as f32 - 1.0) / 2.0; // 127.5 for 256
    let r1: f32 = 30.0;
    let r2: f32 = 90.0;
    let r1_sqr: f32 = r1 * r1;
    let r2_sqr: f32 = r2 * r2;

    // Workload size.
    let hits_per_pos: usize = 200;
    let total_hits: usize = num_scan_positions * hits_per_pos;

    // --- Synthesize deterministic hits (already sorted by scan position) ---
    let mut hits: Vec<Hit> = Vec::with_capacity(total_hits);
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    for scan_id in 0..num_scan_positions {
        for _ in 0..hits_per_pos {
            let x = rng.gen_range(0..detector_size);
            let y = rng.gen_range(0..detector_size);
            // Make dx,dy as integer offsets (truncate .5 consistently).
            let dx = (x as f32 - centre) as i32;
            let dy = (y as f32 - centre) as i32;
            hits.push(Hit {
                dx,
                dy,
                tot: rng.gen_range(1..100),
                scan_id: scan_id as u32,
            });
        }
    }
    assert_eq!(hits.len(), total_hits);

    // --- CPU reference (measure + compute) ---
    let cpu_t0 = Instant::now();
    let mut cpu_sum: Vec<u64> = vec![0; num_scan_positions];
    let mut cpu_count: Vec<u64> = vec![0; num_scan_positions];
    for h in &hits {
        let dx = h.dx as f32;
        let dy = h.dy as f32;
        let dist_sqr = dx * dx + dy * dy;
        if dist_sqr >= r1_sqr && dist_sqr < r2_sqr {
            let id = h.scan_id as usize;
            cpu_sum[id] += h.tot as u64;
            cpu_count[id] += 1;
        }
    }
    let cpu_time = cpu_t0.elapsed();

    // --- GPU kernel (framework helper) ---
    let workgroup_size: u32 = 128; // good default, try {64,128,256}
    let shader_source = format!(r#"
struct Hit {{
    dx: i32,
    dy: i32,
    tot: u32,
    scan_id: u32,
}};

@group(0) @binding(0) var<storage, read> hits: array<Hit>;
@group(0) @binding(1) var<storage, read_write> out: array<atomic<u32>>;

const R1_SQR: f32 = {r1_sqr};
const R2_SQR: f32 = {r2_sqr};

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i: u32 = gid.x;
    if (i >= arrayLength(&hits)) {{
        return;
    }}
    let h = hits[i];
    let dx = f32(h.dx);
    let dy = f32(h.dy);
    let dist_sqr = dx * dx + dy * dy;

    if (dist_sqr >= R1_SQR && dist_sqr < R2_SQR) {{
        let base = h.scan_id * 2u;
        // sum (index 2*id) and count (index 2*id+1)
        atomicAdd(&out[base], h.tot);
        atomicAdd(&out[base + 1u], 1u);
    }}
}}
"#);

    let output_len = num_scan_positions * 2; // [sum0,count0, sum1,count1, ...]
    let gpu_t0 = Instant::now();
    let gpu_raw: Vec<u32> = run_compute_single_input_custom_output::<Hit, u32>(
        &ctx,
        &shader_source,
        "main",
        &hits,
        output_len,
        workgroup_size,
    );
    let gpu_time = gpu_t0.elapsed();

    // Unpack
    let mut gpu_sum: Vec<u64> = vec![0; num_scan_positions];
    let mut gpu_count: Vec<u64> = vec![0; num_scan_positions];
    for i in 0..num_scan_positions {
        gpu_sum[i] = gpu_raw[2 * i] as u64;
        gpu_count[i] = gpu_raw[2 * i + 1] as u64;
    }

    // --- Equality checks (deterministic) ---
    for i in 0..num_scan_positions {
        assert_eq!(gpu_sum[i], cpu_sum[i], "sum mismatch at scan {}", i);
        assert_eq!(gpu_count[i], cpu_count[i], "count mismatch at scan {}", i);
    }

    // Report a few intensities
    println!("OK — GPU == CPU for {} scan positions.", num_scan_positions);
    println!("CPU time: {:?} | GPU time (compute+readback): {:?}", cpu_time, gpu_time);
    println!("vADF intensities (first 10):");
    for i in 0..num_scan_positions.min(10) {
        let avg = if gpu_count[i] > 0 {
            gpu_sum[i] as f64 / gpu_count[i] as f64
        } else { 0.0 };
        println!("pos {i:4}: sum={} count={} avg={:.3}", gpu_sum[i], gpu_count[i], avg);
    }
}
