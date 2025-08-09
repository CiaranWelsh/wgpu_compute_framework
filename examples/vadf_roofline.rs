use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use bytemuck::{Pod, Zeroable};
use wgpu_compute_framework::context::GpuContext;
use wgpu_compute_framework::compute::run_compute_single_input_custom_output_timed;

// ---------------- Data types ----------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Hit {
    dx: i32,
    dy: i32,
    tot: u32,
    scan_id: u32,
}

// ---------------- Kernel ----------------

fn make_shader(r1_sqr: f32, r2_sqr: f32, workgroup_size: u32) -> String {
    format!(r#"
struct Hit {{
  dx: i32,
  dy: i32,
  tot: u32,
  scan_id: u32,
}};

@group(0) @binding(0) var<storage, read>       hits: array<Hit>;
@group(0) @binding(1) var<storage, read_write> out:  array<atomic<u32>>;

const R1_SQR: f32 = {r1_sqr};
const R2_SQR: f32 = {r2_sqr};

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i: u32 = gid.x;
    if (i >= arrayLength(&hits)) {{ return; }}
    let h = hits[i];

    // float math for ROI distance
    let dx = f32(h.dx);
    let dy = f32(h.dy);
    let d2 = dx * dx + dy * dy;  // 2 mul + 1 add = 3 FLOPs per hit

    if (d2 >= R1_SQR && d2 < R2_SQR) {{
        let base = h.scan_id * 2u;
        atomicAdd(&out[base + 0u], h.tot); // sum
        atomicAdd(&out[base + 1u], 1u);    // count
    }}
}}
"#)
}

// ---------------- Synthetic generator ----------------

fn synth_hits(
    n_scan: usize,
    hits_per_pos: usize,
    det: u32,
    seed: u64,
) -> Vec<Hit> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centre = (det as f32 - 1.0) * 0.5;
    let mut hits = Vec::with_capacity(n_scan * hits_per_pos);
    for s in 0..n_scan {
        for _ in 0..hits_per_pos {
            let x = rng.gen_range(0..det);
            let y = rng.gen_range(0..det);
            let dx = (x as f32 - centre) as i32;
            let dy = (y as f32 - centre) as i32;
            hits.push(Hit {
                dx, dy,
                tot: rng.gen_range(1..100),
                scan_id: s as u32,
            });
        }
    }
    hits
}

// ---------------- Main: sweep + CSV ----------------

fn main() {
    env_logger::init();
    let ctx = GpuContext::new_blocking().expect("GpuContext");

    // Sweep space
    let scan_rows = 64u32;
    let scan_cols = 64u32;
    let n_scan = (scan_rows * scan_cols) as usize;
    let det = 256u32;

    let workgroups = [64u32, 128, 256];
    let hits_per_pos_list = [50usize, 200, 800];
    let annuli = [
        (10.0f32, 30.0f32),
        (30.0, 60.0),
        (60.0, 100.0),
    ];

    // CSV header
    println!("variant,scan,det,hits_per_pos,r1,r2,workgroup,total_hits,roi_hits,kernel_ms,submit_ms,GFLOPs,GBps,OI");

    for &wg in &workgroups {
        for &hpp in &hits_per_pos_list {
            for &(r1, r2) in &annuli {
                let r1_sqr = r1 * r1;
                let r2_sqr = r2 * r2;

                let hits = synth_hits(n_scan, hpp, det, 0xC0FFEE);
                let total_hits = hits.len();

                // Output layout: [sum0,count0, sum1,count1, ...]
                let out_len = n_scan * 2;

                let shader = make_shader(r1_sqr, r2_sqr, wg);

                // Timed GPU run (kernel + submit)
                let (gpu_raw, timing) = run_compute_single_input_custom_output_timed::<Hit, u32>(
                    &ctx, &shader, "main", &hits, out_len, wg,
                );

                // Unpack counts to compute ROI fraction
                let mut roi_hits: u64 = 0;
                for i in 0..n_scan {
                    roi_hits += gpu_raw[2*i + 1] as u64;
                }

                // ---- Derived metrics for roofline ----
                // FLOPs: 3 per hit, regardless of ROI (dx*dx + dy*dy)
                let total_flops = 3.0 * total_hits as f64;

                // Bytes touched by kernel:
                // - load Hit (16B) always
                // - if hits in ROI: two u32 atomics (~we approximate as 8B R/W). Use counts to scale.
                let total_bytes = 16.0 * total_hits as f64 + 8.0 * roi_hits as f64;
                let oi = total_flops / total_bytes;                // FLOPs / byte
                let kernel_s = timing.kernel_ns * 1e-9;
                let gflops = (total_flops / kernel_s) / 1e9;
                let gbytes = (total_bytes / kernel_s) / 1e9;

                println!(
                    "vadf,{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.3},{:.3}",
                    n_scan, det, hpp, r1, r2, wg, total_hits, roi_hits,
                    timing.kernel_ns/1e6, timing.submit_ns/1e6,
                    gflops, gbytes, oi
                );
            }
        }
    }
}
