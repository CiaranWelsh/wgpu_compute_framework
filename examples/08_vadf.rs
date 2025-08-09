//! Example 8: Virtual Annular Dark‑Field (vADF) imaging using wGPU.
//!
//! This example demonstrates how to implement a simple vADF algorithm on the GPU
//! using the existing `wgpu_compute_framework`.  We simulate a 4D STEM scan
//! where a beam raster‑scans over a sample and records a 2D diffraction
//! pattern (256×256 pixel detector) at each scan position.  Each detected
//! event (hit) has pixel coordinates `(x, y)`, a time‑of‑arrival (ToA) and a
//! time‑over‑threshold (ToT).  The ToA encodes which scan position the hit
//! belongs to via the dwell time: hits are pre‑sorted by ToA and grouped
//! into scan positions based on this dwell window.  For simplicity our
//! simulation skips fine structure such as TDC rise/fall markers and
//! generates random hits per scan position.
//!
//! The vADF algorithm sums (or averages) the ToT values for hits whose
//! radial distance from the detector centre lies within an annulus defined
//! by radii `r1` and `r2`.  For each scan position we produce a single
//! intensity value: `sum(tot[i] for hits within annulus)` and the count of
//! hits contributing.  Dividing the sum by the count yields the virtual
//! annular dark‑field image.  The GPU kernel uses one thread per hit and
//! accumulates per‑scan sums and counts using atomics.

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use rand::Rng;
use wgpu_compute_framework::context::GpuContext;
use wgpu_compute_framework::compute::run_compute_single_input_custom_output;

// Pollster provides a lightweight block_on that doesn't require an async
// runtime.  It is used here to synchronously wait on GPU buffer mapping
// futures without adding a heavy dependency like Tokio.
use pollster;

/// A single simulated hit.  We store dx and dy (pixel offset from detector
/// centre) as 32‑bit integers to avoid converting in the shader.  `tot`
/// represents the time‑over‑threshold (a proxy for detector intensity) and
/// `scan_id` identifies the scan position this hit belongs to.  Deriving
/// Pod/Zeroable allows safe casting to bytes for GPU upload.
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
    // Create a GPU context using the helper.  This selects a device and
    // queue using wgpu 0.26 under the hood.
    let ctx = GpuContext::new_blocking().expect("failed to create GpuContext");

    // Parameters for the scan and simulation.  We choose a modest grid
    // (e.g. 64×64 = 4096 scan positions) to keep the runtime reasonable on
    // commodity machines.  Increasing the grid or hits_per_pos will
    // proportionally increase the workload.
    let scan_rows: u32 = 64;
    let scan_cols: u32 = 64;
    let num_scan_positions = (scan_rows * scan_cols) as usize;
    let dwell_time_ns: f64 = 10_000.0; // 10 µs dwell time per scan position
    // Detector parameters: resolution and region of interest.  The
    // Timepix3 has 256×256 pixels【250169078949887†L339-L344】.  We place the origin at the
    // detector centre (127.5, 127.5) and compute dx/dy accordingly.  The
    // annular ROI is defined by inner radius r1 and outer radius r2.
    let detector_size: u32 = 256;
    let centre = (detector_size as f32 - 1.0) / 2.0;
    let r1: f32 = 30.0;
    let r2: f32 = 90.0;
    let r1_sqr: f32 = r1 * r1;
    let r2_sqr: f32 = r2 * r2;
    // Simulation: how many hits per scan position on average.  We sample a
    // Poisson‑like distribution with mean hits_per_pos; adjust this to
    // explore performance.  The total number of hits = hits_per_pos * #scan
    // positions.
    let hits_per_pos: usize = 200;
    let total_hits: usize = num_scan_positions * hits_per_pos;

    // Precompute mapping from scan index to its 2D coordinates (row, col).
    // This is useful for computing ToA (time encoding), though the ToA
    // values themselves are not explicitly stored in the Hit structure.
    let dwell_time_ticks: f64 = dwell_time_ns / 1.5625; // ToA units (1 tick = 1.5625 ns)
    // Generate synthetic hits.  Each hit gets a random pixel (x,y) and
    // random ToT.  We compute dx, dy as difference from centre and set
    // scan_id according to the dwell time and ToA ordering.  Hits are
    // appended in scan order so the buffer is already sorted by scan_id.
    let mut hits: Vec<Hit> = Vec::with_capacity(total_hits);
    let mut rng = rand::thread_rng();
    for scan_id in 0..num_scan_positions {
        // Optionally vary hits per scan position (e.g. Poisson); here we fix
        // to hits_per_pos for reproducibility.
        for _ in 0..hits_per_pos {
            let x = rng.gen_range(0..detector_size);
            let y = rng.gen_range(0..detector_size);
            let dx = x as f32 - centre;
            let dy = y as f32 - centre;
            // Quantise dx/dy to i32; storing difference avoids repeated
            // subtraction on the GPU.  Note: we multiply by 1.0 to ensure
            // float→int conversion occurs in CPU, not WGSL.
            let hit = Hit {
                dx: dx as i32,
                dy: dy as i32,
                tot: rng.gen_range(1..100),
                scan_id: scan_id as u32,
            };
            hits.push(hit);
        }
    }
    assert_eq!(hits.len(), total_hits);

    // Compute CPU reference sums and counts for validation.  We convert
    // dx/dy back to floats, compute distance squared and accumulate when
    // within the annulus.
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

    // We'll create a WGSL shader that uses one output array of atomic u32s
    // where each scan position occupies two consecutive slots: the first
    // for the sum of ToT values and the second for the hit count.  The
    // helper `run_compute_single_input_custom_output` accepts exactly one
    // input buffer and one output buffer.  Here we flatten the per‑scan
    // sum and count into a single output buffer of length `2 * num_scan_positions`.
    let workgroup_size: u32 = 64;
    // Embed constants and workgroup size into the WGSL source.  The
    // `scan_id` is used to compute indices into the flattened output
    // array.  Atomic operations accumulate the sum and count.
    let shader_source = format!(r#"
struct Hit {{
    dx: i32,
    dy: i32,
    tot: u32,
    scan_id: u32,
}};

@group(0) @binding(0) var<storage, read> hits: array<Hit>;
@group(0) @binding(1) var<storage, read_write> out: array<atomic<u32>>;

const R1_SQR: f32 = {r1_sqr}f;
const R2_SQR: f32 = {r2_sqr}f;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i: u32 = gid.x;
    if (i >= arrayLength(&hits)) {{
        return;
    }}
    let hit = hits[i];
    let dx = f32(hit.dx);
    let dy = f32(hit.dy);
    let dist_sqr = dx * dx + dy * dy;
    if (dist_sqr >= R1_SQR && dist_sqr < R2_SQR) {{
        let base = hit.scan_id * 2u;
        atomicAdd(&out[base], hit.tot);
        atomicAdd(&out[base + 1u], 1u);
    }}
}}
"#);
    // Run the compute shader using the framework helper.  The output
    // length is twice the number of scan positions.  The call is
    // synchronous and blocks until the GPU work is finished and the
    // results have been copied back to the host.
    let output_len = num_scan_positions * 2;
    let start = Instant::now();
    let gpu_raw: Vec<u32> = run_compute_single_input_custom_output::<Hit, u32>(
        &ctx,
        &shader_source,
        "main",
        &hits,
        output_len,
        workgroup_size,
    );
    let gpu_time = start.elapsed();
    // Split the flattened output into sum and count arrays of u64 for
    // comparison with the CPU reference.  Note: atomic accumulation
    // occurs modulo 2^32; wraparound is ignored for this example.
    let mut gpu_sum: Vec<u64> = vec![0; num_scan_positions];
    let mut gpu_count: Vec<u64> = vec![0; num_scan_positions];
    for i in 0..num_scan_positions {
        gpu_sum[i] = gpu_raw[2 * i] as u64;
        gpu_count[i] = gpu_raw[2 * i + 1] as u64;
    }
    // Validate results
    for i in 0..num_scan_positions {
        assert_eq!(gpu_sum[i], cpu_sum[i], "sum mismatch at {}", i);
        assert_eq!(gpu_count[i], cpu_count[i], "count mismatch at {}", i);
    }
    // Print a few vADF intensities for visual inspection
    println!("GPU vADF intensities (first 10 positions):");
    for i in 0..10.min(num_scan_positions) {
        let intensity = if gpu_count[i] > 0 {
            gpu_sum[i] as f64 / gpu_count[i] as f64
        } else {
            0.0
        };
        println!("pos {} -> sum={} count={} avg={:.3}", i, gpu_sum[i], gpu_count[i], intensity);
    }
    println!("Total hits: {} across {} scan positions", hits.len(), num_scan_positions);
    println!("GPU compute time (framework): {:?}", gpu_time);
}