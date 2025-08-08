//! Advanced example: dense matrix multiplication on the GPU.
//!
//! This example multiplies two square matrices `A` and `B` on the GPU using
//! a compute shader and compares the performance to a naive CPU
//! implementation.  Matrix multiplication has a high arithmetic
//! intensity (O(N^3) operations for N×N matrices) which makes it a
//! candidate for GPU acceleration.  For small matrices the CPU
//! overhead of submission and data transfer may dominate, but as `N`
//! grows the GPU’s parallelism becomes advantageous.  Note that this
//! implementation does not use workgroup shared memory or other
//! optimisations; it is intended as an educational starting point.

use rand::Rng;
use std::time::Instant;
use wgpu_compute_framework::{run_compute_two_inputs, GpuContext};

fn main() {
    env_logger::init();
    let context = GpuContext::new_blocking().expect("failed to initialise GPU context");
    // Matrix dimensions to test.  These values will be used for both
    // the CPU and GPU implementations.  Larger dimensions dramatically
    // increase run time on the CPU (O(N^3)) and memory usage (O(N^2)).
    let dims = [32_usize, 64, 128, 256, 512];
    println!("Square matrix multiplication (N×N) on GPU vs CPU");
    for &n in &dims {
        // Generate two random matrices flattened into 1D arrays in row‑major
        // order.  Each contains `n * n` elements.
        let mut rng = rand::thread_rng();
        let a: Vec<f32> = (0..(n * n)).map(|_| rng.gen_range(0.0f32..1.0)).collect();
        let b: Vec<f32> = (0..(n * n)).map(|_| rng.gen_range(0.0f32..1.0)).collect();
        // CPU: naive triple nested loops.
        let start = Instant::now();
        let mut cpu_out = vec![0.0f32; n * n];
        for row in 0..n {
            for col in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += a[row * n + k] * b[k * n + col];
                }
                cpu_out[row * n + col] = sum;
            }
        }
        let cpu_time = start.elapsed();
        // GPU: generate a WGSL shader with a compile‑time constant for the dimension.
        // The shader treats the matrices as flattened arrays and computes one
        // output element per invocation.  We dispatch `n * n` threads.
        let shader = format!(r#"
const N: u32 = {}u;
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(16)
fn mmul(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx: u32 = global_id.x;
    if (idx >= N * N) {{
        return;
    }}
    let row: u32 = idx / N;
    let col: u32 = idx % N;
    var sum: f32 = 0.0;
    var k: u32 = 0u;
    // Loop over the shared dimension.  Each iteration multiplies one
    // element from row `row` of A with one element from column `col` of B.
    loop {{
        if (k >= N) {{
            break;
        }}
        let a_index: u32 = row * N + k;
        let b_index: u32 = k * N + col;
        sum = sum + a[a_index] * b[b_index];
        k = k + 1u;
    }}
    out[idx] = sum;
}}
"#, n);
        let start = Instant::now();
        // Each compute invocation will process one output element.  We use a 1D workgroup size of 64.
        let gpu_out = run_compute_two_inputs::<f32>(&context, &shader, "mmul", &a, &b, 64);
        let gpu_time = start.elapsed();
        // Validate the first element (0,0) to ensure correctness.  We allow a
        // small relative error because floating point operations may be
        // reordered on the GPU.  A full test would compare all elements.
        let idx = 0;
        let cpu_val = cpu_out[idx];
        let gpu_val = gpu_out[idx];
        let max_abs = cpu_val.abs().max(1.0);
        assert!((cpu_val - gpu_val).abs() < 1e-4 * max_abs);
        println!("N = {:>3}: CPU = {:?}, GPU = {:?}", n, cpu_time, gpu_time);
    }
    println!("\nThis simple implementation of matrix multiplication illustrates how compute workloads with high arithmetic intensity favour the GPU.\nFor very small matrices the CPU loop wins because the submission overhead dominates.  As N grows, the O(N^3) arithmetic cost allows the GPU to shine.\nMore sophisticated compute shaders using tiled algorithms and shared memory can further improve performance.");
}