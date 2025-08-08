//! Beginner example: element‑wise doubling on the GPU versus the CPU.
//!
//! This example demonstrates how to use the `run_compute_single_input` helper
//! to dispatch a WGSL compute shader that multiplies each element in an
//! array by two.  We generate random input data of varying sizes and
//! measure the time taken on the CPU and GPU.  For small inputs the
//! overhead of transferring data to and from the GPU outweighs any
//! computational speed‑up.  As the number of elements grows the GPU
//! becomes competitive.  Use this example to get an intuition for the
//! tipping point on your hardware.

use rand::Rng;
use std::time::Instant;
use wgpu_compute_framework::{run_compute_single_input, GpuContext};

fn main() {
    // Initialise logging so that wgpu can emit diagnostic messages when
    // RUST_LOG is set.  Remove this line if you prefer a quieter output.
    env_logger::init();
    // Create a GPU context.  This discovers and initialises the GPU.
    let context = GpuContext::new_blocking()
        .expect("failed to initialise GPU context");
    // Define a range of sizes to test.  You can adjust these numbers to
    // probe performance on your machine.  Each value represents the
    // number of 32‑bit floats processed in one dispatch.
    let sizes = [1_usize, 1_000, 10_000, 100_000, 1_000_000, 10_000_000,15_000_000 ];
    println!("Element‑wise doubling on GPU vs CPU (times include data transfer)");
    // WGSL compute shader that doubles each element.  The shader reads
    // from `input` and writes the result to `out`.  It guards against
    // out‑of‑bounds access when the total number of invocations is not an
    // exact multiple of the workgroup size.
    const SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn double(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&input)) {
        return;
    }
    out[i] = input[i] * 2.0;
}
"#;
    for &n in &sizes {
        // Generate random input data of length `n`.
        let mut rng = rand::thread_rng();
        let input: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..1.0)).collect();
        // CPU: perform the same operation with a simple loop.  We clone
        // the input vector so that the original data remains available for
        // the GPU invocation.
        let mut cpu_data = input.clone();
        let start = Instant::now();
        for x in &mut cpu_data {
            *x *= 2.0;
        }
        let cpu_time = start.elapsed();
        // GPU: dispatch the compute shader.  `run_compute_single_input` takes
        // ownership of the data slice and returns a new vector containing
        // the results.  The workgroup size must match the value in the
        // shader's `@workgroup_size` attribute.
        let start = Instant::now();
        let gpu_output = run_compute_single_input::<f32>(&context, SHADER, "double", &input, 256);
        let gpu_time = start.elapsed();
        // Sanity check on the first element.  We only check a single
        // element to avoid introducing measurement noise.  A full test
        // suite should compare all elements and handle NaNs appropriately.
        if let Some((cpu_first, gpu_first)) = cpu_data.get(0).zip(gpu_output.get(0)) {
            assert!((*cpu_first - *gpu_first).abs() < 1e-6);
        }
        println!("n = {:>9}: CPU = {:?}, GPU = {:?}", n, cpu_time, gpu_time);
    }
    println!("\nObserve that the GPU timing includes the cost of uploading data, dispatching the shader and reading back the result.\nFor small arrays the CPU is faster because there is little work to amortise the transfer overhead.\nOn sufficiently large inputs the GPU will begin to pull ahead.  The exact crossover depends on your hardware and workload.");
}