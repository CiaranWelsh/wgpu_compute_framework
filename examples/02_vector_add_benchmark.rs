//! Intermediate example: vector addition on the GPU versus the CPU.
//!
//! This program adds two arrays of 32‑bit floats element by element
//! using both the GPU and the CPU and compares their run times.  By
//! varying the length of the inputs you can identify at which point
//! the overhead of transferring data to and from the GPU is
//! outweighed by the compute throughput of your GPU.  This is the
//! simplest non‑trivial example of using `run_compute_two_inputs`.

use rand::Rng;
use std::time::Instant;
use wgpu_compute_framework::{run_compute_two_inputs, GpuContext};

fn main() {
    env_logger::init();
    let context = GpuContext::new_blocking().expect("failed to initialise GPU context");
    // Problem sizes to evaluate.  Feel free to add larger sizes if you
    // have enough memory on your GPU and host system.  Each number
    // denotes the length of the vectors `a` and `b`.
    let sizes = [1_usize, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000];
    // WGSL compute shader for vector addition.  It reads from
    // two storage buffers `a` and `b` and writes their sum into `out`.
    const ADD_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&a)) {
        return;
    }
    out[i] = a[i] + b[i];
}
"#;
    println!("Vector addition on GPU vs CPU (times include data transfer)");
    for &n in &sizes {
        let mut rng = rand::thread_rng();
        // Generate inputs with random values in [0,1).
        let a: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..1.0)).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..1.0)).collect();
        // CPU reference implementation.  This loop is trivially vectorised by
        // modern compilers and serves as a baseline.
        let mut cpu_out = vec![0.0f32; n];
        let start = Instant::now();
        for i in 0..n {
            cpu_out[i] = a[i] + b[i];
        }
        let cpu_time = start.elapsed();
        // GPU implementation.  `run_compute_two_inputs` uploads the data,
        // compiles the shader, dispatches workgroups and reads the
        // result back.  The chosen workgroup size (64) should match
        // the `@workgroup_size` in the shader.
        let start = Instant::now();
        let gpu_out = run_compute_two_inputs::<f32>(&context, ADD_SHADER, "add", &a, &b, 64);
        let gpu_time = start.elapsed();
        // Minimal sanity check on the first element.  In a robust test you
        // should verify the entire output using an appropriate tolerance.
        if let Some((cpu_first, gpu_first)) = cpu_out.get(0).zip(gpu_out.get(0)) {
            assert!((*cpu_first - *gpu_first).abs() < 1e-6);
        }
        println!("n = {:>9}: CPU = {:?}, GPU = {:?}", n, cpu_time, gpu_time);
    }
    println!("\nThe GPU spends time transferring the inputs and outputs as well as compiling and dispatching the compute pipeline.\nWhen the vectors are small, a single CPU core often completes the addition faster.  As the size increases, the GPU’s parallelism offsets the transfer cost.\nTry increasing the sizes or the workgroup size to explore the trade‑offs.");
}