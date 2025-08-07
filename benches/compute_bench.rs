//! Criterion benchmarks comparing GPU and CPU vector addition.
//!
//! To run the benchmarks use `cargo bench`.  Criterion will execute
//! each function multiple times and report statistics such as the
//! median and standard deviation of the run times.  The GPU bench
//! includes the cost of submitting commands and reading back the
//! result, which makes it representative of real‑world latency when
//! processing streaming data.

use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

use wgpu_compute_framework::{run_compute_two_inputs, GpuContext};

fn gpu_add_benchmark(c: &mut Criterion) {
    // Establish a single GPU context up front so that the device and
    // pipeline creation overhead is not included in the benchmark.  In
    // a real application the context would be reused across many
    // compute submissions.
    let context = GpuContext::new_blocking().expect("failed to initialise GPU context");
    // Generate input data.  We choose a moderately large vector so that
    // the GPU has enough work to amortise the submission overhead.
    let n: usize = 100_000_000;
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
    const SHADER: &str = r#"
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
    c.bench_function("gpu vector add", |bencher| {
        bencher.iter(|| {
            // Dispatch the compute shader.  Discard the result to
            // avoid polluting the cache of subsequent iterations.
            let _ = run_compute_two_inputs(&context, SHADER, "add", &a, &b, 64);
        });
    });
    c.bench_function("cpu vector add", |bencher| {
        bencher.iter(|| {
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                out[i] = a[i] + b[i];
            }
        });
    });
}

criterion_group!(benches, gpu_add_benchmark);
criterion_main!(benches);