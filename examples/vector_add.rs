//! Example demonstrating element‑wise vector addition on the GPU.
//!
//! Run this example with:
//!
//! ```sh
//! cargo run --example vector_add
//! ```
//!
//! The program constructs two vectors of 32‑bit floats, dispatches a
//! compute shader to add them together, and prints the result to
//! stdout.  The compute shader is written in WGSL and embedded as a
//! raw string.

use wgpu_compute_framework::{run_compute_two_inputs, GpuContext};

fn main() {
    // Initialize the environment logger so that wgpu can emit debug
    // information if the `RUST_LOG` environment variable is set.  If
    // you don't wish to see wgpu's internal logs you can omit this
    // call.
    env_logger::init();
    // Create a GPU context.  This will block while wgpu discovers the
    // hardware, picks an adapter and creates a device.
    let context = GpuContext::new_blocking().expect("failed to initialise GPU context");
    // Define the input data.  In a real application these would
    // represent a stream of measurements from a sensor such as a
    // Timepix3 or Timepix4 detector.  For demonstration purposes
    // we simply choose small vectors so that the output fits on one
    // line.
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    // WGSL compute shader performing element‑wise addition.  It reads
    // two storage buffers `a` and `b`, sums the elements at the same
    // index, and writes the result into the third storage buffer
    // `out`.  A guard checks the invocation index against the array
    // length to avoid out‑of‑bounds writes if the number of elements
    // isn't an exact multiple of the workgroup size.
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
    // Ensure we don't access beyond the end of the buffer when the
    // number of elements is not divisible by the workgroup size.
    if (i >= arrayLength(&a)) {
        return;
    }
    out[i] = a[i] + b[i];
}
"#;
    // Dispatch the compute shader.  We choose a workgroup size of 64
    // because it is a common value for one‑dimensional workloads.
    let result = run_compute_two_inputs(&context, SHADER, "add", &a, &b, 64);
    println!("Result: {result:?}");
}