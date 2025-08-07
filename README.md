# wgpu_compute_framework

A minimal framework for running compute workloads on the GPU using
[wgpu](https://github.com/gfx-rs/wgpu).  It is intended as a
building block for high‑throughput, low‑latency data processing
applications such as processing streams from Timepix3 and Timepix4
detectors.  The design favours simplicity over abstraction: a single
function call uploads data to the GPU, dispatches a WGSL compute
shader and returns the results to the host.

## Features

* **Headless GPU initialisation.**  The `GpuContext` type hides the
  asynchronous adapter and device selection logic behind a simple
  synchronous API.  It automatically selects a high performance
  adapter and verifies that compute shaders are supported.
* **Typed GPU buffers.**  The `GpuBuffer<T>` wrapper tracks the
  length of typed arrays stored on the GPU, making it easy to avoid
  mismatched buffer sizes.  Functions are provided to upload data
  from the CPU and to download results back to the host.
* **One and two input compute helpers.**  The `run_compute_single_input`
  and `run_compute_two_inputs` functions take care of creating
  bind groups, pipelines and command encoders.  They dispatch a WGSL
  compute shader with a configurable workgroup size and block until
  the result is ready.
* **Criterion benchmarks.**  A benchmark in `benches/compute_bench.rs`
  measures the performance of vector addition on the GPU versus a
  straightforward CPU loop.  This provides a starting point for
  evaluating the throughput of your own compute kernels.

## Usage

Add this crate as a dependency in your `Cargo.toml`:

```toml
[dependencies]
wgpu_compute_framework = { path = "path/to/wgpu_compute_framework" }
```

Create a GPU context once at start up and reuse it for all compute
tasks:

```rust
use wgpu_compute_framework::{GpuContext, run_compute_two_inputs};

fn main() {
    // Create the GPU context.  Panics if no suitable adapter is found.
    let context = GpuContext::new_blocking().unwrap();

    // Prepare your input data.  The example below performs
    // element‑wise addition of two arrays of `f32`.
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    // Write your WGSL compute shader.  It must define bindings for
    // each buffer and a `@compute` entry point.  Guard against
    // out‑of‑bounds accesses when the array length is not divisible by
    // the workgroup size.
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
    if (i >= arrayLength(&a)) { return; }
    out[i] = a[i] + b[i];
}
"#;

    // Dispatch the shader and retrieve the results.  The final
    // argument controls the number of threads per workgroup.
    let result = run_compute_two_inputs(&context, SHADER, "add", &a, &b, 64);
    println!("{result:?}");
}
```

For more examples consult the `examples` directory.  To run the
vector addition example use:

```sh
cargo run --example vector_add
```

To execute the benchmarks:

```sh
cargo bench
```

## Limitations and future work

This framework is intentionally simple.  It does not handle
asynchronous command submission, persistent pipeline caching, or
advanced resource layouts.  Those aspects can be layered on top of
this crate as needed.  Similarly, the helper functions assume that
the output buffer has the same length as the input buffers; more
complex workloads may require bespoke bind group layouts and shader
interfaces.

Contributions and improvements are welcome!