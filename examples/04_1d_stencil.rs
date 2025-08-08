// 04_1d_stencil.rs
//! Lesson: 1D 3-point stencil (moving average).
//! Requirements targeted:
//! - R1: Show basic 1D dispatch and bounds checks.
//! - R2: Keep memory access coalesced and simple.
//! - R3: Provide a tiny test to validate correctness.
//!
//! Mapping: 1D workgroups; each thread writes one output element.
//! Unknown-unknown surfaced: boundary handling and safe oversubscription.

use rand::Rng;
use wgpu_compute_framework::{GpuContext, run_compute_single_input};

// Naive 3-point blur: out[i] = (in[i-1] + in[i] + in[i+1]) / 3
const SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read>  inp: array<f32>;
@group(0) @binding(1)
var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(128)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    let n: u32 = arrayLength(&inp);
    if (i >= n) { return; }

    // Handle boundaries by clamping indices.
    let i0 = select(0u, i - 1u, i > 0u);
    let i1 = i;
    let i2 = select(n - 1u, i + 1u, i + 1u < n);

    let s = inp[i0] + inp[i1] + inp[i2];
    outp[i] = s * 0.33333334;
}
"#;

fn cpu_blur(a: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let i0 = if i == 0 { 0 } else { i - 1 };
        let i2 = if i + 1 >= n { n - 1 } else { i + 1 };
        out[i] = (a[i0] + a[i] + a[i2]) / 3.0;
    }
    out
}

fn main() {
    env_logger::init();
    let ctx = GpuContext::new_blocking().expect("gpu context");

    let n = 1_000_000usize;
    let mut rng = rand::thread_rng();
    let a: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();

    let gpu_out = run_compute_single_input::<f32>(&ctx, SHADER, "blur", &a, 128);
    let cpu_out = cpu_blur(&a);

    // Spot-check a few elements to avoid O(n) comparison cost here.
    for &idx in &[0usize, 1, n/2, n-2, n-1] {
        let d = (gpu_out[idx] - cpu_out[idx]).abs();
        assert!(d < 1e-5, "mismatch at {idx}: gpu={} cpu={}", gpu_out[idx], cpu_out[idx]);
    }
    println!("OK: 1D stencil computed for n={}", n);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiny_correctness() {
        let ctx = GpuContext::new_blocking().unwrap();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gpu = run_compute_single_input::<f32>(&ctx, SHADER, "blur", &a, 4);
        let cpu = cpu_blur(&a);
        for i in 0..a.len() {
            assert!((gpu[i]-cpu[i]).abs() < 1e-6);
        }
    }
}
