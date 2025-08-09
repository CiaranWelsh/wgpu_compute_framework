// 04_1d_stencil.rs
//! Lesson: 1D 3-point stencil (moving average).
//! Requirements targeted:
//! - R1: Show basic 1D dispatch and bounds checks.
//! - R2: Keep memory access coalesced and simple.
//! - R3: Provide a tiny test to validate correctness.
//!
//! Mapping: 1D workgroups; each thread writes one output element.
//! Unknown-unknown surfaced: boundary handling and safe oversubscription.

use log::log;
use rand::Rng;
use wgpu_compute_framework::{run_compute_single_input, GpuContext};

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
    // @const @must_use fn select(f: T, t: T, cond: bool) -> T
    // Returns t when cond is true, and f otherwise.
    // if i > 0u:
    //      i0 = i - 1u;
    // else:
    /       i0 = 0;
    let i0 = select(0u, i - 1u, i > 0u);
    let i1 = i;

    // if i + 1u < n:
    //      i2 = i + 1u;
    // else:
    /       i2 = n - 1u;
    let i2 = select(n - 1u, i + 1u, i + 1u < n);

    let s = inp[i0] + inp[i1] + inp[i2];
    outp[i] = s * 0.33333334;
}
"#;

const IDENTITY_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read>  inp: array<f32>;
@group(0) @binding(1)
var<storage, read_write> outp: array<f32>;

@compute @workgroup_size(128)
fn identity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    // let n: u32 = arrayLength(&inp);
    // if (i >= n) { return; }

    outp[i] = inp[i];
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

    let gpu_out = run_compute_single_input::<f32>(&ctx, SHADER, "blur", &a, 1024);

    println!("{:?}", gpu_out.into_iter().take(10).collect::<Vec<_>>());
    // let cpu_out = cpu_blur(&a);

    // // Spot-check a few elements to avoid O(n) comparison cost here.
    // for &idx in &[0usize, 1, n/2, n-2, n-1] {
    //     let d = (gpu_out[idx] - cpu_out[idx]).abs();
    //     assert!(d < 1e-5, "mismatch at {idx}: gpu={} cpu={}", gpu_out[idx], cpu_out[idx]);
    // }
    // println!("OK: 1D stencil computed for n={}", n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    #[test]
    fn tiny_correctness() {
        let ctx = GpuContext::new_blocking().unwrap();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let gpu = run_compute_single_input::<f32>(&ctx, SHADER, "blur", &a, 4);
        let cpu = cpu_blur(&a);
        for i in 0..a.len() {
            assert!((gpu[i] - cpu[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn identity_test() {
        let ctx = GpuContext::new_blocking().unwrap();

        let mut a = vec![0.0f32; 1024];
        a[0..4].copy_from_slice(&[10., 20., 30., 40.]);

        let gpu = run_compute_single_input::<f32>(&ctx, IDENTITY_SHADER, "identity", &a, 256);
        let expected: Vec<f32> = a.iter().copied().collect();
        assert_eq!(&gpu[0..4], &[10., 20., 30., 40.]);
    }

    #[test]
    fn exercise_2_workgroup_size() {
        env_logger::init();
        let ctx = GpuContext::new_blocking().expect("gpu context");

        for nw in vec![32, 64, 128, 256] {
            for n in vec![1_000_000, 50_000_000] {
                let mut rng = rand::thread_rng();
                let a: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
                let now = Instant::now();
                let gpu_out = run_compute_single_input::<f32>(&ctx, SHADER, "blur", &a, nw);
                println!(
                    "Time for workgroup_size: {nw} and n={n}: {:?}",
                    now.elapsed()
                );
            }
        }
    }

    #[test]
    fn exercise_2_workgroup_size_fixed() {
        use std::time::Instant;
        let ctx = GpuContext::new_blocking().expect("gpu context");

        // Apple/Metal typical binding cap: 128 MiB.
        const BYTES_CAP: usize = 128 * 1024 * 1024;
        let max_elems_f32 = BYTES_CAP / std::mem::size_of::<f32>(); // 33_554_432

        for &nw in &[32u32, 64, 128, 256, 512] {
            for &n in &[1_000_000usize, 50_000_000] {
                let n = n.min(max_elems_f32 - 1024); // pad a bit for safety
                let mut rng = rand::thread_rng();
                let a: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();

                let now = Instant::now();
                let _gpu_out = run_compute_single_input::<f32>(&ctx, SHADER, "blur", &a, nw);
                println!("workgroup_size={nw:>3} n={n:>9} -> {:?}", now.elapsed());
            }
        }
    }
}
