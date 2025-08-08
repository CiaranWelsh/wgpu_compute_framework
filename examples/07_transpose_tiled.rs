// 07_transpose_tiled.rs
//! Lesson: Cache-friendly matrix transpose with padded tiles.
//! Requirements targeted:
//! - R1: Show tile + halo style shared memory, but now for transpose.
//! - R2: Explain +1 padding to avoid shared memory bank conflicts.
//! - R3: Provide a correctness test.
//!
//! Input/Output have the same length; data is a flattened row-major W×H matrix.

use wgpu_compute_framework::{GpuContext, run_compute_single_input};

fn make_transpose_shader(w: usize, h: usize, tile: usize) -> String {
    // +1 column padding avoids shared-memory bank conflicts when threads read columns.
    let pad = 1usize;
    let sh_w = tile + pad;
    format!(r#"
const W: u32 = {w}u;
const H: u32 = {h}u;
const TILE: u32 = {tile}u;
const SH_W: u32 = {sh_w}u; // padded width

@group(0) @binding(0) var<storage, read>       inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;

var<workgroup> tile: array<f32, SH_W * TILE>;

@compute @workgroup_size({tile}, {tile})
fn transpose(@builtin(local_invocation_id)  lid: vec3<u32>,
             @builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(workgroup_id)         wid: vec3<u32>) {{
    let x = gid.x;
    let y = gid.y;
    if (x >= W || y >= H) {{ return; }}

    // Global indices
    let in_idx = x + y * W;

    // Local indices inside the padded shared tile
    let lx = lid.x;
    let ly = lid.y;
    tile[lx + ly * SH_W] = inp[in_idx];
    workgroupBarrier();

    // Compute the transposed coordinates
    let tx = wid.y * TILE + lx;
    let ty = wid.x * TILE + ly;
    if (tx < H && ty < W) {{
        let out_idx = ty + tx * H;
        // Read from the transposed location in shared memory (note SH_W!)
        outp[out_idx] = tile[ly + lx * SH_W];
    }}
}}
"#)
}

fn main() {
    env_logger::init();
    let ctx = GpuContext::new_blocking().expect("gpu context");

    let (w, h) = (1024usize, 768usize);
    let n = w*h;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let shader = make_transpose_shader(w, h, 16);
    let out = run_compute_single_input::<f32>(&ctx, &shader, "transpose", &a, 256);

    // Validate: (x,y) in -> (y,x) out
    let idx = |x: usize, y: usize| -> usize { x + y*w };
    let idx_t = |x: usize, y: usize| -> usize { y + x*h };
    for &sample in &[(0,0), (1,2), (w-1,h-1), (10,20)] {
        let (x,y) = sample;
        assert!((out[idx_t(x,y)] - a[idx(x,y)]).abs() < 1e-6);
    }
    println!("OK: transpose {}x{}", w, h);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiny_transpose() {
        let ctx = GpuContext::new_blocking().unwrap();
        let (w,h) = (5usize,4usize);
        let n = w*h;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let shader = make_transpose_shader(w,h,4);
        let out = run_compute_single_input::<f32>(&ctx, &shader, "transpose", &a, 16);
        let idx = |x: usize, y: usize| -> usize { x + y*w };
        let idx_t = |x: usize, y: usize| -> usize { y + x*h };
        for y in 0..h {
            for x in 0..w {
                assert!((out[idx_t(x,y)] - a[idx(x,y)]).abs() < 1e-6);
            }
        }
    }
}
