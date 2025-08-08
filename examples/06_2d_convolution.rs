// 06_2d_convolution.rs
//! Lesson: 2D 3×3 blur (image convolution).
//! Requirements targeted:
//! - R1: Show natural 2D mapping and boundary handling.
//! - R2: Contrast naive global-memory version with tiled shared-memory version.
//! - R3: Small test for a tiny image.
//!
//! We encode (W,H) as compile-time constants in the shader to keep the host simple.

use wgpu_compute_framework::{GpuContext, run_compute_single_input};
use rand::Rng;

fn make_naive_shader(w: usize, h: usize) -> String {
    format!(r#"
const W: u32 = {w}u;
const H: u32 = {h}u;

@group(0) @binding(0) var<storage, read>        img: array<f32>;
@group(0) @binding(1) var<storage, read_write>  outp: array<f32>;

fn at(x: i32, y: i32) -> f32 {{
    let xx = clamp(x, 0, i32(W) - 1);
    let yy = clamp(y, 0, i32(H) - 1);
    let idx = u32(xx) + u32(yy) * W;
    return img[idx];
}}

@compute @workgroup_size(16, 16)
fn conv_naive(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let x = i32(gid.x);
    let y = i32(gid.y);
    if (gid.x >= W || gid.y >= H) {{ return; }}

    var sum: f32 = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {{
        for (var dx = -1; dx <= 1; dx = dx + 1) {{
            sum = sum + at(x + dx, y + dy);
        }}
    }}
    let idx = gid.x + gid.y * W;
    outp[idx] = sum / 9.0;
}}
"#)
}

fn make_tiled_shader(w: usize, h: usize, tx: usize, ty: usize) -> String {
    // Add +2 halo on each axis for radius=1.
    let tilew = tx + 2;
    let tileh = ty + 2;
    format!(r#"
const W: u32 = {w}u;
const H: u32 = {h}u;

@group(0) @binding(0) var<storage, read>        img: array<f32>;
@group(0) @binding(1) var<storage, read_write>  outp: array<f32>;

var<workgroup> tile: array<f32, {tilew}u*{tileh}u>;

fn clampu(v: i32, lo: i32, hi: i32) -> u32 {{
  return u32(clamp(v, lo, hi));
}}

fn at(x: i32, y: i32) -> f32 {{
    let xx = clampu(x, 0, i32(W)-1);
    let yy = clampu(y, 0, i32(H)-1);
    return img[xx + yy * W];
}}

@compute @workgroup_size({tx}, {ty})
fn conv_tiled(@builtin(local_invocation_id)  lid: vec3<u32>,
              @builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(workgroup_id)         wid: vec3<u32>) {{
    if (gid.x >= W || gid.y >= H) {{ return; }}

    // Each thread loads its pixel into the interior of the tile.
    let lx = i32(lid.x) + 1;
    let ly = i32(lid.y) + 1;
    let gx = i32(gid.x);
    let gy = i32(gid.y);
    let tw = {tilew}u;

    tile[u32(lx) + u32(ly) * tw] = at(gx, gy);

    // Threads at edges load halo (simple version; a few redundant loads are fine for clarity).
    if (lid.x == 0u) {{
        tile[0u + u32(ly)*tw] = at(gx-1, gy);
    }}
    if (lid.x == {tx-1}u) {{
        tile[u32({tilew-1}) + u32(ly)*tw] = at(gx+1, gy);
    }}
    if (lid.y == 0u) {{
        tile[u32(lx) + 0u*tw] = at(gx, gy-1);
    }}
    if (lid.y == {ty-1}u) {{
        tile[u32(lx) + u32({tileh-1})*tw] = at(gx, gy+1);
    }}

    // Corners (load once)
    if (lid.x == 0u && lid.y == 0u) {{
        tile[0u] = at(gx-1, gy-1);
    }}
    if (lid.x == {tx-1}u && lid.y == 0u) {{
        tile[u32({tilew-1})] = at(gx+1, gy-1);
    }}
    if (lid.x == 0u && lid.y == {ty-1}u) {{
        tile[0u + u32({tileh-1})*tw] = at(gx-1, gy+1);
    }}
    if (lid.x == {tx-1}u && lid.y == {ty-1}u) {{
        tile[u32({tilew-1}) + u32({tileh-1})*tw] = at(gx+1, gy+1);
    }}
    workgroupBarrier();

    // Compute from tile (interior index in tile)
    var sum: f32 = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {{
        for (var dx = -1; dx <= 1; dx = dx + 1) {{
            let ix = u32(i32(lx) + dx);
            let iy = u32(i32(ly) + dy);
            sum = sum + tile[ix + iy * tw];
        }}
    }}
    let out_idx = gid.x + gid.y * W;
    outp[out_idx] = sum / 9.0;
}}
"#)
}

fn main() {
    env_logger::init();
    let ctx = GpuContext::new_blocking().expect("gpu context");

    // Treat this as an image of shape (H, W) flattened row-major.
    let (w, h) = (1024usize, 1024usize);
    let n = w * h;
    let mut rng = rand::thread_rng();
    let img: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();

    // Naive
    let naive = make_naive_shader(w, h);
    let out_naive = run_compute_single_input::<f32>(&ctx, &naive, "conv_naive", &img, 256);

    // Tiled
    let tiled = make_tiled_shader(w, h, 16, 16);
    let out_tiled = run_compute_single_input::<f32>(&ctx, &tiled, "conv_tiled", &img, 256);

    // Sanity: results should be close.
    for &idx in &[0usize, w+1, n-1] {
        assert!((out_naive[idx] - out_tiled[idx]).abs() < 1e-4);
    }
    println!("OK: 2D convolution naive vs tiled match ({}x{})", w, h);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tiny_image_matches() {
        let ctx = GpuContext::new_blocking().unwrap();
        let (w, h) = (4usize, 3usize);
        let n = w*h;
        let img: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let naive = make_naive_shader(w, h);
        let tiled = make_tiled_shader(w, h, 2, 2);
        let a = run_compute_single_input::<f32>(&ctx, &naive, "conv_naive", &img, 4);
        let b = run_compute_single_input::<f32>(&ctx, &tiled, "conv_tiled", &img, 4);
        for i in 0..n { assert!((a[i]-b[i]).abs() < 1e-5); }
    }
}
