// 05_prefix_sum_block_scan.rs
//! Lesson: Inclusive scan (prefix sum) using workgroup shared memory + barriers.
//! Requirements targeted:
//! - R1: Introduce `var<workgroup>` and `workgroupBarrier()`.
//! - R2: Show a simple, debuggable **two-pass** approach without new host helpers.
//! - R3: Tests document assumptions and expected behavior.
//!
//! Strategy:
//! 1) Pass A (GPU): per-workgroup inclusive scan (Hillis–Steele) over fixed-size tiles.
//!    Writes the scanned array to `out`. The last element of each tile is the tile sum.
//! 2) CPU: build an array `tile_offsets_exclusive` by scanning the tile sums on CPU
//!    (small; #tiles << N). Expand to length N by repeating each tile's exclusive prefix.
//! 3) Pass B (GPU): elementwise add `out += expanded_offsets` using the 2-input helper.
//!
//! This avoids adding new framework functions while still teaching shared memory.

use wgpu_compute_framework::{GpuContext, run_compute_single_input, run_compute_two_inputs};

// --- Pass A: per-tile inclusive scan using workgroup memory ---
const SCAN_TILE_WGSL: &str = r#"
const TILE: u32 = 128u;

@group(0) @binding(0) var<storage, read>        inp: array<f32>;
@group(0) @binding(1) var<storage, read_write>  outp: array<f32>;

var<workgroup> scratch: array<f32, TILE>;

@compute @workgroup_size(TILE)
fn scan_tile(@builtin(local_invocation_id)  lid: vec3<u32>,
             @builtin(global_invocation_id) gid: vec3<u32>,
             @builtin(workgroup_id)         wid: vec3<u32>) {
    let n = arrayLength(&inp);
    let i = gid.x;
    let lane = lid.x;

    // Load into shared memory (or 0 beyond bounds).
    var x = 0.0;
    if (i < n) {
        x = inp[i];
    }
    scratch[lane] = x;
    workgroupBarrier();

    // Hillis–Steele inclusive scan in shared memory.
    var offset: u32 = 1u;
    loop {
        if (offset >= TILE) { break; }
        let prev = scratch[lane];
        // Read neighbor if exists within this tile.
        let neighbor_idx = select(0u, lane - offset, lane >= offset);
        let addend = select(0.0, scratch[neighbor_idx], lane >= offset);
        workgroupBarrier(); // ensure all read old values
        scratch[lane] = prev + addend;
        workgroupBarrier();
        offset = offset << 1u;
    }

    if (i < n) {
        outp[i] = scratch[lane];
    }
}
"#;

// --- Pass B: add expanded tile offsets: out[i] = out[i] + offs[i] ---
const ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(256)
fn add2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&a)) { return; }
    o[i] = a[i] + b[i];
}
"#;

fn cpu_expand_offsets(tile_prefix_exclusive: &[f32], tile: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (t, &off) in tile_prefix_exclusive.iter().enumerate() {
        let start = t * tile;
        let end = (start + tile).min(n);
        for i in start..end {
            out[i] = off;
        }
    }
    out
}

fn main() {
    env_logger::init();
    let ctx = GpuContext::new_blocking().expect("gpu context");

    let n = 1_000_000usize;
    let tile = 128usize;

    // Data: all 1's → inclusive scan should be [1,2,3,...].
    let a: Vec<f32> = vec![1.0; n];

    // Pass A: per-tile scan on GPU.
    let scanned = run_compute_single_input::<f32>(&ctx, SCAN_TILE_WGSL, "scan_tile", &a, 128);

    // Build tile sums (last element of each tile) and exclusive-scan them on CPU.
    let tiles = (n + tile - 1) / tile;
    let mut tile_sums = vec![0.0f32; tiles];
    for t in 0..tiles {
        let end = ((t + 1) * tile).min(n);
        tile_sums[t] = scanned[end - 1];
    }
    // Exclusive scan of tile_sums.
    let mut tile_prefix_excl = vec![0.0f32; tiles];
    let mut acc = 0.0f32;
    for t in 0..tiles {
        tile_prefix_excl[t] = acc;
        acc += tile_sums[t];
    }
    // Expand to length n so we can reuse the 2-input helper without changing your framework.
    let expanded = cpu_expand_offsets(&tile_prefix_excl, tile, n);

    // Pass B: add offsets (GPU).
    let final_out = run_compute_two_inputs::<f32>(&ctx, ADD_WGSL, "add2", &scanned, &expanded, 256);

    // Spot checks.
    for &idx in &[0usize, 1, tile-1, tile, n-1] {
        let exp = (idx + 1) as f32;
        assert!((final_out[idx] - exp).abs() < 1e-4, "mismatch at {idx}");
    }
    println!("OK: scan computed for n={} (two-pass)", n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_scan_correctness() {
        let ctx = GpuContext::new_blocking().unwrap();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scanned = run_compute_single_input::<f32>(&ctx, SCAN_TILE_WGSL, "scan_tile", &a, 4);
        // Build CPU tile-fixup.
        let n = a.len();
        let tile = 4usize;
        let tiles = (n + tile - 1) / tile;
        let mut tile_sums = vec![0.0f32; tiles];
        for t in 0..tiles {
            let end = ((t + 1) * tile).min(n);
            tile_sums[t] = scanned[end - 1];
        }
        let mut excl = vec![0.0f32; tiles];
        let mut acc = 0.0f32;
        for t in 0..tiles { excl[t] = acc; acc += tile_sums[t]; }
        let expanded = cpu_expand_offsets(&excl, tile, n);
        let out = run_compute_two_inputs::<f32>(&ctx, ADD_WGSL, "add2", &scanned, &expanded, 4);

        // CPU truth
        let mut truth = vec![0.0; n];
        let mut s = 0.0;
        for i in 0..n { s += a[i]; truth[i] = s; }
        for i in 0..n { assert!((out[i]-truth[i]).abs() < 1e-5); }
    }
}
