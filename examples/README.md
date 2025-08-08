# Next Baby Steps with Compute Shaders (WGSL + Rust/wgpu)

You asked for the next progressive set of examples _after_ `01_double`, `02_vector_add_benchmark`, and `03_matrix_multiplication`.
These extend your existing `wgpu_compute_framework` crate (no changes needed) and focus on two things:
1) **How to code shaders (WGSL)** — syntax, bindings, workgroup memory, barriers, atomics.
2) **How to map data to workgroups** — 1D vs 2D vs 3D; tiling; when to choose which; and what to watch out for.

## What’s included

- `04_1d_stencil.rs` — 1D 3-point blur (moving average). Simple 1D mapping; boundary handling. 
- `05_prefix_sum_block_scan.rs` — Workgroup-shared-memory **scan** (prefix sum) in two passes (GPU + small CPU assist). Introduces `var<workgroup>` and barriers.
- `06_2d_convolution.rs` — 3×3 image blur two ways: naive global reads, then **tiled** with shared memory. Shows 2D workgroups.
- `07_transpose_tiled.rs` — Cache-friendly matrix **transpose** using padded tiles to avoid bank conflicts.
- `WGSL_101.md` — short WGSL primer.
- `WORKGROUP_MAPPING_GUIDE.md` — 1D/2D/3D mapping tactics, tile sizing, occupancy heuristics.
- `UNKNOWN_UNKNOWNS.md` — the checklist of “stuff to learn next” with quick definitions.

All examples assume your existing crate exports:
`GpuContext`, `run_compute_single_input`, `run_compute_two_inputs`.

> **Run:** `cargo run --example 04_1d_stencil` etc., or compile each file as a binary target. 
> These are self-contained binaries; tests live at bottom of each file (`#[cfg(test)]`).

