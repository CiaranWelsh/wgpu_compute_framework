//! A lightweight framework for running compute workloads on the GPU using
//! [wgpu](https://github.com/gfx-rs/wgpu).  The goal of this crate is to
//! simplify sending data to the GPU, executing a compute shader on that
//! data, and retrieving the results back to the CPU.  While the
//! framework is intentionally minimal, it demonstrates how to build a
//! high‑throughput pipeline for data processing that incurs little
//! overhead beyond the actual computation.  The provided API is
//! synchronous and blocking: it waits for the GPU to complete before
//! returning results.  For asynchronous usage you can extract the
//! underlying `wgpu::Device` and `wgpu::Queue` and integrate the
//! framework into your own event loop.

pub mod context;
pub mod buffer;
pub mod compute;

// Re‑export the most common types at the crate root so that users can
// simply `use wgpu_compute_framework::*;`.
pub use context::GpuContext;
pub use buffer::{GpuBuffer};
// Re-export helper functions so users can import directly from the crate root.
pub use compute::{
    run_compute_single_input,
    run_compute_two_inputs,
    run_compute_single_input_custom_output,
};