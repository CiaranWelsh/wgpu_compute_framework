# WORKGROUP_MAPPING_GUIDE.md

**Opinionated TL;DR:** Start simple — 1D for 1D data, 2D for images/matrices, and only go 3D when you truly have a 3D neighborhood (volumes). Prefer square-ish 2D tiles like 16×16 or 32×8. Keep total `@workgroup_size` in the 64–256 range unless you have a reason not to.

## When to use 1D / 2D / 3D
- **1D:** reductions, scans, elementwise ops, 1D stencils. Easiest to reason about.
- **2D:** images/matrices, 2D stencils/convolutions, transpose. Natural indexing & coalesced accesses.
- **3D:** volumes (medical CT, fluid sims). Only if neighbors exist in z, otherwise fake it with 2D×loops.

## Picking a workgroup size
- Use something near the GPU’s native wave/warp size times an integer: 32, 64, 128, 256 are safe bets.
- Start with **64 or 128** for 1D; **16×16** for 2D. Measure and adjust.
- Keep in mind device limits: `max_compute_invocations_per_workgroup` and `max_compute_workgroup_size_{x,y,z}`. wgpu exposes these via `device.limits()`.

## Grid sizing
- Total threads = ceil_div(N, WG_SIZE) × WG_SIZE. It’s fine to overshoot and guard with `if (i >= N) return;`.
- For 2D, use `(ceil_div(W, TX), ceil_div(H, TY))` workgroups with `@workgroup_size(TX, TY)`.

## Tiling & shared memory
- Load a **tile + halo** into `var<workgroup>`, `workgroupBarrier()`, compute from the tile, write out.
- For transpose, pad the tile width by +1 to avoid shared memory bank conflicts.

## Memory access patterns
- **Coalesced**: threads next to each other read/write adjacent elements.
- Avoid **divergent branches** inside warps/wavefronts.
- Minimize global memory traffic; prefer reusing data in shared memory.

## Back-of-the-envelope heuristics
- If each output element needs < 32 bytes of input and trivial math → likely **bandwidth-bound**. Aim for coalesced loads & tiling.
- If each output element does a lot of math → **compute-bound**. Increase occupancy, keep ALUs fed.
- If you use atomics heavily and it’s slow → **contention-bound**. Privatize per-workgroup, then reduce.
