# GPU Compute Worksheet — “Do It Yourself” Progression (WGSL + Rust/wgpu)

**Opinionated take:** you’ll learn fastest by iterating on *simple kernels* and measuring the blast radius of small changes. These exercises climb from 1D indexing → shared memory → 2D tiling → data hazards → 3D mapping → atomics and reductions. Keep code changes minimal; add focused tests; write down what you expected and what actually happened.

**Assumptions**
- You have the examples: `01_double.rs`, `02_vector_add_benchmark.rs`, `03_matrix_multiplication.rs`, `04_1d_stencil.rs`, `05_prefix_sum_block_scan.rs`, `06_2d_convolution.rs`, `07_transpose_tiled.rs`.
- Your host framework exports `GpuContext`, `run_compute_single_input`, and `run_compute_two_inputs`.
- You can run `cargo test` and `cargo run --example ...` or as binaries.
- When in doubt, prefer correctness-first, then make one performance tweak at a time.

**Testing stance (what I expect you to write, minimally per exercise)**
- A small `#[cfg(test)]` unit test covering a tiny synthetic input.
- Comment in the test describing **purpose**, **assumptions**, **expected result**, and which **requirement** the test targets.
- For perf-focused items, capture a simple timing (wall-clock around dispatch is fine for now) and record results in a markdown table in the test file comment.

---

## Exercise 1 — Indexing sanity (1D)
**Goal:** Prove to yourself that `global_invocation_id.x` maps to linear indices as you think.

**Start from:** `04_1d_stencil.rs`.

**Tasks**
- Add a new “identity copy” kernel (same bindings) that writes `out[i] = inp[i]` and early-returns on `i >= n`.
- In a test, feed `[10., 20., 30., 40.]` and assert equality, then oversubscribe threads (e.g., workgroup size 256, dispatch enough groups for 1024 threads) and assert correctness still holds.

**Acceptance**
- Unit test passes; include a comment explaining why oversubscription + bounds checks is safe.

---

## Exercise 2 — Workgroup size sweep (1D)
**Goal:** Understand the effect of workgroup size on a bandwidth-bound kernel.

**Start from:** `04_1d_stencil.rs` (blur kernel).

**Tasks**
- Run with workgroup sizes: 32, 64, 128, 256.
- Record runtime for `n = 1_000_000` and `n = 50_000_000`.
- Keep the same dispatch math (ceil-div) and bounds check.

**Acceptance**
- Table of timings. Pick the fastest config; explain (briefly) why it might be best on your device (warp/wave size, occupancy).

---

## Exercise 3 — Boundary modes (1D)
**Goal:** Implement **clamp**, **wrap**, and **mirror** boundary handling and verify each against a CPU reference.

**Start from:** `04_1d_stencil.rs`.

**Tasks**
- Parameterize the WGSL kernel using an `override` constant (e.g. `MODE: u32`) for boundary mode.
- Add CPU reference functions for the three modes.
- Tiny tests per mode with handcrafted vectors that hit boundaries.
[02_vector_add_benchmark.rs](02_vector_add_benchmark.rs)
**Acceptance**
- All modes match CPU within `1e-6`. Include a comment diagramming indices used for the first and last output elements.

---

## Exercise 4 — Wider stencil (1D, 5-point)
**Goal:** Scale the same pattern to a 5-tap weighted blur.

**Start from:** `04_1d_stencil.rs`.

**Tasks**
- Implement weights `[1, 4, 6, 4, 1]/16` (Pascal row) in WGSL and CPU.
- Keep coalesced reads; still one thread → one element.

**Acceptance**
- Unit test on small arrays + a random array (seeded).

---

## Exercise 5 — f16 variant (if supported)
**Goal:** Explore precision/perf trade-offs.

**Start from:** Exercise 4’s kernel.

**Tasks**
- Add an `f16` variant (guard with feature detection on the device). Fall back to `f32` if unsupported.
- Compare runtime and max absolute error vs `f32` result.

**Acceptance**
- Report speedup (if any) and error; explain whether it’s acceptable for your use-case.

---

## Exercise 6 — Strided access (AoS vs SoA thought experiment)
**Goal:** Feel why memory coalescing matters.

**Start from:** `04_1d_stencil.rs`.

**Tasks**
- Add a kernel that reads every `k`-th element (e.g., `inp[i*k]` modulo `n`) to simulate poor locality.
- Sweep `k ∈ {1, 2, 4, 8, 16}` and time it.

**Acceptance**
- Table of k vs time; 1–2 sentence explanation tying slowdowns to non-coalesced reads.

---

## Exercise 7 — Inclusive → Exclusive scan
**Goal:** Adjust the algorithm spec without rewriting everything.

**Start from:** `05_prefix_sum_block_scan.rs` (two-pass approach).

**Tasks**
- Make the **per-tile** scan produce **exclusive** results (shift by one) but keep the overall two-pass structure.
- Update CPU fix-up accordingly.

**Acceptance**
- Unit test against a CPU exclusive scan on a small vector with varied values (including zeros and negatives).

---

## Exercise 8 — Parameterize tile size (scan)
**Goal:** Make kernels configurable without code duplication.

**Start from:** `05_prefix_sum_block_scan.rs`.

**Tasks**
- Replace hard-coded `TILE = 128` with a WGSL `override TILE: u32`.
- Drive it from the host using a `PipelineConstant` (specialization constant) or by string substitution if your framework doesn’t support overrides yet.
- Try TILE ∈ {64, 128, 256}.

**Acceptance**
- Same results across TILE values; capture timings and pick a winner with a short rationale.

---

## Exercise 9 — Fully on-GPU scan (no CPU fix-up)
**Goal:** Add a second GPU pass to scan the tile-sums buffer and then add back to the main buffer.

**Start from:** `05_prefix_sum_block_scan.rs`.

**Tasks**
- Emit per-tile sums to a small buffer during Pass A.
- Implement Pass A2: scan that small buffer on GPU (can reuse the same kernel on a smaller input).
- Implement Pass B: add scanned tile offsets to the big buffer (already present).
- Keep host changes minimal; reuse existing helpers.

**Acceptance**
- Unit test vs CPU scan; timings vs the CPU-assisted version. Note any overhead from extra passes.

---

## Exercise 10 — Segmented scan (flags)
**Goal:** Introduce a control dimension without blowing up divergence.

**Start from:** Exercise 9’s scan.

**Tasks**
- Provide a second input buffer `flags[i] ∈ {0,1}`; when `flags[i]==1`, start a new segment (reset the prefix).
- Implement per-tile segmented scan (you may treat segment boundaries that cross tiles as a second-pass fix-up, similar to Exercise 9).

**Acceptance**
- Unit tests: several tiny cases with segments at tile edges and within tiles.

---

## Exercise 11 — 2D naive vs separable blur
**Goal:** Reduce global memory traffic via separability.

**Start from:** `06_2d_convolution.rs`.

**Tasks**
- Implement two-pass separable blur (horizontal then vertical) using 1D 3-tap kernels.
- Compare with the 3×3 naive single-pass kernel for correctness and time.

**Acceptance**
- Max absolute diff < 1e-5. Provide a timing table and a sentence on why separable is faster (fewer reads).

---

## Exercise 12 — Tiled blur with variable halo
**Goal:** Generalize tile + halo math.

**Start from:** `06_2d_convolution.rs` (tiled).

**Tasks**
- Increase radius to `R=2` (5×5). Update shared tile shape to include halo.
- Add assertions/comments showing the shared memory size math.
- Run for (W,H) = (2048,2048).

**Acceptance**
- Matches naive/blessed CPU; note the change in shared memory usage and whether it throttles occupancy.

---

## Exercise 13 — Sobel edge detector
**Goal:** Implement a slightly more branchy stencil and keep it coalesced.

**Start from:** `06_2d_convolution.rs` (tiled).

**Tasks**
- Implement Sobel X and Y, output gradient magnitude.
- Keep boundaries clamped; prefer shared-memory reads like the blur.

**Acceptance**
- Unit test on a 5×5 pattern with a known edge; spot-check a few pixels; visually sanity-check by printing a small grid in the test output.

---

## Exercise 14 — Tile shape exploration (2D)
**Goal:** Understand why 16×16 isn’t always best.

**Start from:** `06_2d_convolution.rs` (tiled).

**Tasks**
- Try (TX,TY) ∈ {(8,32), (16,16), (32,8)} with `R=1` and `R=2`.
- Keep total threads per workgroup between 128 and 256 where possible.
- Record runtimes.

**Acceptance**
- A table: tile shape vs time for both radii; a short note linking best shape to memory access pattern and occupancy constraints.

---

## Exercise 15 — Transpose without padding (bank conflicts)
**Goal:** See the cost of shared-memory bank conflicts.

**Start from:** `07_transpose_tiled.rs`.

**Tasks**
- Remove the `+1` padding column in shared memory and re-run timings.
- Keep everything else identical.

**Acceptance**
- Same correctness; likely worse perf. Write a 2–3 sentence explanation of bank conflicts and why padding helps.

---

## Exercise 16 — In-place transpose (square matrices only)
**Goal:** Perform an in-place transpose safely with tiling.

**Start from:** `07_transpose_tiled.rs`.

**Tasks**
- For square matrices, implement in-place using block swaps (off-diagonal tiles swap; diagonal tiles transpose in-place).
- Use two dispatches if simpler; correctness over cleverness.

**Acceptance**
- Unit test on 8×8 and 17×17. No temporary full-size buffer used.

---

## Exercise 17 — 3D box blur (first 3D dispatch)
**Goal:** When 3D makes sense.

**Start from:** `06_2d_convolution.rs` (tiled idea) and generalize.

**Tasks**
- Treat input as (D,H,W); implement a radius-1 3D box blur using `@workgroup_size(TX,TY,TZ)` with small tiles.
- Add bounds checks; discuss memory footprint.

**Acceptance**
- Unit test on a 4×3×2 volume; compare to CPU reference; note how dispatch dims map to indices.

---

## Exercise 18 — Histogram with atomics (contention & privatization)
**Goal:** Intro to atomics and reducing contention.

**Tasks**
- Given a grayscale image (values 0–255 as `u32`), compute a 256-bin histogram.
- First do the *naive* version: each thread `atomicAdd` into a single global array.
- Then **privatize**: one histogram per workgroup in shared memory, then atomically merge to global once per bin.

**Acceptance**
- Correctness: sum of all bins equals number of pixels.
- Timing: privatized version faster than naive for large images; include a short explanation about contention.

---

## Exercise 19 — Parallel reduction (sum) with two-pass design
**Goal:** Classic reduction pattern.

**Tasks**
- Implement per-workgroup reduction to partial sums buffer; then a second pass to reduce partials to one value.
- Compare against CPU `sum()`.

**Acceptance**
- Numeric match within tolerance; timing vs scan-based “sum(last element)”. Note pros/cons of reduction vs scan for sums.

---

## Exercise 20 — Parameter sweep harness (bring it together)
**Goal:** Systematic exploration mindset.

**Tasks**
- Build a small harness (reuse your `02_vector_add_benchmark.rs` patterns) that sweeps workgroup sizes and tile shapes for: 1D blur, 2D tiled blur (R=1,2), transpose.
- Emit a CSV with columns: kernel, N/W/H, WG_SIZE/TILE, runtime, bytes_moved, GB/s, notes.
- Keep changes minimal; don’t over-architect.

**Acceptance**
- CSV produced and checked into repo; a short “Findings.md” with 3–5 bullets on what settings worked best and why.

---

## Hints & Guardrails
- If something unexpectedly slows down, check: **occupancy, register pressure, shared memory usage, bank conflicts, divergence, coalescing, atomics contention, precision issues**. (See your `UNKNOWN_UNKNOWNS.md` for quick definitions.)
- Always add a bounds check before writing. Oversubscription is a feature, not a bug.
- Prefer 1 pass → test → 1 tweak → test. Avoid multi-tweak debugging.
- When comparing floats, use a small epsilon and document it in the test.

## Optional “extra credit” directions
- Add device limit checks (max invocations per workgroup, max shared memory per workgroup) and print them at startup.
- Try fp16 for the 2D blur and transpose, measure speed & error.
- Introduce a “debug” kernel path that writes intermediate tiles to a buffer you can inspect on CPU.

---

Good luck. If you get stuck, bring me (a) your kernel, (b) a tiny failing input and expected output, and (c) your current workgroup/tile config—then we’ll debug like systems engineers, not magicians.
