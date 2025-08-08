# UNKNOWN_UNKNOWNS.md

Stuff you’ll bump into soon, with 1-liners:
- **Occupancy:** How many workgroups can run concurrently per SM/CU; driven by registers, shared memory, and workgroup size.
- **Register pressure:** Big kernels spill to memory → slow.
- **Shared memory bank conflicts:** Multiple lanes hitting same bank serialize; +1 padding trick helps (see transpose sample).
- **Memory coalescing:** Adjacent lanes should read adjacent addresses.
- **Divergence:** If threads in a warp take different branches, you serialize.
- **Subgroups (wave/warp intrinsics):** `subgroupBroadcast`, `subgroupAdd` (future WGSL features) let you avoid shared memory for some patterns.
- **Precision & denormals:** Mixing `f16/f32`; beware non-associativity of FP math when validating.
- **Asynchronous copies / copy engines:** Overlap transfers with compute where possible.
- **Pinned memory / staging buffers:** Host ↔ device transfers are not equal.
- **Dispatch vs draw / graphics interop:** Compute pipelines can feed graphics (and vice versa).

Skim this list when something “mysteriously” underperforms.
