# WGSL 101 (Just Enough Shader Language)

## Minimal compute entry
```wgsl
@group(0) @binding(0) var<storage, read>  inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>,
        @builtin(workgroup_id)         wid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&inp)) { return; }
    out[i] = inp[i] * 2.0;
}
```

- `@group(0) @binding(n)` match host-side bind group layout.
- Storage buffers: `var<storage, read>` or `read_write`.
- Workgroup size = threads per **workgroup**; you dispatch many workgroups.
- Builtins you’ll use a lot:
  - `global_invocation_id` = unique thread id in the whole grid (x,y,z)
  - `local_invocation_id`  = lane id within a workgroup (0..workgroup_size-1 across x/y/z)
  - `workgroup_id`         = which workgroup this thread belongs to

## Workgroup shared memory
```wgsl
var<workgroup> tile: array<f32, 256>;
workgroupBarrier();
```

- Fast on-chip memory shared by threads in a workgroup.
- Use `workgroupBarrier()` to synchronize **reads/writes** to `var<workgroup>`.
- Size must be compile-time constant.

## Atomics (quick taste)
```wgsl
var<workgroup> counter: atomic<u32>;
atomicAdd(&counter, 1u);
```

- Atomics exist for `i32`/`u32`. Use sparingly; contention kills perf.
- Use them to accumulate into shared memory first, then flush to global.

## Indexing helpers
- `arrayLength(&buf)` reads the runtime length of a runtime-sized array in a storage buffer.
- Map multi-d indices to 1D: `idx = x + y*W + z*W*H`.

See the samples for complete, heavily-commented shaders.
