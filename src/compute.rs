//! Helpers for running compute shaders.
//!
//! This module provides two convenience functions for dispatching
//! compute shaders on the GPU.  They support either a single input
//! buffer (producing one output buffer of the same length) or two
//! input buffers of equal length (again producing one output buffer).
//! The functions encapsulate the wgpu boilerplate of creating shader
//! modules, bind groups, pipelines and command buffers.  Both
//! functions block the current thread until the GPU has completed
//! executing the compute workload and the results have been copied
//! back to a host-accessible buffer.

use std::num::NonZeroU64;

use bytemuck::Pod;
use wgpu::{self, util::DeviceExt, ShaderModuleDescriptor, ShaderSource, BufferUsages};

use crate::{buffer::GpuBuffer, context::GpuContext};

/// Calculate an (x, y) workgroup grid that covers `total_groups`
/// workgroups without exceeding the per-dimension limit.
fn split_workgroups(total_groups: u32, limit: u32) -> (u32, u32) {
    if total_groups <= limit {
        (total_groups, 1)
    } else {
        let x = limit;
        let y = (total_groups + limit - 1) / limit; // ceiling-divide
        (x, y)
    }
}

/// Dispatch a compute shader with a single input buffer and return the
/// results.
///
/// The shader must declare two bindings: a read-only storage buffer
/// for the input data at binding 0 and a read/write storage buffer
/// for the output data at binding 1.  The number of elements in
/// `input` determines how many workgroups are dispatched.  If the
/// length is not a multiple of `workgroup_size`, the final
/// workgroup will contain unused invocations; the shader should
/// guard against out-of-bounds accesses.  The `entry_point` must
/// match the name of your `@compute` function in the WGSL source.
pub fn run_compute_single_input<T: Pod + Copy>(
    context: &GpuContext,
    shader_source: &str,
    entry_point: &str,
    input: &[T],
    workgroup_size: u32,
) -> Vec<T> {
    assert!(!input.is_empty(), "input slice must not be empty");

    let module = context
        .device
        .create_shader_module(ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

    let input_buffer = GpuBuffer::<T>::from_slice(context, input, BufferUsages::empty());
    let output_buffer = GpuBuffer::<T>::new_output(context, input.len(), BufferUsages::empty());
    let download_buffer = GpuBuffer::<T>::new_download(context, input.len());

    let element_size = std::mem::size_of::<T>() as u64;
    let min_binding_size = NonZeroU64::new(element_size).unwrap();

    let bind_group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            },
        ],
    });

    let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("compute_encoder") });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        let limits = context.device.limits();
        let total_groups = ((input_buffer.len as u32) + workgroup_size - 1) / workgroup_size;
        let (groups_x, groups_y) =
            split_workgroups(total_groups, limits.max_compute_workgroups_per_dimension);

        cpass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    encoder.copy_buffer_to_buffer(
        &output_buffer.buffer,
        0,
        &download_buffer.buffer,
        0,
        (input_buffer.len * std::mem::size_of::<T>()) as u64,
    );

    let command_buffer = encoder.finish();
    context.queue.submit([command_buffer]);
    download_buffer.read_to_vec(context)
}

/// Dispatch a compute shader with two input buffers and return the results.
///
/// The shader must declare three bindings: two read-only storage
/// buffers at bindings 0 and 1, and a read/write storage buffer at
/// binding 2.  Each input buffer must contain the same number of
/// elements.  The returned vector will have that same length.
pub fn run_compute_two_inputs<T: Pod + Copy>(
    context: &GpuContext,
    shader_source: &str,
    entry_point: &str,
    input_a: &[T],
    input_b: &[T],
    workgroup_size: u32,
) -> Vec<T> {
    assert_eq!(input_a.len(), input_b.len(), "input slices must have equal length");
    assert!(!input_a.is_empty(), "input slices must not be empty");

    let module = context
        .device
        .create_shader_module(ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

    let buffer_a = GpuBuffer::<T>::from_slice(context, input_a, BufferUsages::empty());
    let buffer_b = GpuBuffer::<T>::from_slice(context, input_b, BufferUsages::empty());
    let output_buffer = GpuBuffer::<T>::new_output(context, input_a.len(), BufferUsages::empty());
    let download_buffer = GpuBuffer::<T>::new_download(context, input_a.len());

    let element_size = std::mem::size_of::<T>() as u64;
    let min_binding_size = NonZeroU64::new(element_size).unwrap();

    let bind_group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(min_binding_size),
                },
                count: None,
            },
        ],
    });

    let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("compute_encoder") });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        let limits = context.device.limits();
        let total_groups = ((input_a.len() as u32) + workgroup_size - 1) / workgroup_size;
        let (groups_x, groups_y) = split_workgroups(total_groups, limits.max_compute_workgroups_per_dimension);

        cpass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    encoder.copy_buffer_to_buffer(
        &output_buffer.buffer,
        0,
        &download_buffer.buffer,
        0,
        (input_a.len() * std::mem::size_of::<T>()) as u64,
    );

    let command_buffer = encoder.finish();
    context.queue.submit([command_buffer]);
    download_buffer.read_to_vec(context)
}

pub fn run_compute_single_input_with_params<In: Pod + Copy, P: Pod + Copy, Out: Pod + Copy>(
    context: &GpuContext,
    shader_source: &str,
    entry_point: &str,
    input: &[In],        // streaming hits
    params: &P,          // small, sent once (you can reuse the buffer; see example)
    out_len: usize,      // n_scan bins
    workgroup_size: u32, // must match @workgroup_size in WGSL
) -> Vec<Out> {
    assert!(out_len > 0, "out_len must be > 0");

    let module = context.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("vadf_shader"),
        source: ShaderSource::Wgsl(shader_source.into()),
    });

    let hits_buf   = GpuBuffer::<In>::from_slice(context, input, BufferUsages::empty());
    let params_buf = GpuBuffer::<P>::from_slice(context, std::slice::from_ref(params), BufferUsages::empty());
    let out_buf    = GpuBuffer::<Out>::new_output(context, out_len, BufferUsages::empty());
    let dl_buf     = GpuBuffer::<Out>::new_download(context, out_len);

    let in_size  = NonZeroU64::new(std::mem::size_of::<In>()  as u64).unwrap();
    let out_size = NonZeroU64::new(std::mem::size_of::<Out>() as u64).unwrap();
    let prm_size = NonZeroU64::new(std::mem::size_of::<P>()   as u64).unwrap();

    let bgl = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vadf_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(in_size),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(out_size),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(prm_size),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vadf_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("vadf_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vadf_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: hits_buf.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out_buf.buffer.as_entire_binding()  },
            wgpu::BindGroupEntry { binding: 2, resource: params_buf.buffer.as_entire_binding() },
        ],
    });

    let total_groups = ((hits_buf.len as u32) + workgroup_size - 1) / workgroup_size;
    let limits = context.device.limits();
    let (groups_x, groups_y) = split_workgroups(total_groups, limits.max_compute_workgroups_per_dimension);

    let mut encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vadf_enc") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("vadf_pass"), timestamp_writes: None });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    encoder.copy_buffer_to_buffer(
        &out_buf.buffer, 0, &dl_buf.buffer, 0,
        (out_len * std::mem::size_of::<Out>()) as u64,
    );
    context.queue.submit([encoder.finish()]);
    dl_buf.read_to_vec(context)
}

#[derive(Clone, Copy, Debug)]
pub struct GpuTiming {
    /// Kernel time in nanoseconds (GPU timestamps if supported; otherwise host wall clock).
    pub kernel_ns: f64,
    /// Total command submission + completion time in nanoseconds (host wall clock).
    pub submit_ns: f64,
    /// Bytes uploaded to the GPU for inputs (approx).
    pub bytes_upload: u64,
    /// Bytes read back from the GPU for outputs (approx).
    pub bytes_download: u64,
}

/// Run a single-input compute shader that produces a custom-length output,
/// and return (output, timing). Uses correct QUERY_RESOLVE handling (wgpu 0.26).
pub fn run_compute_single_input_custom_output_timed<T: bytemuck::Pod, Out: bytemuck::Pod>(
    context: &GpuContext,
    shader_source: &str,
    entry_point: &str,
    input: &[T],
    output_len: usize,
    workgroup_size: u32,
) -> (Vec<Out>, GpuTiming) {
    use wgpu::PipelineCompilationOptions;

    let device = &context.device;
    let queue = &context.queue;

    // Compile shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("timed_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Buffers
    let input_buffer = GpuBuffer::<T>::from_slice(context, input, BufferUsages::empty());
    let output_buffer = GpuBuffer::<Out>::new_output(context, output_len, BufferUsages::empty());
    let download_buffer = GpuBuffer::<Out>::new_download(context, output_len);

    // BGL
    let min_in = std::num::NonZeroU64::new(std::mem::size_of::<T>() as u64).unwrap();
    let min_out = std::num::NonZeroU64::new(std::mem::size_of::<Out>() as u64).unwrap();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("timed_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(min_in),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(min_out),
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("timed_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("timed_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("timed_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.buffer.as_entire_binding() },
        ],
    });

    // Dispatch sizing
    let total_invocations = input.len() as u32;
    let total_groups = (total_invocations + workgroup_size - 1) / workgroup_size;
    let limits = device.limits();
    let max_per_dim = limits.max_compute_workgroups_per_dimension;
    let (groups_x, groups_y) = if total_groups <= max_per_dim {
        (total_groups, 1)
    } else {
        let y = (total_groups + max_per_dim - 1) / max_per_dim;
        let x = (total_groups + y - 1) / y;
        (x, y)
    };

    // Timestamp support?
    let have_ts = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let mut kernel_ns = None;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("timed_enc") });

    // Create timestamp query set. Resolve into a buffer with QUERY_RESOLVE, then copy to a MAP_READ buffer.
    let (query_set, ts_resolve_buf, ts_download) = if have_ts {
        let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("ts_qs"),
            count: 2,
            ty: wgpu::QueryType::Timestamp,
        });
        // 2 * u64 = 16 bytes
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dl = GpuBuffer::<u64>::new_download(context, 2); // MAP_READ | COPY_DST
        (Some(qs), Some(resolve_buf), Some(dl))
    } else {
        (None, None, None)
    };

    // Begin pass with timestamp writes (wgpu 0.26)
    {
        let ts_writes = query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("timed_pass"),
            timestamp_writes: ts_writes,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(groups_x, groups_y, 1);
    }

    // Resolve timestamps into the QUERY_RESOLVE buffer, then copy to the mappable buffer.
    if let (Some(qs), Some(res_buf)) = (query_set.as_ref(), ts_resolve_buf.as_ref()) {
        encoder.resolve_query_set(qs, 0..2, res_buf, 0);
        if let Some(dl) = ts_download.as_ref() {
            encoder.copy_buffer_to_buffer(res_buf, 0, &dl.buffer, 0, 16);
        }
    }

    // Output readback
    encoder.copy_buffer_to_buffer(
        &output_buffer.buffer, 0,
        &download_buffer.buffer, 0,
        (output_len * std::mem::size_of::<Out>()) as u64,
    );

    // Submit + host timing
    let submit_t0 = std::time::Instant::now();
    let cb = encoder.finish();
    queue.submit([cb]);
    device.poll(wgpu::PollType::Wait);
    let submit_ns = submit_t0.elapsed().as_nanos() as f64;

    // GPU kernel ns (if supported)
    if let Some(ts_dl) = ts_download {
        let ts = ts_dl.read_to_vec(context);
        let period_ns = queue.get_timestamp_period() as f64;
        // 2 u64 timestamps: end - begin
        let dt = (ts[1] - ts[0]) as f64 * period_ns;
        kernel_ns = Some(dt);
    }

    // Fallback if no timestamps: use host time as kernel proxy
    let kernel_ns = kernel_ns.unwrap_or(submit_ns);

    let bytes_upload = (input.len() * std::mem::size_of::<T>()) as u64;
    let bytes_download = (output_len * std::mem::size_of::<Out>()) as u64;

    let out = download_buffer.read_to_vec(context);

    (out, GpuTiming { kernel_ns, submit_ns, bytes_upload, bytes_download })
}
