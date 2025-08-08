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
//! back to a host‑accessible buffer.

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
/// The shader must declare two bindings: a read‑only storage buffer
/// for the input data at binding 0 and a read/write storage buffer
/// for the output data at binding 1.  The number of elements in
/// `input` determines how many workgroups are dispatched.  If the
/// length is not a multiple of `workgroup_size`, the final
/// workgroup will contain unused invocations; the shader should
/// guard against out‑of‑bounds accesses.  The `entry_point` must
/// match the name of your `@compute` function in the WGSL source.
pub fn run_compute_single_input<T: Pod + Copy>(
    context: &GpuContext,
    shader_source: &str,
    entry_point: &str,
    input: &[T],
    workgroup_size: u32,
) -> Vec<T> {
    assert!(!input.is_empty(), "input slice must not be empty");
    // Compile the shader module from WGSL source.  Any syntax errors
    // will be reported at runtime.  Shader modules are cached by the
    // device internally.
    let module = context
        .device
        .create_shader_module(ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
    // Create GPU buffers.  The input buffer uses only STORAGE
    // capability because we upload its contents via the queue.  The
    // output buffer allows COPY_SRC so that we can copy into the
    // download buffer later.
    let input_buffer = GpuBuffer::<T>::from_slice(context, input, BufferUsages::empty());
    let output_buffer = GpuBuffer::<T>::new_output(context, input.len(), BufferUsages::empty());
    let download_buffer = GpuBuffer::<T>::new_download(context, input.len());
    // Describe the bind group layout.  Two entries: input and output.
    let element_size = std::mem::size_of::<T>() as u64;
    // NonZeroU64 requirement: element_size > 0 because T: Pod implies
    // non‑zero sized type.
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
    // Create a bind group tying our buffers to the bindings.
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
    // Build the pipeline layout and the compute pipeline.
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
    // Record commands into a command encoder.
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

        // ------------- NEW 2-D DISPATCH LOGIC -------------
        let limits = context.device.limits();
        let total_groups = ((input_buffer.len as u32) + workgroup_size - 1) / workgroup_size;
        let (groups_x, groups_y) =
            split_workgroups(total_groups, limits.max_compute_workgroups_per_dimension);

        cpass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    // Copy the output buffer into the download buffer so that we can map
    // it on the CPU.  Both buffers have equal length in bytes.
    encoder.copy_buffer_to_buffer(
        &output_buffer.buffer,
        0,
        &download_buffer.buffer,
        0,
        (input_buffer.len * std::mem::size_of::<T>()) as u64,
    );
    // Submit the command buffer to the queue.  We deliberately drop
    // the returned submission index because we don't need to track
    // fences here.
    let command_buffer = encoder.finish();
    context.queue.submit([command_buffer]);
    // Map the download buffer and read back the results.
    download_buffer.read_to_vec(context)
}

/// Dispatch a compute shader with two input buffers and return the results.
///
/// The shader must declare three bindings: two read‑only storage
/// buffers at bindings 0 and 1, and a read/write storage buffer at
/// binding 2.  Each input buffer must contain the same number of
/// elements.  The returned vector will have that same length.  The
/// behaviour of the shader outside this contract is unspecified.
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