//! Typed GPU buffers and host readback utilities.
//!
//! This module defines a [`GpuBuffer`] wrapper around [`wgpu::Buffer`]
//! that tracks the number of typed elements stored in the buffer and
//! provides convenience methods for uploading and downloading data.
//! The buffer itself does not maintain ownership of the CPU data; it
//! merely references GPU memory.  All interactions with the GPU are
//! performed through a [`crate::GpuContext`].

use std::num::NonZeroU64;

use bytemuck::{cast_slice, Pod};
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use crate::GpuContext;

/// A typed GPU buffer.
///
/// This struct wraps a `wgpu::Buffer` together with the element
/// length and a phantom type parameter.  The length records how many
/// elements of type `T` are stored in the buffer.  Note that the
/// underlying buffer size in bytes is `len * std::mem::size_of::<T>()`.
pub struct GpuBuffer<T: Pod> {
    pub buffer: Buffer,
    pub len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Pod> GpuBuffer<T> {
    /// Create a new storage buffer from a slice of data.
    ///
    /// The buffer will have usage `STORAGE | COPY_DST` by default.
    /// Additional usages can be passed in via the `usage` parameter.
    /// When reading back from the GPU you must ensure that the buffer
    /// includes the `COPY_SRC` usage flag.
    pub fn from_slice(
        context: &GpuContext,
        data: &[T],
        usage: BufferUsages,
    ) -> Self {
        let bytes = cast_slice(data);
        let buffer = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_buffer_input"),
            size: bytes.len() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | usage,
            mapped_at_creation: false,
        });
        // Write the contents into the buffer via a queue write.  This
        // avoids requiring the `MAP_WRITE` usage flag.  Note that
        // writing immediately after creation is safe because the GPU
        // has not yet seen the buffer.
        context.queue.write_buffer(&buffer, 0, bytes);
        Self {
            buffer,
            len: data.len(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new output buffer of `len` elements.  The buffer will
    /// have usage flags `STORAGE` and `COPY_SRC` so that it can be
    /// bound to a compute shader and read back to the host.  If you
    /// intend to copy data into the buffer you should add the
    /// `COPY_DST` usage via the `usage` parameter.
    pub fn new_output(context: &GpuContext, len: usize, usage: BufferUsages) -> Self {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let buffer = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_buffer_output"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a new download buffer sized to hold `len` elements.  The
    /// buffer is mapped for reading by the CPU and has usage flags
    /// `COPY_DST` and `MAP_READ`.  It cannot be bound directly to a
    /// shader.
    pub fn new_download(context: &GpuContext, len: usize) -> Self {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let buffer = context.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_buffer_download"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Read the contents of the buffer back to the CPU.
    ///
    /// The buffer must have been filled with data and copied to a
    /// download buffer with the `MAP_READ` usage.  This method will
    /// block the current thread until the GPU has finished writing to
    /// the buffer and the data is ready to be read.  After reading
    /// the data the buffer is unmapped.
    pub fn read_to_vec(&self, context: &GpuContext) -> Vec<T> {
        // Create a slice covering the entire buffer.
        let slice = self.buffer.slice(..);
        // Begin an asynchronous mapping operation.  The callback is
        // unused because we synchronously poll the device below.
        slice.map_async(wgpu::MapMode::Read, |_| {});

        // Block until the mapping is ready.  PollType::Wait keeps the
        // CPU thread idle until the device has completed all work.
        context
            .device
            .poll(wgpu::PollType::Wait)
            .expect("device polling failed");

        // Obtain the mapped range as a byte slice.  We can cast this
        // slice to our target element type because the buffer was
        // created with the correct size and alignment.
        let data = slice.get_mapped_range();
        let result: Vec<T> = cast_slice(&data).to_vec();
        // Explicitly drop the view before unmapping.  Dropping the
        // mapped range view releases the borrow.
        drop(data);
        self.buffer.unmap();
        result
    }
}