//! GPU context initialization.
//!
//! This module provides a thin wrapper around wgpu's instance, adapter,
//! device and queue objects.  Creating a [`GpuContext`] lazily
//! instantiates the GPU and prepares it for compute workloads.  The
//! `new_blocking` constructor hides the asynchronous nature of
//! requesting an adapter and device by using the [`pollster`]
//! crate.

use wgpu::{Adapter, Device, Instance, Queue};

/// A GPU context encapsulates all state needed to submit compute work.
///
/// The context holds on to the `Instance`, `Adapter`, `Device` and
/// `Queue`.  Those types have internal reference counting so they can
/// cheaply be cloned if you need multiple references.  Creating a
/// context will pick the default high performance adapter on the
/// system.  If no adapter is available or it does not support compute
/// shaders, an error is returned.
pub struct GpuContext {
    /// The global GPU instance.  Keeps track of available backends and
    /// manages surface creation on windowed platforms.  In headless
    /// compute applications the instance is still required to request
    /// an adapter.
    pub instance: Instance,
    /// The physical device selected for computation.  The adapter
    /// exposes downlevel capabilities and limits which can be
    /// inspected by the caller.
    pub adapter: Adapter,
    /// Logical device used to create resources and command encoders.
    pub device: Device,
    /// Command submission queue used to send recorded command buffers
    /// to the GPU.
    pub queue: Queue,
}

impl GpuContext {
    /// Create a new GPU context synchronously.
    ///
    /// This function will block the current thread while waiting for
    /// the asynchronous adapter and device requests to finish.  If you
    /// require asynchronous initialization, use the [`Self::new_async`]
    /// method instead.
    pub fn new_blocking() -> Result<Self, String> {
        // Create an instance with default backends.  In wgpu 26 the
        // instance is constructed via an `InstanceDescriptor`, which
        // specifies the backends and other global configuration.  The
        // default enables all supported backends.
        let instance = Instance::new(&wgpu::InstanceDescriptor::default());
        // Request an adapter that supports compute.  We don't specify
        // surface or power preference here because this is a headless
        // framework.  The first high performance adapter is used.
        // Request an adapter.  The default options pick a high performance
        // adapter if available and are sufficient for compute workloads.
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .map_err(|e| "Unable to find a suitable GPU adapter".to_string())?;
        // Verify that the adapter supports compute shaders.  Downlevel
        // devices may not support compute on all backends; abort early
        // if unsupported.
        let capabilities = adapter.get_downlevel_capabilities();
        if !capabilities.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
            return Err("Selected adapter does not support compute shaders".into());
        }
        // Request a logical device and a queue.  We require no special
        // features or limits beyond the defaults.  A second parameter
        // allows capturing a GPU trace; we pass `None` to disable
        // tracing.
        // Request a logical device and queue.  In wgpu 26 the
        // `DeviceDescriptor` explicitly separates required features and
        // limits, and introduces `memory_hints` and `trace` fields.  We
        // require no special features and use downlevel defaults for
        // limits.  `MemoryHints::MemoryUsage` selects balanced memory
        // allocation behaviour, and `Trace::Off` disables GPU trace
        // capture.  Note that `request_device` in this release takes
        // only the descriptor.
        let avail = adapter.features();
        let mut features = wgpu::Features::empty();
        if avail.contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu_compute_device"),
            required_features: features,
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("Failed to create GPU device: {e}"))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Create a new GPU context asynchronously.
    ///
    /// This function returns a future that resolves to a new context.
    /// It can be awaited inside an asynchronous runtime.  See
    /// [`Self::new_blocking`] for a synchronous alternative.
    pub async fn new_async() -> Result<Self, String> {
        let instance = Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| "Unable to find a suitable GPU adapter".to_string())?;
        let capabilities = adapter.get_downlevel_capabilities();
        if !capabilities.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
            return Err("Selected adapter does not support compute shaders".into());
        }
        let avail = adapter.features();
        let mut features = wgpu::Features::empty();
        if avail.contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("wgpu_compute_device"),
                    required_features: features,
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .map_err(|e| format!("Failed to create GPU device: {e}"))?;
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}