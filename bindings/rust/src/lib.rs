//! Rust bindings for Psyne zero-copy messaging library
//!
//! Psyne provides high-performance, zero-copy messaging for AI/ML applications.
//! This crate provides safe Rust bindings to the Psyne C API.
//!
//! # Example
//!
//! ```no_run
//! use psyne::{Channel, ChannelMode, ChannelType};
//!
//! fn main() -> Result<(), psyne::Error> {
//!     // Initialize the library
//!     psyne::init()?;
//!
//!     // Create a channel
//!     let channel = Channel::create(
//!         "memory://demo",
//!         1024 * 1024,  // 1MB buffer
//!         ChannelMode::Spsc,
//!         ChannelType::Multi,
//!     )?;
//!
//!     // Send a message
//!     let data = b"Hello from Rust!";
//!     channel.send_data(data, 1)?;
//!
//!     // Receive a message
//!     let mut buffer = vec![0u8; 1024];
//!     if let Some((size, msg_type)) = channel.receive_data(&mut buffer, 0)? {
//!         println!("Received: {}", std::str::from_utf8(&buffer[..size]).unwrap());
//!     }
//!
//!     Ok(())
//! }
//! ```

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr;
use std::slice;
use std::time::Duration;
use thiserror::Error;

// Include the auto-generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Psyne error type
#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid argument")]
    InvalidArgument,
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Channel full")]
    ChannelFull,
    #[error("No message available")]
    NoMessage,
    #[error("Channel stopped")]
    ChannelStopped,
    #[error("Operation not supported")]
    Unsupported,
    #[error("I/O error")]
    IoError,
    #[error("Timeout")]
    Timeout,
    #[error("Unknown error")]
    Unknown,
    #[error("Invalid UTF-8")]
    InvalidUtf8(#[from] std::str::Utf8Error),
    #[error("Null pointer")]
    NullPointer,
}

impl From<psyne_error_t> for Error {
    fn from(err: psyne_error_t) -> Self {
        match err {
            psyne_error_PSYNE_ERROR_INVALID_ARGUMENT => Error::InvalidArgument,
            psyne_error_PSYNE_ERROR_OUT_OF_MEMORY => Error::OutOfMemory,
            psyne_error_PSYNE_ERROR_CHANNEL_FULL => Error::ChannelFull,
            psyne_error_PSYNE_ERROR_NO_MESSAGE => Error::NoMessage,
            psyne_error_PSYNE_ERROR_CHANNEL_STOPPED => Error::ChannelStopped,
            psyne_error_PSYNE_ERROR_UNSUPPORTED => Error::Unsupported,
            psyne_error_PSYNE_ERROR_IO => Error::IoError,
            psyne_error_PSYNE_ERROR_TIMEOUT => Error::Timeout,
            _ => Error::Unknown,
        }
    }
}

/// Result type for Psyne operations
pub type Result<T> = std::result::Result<T, Error>;

/// Channel synchronization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelMode {
    /// Single Producer, Single Consumer
    Spsc,
    /// Single Producer, Multiple Consumer
    Spmc,
    /// Multiple Producer, Single Consumer
    Mpsc,
    /// Multiple Producer, Multiple Consumer
    Mpmc,
}

impl From<ChannelMode> for psyne_channel_mode_t {
    fn from(mode: ChannelMode) -> Self {
        match mode {
            ChannelMode::Spsc => psyne_channel_mode_PSYNE_MODE_SPSC,
            ChannelMode::Spmc => psyne_channel_mode_PSYNE_MODE_SPMC,
            ChannelMode::Mpsc => psyne_channel_mode_PSYNE_MODE_MPSC,
            ChannelMode::Mpmc => psyne_channel_mode_PSYNE_MODE_MPMC,
        }
    }
}

/// Channel type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    /// Single message type
    Single,
    /// Multiple message types
    Multi,
}

impl From<ChannelType> for psyne_channel_type_t {
    fn from(typ: ChannelType) -> Self {
        match typ {
            ChannelType::Single => psyne_channel_type_PSYNE_TYPE_SINGLE,
            ChannelType::Multi => psyne_channel_type_PSYNE_TYPE_MULTI,
        }
    }
}

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None,
    Lz4,
    Zstd,
    Snappy,
}

impl From<CompressionType> for psyne_compression_type_t {
    fn from(typ: CompressionType) -> Self {
        match typ {
            CompressionType::None => psyne_compression_type_PSYNE_COMPRESSION_NONE,
            CompressionType::Lz4 => psyne_compression_type_PSYNE_COMPRESSION_LZ4,
            CompressionType::Zstd => psyne_compression_type_PSYNE_COMPRESSION_ZSTD,
            CompressionType::Snappy => psyne_compression_type_PSYNE_COMPRESSION_SNAPPY,
        }
    }
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub compression_type: CompressionType,
    pub level: i32,
    pub min_size_threshold: usize,
    pub enable_checksum: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            compression_type: CompressionType::None,
            level: 1,
            min_size_threshold: 128,
            enable_checksum: true,
        }
    }
}

/// Channel metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct Metrics {
    pub messages_sent: u64,
    pub bytes_sent: u64,
    pub messages_received: u64,
    pub bytes_received: u64,
    pub send_blocks: u64,
    pub receive_blocks: u64,
}

impl From<psyne_metrics_t> for Metrics {
    fn from(m: psyne_metrics_t) -> Self {
        Self {
            messages_sent: m.messages_sent,
            bytes_sent: m.bytes_sent,
            messages_received: m.messages_received,
            bytes_received: m.bytes_received,
            send_blocks: m.send_blocks,
            receive_blocks: m.receive_blocks,
        }
    }
}

/// Initialize the Psyne library
pub fn init() -> Result<()> {
    unsafe {
        let err = psyne_init();
        if err == psyne_error_PSYNE_OK {
            Ok(())
        } else {
            Err(err.into())
        }
    }
}

/// Cleanup the Psyne library
pub fn cleanup() {
    unsafe {
        psyne_cleanup();
    }
}

/// Get version string
pub fn version() -> &'static str {
    unsafe {
        let ptr = psyne_version();
        CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
    }
}

/// Channel handle
pub struct Channel {
    ptr: *mut psyne_channel_t,
    _phantom: PhantomData<psyne_channel_t>,
}

impl Channel {
    /// Create a new channel
    pub fn create(
        uri: &str,
        buffer_size: usize,
        mode: ChannelMode,
        channel_type: ChannelType,
    ) -> Result<Self> {
        let c_uri = CString::new(uri).map_err(|_| Error::InvalidArgument)?;
        let mut channel_ptr: *mut psyne_channel_t = ptr::null_mut();
        
        unsafe {
            let err = psyne_channel_create(
                c_uri.as_ptr(),
                buffer_size,
                mode.into(),
                channel_type.into(),
                &mut channel_ptr,
            );
            
            if err == psyne_error_PSYNE_OK {
                Ok(Self {
                    ptr: channel_ptr,
                    _phantom: PhantomData,
                })
            } else {
                Err(err.into())
            }
        }
    }
    
    /// Create a channel with compression
    pub fn create_compressed(
        uri: &str,
        buffer_size: usize,
        mode: ChannelMode,
        channel_type: ChannelType,
        compression: CompressionConfig,
    ) -> Result<Self> {
        let c_uri = CString::new(uri).map_err(|_| Error::InvalidArgument)?;
        let mut channel_ptr: *mut psyne_channel_t = ptr::null_mut();
        
        let c_compression = psyne_compression_config_t {
            type_: compression.compression_type.into(),
            level: compression.level,
            min_size_threshold: compression.min_size_threshold,
            enable_checksum: compression.enable_checksum,
        };
        
        unsafe {
            let err = psyne_channel_create_compressed(
                c_uri.as_ptr(),
                buffer_size,
                mode.into(),
                channel_type.into(),
                &c_compression,
                &mut channel_ptr,
            );
            
            if err == psyne_error_PSYNE_OK {
                Ok(Self {
                    ptr: channel_ptr,
                    _phantom: PhantomData,
                })
            } else {
                Err(err.into())
            }
        }
    }
    
    /// Stop the channel
    pub fn stop(&self) -> Result<()> {
        unsafe {
            let err = psyne_channel_stop(self.ptr);
            if err == psyne_error_PSYNE_OK {
                Ok(())
            } else {
                Err(err.into())
            }
        }
    }
    
    /// Check if channel is stopped
    pub fn is_stopped(&self) -> Result<bool> {
        let mut stopped = false;
        unsafe {
            let err = psyne_channel_is_stopped(self.ptr, &mut stopped);
            if err == psyne_error_PSYNE_OK {
                Ok(stopped)
            } else {
                Err(err.into())
            }
        }
    }
    
    /// Get channel URI
    pub fn uri(&self) -> Result<String> {
        let mut buffer = vec![0u8; 256];
        unsafe {
            let err = psyne_channel_get_uri(
                self.ptr,
                buffer.as_mut_ptr() as *mut i8,
                buffer.len(),
            );
            
            if err == psyne_error_PSYNE_OK {
                let c_str = CStr::from_ptr(buffer.as_ptr() as *const i8);
                Ok(c_str.to_string_lossy().into_owned())
            } else {
                Err(err.into())
            }
        }
    }
    
    /// Get channel metrics
    pub fn metrics(&self) -> Result<Metrics> {
        let mut metrics = psyne_metrics_t {
            messages_sent: 0,
            bytes_sent: 0,
            messages_received: 0,
            bytes_received: 0,
            send_blocks: 0,
            receive_blocks: 0,
        };
        
        unsafe {
            let err = psyne_channel_get_metrics(self.ptr, &mut metrics);
            if err == psyne_error_PSYNE_OK {
                Ok(metrics.into())
            } else {
                Err(err.into())
            }
        }
    }
    
    /// Send raw data
    pub fn send_data(&self, data: &[u8], msg_type: u32) -> Result<()> {
        unsafe {
            let err = psyne_send_data(
                self.ptr,
                data.as_ptr() as *const std::ffi::c_void,
                data.len(),
                msg_type,
            );
            
            if err == psyne_error_PSYNE_OK {
                Ok(())
            } else {
                Err(err.into())
            }
        }
    }
    
    /// Receive raw data
    pub fn receive_data(
        &self,
        buffer: &mut [u8],
        timeout: Duration,
    ) -> Result<Option<(usize, u32)>> {
        let mut received_size = 0;
        let mut msg_type = 0;
        let timeout_ms = timeout.as_millis() as u32;
        
        unsafe {
            let err = psyne_receive_data(
                self.ptr,
                buffer.as_mut_ptr() as *mut std::ffi::c_void,
                buffer.len(),
                &mut received_size,
                &mut msg_type,
                timeout_ms,
            );
            
            match err {
                psyne_error_PSYNE_OK => Ok(Some((received_size, msg_type))),
                psyne_error_PSYNE_ERROR_NO_MESSAGE | psyne_error_PSYNE_ERROR_TIMEOUT => Ok(None),
                _ => Err(err.into()),
            }
        }
    }
    
    /// Reset metrics
    pub fn reset_metrics(&self) -> Result<()> {
        unsafe {
            let err = psyne_channel_reset_metrics(self.ptr);
            if err == psyne_error_PSYNE_OK {
                Ok(())
            } else {
                Err(err.into())
            }
        }
    }
}

impl Drop for Channel {
    fn drop(&mut self) {
        unsafe {
            psyne_channel_destroy(self.ptr);
        }
    }
}

// Safety: Channel can be sent between threads
unsafe impl Send for Channel {}
// Safety: Channel operations are thread-safe based on mode
unsafe impl Sync for Channel {}

/// Message handle for manual message operations
pub struct Message<'a> {
    ptr: *mut psyne_message_t,
    _phantom: PhantomData<&'a Channel>,
}

impl<'a> Message<'a> {
    /// Reserve a message
    pub fn reserve(channel: &'a Channel, size: usize) -> Result<Self> {
        let mut msg_ptr: *mut psyne_message_t = ptr::null_mut();
        
        unsafe {
            let err = psyne_message_reserve(channel.ptr, size, &mut msg_ptr);
            
            if err == psyne_error_PSYNE_OK {
                Ok(Self {
                    ptr: msg_ptr,
                    _phantom: PhantomData,
                })
            } else {
                Err(err.into())
            }
        }
    }
    
    /// Get mutable data slice
    pub fn data_mut(&mut self) -> Result<&mut [u8]> {
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let mut size = 0;
        
        unsafe {
            let err = psyne_message_get_data(self.ptr, &mut data_ptr, &mut size);
            
            if err == psyne_error_PSYNE_OK {
                if data_ptr.is_null() {
                    Err(Error::NullPointer)
                } else {
                    Ok(slice::from_raw_parts_mut(data_ptr as *mut u8, size))
                }
            } else {
                Err(err.into())
            }
        }
    }
    
    /// Send the message
    pub fn send(self, msg_type: u32) -> Result<()> {
        unsafe {
            let err = psyne_message_send(self.ptr, msg_type);
            std::mem::forget(self); // Message is consumed
            
            if err == psyne_error_PSYNE_OK {
                Ok(())
            } else {
                Err(err.into())
            }
        }
    }
}

impl<'a> Drop for Message<'a> {
    fn drop(&mut self) {
        unsafe {
            psyne_message_cancel(self.ptr);
        }
    }
}

/// Builder pattern for channel creation
pub struct ChannelBuilder {
    uri: String,
    buffer_size: usize,
    mode: ChannelMode,
    channel_type: ChannelType,
    compression: Option<CompressionConfig>,
}

impl ChannelBuilder {
    /// Create a new channel builder
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            buffer_size: 1024 * 1024, // 1MB default
            mode: ChannelMode::Spsc,
            channel_type: ChannelType::Multi,
            compression: None,
        }
    }
    
    /// Set buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
    
    /// Set channel mode
    pub fn mode(mut self, mode: ChannelMode) -> Self {
        self.mode = mode;
        self
    }
    
    /// Set channel type
    pub fn channel_type(mut self, typ: ChannelType) -> Self {
        self.channel_type = typ;
        self
    }
    
    /// Enable compression
    pub fn compression(mut self, config: CompressionConfig) -> Self {
        self.compression = Some(config);
        self
    }
    
    /// Build the channel
    pub fn build(self) -> Result<Channel> {
        match self.compression {
            Some(compression) => Channel::create_compressed(
                &self.uri,
                self.buffer_size,
                self.mode,
                self.channel_type,
                compression,
            ),
            None => Channel::create(
                &self.uri,
                self.buffer_size,
                self.mode,
                self.channel_type,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
    }
    
    #[test]
    fn test_channel_creation() {
        init().unwrap();
        
        let channel = ChannelBuilder::new("memory://test")
            .buffer_size(64 * 1024)
            .mode(ChannelMode::Spsc)
            .build()
            .unwrap();
        
        assert_eq!(channel.uri().unwrap(), "memory://test");
        assert_eq!(channel.is_stopped().unwrap(), false);
        
        cleanup();
    }
    
    #[test]
    fn test_send_receive() {
        init().unwrap();
        
        let channel = Channel::create(
            "memory://test",
            1024 * 1024,
            ChannelMode::Spsc,
            ChannelType::Multi,
        ).unwrap();
        
        // Send
        let data = b"Hello from Rust!";
        channel.send_data(data, 42).unwrap();
        
        // Receive
        let mut buffer = vec![0u8; 1024];
        let result = channel.receive_data(&mut buffer, Duration::from_secs(1)).unwrap();
        
        assert!(result.is_some());
        let (size, msg_type) = result.unwrap();
        assert_eq!(size, data.len());
        assert_eq!(&buffer[..size], data);
        
        cleanup();
    }
}