"""
# Psyne.jl

High-performance zero-copy messaging library for Julia, optimized for AI/ML applications.

Psyne provides efficient inter-process and inter-thread communication with minimal overhead
through zero-copy message passing. This Julia package provides idiomatic bindings to the
native Psyne C++ library.

## Features

- Zero-copy messaging for maximum performance
- Multiple transport types (memory, IPC, TCP, Unix sockets, multicast)
- Configurable synchronization modes (SPSC, SPMC, MPSC, MPMC)
- Built-in compression support (LZ4, Zstd, Snappy)
- Type-safe operations with Julia's type system
- Performance metrics and monitoring
- Array types and broadcasting support

## Quick Start

```julia
using Psyne

# Create a channel
channel = Channel("memory://buffer1")

# Send data
data = Float32[1.0, 2.0, 3.0, 4.0]
send(channel, data)

# Receive data
received = receive(channel, Vector{Float32})
```

## Exports

### Core Types
- `PsyneChannel`: Main channel abstraction
- `ChannelMode`: Synchronization modes (SPSC, SPMC, MPSC, MPMC)
- `ChannelType`: Single or multi-type channels
- `CompressionType`: Compression algorithms

### Functions
- `channel`: Create channels with various transports
- `send`: Send messages through channels
- `receive`: Receive messages from channels
- `metrics`: Get channel performance metrics

### Message Types
- `FloatVector`: Dynamic arrays of Float32
- `DoubleMatrix`: 2D matrices of Float64
- `ByteVector`: Raw byte arrays
- `ComplexVector`: Arrays of complex numbers

See the examples directory for detailed usage examples.
"""
module Psyne

export PsyneChannel, ChannelMode, ChannelType, CompressionType, CompressionConfig,
       Metrics, PsyneError,
       channel, send, receive, stop!, is_stopped, get_metrics, reset_metrics!,
       reserve_write_slot, notify_message_ready, advance_read_pointer, get_buffer_view,
       create_multicast_publisher, create_multicast_subscriber, create_webrtc_channel,
       FloatVector, DoubleMatrix, ByteVector, ComplexVector

# Standard library imports
using Base: @kwdef

# Load the native library
const libpsyne = "psyne"  # Assumes libpsyne.so is in the library path

# Error handling
struct PsyneError <: Exception
    message::String
    code::Int32
end

Base.showerror(io::IO, e::PsyneError) = print(io, "PsyneError($(e.code)): $(e.message)")

# Error codes from C API
const PSYNE_OK = 0
const PSYNE_ERROR_INVALID_ARGUMENT = -1
const PSYNE_ERROR_OUT_OF_MEMORY = -2
const PSYNE_ERROR_CHANNEL_FULL = -3
const PSYNE_ERROR_NO_MESSAGE = -4
const PSYNE_ERROR_CHANNEL_STOPPED = -5
const PSYNE_ERROR_UNSUPPORTED = -6
const PSYNE_ERROR_IO = -7
const PSYNE_ERROR_TIMEOUT = -8
const PSYNE_ERROR_UNKNOWN = -99

# Error code to message mapping
const ERROR_MESSAGES = Dict(
    PSYNE_ERROR_INVALID_ARGUMENT => "Invalid argument",
    PSYNE_ERROR_OUT_OF_MEMORY => "Out of memory",
    PSYNE_ERROR_CHANNEL_FULL => "Channel is full",
    PSYNE_ERROR_NO_MESSAGE => "No message available",
    PSYNE_ERROR_CHANNEL_STOPPED => "Channel is stopped",
    PSYNE_ERROR_UNSUPPORTED => "Operation not supported",
    PSYNE_ERROR_IO => "I/O error",
    PSYNE_ERROR_TIMEOUT => "Operation timed out",
    PSYNE_ERROR_UNKNOWN => "Unknown error"
)

function check_error(code::Int32)
    if code != PSYNE_OK
        message = get(ERROR_MESSAGES, code, "Unknown error")
        throw(PsyneError(message, code))
    end
end

# Channel modes
@enum ChannelMode begin
    SPSC = 0  # Single Producer, Single Consumer
    SPMC = 1  # Single Producer, Multiple Consumer
    MPSC = 2  # Multiple Producer, Single Consumer
    MPMC = 3  # Multiple Producer, Multiple Consumer
end

# Channel types
@enum ChannelType begin
    SingleType = 0  # Single message type
    MultiType = 1   # Multiple message types
end

# Compression types
@enum CompressionType begin
    None = 0
    LZ4 = 1
    Zstd = 2
    Snappy = 3
end

# Compression configuration
@kwdef struct CompressionConfig
    type::CompressionType = None
    level::Int32 = 1
    min_size_threshold::UInt64 = 128
    enable_checksum::Bool = true
end

# Channel metrics
struct Metrics
    messages_sent::UInt64
    bytes_sent::UInt64
    messages_received::UInt64
    bytes_received::UInt64
    send_blocks::UInt64
    receive_blocks::UInt64
end

# Native library functions
function psyne_init()
    code = ccall((:psyne_init, libpsyne), Int32, ())
    check_error(code)
end

function psyne_cleanup()
    ccall((:psyne_cleanup, libpsyne), Cvoid, ())
end

function psyne_version()
    ptr = ccall((:psyne_version, libpsyne), Ptr{UInt8}, ())
    return unsafe_string(ptr)
end

# Initialize library when module loads
function __init__()
    try
        psyne_init()
    catch e
        @warn "Failed to initialize Psyne library: $e"
    end
end

# Include sub-modules
include("channel.jl")
include("message.jl")
include("compression.jl")
include("metrics.jl")

end # module Psyne