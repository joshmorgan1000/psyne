"""
Channel implementation for Psyne.jl

Provides Julia-style interface to Psyne channels with type safety and proper resource management.
"""

"""
    PsyneChannel

Represents a Psyne communication channel.

A channel provides zero-copy message passing between processes or threads.
Channels are created with a URI that specifies the transport mechanism:

- `memory://name` - In-process shared memory
- `ipc://name` - Inter-process communication
- `tcp://host:port` - TCP network communication
- `unix:///path` - Unix domain sockets
- `udp://address:port` - UDP multicast

# Examples

```julia
# Create a memory channel
ch = channel("memory://buffer1")

# Create a TCP server channel with custom buffer size
ch = channel("tcp://:8080", buffer_size=2^20, mode=MPMC)

# Create a compressed IPC channel
config = CompressionConfig(type=LZ4, level=3)
ch = channel("ipc://data", compression=config)
```
"""
mutable struct PsyneChannel
    handle::Ptr{Cvoid}
    uri::String
    buffer_size::UInt64
    mode::ChannelMode
    type::ChannelType
    
    function PsyneChannel(handle::Ptr{Cvoid}, uri::String, buffer_size::UInt64, 
                         mode::ChannelMode, type::ChannelType)
        channel = new(handle, uri, buffer_size, mode, type)
        finalizer(close, channel)
        return channel
    end
end

"""
    channel(uri::AbstractString; 
            buffer_size::Integer = 1024*1024,
            mode::ChannelMode = SPSC,
            type::ChannelType = MultiType,
            compression::Union{CompressionConfig, Nothing} = nothing) -> PsyneChannel

Create a new Psyne channel.

# Arguments
- `uri`: Channel URI specifying transport and location
- `buffer_size`: Size of internal buffer in bytes (default: 1MB)
- `mode`: Synchronization mode (default: SPSC)
- `type`: Single or multi-type channel (default: MultiType)
- `compression`: Optional compression configuration

# Returns
A new `PsyneChannel` instance.

# Throws
- `PsyneError`: If channel creation fails
- `ArgumentError`: If URI format is invalid

# Examples

```julia
# Basic memory channel
ch = channel("memory://test")

# High-performance SPSC channel
ch = channel("ipc://fast", mode=SPSC, buffer_size=4*1024*1024)

# Compressed network channel
config = CompressionConfig(type=LZ4, level=3)
ch = channel("tcp://localhost:8080", compression=config)
```
"""
function channel(uri::AbstractString; 
                buffer_size::Integer = 1024*1024,
                mode::ChannelMode = SPSC,
                type::ChannelType = MultiType,
                compression::Union{CompressionConfig, Nothing} = nothing)
    
    # Validate arguments
    buffer_size > 0 || throw(ArgumentError("buffer_size must be positive"))
    
    handle_ref = Ref{Ptr{Cvoid}}()
    
    if compression === nothing
        # Create channel without compression
        code = ccall((:psyne_channel_create, libpsyne), Int32,
                    (Ptr{UInt8}, UInt64, Int32, Int32, Ptr{Ptr{Cvoid}}),
                    uri, UInt64(buffer_size), Int32(mode), Int32(type), handle_ref)
    else
        # Create channel with compression
        comp_config = (Int32(compression.type), compression.level, 
                      compression.min_size_threshold, compression.enable_checksum)
        code = ccall((:psyne_channel_create_compressed, libpsyne), Int32,
                    (Ptr{UInt8}, UInt64, Int32, Int32, Ptr{NTuple{4, Any}}, Ptr{Ptr{Cvoid}}),
                    uri, UInt64(buffer_size), Int32(mode), Int32(type), 
                    Ref(comp_config), handle_ref)
    end
    
    check_error(code)
    
    return PsyneChannel(handle_ref[], String(uri), UInt64(buffer_size), mode, type)
end

"""
    close(channel::PsyneChannel)

Close and destroy a channel, releasing all resources.

This is automatically called when the channel is garbage collected,
but can be called explicitly for immediate cleanup.

# Examples

```julia
ch = channel("memory://temp")
# ... use channel ...
close(ch)
```
"""
function Base.close(channel::PsyneChannel)
    if channel.handle != C_NULL
        ccall((:psyne_channel_destroy, libpsyne), Cvoid, (Ptr{Cvoid},), channel.handle)
        channel.handle = C_NULL
    end
end

"""
    stop!(channel::PsyneChannel)

Stop a channel, preventing further send/receive operations.

This gracefully shuts down the channel while preserving existing messages.

# Examples

```julia
ch = channel("memory://test")
# ... use channel ...
stop!(ch)
```
"""
function stop!(channel::PsyneChannel)
    channel.handle == C_NULL && return
    code = ccall((:psyne_channel_stop, libpsyne), Int32, (Ptr{Cvoid},), channel.handle)
    check_error(code)
end

"""
    is_stopped(channel::PsyneChannel) -> Bool

Check if a channel has been stopped.

# Returns
`true` if the channel is stopped, `false` otherwise.

# Examples

```julia
ch = channel("memory://test")
@assert !is_stopped(ch)
stop!(ch)
@assert is_stopped(ch)
```
"""
function is_stopped(channel::PsyneChannel)
    channel.handle == C_NULL && return true
    
    stopped_ref = Ref{Bool}()
    code = ccall((:psyne_channel_is_stopped, libpsyne), Int32, 
                (Ptr{Cvoid}, Ptr{Bool}), channel.handle, stopped_ref)
    check_error(code)
    
    return stopped_ref[]
end

"""
    get_uri(channel::PsyneChannel) -> String

Get the URI of a channel.

# Returns
The channel's URI string.

# Examples

```julia
ch = channel("memory://test")
@assert get_uri(ch) == "memory://test"
```
"""
function get_uri(channel::PsyneChannel)
    return channel.uri
end

"""
    get_buffer_size(channel::PsyneChannel) -> UInt64

Get the buffer size of a channel.

# Returns
The channel's buffer size in bytes.

# Examples

```julia
ch = channel("memory://test", buffer_size=2^20)
@assert get_buffer_size(ch) == 2^20
```
"""
function get_buffer_size(channel::PsyneChannel)
    return channel.buffer_size
end

"""
    get_mode(channel::PsyneChannel) -> ChannelMode

Get the synchronization mode of a channel.

# Returns
The channel's mode (SPSC, SPMC, MPSC, or MPMC).
"""
function get_mode(channel::PsyneChannel)
    return channel.mode
end

"""
    get_type(channel::PsyneChannel) -> ChannelType

Get the type configuration of a channel.

# Returns
The channel's type (SingleType or MultiType).
"""
function get_type(channel::PsyneChannel)
    return channel.type
end

# Pretty printing
function Base.show(io::IO, channel::PsyneChannel)
    status = channel.handle == C_NULL ? "closed" : (is_stopped(channel) ? "stopped" : "active")
    print(io, "PsyneChannel(\"$(channel.uri)\", $(channel.buffer_size) bytes, $(channel.mode), $status)")
end

function Base.show(io::IO, ::MIME"text/plain", channel::PsyneChannel)
    println(io, "PsyneChannel:")
    println(io, "  URI: $(channel.uri)")
    println(io, "  Buffer size: $(channel.buffer_size) bytes")
    println(io, "  Mode: $(channel.mode)")
    println(io, "  Type: $(channel.type)")
    status = channel.handle == C_NULL ? "closed" : (is_stopped(channel) ? "stopped" : "active")
    println(io, "  Status: $status")
end

# Iterator interface for receiving messages
struct ChannelIterator{T}
    channel::PsyneChannel
    timeout_ms::UInt32
end

"""
    iterate(channel::PsyneChannel, ::Type{T}; timeout_ms::Integer = 0) where T

Create an iterator for receiving messages of type T from a channel.

# Arguments
- `channel`: The channel to iterate over
- `T`: The message type to receive
- `timeout_ms`: Timeout in milliseconds (0 = non-blocking, default)

# Examples

```julia
ch = channel("memory://data")

# Iterate over Float32 vectors with 100ms timeout
for msg in iterate(ch, Vector{Float32}, timeout_ms=100)
    println("Received: ", msg)
end
```
"""
function Base.iterate(channel::PsyneChannel, ::Type{T}; timeout_ms::Integer = 0) where T
    return ChannelIterator{T}(channel, UInt32(timeout_ms))
end

function Base.iterate(iter::ChannelIterator{T}, state=nothing) where T
    try
        msg = receive(iter.channel, T, timeout_ms=iter.timeout_ms)
        return (msg, nothing)
    catch e
        if e isa PsyneError && e.code == PSYNE_ERROR_NO_MESSAGE
            return nothing
        else
            rethrow(e)
        end
    end
end

Base.IteratorSize(::Type{<:ChannelIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{ChannelIterator{T}}) where T = Base.HasEltype()
Base.eltype(::Type{ChannelIterator{T}}) where T = T