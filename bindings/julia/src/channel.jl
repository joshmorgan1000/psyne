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

# ===================================================================
# Zero-copy API methods (v1.3.0)
# ===================================================================

"""
    reserve_write_slot(channel::PsyneChannel, size::Integer) -> UInt32

Reserve space in ring buffer and return offset (zero-copy API).

This is part of the zero-copy API for maximum performance. After reserving
a slot, you can write directly to the buffer and then notify when ready.

# Arguments
- `channel`: The channel to reserve space in
- `size`: Size of message to reserve in bytes

# Returns
Offset within ring buffer where data should be written, or UInt32(0xFFFFFFFF) if buffer is full.

# Throws
- `PsyneError`: If reservation fails

# Examples

```julia
ch = channel("memory://zerocopy")
offset = reserve_write_slot(ch, 1024)
if offset != 0xFFFFFFFF
    # Write data at offset...
    notify_message_ready(ch, offset, 1024)
end
```
"""
function reserve_write_slot(channel::PsyneChannel, size::Integer)
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    offset_ref = Ref{UInt32}()
    code = ccall((:psyne_channel_reserve_write_slot, libpsyne), Int32,
                (Ptr{Cvoid}, UInt64, Ptr{UInt32}),
                channel.handle, UInt64(size), offset_ref)
    check_error(code)
    
    return offset_ref[]
end

"""
    notify_message_ready(channel::PsyneChannel, offset::UInt32, size::Integer)

Notify receiver that message is ready at offset (zero-copy API).

This completes the zero-copy send operation after writing data to the
reserved buffer location.

# Arguments
- `channel`: The channel containing the message
- `offset`: Offset within ring buffer where message data starts
- `size`: Size of the message in bytes

# Throws
- `PsyneError`: If notification fails

# Examples

```julia
ch = channel("memory://zerocopy")
offset = reserve_write_slot(ch, 1024)
# ... write data to buffer at offset ...
notify_message_ready(ch, offset, 1024)
```
"""
function notify_message_ready(channel::PsyneChannel, offset::UInt32, size::Integer)
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    code = ccall((:psyne_channel_notify_message_ready, libpsyne), Int32,
                (Ptr{Cvoid}, UInt32, UInt64),
                channel.handle, offset, UInt64(size))
    check_error(code)
end

"""
    advance_read_pointer(channel::PsyneChannel, size::Integer)

Consumer advances read pointer after processing message (zero-copy API).

This completes the zero-copy receive operation after processing the data
directly from the ring buffer.

# Arguments
- `channel`: The channel to advance the read pointer in
- `size`: Size of message that was consumed in bytes

# Throws
- `PsyneError`: If advancing read pointer fails

# Examples

```julia
ch = channel("memory://zerocopy")
buffer_view = get_buffer_view(ch)
if buffer_view !== nothing
    # ... process data directly from buffer_view ...
    advance_read_pointer(ch, processed_size)
end
```
"""
function advance_read_pointer(channel::PsyneChannel, size::Integer)
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    code = ccall((:psyne_channel_advance_read_pointer, libpsyne), Int32,
                (Ptr{Cvoid}, UInt64),
                channel.handle, UInt64(size))
    check_error(code)
end

"""
    get_buffer_view(channel::PsyneChannel) -> Union{Vector{UInt8}, Nothing}

Get a view of the ring buffer for zero-copy access.

# Arguments
- `channel`: The channel to get buffer view from

# Returns
A `Vector{UInt8}` view of the ring buffer, or `nothing` if not available.

# Throws
- `PsyneError`: If getting buffer view fails

# Safety Warning
The returned buffer view is only valid while the channel exists and
the ring buffer is not reallocated. Use with extreme caution.

# Examples

```julia
ch = channel("memory://zerocopy")
buffer_view = get_buffer_view(ch)
if buffer_view !== nothing
    # Read/write directly to buffer_view
    # Remember to call advance_read_pointer when done reading
end
```
"""
function get_buffer_view(channel::PsyneChannel)
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    ptr_ref = Ref{Ptr{UInt8}}()
    size_ref = Ref{UInt64}()
    
    code = ccall((:psyne_channel_get_buffer_span, libpsyne), Int32,
                (Ptr{Cvoid}, Ptr{Ptr{UInt8}}, Ptr{UInt64}),
                channel.handle, ptr_ref, size_ref)
    check_error(code)
    
    ptr = ptr_ref[]
    size = size_ref[]
    
    if ptr == C_NULL || size == 0
        return nothing
    end
    
    # Create a Julia array that wraps the C memory (unsafe!)
    # This does not copy the data but creates a view
    return unsafe_wrap(Array{UInt8}, ptr, size; own=false)
end

# ===================================================================
# v1.3.0 Transport Factory Functions
# ===================================================================

"""
    create_multicast_publisher(multicast_address::AbstractString, port::Integer; 
                              buffer_size::Integer = 1024*1024,
                              compression::Union{CompressionConfig, Nothing} = nothing) -> PsyneChannel

Create a UDP multicast publisher channel for one-to-many messaging.

# Arguments
- `multicast_address`: The multicast group address (e.g., "239.255.0.1")
- `port`: The port number
- `buffer_size`: Size of internal buffer in bytes (default: 1MB)
- `compression`: Optional compression configuration

# Returns
A new `PsyneChannel` configured for multicast publishing.

# Examples

```julia
# Create basic multicast publisher
pub = create_multicast_publisher("239.255.0.1", 12345)

# Create compressed multicast publisher
config = CompressionConfig(type=LZ4, level=3)
pub = create_multicast_publisher("239.255.0.1", 12345, compression=config)
```
"""
function create_multicast_publisher(multicast_address::AbstractString, port::Integer;
                                  buffer_size::Integer = 1024*1024,
                                  compression::Union{CompressionConfig, Nothing} = nothing)
    uri = "udp://$(multicast_address):$(port)"
    return channel(uri, buffer_size=buffer_size, mode=SPSC, type=MultiType, compression=compression)
end

"""
    create_multicast_subscriber(multicast_address::AbstractString, port::Integer;
                               buffer_size::Integer = 1024*1024,
                               interface_address::Union{AbstractString, Nothing} = nothing) -> PsyneChannel

Create a UDP multicast subscriber channel for receiving multicast messages.

# Arguments
- `multicast_address`: The multicast group address (e.g., "239.255.0.1")
- `port`: The port number
- `buffer_size`: Size of internal buffer in bytes (default: 1MB)
- `interface_address`: Optional specific network interface to bind to

# Returns
A new `PsyneChannel` configured for multicast subscription.

# Examples

```julia
# Create basic multicast subscriber
sub = create_multicast_subscriber("239.255.0.1", 12345)

# Create multicast subscriber bound to specific interface
sub = create_multicast_subscriber("239.255.0.1", 12345, interface_address="192.168.1.100")
```
"""
function create_multicast_subscriber(multicast_address::AbstractString, port::Integer;
                                   buffer_size::Integer = 1024*1024,
                                   interface_address::Union{AbstractString, Nothing} = nothing)
    uri = "udp://$(multicast_address):$(port)"
    if interface_address !== nothing
        uri *= "?interface=$(interface_address)"
    end
    return channel(uri, buffer_size=buffer_size, mode=SPSC, type=MultiType)
end

"""
    create_webrtc_channel(peer_id::AbstractString;
                         buffer_size::Integer = 1024*1024,
                         signaling_server_uri::AbstractString = "ws://localhost:8080") -> PsyneChannel

Create a WebRTC channel for peer-to-peer communication with NAT traversal.

WebRTC provides direct peer-to-peer communication that can traverse NAT and
firewall boundaries, making it ideal for distributed applications.

# Arguments
- `peer_id`: The target peer identifier
- `buffer_size`: Size of internal buffer in bytes (default: 1MB)
- `signaling_server_uri`: The WebSocket signaling server URI (default: ws://localhost:8080)

# Returns
A new `PsyneChannel` configured for WebRTC communication.

# Examples

```julia
# Create basic WebRTC channel
peer = create_webrtc_channel("peer-123")

# Create WebRTC channel with custom signaling server
peer = create_webrtc_channel("peer-456", signaling_server_uri="wss://signaling.example.com:443")
```
"""
function create_webrtc_channel(peer_id::AbstractString;
                             buffer_size::Integer = 1024*1024,
                             signaling_server_uri::AbstractString = "ws://localhost:8080")
    uri = "webrtc://$(peer_id)?signaling=$(signaling_server_uri)"
    return channel(uri, buffer_size=buffer_size, mode=SPSC, type=MultiType)
end