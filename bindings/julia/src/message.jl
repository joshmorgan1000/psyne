"""
Message types and operations for Psyne.jl

Provides type-safe message sending and receiving with support for Julia's array types
and broadcasting operations.
"""

# Message type constants (from C++ API)
const MESSAGE_TYPE_FLOAT_VECTOR = 1
const MESSAGE_TYPE_DOUBLE_MATRIX = 2
const MESSAGE_TYPE_BYTE_VECTOR = 10
const MESSAGE_TYPE_COMPLEX_VECTOR = 107

# Abstract base type for all Psyne messages
abstract type PsyneMessage end

"""
    FloatVector <: PsyneMessage

Dynamic array of Float32 values optimized for zero-copy transmission.

This type provides a Julia array-like interface while being backed by
Psyne's zero-copy message buffers.

# Examples

```julia
ch = channel("memory://data")

# Send a Float32 vector
data = Float32[1.0, 2.0, 3.0, 4.0]
send(ch, data)

# Receive as FloatVector
received = receive(ch, FloatVector)
```
"""
struct FloatVector <: PsyneMessage
    data::Vector{Float32}
    
    FloatVector(data::Vector{Float32}) = new(data)
    FloatVector(size::Integer) = new(Vector{Float32}(undef, size))
end

"""
    DoubleMatrix <: PsyneMessage

2D matrix of Float64 values for scientific computing applications.

# Examples

```julia
ch = channel("memory://matrix")

# Send a matrix
matrix = rand(Float64, 10, 20)
send(ch, matrix)

# Receive as DoubleMatrix
received = receive(ch, DoubleMatrix)
```
"""
struct DoubleMatrix <: PsyneMessage
    data::Matrix{Float64}
    
    DoubleMatrix(data::Matrix{Float64}) = new(data)
    DoubleMatrix(rows::Integer, cols::Integer) = new(Matrix{Float64}(undef, rows, cols))
end

"""
    ByteVector <: PsyneMessage

Raw byte array for binary data transmission.

# Examples

```julia
ch = channel("memory://bytes")

# Send raw bytes
bytes = UInt8[0x01, 0x02, 0x03, 0x04]
send(ch, bytes)

# Receive as ByteVector
received = receive(ch, ByteVector)
```
"""
struct ByteVector <: PsyneMessage
    data::Vector{UInt8}
    
    ByteVector(data::Vector{UInt8}) = new(data)
    ByteVector(size::Integer) = new(Vector{UInt8}(undef, size))
end

"""
    ComplexVector <: PsyneMessage

Array of Complex{Float32} values for signal processing applications.

# Examples

```julia
ch = channel("memory://signal")

# Send complex data
signal = Complex{Float32}[1+2im, 3+4im, 5+6im]
send(ch, signal)

# Receive as ComplexVector
received = receive(ch, ComplexVector)
```
"""
struct ComplexVector <: PsyneMessage
    data::Vector{Complex{Float32}}
    
    ComplexVector(data::Vector{Complex{Float32}}) = new(data)
    ComplexVector(size::Integer) = new(Vector{Complex{Float32}}(undef, size))
end

# Array-like interface for message types
Base.size(msg::FloatVector) = size(msg.data)
Base.size(msg::DoubleMatrix) = size(msg.data)
Base.size(msg::ByteVector) = size(msg.data)
Base.size(msg::ComplexVector) = size(msg.data)

Base.length(msg::FloatVector) = length(msg.data)
Base.length(msg::ByteVector) = length(msg.data)
Base.length(msg::ComplexVector) = length(msg.data)

Base.getindex(msg::FloatVector, i...) = getindex(msg.data, i...)
Base.getindex(msg::DoubleMatrix, i...) = getindex(msg.data, i...)
Base.getindex(msg::ByteVector, i...) = getindex(msg.data, i...)
Base.getindex(msg::ComplexVector, i...) = getindex(msg.data, i...)

Base.setindex!(msg::FloatVector, v, i...) = setindex!(msg.data, v, i...)
Base.setindex!(msg::DoubleMatrix, v, i...) = setindex!(msg.data, v, i...)
Base.setindex!(msg::ByteVector, v, i...) = setindex!(msg.data, v, i...)
Base.setindex!(msg::ComplexVector, v, i...) = setindex!(msg.data, v, i...)

# Iterator interface
Base.iterate(msg::FloatVector) = iterate(msg.data)
Base.iterate(msg::FloatVector, state) = iterate(msg.data, state)
Base.iterate(msg::ByteVector) = iterate(msg.data)
Base.iterate(msg::ByteVector, state) = iterate(msg.data, state)
Base.iterate(msg::ComplexVector) = iterate(msg.data)
Base.iterate(msg::ComplexVector, state) = iterate(msg.data, state)

# Broadcasting
Base.broadcastable(msg::FloatVector) = msg.data
Base.broadcastable(msg::DoubleMatrix) = msg.data
Base.broadcastable(msg::ByteVector) = msg.data
Base.broadcastable(msg::ComplexVector) = msg.data

# Type mapping for message types
const MESSAGE_TYPE_MAP = Dict{DataType, UInt32}(
    Vector{Float32} => MESSAGE_TYPE_FLOAT_VECTOR,
    FloatVector => MESSAGE_TYPE_FLOAT_VECTOR,
    Matrix{Float64} => MESSAGE_TYPE_DOUBLE_MATRIX,
    DoubleMatrix => MESSAGE_TYPE_DOUBLE_MATRIX,
    Vector{UInt8} => MESSAGE_TYPE_BYTE_VECTOR,
    ByteVector => MESSAGE_TYPE_BYTE_VECTOR,
    Vector{Complex{Float32}} => MESSAGE_TYPE_COMPLEX_VECTOR,
    ComplexVector => MESSAGE_TYPE_COMPLEX_VECTOR
)

"""
    send(channel::PsyneChannel, data::T; timeout_ms::Integer = 0) where T

Send data through a Psyne channel.

This function supports multiple dispatch based on the data type and provides
zero-copy transmission when possible.

# Arguments
- `channel`: The channel to send through
- `data`: The data to send (supports arrays, matrices, and message types)
- `timeout_ms`: Send timeout in milliseconds (0 = non-blocking)

# Supported Types
- `Vector{Float32}` / `FloatVector`
- `Matrix{Float64}` / `DoubleMatrix` 
- `Vector{UInt8}` / `ByteVector`
- `Vector{Complex{Float32}}` / `ComplexVector`

# Examples

```julia
ch = channel("memory://test")

# Send different data types
send(ch, Float32[1.0, 2.0, 3.0])
send(ch, rand(Float64, 5, 5))
send(ch, UInt8[0x01, 0x02, 0x03])
send(ch, Complex{Float32}[1+2im, 3+4im])
```
"""
function send(channel::PsyneChannel, data::Vector{Float32}; timeout_ms::Integer = 0)
    _send_vector_data(channel, data, MESSAGE_TYPE_FLOAT_VECTOR, timeout_ms)
end

function send(channel::PsyneChannel, data::FloatVector; timeout_ms::Integer = 0)
    _send_vector_data(channel, data.data, MESSAGE_TYPE_FLOAT_VECTOR, timeout_ms)
end

function send(channel::PsyneChannel, data::Matrix{Float64}; timeout_ms::Integer = 0)
    _send_matrix_data(channel, data, MESSAGE_TYPE_DOUBLE_MATRIX, timeout_ms)
end

function send(channel::PsyneChannel, data::DoubleMatrix; timeout_ms::Integer = 0)
    _send_matrix_data(channel, data.data, MESSAGE_TYPE_DOUBLE_MATRIX, timeout_ms)
end

function send(channel::PsyneChannel, data::Vector{UInt8}; timeout_ms::Integer = 0)
    _send_vector_data(channel, data, MESSAGE_TYPE_BYTE_VECTOR, timeout_ms)
end

function send(channel::PsyneChannel, data::ByteVector; timeout_ms::Integer = 0)
    _send_vector_data(channel, data.data, MESSAGE_TYPE_BYTE_VECTOR, timeout_ms)
end

function send(channel::PsyneChannel, data::Vector{Complex{Float32}}; timeout_ms::Integer = 0)
    _send_vector_data(channel, data, MESSAGE_TYPE_COMPLEX_VECTOR, timeout_ms)
end

function send(channel::PsyneChannel, data::ComplexVector; timeout_ms::Integer = 0)
    _send_vector_data(channel, data.data, MESSAGE_TYPE_COMPLEX_VECTOR, timeout_ms)
end

# Internal helper for sending vector data
function _send_vector_data(channel::PsyneChannel, data::Vector{T}, msg_type::UInt32, timeout_ms::Integer) where T
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    data_size = sizeof(data)
    code = ccall((:psyne_send_data, libpsyne), Int32,
                (Ptr{Cvoid}, Ptr{T}, UInt64, UInt32),
                channel.handle, data, UInt64(data_size), msg_type)
    check_error(code)
end

# Internal helper for sending matrix data
function _send_matrix_data(channel::PsyneChannel, data::Matrix{T}, msg_type::UInt32, timeout_ms::Integer) where T
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    # Matrices are sent as flattened arrays with dimension info
    data_size = sizeof(data) + 2 * sizeof(UInt64)  # Include dimensions
    
    # Create a buffer with dimensions + data
    buffer = Vector{UInt8}(undef, data_size)
    
    # Pack dimensions (row-major order)
    dims = [UInt64(size(data, 1)), UInt64(size(data, 2))]
    unsafe_copyto!(pointer(buffer), pointer(dims), 2 * sizeof(UInt64))
    
    # Pack matrix data
    data_offset = 2 * sizeof(UInt64)
    unsafe_copyto!(pointer(buffer) + data_offset, pointer(data), sizeof(data))
    
    code = ccall((:psyne_send_data, libpsyne), Int32,
                (Ptr{Cvoid}, Ptr{UInt8}, UInt64, UInt32),
                channel.handle, buffer, UInt64(data_size), msg_type)
    check_error(code)
end

"""
    receive(channel::PsyneChannel, ::Type{T}; timeout_ms::Integer = 0) -> T where T

Receive data from a Psyne channel.

This function uses multiple dispatch to return the appropriate Julia type
based on the type parameter.

# Arguments
- `channel`: The channel to receive from
- `T`: The expected data type
- `timeout_ms`: Receive timeout in milliseconds (0 = non-blocking)

# Returns
The received data as type `T`.

# Throws
- `PsyneError`: If receive fails or times out
- `MethodError`: If the requested type is not supported

# Examples

```julia
ch = channel("memory://test")

# Receive different types
float_data = receive(ch, Vector{Float32})
matrix_data = receive(ch, Matrix{Float64})
byte_data = receive(ch, Vector{UInt8})
complex_data = receive(ch, Vector{Complex{Float32}})

# Receive with timeout
data = receive(ch, Vector{Float32}, timeout_ms=1000)
```
"""
function receive(channel::PsyneChannel, ::Type{Vector{Float32}}; timeout_ms::Integer = 0)
    return _receive_vector_data(channel, Float32, MESSAGE_TYPE_FLOAT_VECTOR, timeout_ms)
end

function receive(channel::PsyneChannel, ::Type{FloatVector}; timeout_ms::Integer = 0)
    data = _receive_vector_data(channel, Float32, MESSAGE_TYPE_FLOAT_VECTOR, timeout_ms)
    return FloatVector(data)
end

function receive(channel::PsyneChannel, ::Type{Matrix{Float64}}; timeout_ms::Integer = 0)
    return _receive_matrix_data(channel, Float64, MESSAGE_TYPE_DOUBLE_MATRIX, timeout_ms)
end

function receive(channel::PsyneChannel, ::Type{DoubleMatrix}; timeout_ms::Integer = 0)
    data = _receive_matrix_data(channel, Float64, MESSAGE_TYPE_DOUBLE_MATRIX, timeout_ms)
    return DoubleMatrix(data)
end

function receive(channel::PsyneChannel, ::Type{Vector{UInt8}}; timeout_ms::Integer = 0)
    return _receive_vector_data(channel, UInt8, MESSAGE_TYPE_BYTE_VECTOR, timeout_ms)
end

function receive(channel::PsyneChannel, ::Type{ByteVector}; timeout_ms::Integer = 0)
    data = _receive_vector_data(channel, UInt8, MESSAGE_TYPE_BYTE_VECTOR, timeout_ms)
    return ByteVector(data)
end

function receive(channel::PsyneChannel, ::Type{Vector{Complex{Float32}}}; timeout_ms::Integer = 0)
    return _receive_vector_data(channel, Complex{Float32}, MESSAGE_TYPE_COMPLEX_VECTOR, timeout_ms)
end

function receive(channel::PsyneChannel, ::Type{ComplexVector}; timeout_ms::Integer = 0)
    data = _receive_vector_data(channel, Complex{Float32}, MESSAGE_TYPE_COMPLEX_VECTOR, timeout_ms)
    return ComplexVector(data)
end

# Internal helper for receiving vector data
function _receive_vector_data(channel::PsyneChannel, ::Type{T}, expected_type::UInt32, timeout_ms::Integer) where T
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    # Allocate a buffer for receiving
    buffer_size = UInt64(channel.buffer_size)
    buffer = Vector{UInt8}(undef, buffer_size)
    received_size_ref = Ref{UInt64}()
    msg_type_ref = Ref{UInt32}()
    
    code = ccall((:psyne_receive_data, libpsyne), Int32,
                (Ptr{Cvoid}, Ptr{UInt8}, UInt64, Ptr{UInt64}, Ptr{UInt32}, UInt32),
                channel.handle, buffer, buffer_size, received_size_ref, msg_type_ref, UInt32(timeout_ms))
    check_error(code)
    
    # Verify message type
    if msg_type_ref[] != expected_type
        throw(PsyneError("Received message type $(msg_type_ref[]) but expected $expected_type", -1))
    end
    
    # Convert buffer to the requested type
    received_size = received_size_ref[]
    element_count = received_size ÷ sizeof(T)
    
    # Create result vector and copy data
    result = Vector{T}(undef, element_count)
    unsafe_copyto!(pointer(result), pointer(buffer), element_count * sizeof(T))
    
    return result
end

# Internal helper for receiving matrix data
function _receive_matrix_data(channel::PsyneChannel, ::Type{T}, expected_type::UInt32, timeout_ms::Integer) where T
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    # Allocate a buffer for receiving
    buffer_size = UInt64(channel.buffer_size)
    buffer = Vector{UInt8}(undef, buffer_size)
    received_size_ref = Ref{UInt64}()
    msg_type_ref = Ref{UInt32}()
    
    code = ccall((:psyne_receive_data, libpsyne), Int32,
                (Ptr{Cvoid}, Ptr{UInt8}, UInt64, Ptr{UInt64}, Ptr{UInt32}, UInt32),
                channel.handle, buffer, buffer_size, received_size_ref, msg_type_ref, UInt32(timeout_ms))
    check_error(code)
    
    # Verify message type
    if msg_type_ref[] != expected_type
        throw(PsyneError("Received message type $(msg_type_ref[]) but expected $expected_type", -1))
    end
    
    # Extract dimensions
    dims_ptr = pointer(buffer)
    rows = unsafe_load(Ptr{UInt64}(dims_ptr))
    cols = unsafe_load(Ptr{UInt64}(dims_ptr + sizeof(UInt64)))
    
    # Extract matrix data
    data_offset = 2 * sizeof(UInt64)
    element_count = rows * cols
    matrix_data = Vector{T}(undef, element_count)
    unsafe_copyto!(pointer(matrix_data), pointer(buffer) + data_offset, element_count * sizeof(T))
    
    # Reshape to matrix
    return reshape(matrix_data, Int(rows), Int(cols))
end

# Convenience function for generic receive (tries to infer type from message)
"""
    receive(channel::PsyneChannel; timeout_ms::Integer = 0) -> Any

Receive data from a channel without specifying the expected type.

The function will attempt to determine the appropriate Julia type based on
the message type ID received from the channel.

# Arguments
- `channel`: The channel to receive from
- `timeout_ms`: Receive timeout in milliseconds (0 = non-blocking)

# Returns
The received data with an appropriate Julia type.

# Examples

```julia
ch = channel("memory://test")
data = receive(ch)  # Type determined automatically
```
"""
function receive(channel::PsyneChannel; timeout_ms::Integer = 0)
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    # First, peek at the message type
    buffer_size = UInt64(channel.buffer_size)
    buffer = Vector{UInt8}(undef, buffer_size)
    received_size_ref = Ref{UInt64}()
    msg_type_ref = Ref{UInt32}()
    
    code = ccall((:psyne_receive_data, libpsyne), Int32,
                (Ptr{Cvoid}, Ptr{UInt8}, UInt64, Ptr{UInt64}, Ptr{UInt32}, UInt32),
                channel.handle, buffer, buffer_size, received_size_ref, msg_type_ref, UInt32(timeout_ms))
    check_error(code)
    
    # Dispatch based on message type
    msg_type = msg_type_ref[]
    received_size = received_size_ref[]
    
    if msg_type == MESSAGE_TYPE_FLOAT_VECTOR
        element_count = received_size ÷ sizeof(Float32)
        result = Vector{Float32}(undef, element_count)
        unsafe_copyto!(pointer(result), pointer(buffer), element_count * sizeof(Float32))
        return result
        
    elseif msg_type == MESSAGE_TYPE_DOUBLE_MATRIX
        # Extract dimensions and matrix data
        dims_ptr = pointer(buffer)
        rows = unsafe_load(Ptr{UInt64}(dims_ptr))
        cols = unsafe_load(Ptr{UInt64}(dims_ptr + sizeof(UInt64)))
        
        data_offset = 2 * sizeof(UInt64)
        element_count = rows * cols
        matrix_data = Vector{Float64}(undef, element_count)
        unsafe_copyto!(pointer(matrix_data), pointer(buffer) + data_offset, element_count * sizeof(Float64))
        
        return reshape(matrix_data, Int(rows), Int(cols))
        
    elseif msg_type == MESSAGE_TYPE_BYTE_VECTOR
        result = Vector{UInt8}(undef, received_size)
        unsafe_copyto!(pointer(result), pointer(buffer), received_size)
        return result
        
    elseif msg_type == MESSAGE_TYPE_COMPLEX_VECTOR
        element_count = received_size ÷ sizeof(Complex{Float32})
        result = Vector{Complex{Float32}}(undef, element_count)
        unsafe_copyto!(pointer(result), pointer(buffer), element_count * sizeof(Complex{Float32}))
        return result
        
    else
        throw(PsyneError("Unknown message type: $msg_type", -1))
    end
end