#!/usr/bin/env julia

"""
Basic Psyne.jl Example

This example demonstrates the fundamental usage of Psyne.jl for zero-copy
message passing between processes or threads.

Run this example with:
    julia basic_example.jl
"""

using Psyne

println("=== Psyne.jl Basic Example ===")
println("Psyne version: ", psyne_version())
println()

# Create a memory channel for in-process communication
println("1. Creating a memory channel...")
channel_uri = "memory://basic_example"
ch = channel(channel_uri)
println("   Created channel: $ch")
println()

# Send some basic data types
println("2. Sending different data types...")

# Send a Float32 vector
println("   Sending Float32 vector...")
float_data = Float32[1.0, 2.0, 3.14159, 4.0, 5.0]
send(ch, float_data)
println("   Sent: $float_data")

# Send a Float64 matrix
println("   Sending Float64 matrix...")
matrix_data = rand(Float64, 3, 4)
send(ch, matrix_data)
println("   Sent: $(size(matrix_data)) matrix")

# Send raw bytes
println("   Sending byte array...")
byte_data = UInt8[0x48, 0x65, 0x6C, 0x6C, 0x6F]  # "Hello" in ASCII
send(ch, byte_data)
println("   Sent: $byte_data ($(String(byte_data)))")

# Send complex numbers
println("   Sending complex vector...")
complex_data = Complex{Float32}[1+2im, 3+4im, 5+6im]
send(ch, complex_data)
println("   Sent: $complex_data")
println()

# Receive the data back
println("3. Receiving data...")

# Receive Float32 vector
println("   Receiving Float32 vector...")
received_float = receive(ch, Vector{Float32})
println("   Received: $received_float")
println("   Match: $(received_float == float_data)")

# Receive Float64 matrix
println("   Receiving Float64 matrix...")
received_matrix = receive(ch, Matrix{Float64})
println("   Received: $(size(received_matrix)) matrix")
println("   Match: $(received_matrix == matrix_data)")

# Receive bytes
println("   Receiving byte array...")
received_bytes = receive(ch, Vector{UInt8})
println("   Received: $received_bytes ($(String(received_bytes)))")
println("   Match: $(received_bytes == byte_data)")

# Receive complex numbers
println("   Receiving complex vector...")
received_complex = receive(ch, Vector{Complex{Float32}})
println("   Received: $received_complex")
println("   Match: $(received_complex == complex_data)")
println()

# Demonstrate automatic type detection
println("4. Demonstrating automatic type detection...")
send(ch, Float32[10.0, 20.0, 30.0])
auto_received = receive(ch)  # No type specified
println("   Auto-detected type: $(typeof(auto_received))")
println("   Value: $auto_received")
println()

# Demonstrate Psyne message types
println("5. Using Psyne message types...")

# Create a FloatVector message
println("   Creating FloatVector...")
float_msg = FloatVector(10)
for i in 1:length(float_msg)
    float_msg[i] = sin(i * 0.1)
end
send(ch, float_msg)
println("   Sent FloatVector of length $(length(float_msg))")

received_float_msg = receive(ch, FloatVector)
println("   Received FloatVector: first 5 elements = $(received_float_msg[1:5])")
println()

# Demonstrate array operations on message types
println("6. Array operations on message types...")
byte_msg = ByteVector(5)
byte_msg.data .= [1, 2, 3, 4, 5]
println("   Created ByteVector: $(byte_msg.data)")

# Broadcasting works
byte_msg.data .*= 2
println("   After broadcasting (*= 2): $(byte_msg.data)")

send(ch, byte_msg)
received_byte_msg = receive(ch, ByteVector)
println("   Received ByteVector: $(received_byte_msg.data)")
println()

# Clean up
println("7. Cleaning up...")
close(ch)
println("   Channel closed")

println()
println("=== Basic Example Complete ===")
println("This example demonstrated:")
println("  ✓ Creating channels")
println("  ✓ Sending/receiving built-in Julia types")
println("  ✓ Automatic type detection")
println("  ✓ Psyne message types")
println("  ✓ Array operations and broadcasting")
println("  ✓ Proper resource cleanup")