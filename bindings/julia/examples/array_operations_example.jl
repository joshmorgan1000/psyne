#!/usr/bin/env julia

"""
Array Operations Example for Psyne.jl

This example demonstrates advanced array operations, broadcasting, and
scientific computing patterns using Psyne's message types.

Run this example with:
    julia array_operations_example.jl
"""

using Psyne
using Printf
using LinearAlgebra
using Statistics

println("=== Psyne.jl Array Operations Example ===")
println()

# Create a high-performance channel for array operations
ch = channel("memory://arrays", 
            buffer_size=16*1024*1024,  # 16MB for large arrays
            mode=SPSC)

enable_metrics!(ch, true)
println("Created channel: $ch")
println()

# Example 1: Basic array operations
println("1. Basic Array Operations")
println("-" * 25)

# Create and send various array types
println("Creating and sending arrays...")

# Float32 vector
float_data = Float32[sin(i * 0.1) for i in 1:1000]
send(ch, float_data)
println("  Sent Float32 vector: $(length(float_data)) elements")

# Float64 matrix
matrix_data = rand(Float64, 50, 100)
send(ch, matrix_data)
println("  Sent Float64 matrix: $(size(matrix_data))")

# Complex array
complex_data = Complex{Float32}[exp(im * i * 0.1) for i in 1:500]
send(ch, complex_data)
println("  Sent Complex{Float32} vector: $(length(complex_data)) elements")

# Receive and verify
received_float = receive(ch, Vector{Float32})
received_matrix = receive(ch, Matrix{Float64})
received_complex = receive(ch, Vector{Complex{Float32}})

println("Received arrays:")
@printf("  Float32 vector: %d elements, sum=%.3f\n", length(received_float), sum(received_float))
@printf("  Float64 matrix: %s, mean=%.3f\n", string(size(received_matrix)), mean(received_matrix))
@printf("  Complex vector: %d elements, magnitude=%.3f\n", 
        length(received_complex), sum(abs.(received_complex)))
println()

# Example 2: Broadcasting operations
println("2. Broadcasting Operations")
println("-" * 26)

# Create FloatVector message type for in-place operations
float_msg = FloatVector(1000)
for i in 1:length(float_msg)
    float_msg[i] = Float32(i)
end

println("Original FloatVector: first 5 = $(float_msg[1:5])")

# Broadcasting operations
float_msg.data .*= 2.0f0  # Scale by 2
println("After .*= 2: first 5 = $(float_msg[1:5])")

float_msg.data .+= 10.0f0  # Add offset
println("After .+= 10: first 5 = $(float_msg[1:5])")

# Apply function broadcast
float_msg.data .= sin.(float_msg.data .* 0.01f0)
println("After sin broadcast: first 5 = $(float_msg[1:5])")

# Send the modified message
send(ch, float_msg)
received_float_msg = receive(ch, FloatVector)
println("Sent and received FloatVector successfully")
println("  Verification: first 5 = $(received_float_msg[1:5])")
println()

# Example 3: Matrix operations and linear algebra
println("3. Matrix Operations and Linear Algebra")
println("-" * 38)

# Create test matrices
A = rand(Float64, 100, 100)
B = rand(Float64, 100, 100)

println("Created matrices A and B: $(size(A))")

# Send matrices
send(ch, A)
send(ch, B)

# Receive matrices
recv_A = receive(ch, Matrix{Float64})
recv_B = receive(ch, Matrix{Float64})

println("Performing linear algebra operations...")

# Matrix multiplication
C = recv_A * recv_B
@printf("  A * B = %s matrix, trace=%.3f\n", string(size(C)), tr(C))

# Eigenvalues (for a smaller matrix)
small_A = recv_A[1:20, 1:20]
eigenvals = eigvals(small_A)
@printf("  Eigenvalues of A[1:20,1:20]: max=%.3f, min=%.3f\n", 
        maximum(real.(eigenvals)), minimum(real.(eigenvals)))

# QR decomposition
Q, R = qr(recv_A)
@printf("  QR decomposition: Q is %s, R is %s\n", string(size(Q)), string(size(R)))

# Send results back
send(ch, C)
send(ch, Matrix(Q))  # Convert to regular matrix

received_C = receive(ch, Matrix{Float64})
received_Q = receive(ch, Matrix{Float64})
println("Sent and received computation results")
println()

# Example 4: Large array performance test
println("4. Large Array Performance Test")
println("-" * 31)

sizes = [1000, 10000, 100000, 1000000]
println("Testing different array sizes...")
println("Size (elements) | Send Time (ms) | Recv Time (ms) | Throughput (MB/s)")
println("----------------|----------------|----------------|------------------")

for size in sizes
    # Create large array
    large_array = rand(Float32, size)
    array_size_mb = sizeof(large_array) / 1e6
    
    # Measure send time
    send_start = time()
    send(ch, large_array)
    send_time = (time() - send_start) * 1000
    
    # Measure receive time
    recv_start = time()
    received_large = receive(ch, Vector{Float32})
    recv_time = (time() - recv_start) * 1000
    
    # Calculate throughput
    total_time = (send_time + recv_time) / 1000
    throughput = (array_size_mb * 2) / total_time  # Send + receive
    
    @printf("%15d | %14.2f | %14.2f | %16.2f\n",
            size, send_time, recv_time, throughput)
    
    # Verify data integrity
    if !isapprox(large_array, received_large, rtol=1e-6)
        println("Warning: Data integrity check failed for size $size")
    end
end
println()

# Example 5: Scientific computing workflow
println("5. Scientific Computing Workflow")
println("-" * 32)

# Simulate a typical scientific computing workflow
println("Simulating signal processing workflow...")

# Generate signal data
sample_rate = 44100  # Hz
duration = 1.0       # seconds
t = 0:1/sample_rate:duration-1/sample_rate
freq1, freq2 = 440, 880  # A4 and A5 notes

# Composite signal
signal = Float32.(sin.(2π * freq1 * t) + 0.5 * sin.(2π * freq2 * t))
noise = 0.1f0 * randn(Float32, length(signal))
noisy_signal = signal + noise

println("  Generated signal: $(length(signal)) samples")
@printf("  Sample rate: %d Hz, Duration: %.1f s\n", sample_rate, duration)

# Send raw signal
send(ch, noisy_signal)
received_signal = receive(ch, Vector{Float32})

# Apply simple low-pass filter (moving average)
window_size = 10
filtered_signal = similar(received_signal)
for i in eachindex(filtered_signal)
    start_idx = max(1, i - window_size ÷ 2)
    end_idx = min(length(received_signal), i + window_size ÷ 2)
    filtered_signal[i] = mean(received_signal[start_idx:end_idx])
end

println("  Applied moving average filter")

# Compute frequency domain representation (simplified)
# For demo purposes, we'll just compute power spectrum approximation
power_spectrum = abs.(filtered_signal).^2
max_power_idx = argmax(power_spectrum)
dominant_freq = (max_power_idx - 1) * sample_rate / length(power_spectrum)

@printf("  Dominant frequency: %.1f Hz\n", dominant_freq)

# Send processed results
send(ch, filtered_signal)
send(ch, power_spectrum)

received_filtered = receive(ch, Vector{Float32})
received_spectrum = receive(ch, Vector{Float32})

println("  Workflow completed: original → filtered → spectrum")
@printf("  Signal power reduction: %.1f%%\n", 
        (1 - sum(received_filtered.^2) / sum(received_signal.^2)) * 100)
println()

# Example 6: Memory-efficient operations with ByteVector
println("6. Memory-Efficient Operations with ByteVector")
println("-" * 45)

# Create a ByteVector for binary data
byte_msg = ByteVector(10000)

# Fill with pattern
for i in 1:length(byte_msg)
    byte_msg[i] = UInt8(i % 256)
end

println("Created ByteVector with pattern: first 10 = $(byte_msg[1:10])")

# Send as raw bytes
send(ch, byte_msg)
received_bytes = receive(ch, ByteVector)

# Reinterpret as different types
as_int16 = reinterpret(Int16, received_bytes.data)
as_float32 = reinterpret(Float32, received_bytes.data)

@printf("Reinterpreted as Int16: %d elements, first = %d\n", 
        length(as_int16), as_int16[1])
@printf("Reinterpreted as Float32: %d elements, first = %.3e\n", 
        length(as_float32), as_float32[1])
println()

# Example 7: Complex number operations
println("7. Complex Number Operations")
println("-" * 28)

# Create complex signal
N = 1000
complex_signal = ComplexVector(N)
for i in 1:N
    # Generate complex exponential
    complex_signal[i] = Complex{Float32}(exp(im * 2π * i / N))
end

println("Created complex exponential signal: $(length(complex_signal)) samples")

# Complex operations
magnitude = abs.(complex_signal.data)
phase = angle.(complex_signal.data)

@printf("  Magnitude range: [%.3f, %.3f]\n", minimum(magnitude), maximum(magnitude))
@printf("  Phase range: [%.3f, %.3f]\n", minimum(phase), maximum(phase))

# Send complex data
send(ch, complex_signal)
received_complex_msg = receive(ch, ComplexVector)

# Verify complex conjugate property
conjugated = conj.(received_complex_msg.data)
product = received_complex_msg.data .* conjugated
magnitude_squared = real.(product)

println("  Complex conjugate verification:")
@printf("    |z|² range: [%.6f, %.6f] (should be ~1.0)\n", 
        minimum(magnitude_squared), maximum(magnitude_squared))
println()

# Performance summary
final_metrics = get_metrics(ch)
println("8. Performance Summary")
println("-" * 21)
println(final_metrics)

total_time = time() - 0  # Would need to track start time properly
if final_metrics.messages_sent > 0
    @printf("Average message size: %.1f KB\n", 
            final_metrics.bytes_sent / final_metrics.messages_sent / 1024)
end

# Health check
health = health_check(ch)
println("Channel health: $(health.status)")

close(ch)

println()
println("=== Array Operations Example Complete ===")
println("This example demonstrated:")
println("  ✓ Basic array operations and verification")
println("  ✓ Broadcasting and in-place operations")
println("  ✓ Linear algebra with received matrices")
println("  ✓ Performance testing with different array sizes")
println("  ✓ Scientific computing workflow simulation")
println("  ✓ Memory-efficient operations with ByteVector")
println("  ✓ Complex number operations and verification")
println("  ✓ Type reinterpretation for binary data")