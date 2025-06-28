#!/usr/bin/env julia

"""
Performance Example for Psyne.jl

This example demonstrates performance monitoring, benchmarking, and optimization
techniques for Psyne channels.

Run this example with:
    julia performance_example.jl
"""

using Psyne
using Printf

println("=== Psyne.jl Performance Example ===")
println()

# Create a high-performance channel with metrics enabled
println("1. Creating high-performance channel...")
ch = channel("memory://perf_test", 
            buffer_size=4*1024*1024,  # 4MB buffer
            mode=SPSC)                # Single producer, single consumer for max speed

# Enable metrics for monitoring
enable_metrics!(ch, true)
println("   Created SPSC channel with 4MB buffer and metrics enabled")
println("   Channel: $ch")
println()

# Performance test function
function performance_test(channel, data_size::Int, num_messages::Int, test_name::String)
    println("2. Running $test_name...")
    println("   Data size: $data_size bytes per message")
    println("   Messages: $num_messages")
    
    # Reset metrics
    reset_metrics!(channel)
    
    # Create test data
    test_data = rand(Float32, data_size ÷ sizeof(Float32))
    
    # Measure send performance
    println("   Sending messages...")
    send_start = time()
    for i in 1:num_messages
        send(channel, test_data)
    end
    send_time = time() - send_start
    
    # Measure receive performance
    println("   Receiving messages...")
    receive_start = time()
    for i in 1:num_messages
        received = receive(channel, Vector{Float32})
    end
    receive_time = time() - receive_start
    
    total_time = send_time + receive_time
    total_bytes = num_messages * data_size
    
    # Get final metrics
    metrics = get_metrics(channel)
    
    # Calculate statistics
    throughput_stats = throughput(metrics, total_time)
    efficiency_stats = efficiency(metrics)
    
    println("   Results:")
    @printf("     Send time: %.3f seconds\n", send_time)
    @printf("     Receive time: %.3f seconds\n", receive_time)
    @printf("     Total time: %.3f seconds\n", total_time)
    @printf("     Throughput: %.2f MB/s (%.2f Mbps)\n", 
            throughput_stats.bytes_per_second / 1e6, throughput_stats.mbps)
    @printf("     Message rate: %.0f messages/second\n", throughput_stats.messages_per_second)
    @printf("     Send blocking rate: %.1f%%\n", efficiency_stats.send_block_rate * 100)
    @printf("     Receive blocking rate: %.1f%%\n", efficiency_stats.receive_block_rate * 100)
    @printf("     Channel utilization: %.1f%%\n", efficiency_stats.utilization * 100)
    println()
    
    return throughput_stats
end

# Run multiple performance tests
results = []

# Small messages, high frequency
result1 = performance_test(ch, 1024, 10000, "Small Messages Test")
push!(results, ("Small (1KB)", result1))

# Medium messages
result2 = performance_test(ch, 64*1024, 1000, "Medium Messages Test")
push!(results, ("Medium (64KB)", result2))

# Large messages
result3 = performance_test(ch, 1024*1024, 100, "Large Messages Test")
push!(results, ("Large (1MB)", result3))

# Summary comparison
println("3. Performance Summary:")
println("   Test Name         | Throughput (MB/s) | Message Rate (msg/s)")
println("   ------------------|-------------------|--------------------")
for (name, stats) in results
    @printf("   %-17s | %13.2f | %15.0f\n", 
            name, stats.bytes_per_second / 1e6, stats.messages_per_second)
end
println()

# Demonstrate compression impact
println("4. Testing compression impact...")

# Create compressed channel
comp_config = lz4_config(level=1, min_size=1024)
ch_compressed = channel("memory://perf_compressed", 
                       buffer_size=4*1024*1024,
                       mode=SPSC,
                       compression=comp_config)
enable_metrics!(ch_compressed, true)

# Test with compressible data (repeated pattern)
compressible_data = Float32[sin(i * 0.1) for i in 1:1000 for _ in 1:10]  # Repeated sine wave
println("   Testing with compressible data ($(length(compressible_data)) Float32s)...")

# Estimate compression ratio
estimated_ratio = estimate_compression_ratio(compressible_data, comp_config)
println("   Estimated compression ratio: $(round(estimated_ratio, digits=3))")

# Test uncompressed
reset_metrics!(ch)
start_time = time()
for i in 1:100
    send(ch, compressible_data)
    received = receive(ch, Vector{Float32})
end
uncompressed_time = time() - start_time
uncompressed_metrics = get_metrics(ch)

# Test compressed
reset_metrics!(ch_compressed)
start_time = time()
for i in 1:100
    send(ch_compressed, compressible_data)
    received = receive(ch_compressed, Vector{Float32})
end
compressed_time = time() - start_time
compressed_metrics = get_metrics(ch_compressed)

println("   Compression results:")
@printf("     Uncompressed time: %.3f seconds\n", uncompressed_time)
@printf("     Compressed time: %.3f seconds\n", compressed_time)
@printf("     Speed ratio: %.2fx %s\n", 
        uncompressed_time / compressed_time,
        compressed_time < uncompressed_time ? "(compressed faster)" : "(uncompressed faster)")

close(ch_compressed)
println()

# Demonstrate real-time monitoring
println("5. Real-time monitoring demo...")
println("   Monitoring channel for 5 seconds with background activity...")

# Start background activity in a task
background_task = @async begin
    test_data = rand(Float32, 1000)
    for i in 1:1000
        send(ch, test_data)
        received = receive(ch, Vector{Float32})
        sleep(0.005)  # 5ms between messages
    end
end

# Monitor the channel
snapshots = monitor(ch, 5.0, interval_seconds=1.0)

println("   Monitoring results:")
println("   Time (s) | Messages/s | MB/s   | Blocking %")
println("   ---------|------------|--------|----------")
for (i, snapshot) in enumerate(snapshots)
    if i > 1
        delta = snapshot - snapshots[i-1]
        tput = throughput(delta, 1.0)
        eff = efficiency(delta)
        blocking_pct = max(eff.send_block_rate, eff.receive_block_rate) * 100
        @printf("   %8d | %10.0f | %6.2f | %8.1f\n", 
                i-1, tput.messages_per_second, tput.bytes_per_second/1e6, blocking_pct)
    end
end

# Wait for background task to complete
wait(background_task)
println()

# Health check
println("6. Channel health check...")
health = health_check(ch)
println("   Status: $(health.status)")
if !isempty(health.issues)
    println("   Issues:")
    for issue in health.issues
        println("     - $issue")
    end
end
if !isempty(health.recommendations)
    println("   Recommendations:")
    for rec in health.recommendations
        println("     - $rec")
    end
end
println()

# Benchmark different algorithms
println("7. Compression algorithm benchmark...")
benchmark_data = rand(Float32, 10000)
bench_results = compression_benchmark(benchmark_data)

println("   Algorithm | Compression Ratio | Est. Time (ms) | Throughput (MB/s)")
println("   ----------|-------------------|----------------|------------------")
for (alg, metrics) in bench_results
    @printf("   %-9s | %13.3f | %10.2f | %13.2f\n",
            alg, metrics[:ratio], metrics[:time_ms], metrics[:throughput_mbps])
end

# Get recommendation
recommendation = recommend_compression(benchmark_data, priority=:balanced)
println("   Recommended configuration: $recommendation")
println()

# Cleanup
println("8. Cleaning up...")
close(ch)
println("   All channels closed")

println()
println("=== Performance Example Complete ===")
println("This example demonstrated:")
println("  ✓ High-performance channel configuration")
println("  ✓ Performance testing and benchmarking")
println("  ✓ Metrics collection and analysis")
println("  ✓ Compression impact measurement")
println("  ✓ Real-time monitoring")
println("  ✓ Health checking")
println("  ✓ Algorithm benchmarking and recommendations")