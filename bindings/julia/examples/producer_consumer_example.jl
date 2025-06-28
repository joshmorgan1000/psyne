#!/usr/bin/env julia

"""
Producer-Consumer Example for Psyne.jl

This example demonstrates multi-threaded producer-consumer patterns using
Psyne channels with different synchronization modes.

Run this example with:
    julia -t 4 producer_consumer_example.jl

The -t 4 flag enables 4 Julia threads for parallel execution.
"""

using Psyne
using Printf
using Base.Threads

println("=== Psyne.jl Producer-Consumer Example ===")
println("Julia threads available: $(nthreads())")
println()

if nthreads() < 2
    println("Warning: This example works best with multiple Julia threads.")
    println("Run with: julia -t 4 producer_consumer_example.jl")
    println()
end

# Configuration
const PRODUCER_COUNT = 2
const CONSUMER_COUNT = 2
const MESSAGES_PER_PRODUCER = 1000
const MESSAGE_SIZE = 1000  # Float32 elements per message

# Shared statistics
const stats_lock = SpinLock()
mutable struct Statistics
    total_produced::Int
    total_consumed::Int
    total_bytes_produced::Int
    total_bytes_consumed::Int
    producer_times::Vector{Float64}
    consumer_times::Vector{Float64}
end

stats = Statistics(0, 0, 0, 0, Float64[], Float64[])

function update_producer_stats(count::Int, bytes::Int, time::Float64)
    lock(stats_lock) do
        stats.total_produced += count
        stats.total_bytes_produced += bytes
        push!(stats.producer_times, time)
    end
end

function update_consumer_stats(count::Int, bytes::Int, time::Float64)
    lock(stats_lock) do
        stats.total_consumed += count
        stats.total_bytes_consumed += bytes
        push!(stats.consumer_times, time)
    end
end

# Producer function
function producer(producer_id::Int, channel::PsyneChannel, message_count::Int)
    println("Producer $producer_id starting...")
    
    local_produced = 0
    local_bytes = 0
    start_time = time()
    
    try
        for i in 1:message_count
            # Create unique data for this producer and message
            data = Float32[producer_id * 1000 + i + sin(j * 0.1) for j in 1:MESSAGE_SIZE]
            
            # Send the message
            send(channel, data)
            
            local_produced += 1
            local_bytes += sizeof(data)
            
            # Small random delay to simulate real work
            sleep(rand() * 0.001)  # 0-1ms
            
            # Progress report every 100 messages
            if i % 100 == 0
                @printf("Producer %d: %d/%d messages sent\n", producer_id, i, message_count)
            end
        end
        
    catch e
        println("Producer $producer_id error: $e")
    finally
        end_time = time()
        producer_time = end_time - start_time
        
        update_producer_stats(local_produced, local_bytes, producer_time)
        
        @printf("Producer %d finished: %d messages in %.3f seconds (%.1f msg/s)\n",
                producer_id, local_produced, producer_time, local_produced / producer_time)
    end
end

# Consumer function
function consumer(consumer_id::Int, channel::PsyneChannel, expected_total::Int)
    println("Consumer $consumer_id starting...")
    
    local_consumed = 0
    local_bytes = 0
    start_time = time()
    received_from_producers = Dict{Int, Int}()
    
    try
        while local_consumed < expected_total
            try
                # Receive a message with timeout
                data = receive(channel, Vector{Float32}, timeout_ms=1000)
                
                local_consumed += 1
                local_bytes += sizeof(data)
                
                # Determine which producer this came from (decode from data)
                if length(data) >= 1
                    producer_id = Int(floor(data[1] / 1000))
                    received_from_producers[producer_id] = get(received_from_producers, producer_id, 0) + 1
                end
                
                # Simulate some processing time
                sleep(rand() * 0.0005)  # 0-0.5ms
                
                # Progress report every 100 messages
                if local_consumed % 100 == 0
                    @printf("Consumer %d: %d/%d messages received\n", 
                            consumer_id, local_consumed, expected_total)
                end
                
            catch e
                if isa(e, PsyneError) && (e.code == PSYNE_ERROR_NO_MESSAGE || e.code == PSYNE_ERROR_TIMEOUT)
                    # Check if all producers are done
                    lock(stats_lock) do
                        if stats.total_produced >= PRODUCER_COUNT * MESSAGES_PER_PRODUCER
                            @printf("Consumer %d: All producers finished, stopping early\n", consumer_id)
                            break
                        end
                    end
                    continue
                else
                    println("Consumer $consumer_id error: $e")
                    break
                end
            end
        end
        
    finally
        end_time = time()
        consumer_time = end_time - start_time
        
        update_consumer_stats(local_consumed, local_bytes, consumer_time)
        
        @printf("Consumer %d finished: %d messages in %.3f seconds (%.1f msg/s)\n",
                consumer_id, local_consumed, consumer_time, local_consumed / consumer_time)
        
        println("Consumer $consumer_id received from producers: $received_from_producers")
    end
end

# Test different channel modes
function test_channel_mode(mode::ChannelMode, mode_name::String)
    println("\n" * "="^60)
    println("Testing $mode_name mode")
    println("="^60)
    
    # Reset statistics
    global stats = Statistics(0, 0, 0, 0, Float64[], Float64[])
    
    # Create channel for this test
    ch = channel("memory://prod_cons_$(mode_name)", 
                buffer_size=8*1024*1024,  # 8MB buffer
                mode=mode)
    
    enable_metrics!(ch, true)
    println("Created channel: $ch")
    
    # Calculate expected messages per consumer
    total_messages = PRODUCER_COUNT * MESSAGES_PER_PRODUCER
    messages_per_consumer = total_messages ÷ CONSUMER_COUNT
    
    println("Configuration:")
    println("  Producers: $PRODUCER_COUNT")
    println("  Consumers: $CONSUMER_COUNT") 
    println("  Messages per producer: $MESSAGES_PER_PRODUCER")
    println("  Total messages: $total_messages")
    println("  Messages per consumer: $messages_per_consumer")
    println("  Message size: $MESSAGE_SIZE Float32 elements ($(MESSAGE_SIZE * 4) bytes)")
    println()
    
    # Start all tasks
    overall_start = time()
    
    # Start producers
    producer_tasks = [Threads.@spawn producer(i, ch, MESSAGES_PER_PRODUCER) for i in 1:PRODUCER_COUNT]
    
    # Start consumers
    consumer_tasks = [Threads.@spawn consumer(i, ch, messages_per_consumer) for i in 1:CONSUMER_COUNT]
    
    # Wait for all producers to finish
    println("Waiting for producers to finish...")
    for task in producer_tasks
        wait(task)
    end
    println("All producers finished")
    
    # Wait for all consumers to finish
    println("Waiting for consumers to finish...")
    for task in consumer_tasks
        wait(task)
    end
    println("All consumers finished")
    
    overall_time = time() - overall_start
    
    # Get final channel metrics
    final_metrics = get_metrics(ch)
    
    # Print comprehensive results
    println("\n" * "-"^50)
    println("Results for $mode_name mode:")
    println("-"^50)
    
    @printf("Overall execution time: %.3f seconds\n", overall_time)
    @printf("Total messages produced: %d\n", stats.total_produced)
    @printf("Total messages consumed: %d\n", stats.total_consumed)
    @printf("Total bytes produced: %.2f MB\n", stats.total_bytes_produced / 1e6)
    @printf("Total bytes consumed: %.2f MB\n", stats.total_bytes_consumed / 1e6)
    
    if length(stats.producer_times) > 0
        avg_producer_time = sum(stats.producer_times) / length(stats.producer_times)
        @printf("Average producer time: %.3f seconds\n", avg_producer_time)
    end
    
    if length(stats.consumer_times) > 0
        avg_consumer_time = sum(stats.consumer_times) / length(stats.consumer_times)
        @printf("Average consumer time: %.3f seconds\n", avg_consumer_time)
    end
    
    # Channel metrics
    channel_tput = throughput(final_metrics, overall_time)
    channel_eff = efficiency(final_metrics)
    
    @printf("Channel throughput: %.2f MB/s (%.2f Mbps)\n", 
            channel_tput.bytes_per_second / 1e6, channel_tput.mbps)
    @printf("Channel message rate: %.0f messages/second\n", channel_tput.messages_per_second)
    @printf("Send blocking rate: %.1f%%\n", channel_eff.send_block_rate * 100)
    @printf("Receive blocking rate: %.1f%%\n", channel_eff.receive_block_rate * 100)
    @printf("Channel utilization: %.1f%%\n", channel_eff.utilization * 100)
    
    # Health check
    health = health_check(ch)
    println("Channel health: $(health.status)")
    if !isempty(health.issues)
        println("Issues: $(join(health.issues, ", "))")
    end
    
    close(ch)
    
    return (
        overall_time = overall_time,
        throughput_mbps = channel_tput.mbps,
        message_rate = channel_tput.messages_per_second,
        utilization = channel_eff.utilization
    )
end

# Run tests for different modes
println("Starting producer-consumer tests...")
println()

# Test results storage
results = Dict{String, NamedTuple}()

# Test SPSC mode (fastest, but limited to 1 producer and 1 consumer)
if PRODUCER_COUNT == 1 && CONSUMER_COUNT == 1
    results["SPSC"] = test_channel_mode(SPSC, "SPSC")
end

# Test MPSC mode (multiple producers, single consumer)
if CONSUMER_COUNT == 1
    results["MPSC"] = test_channel_mode(MPSC, "MPSC")
end

# Test SPMC mode (single producer, multiple consumers)
if PRODUCER_COUNT == 1
    results["SPMC"] = test_channel_mode(SPMC, "SPMC")
end

# Test MPMC mode (multiple producers, multiple consumers)
results["MPMC"] = test_channel_mode(MPMC, "MPMC")

# Summary comparison
if length(results) > 1
    println("\n" * "="^60)
    println("PERFORMANCE COMPARISON")
    println("="^60)
    
    println("Mode  | Time (s) | Throughput (Mbps) | Message Rate (msg/s) | Utilization (%)")
    println("------|----------|-------------------|----------------------|----------------")
    
    for (mode_name, result) in results
        @printf("%-5s | %8.3f | %17.2f | %20.0f | %14.1f\n",
                mode_name, result.overall_time, result.throughput_mbps, 
                result.message_rate, result.utilization * 100)
    end
end

println("\n=== Producer-Consumer Example Complete ===")
println("This example demonstrated:")
println("  ✓ Multi-threaded producer-consumer patterns")
println("  ✓ Different channel synchronization modes")
println("  ✓ Performance monitoring and comparison")
println("  ✓ Thread-safe statistics collection")
println("  ✓ Timeout handling and error recovery")
println("  ✓ Channel health monitoring")

if nthreads() == 1
    println("\nTip: Run with julia -t N for better parallelism (N = number of threads)")
end