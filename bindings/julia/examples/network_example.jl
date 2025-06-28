#!/usr/bin/env julia

"""
Network Communication Example for Psyne.jl

This example demonstrates network-based communication using TCP channels
with different modes and configurations.

Run this example with:
    julia network_example.jl [server|client] [port]

Default: runs as server on port 8080
"""

using Psyne
using Printf

function print_banner(title::String)
    println("=" ^ (length(title) + 8))
    println("=== $title ===")
    println("=" ^ (length(title) + 8))
end

function run_server(port::Int)
    print_banner("Psyne.jl Network Server")
    println("Starting server on port $port...")
    println()
    
    # Create a TCP server channel
    server_uri = "tcp://:$port"
    server_ch = channel(server_uri, 
                       buffer_size=2*1024*1024,  # 2MB buffer
                       mode=MPMC,                # Support multiple clients
                       compression=lz4_config(level=3))
    
    enable_metrics!(server_ch, true)
    println("Server channel created: $server_ch")
    println("Waiting for client connections...")
    println("Press Ctrl+C to stop server")
    println()
    
    message_count = 0
    start_time = time()
    last_metrics_time = start_time
    
    try
        while true
            try
                # Try to receive a message (non-blocking)
                data = receive(server_ch, timeout_ms=100)
                message_count += 1
                
                println("[$message_count] Received $(typeof(data)): $(summary(data))")
                
                # Echo the data back with some processing
                if isa(data, Vector{Float32})
                    # Apply some processing (e.g., scaling)
                    processed = data .* 2.0f0
                    send(server_ch, processed)
                    println("[$message_count] Sent processed Float32 vector (scaled by 2)")
                    
                elseif isa(data, Matrix{Float64})
                    # Apply matrix processing
                    processed = data .+ 1.0
                    send(server_ch, processed)
                    println("[$message_count] Sent processed Float64 matrix (+1)")
                    
                elseif isa(data, Vector{UInt8})
                    # Echo bytes back as-is
                    send(server_ch, data)
                    println("[$message_count] Echoed byte vector: $(String(data))")
                    
                else
                    # Echo unknown types back
                    send(server_ch, data)
                    println("[$message_count] Echoed unknown type")
                end
                
                # Print metrics every 10 messages
                if message_count % 10 == 0
                    current_time = time()
                    elapsed = current_time - last_metrics_time
                    
                    metrics = get_metrics(server_ch)
                    tput = throughput(metrics, current_time - start_time)
                    
                    @printf("--- Metrics after %d messages ---\n", message_count)
                    @printf("Throughput: %.2f MB/s (%.2f Mbps)\n", 
                            tput.bytes_per_second / 1e6, tput.mbps)
                    @printf("Message rate: %.1f messages/second\n", tput.messages_per_second)
                    println()
                    
                    last_metrics_time = current_time
                end
                
            catch e
                if isa(e, PsyneError) && e.code == PSYNE_ERROR_NO_MESSAGE
                    # No message available, continue waiting
                    continue
                elseif isa(e, PsyneError) && e.code == PSYNE_ERROR_TIMEOUT
                    # Timeout, continue waiting
                    continue
                else
                    println("Error receiving message: $e")
                    break
                end
            end
        end
        
    catch e
        if isa(e, InterruptException)
            println("\nServer interrupted by user")
        else
            println("Server error: $e")
        end
    finally
        println("\nShutting down server...")
        
        # Print final metrics
        if message_count > 0
            final_metrics = get_metrics(server_ch)
            total_time = time() - start_time
            final_tput = throughput(final_metrics, total_time)
            
            println("Final server statistics:")
            @printf("  Total messages: %d\n", message_count)
            @printf("  Total time: %.2f seconds\n", total_time)
            @printf("  Average throughput: %.2f MB/s\n", final_tput.bytes_per_second / 1e6)
            @printf("  Average message rate: %.1f messages/second\n", final_tput.messages_per_second)
        end
        
        close(server_ch)
        println("Server stopped")
    end
end

function run_client(host::String, port::Int)
    print_banner("Psyne.jl Network Client")
    println("Connecting to server at $host:$port...")
    println()
    
    # Create a TCP client channel
    client_uri = "tcp://$host:$port"
    client_ch = channel(client_uri,
                       buffer_size=2*1024*1024,  # 2MB buffer
                       mode=SPSC,                # Single client connection
                       compression=lz4_config(level=3))
    
    enable_metrics!(client_ch, true)
    println("Client channel created: $client_ch")
    println()
    
    # Test different data types
    test_cases = [
        ("Float32 vector", () -> rand(Float32, 1000)),
        ("Float64 matrix", () -> rand(Float64, 10, 20)),
        ("Byte message", () -> UInt8.(collect("Hello from Julia client!"))),
        ("Large Float32 array", () -> rand(Float32, 50000)),
        ("Complex numbers", () -> Complex{Float32}[1+2im, 3+4im, 5+6im]),
    ]
    
    println("Running test cases...")
    
    for (i, (test_name, data_generator)) in enumerate(test_cases)
        println("Test $i: $test_name")
        
        # Generate test data
        test_data = data_generator()
        println("  Generated: $(summary(test_data))")
        
        # Send data
        send_start = time()
        send(client_ch, test_data)
        send_time = time() - send_start
        println("  Sent in $(round(send_time * 1000, digits=2)) ms")
        
        # Receive response
        receive_start = time()
        response = receive(client_ch, typeof(test_data), timeout_ms=5000)
        receive_time = time() - receive_start
        println("  Received response in $(round(receive_time * 1000, digits=2)) ms")
        println("  Response: $(summary(response))")
        
        # Verify response (basic check)
        if isa(test_data, Vector{Float32}) && isa(response, Vector{Float32})
            expected = test_data .* 2.0f0
            is_correct = isapprox(response, expected, rtol=1e-6)
            println("  Processing correct: $is_correct")
        elseif isa(test_data, Matrix{Float64}) && isa(response, Matrix{Float64})
            expected = test_data .+ 1.0
            is_correct = isapprox(response, expected, rtol=1e-10)
            println("  Processing correct: $is_correct")
        elseif isa(test_data, Vector{UInt8}) && isa(response, Vector{UInt8})
            is_correct = response == test_data
            println("  Echo correct: $is_correct")
        end
        
        println()
        
        # Small delay between tests
        sleep(0.5)
    end
    
    # Performance test
    println("Running performance test...")
    println("Sending 1000 small messages rapidly...")
    
    reset_metrics!(client_ch)
    perf_start = time()
    
    for i in 1:1000
        small_data = rand(Float32, 100)
        send(client_ch, small_data)
        response = receive(client_ch, Vector{Float32}, timeout_ms=1000)
    end
    
    perf_time = time() - perf_start
    perf_metrics = get_metrics(client_ch)
    perf_tput = throughput(perf_metrics, perf_time)
    
    println("Performance test results:")
    @printf("  Time: %.3f seconds\n", perf_time)
    @printf("  Throughput: %.2f MB/s (%.2f Mbps)\n", 
            perf_tput.bytes_per_second / 1e6, perf_tput.mbps)
    @printf("  Round-trip rate: %.0f messages/second\n", perf_tput.messages_per_second / 2)
    println()
    
    # Test with different message sizes
    println("Testing different message sizes...")
    sizes = [100, 1000, 10000, 100000]  # Number of Float32 elements
    
    println("Size (elements) | Send Time (ms) | Recv Time (ms) | Round-trip (ms)")
    println("----------------|----------------|----------------|----------------")
    
    for size in sizes
        large_data = rand(Float32, size)
        
        # Measure send time
        send_start = time()
        send(client_ch, large_data)
        send_time = (time() - send_start) * 1000
        
        # Measure receive time
        recv_start = time()
        response = receive(client_ch, Vector{Float32}, timeout_ms=10000)
        recv_time = (time() - recv_start) * 1000
        
        total_time = send_time + recv_time
        
        @printf("%15d | %14.2f | %14.2f | %14.2f\n",
                size, send_time, recv_time, total_time)
    end
    println()
    
    # Test compression effectiveness
    println("Testing compression effectiveness...")
    
    # Highly compressible data
    compressible = Float32[sin(i * 0.01) for i in 1:10000 for _ in 1:10]
    send(client_ch, compressible)
    comp_response = receive(client_ch, Vector{Float32}, timeout_ms=5000)
    
    # Random data (less compressible)
    random_data = rand(Float32, length(compressible))
    send(client_ch, random_data)
    rand_response = receive(client_ch, Vector{Float32}, timeout_ms=5000)
    
    println("  Sent compressible data ($(length(compressible)) elements)")
    println("  Sent random data ($(length(random_data)) elements)")
    println("  Both received successfully - compression handled transparently")
    println()
    
    # Final metrics
    final_metrics = get_metrics(client_ch)
    println("Final client metrics:")
    println(final_metrics)
    
    # Cleanup
    close(client_ch)
    println("Client disconnected")
end

# Parse command line arguments
function main()
    args = ARGS
    
    if length(args) == 0
        # Default to server mode
        mode = "server"
        port = 8080
        host = "localhost"
    elseif length(args) == 1
        mode = args[1]
        port = 8080
        host = "localhost"
    elseif length(args) == 2
        mode = args[1]
        port = parse(Int, args[2])
        host = "localhost"
    elseif length(args) == 3
        mode = args[1]
        host = args[2]
        port = parse(Int, args[3])
    else
        println("Usage: julia network_example.jl [server|client] [host] [port]")
        println("Examples:")
        println("  julia network_example.jl                    # Server on port 8080")
        println("  julia network_example.jl server 9000       # Server on port 9000")
        println("  julia network_example.jl client 8080       # Client to localhost:8080")
        println("  julia network_example.jl client server.com 8080  # Client to server.com:8080")
        return
    end
    
    if mode == "server"
        run_server(port)
    elseif mode == "client"
        run_client(host, port)
    else
        println("Error: Mode must be 'server' or 'client'")
    end
end

# Only run main if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end