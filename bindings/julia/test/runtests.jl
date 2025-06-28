#!/usr/bin/env julia

"""
Test suite for Psyne.jl

Comprehensive tests for the Julia bindings to ensure correctness,
performance, and reliability.

Run with: julia --project=.. test/runtests.jl
Or: julia --project=.. -e "using Pkg; Pkg.test()"
"""

using Test
using Psyne

# Test configuration
const TEST_BUFFER_SIZE = 1024 * 1024  # 1MB
const TEST_TIMEOUT_MS = 1000

@testset "Psyne.jl Tests" begin

    @testset "Library Initialization" begin
        @test psyne_version() isa String
        @test length(psyne_version()) > 0
        @test occursin(".", psyne_version())  # Should contain version dots
    end

    @testset "Error Handling" begin
        # Test PsyneError
        err = PsyneError("Test error", -1)
        @test err.message == "Test error"
        @test err.code == -1
        
        # Test error message formatting
        @test occursin("Test error", string(err))
        @test occursin("-1", string(err))
    end

    @testset "Channel Creation and Management" begin
        @testset "Memory Channels" begin
            # Basic memory channel
            ch = channel("memory://test1")
            @test ch isa PsyneChannel
            @test get_uri(ch) == "memory://test1"
            @test get_buffer_size(ch) == 1024*1024  # Default size
            @test get_mode(ch) == SPSC  # Default mode
            @test get_type(ch) == MultiType  # Default type
            @test !is_stopped(ch)
            
            # Close and test
            close(ch)
            @test ch.handle == C_NULL
            
            # Custom configuration
            ch2 = channel("memory://test2", 
                         buffer_size=2*1024*1024,
                         mode=MPMC,
                         type=SingleType)
            @test get_buffer_size(ch2) == 2*1024*1024
            @test get_mode(ch2) == MPMC
            @test get_type(ch2) == SingleType
            close(ch2)
        end

        @testset "Channel Operations" begin
            ch = channel("memory://ops_test")
            
            # Stop functionality
            @test !is_stopped(ch)
            stop!(ch)
            @test is_stopped(ch)
            
            close(ch)
        end
    end

    @testset "Message Types and Operations" begin
        ch = channel("memory://message_test", buffer_size=TEST_BUFFER_SIZE)
        enable_metrics!(ch, true)
        
        @testset "Float32 Vectors" begin
            # Send and receive Vector{Float32}
            original = Float32[1.0, 2.0, 3.14159, 4.0, 5.0]
            send(ch, original)
            received = receive(ch, Vector{Float32})
            @test received == original
            @test typeof(received) == Vector{Float32}
            
            # FloatVector message type
            float_msg = FloatVector(10)
            for i in 1:length(float_msg)
                float_msg[i] = Float32(i * 0.1)
            end
            send(ch, float_msg)
            received_msg = receive(ch, FloatVector)
            @test received_msg.data == float_msg.data
        end

        @testset "Float64 Matrices" begin
            # Send and receive Matrix{Float64}
            original = rand(Float64, 5, 10)
            send(ch, original)
            received = receive(ch, Matrix{Float64})
            @test received == original
            @test size(received) == size(original)
            
            # DoubleMatrix message type
            matrix_msg = DoubleMatrix(3, 4)
            matrix_msg.data .= reshape(1.0:12.0, 3, 4)
            send(ch, matrix_msg)
            received_matrix_msg = receive(ch, DoubleMatrix)
            @test received_matrix_msg.data == matrix_msg.data
        end

        @testset "Byte Vectors" begin
            # Send and receive Vector{UInt8}
            original = UInt8[0x48, 0x65, 0x6C, 0x6C, 0x6F]  # "Hello"
            send(ch, original)
            received = receive(ch, Vector{UInt8})
            @test received == original
            @test String(received) == "Hello"
            
            # ByteVector message type
            byte_msg = ByteVector(5)
            byte_msg.data .= [1, 2, 3, 4, 5]
            send(ch, byte_msg)
            received_byte_msg = receive(ch, ByteVector)
            @test received_byte_msg.data == byte_msg.data
        end

        @testset "Complex Vectors" begin
            # Send and receive Vector{Complex{Float32}}
            original = Complex{Float32}[1+2im, 3+4im, 5+6im]
            send(ch, original)
            received = receive(ch, Vector{Complex{Float32}})
            @test received == original
            
            # ComplexVector message type
            complex_msg = ComplexVector(3)
            complex_msg.data .= [1+2im, 3+4im, 5+6im]
            send(ch, complex_msg)
            received_complex_msg = receive(ch, ComplexVector)
            @test received_complex_msg.data == complex_msg.data
        end

        @testset "Automatic Type Detection" begin
            # Send different types and receive without type specification
            test_data = [
                Float32[1.0, 2.0, 3.0],
                rand(Float64, 3, 3),
                UInt8[1, 2, 3, 4],
                Complex{Float32}[1+im, 2+2im]
            ]
            
            for data in test_data
                send(ch, data)
                received = receive(ch)
                @test received == data
                @test typeof(received) == typeof(data)
            end
        end

        close(ch)
    end

    @testset "Array Operations and Broadcasting" begin
        ch = channel("memory://array_test")
        
        @testset "Broadcasting on Message Types" begin
            # FloatVector broadcasting
            fv = FloatVector(5)
            fv.data .= [1.0, 2.0, 3.0, 4.0, 5.0]
            
            # Test broadcasting operations
            fv.data .*= 2.0f0
            @test fv.data == [2.0, 4.0, 6.0, 8.0, 10.0]
            
            fv.data .+= 1.0f0
            @test fv.data == [3.0, 5.0, 7.0, 9.0, 11.0]
            
            # Function broadcasting
            fv.data .= sin.(fv.data)
            @test all(isfinite.(fv.data))
        end

        @testset "Array Interface" begin
            # Test array-like interface for message types
            fv = FloatVector(10)
            
            # Size and length
            @test size(fv) == (10,)
            @test length(fv) == 10
            
            # Indexing
            fv[1] = 1.0f0
            fv[end] = 10.0f0
            @test fv[1] == 1.0f0
            @test fv[end] == 10.0f0
            
            # Iteration
            count = 0
            for x in fv
                count += 1
            end
            @test count == 10
        end

        close(ch)
    end

    @testset "Compression" begin
        @testset "Compression Configurations" begin
            # Test compression config creation
            lz4_conf = lz4_config()
            @test lz4_conf.type == LZ4
            @test lz4_conf.level == 1
            @test lz4_conf.enable_checksum == true
            
            zstd_conf = zstd_config(level=5)
            @test zstd_conf.type == Zstd
            @test zstd_conf.level == 5
            
            snappy_conf = snappy_config(min_size=256)
            @test snappy_conf.type == Snappy
            @test snappy_conf.min_size_threshold == 256
            
            none_conf = no_compression()
            @test none_conf.type == None
        end

        @testset "Compression Estimation" begin
            # Test compression ratio estimation
            random_data = rand(Float32, 1000)
            patterned_data = Float32[sin(i * 0.1) for i in 1:1000 for _ in 1:10]
            
            config = lz4_config()
            
            random_ratio = estimate_compression_ratio(random_data, config)
            pattern_ratio = estimate_compression_ratio(patterned_data, config)
            
            @test 0.0 < random_ratio <= 1.0
            @test 0.0 < pattern_ratio <= 1.0
            # Patterned data should compress better
            @test pattern_ratio <= random_ratio
        end

        @testset "Compression Recommendations" begin
            test_data = rand(Float32, 5000)
            
            speed_config = recommend_compression(test_data, priority=:speed)
            @test speed_config.type in [LZ4, Snappy, None]
            
            ratio_config = recommend_compression(test_data, priority=:ratio)
            @test ratio_config.type in [Zstd, Snappy, None]
            
            balanced_config = recommend_compression(test_data, priority=:balanced)
            @test balanced_config.type in [LZ4, Zstd, Snappy, None]
        end
    end

    @testset "Metrics and Monitoring" begin
        ch = channel("memory://metrics_test")
        enable_metrics!(ch, true)
        
        @testset "Basic Metrics" begin
            # Reset metrics
            reset_metrics!(ch)
            initial_metrics = get_metrics(ch)
            @test initial_metrics.messages_sent == 0
            @test initial_metrics.bytes_sent == 0
            @test initial_metrics.messages_received == 0
            @test initial_metrics.bytes_received == 0
            
            # Send some data
            test_data = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
            send(ch, test_data)
            received_data = receive(ch, Vector{Float32})
            
            # Check metrics updated
            metrics = get_metrics(ch)
            @test metrics.messages_sent == 1
            @test metrics.messages_received == 1
            @test metrics.bytes_sent > 0
            @test metrics.bytes_received > 0
        end

        @testset "Metrics Arithmetic" begin
            # Test metrics arithmetic operations
            m1 = Metrics(10, 1000, 5, 500, 1, 0)
            m2 = Metrics(5, 250, 10, 750, 0, 1)
            
            sum_metrics = m1 + m2
            @test sum_metrics.messages_sent == 15
            @test sum_metrics.bytes_sent == 1250
            @test sum_metrics.messages_received == 15
            @test sum_metrics.bytes_received == 1250
            
            diff_metrics = m1 - m2
            @test diff_metrics.messages_sent == 5
            @test diff_metrics.bytes_sent == 750
        end

        @testset "Throughput and Efficiency" begin
            # Create test metrics
            test_metrics = Metrics(1000, 100000, 950, 95000, 50, 25)
            
            # Test throughput calculation
            tput = throughput(test_metrics, 10.0)  # 10 seconds
            @test tput.messages_per_second ≈ 195.0  # (1000+950)/10
            @test tput.bytes_per_second ≈ 19500.0   # (100000+95000)/10
            @test tput.mbps ≈ 1.56  # 19500*8/1e6
            
            # Test efficiency calculation
            eff = efficiency(test_metrics)
            @test 0.0 <= eff.send_block_rate <= 1.0
            @test 0.0 <= eff.receive_block_rate <= 1.0
            @test 0.0 <= eff.utilization <= 1.0
        end

        @testset "Health Check" begin
            health = health_check(ch)
            @test health.status in [:healthy, :warning, :error]
            @test health.issues isa Vector{String}
            @test health.recommendations isa Vector{String}
        end

        close(ch)
    end

    @testset "Error Conditions and Edge Cases" begin
        @testset "Invalid Channel Operations" begin
            # Test operations on closed channel
            ch = channel("memory://error_test")
            close(ch)
            
            # Should not crash, but may throw PsyneError
            @test_throws PsyneError send(ch, Float32[1.0, 2.0])
            @test_throws PsyneError receive(ch, Vector{Float32})
        end

        @testset "Timeout Operations" begin
            ch = channel("memory://timeout_test")
            
            # Receive with timeout on empty channel
            @test_throws PsyneError receive(ch, Vector{Float32}, timeout_ms=100)
            
            close(ch)
        end

        @testset "Large Data" begin
            ch = channel("memory://large_test", buffer_size=16*1024*1024)
            
            # Test with large array (close to buffer size)
            large_data = rand(Float32, 1000000)  # ~4MB
            send(ch, large_data)
            received_large = receive(ch, Vector{Float32})
            @test received_large == large_data
            
            close(ch)
        end
    end

    @testset "Performance Characteristics" begin
        @testset "Basic Performance" begin
            ch = channel("memory://perf_test", mode=SPSC, buffer_size=4*1024*1024)
            enable_metrics!(ch, true)
            
            # Performance test with moderate load
            num_messages = 100
            message_size = 1000
            
            reset_metrics!(ch)
            start_time = time()
            
            for i in 1:num_messages
                data = rand(Float32, message_size)
                send(ch, data)
                received = receive(ch, Vector{Float32})
                @test length(received) == message_size
            end
            
            end_time = time()
            duration = end_time - start_time
            
            # Basic performance checks
            @test duration > 0.0
            @test duration < 10.0  # Should complete within 10 seconds
            
            metrics = get_metrics(ch)
            @test metrics.messages_sent == num_messages
            @test metrics.messages_received == num_messages
            
            tput = throughput(metrics, duration)
            @test tput.messages_per_second > 0
            @test tput.bytes_per_second > 0
            
            close(ch)
        end
    end

end

println("All tests completed!")