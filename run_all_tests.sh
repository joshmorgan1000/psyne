#!/bin/bash

# PSYNE Test Runner Script
# Runs all examples correctly, handling paired examples and command line arguments

cd build || exit 1

echo "=== PSYNE EXAMPLES TEST REPORT ==="
echo "Generated: $(date)"
echo "=================================="
echo ""

# Counter for pass/fail
PASSED=0
FAILED=0

# Function to run a test and report result
run_test() {
    local name="$1"
    local command="$2"
    local timeout="${3:-10}"
    
    echo -n "Testing $name... "
    if timeout ${timeout}s bash -c "$command" > /tmp/test_output_$$.txt 2>&1; then
        echo "PASS"
        ((PASSED++))
    else
        echo "FAIL"
        ((FAILED++))
        # Show first few lines of error
        head -5 /tmp/test_output_$$.txt | sed 's/^/  > /'
    fi
    rm -f /tmp/test_output_$$.txt
}

echo "=== Standalone Examples ==="
echo ""

# Simple examples that should just work
for example in arrow_demo async_messaging_demo c_api_demo collective_simple_test \
               compression_demo custom_allocator_demo fixed_size_demo \
               performance_demo simd_demo tensor_optimization_demo \
               test_bytevector test_floatvector windows_test \
               simple_messaging_zero_copy filtered_fanout_demo \
               pair_pattern_demo modern_cpp20_demo enhanced_types_demo \
               message_types_demo simple_enhanced_types_test \
               high_performance_messaging metrics_demo debug_demo \
               zero_copy_showcase simple_messaging dynamic_allocation_demo \
               multi_type_channel coroutine_example publish_subscribe_demo \
               request_reply_demo routing_demo_final rudp_demo; do
    if [[ -x "examples/$example" ]]; then
        run_test "$example" "./examples/$example"
    fi
done

echo ""
echo "=== Examples with Command Line Arguments ==="
echo ""

# channel_factory_demo - needs mode argument
if [[ -x "examples/channel_factory_demo" ]]; then
    run_test "channel_factory_demo (memory)" "./examples/channel_factory_demo memory"
fi

# collective_demo - needs rank argument
if [[ -x "examples/collective_demo" ]]; then
    run_test "collective_demo" "./examples/collective_demo 0"
fi

# Examples that need "server" or "client" mode
for example in tcp_demo unix_socket_demo websocket_demo; do
    if [[ -x "examples/$example" ]]; then
        # Run server in background, then client
        run_test "$example" "
            ./examples/$example server > /dev/null 2>&1 &
            SERVER_PID=\$!
            sleep 1
            ./examples/$example client
            kill \$SERVER_PID 2>/dev/null
            wait \$SERVER_PID 2>/dev/null
        "
    fi
done

echo ""
echo "=== IPC Producer/Consumer Pair ==="
echo ""

if [[ -x "examples/ipc_producer" && -x "examples/ipc_consumer" ]]; then
    run_test "ipc_producer_consumer" "
        ./examples/ipc_producer > /dev/null 2>&1 &
        PRODUCER_PID=\$!
        sleep 1
        ./examples/ipc_consumer
        kill \$PRODUCER_PID 2>/dev/null
        wait \$PRODUCER_PID 2>/dev/null
    " 20
fi

echo ""
echo "=== TCP Server/Client Pair ==="
echo ""

if [[ -x "examples/tcp_server" && -x "examples/tcp_client" ]]; then
    run_test "tcp_server_client" "
        ./examples/tcp_server > /dev/null 2>&1 &
        SERVER_PID=\$!
        sleep 2
        ./examples/tcp_client
        kill \$SERVER_PID 2>/dev/null
        wait \$SERVER_PID 2>/dev/null
    " 20
fi

echo ""
echo "=== Producer/Consumer (Threaded) ==="
echo ""

if [[ -x "examples/producer_consumer" ]]; then
    run_test "producer_consumer" "./examples/producer_consumer" 15
fi

echo ""
echo "=== Channel Patterns ==="
echo ""

if [[ -x "examples/channel_patterns_showcase" ]]; then
    run_test "channel_patterns_showcase" "./examples/channel_patterns_showcase"
fi

echo ""
echo "=== Network Examples (May Fail Without Setup) ==="
echo ""

# These might fail due to network requirements
for example in udp_multicast_demo quic_demo webrtc_demo debug_multicast; do
    if [[ -x "examples/$example" ]]; then
        run_test "$example" "./examples/$example" 5
    fi
done

# grpc_demo needs mode argument
if [[ -x "examples/grpc_demo" ]]; then
    run_test "grpc_demo (tcp-client)" "./examples/grpc_demo tcp-client" 5
fi

# webrtc examples need peer names
if [[ -x "examples/webrtc_p2p_demo" ]]; then
    run_test "webrtc_p2p_demo" "./examples/webrtc_p2p_demo player1" 5
fi

if [[ -x "examples/webrtc_simple_example" ]]; then
    run_test "webrtc_simple_example" "./examples/webrtc_simple_example offerer" 5
fi

echo ""
echo "=== IPC Demo (Special Case) ==="
echo ""

if [[ -x "examples/ipc_demo" ]]; then
    # ipc_demo needs producer/consumer argument
    run_test "ipc_demo" "
        ./examples/ipc_demo producer > /dev/null 2>&1 &
        PRODUCER_PID=\$!
        sleep 1
        ./examples/ipc_demo consumer
        kill \$PRODUCER_PID 2>/dev/null
        wait \$PRODUCER_PID 2>/dev/null
    " 15
fi

echo ""
echo "=== IPC Test ==="
echo ""

if [[ -x "examples/ipc_test" ]]; then
    run_test "ipc_test" "./examples/ipc_test"
fi

echo ""
echo "=================================="
echo "SUMMARY:"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo "  Total:  $((PASSED + FAILED))"
echo "=================================="
echo ""

# Exit with error if any tests failed
if [[ $FAILED -gt 0 ]]; then
    echo "Some tests failed!"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi