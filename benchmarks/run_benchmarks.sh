#!/bin/bash

# Psyne Comprehensive Benchmark Suite
# Tests various messaging patterns to saturate hardware

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Psyne Messaging Patterns Benchmark Suite${NC}"
echo "======================================="
echo ""

# Check if benchmark binary exists
BENCHMARK_BIN="./messaging_patterns_benchmark"
if [ ! -f "$BENCHMARK_BIN" ]; then
    echo -e "${RED}Error: Benchmark binary not found at $BENCHMARK_BIN${NC}"
    echo "Please build the benchmarks first"
    exit 1
fi

# Get system info
echo -e "${YELLOW}System Information:${NC}"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "CPU: $(lscpu | grep 'Model name' | cut -d ':' -f2 | xargs)"
    echo "Cores: $(nproc)"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
    
    # Set performance governor if available
    if command -v cpupower &> /dev/null; then
        echo "Setting CPU governor to performance..."
        sudo cpupower frequency-set -g performance 2>/dev/null || true
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
    echo "Cores: $(sysctl -n hw.ncpu)"
    echo "Memory: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 "GB"}')"
fi
echo ""

# Function to run a benchmark and capture results
run_benchmark() {
    local name="$1"
    local transport="$2"
    local pattern="$3"
    local extra_args="${4:-}"
    
    echo -e "${YELLOW}Running $name...${NC}"
    $BENCHMARK_BIN --transport $transport --pattern $pattern $extra_args 2>&1 | tee benchmark_${name// /_}.log
    echo ""
}

# Test 1: Throughput Tests (Small Messages)
echo -e "${GREEN}=== THROUGHPUT TESTS (Small Messages) ===${NC}"
echo "Testing maximum message rate with 64-byte messages"
echo ""

# SPSC In-Process
run_benchmark "SPSC Throughput" "spsc" "throughput" "--count 10000000 --size 64"

# IPC 
run_benchmark "IPC Throughput" "ipc" "throughput" "--count 10000000 --size 64"

# TCP (localhost)
run_benchmark "TCP Throughput" "tcp" "throughput" "--count 1000000 --size 64"

# Test 2: Bandwidth Tests (Large Messages)
echo -e "${GREEN}=== BANDWIDTH TESTS (Large Messages) ===${NC}"
echo "Testing maximum bandwidth with various message sizes"
echo ""

run_benchmark "SPSC Bandwidth" "spsc" "bandwidth"
run_benchmark "IPC Bandwidth" "ipc" "bandwidth"
run_benchmark "TCP Bandwidth" "tcp" "bandwidth"

# Test 3: Burst Tests
echo -e "${GREEN}=== BURST TESTS ===${NC}"
echo "Testing behavior under bursty traffic patterns"
echo ""

run_benchmark "IPC Burst Small" "ipc" "burst" "--size 64 --burst 10000 --count 100000"
run_benchmark "IPC Burst Large" "ipc" "burst" "--size 65536 --burst 100 --count 1000"

# Test 4: CPU Saturation Tests
echo -e "${GREEN}=== CPU SATURATION TESTS ===${NC}"
echo "Testing with CPU affinity to maximize performance"
echo ""

# Get number of physical cores
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PHYSICAL_CORES=$(lscpu | grep "Core(s) per socket" | awk '{print $4}')
    SOCKETS=$(lscpu | grep "Socket(s)" | awk '{print $2}')
    TOTAL_CORES=$((PHYSICAL_CORES * SOCKETS))
elif [[ "$OSTYPE" == "darwin"* ]]; then
    TOTAL_CORES=$(sysctl -n hw.physicalcpu)
fi

echo "Using $TOTAL_CORES physical cores for saturation test"

# Single producer/consumer with affinity
run_benchmark "IPC CPU Affinity" "ipc" "throughput" "--count 10000000 --size 256 --affinity"

# Multi-threaded stress test
echo -e "${GREEN}=== MULTI-THREADED STRESS TESTS ===${NC}"
echo "Testing with multiple producers and consumers"
echo ""

# Scale up producers/consumers
for np in 1 2 4; do
    for nc in 1 2 4; do
        if [ $((np * nc)) -le $TOTAL_CORES ]; then
            run_benchmark "Stress Test ${np}P-${nc}C" "ipc" "stress" \
                "--producers $np --consumers $nc --count 1000000 --size 512 --affinity"
        fi
    done
done

# Test 5: Memory Pressure Test
echo -e "${GREEN}=== MEMORY PRESSURE TEST ===${NC}"
echo "Testing with large messages to stress memory subsystem"
echo ""

# Very large messages
run_benchmark "Memory Pressure 1MB" "ipc" "throughput" "--count 10000 --size 1048576"
run_benchmark "Memory Pressure 16MB" "ipc" "throughput" "--count 1000 --size 16777216"

# Test 6: Latency Distribution Test
echo -e "${GREEN}=== LATENCY DISTRIBUTION TEST ===${NC}"
echo "Measuring latency percentiles"
echo ""

# Create a special latency test that captures detailed statistics
cat > latency_test.cpp << 'EOF'
#include "psyne/psyne.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

using namespace psyne;
using namespace std::chrono;

int main() {
    ChannelConfig config;
    config.name = "latency_test_channel";
    config.size_mb = 16;
    config.mode = ChannelMode::SPSC;
    config.transport = ChannelTransport::IPC;
    config.blocking = true;
    
    const size_t num_samples = 100000;
    std::vector<uint64_t> latencies;
    latencies.reserve(num_samples);
    
    // Producer thread
    std::thread producer([&]() {
        config.is_producer = true;
        auto channel = Channel<uint64_t>::create(config);
        
        for (size_t i = 0; i < num_samples; ++i) {
            auto msg = channel->allocate();
            *msg = duration_cast<nanoseconds>(
                high_resolution_clock::now().time_since_epoch()).count();
            msg.send();
            
            // Pace messages to avoid queue buildup
            if (i % 100 == 0) {
                std::this_thread::sleep_for(microseconds(1));
            }
        }
    });
    
    // Consumer thread
    std::thread consumer([&]() {
        config.is_producer = false;
        auto channel = Channel<uint64_t>::create(config);
        
        for (size_t i = 0; i < num_samples; ++i) {
            auto msg = channel->receive();
            uint64_t now = duration_cast<nanoseconds>(
                high_resolution_clock::now().time_since_epoch()).count();
            latencies.push_back(now - *msg);
        }
    });
    
    producer.join();
    consumer.join();
    
    // Calculate percentiles
    std::sort(latencies.begin(), latencies.end());
    
    std::cout << "Latency Distribution (nanoseconds):" << std::endl;
    std::cout << "Min:    " << latencies[0] << " ns" << std::endl;
    std::cout << "50th:   " << latencies[num_samples * 50 / 100] << " ns" << std::endl;
    std::cout << "90th:   " << latencies[num_samples * 90 / 100] << " ns" << std::endl;
    std::cout << "95th:   " << latencies[num_samples * 95 / 100] << " ns" << std::endl;
    std::cout << "99th:   " << latencies[num_samples * 99 / 100] << " ns" << std::endl;
    std::cout << "99.9th: " << latencies[num_samples * 999 / 1000] << " ns" << std::endl;
    std::cout << "Max:    " << latencies.back() << " ns" << std::endl;
    
    return 0;
}
EOF

# Compile and run if compiler is available
if command -v g++ &> /dev/null; then
    echo "Compiling latency test..."
    g++ -std=c++20 -O3 -I../include latency_test.cpp -L../build -lpsyne -lpthread -o latency_test
    if [ -f "./latency_test" ]; then
        echo "Running latency distribution test..."
        ./latency_test
    fi
fi

# Generate summary report
echo -e "${GREEN}=== BENCHMARK SUMMARY ===${NC}"
echo "Generating summary report..."

# Parse log files and extract key metrics
cat > benchmark_summary.txt << EOF
Psyne Benchmark Summary
=====================

Date: $(date)
System: $(uname -a)

Key Results:
-----------
EOF

# Extract throughput results
echo "" >> benchmark_summary.txt
echo "Throughput (messages/second):" >> benchmark_summary.txt
grep -h "msg/s" benchmark_*.log | sort -k2 -nr >> benchmark_summary.txt || true

echo "" >> benchmark_summary.txt
echo "Bandwidth (MB/s):" >> benchmark_summary.txt
grep -h "MB/s" benchmark_*.log | grep -v "msg/s" | sort -k2 -nr >> benchmark_summary.txt || true

echo "" >> benchmark_summary.txt
echo "Latency (microseconds):" >> benchmark_summary.txt
grep -h "latency:" benchmark_*.log | sort -k3 -n >> benchmark_summary.txt || true

echo ""
echo -e "${GREEN}Benchmark suite completed!${NC}"
echo "Results saved to:"
echo "  - Individual logs: benchmark_*.log"
echo "  - Summary: benchmark_summary.txt"

# Reset CPU governor if we changed it
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v cpupower &> /dev/null; then
    echo ""
    echo "Resetting CPU governor to ondemand..."
    sudo cpupower frequency-set -g ondemand 2>/dev/null || true
fi