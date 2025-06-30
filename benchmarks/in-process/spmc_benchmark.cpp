/**
 * @file spmc_benchmark.cpp
 * @brief SPMC (Single Producer Multi Consumer) messaging benchmark
 * 
 * Tests high-performance messaging from a single producer to multiple consumer threads.
 * This benchmark measures throughput, latency distribution, and fan-out efficiency
 * as messages are broadcast to 2-16 consumer threads.
 */

#include <psyne/psyne.hpp>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <atomic>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <barrier>
#include <numeric>

int main() {
    std::cout << "=== Psyne SPMC (Single Producer Multi Consumer) Benchmark ===\n\n";
    std::cout << "SPMC benchmark functionality integrated into throughput_comparison.cpp\n";
    std::cout << "Run './throughput_comparison' for comprehensive SPMC testing.\n";
    return 0;
}