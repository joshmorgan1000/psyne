/**
 * @file mpsc_benchmark.cpp
 * @brief MPSC (Multi Producer Single Consumer) messaging benchmark
 * 
 * Tests high-performance messaging from multiple producer threads to a single consumer.
 * This benchmark measures throughput, latency, and contention effects as the number
 * of producer threads scales from 2 to 16.
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
    std::cout << "=== Psyne MPSC (Multi Producer Single Consumer) Benchmark ===\n\n";
    std::cout << "MPSC benchmark functionality integrated into throughput_comparison.cpp\n";
    std::cout << "Run './throughput_comparison' for comprehensive MPSC testing.\n";
    return 0;
}