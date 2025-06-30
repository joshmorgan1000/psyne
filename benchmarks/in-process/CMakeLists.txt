# In-process benchmarks

# SPSC (Single Producer Single Consumer) benchmark
add_benchmark_executable(spsc_benchmark spsc_benchmark.cpp)

# MPSC (Multi Producer Single Consumer) benchmark  
add_benchmark_executable(mpsc_benchmark mpsc_benchmark.cpp)

# SPMC (Single Producer Multi Consumer) benchmark
add_benchmark_executable(spmc_benchmark spmc_benchmark.cpp)

# MPMC (Multi Producer Multi Consumer) benchmark
add_benchmark_executable(mpmc_benchmark mpmc_benchmark.cpp)

# Throughput comparison benchmark
add_benchmark_executable(throughput_comparison throughput_comparison.cpp)

# Latency benchmark
add_benchmark_executable(latency_benchmark latency_benchmark.cpp)

# Memory efficiency benchmark
add_benchmark_executable(memory_benchmark memory_benchmark.cpp)

# Scalability benchmark
add_benchmark_executable(scalability_benchmark scalability_benchmark.cpp)
