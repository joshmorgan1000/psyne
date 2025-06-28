#include <gtest/gtest.h>
#include <psyne/psyne.hpp>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>

/**
 * @brief Comprehensive comparison of Psyne against other IPC mechanisms
 * 
 * This benchmark compares Psyne's performance against:
 * - Unix domain sockets
 * - TCP localhost sockets  
 * - Shared memory (raw)
 * - Named pipes (FIFOs)
 */
class IPCComparisonBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for comparison tests
        results_.clear();
    }
    
    void TearDown() override {
        // Generate comparison report
        generate_comparison_report();
    }
    
    struct BenchmarkResult {
        std::string mechanism;
        double latency_us;        // Average latency in microseconds
        double throughput_mbps;   // Throughput in MB/s
        double message_rate;      // Messages per second
        size_t message_size;      // Message size tested
        std::string notes;        // Additional information
    };
    
    std::vector<BenchmarkResult> results_;
    
    void add_result(const std::string& mechanism, double latency_us, 
                   double throughput_mbps, double message_rate, 
                   size_t message_size, const std::string& notes = "") {
        results_.push_back({mechanism, latency_us, throughput_mbps, 
                           message_rate, message_size, notes});
    }
    
    void generate_comparison_report() {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "            IPC MECHANISM PERFORMANCE COMPARISON\n";
        std::cout << std::string(80, '=') << "\n\n";
        
        std::cout << std::left << std::setw(20) << "Mechanism"
                  << std::setw(12) << "Latency(μs)"
                  << std::setw(15) << "Throughput(MB/s)"
                  << std::setw(15) << "Msg Rate(K/s)"
                  << std::setw(12) << "Msg Size"
                  << "Notes\n";
        std::cout << std::string(80, '-') << "\n";
        
        for (const auto& result : results_) {
            std::cout << std::left << std::setw(20) << result.mechanism
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.latency_us
                      << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput_mbps
                      << std::setw(15) << std::fixed << std::setprecision(1) << (result.message_rate / 1000.0)
                      << std::setw(12) << result.message_size
                      << result.notes << "\n";
        }
        
        // Find Psyne results for comparison
        auto psyne_memory = std::find_if(results_.begin(), results_.end(),
            [](const BenchmarkResult& r) { return r.mechanism == "Psyne (Memory)"; });
        
        if (psyne_memory != results_.end()) {
            std::cout << "\n" << std::string(50, '=') << "\n";
            std::cout << "PSYNE PERFORMANCE ADVANTAGES:\n";
            std::cout << std::string(50, '=') << "\n";
            
            for (const auto& result : results_) {
                if (result.mechanism.find("Psyne") == std::string::npos) {
                    double latency_advantage = result.latency_us / psyne_memory->latency_us;
                    double throughput_advantage = psyne_memory->throughput_mbps / result.throughput_mbps;
                    
                    std::cout << "vs " << result.mechanism << ":\n";
                    std::cout << "  - " << std::fixed << std::setprecision(1) 
                              << latency_advantage << "x faster latency\n";
                    std::cout << "  - " << std::fixed << std::setprecision(1) 
                              << throughput_advantage << "x higher throughput\n\n";
                }
            }
        }
    }
    
    // Helper to measure execution time
    template<typename Func>
    double measure_time_us(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count();
    }
};

/**
 * @brief Benchmark Psyne memory channels
 */
TEST_F(IPCComparisonBenchmark, PsyneMemoryChannel) {
    const size_t message_size = 1024;  // 1KB messages
    const int num_messages = 10000;
    
    auto channel = psyne::Channel::create("memory://benchmark", 1024 * 1024);
    
    auto latency = measure_time_us([&]() {
        psyne::ByteVector msg(*channel);
        msg.resize(message_size);
        msg.send();
        
        auto received = channel->receive_single<psyne::ByteVector>();
        EXPECT_TRUE(received.has_value());
    });
    
    // Throughput test
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_messages; ++i) {
        psyne::ByteVector msg(*channel);
        msg.resize(message_size);
        msg.send();
        
        auto received = channel->receive_single<psyne::ByteVector>();
        EXPECT_TRUE(received.has_value());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    
    double throughput_mbps = (num_messages * message_size) / (duration * 1024 * 1024);
    double message_rate = num_messages / duration;
    
    add_result("Psyne (Memory)", latency, throughput_mbps, message_rate, 
               message_size, "Zero-copy, lock-free");
}

/**
 * @brief Benchmark Psyne IPC channels
 */
TEST_F(IPCComparisonBenchmark, PsyneIPCChannel) {
    const size_t message_size = 1024;
    const int num_messages = 10000;
    
    try {
        auto channel = psyne::Channel::create("ipc://benchmark_ipc", 1024 * 1024);
        
        auto latency = measure_time_us([&]() {
            psyne::ByteVector msg(*channel);
            msg.resize(message_size);
            msg.send();
            
            auto received = channel->receive_single<psyne::ByteVector>();
            EXPECT_TRUE(received.has_value());
        });
        
        // Throughput test
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_messages; ++i) {
            psyne::ByteVector msg(*channel);
            msg.resize(message_size);
            msg.send();
            
            auto received = channel->receive_single<psyne::ByteVector>();
            EXPECT_TRUE(received.has_value());
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start).count();
        
        double throughput_mbps = (num_messages * message_size) / (duration * 1024 * 1024);
        double message_rate = num_messages / duration;
        
        add_result("Psyne (IPC)", latency, throughput_mbps, message_rate, 
                   message_size, "Shared memory");
    } catch (const std::exception& e) {
        add_result("Psyne (IPC)", 0, 0, 0, message_size, "Not available");
    }
}

/**
 * @brief Benchmark Unix domain sockets
 */
TEST_F(IPCComparisonBenchmark, UnixDomainSockets) {
    const size_t message_size = 1024;
    const int num_messages = 1000;  // Fewer messages due to overhead
    
    // Create socket pair
    int sock_fds[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sock_fds) == -1) {
        add_result("Unix Sockets", 0, 0, 0, message_size, "Creation failed");
        return;
    }
    
    std::vector<uint8_t> test_data(message_size, 0x42);
    std::vector<uint8_t> recv_buffer(message_size);
    
    auto latency = measure_time_us([&]() {
        send(sock_fds[0], test_data.data(), message_size, 0);
        recv(sock_fds[1], recv_buffer.data(), message_size, 0);
    });
    
    // Throughput test
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_messages; ++i) {
        send(sock_fds[0], test_data.data(), message_size, 0);
        recv(sock_fds[1], recv_buffer.data(), message_size, 0);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    
    double throughput_mbps = (num_messages * message_size) / (duration * 1024 * 1024);
    double message_rate = num_messages / duration;
    
    close(sock_fds[0]);
    close(sock_fds[1]);
    
    add_result("Unix Sockets", latency, throughput_mbps, message_rate, 
               message_size, "Kernel copy overhead");
}

/**
 * @brief Benchmark TCP localhost sockets
 */
TEST_F(IPCComparisonBenchmark, TCPLocalhost) {
    const size_t message_size = 1024;
    const int num_messages = 500;  // Even fewer due to TCP overhead
    
    // Create server socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        add_result("TCP Localhost", 0, 0, 0, message_size, "Socket creation failed");
        return;
    }
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(0);  // Let OS choose port
    
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        close(server_fd);
        add_result("TCP Localhost", 0, 0, 0, message_size, "Bind failed");
        return;
    }
    
    // Get assigned port
    socklen_t addr_len = sizeof(addr);
    getsockname(server_fd, (struct sockaddr*)&addr, &addr_len);
    int port = ntohs(addr.sin_port);
    
    listen(server_fd, 1);
    
    // Connect in separate thread to avoid blocking
    std::thread server_thread([&]() {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd == -1) return;
        
        std::vector<uint8_t> recv_buffer(message_size);
        for (int i = 0; i < num_messages + 1; ++i) {  // +1 for latency test
            recv(client_fd, recv_buffer.data(), message_size, MSG_WAITALL);
            send(client_fd, recv_buffer.data(), message_size, 0);  // Echo back
        }
        
        close(client_fd);
    });
    
    // Client side
    int client_fd = socket(AF_INET, SOCK_STREAM, 0);
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    
    if (connect(client_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        std::vector<uint8_t> test_data(message_size, 0x42);
        std::vector<uint8_t> recv_buffer(message_size);
        
        // Latency test
        auto latency = measure_time_us([&]() {
            send(client_fd, test_data.data(), message_size, 0);
            recv(client_fd, recv_buffer.data(), message_size, MSG_WAITALL);
        });
        
        // Throughput test
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_messages; ++i) {
            send(client_fd, test_data.data(), message_size, 0);
            recv(client_fd, recv_buffer.data(), message_size, MSG_WAITALL);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start).count();
        
        double throughput_mbps = (num_messages * message_size) / (duration * 1024 * 1024);
        double message_rate = num_messages / duration;
        
        add_result("TCP Localhost", latency, throughput_mbps, message_rate, 
                   message_size, "Network stack overhead");
        
        close(client_fd);
    } else {
        add_result("TCP Localhost", 0, 0, 0, message_size, "Connection failed");
    }
    
    server_thread.join();
    close(server_fd);
}

/**
 * @brief Benchmark named pipes (FIFOs)
 */
TEST_F(IPCComparisonBenchmark, NamedPipes) {
    const size_t message_size = 1024;
    const int num_messages = 1000;
    
    const char* fifo_path = "/tmp/psyne_benchmark_fifo";
    
    // Clean up any existing FIFO
    unlink(fifo_path);
    
    if (mkfifo(fifo_path, 0666) == -1) {
        add_result("Named Pipes", 0, 0, 0, message_size, "FIFO creation failed");
        return;
    }
    
    std::thread writer_thread([&]() {
        int write_fd = open(fifo_path, O_WRONLY);
        if (write_fd == -1) return;
        
        std::vector<uint8_t> test_data(message_size, 0x42);
        for (int i = 0; i < num_messages + 1; ++i) {  // +1 for latency test
            write(write_fd, test_data.data(), message_size);
        }
        
        close(write_fd);
    });
    
    int read_fd = open(fifo_path, O_RDONLY);
    if (read_fd != -1) {
        std::vector<uint8_t> recv_buffer(message_size);
        
        // Latency test
        auto latency = measure_time_us([&]() {
            read(read_fd, recv_buffer.data(), message_size);
        });
        
        // Throughput test
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_messages; ++i) {
            read(read_fd, recv_buffer.data(), message_size);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start).count();
        
        double throughput_mbps = (num_messages * message_size) / (duration * 1024 * 1024);
        double message_rate = num_messages / duration;
        
        add_result("Named Pipes", latency, throughput_mbps, message_rate, 
                   message_size, "File system overhead");
        
        close(read_fd);
    } else {
        add_result("Named Pipes", 0, 0, 0, message_size, "Open failed");
    }
    
    writer_thread.join();
    unlink(fifo_path);
}

/**
 * @brief Summary test that runs all benchmarks
 */
TEST_F(IPCComparisonBenchmark, ComprehensiveComparison) {
    std::cout << "\nRunning comprehensive IPC comparison...\n";
    
    // Run all individual benchmarks
    PsyneMemoryChannel();
    PsyneIPCChannel();
    UnixDomainSockets();
    TCPLocalhost();
    NamedPipes();
    
    std::cout << "\nComparison complete! See detailed report above.\n";
}