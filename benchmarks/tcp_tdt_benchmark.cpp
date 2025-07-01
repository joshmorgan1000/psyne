/**
 * @file tcp_tdt_benchmark.cpp
 * @brief TCP Server/Client benchmark with TDT compression for cross-machine
 * testing
 *
 * Usage:
 *   Server: ./tcp_tdt_benchmark server [port]
 *   Client: ./tcp_tdt_benchmark client <server_ip> [port]
 *
 * Tests tensor data compression over TCP networks between different machines.
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

// Network includes
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace std::chrono;

// Simple TDT compression implementation
class SimpleTDT {
public:
    struct CompressionStats {
        size_t original_size = 0;
        size_t compressed_size = 0;
        double compression_ratio = 0.0;
        double compression_time_us = 0.0;
        double decompression_time_us = 0.0;
    };

    // Simple float compression using quantization and run-length encoding
    static std::vector<uint8_t> compress(const float *data, size_t count,
                                         CompressionStats &stats) {
        auto start = high_resolution_clock::now();

        std::vector<uint8_t> compressed;
        compressed.reserve(count * 2); // Conservative estimate

        // Simple quantization to 16-bit
        uint16_t prev = 0;
        size_t run_length = 0;

        for (size_t i = 0; i < count; ++i) {
            // Quantize float to 16-bit
            uint16_t quantized = static_cast<uint16_t>(
                std::clamp(data[i] * 32767.0f + 32768.0f, 0.0f, 65535.0f));

            if (quantized == prev && run_length < 255) {
                run_length++;
            } else {
                // Emit previous run
                if (run_length > 0) {
                    compressed.push_back(0xFF); // RLE marker
                    compressed.push_back(static_cast<uint8_t>(run_length));
                    compressed.push_back(prev & 0xFF);
                    compressed.push_back((prev >> 8) & 0xFF);
                }

                // Start new run
                prev = quantized;
                run_length = 1;
            }
        }

        // Emit final run
        if (run_length > 0) {
            compressed.push_back(0xFF);
            compressed.push_back(static_cast<uint8_t>(run_length));
            compressed.push_back(prev & 0xFF);
            compressed.push_back((prev >> 8) & 0xFF);
        }

        auto end = high_resolution_clock::now();

        stats.original_size = count * sizeof(float);
        stats.compressed_size = compressed.size();
        stats.compression_ratio =
            static_cast<double>(stats.original_size) / stats.compressed_size;
        stats.compression_time_us =
            duration_cast<microseconds>(end - start).count();

        return compressed;
    }

    static std::vector<float> decompress(const uint8_t *data, size_t size,
                                         CompressionStats &stats) {
        auto start = high_resolution_clock::now();

        std::vector<float> result;
        result.reserve(size * 2); // Conservative estimate

        for (size_t i = 0; i < size;) {
            if (data[i] == 0xFF && i + 3 < size) {
                // RLE block
                uint8_t run_length = data[i + 1];
                uint16_t value =
                    data[i + 2] | (static_cast<uint16_t>(data[i + 3]) << 8);

                // Dequantize and add to result
                float float_val =
                    (static_cast<float>(value) - 32768.0f) / 32767.0f;
                for (int j = 0; j < run_length; ++j) {
                    result.push_back(float_val);
                }

                i += 4;
            } else {
                // This shouldn't happen with our simple format
                i++;
            }
        }

        auto end = high_resolution_clock::now();
        stats.decompression_time_us =
            duration_cast<microseconds>(end - start).count();

        return result;
    }
};

// Tensor data generator
class TensorGenerator {
public:
    enum DataType { RANDOM, GRADIENTS, ACTIVATIONS, WEIGHTS };

    static std::vector<float> generate(size_t count, DataType type) {
        std::vector<float> data(count);
        std::random_device rd;
        std::mt19937 gen(rd());

        switch (type) {
        case RANDOM: {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &val : data)
                val = dist(gen);
        } break;

        case GRADIENTS: {
            // Gradients tend to be small and sparse
            std::normal_distribution<float> dist(0.0f, 0.001f);
            for (auto &val : data) {
                val = dist(gen);
                if (std::abs(val) < 0.0001f)
                    val = 0.0f; // Sparsity
            }
        } break;

        case ACTIVATIONS: {
            // ReLU-like activations (many zeros, positive values)
            std::exponential_distribution<float> dist(2.0f);
            std::bernoulli_distribution zero_dist(0.3); // 30% zeros
            for (auto &val : data) {
                val = zero_dist(gen) ? 0.0f : dist(gen);
            }
        } break;

        case WEIGHTS: {
            // Xavier initialization
            std::normal_distribution<float> dist(0.0f, 0.1f);
            for (auto &val : data)
                val = dist(gen);
        } break;
        }

        return data;
    }

    static const char *type_name(DataType type) {
        switch (type) {
        case RANDOM:
            return "Random";
        case GRADIENTS:
            return "Gradients";
        case ACTIVATIONS:
            return "Activations";
        case WEIGHTS:
            return "Weights";
        default:
            return "Unknown";
        }
    }
};

// Network utilities
class NetworkUtils {
public:
    static int create_server_socket(int port) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            perror("socket");
            return -1;
        }

        int opt = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("bind");
            close(sock);
            return -1;
        }

        if (listen(sock, 5) < 0) {
            perror("listen");
            close(sock);
            return -1;
        }

        return sock;
    }

    static int connect_to_server(const char *host, int port) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            perror("socket");
            return -1;
        }

        struct hostent *server = gethostbyname(host);
        if (!server) {
            std::cerr << "Error: Cannot resolve host " << host << std::endl;
            close(sock);
            return -1;
        }

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
        addr.sin_port = htons(port);

        if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            perror("connect");
            close(sock);
            return -1;
        }

        return sock;
    }

    static bool send_all(int sock, const void *data, size_t size) {
        const uint8_t *ptr = static_cast<const uint8_t *>(data);
        size_t sent = 0;

        while (sent < size) {
            ssize_t result = send(sock, ptr + sent, size - sent, 0);
            if (result <= 0) {
                perror("send");
                return false;
            }
            sent += result;
        }
        return true;
    }

    static bool recv_all(int sock, void *data, size_t size) {
        uint8_t *ptr = static_cast<uint8_t *>(data);
        size_t received = 0;

        while (received < size) {
            ssize_t result = recv(sock, ptr + received, size - received, 0);
            if (result <= 0) {
                perror("recv");
                return false;
            }
            received += result;
        }
        return true;
    }
};

// Benchmark configuration
struct BenchmarkConfig {
    size_t tensor_count = 1000;
    size_t tensor_size = 1024 * 1024; // 1M floats = 4MB
    TensorGenerator::DataType data_type = TensorGenerator::GRADIENTS;
    bool verbose = true;
};

// Server implementation
void run_server(int port, const BenchmarkConfig &config) {
    std::cout << "🚀 TDT TCP Server starting on port " << port << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Tensor count: " << config.tensor_count << std::endl;
    std::cout << "  Tensor size:  " << config.tensor_size << " floats ("
              << (config.tensor_size * sizeof(float) / 1024 / 1024) << " MB)"
              << std::endl;
    std::cout << "  Data type:    "
              << TensorGenerator::type_name(config.data_type) << std::endl;
    std::cout << std::endl;

    int server_sock = NetworkUtils::create_server_socket(port);
    if (server_sock < 0) {
        std::cerr << "Failed to create server socket" << std::endl;
        return;
    }

    std::cout << "Waiting for client connection..." << std::endl;

    struct sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    int client_sock =
        accept(server_sock, (struct sockaddr *)&client_addr, &client_len);

    if (client_sock < 0) {
        perror("accept");
        close(server_sock);
        return;
    }

    std::cout << "Client connected from " << inet_ntoa(client_addr.sin_addr)
              << std::endl;

    // Send benchmark configuration
    if (!NetworkUtils::send_all(client_sock, &config, sizeof(config))) {
        std::cerr << "Failed to send config" << std::endl;
        close(client_sock);
        close(server_sock);
        return;
    }

    // Generate and send tensors
    std::vector<SimpleTDT::CompressionStats> stats(config.tensor_count);
    auto total_start = high_resolution_clock::now();

    for (size_t i = 0; i < config.tensor_count; ++i) {
        // Generate tensor data
        auto tensor =
            TensorGenerator::generate(config.tensor_size, config.data_type);

        // Compress
        auto compressed =
            SimpleTDT::compress(tensor.data(), tensor.size(), stats[i]);

        // Send compressed size first
        uint32_t compressed_size = compressed.size();
        if (!NetworkUtils::send_all(client_sock, &compressed_size,
                                    sizeof(compressed_size))) {
            std::cerr << "Failed to send compressed size" << std::endl;
            break;
        }

        // Send compressed data
        if (!NetworkUtils::send_all(client_sock, compressed.data(),
                                    compressed.size())) {
            std::cerr << "Failed to send compressed data" << std::endl;
            break;
        }

        if (config.verbose && (i + 1) % 100 == 0) {
            std::cout << "Sent " << (i + 1) << "/" << config.tensor_count
                      << " tensors (compression: " << std::fixed
                      << std::setprecision(2) << stats[i].compression_ratio
                      << "x)" << std::endl;
        }
    }

    auto total_end = high_resolution_clock::now();

    // Calculate statistics
    double total_time_s =
        duration_cast<microseconds>(total_end - total_start).count() / 1e6;
    size_t total_original = std::accumulate(
        stats.begin(), stats.end(), 0UL,
        [](size_t sum, const auto &s) { return sum + s.original_size; });
    size_t total_compressed = std::accumulate(
        stats.begin(), stats.end(), 0UL,
        [](size_t sum, const auto &s) { return sum + s.compressed_size; });

    std::cout << std::endl << "📊 Server Results:" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Total time:           " << std::fixed << std::setprecision(3)
              << total_time_s << " seconds" << std::endl;
    std::cout << "Tensors sent:         " << config.tensor_count << std::endl;
    std::cout << "Original data:        " << (total_original / 1024 / 1024)
              << " MB" << std::endl;
    std::cout << "Compressed data:      " << (total_compressed / 1024 / 1024)
              << " MB" << std::endl;
    std::cout << "Compression ratio:    " << std::fixed << std::setprecision(2)
              << (static_cast<double>(total_original) / total_compressed) << "x"
              << std::endl;
    std::cout << "Network throughput:   " << std::fixed << std::setprecision(2)
              << (total_compressed / 1024.0 / 1024.0 / total_time_s)
              << " MB/s (compressed)" << std::endl;
    std::cout << "Effective throughput: " << std::fixed << std::setprecision(2)
              << (total_original / 1024.0 / 1024.0 / total_time_s)
              << " MB/s (original)" << std::endl;

    close(client_sock);
    close(server_sock);
}

// Client implementation
void run_client(const char *host, int port) {
    std::cout << "🔌 TDT TCP Client connecting to " << host << ":" << port
              << std::endl;

    int sock = NetworkUtils::connect_to_server(host, port);
    if (sock < 0) {
        std::cerr << "Failed to connect to server" << std::endl;
        return;
    }

    std::cout << "Connected to server!" << std::endl;

    // Receive benchmark configuration
    BenchmarkConfig config;
    if (!NetworkUtils::recv_all(sock, &config, sizeof(config))) {
        std::cerr << "Failed to receive config" << std::endl;
        close(sock);
        return;
    }

    std::cout << "Benchmark configuration:" << std::endl;
    std::cout << "  Tensor count: " << config.tensor_count << std::endl;
    std::cout << "  Tensor size:  " << config.tensor_size << " floats"
              << std::endl;
    std::cout << "  Data type:    "
              << TensorGenerator::type_name(config.data_type) << std::endl;
    std::cout << std::endl;

    // Receive and decompress tensors
    std::vector<SimpleTDT::CompressionStats> stats(config.tensor_count);
    auto total_start = high_resolution_clock::now();

    for (size_t i = 0; i < config.tensor_count; ++i) {
        // Receive compressed size
        uint32_t compressed_size;
        if (!NetworkUtils::recv_all(sock, &compressed_size,
                                    sizeof(compressed_size))) {
            std::cerr << "Failed to receive compressed size" << std::endl;
            break;
        }

        // Receive compressed data
        std::vector<uint8_t> compressed(compressed_size);
        if (!NetworkUtils::recv_all(sock, compressed.data(), compressed_size)) {
            std::cerr << "Failed to receive compressed data" << std::endl;
            break;
        }

        // Decompress
        auto decompressed = SimpleTDT::decompress(compressed.data(),
                                                  compressed.size(), stats[i]);

        stats[i].original_size = decompressed.size() * sizeof(float);
        stats[i].compressed_size = compressed_size;
        stats[i].compression_ratio =
            static_cast<double>(stats[i].original_size) /
            stats[i].compressed_size;

        if (config.verbose && (i + 1) % 100 == 0) {
            std::cout << "Received " << (i + 1) << "/" << config.tensor_count
                      << " tensors (decompressed to " << decompressed.size()
                      << " floats)" << std::endl;
        }
    }

    auto total_end = high_resolution_clock::now();

    // Calculate statistics
    double total_time_s =
        duration_cast<microseconds>(total_end - total_start).count() / 1e6;
    size_t total_original = std::accumulate(
        stats.begin(), stats.end(), 0UL,
        [](size_t sum, const auto &s) { return sum + s.original_size; });
    size_t total_compressed = std::accumulate(
        stats.begin(), stats.end(), 0UL,
        [](size_t sum, const auto &s) { return sum + s.compressed_size; });

    std::cout << std::endl << "📈 Client Results:" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Total time:           " << std::fixed << std::setprecision(3)
              << total_time_s << " seconds" << std::endl;
    std::cout << "Tensors received:     " << config.tensor_count << std::endl;
    std::cout << "Decompressed data:    " << (total_original / 1024 / 1024)
              << " MB" << std::endl;
    std::cout << "Network data:         " << (total_compressed / 1024 / 1024)
              << " MB" << std::endl;
    std::cout << "Compression ratio:    " << std::fixed << std::setprecision(2)
              << (static_cast<double>(total_original) / total_compressed) << "x"
              << std::endl;
    std::cout << "Receive throughput:   " << std::fixed << std::setprecision(2)
              << (total_compressed / 1024.0 / 1024.0 / total_time_s)
              << " MB/s (compressed)" << std::endl;
    std::cout << "Effective throughput: " << std::fixed << std::setprecision(2)
              << (total_original / 1024.0 / 1024.0 / total_time_s)
              << " MB/s (original)" << std::endl;

    // Calculate average decompression times
    double avg_decomp_time =
        std::accumulate(stats.begin(), stats.end(), 0.0,
                        [](double sum, const auto &s) {
                            return sum + s.decompression_time_us;
                        }) /
        stats.size();

    std::cout << "Avg decompression:    " << std::fixed << std::setprecision(2)
              << avg_decomp_time << " μs per tensor" << std::endl;

    close(sock);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  Server: " << argv[0] << " server [port]" << std::endl;
        std::cout << "  Client: " << argv[0] << " client <server_ip> [port]"
                  << std::endl;
        std::cout << std::endl;
        std::cout << "Example:" << std::endl;
        std::cout << "  On Linux:  " << argv[0] << " server 8080" << std::endl;
        std::cout << "  On Mac:    " << argv[0] << " client 192.168.1.100 8080"
                  << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    int port = 8080; // Default port

    if (mode == "server") {
        if (argc > 2)
            port = std::atoi(argv[2]);

        BenchmarkConfig config;
        config.tensor_count = 1000;
        config.tensor_size = 256 * 1024; // 256K floats = 1MB
        config.data_type = TensorGenerator::GRADIENTS;

        run_server(port, config);

    } else if (mode == "client") {
        if (argc < 3) {
            std::cerr << "Error: Client mode requires server IP address"
                      << std::endl;
            return 1;
        }

        const char *host = argv[2];
        if (argc > 3)
            port = std::atoi(argv[3]);

        run_client(host, port);

    } else {
        std::cerr << "Error: Mode must be 'server' or 'client'" << std::endl;
        return 1;
    }

    return 0;
}