/**
 * @file tcp_client.cpp
 * @brief Advanced TCP client for AI/ML data streaming and testing
 * 
 * This example demonstrates:
 * - High-throughput tensor streaming to servers
 * - Latency testing and benchmarking
 * - Realistic AI/ML data patterns
 * - Connection resilience and error handling
 * - Performance measurement and analysis
 */

#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <thread>
#include <vector>
#include <cmath>

using namespace psyne;

class TensorGenerator {
private:
    std::mt19937 gen_;
    std::normal_distribution<float> dist_;

public:
    TensorGenerator() : gen_(std::random_device{}()), dist_(0.0f, 1.0f) {}

    void generate_synthetic_data(FloatVector& tensor, size_t size, const std::string& pattern) {
        tensor.resize(size);
        
        if (pattern == "gaussian") {
            // Generate Gaussian distributed data (common in ML)
            for (size_t i = 0; i < size; ++i) {
                tensor[i] = dist_(gen_);
            }
        } else if (pattern == "image") {
            // Simulate image-like data with spatial correlation
            size_t width = static_cast<size_t>(std::sqrt(size));
            for (size_t i = 0; i < size; ++i) {
                size_t x = i % width;
                size_t y = i / width;
                // Create smooth gradient pattern like image data
                tensor[i] = std::sin(x * 0.1f) * std::cos(y * 0.1f) * 0.5f + 0.5f;
            }
        } else if (pattern == "sequence") {
            // Time series / sequence data
            for (size_t i = 0; i < size; ++i) {
                tensor[i] = std::sin(i * 0.01f) + 0.1f * dist_(gen_);
            }
        } else if (pattern == "sparse") {
            // Sparse data (common in NLP embeddings)
            std::fill(tensor.begin(), tensor.end(), 0.0f);
            size_t num_nonzero = size / 10;  // 10% non-zero
            for (size_t i = 0; i < num_nonzero; ++i) {
                size_t idx = gen_() % size;
                tensor[idx] = dist_(gen_);
            }
        }
    }
};

void run_ai_client() {
    std::cout << "🧠 AI/ML Data Streaming Client\n";
    std::cout << "==============================\n";
    std::cout << "Streaming synthetic tensors to AI inference server\n\n";

    try {
        auto channel = Channel::create("tcp://localhost:8080",
                                     32 * 1024 * 1024, // 32MB buffer
                                     ChannelMode::SPSC, 
                                     ChannelType::SingleType);

        std::cout << "✓ Connected to AI server at localhost:8080\n";
        std::cout << "✓ Ready for tensor streaming\n\n";

        TensorGenerator generator;
        
        // Different tensor sizes and patterns to test
        std::vector<std::pair<size_t, std::string>> test_cases = {
            {1000, "gaussian"},      // Small dense tensor
            {10000, "image"},        // Medium image-like data
            {50000, "sequence"},     // Large sequence data
            {100000, "sparse"},      // Large sparse tensor
            {224*224*3, "image"},    // Typical CNN input (224x224 RGB)
        };

        // Start result receiver thread
        std::atomic<int> results_received{0};
        std::vector<std::chrono::microseconds> processing_times;
        std::mutex times_mutex;
        
        std::thread receiver([&]() {
            while (results_received < static_cast<int>(test_cases.size() * 2)) {  // 2 rounds
                auto result = channel->receive<FloatVector>(std::chrono::milliseconds(5000));
                if (result) {
                    results_received++;
                    std::cout << "📥 Received processed tensor: " << result->size() 
                             << " elements\n";
                    
                    // Show some statistics of processed data
                    if (result->size() > 0) {
                        float sum = 0.0f;
                        for (size_t i = 0; i < result->size(); ++i) {
                            sum += (*result)[i];
                        }
                        float mean = sum / result->size();
                        std::cout << "   Result mean: " << mean << "\n";
                    }
                }
            }
        });

        auto overall_start = std::chrono::high_resolution_clock::now();
        
        // Run test cases twice
        for (int round = 0; round < 2; ++round) {
            std::cout << "\n🚀 Round " << (round + 1) << " - Streaming different tensor patterns:\n";
            
            for (const auto& [size, pattern] : test_cases) {
                std::cout << "\n📤 Generating " << pattern << " tensor (" << size 
                         << " elements, " << (size * sizeof(float) / 1024) << " KB)...\n";
                
                // Create tensor directly in ring buffer (zero-copy)
                FloatVector tensor(channel);
                generator.generate_synthetic_data(tensor, size, pattern);
                
                auto send_start = std::chrono::high_resolution_clock::now();
                tensor.send();
                auto send_end = std::chrono::high_resolution_clock::now();
                
                auto send_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    send_end - send_start);
                
                {
                    std::lock_guard<std::mutex> lock(times_mutex);
                    processing_times.push_back(send_time);
                }
                
                std::cout << "   ✓ Sent in " << send_time.count() << " μs\n";
                
                // Brief pause between tensors
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }

        std::cout << "\n⏳ Waiting for all results...\n";
        receiver.join();
        
        auto overall_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            overall_end - overall_start);

        // Calculate statistics
        auto total_send_time = std::accumulate(processing_times.begin(), processing_times.end(),
                                             std::chrono::microseconds{0});
        
        std::cout << "\n📊 AI Client Performance Summary:\n";
        std::cout << "   Tensors sent: " << processing_times.size() << "\n";
        std::cout << "   Results received: " << results_received.load() << "\n";
        std::cout << "   Total time: " << total_duration.count() << " ms\n";
        std::cout << "   Average send time: " << (total_send_time.count() / processing_times.size()) << " μs\n";
        std::cout << "   Throughput: " << (processing_times.size() * 1000.0 / total_duration.count()) << " tensors/s\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ AI client error: " << e.what() << std::endl;
    }
}

void run_latency_test() {
    std::cout << "⚡ TCP Latency Benchmark\n";
    std::cout << "=======================\n";
    std::cout << "Measuring round-trip latency with echo server\n\n";

    try {
        auto channel = Channel::create("tcp://localhost:8080",
                                     4 * 1024 * 1024, // 4MB buffer
                                     ChannelMode::SPSC, 
                                     ChannelType::SingleType);

        std::cout << "✓ Connected to echo server at localhost:8080\n\n";

        const int num_tests = 100;
        const std::vector<size_t> message_sizes = {100, 1000, 10000, 100000};
        
        for (size_t msg_size : message_sizes) {
            std::cout << "🔬 Testing " << msg_size << " element messages (" 
                     << (msg_size * sizeof(float) / 1024) << " KB):\n";
            
            std::vector<std::chrono::microseconds> latencies;
            latencies.reserve(num_tests);
            
            for (int i = 0; i < num_tests; ++i) {
                // Create test message
                FloatVector test_msg(channel);
                test_msg.resize(msg_size);
                
                // Fill with pattern for verification
                for (size_t j = 0; j < msg_size; ++j) {
                    test_msg[j] = static_cast<float>(i * 1000 + j);
                }
                
                auto start_time = std::chrono::high_resolution_clock::now();
                test_msg.send();
                
                // Wait for echo response
                auto echo = channel->receive<FloatVector>(std::chrono::milliseconds(1000));
                auto end_time = std::chrono::high_resolution_clock::now();
                
                if (echo && echo->size() == msg_size) {
                    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time);
                    latencies.push_back(latency);
                    
                    // Verify echo integrity
                    bool valid = true;
                    for (size_t j = 0; j < msg_size && j < 10; ++j) {
                        if (std::abs((*echo)[j] - test_msg[j]) > 1e-6f) {
                            valid = false;
                            break;
                        }
                    }
                    
                    if (!valid) {
                        std::cout << "   ⚠️ Echo verification failed for message " << i << "\n";
                    }
                } else {
                    std::cout << "   ❌ No echo received for message " << i << "\n";
                }
                
                // Progress indicator
                if ((i + 1) % 20 == 0) {
                    std::cout << "   Progress: " << (i + 1) << " / " << num_tests << "\n";
                }
            }
            
            if (!latencies.empty()) {
                // Calculate statistics
                std::sort(latencies.begin(), latencies.end());
                
                auto min_latency = latencies.front();
                auto max_latency = latencies.back();
                auto median_latency = latencies[latencies.size() / 2];
                auto p95_latency = latencies[static_cast<size_t>(latencies.size() * 0.95)];
                
                auto total_latency = std::accumulate(latencies.begin(), latencies.end(),
                                                   std::chrono::microseconds{0});
                auto avg_latency = total_latency / latencies.size();
                
                std::cout << "   📊 Results for " << msg_size << " elements:\n";
                std::cout << "      Min latency:    " << min_latency.count() << " μs\n";
                std::cout << "      Average latency:" << avg_latency.count() << " μs\n";
                std::cout << "      Median latency: " << median_latency.count() << " μs\n";
                std::cout << "      95th percentile:" << p95_latency.count() << " μs\n";
                std::cout << "      Max latency:    " << max_latency.count() << " μs\n";
                std::cout << "      Success rate:   " << (latencies.size() * 100 / num_tests) << "%\n\n";
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "❌ Latency test error: " << e.what() << std::endl;
    }
}

void run_throughput_test() {
    std::cout << "🚀 TCP Throughput Benchmark\n";
    std::cout << "===========================\n";
    std::cout << "Maximum throughput stress test\n\n";

    try {
        auto channel = Channel::create("tcp://localhost:8080",
                                     64 * 1024 * 1024, // 64MB buffer for maximum throughput
                                     ChannelMode::SPSC, 
                                     ChannelType::SingleType);

        std::cout << "✓ Connected for throughput testing\n";
        std::cout << "✓ Using 64MB ring buffer\n\n";

        const size_t tensor_size = 256 * 256;  // 256K floats = 1MB per tensor
        const int num_tensors = 100;
        
        std::cout << "📊 Test parameters:\n";
        std::cout << "   Tensor size: " << tensor_size << " floats (" 
                 << (tensor_size * sizeof(float) / 1024 / 1024) << " MB each)\n";
        std::cout << "   Number of tensors: " << num_tensors << "\n";
        std::cout << "   Total data: " << (num_tensors * tensor_size * sizeof(float) / 1024 / 1024) 
                 << " MB\n\n";

        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_tensors; ++i) {
            FloatVector tensor(channel);
            tensor.resize(tensor_size);
            
            // Fill with computational pattern
            for (size_t j = 0; j < tensor_size; ++j) {
                tensor[j] = std::sin(static_cast<float>(i * tensor_size + j) * 0.0001f);
            }
            
            tensor.send();
            
            if ((i + 1) % 10 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
                double current_throughput = ((i + 1) * tensor_size * sizeof(float) * 1000.0) / 
                                          (elapsed.count() * 1024 * 1024);
                
                std::cout << "📤 Sent " << (i + 1) << " tensors (current: " 
                         << current_throughput << " MB/s)\n";
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        size_t total_bytes = num_tensors * tensor_size * sizeof(float);
        double throughput_mbps = (total_bytes * 1000000.0) / (duration.count() * 1024 * 1024);
        double tensor_rate = (num_tensors * 1000000.0) / duration.count();
        
        std::cout << "\n🏆 Throughput Test Results:\n";
        std::cout << "   Total data sent: " << (total_bytes / 1024 / 1024) << " MB\n";
        std::cout << "   Duration: " << duration.count() << " μs\n";
        std::cout << "   Peak throughput: " << throughput_mbps << " MB/s\n";
        std::cout << "   Tensor rate: " << tensor_rate << " tensors/s\n";
        std::cout << "   Average latency per tensor: " << (duration.count() / num_tensors) << " μs\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ Throughput test error: " << e.what() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::cout << "Psyne TCP Client Examples\n";
    std::cout << "=========================\n\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [ai|latency|throughput]\n\n";
        std::cout << "Client modes:\n";
        std::cout << "  ai         # AI/ML tensor streaming client\n";
        std::cout << "  latency    # Round-trip latency measurement\n";
        std::cout << "  throughput # Maximum throughput stress test\n\n";
        std::cout << "Features:\n";
        std::cout << "  ✓ Zero-copy tensor generation and streaming\n";
        std::cout << "  ✓ Realistic AI/ML data patterns\n";
        std::cout << "  ✓ Comprehensive performance analysis\n";
        std::cout << "  ✓ Connection resilience testing\n";
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "ai") {
        run_ai_client();
    } else if (mode == "latency") {
        run_latency_test();
    } else if (mode == "throughput") {
        run_throughput_test();
    } else {
        std::cerr << "❌ Invalid mode. Use 'ai', 'latency', or 'throughput'\n";
        return 1;
    }

    return 0;
}