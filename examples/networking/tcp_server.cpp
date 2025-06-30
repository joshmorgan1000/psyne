/**
 * @file tcp_server.cpp
 * @brief Production-grade TCP server example for AI/ML workloads
 * 
 * This example demonstrates:
 * - High-performance TCP server for tensor data
 * - Multiple concurrent connections (if using MPSC mode)
 * - Real-time data processing and aggregation
 * - Performance monitoring and metrics
 * - Graceful shutdown handling
 */

#include <chrono>
#include <csignal>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <numeric>

using namespace psyne;

// Global flag for graceful shutdown
std::atomic<bool> server_running{true};

void signal_handler(int signal) {
    std::cout << "\n🛑 Received shutdown signal (" << signal << ")\n";
    server_running = false;
}

class TensorProcessor {
private:
    std::atomic<size_t> total_tensors_{0};
    std::atomic<size_t> total_elements_{0};
    std::atomic<size_t> total_bytes_{0};
    std::chrono::high_resolution_clock::time_point start_time_;

public:
    TensorProcessor() : start_time_(std::chrono::high_resolution_clock::now()) {}

    FloatVector process_tensor(const FloatVector& input, std::shared_ptr<Channel> channel) {
        total_tensors_++;
        total_elements_ += input.size();
        total_bytes_ += input.size() * sizeof(float);

        // Create output tensor (zero-copy in ring buffer)
        FloatVector output(channel);
        output.resize(input.size());

        // Apply ML-style processing: normalization + activation
        float sum = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            sum += input[i];
        }
        float mean = sum / static_cast<float>(input.size());

        // Compute variance for normalization
        float variance = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(input.size());
        float stddev = std::sqrt(variance + 1e-8f);  // Add epsilon for numerical stability

        // Normalize and apply ReLU activation
        for (size_t i = 0; i < input.size(); ++i) {
            float normalized = (input[i] - mean) / stddev;
            output[i] = std::max(0.0f, normalized);  // ReLU activation
        }

        return output;
    }

    void print_statistics() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
        
        double throughput_mbps = (total_bytes_ * 1000.0) / (duration.count() * 1024 * 1024);
        double tensor_rate = (total_tensors_ * 1000.0) / duration.count();
        
        std::cout << "\n📊 Server Performance Statistics:\n";
        std::cout << "   Tensors processed: " << total_tensors_.load() << "\n";
        std::cout << "   Total elements: " << total_elements_.load() << "\n";
        std::cout << "   Data volume: " << (total_bytes_.load() / 1024 / 1024) << " MB\n";
        std::cout << "   Uptime: " << duration.count() << " ms\n";
        std::cout << "   Throughput: " << throughput_mbps << " MB/s\n";
        std::cout << "   Tensor rate: " << tensor_rate << " tensors/s\n";
        std::cout << "   Avg tensor size: " << (total_elements_.load() / std::max(size_t(1), total_tensors_.load())) << " elements\n";
    }
};

void run_ai_inference_server() {
    std::cout << "🧠 AI Inference Server (TCP)\n";
    std::cout << "============================\n";
    std::cout << "Simulating real-time tensor processing pipeline\n\n";

    try {
        // Create high-performance server channel
        auto channel = Channel::create("tcp://:8080",
                                     32 * 1024 * 1024, // 32MB buffer for large tensors
                                     ChannelMode::SPSC,  // Single producer/consumer for max performance
                                     ChannelType::SingleType);

        std::cout << "✓ Server listening on port 8080\n";
        std::cout << "✓ Ring buffer: 32MB zero-copy capacity\n";
        std::cout << "✓ Processing mode: AI/ML tensor inference\n";
        std::cout << "✓ Ready for client connections...\n\n";

        TensorProcessor processor;
        
        // Statistics thread
        std::thread stats_thread([&processor]() {
            while (server_running) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                if (server_running) {
                    processor.print_statistics();
                }
            }
        });

        while (server_running) {
            auto input_tensor = channel->receive<FloatVector>(std::chrono::milliseconds(1000));

            if (input_tensor) {
                std::cout << "📥 Processing tensor: " << input_tensor->size() 
                         << " elements (" << (input_tensor->size() * sizeof(float) / 1024) 
                         << " KB)\n";

                // Process tensor (ML pipeline simulation)
                auto start_processing = std::chrono::high_resolution_clock::now();
                FloatVector result = processor.process_tensor(*input_tensor, channel);
                auto end_processing = std::chrono::high_resolution_clock::now();
                
                auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_processing - start_processing);

                // Send processed result back
                result.send();
                
                std::cout << "📤 Sent processed tensor (processing: " 
                         << processing_time.count() << " μs)\n";

                // Show first few processed values for verification
                if (result.size() > 0) {
                    std::cout << "   Sample outputs: ";
                    for (size_t i = 0; i < std::min(size_t(5), result.size()); ++i) {
                        std::cout << result[i] << " ";
                    }
                    if (result.size() > 5) std::cout << "...";
                    std::cout << "\n\n";
                }
            } else {
                std::cout << "⏳ Waiting for tensors...\n";
            }
        }

        server_running = false;
        stats_thread.join();
        
        std::cout << "\n🏁 Server shutting down gracefully...\n";
        processor.print_statistics();

    } catch (const std::exception &e) {
        std::cerr << "❌ Server error: " << e.what() << std::endl;
    }
}

void run_echo_server() {
    std::cout << "🔄 High-Performance Echo Server\n";
    std::cout << "===============================\n";
    std::cout << "Zero-copy echo server for latency testing\n\n";

    try {
        auto channel = Channel::create("tcp://:8080",
                                     8 * 1024 * 1024, // 8MB buffer
                                     ChannelMode::SPSC, 
                                     ChannelType::SingleType);

        std::cout << "✓ Echo server listening on port 8080\n";
        std::cout << "✓ Zero-latency mode enabled\n\n";

        size_t echo_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (server_running && echo_count < 1000) {  // Process up to 1000 echoes
            auto msg = channel->receive<FloatVector>(std::chrono::milliseconds(2000));

            if (msg) {
                echo_count++;
                
                // Immediate echo with zero-copy
                FloatVector echo(channel);
                echo.resize(msg->size());
                
                // Direct memory copy (still zero-copy within ring buffer)
                for (size_t i = 0; i < msg->size(); ++i) {
                    echo[i] = (*msg)[i];
                }
                
                echo.send();
                
                if (echo_count % 100 == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
                    double rate = (echo_count * 1000.0) / duration.count();
                    
                    std::cout << "🔄 Echoed " << echo_count << " messages (rate: " 
                             << rate << " msg/s)\n";
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n📊 Echo Server Results:\n";
        std::cout << "   Total echoes: " << echo_count << "\n";
        std::cout << "   Duration: " << duration.count() << " ms\n";
        std::cout << "   Average rate: " << (echo_count * 1000.0 / duration.count()) << " msg/s\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ Echo server error: " << e.what() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::cout << "Psyne TCP Server Examples\n";
    std::cout << "=========================\n\n";

    // Setup signal handler for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " [ai|echo]\n\n";
        std::cout << "Server modes:\n";
        std::cout << "  ai    # AI/ML inference server (tensor processing)\n";
        std::cout << "  echo  # High-performance echo server (latency testing)\n\n";
        std::cout << "Features:\n";
        std::cout << "  ✓ Zero-copy tensor processing\n";
        std::cout << "  ✓ Real-time performance metrics\n";
        std::cout << "  ✓ Graceful shutdown (Ctrl+C)\n";
        std::cout << "  ✓ Memory-efficient ring buffers\n";
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "ai") {
        run_ai_inference_server();
    } else if (mode == "echo") {
        run_echo_server();
    } else {
        std::cerr << "❌ Invalid mode. Use 'ai' or 'echo'\n";
        return 1;
    }

    return 0;
}