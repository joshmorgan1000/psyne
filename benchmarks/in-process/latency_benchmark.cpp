/**
 * @file latency_benchmark.cpp
 * @brief Latency-focused benchmark for measuring round-trip and one-way latencies
 */

#include <psyne/psyne.hpp>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace psyne_bench {

class LatencyMessage : public psyne::Message<LatencyMessage> {
public:
    static constexpr uint32_t message_type = 400;
    using Message<LatencyMessage>::Message;
    
    static size_t calculate_size() { return 4096; } // Small messages for latency
    
    void set_timestamp(std::chrono::high_resolution_clock::time_point ts) {
        if (is_valid()) {
            std::memcpy(data(), &ts, sizeof(ts));
        }
    }
    
    std::chrono::high_resolution_clock::time_point get_timestamp() const {
        std::chrono::high_resolution_clock::time_point ts;
        if (is_valid()) {
            std::memcpy(&ts, data(), sizeof(ts));
        }
        return ts;
    }
};

void run_latency_test(const std::string& name, size_t num_messages) {
    std::cout << "Running " << name << " latency test (" << num_messages << " messages)..." << std::endl;
    
    auto channel = psyne::create_channel("memory://latency_test", 64*1024*1024, psyne::ChannelMode::SPSC);
    std::vector<std::chrono::nanoseconds> latencies;
    latencies.reserve(num_messages);
    
    std::atomic<bool> done{false};
    
    // Consumer thread
    std::thread consumer([&]() {
        size_t received = 0;
        while (received < num_messages) {
            auto msg_opt = channel->receive<LatencyMessage>(std::chrono::milliseconds(100));
            if (msg_opt) {
                auto receive_time = std::chrono::high_resolution_clock::now();
                auto send_time = msg_opt->get_timestamp();
                latencies.push_back(receive_time - send_time);
                received++;
            }
        }
        done = true;
    });
    
    // Producer thread
    std::thread producer([&]() {
        for (size_t i = 0; i < num_messages; ++i) {
            bool sent = false;
            while (!sent) {
                try {
                    LatencyMessage msg(*channel);
                    msg.set_timestamp(std::chrono::high_resolution_clock::now());
                    msg.send();
                    sent = true;
                } catch (...) {
                    std::this_thread::yield();
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10)); // Small delay
        }
    });
    
    producer.join();
    consumer.join();
    
    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    auto sum = std::accumulate(latencies.begin(), latencies.end(), std::chrono::nanoseconds(0));
    
    std::cout << "  Messages: " << latencies.size() << "\n";
    std::cout << "  Min latency: " << latencies.front().count() << " ns\n";
    std::cout << "  Max latency: " << latencies.back().count() << " ns\n";
    std::cout << "  Avg latency: " << (sum.count() / latencies.size()) << " ns\n";
    std::cout << "  P50 latency: " << latencies[latencies.size()/2].count() << " ns\n";
    std::cout << "  P95 latency: " << latencies[latencies.size()*95/100].count() << " ns\n";
    std::cout << "  P99 latency: " << latencies[latencies.size()*99/100].count() << " ns\n\n";
}

} // namespace psyne_bench

int main() {
    std::cout << "=== Psyne Latency Benchmark ===\n\n";
    
    using namespace psyne_bench;
    
    run_latency_test("Low Load", 10000);
    run_latency_test("Medium Load", 100000);
    run_latency_test("High Load", 1000000);
    
    std::cout << "Latency benchmark completed!\n";
    return 0;
}