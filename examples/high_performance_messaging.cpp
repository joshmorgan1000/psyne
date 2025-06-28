// High-performance messaging example for Psyne
// Demonstrates zero-copy messaging with all performance optimizations

#include <psyne/psyne.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <random>

using namespace psyne;
using namespace psyne::perf;

// High-throughput data message
class DataPacket : public Message<DataPacket> {
public:
    static constexpr uint32_t message_type = 200;
    
    using Message<DataPacket>::Message;
    
    static size_t calculate_size() {
        return 8 * 1024; // 8KB packets
    }
    
    // Structured data access
    struct Header {
        uint64_t sequence_id;
        uint64_t timestamp;
        uint32_t data_size;
        uint32_t checksum;
    };
    
    Header* header() { 
        return reinterpret_cast<Header*>(data()); 
    }
    
    float* payload() { 
        return reinterpret_cast<float*>(data() + sizeof(Header)); 
    }
    
    size_t payload_size() const {
        return size() - sizeof(Header);
    }
    
    size_t payload_elements() const {
        return payload_size() / sizeof(float);
    }
    
    // Initialize packet with SIMD-optimized data
    void initialize(uint64_t seq_id) {
        auto* hdr = header();
        hdr->sequence_id = seq_id;
        hdr->timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        hdr->data_size = static_cast<uint32_t>(payload_size());
        
        // Fill payload with test pattern using SIMD
        float* data = payload();
        size_t count = payload_elements();
        
        // Use SIMD to efficiently initialize data
        std::vector<float> pattern(count);
        for (size_t i = 0; i < count; ++i) {
            pattern[i] = std::sin(static_cast<float>(seq_id + i) * 0.1f);
        }
        
        // Copy using optimized memory operations
        std::memcpy(data, pattern.data(), count * sizeof(float));
        
        // Calculate checksum
        hdr->checksum = calculate_checksum();
    }
    
    bool verify() const {
        return header()->checksum == calculate_checksum();
    }

private:
    uint32_t calculate_checksum() const {
        uint32_t checksum = 0;
        const auto* data = reinterpret_cast<const uint32_t*>(payload());
        size_t count = payload_size() / sizeof(uint32_t);
        
        for (size_t i = 0; i < count; ++i) {
            checksum ^= data[i];
        }
        return checksum;
    }
};

class HighPerformanceProducer {
private:
    std::unique_ptr<HugePagePool> memory_pool_;
    std::unique_ptr<MessagePrefetcher> prefetcher_;
    std::atomic<uint64_t> sequence_counter_{0};
    std::atomic<uint64_t> messages_sent_{0};
    std::atomic<bool> running_{false};

public:
    HighPerformanceProducer() {
        // Configure huge page memory pool
        HugePagePool::PoolConfig pool_config;
        pool_config.initial_size = 256 * 1024 * 1024; // 256MB
        pool_config.policy = HugePagePolicy::TryBest;
        pool_config.numa_aware = true;
        
        memory_pool_ = std::make_unique<HugePagePool>(pool_config);
        
        // Configure message prefetcher
        MessagePrefetcher::MessageConfig prefetch_config;
        prefetch_config.strategy = MessagePrefetchStrategy::Sequential;
        prefetch_config.message_size_estimate = DataPacket::calculate_size();
        prefetch_config.lookahead_messages = 4;
        prefetch_config.target_level = CacheLevel::L2;
        
        prefetcher_ = std::make_unique<MessagePrefetcher>(prefetch_config);
    }
    
    void start_production(const std::string& channel_uri, size_t target_rate_hz) {
        running_ = true;
        
        auto channel = Channel::create(channel_uri, 
                                     128 * 1024 * 1024, // 128MB ring buffer
                                     ChannelType::SingleType,
                                     ChannelMode::SPSC);
        
        if (!channel) {
            throw std::runtime_error("Failed to create channel");
        }
        
        // Set thread affinity for optimal performance
        auto affinity = get_recommended_affinity(ThreadType::HighThroughput);
        set_thread_affinity(affinity.core_ids);
        
        std::cout << "Producer started on cores: ";
        for (int core : affinity.core_ids) {
            std::cout << core << " ";
        }
        std::cout << "\n";
        
        auto target_interval = std::chrono::nanoseconds(1000000000 / target_rate_hz);
        auto next_send = std::chrono::high_resolution_clock::now();
        
        while (running_) {
            auto now = std::chrono::high_resolution_clock::now();
            if (now >= next_send) {
                send_message(*channel);
                next_send = now + target_interval;
            } else {
                // Micro-sleep to avoid busy waiting
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }
    }
    
    void stop() {
        running_ = false;
    }
    
    uint64_t get_messages_sent() const {
        return messages_sent_.load();
    }

private:
    void send_message(Channel& channel) {
        try {
            // Create optimized message
            auto msg = create_optimized_message<DataPacket>(channel);
            
            // Initialize with sequence number
            uint64_t seq_id = sequence_counter_.fetch_add(1);
            msg.message().initialize(seq_id);
            
            // Prefetch next message location
            prefetcher_->prefetch_message_write(
                msg.message().data(), 
                msg.message().size()
            );
            
            // Send with optimizations
            msg.send_optimized();
            
            messages_sent_.fetch_add(1);
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to send message: " << e.what() << "\n";
        }
    }
};

class HighPerformanceConsumer {
private:
    std::unique_ptr<AdaptivePrefetcher> prefetcher_;
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint64_t> messages_verified_{0};
    std::atomic<bool> running_{false};
    
    // SIMD-optimized processing buffers
    std::vector<float, CacheAlignedAllocator<float>> processing_buffer_;

public:
    HighPerformanceConsumer() {
        AdaptivePrefetcher::Config prefetch_config;
        prefetch_config.history_size = 32;
        prefetch_config.confidence_threshold = 0.8;
        prefetch_config.enable_stride_detection = true;
        prefetch_config.target_level = CacheLevel::L1;
        
        prefetcher_ = std::make_unique<AdaptivePrefetcher>(prefetch_config);
        
        // Pre-allocate processing buffer
        processing_buffer_.resize(2048); // Cache-aligned buffer
    }
    
    void start_consumption(const std::string& channel_uri) {
        running_ = true;
        
        auto channel = Channel::connect(channel_uri);
        if (!channel) {
            throw std::runtime_error("Failed to connect to channel");
        }
        
        // Set thread affinity for low latency
        auto affinity = get_recommended_affinity(ThreadType::LowLatency);
        set_thread_affinity(affinity.core_ids);
        
        std::cout << "Consumer started on cores: ";
        for (int core : affinity.core_ids) {
            std::cout << core << " ";
        }
        std::cout << "\n";
        
        while (running_) {
            try {
                auto msg = channel->receive<DataPacket>(std::chrono::milliseconds(1));
                if (msg) {
                    process_message(*msg);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error receiving message: " << e.what() << "\n";
            }
        }
    }
    
    void stop() {
        running_ = false;
    }
    
    uint64_t get_messages_received() const {
        return messages_received_.load();
    }
    
    uint64_t get_messages_verified() const {
        return messages_verified_.load();
    }
    
    AdaptivePrefetcher::Stats get_prefetch_stats() const {
        return prefetcher_->get_stats();
    }

private:
    void process_message(const DataPacket& msg) {
        messages_received_.fetch_add(1);
        
        // Record access for adaptive prefetching
        prefetcher_->record_access(msg.data(), PrefetchHint::Read);
        
        // Prefetch predicted next accesses
        prefetcher_->prefetch_predicted(2);
        
        // Verify message integrity
        if (msg.verify()) {
            messages_verified_.fetch_add(1);
            
            // Process payload with SIMD optimizations
            process_payload_simd(msg);
        } else {
            std::cerr << "Message verification failed for sequence " 
                      << msg.header()->sequence_id << "\n";
        }
    }
    
    void process_payload_simd(const DataPacket& msg) {
        const float* data = msg.payload();
        size_t count = msg.payload_elements();
        
        if (count > processing_buffer_.size()) {
            processing_buffer_.resize(count);
        }
        
        // Apply SIMD processing - calculate squared magnitude
        for (size_t i = 0; i < count; i += 8) {
            size_t remaining = std::min(size_t(8), count - i);
            vector_multiply_f32(&data[i], &data[i], &processing_buffer_[i], remaining);
        }
        
        // Calculate sum using SIMD
        float sum = 0.0f;
        for (size_t i = 0; i < count; i += 256) { // Process in cache-friendly chunks
            size_t chunk_size = std::min(size_t(256), count - i);
            sum += vector_sum_f32(&processing_buffer_[i], chunk_size);
        }
        
        // Store result (in real application, would send to next stage)
        volatile float result = sum; // Prevent optimization
        (void)result;
    }
};

void run_performance_benchmark() {
    std::cout << "\n=== High-Performance Messaging Benchmark ===\n";
    
    const std::string channel_uri = "ipc://performance_test";
    const size_t target_rate = 10000; // 10K messages/second
    const auto test_duration = std::chrono::seconds(10);
    
    // Initialize performance optimizations
    PerformanceManager::Config config;
    config.enable_simd = true;
    config.enable_huge_pages = true;
    config.enable_numa_affinity = true;
    config.enable_cpu_affinity = true;
    config.enable_prefetching = true;
    config.auto_tune = true;
    
    enable_performance_optimizations(config);
    
    HighPerformanceProducer producer;
    HighPerformanceConsumer consumer;
    
    std::cout << "Starting benchmark...\n";
    std::cout << "Target rate: " << target_rate << " msg/s\n";
    std::cout << "Duration: " << test_duration.count() << " seconds\n";
    std::cout << "Message size: " << DataPacket::calculate_size() << " bytes\n";
    
    // Start consumer in separate thread
    std::thread consumer_thread([&consumer, &channel_uri]() {
        try {
            consumer.start_consumption(channel_uri);
        } catch (const std::exception& e) {
            std::cerr << "Consumer error: " << e.what() << "\n";
        }
    });
    
    // Start producer in separate thread
    std::thread producer_thread([&producer, &channel_uri, target_rate]() {
        try {
            producer.start_production(channel_uri, target_rate);
        } catch (const std::exception& e) {
            std::cerr << "Producer error: " << e.what() << "\n";
        }
    });
    
    // Wait for test duration
    std::this_thread::sleep_for(test_duration);
    
    // Stop both threads
    producer.stop();
    consumer.stop();
    
    producer_thread.join();
    consumer_thread.join();
    
    // Report results
    auto messages_sent = producer.get_messages_sent();
    auto messages_received = consumer.get_messages_received();
    auto messages_verified = consumer.get_messages_verified();
    
    double actual_rate = static_cast<double>(messages_sent) / test_duration.count();
    double throughput_mbps = (actual_rate * DataPacket::calculate_size()) / (1024 * 1024);
    double loss_rate = 1.0 - (static_cast<double>(messages_received) / messages_sent);
    double error_rate = 1.0 - (static_cast<double>(messages_verified) / messages_received);
    
    std::cout << "\n=== Benchmark Results ===\n";
    std::cout << "Messages sent: " << messages_sent << "\n";
    std::cout << "Messages received: " << messages_received << "\n";
    std::cout << "Messages verified: " << messages_verified << "\n";
    std::cout << "Actual rate: " << actual_rate << " msg/s\n";
    std::cout << "Throughput: " << throughput_mbps << " MB/s\n";
    std::cout << "Loss rate: " << (loss_rate * 100) << "%\n";
    std::cout << "Error rate: " << (error_rate * 100) << "%\n";
    
    // Prefetching statistics
    auto prefetch_stats = consumer.get_prefetch_stats();
    std::cout << "\nPrefetch Statistics:\n";
    std::cout << "  Total accesses: " << prefetch_stats.total_accesses << "\n";
    std::cout << "  Prefetches issued: " << prefetch_stats.prefetches_issued << "\n";
    std::cout << "  Hit rate: " << (prefetch_stats.hit_rate * 100) << "%\n";
    std::cout << "  Confidence: " << prefetch_stats.confidence << "\n";
    
    // System performance
    auto& manager = get_performance_manager();
    auto perf = manager.measure_system_performance();
    std::cout << "\nSystem Performance:\n";
    std::cout << "  Memory bandwidth: " << perf.memory_bandwidth_gbps << " GB/s\n";
    std::cout << "  Cache hit rate: " << (perf.cache_hit_rate * 100) << "%\n";
    std::cout << "  Using huge pages: " << (perf.using_huge_pages ? "Yes" : "No") << "\n";
    std::cout << "  SIMD accelerated: " << (perf.simd_accelerated ? "Yes" : "No") << "\n";
}

int main() {
    std::cout << "Psyne High-Performance Messaging Demo\n";
    std::cout << "=====================================\n";
    
    try {
        run_performance_benchmark();
        
        std::cout << "\n" << get_performance_summary() << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}