/**
 * @file quick_beast_test.cpp
 * @brief QUICK HIGH THREAD COUNT TEST - 5 seconds each
 */

#include "../include/psyne/core/behaviors.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <vector>
#include <iomanip>

struct TestMessage {
    uint64_t id;
    uint32_t producer_id;
    char padding[48];
};

// Reuse substrate/pattern classes from before but with names...
class QuickInProcess : public psyne::behaviors::SubstrateBehavior {
public:
    void* allocate_memory_slab(size_t size_bytes) override {
        allocated_memory_ = std::aligned_alloc(64, size_bytes);
        return allocated_memory_;
    }
    void deallocate_memory_slab(void* memory) override { if (memory == allocated_memory_) std::free(memory); }
    void transport_send(void* data, size_t size) override {}
    void transport_receive(void* buffer, size_t buffer_size) override {}
    const char* substrate_name() const override { return "InProcess"; }
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }
private:
    void* allocated_memory_ = nullptr;
};

// Pattern implementations with 64K buffer
template<bool NeedLocks>
class QuickPattern : public psyne::behaviors::PatternBehavior {
protected:
    static constexpr size_t buffer_size_ = 65536; // 64K slots
    std::atomic<size_t> write_pos_{0}; 
    std::atomic<size_t> read_pos_{0}; 
    void* slab_memory_ = nullptr; 
    size_t message_size_ = 0;
    std::mutex init_mutex_;
};

class QuickMPSC : public QuickPattern<true> {
public:
    void* coordinate_allocation(void* slab_memory, size_t message_size) override {
        if (!slab_memory_) { 
            std::lock_guard<std::mutex> lock(init_mutex_);
            if (!slab_memory_) { slab_memory_ = slab_memory; message_size_ = message_size; }
        }
        size_t slot = write_pos_.fetch_add(1) % buffer_size_;
        return static_cast<char*>(slab_memory) + (slot * message_size);
    }
    void* coordinate_receive() override {
        size_t current_read = read_pos_.load(); 
        size_t current_write = write_pos_.load();
        if (current_read >= current_write) return nullptr;
        size_t slot = current_read % buffer_size_; 
        read_pos_.fetch_add(1);
        return static_cast<char*>(slab_memory_) + (slot * message_size_);
    }
    void producer_sync() override {} 
    void consumer_sync() override {}
    const char* pattern_name() const override { return "MPSC"; }
    bool needs_locks() const override { return true; }
    size_t max_producers() const override { return SIZE_MAX; } 
    size_t max_consumers() const override { return 1; }
};

class QuickSPMC : public QuickPattern<true> {
public:
    void* coordinate_allocation(void* slab_memory, size_t message_size) override {
        if (!slab_memory_) { slab_memory_ = slab_memory; message_size_ = message_size; }
        size_t slot = write_pos_.fetch_add(1) % buffer_size_;
        return static_cast<char*>(slab_memory) + (slot * message_size);
    }
    void* coordinate_receive() override {
        size_t current_read = read_pos_.fetch_add(1);
        size_t current_write = write_pos_.load();
        if (current_read >= current_write) {
            read_pos_.fetch_sub(1);
            return nullptr;
        }
        size_t slot = current_read % buffer_size_;
        return static_cast<char*>(slab_memory_) + (slot * message_size_);
    }
    void producer_sync() override {} 
    void consumer_sync() override {}
    const char* pattern_name() const override { return "SPMC"; }
    bool needs_locks() const override { return true; }
    size_t max_producers() const override { return 1; } 
    size_t max_consumers() const override { return SIZE_MAX; }
};

template<typename PatternType>
void quick_beast_test(const std::string& name, size_t producers, size_t consumers) {
    std::cout << std::setw(20) << name << " (" << std::setw(2) << producers << "P × " 
              << std::setw(2) << consumers << "C): ";
    std::cout.flush();
    
    try {
        using ChannelType = psyne::behaviors::ChannelBridge<TestMessage, QuickInProcess, PatternType>;
        ChannelType channel(32 * 1024 * 1024); // 32MB
        
        std::atomic<size_t> sent{0}, received{0};
        std::atomic<bool> running{true};
        
        // Producers
        std::vector<std::thread> producers_vec;
        for (size_t i = 0; i < producers; ++i) {
            producers_vec.emplace_back([&]() {
                while (running.load()) {
                    try {
                        auto msg = channel.create_message();
                        msg->id = sent.fetch_add(1);
                        channel.send_message(msg);
                    } catch (...) { std::this_thread::yield(); }
                }
            });
        }
        
        // Consumers
        std::vector<std::thread> consumers_vec;
        for (size_t i = 0; i < consumers; ++i) {
            consumers_vec.emplace_back([&]() {
                while (running.load()) {
                    if (channel.try_receive()) { 
                        received.fetch_add(1);
                    } else { 
                        std::this_thread::yield();
                    }
                }
            });
        }
        
        // Run for 5 seconds
        std::this_thread::sleep_for(std::chrono::seconds(5));
        running.store(false);
        
        for (auto& t : producers_vec) t.join();
        for (auto& t : consumers_vec) t.join();
        
        double throughput = received.load() / 5.0 / 1'000'000.0;
        std::cout << std::fixed << std::setprecision(2) << std::setw(6) << throughput << "M msgs/s\n";
        
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << "\n";
    }
}

int main() {
    auto cores = std::thread::hardware_concurrency();
    std::cout << "🚀 PSYNE v2.0 HIGH THREAD COUNT TEST\n";
    std::cout << "====================================\n";
    std::cout << "System cores: " << cores << "\n\n";
    
    std::cout << "🔥 MPSC SCALING (Many Producers → 1 Consumer)\n";
    std::cout << "=============================================\n";
    quick_beast_test<QuickMPSC>("MPSC", 1, 1);
    quick_beast_test<QuickMPSC>("MPSC", 2, 1);
    quick_beast_test<QuickMPSC>("MPSC", 4, 1);
    quick_beast_test<QuickMPSC>("MPSC", 8, 1);
    quick_beast_test<QuickMPSC>("MPSC", 16, 1);
    quick_beast_test<QuickMPSC>("MPSC", 32, 1);
    if (cores >= 64) quick_beast_test<QuickMPSC>("MPSC", 64, 1);
    
    std::cout << "\n📡 SPMC SCALING (1 Producer → Many Consumers) [KEY FOR MANIFOLDB!]\n";
    std::cout << "================================================================\n";
    quick_beast_test<QuickSPMC>("SPMC", 1, 1);
    quick_beast_test<QuickSPMC>("SPMC", 1, 2);
    quick_beast_test<QuickSPMC>("SPMC", 1, 4);
    quick_beast_test<QuickSPMC>("SPMC", 1, 8);
    quick_beast_test<QuickSPMC>("SPMC", 1, 16);
    quick_beast_test<QuickSPMC>("SPMC", 1, 32);
    if (cores >= 64) quick_beast_test<QuickSPMC>("SPMC", 1, 64);
    
    std::cout << "\n🎉 SHIP IT TO PSYNETICS! 🚀\n";
    
    return 0;
}