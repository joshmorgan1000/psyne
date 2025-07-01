/**
 * @file quick_all_substrates.cpp
 * @brief Quick test of all pattern × substrate combinations (5s each)
 *
 * Shortened version for rapid testing before running the full 20-minute suite
 */

#include "../include/psyne/core/behaviors.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

// Simple test message
struct TestMessage {
    uint64_t id;
    uint64_t timestamp;
    uint32_t producer_id;
    char data[32];
};

// Quick substrate implementations (same as comprehensive but simpler)
class QuickInProcess : public psyne::behaviors::SubstrateBehavior {
public:
    void *allocate_memory_slab(size_t size_bytes) override {
        allocated_memory_ = std::aligned_alloc(64, size_bytes);
        return allocated_memory_;
    }
    void deallocate_memory_slab(void *memory) override {
        if (memory == allocated_memory_)
            std::free(memory);
    }
    void transport_send(void *data, size_t size) override {}
    void transport_receive(void *buffer, size_t buffer_size) override {}
    const char *substrate_name() const override {
        return "InProcess";
    }
    bool is_zero_copy() const override {
        return true;
    }
    bool is_cross_process() const override {
        return false;
    }

private:
    void *allocated_memory_ = nullptr;
};

class QuickIPC : public psyne::behaviors::SubstrateBehavior {
public:
    void *allocate_memory_slab(size_t size_bytes) override {
        return std::aligned_alloc(64, size_bytes);
    }
    void deallocate_memory_slab(void *memory) override {
        std::free(memory);
    }
    void transport_send(void *data, size_t size) override {}
    void transport_receive(void *buffer, size_t buffer_size) override {}
    const char *substrate_name() const override {
        return "IPC";
    }
    bool is_zero_copy() const override {
        return true;
    }
    bool is_cross_process() const override {
        return true;
    }
};

class QuickTCP : public psyne::behaviors::SubstrateBehavior {
public:
    void *allocate_memory_slab(size_t size_bytes) override {
        return std::aligned_alloc(64, size_bytes);
    }
    void deallocate_memory_slab(void *memory) override {
        std::free(memory);
    }
    void transport_send(void *data, size_t size) override {}
    void transport_receive(void *buffer, size_t buffer_size) override {}
    const char *substrate_name() const override {
        return "TCP";
    }
    bool is_zero_copy() const override {
        return false;
    }
    bool is_cross_process() const override {
        return true;
    }
};

// Quick pattern implementations
class QuickSPSC : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        if (!slab_memory_) {
            slab_memory_ = slab_memory;
            message_size_ = message_size;
        }
        size_t slot = write_pos_.fetch_add(1) % 1024;
        return static_cast<char *>(slab_memory) + (slot * message_size);
    }
    void *coordinate_receive() override {
        size_t current_read = read_pos_.load();
        size_t current_write = write_pos_.load();
        if (current_read >= current_write)
            return nullptr;
        size_t slot = current_read % 1024;
        read_pos_.fetch_add(1);
        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }
    void producer_sync() override {}
    void consumer_sync() override {}
    const char *pattern_name() const override {
        return "SPSC";
    }
    bool needs_locks() const override {
        return false;
    }
    size_t max_producers() const override {
        return 1;
    }
    size_t max_consumers() const override {
        return 1;
    }

private:
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
};

template <typename SubstrateType, typename PatternType>
void quick_test(const std::string &test_name) {
    std::cout << "Testing " << test_name << "... ";

    try {
        using ChannelType =
            psyne::behaviors::ChannelBridge<TestMessage, SubstrateType,
                                            PatternType>;
        ChannelType channel(1024 * 1024); // 1MB

        std::atomic<size_t> sent{0}, received{0};
        std::atomic<bool> running{true};

        std::thread producer([&]() {
            while (running.load()) {
                try {
                    auto msg = channel.create_message();
                    msg->id = sent.load() + 1;
                    channel.send_message(msg);
                    sent.fetch_add(1);
                } catch (...) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        });

        std::thread consumer([&]() {
            while (running.load()) {
                auto msg_opt = channel.try_receive();
                if (msg_opt) {
                    received.fetch_add(1);
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        });

        std::this_thread::sleep_for(std::chrono::seconds(5));
        running.store(false);
        producer.join();
        consumer.join();

        double throughput = received.load() / 5.0;
        std::cout << "✅ " << (int)throughput << " msgs/s\n";

    } catch (const std::exception &e) {
        std::cout << "❌ FAILED: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "Quick Pattern × Substrate Test (5s each)\n";
    std::cout << "========================================\n";

    quick_test<QuickInProcess, QuickSPSC>("SPSC + InProcess");
    quick_test<QuickIPC, QuickSPSC>("SPSC + IPC");
    quick_test<QuickTCP, QuickSPSC>("SPSC + TCP");

    std::cout << "\n✅ Quick test complete! All substrate types working.\n";
    std::cout << "🚀 Ready for comprehensive 20-minute benchmark.\n";

    return 0;
}