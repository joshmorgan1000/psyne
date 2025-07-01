/**
 * @file test_mpmc_simple.cpp
 * @brief Simple MPMC test to isolate the crash
 */

#include "../include/psyne/core/behaviors.hpp"
#include "../include/psyne/channel/pattern/mpmc.hpp"
#include <iostream>
#include <thread>
#include <atomic>

struct SimpleMessage {
    uint64_t id;
    uint64_t producer_id;
    char data[32];
    
    SimpleMessage() : id(0), producer_id(0) {
        std::memset(data, 0, sizeof(data));
    }
    
    SimpleMessage(uint64_t msg_id, uint64_t prod_id) : id(msg_id), producer_id(prod_id) {
        std::snprintf(data, sizeof(data), "Msg_%llu_P_%llu", msg_id, prod_id);
    }
};

// Simple substrate for testing
class TestSubstrate : public psyne::behaviors::SubstrateBehavior {
public:
    void* allocate_memory_slab(size_t size_bytes) override {
        allocated_memory_ = std::aligned_alloc(64, size_bytes);
        slab_size_ = size_bytes;
        return allocated_memory_;
    }
    
    void deallocate_memory_slab(void* memory) override {
        if (memory && memory == allocated_memory_) {
            std::free(memory);
            allocated_memory_ = nullptr;
        }
    }
    
    void transport_send(void* data, size_t size) override {
        // In-process: no transport needed
    }
    
    void transport_receive(void* buffer, size_t buffer_size) override {
        // In-process: no transport needed
    }
    
    const char* substrate_name() const override { return "TestSubstrate"; }
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }

private:
    void* allocated_memory_ = nullptr;
    size_t slab_size_ = 0;
};

int main() {
    std::cout << "Simple MPMC Test\n";
    std::cout << "================\n";
    
    try {
        using ChannelType = psyne::behaviors::ChannelBridge<SimpleMessage, TestSubstrate, psyne::patterns::MPMC>;
        
        std::cout << "Creating MPMC channel...\n";
        ChannelType channel(8192);  // Smaller size for testing
        
        std::atomic<int> messages_sent{0};
        std::atomic<int> messages_received{0};
        
        // 2 producers, 2 consumers, 5 messages each
        const int num_producers = 2;
        const int num_consumers = 2;
        const int messages_per_producer = 5;
        const int total_messages = num_producers * messages_per_producer;
        
        std::cout << "Starting " << num_producers << " producers, " << num_consumers << " consumers\n";
        
        // Producers
        std::vector<std::thread> producers;
        for (int p = 0; p < num_producers; ++p) {
            producers.emplace_back([&, producer_id = p + 1]() {
                std::cout << "Producer " << producer_id << " starting\n";
                try {
                    for (int i = 1; i <= messages_per_producer; ++i) {
                        auto msg = channel.create_message(i, producer_id);
                        channel.send_message(msg);
                        messages_sent.fetch_add(1);
                        std::cout << "Producer " << producer_id << " sent message " << i << "\n";
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    std::cout << "Producer " << producer_id << " finished\n";
                } catch (const std::exception& e) {
                    std::cout << "Producer " << producer_id << " error: " << e.what() << "\n";
                }
            });
        }
        
        // Consumers
        std::vector<std::thread> consumers;
        for (int c = 0; c < num_consumers; ++c) {
            consumers.emplace_back([&, consumer_id = c + 1]() {
                std::cout << "Consumer " << consumer_id << " starting\n";
                try {
                    while (messages_received.load() < total_messages) {
                        auto msg_opt = channel.try_receive();
                        if (msg_opt) {
                            messages_received.fetch_add(1);
                            std::cout << "Consumer " << consumer_id << " received: ID=" << (*msg_opt)->id 
                                      << " Producer=" << (*msg_opt)->producer_id << "\n";
                        } else {
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        }
                    }
                    std::cout << "Consumer " << consumer_id << " finished\n";
                } catch (const std::exception& e) {
                    std::cout << "Consumer " << consumer_id << " error: " << e.what() << "\n";
                }
            });
        }
        
        // Wait for all threads
        for (auto& p : producers) {
            p.join();
        }
        for (auto& c : consumers) {
            c.join();
        }
        
        std::cout << "\nMPMC Test Results:\n";
        std::cout << "  Messages sent: " << messages_sent.load() << "\n";
        std::cout << "  Messages received: " << messages_received.load() << "\n";
        std::cout << "  Success: " << (messages_sent.load() == messages_received.load() ? "✅ PASSED" : "❌ FAILED") << "\n";
        
    } catch (const std::exception& e) {
        std::cout << "❌ MPMC test failed: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}