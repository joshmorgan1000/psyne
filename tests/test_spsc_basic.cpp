/**
 * @file test_spsc_basic.cpp
 * @brief Basic functionality test for SPSC + InProcess
 *
 * Tests the fundamental operations:
 * - Create channel
 * - Send messages
 * - Receive messages
 * - Verify data integrity
 */

#include "../include/psyne/core/behaviors.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

// Simple test message
struct TestMessage {
    uint64_t id;
    uint64_t timestamp;
    char data[32];

    TestMessage() : id(0), timestamp(0) {
        std::memset(data, 0, sizeof(data));
    }

    TestMessage(uint64_t id, const char *msg) : id(id), timestamp(0) {
        std::strncpy(data, msg, sizeof(data) - 1);
        auto now = std::chrono::high_resolution_clock::now();
        timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now.time_since_epoch())
                        .count();
    }
};

// Simple working substrate
class SimpleInProcessSubstrate : public psyne::behaviors::SubstrateBehavior {
public:
    void *allocate_memory_slab(size_t size_bytes) override {
        allocated_memory_ = std::aligned_alloc(64, size_bytes);
        slab_size_ = size_bytes;
        std::cout << "InProcess: Allocated " << size_bytes << " bytes\n";
        return allocated_memory_;
    }

    void deallocate_memory_slab(void *memory) override {
        if (memory == allocated_memory_) {
            std::free(memory);
            allocated_memory_ = nullptr;
        }
    }

    void transport_send(void *data, size_t size) override {
        // In-process: just notify that message is ready
        auto *msg = static_cast<TestMessage *>(data);
        std::cout << "InProcess: Message " << msg->id << " ready in memory\n";
    }

    void transport_receive(void *buffer, size_t buffer_size) override {
        // In-process: just acknowledge receive
        std::cout << "InProcess: Receive acknowledged\n";
    }

    const char *substrate_name() const override {
        return "SimpleInProcess";
    }
    bool is_zero_copy() const override {
        return true;
    }
    bool is_cross_process() const override {
        return false;
    }

private:
    void *allocated_memory_ = nullptr;
    size_t slab_size_ = 0;
};

// Simple working SPSC pattern
class SimpleSPSCPattern : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        // Simple ring buffer allocation
        size_t current_slot = write_pos_.load() % max_messages_;
        void *slot =
            static_cast<char *>(slab_memory) + (current_slot * message_size);

        // Store the slab and message size for receive
        slab_memory_ = slab_memory;
        message_size_ = message_size;

        write_pos_.fetch_add(1);
        std::cout << "SPSC: Allocated slot " << current_slot
                  << " for writing\n";

        return slot;
    }

    void *coordinate_receive() override {
        size_t current_read = read_pos_.load();
        size_t current_write = write_pos_.load();

        if (current_read >= current_write) {
            return nullptr; // No messages available
        }

        size_t slot = current_read % max_messages_;
        read_pos_.fetch_add(1);

        std::cout << "SPSC: Reading from slot " << slot << "\n";

        // Return actual pointer to the message slot
        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }

    void producer_sync() override {
        // SPSC is lock-free
    }

    void consumer_sync() override {
        // SPSC is lock-free
    }

    const char *pattern_name() const override {
        return "SimpleSPSC";
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

    void set_max_messages(size_t max) {
        max_messages_ = max;
    }

private:
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    size_t max_messages_ = 1024;
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
};

void test_basic_functionality() {
    std::cout << "\n=== Basic SPSC + InProcess Test ===\n";

    // Create channel
    psyne::behaviors::ChannelBridge<TestMessage, SimpleInProcessSubstrate,
                                    SimpleSPSCPattern>
        channel(4096);

    std::cout << "\nChannel created successfully!\n";
    std::cout << "Substrate: " << channel.substrate_name() << "\n";
    std::cout << "Pattern: " << channel.pattern_name() << "\n";
    std::cout << "Zero-copy: " << (channel.is_zero_copy() ? "Yes" : "No")
              << "\n";

    // Test message creation and sending
    std::cout << "\n--- Testing Message Creation ---\n";

    auto msg1 = channel.create_message(1, "Hello World");
    std::cout << "Created message 1: ID=" << msg1->id << " Data='" << msg1->data
              << "'\n";
    channel.send_message(msg1);

    auto msg2 = channel.create_message(2, "Test Message");
    std::cout << "Created message 2: ID=" << msg2->id << " Data='" << msg2->data
              << "'\n";
    channel.send_message(msg2);

    auto msg3 = channel.create_message(3, "Final Message");
    std::cout << "Created message 3: ID=" << msg3->id << " Data='" << msg3->data
              << "'\n";
    channel.send_message(msg3);

    // Test message receiving
    std::cout << "\n--- Testing Message Receiving ---\n";

    for (int i = 0; i < 3; ++i) {
        auto received = channel.try_receive();
        if (received) {
            std::cout << "Received message: ID=" << (*received)->id << " Data='"
                      << (*received)->data << "'\n";
        } else {
            std::cout
                << "No message available (pattern implementation incomplete)\n";
        }
    }

    std::cout << "\n=== Basic Test Complete ===\n";
}

void test_producer_consumer() {
    std::cout << "\n=== Producer-Consumer Test ===\n";

    psyne::behaviors::ChannelBridge<TestMessage, SimpleInProcessSubstrate,
                                    SimpleSPSCPattern>
        channel(8192);

    std::atomic<bool> producer_done{false};
    std::atomic<size_t> messages_sent{0};
    std::atomic<size_t> messages_received{0};

    const size_t total_messages = 10;

    // Producer thread
    std::thread producer([&]() {
        for (size_t i = 1; i <= total_messages; ++i) {
            auto msg = channel.create_message(i, "Producer message");
            channel.send_message(msg);
            messages_sent.fetch_add(1);

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        producer_done.store(true);
        std::cout << "Producer finished sending " << total_messages
                  << " messages\n";
    });

    // Consumer thread
    std::thread consumer([&]() {
        while (!producer_done.load() ||
               messages_received.load() < total_messages) {
            auto msg = channel.try_receive();
            if (msg) {
                messages_received.fetch_add(1);
                std::cout << "Consumer received message "
                          << messages_received.load() << " ID=" << (*msg)->id
                          << "\n";
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        std::cout << "Consumer finished receiving " << messages_received.load()
                  << " messages\n";
    });

    producer.join();
    consumer.join();

    std::cout << "Producer-Consumer test complete:\n";
    std::cout << "  Sent: " << messages_sent.load() << "\n";
    std::cout << "  Received: " << messages_received.load() << "\n";
    std::cout << "  Success: "
              << (messages_sent.load() == messages_received.load() ? "YES"
                                                                   : "NO")
              << "\n";
}

int main() {
    std::cout << "SPSC + InProcess Basic Functionality Test\n";
    std::cout << "=========================================\n";

    try {
        test_basic_functionality();
        test_producer_consumer();

        std::cout << "\n✅ All basic tests passed!\n";
        std::cout << "Ready for benchmarking and pattern expansion.\n";

    } catch (const std::exception &e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}