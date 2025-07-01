/**
 * @file behaviors_demo.cpp
 * @brief Demonstration of the clean Psyne behaviors architecture
 *
 * This example shows:
 * - Substrate owns memory + transport
 * - Message is a lens into substrate memory
 * - Channel bridges substrate, message lens, and pattern
 * - No circular dependencies!
 */

#include "../include/psyne/core/behaviors.hpp"
#include <iostream>
#include <memory>
#include <vector>

// Simple message type that works with substrate memory
struct SimpleMessage {
    int id;
    float value;
    char data[64];

    SimpleMessage() : id(0), value(0.0f) {
        std::memset(data, 0, sizeof(data));
    }

    SimpleMessage(int id, float value) : id(id), value(value) {
        std::memset(data, 0, sizeof(data));
        std::snprintf(data, sizeof(data), "Message_%d_%.2f", id, value);
    }
};

// Simple substrate implementation
class DemoSubstrate : public psyne::behaviors::SubstrateBehavior {
public:
    void *allocate_memory_slab(size_t size_bytes) override {
        std::cout << "DemoSubstrate: Allocating " << size_bytes << " bytes\n";
        allocated_memory_ = std::aligned_alloc(64, size_bytes);
        slab_size_ = size_bytes;
        return allocated_memory_;
    }

    void deallocate_memory_slab(void *memory) override {
        std::cout << "DemoSubstrate: Deallocating memory slab\n";
        if (memory == allocated_memory_) {
            std::free(memory);
            allocated_memory_ = nullptr;
        }
    }

    void transport_send(void *data, size_t size) override {
        std::cout << "DemoSubstrate: Sending " << size
                  << " bytes via transport\n";
        // In real implementation, this would send over network/IPC/etc
        auto *msg = static_cast<SimpleMessage *>(data);
        std::cout << "  -> Message ID: " << msg->id << ", Value: " << msg->value
                  << ", Data: " << msg->data << "\n";
    }

    void transport_receive(void *buffer, size_t buffer_size) override {
        std::cout << "DemoSubstrate: Ready to receive up to " << buffer_size
                  << " bytes\n";
    }

    const char *substrate_name() const override {
        return "DemoSubstrate";
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

// Simple pattern implementation
class DemoPattern : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        // Simple sequential allocation
        size_t offset = allocation_count_ * message_size;
        allocation_count_++;

        std::cout << "DemoPattern: Coordinating allocation #"
                  << allocation_count_ << " at offset " << offset << "\n";

        return static_cast<char *>(slab_memory) + offset;
    }

    void *coordinate_receive() override {
        std::cout << "DemoPattern: Coordinating receive operation\n";
        // In real implementation, this would return next available message
        return nullptr;
    }

    void producer_sync() override {
        std::cout << "DemoPattern: Producer synchronization\n";
    }

    void consumer_sync() override {
        std::cout << "DemoPattern: Consumer synchronization\n";
    }

    const char *pattern_name() const override {
        return "DemoPattern";
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
    size_t allocation_count_ = 0;
};

void demonstrate_behaviors() {
    std::cout << "\n=== Psyne Behaviors Architecture Demo ===\n";
    std::cout << "Substrate = Memory Owner + Transport Owner\n";
    std::cout << "Message = Lens into Substrate Memory\n";
    std::cout << "Pattern = Coordination + Synchronization\n";
    std::cout << "Channel = Bridge Between All Three\n\n";

    // Create channel bridge that orchestrates everything
    psyne::behaviors::ChannelBridge<SimpleMessage, DemoSubstrate, DemoPattern>
        channel(1024);

    std::cout << "\nChannel created successfully!\n";
    std::cout << "- Substrate: " << channel.substrate_name() << "\n";
    std::cout << "- Pattern: " << channel.pattern_name() << "\n";
    std::cout << "- Zero-copy: " << (channel.is_zero_copy() ? "Yes" : "No")
              << "\n";
    std::cout << "- Needs locks: " << (channel.needs_locks() ? "Yes" : "No")
              << "\n\n";

    // Create messages using the bridge
    std::cout << "Creating messages through channel bridge...\n\n";

    // Message 1: Default construction
    {
        std::cout << "1. Creating default message:\n";
        auto msg1 = channel.create_message();

        // Access message data through lens
        msg1->id = 100;
        msg1->value = 3.14f;
        std::strcpy(msg1->data, "Hello from lens!");

        std::cout << "   Message data: ID=" << msg1->id
                  << ", Value=" << msg1->value << ", Data='" << msg1->data
                  << "'\n";

        // Send via substrate transport
        channel.send_message(msg1);
        std::cout << "\n";
    }

    // Message 2: Construction with arguments
    {
        std::cout << "2. Creating message with arguments:\n";
        auto msg2 = channel.create_message(200, 2.71f);

        std::cout << "   Message data: ID=" << msg2->id
                  << ", Value=" << msg2->value << ", Data='" << msg2->data
                  << "'\n";

        channel.send_message(msg2);
        std::cout << "\n";
    }

    // Message 3: Another message to show pattern coordination
    {
        std::cout << "3. Creating third message:\n";
        auto msg3 = channel.create_message(300, 1.41f);

        std::cout << "   Message data: ID=" << msg3->id
                  << ", Value=" << msg3->value << ", Data='" << msg3->data
                  << "'\n";

        channel.send_message(msg3);
        std::cout << "\n";
    }

    // Try to receive (would work with proper pattern implementation)
    std::cout << "4. Attempting to receive message:\n";
    auto received = channel.try_receive();
    if (!received) {
        std::cout << "   No messages available (pattern coordination returned "
                     "nullptr)\n";
    }

    std::cout << "\n=== Demo Complete ===\n";
    std::cout << "✅ Substrate allocated and managed memory\n";
    std::cout << "✅ Messages acted as lenses into substrate memory\n";
    std::cout << "✅ Pattern coordinated allocations\n";
    std::cout << "✅ Channel bridged all components\n";
    std::cout << "✅ NO CIRCULAR DEPENDENCIES!\n";
    std::cout << "✅ CLEAN SEPARATION OF CONCERNS!\n";
    std::cout << "🚀 PSYNE = VICTORY!\n\n";
}

int main() {
    std::cout << "Psyne Behaviors Architecture Demonstration\n";
    std::cout << "==========================================\n";

    try {
        demonstrate_behaviors();

        std::cout << "Architecture Summary:\n";
        std::cout << "====================\n";
        std::cout << "• Substrate owns memory slab and transport protocol\n";
        std::cout << "• Messages are lenses that provide typed access to "
                     "substrate memory\n";
        std::cout << "• Patterns coordinate producer/consumer access and "
                     "synchronization\n";
        std::cout << "• Channels bridge all three components together\n";
        std::cout << "• Zero circular dependencies - clean and extensible!\n\n";

        std::cout << "Plugin Potential:\n";
        std::cout << "================\n";
        std::cout << "• InfiniBand vendors can create custom substrates\n";
        std::cout << "• GPU vendors can implement device-specific memory "
                     "management\n";
        std::cout
            << "• Pattern developers can create new coordination algorithms\n";
        std::cout << "• Message types can be hardware-aware and optimized\n";
        std::cout << "• Full ecosystem extensibility achieved!\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}