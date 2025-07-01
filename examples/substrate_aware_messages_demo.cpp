/**
 * @file substrate_aware_messages_demo.cpp
 * @brief Demonstration of substrate-aware messages
 * 
 * Shows how messages can take substrates in their constructors and
 * use them for sophisticated resource management, compression, etc.
 */

#include "psyne/channel/channel_v3.hpp"
#include "psyne/message/substrate_aware_types.hpp"
#include "psyne/global/logger.hpp"
#include <iostream>
#include <vector>
#include <any>

using namespace psyne;

/**
 * @brief Enhanced substrate that provides additional services to messages
 */
template<typename T>
class EnhancedSubstrate : public substrate::SubstrateBase<T> {
private:
    size_t allocations_count_ = 0;
    size_t message_registrations_ = 0;
    
public:
    T* allocate_slab(size_t size_bytes) override {
        allocations_count_++;
        LOG_INFO("EnhancedSubstrate: Allocating slab {} bytes (allocation #{})", 
                 size_bytes, allocations_count_);
        return static_cast<T*>(std::aligned_alloc(alignof(T), size_bytes));
    }
    
    void deallocate_slab(T* ptr) override {
        std::free(ptr);
    }
    
    void send_message(T* msg_ptr, std::vector<std::function<void(T*)>>& listeners) override {
        for (auto& listener : listeners) {
            listener(msg_ptr);
        }
    }
    
    boost::asio::awaitable<void> async_send_message(T* msg_ptr, 
                                                   std::vector<std::function<void(T*)>>& listeners) override {
        send_message(msg_ptr, listeners);
        co_return;
    }
    
    // Enhanced services for substrate-aware messages
    
    /**
     * @brief Allocate additional memory (used by DynamicVectorMessage)
     */
    void* allocate_additional(size_t size_bytes) {
        LOG_INFO("EnhancedSubstrate: Allocating additional {} bytes", size_bytes);
        return std::aligned_alloc(64, size_bytes);
    }
    
    /**
     * @brief Compress string data (used by StringMessage)
     */
    std::vector<uint8_t> compress_string(const std::string& str) {
        LOG_INFO("EnhancedSubstrate: Compressing string of length {}", str.length());
        
        // Simple "compression" - just convert to bytes with length prefix
        std::vector<uint8_t> compressed;
        compressed.reserve(str.length() + 4);
        
        // Store length
        uint32_t len = str.length();
        compressed.insert(compressed.end(), 
                         reinterpret_cast<uint8_t*>(&len), 
                         reinterpret_cast<uint8_t*>(&len) + 4);
        
        // Store data
        compressed.insert(compressed.end(), str.begin(), str.end());
        
        return compressed;
    }
    
    /**
     * @brief Register message type (used by SelfDescribingMessage)
     */
    void register_message_type(const std::string& type_name) {
        message_registrations_++;
        LOG_INFO("EnhancedSubstrate: Registered message type '{}' (registration #{})", 
                 type_name, message_registrations_);
    }
    
    /**
     * @brief Callback when message is destroyed
     */
    void on_message_destroyed() {
        LOG_DEBUG("EnhancedSubstrate: Message destroyed");
    }
    
    /**
     * @brief Serialize payload (used by SelfDescribingMessage)
     */
    template<typename U>
    std::vector<uint8_t> serialize(const U& payload) {
        LOG_INFO("EnhancedSubstrate: Serializing payload of type {}", typeid(U).name());
        
        // Simple serialization - just copy bytes
        std::vector<uint8_t> serialized(sizeof(U));
        std::memcpy(serialized.data(), &payload, sizeof(U));
        return serialized;
    }
    
    // Substrate interface
    bool needs_serialization() const override { return false; }
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }
    const char* name() const override { return "Enhanced"; }
    
    // Stats
    size_t get_allocations_count() const { return allocations_count_; }
    size_t get_registrations_count() const { return message_registrations_; }
};

/**
 * @brief Demo dynamic vector messages with substrate allocation
 */
void demo_dynamic_vectors() {
    std::cout << "\n=== Substrate-Aware Dynamic Vectors ===\n";
    
    using DynVectorMsg = message::DynamicVectorMessage<EnhancedSubstrate<float>>;
    using VectorChannel = Channel<DynVectorMsg,
                                 EnhancedSubstrate<DynVectorMsg>,
                                 pattern::SPSC<DynVectorMsg, EnhancedSubstrate<DynVectorMsg>>>;
    
    auto channel = std::make_shared<VectorChannel>();
    
    channel->register_listener([](DynVectorMsg* msg) {
        std::cout << "Received dynamic vector of size " << msg->size() << "\n";
        if (msg->size() > 0) {
            std::cout << "  First element: " << (*msg)[0] << "\n";
            std::cout << "  Substrate: " << msg->substrate().name() << "\n";
        }
    });
    
    // Create message with substrate-allocated additional memory
    {
        // Message constructor takes substrate reference!
        Message<DynVectorMsg, EnhancedSubstrate<DynVectorMsg>, 
               pattern::SPSC<DynVectorMsg, EnhancedSubstrate<DynVectorMsg>>> 
               msg(*channel, 1000); // size = 1000
        
        // Fill with data
        for (size_t i = 0; i < msg->size(); ++i) {
            (*msg)[i] = static_cast<float>(i * 0.1f);
        }
        
        std::cout << "Created dynamic vector with " << msg->size() << " elements\n";
        msg.send();
    }
}

/**
 * @brief Demo string messages with substrate compression
 */
void demo_string_messages() {
    std::cout << "\n=== Substrate-Aware String Messages ===\n";
    
    using StringMsg = message::StringMessage<EnhancedSubstrate<std::string>>;
    using StringChannel = Channel<StringMsg,
                                 EnhancedSubstrate<StringMsg>,
                                 pattern::SPSC<StringMsg, EnhancedSubstrate<StringMsg>>>;
    
    auto channel = std::make_shared<StringChannel>();
    
    channel->register_listener([](StringMsg* msg) {
        std::cout << "Received string message: \"" << msg->content() << "\"\n";
        std::cout << "  Substrate: " << msg->substrate().name() << "\n";
    });
    
    // Create string message with substrate compression
    {
        Message<StringMsg, EnhancedSubstrate<StringMsg>,
               pattern::SPSC<StringMsg, EnhancedSubstrate<StringMsg>>>
               msg(*channel, "Hello from substrate-aware messaging!");
        
        std::cout << "Created string message with substrate compression\n";
        msg.send();
    }
}

/**
 * @brief Demo self-describing messages with substrate serialization
 */
void demo_self_describing_messages() {
    std::cout << "\n=== Self-Describing Messages ===\n";
    
    using SelfDescMsg = message::SelfDescribingMessage<EnhancedSubstrate<std::any>>;
    using SelfDescChannel = Channel<SelfDescMsg,
                                   EnhancedSubstrate<SelfDescMsg>,
                                   pattern::SPSC<SelfDescMsg, EnhancedSubstrate<SelfDescMsg>>>;
    
    auto channel = std::make_shared<SelfDescChannel>();
    
    channel->register_listener([](SelfDescMsg* msg) {
        std::cout << "Received self-describing message of type: " << msg->type_name() << "\n";
        std::cout << "  Substrate: " << msg->substrate().name() << "\n";
    });
    
    // Create self-describing message with various payload types
    {
        Message<SelfDescMsg, EnhancedSubstrate<SelfDescMsg>,
               pattern::SPSC<SelfDescMsg, EnhancedSubstrate<SelfDescMsg>>>
               msg(*channel, 42.5f); // Float payload
        
        std::cout << "Created self-describing message with float payload\n";
        msg.send();
    }
    
    {
        Message<SelfDescMsg, EnhancedSubstrate<SelfDescMsg>,
               pattern::SPSC<SelfDescMsg, EnhancedSubstrate<SelfDescMsg>>>
               msg(*channel, std::string("Complex payload"));
        
        std::cout << "Created self-describing message with string payload\n";
        msg.send();
    }
}

/**
 * @brief Show substrate statistics
 */
void show_substrate_stats() {
    std::cout << "\n=== Substrate Statistics ===\n";
    
    // Create a substrate and show its stats
    EnhancedSubstrate<float> substrate;
    
    // Simulate some operations
    auto ptr1 = substrate.allocate_additional(1024);
    auto ptr2 = substrate.allocate_additional(2048);
    
    substrate.register_message_type("TestMessage1");
    substrate.register_message_type("TestMessage2");
    
    std::cout << "Substrate Stats:\n";
    std::cout << "  Allocations: " << substrate.get_allocations_count() << "\n";
    std::cout << "  Registrations: " << substrate.get_registrations_count() << "\n";
    std::cout << "  Zero-copy: " << substrate.is_zero_copy() << "\n";
    std::cout << "  Cross-process: " << substrate.is_cross_process() << "\n";
    
    std::free(ptr1);
    std::free(ptr2);
}

int main() {
    LogManager::set_level(LogLevel::INFO);
    
    std::cout << "Substrate-Aware Messages Demo\n";
    std::cout << "============================\n";
    std::cout << "Demonstrating messages that take substrates in constructors\n";
    std::cout << "and use them for sophisticated resource management!\n";
    
    try {
        demo_dynamic_vectors();
        demo_string_messages();
        demo_self_describing_messages();
        show_substrate_stats();
        
        std::cout << "\n🎉 Substrate-aware messaging works perfectly!\n";
        std::cout << "Messages can now use substrates for:\n";
        std::cout << "- Additional memory allocation\n";
        std::cout << "- Compression and serialization\n";
        std::cout << "- Type registration and metadata\n";
        std::cout << "- Resource management and cleanup\n";
        std::cout << "\nThis opens up INFINITE possibilities for custom message types! 🚀\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}