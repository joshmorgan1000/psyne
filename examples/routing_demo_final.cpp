#include <psyne/psyne.hpp>
#include "../src/routing/routing.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

using namespace psyne;
using namespace psyne::routing;

void demo_basic_routing() {
    std::cout << "=== Basic Message Routing Demo ===" << std::endl;
    
    // Create a channel
    auto channel = create_channel("memory://routing_demo", 64 * 1024);
    
    // Create a message router
    MessageRouter router;
    
    // Add route for FloatVector messages
    router.add_route<FloatVector>([](FloatVector&& msg) {
        std::cout << "Received FloatVector with " << msg.size() << " floats: ";
        for (size_t i = 0; i < std::min(msg.size(), size_t(5)); ++i) {
            std::cout << msg[i] << " ";
        }
        if (msg.size() > 5) std::cout << "...";
        std::cout << std::endl;
    });
    
    // Add route for ByteVector messages
    router.add_route<ByteVector>([](ByteVector&& msg) {
        std::cout << "Received ByteVector with " << msg.size() << " bytes" << std::endl;
    });
    
    // Add default handler for unmatched messages
    router.set_default_route([](uint32_t type, void* data, size_t size) {
        (void)data; // Unused
        std::cout << "Unmatched message: type=" << type << ", size=" << size << std::endl;
    });
    
    // Start the router
    router.start(*channel);
    
    // Send various messages
    std::cout << "\nSending messages..." << std::endl;
    
    // Send FloatVector
    {
        FloatVector msg(*channel);
        msg.resize(10);
        for (size_t i = 0; i < 10; ++i) {
            msg[i] = static_cast<float>(i * 1.5);
        }
        channel->send(msg);
    }
    
    // Send another FloatVector
    {
        FloatVector msg(*channel);
        msg.resize(3);
        msg[0] = 3.14f;
        msg[1] = 2.71f;
        msg[2] = 1.41f;
        channel->send(msg);
    }
    
    // Send ByteVector
    {
        ByteVector msg(*channel);
        size_t size = std::min(size_t(100), msg.capacity());
        msg.resize(size);
        for (size_t i = 0; i < size; ++i) {
            msg[i] = static_cast<uint8_t>(i % 256);
        }
        channel->send(msg);
    }
    
    // Wait for messages to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Stop the router
    router.stop();
    
    std::cout << "Basic routing demo completed!" << std::endl;
}

void demo_filtered_channels() {
    std::cout << "\n=== Filtered Channels Demo ===" << std::endl;
    
    // Create base channel
    auto base_channel = create_channel("memory://filtered_demo", 64 * 1024);
    
    // Create a channel that only receives large messages (> 100 bytes)
    auto large_msg_channel = create_filtered_channel(
        create_channel("memory://filtered_demo", 64 * 1024),
        std::make_unique<SizeFilter>(100, SIZE_MAX)
    );
    
    // Send messages in a separate thread
    std::thread sender([&base_channel]() {
        std::cout << "Sending messages of various sizes..." << std::endl;
        
        // Small message
        {
            FloatVector msg(*base_channel);
            msg.resize(10);  // ~40 bytes
            base_channel->send(msg);
            std::cout << "Sent FloatVector with 10 floats (~40 bytes)" << std::endl;
        }
        
        // Large message
        {
            FloatVector msg(*base_channel);
            msg.resize(50);  // ~200 bytes
            base_channel->send(msg);
            std::cout << "Sent FloatVector with 50 floats (~200 bytes)" << std::endl;
        }
        
        // Another small message
        {
            ByteVector msg(*base_channel);
            size_t size = std::min(size_t(50), msg.capacity());
            msg.resize(size);
            base_channel->send(msg);
            std::cout << "Sent ByteVector with " << size << " bytes" << std::endl;
        }
        
        // Another large message
        {
            ByteVector msg(*base_channel);
            size_t size = std::min(size_t(200), msg.capacity());
            msg.resize(size);
            base_channel->send(msg);
            std::cout << "Sent ByteVector with " << size << " bytes" << std::endl;
        }
    });
    
    // Receive only large messages
    std::cout << "\nReceiving only large messages (> 100 bytes):" << std::endl;
    int received = 0;
    auto start = std::chrono::steady_clock::now();
    
    while (received < 2 && std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
        size_t size;
        uint32_t type;
        void* data = large_msg_channel->receive_raw_message(size, type);
        if (data) {
            std::cout << "Received large message: type=" << type 
                      << ", size=" << size << " bytes" << std::endl;
            large_msg_channel->release_raw_message(data);
            received++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    sender.join();
    
    std::cout << "Filtered channels demo completed!" << std::endl;
}

void demo_composite_filters() {
    std::cout << "\n=== Composite Filters Demo ===" << std::endl;
    
    // Create a channel
    auto channel = create_channel("memory://composite_demo", 64 * 1024);
    
    // Create a composite filter: (Type is FloatVector) AND (Size > 100 bytes)
    auto composite_filter = std::make_unique<CompositeFilter>(CompositeFilter::Mode::AND);
    composite_filter->add_filter(std::make_unique<TypeFilter>(FloatVector::message_type));
    composite_filter->add_filter(std::make_unique<SizeFilter>(100, SIZE_MAX));
    
    // Create router with composite filter
    MessageRouter router;
    router.add_route(std::move(composite_filter), 
        [](uint32_t type, void* data, size_t size) {
            (void)type; // Unused
            (void)data; // Unused
            std::cout << "Received large FloatVector: size=" << size << " bytes" << std::endl;
        });
    
    // Add default route for other messages
    router.set_default_route([](uint32_t type, void* data, size_t size) {
        (void)data; // Unused
        std::cout << "Other message: type=" << type << ", size=" << size << " bytes" << std::endl;
    });
    
    router.start(*channel);
    
    std::cout << "\nSending various messages..." << std::endl;
    
    // Send small FloatVector (should go to default)
    {
        FloatVector msg(*channel);
        msg.resize(10);  // ~40 bytes + header
        channel->send(msg);
        std::cout << "Sent small FloatVector (10 elements)" << std::endl;
    }
    
    // Send large FloatVector (should match composite filter)
    {
        FloatVector msg(*channel);
        msg.resize(50);  // ~200 bytes + header
        channel->send(msg);
        std::cout << "Sent large FloatVector (50 elements)" << std::endl;
    }
    
    // Send large ByteVector (should go to default - wrong type)
    {
        ByteVector msg(*channel);
        size_t size = std::min(size_t(200), msg.capacity());
        msg.resize(size);
        channel->send(msg);
        std::cout << "Sent large ByteVector" << std::endl;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    router.stop();
    
    std::cout << "Composite filters demo completed!" << std::endl;
}

void demo_performance() {
    std::cout << "\n=== Routing Performance Demo ===" << std::endl;
    
    auto channel = create_channel("memory://perf_demo", 1024 * 1024);
    
    MessageRouter router;
    std::atomic<int> float_count{0};
    std::atomic<int> byte_count{0};
    
    // Add routes with counters
    router.add_route<FloatVector>([&float_count](FloatVector&& msg) {
        (void)msg; // Unused
        float_count++;
    });
    
    router.add_route<ByteVector>([&byte_count](ByteVector&& msg) {
        (void)msg; // Unused
        byte_count++;
    });
    
    router.start(*channel);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Send many messages
    const int messages_per_type = 10000;
    for (int i = 0; i < messages_per_type; ++i) {
        // Alternate between message types
        if (i % 2 == 0) {
            FloatVector msg(*channel);
            msg.resize(10);
            channel->send(msg);
        } else {
            ByteVector msg(*channel);
            size_t size = std::min(size_t(100), msg.capacity());
            msg.resize(size);
            channel->send(msg);
        }
    }
    
    // Wait for all messages to be routed
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    router.stop();
    
    std::cout << "Routing performance results:" << std::endl;
    std::cout << "  FloatVector messages: " << float_count << std::endl;
    std::cout << "  ByteVector messages: " << byte_count << std::endl;
    std::cout << "  Total messages: " << (float_count + byte_count) << std::endl;
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Throughput: " << (messages_per_type * 2 * 1000 / duration.count()) 
              << " messages/second" << std::endl;
    
    std::cout << "Performance demo completed!" << std::endl;
}

void demo_predicate_filter() {
    std::cout << "\n=== Predicate Filter Demo ===" << std::endl;
    
    auto channel = create_channel("memory://predicate_demo", 64 * 1024);
    
    MessageRouter router;
    
    // Route FloatVectors with first element > 5.0
    auto predicate = [](uint32_t type, const void* data, size_t size) -> bool {
        if (type != FloatVector::message_type) return false;
        if (size < sizeof(size_t) + sizeof(float)) return false;
        
        // Skip the size header
        const float* floats = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(data) + sizeof(size_t));
        return floats[0] > 5.0f;
    };
    
    router.add_route(
        std::make_unique<PredicateFilter>(predicate),
        [](uint32_t type, void* data, size_t size) {
            (void)type; // Unused
            (void)size; // Unused
            const float* floats = reinterpret_cast<const float*>(
                static_cast<const uint8_t*>(data) + sizeof(size_t));
            std::cout << "FloatVector with first element > 5.0: " << floats[0] << std::endl;
        });
    
    // Default route
    router.set_default_route([](uint32_t type, void* data, size_t size) {
        (void)size; // Unused
        if (type == FloatVector::message_type) {
            const float* floats = reinterpret_cast<const float*>(
                static_cast<const uint8_t*>(data) + sizeof(size_t));
            std::cout << "FloatVector with first element <= 5.0: " << floats[0] << std::endl;
        } else {
            std::cout << "Other message type: " << type << std::endl;
        }
    });
    
    router.start(*channel);
    
    std::cout << "\nSending FloatVectors with different first elements..." << std::endl;
    
    // Send FloatVector with first element = 3.0
    {
        FloatVector msg(*channel);
        msg.resize(5);
        msg[0] = 3.0f;
        channel->send(msg);
        std::cout << "Sent FloatVector with first element = 3.0" << std::endl;
    }
    
    // Send FloatVector with first element = 7.5
    {
        FloatVector msg(*channel);
        msg.resize(5);
        msg[0] = 7.5f;
        channel->send(msg);
        std::cout << "Sent FloatVector with first element = 7.5" << std::endl;
    }
    
    // Send FloatVector with first element = 5.0
    {
        FloatVector msg(*channel);
        msg.resize(5);
        msg[0] = 5.0f;
        channel->send(msg);
        std::cout << "Sent FloatVector with first element = 5.0" << std::endl;
    }
    
    // Send FloatVector with first element = 10.0
    {
        FloatVector msg(*channel);
        msg.resize(5);
        msg[0] = 10.0f;
        channel->send(msg);
        std::cout << "Sent FloatVector with first element = 10.0" << std::endl;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    router.stop();
    
    std::cout << "Predicate filter demo completed!" << std::endl;
}

int main() {
    try {
        std::cout << "Psyne Message Routing and Filtering Demo" << std::endl;
        std::cout << "========================================" << std::endl;
        
        demo_basic_routing();
        demo_filtered_channels();
        demo_composite_filters();
        demo_predicate_filter();
        demo_performance();
        
        std::cout << "\n✅ All routing demos completed successfully!" << std::endl;
        
        std::cout << "\nRouting features demonstrated:" << std::endl;
        std::cout << "  • Type-based message routing" << std::endl;
        std::cout << "  • Size-based filtering" << std::endl;
        std::cout << "  • Custom predicate filters" << std::endl;
        std::cout << "  • Filtered channel wrappers" << std::endl;
        std::cout << "  • Composite filters (AND/OR logic)" << std::endl;
        std::cout << "  • Default route handling" << std::endl;
        std::cout << "  • High-performance routing" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}