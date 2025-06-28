#include <psyne/psyne.hpp>
#include "../src/routing/routing.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

using namespace psyne;
using namespace psyne::routing;
using namespace psyne::types;

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
    
    // Add route for Matrix4x4f messages
    router.add_route<Matrix4x4f>([](Matrix4x4f&& msg) {
        std::cout << "Received Matrix4x4f with determinant: " << msg.determinant() << std::endl;
    });
    
    // Add route for Vector3f messages  
    router.add_route<Vector3f>([](Vector3f&& msg) {
        std::cout << "Received Vector3f: (" << msg.x() << ", " << msg.y() << ", " << msg.z() 
                  << "), magnitude: " << msg.length() << std::endl;
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
    
    // Send Matrix4x4f
    {
        Matrix4x4f msg(*channel);
        // Set to identity matrix with scaled X
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                msg(i, j) = (i == j) ? 1.0f : 0.0f;
            }
        }
        msg(0, 0) = 2.0f;  // Scale X by 2
        channel->send(msg);
    }
    
    // Send Vector3f
    {
        Vector3f msg(*channel);
        msg.x() = 3.0f;
        msg.y() = 4.0f;
        msg.z() = 0.0f;
        channel->send(msg);
    }
    
    // Send ByteVector (no specific route, should go to default)
    {
        ByteVector msg(*channel);
        msg.resize(100);
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
            Vector3f msg(*base_channel);
            base_channel->send(msg);
            std::cout << "Sent Vector3f (~12 bytes)" << std::endl;
        }
        
        // Another large message
        {
            Matrix4x4f msg(*base_channel);
            base_channel->send(msg);
            std::cout << "Sent Matrix4x4f (~64 bytes)" << std::endl;
        }
    });
    
    // Receive only large messages
    std::cout << "\nReceiving only large messages (> 100 bytes):" << std::endl;
    int received = 0;
    auto start = std::chrono::steady_clock::now();
    
    while (received < 1 && std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
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
    
    // Send Matrix4x4f (should go to default)
    {
        Matrix4x4f msg(*channel);
        channel->send(msg);
        std::cout << "Sent Matrix4x4f" << std::endl;
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
    std::atomic<int> matrix_count{0};
    std::atomic<int> vector_count{0};
    
    // Add routes with counters
    router.add_route<FloatVector>([&float_count](FloatVector&& msg) {
        (void)msg; // Unused
        float_count++;
    });
    
    router.add_route<Matrix4x4f>([&matrix_count](Matrix4x4f&& msg) {
        (void)msg; // Unused
        matrix_count++;
    });
    
    router.add_route<Vector3f>([&vector_count](Vector3f&& msg) {
        (void)msg; // Unused
        vector_count++;
    });
    
    router.start(*channel);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Send many messages
    const int messages_per_type = 10000;
    for (int i = 0; i < messages_per_type; ++i) {
        // Alternate between message types
        switch (i % 3) {
            case 0: {
                FloatVector msg(*channel);
                msg.resize(10);
                channel->send(msg);
                break;
            }
            case 1: {
                Matrix4x4f msg(*channel);
                channel->send(msg);
                break;
            }
            case 2: {
                Vector3f msg(*channel);
                channel->send(msg);
                break;
            }
        }
    }
    
    // Wait for all messages to be routed
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    router.stop();
    
    std::cout << "Routing performance results:" << std::endl;
    std::cout << "  FloatVector messages: " << float_count << std::endl;
    std::cout << "  Matrix4x4f messages: " << matrix_count << std::endl;
    std::cout << "  Vector3f messages: " << vector_count << std::endl;
    std::cout << "  Total messages: " << (float_count + matrix_count + vector_count) << std::endl;
    std::cout << "  Time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Throughput: " << (messages_per_type * 1000 / duration.count()) 
              << " messages/second" << std::endl;
    
    std::cout << "Performance demo completed!" << std::endl;
}

void demo_type_range_filter() {
    std::cout << "\n=== Type Range Filter Demo ===" << std::endl;
    
    auto channel = create_channel("memory://range_demo", 64 * 1024);
    
    MessageRouter router;
    
    // Route messages with type ID between 1-10 (includes FloatVector=1)
    router.add_route(
        std::make_unique<RangeFilter>(1, 10),
        [](uint32_t type, void* data, size_t size) {
            (void)data; // Unused
            std::cout << "Low type ID message: type=" << type 
                      << ", size=" << size << std::endl;
        });
    
    // Route messages with type ID > 100
    router.add_route(
        std::make_unique<RangeFilter>(100, UINT32_MAX),
        [](uint32_t type, void* data, size_t size) {
            (void)data; // Unused
            std::cout << "High type ID message: type=" << type 
                      << ", size=" << size << std::endl;
        });
    
    router.start(*channel);
    
    std::cout << "\nSending messages with different type IDs..." << std::endl;
    
    // Send FloatVector (type=1)
    {
        FloatVector msg(*channel);
        channel->send(msg);
        std::cout << "Sent FloatVector (type=1)" << std::endl;
    }
    
    // Send Matrix4x4f (type=103)
    {
        Matrix4x4f msg(*channel);
        channel->send(msg);
        std::cout << "Sent Matrix4x4f (type=103)" << std::endl;
    }
    
    // Send Vector3f (type=104)
    {
        Vector3f msg(*channel);
        channel->send(msg);
        std::cout << "Sent Vector3f (type=104)" << std::endl;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    router.stop();
    
    std::cout << "Type range filter demo completed!" << std::endl;
}

int main() {
    try {
        std::cout << "Psyne Message Routing and Filtering Demo" << std::endl;
        std::cout << "========================================" << std::endl;
        
        demo_basic_routing();
        demo_filtered_channels();
        demo_composite_filters();
        demo_type_range_filter();
        demo_performance();
        
        std::cout << "\n✅ All routing demos completed successfully!" << std::endl;
        
        std::cout << "\nRouting features demonstrated:" << std::endl;
        std::cout << "  • Type-based message routing" << std::endl;
        std::cout << "  • Size-based filtering" << std::endl;
        std::cout << "  • Type range filtering" << std::endl;
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