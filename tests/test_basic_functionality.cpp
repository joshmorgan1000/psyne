/**
 * @file test_basic_functionality.cpp
 * @brief Basic functionality tests for the clean architecture
 */

#include "psyne/channel/channel_v3.hpp"
#include "psyne/global/logger.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

/**
 * @brief Test basic message allocation and sending
 */
void test_basic_messaging() {
    std::cout << "Testing basic messaging...\n";
    
    auto channel = make_fast_channel<message::Vector64>();
    
    bool received = false;
    channel->register_listener([&received](message::Vector64* msg) {
        assert(msg->sequence_id == 1234);
        assert(msg->as_eigen().norm() > 0);
        received = true;
        std::cout << "✓ Message received correctly\n";
    });
    
    // Test message allocation and sending
    {
        Message<message::Vector64,
               substrate::InProcess<message::Vector64>,
               pattern::SPSC<message::Vector64, substrate::InProcess<message::Vector64>>> msg(*channel);
        
        msg->as_eigen().setConstant(1.0f);
        msg->sequence_id = 1234;
        
        msg.send();
    }
    
    // Give listener time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    assert(received);
    std::cout << "✓ Basic messaging test passed\n";
}

/**
 * @brief Test channel state and capacity
 */
void test_channel_state() {
    std::cout << "Testing channel state...\n";
    
    auto channel = make_fast_channel<message::Vector128>();
    
    // Test initial state
    assert(channel->empty());
    assert(!channel->full());
    assert(channel->size() == 0);
    assert(channel->capacity() > 0);
    
    // Test channel info
    assert(std::string(channel->substrate_name()) == "InProcess");
    assert(std::string(channel->pattern_name()) == "SPSC");
    assert(channel->is_zero_copy());
    assert(!channel->needs_locks());
    assert(!channel->needs_serialization());
    
    std::cout << "✓ Channel state test passed\n";
}

/**
 * @brief Test producer-consumer pattern
 */
void test_producer_consumer() {
    std::cout << "Testing producer-consumer pattern...\n";
    
    auto channel = make_fast_channel<message::Matrix4x4>();
    
    constexpr int NUM_MESSAGES = 10;
    std::atomic<int> received_count{0};
    
    channel->register_listener([&received_count](message::Matrix4x4* msg) {
        received_count++;
    });
    
    // Producer thread
    std::thread producer([&channel]() {
        for (int i = 0; i < NUM_MESSAGES; ++i) {
            Message<message::Matrix4x4,
                   substrate::InProcess<message::Matrix4x4>,
                   pattern::SPSC<message::Matrix4x4, substrate::InProcess<message::Matrix4x4>>> msg(*channel);
            
            msg->as_eigen().setIdentity();
            msg->as_eigen() *= (i + 1);
            msg->sequence_id = i;
            
            msg.send();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    producer.join();
    
    // Wait for all messages to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    assert(received_count == NUM_MESSAGES);
    std::cout << "✓ Producer-consumer test passed\n";
}

/**
 * @brief Test async operations
 */
void test_async_operations() {
    std::cout << "Testing async operations...\n";
    
    boost::asio::io_context io_context;
    auto channel = make_fast_channel<message::Vector256>();
    
    bool async_test_passed = false;
    
    auto test_coroutine = [&]() -> boost::asio::awaitable<void> {
        try {
            // Send a message
            {
                Message<message::Vector256,
                       substrate::InProcess<message::Vector256>,
                       pattern::SPSC<message::Vector256, substrate::InProcess<message::Vector256>>> msg(*channel);
                
                msg->as_eigen().setConstant(42.0f);
                msg->sequence_id = 9999;
                
                co_await msg.async_send();
            }
            
            // Try to receive it
            auto received_msg = co_await channel->async_receive(io_context, std::chrono::milliseconds(100));
            
            if (received_msg && received_msg->sequence_id == 9999) {
                async_test_passed = true;
                std::cout << "✓ Async message sent and received correctly\n";
            }
            
        } catch (const std::exception& e) {
            std::cout << "Async test error: " << e.what() << "\n";
        }
    };
    
    boost::asio::co_spawn(io_context, test_coroutine(), boost::asio::detached);
    io_context.run();
    
    assert(async_test_passed);
    std::cout << "✓ Async operations test passed\n";
}

/**
 * @brief Test concepts enforcement
 */
void test_concepts() {
    std::cout << "Testing concepts enforcement...\n";
    
    // These should compile (valid configurations)
    static_assert(concepts::MessageType<message::Vector64>);
    static_assert(concepts::SubstrateType<substrate::InProcess<message::Vector64>, message::Vector64>);
    static_assert(concepts::PatternType<pattern::SPSC<message::Vector64, substrate::InProcess<message::Vector64>>, 
                                      message::Vector64, 
                                      substrate::InProcess<message::Vector64>>);
    
    // Test complete channel configuration
    static_assert(concepts::ChannelConfiguration<message::Vector64, 
                                               substrate::InProcess<message::Vector64>,
                                               pattern::SPSC<message::Vector64, substrate::InProcess<message::Vector64>>>);
    
    std::cout << "✓ Concepts enforcement test passed\n";
}

/**
 * @brief Performance smoke test
 */
void test_performance() {
    std::cout << "Testing performance...\n";
    
    auto channel = make_fast_channel<message::Vector512>();
    
    constexpr int NUM_MESSAGES = 10000;
    std::atomic<int> received_count{0};
    
    channel->register_listener([&received_count](message::Vector512* msg) {
        received_count++;
    });
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Send messages as fast as possible
    for (int i = 0; i < NUM_MESSAGES; ++i) {
        Message<message::Vector512,
               substrate::InProcess<message::Vector512>,
               pattern::SPSC<message::Vector512, substrate::InProcess<message::Vector512>>> msg(*channel);
        
        msg->as_eigen().setConstant(i);
        msg->sequence_id = i;
        
        msg.send();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    double throughput = (NUM_MESSAGES * 1000000.0) / duration.count();
    
    std::cout << "Performance Results:\n";
    std::cout << "  Messages: " << NUM_MESSAGES << "\n";
    std::cout << "  Time: " << duration.count() << " μs\n";
    std::cout << "  Throughput: " << throughput << " msgs/sec\n";
    std::cout << "  Received: " << received_count.load() << "/" << NUM_MESSAGES << "\n";
    
    assert(received_count == NUM_MESSAGES);
    assert(throughput > 100000); // Should be able to do 100k+ msgs/sec
    
    std::cout << "✓ Performance test passed\n";
}

int main() {
    LogManager::set_level(LogLevel::INFO);
    
    std::cout << "Psyne Basic Functionality Tests\n";
    std::cout << "===============================\n";
    
    try {
        test_concepts();
        test_channel_state();
        test_basic_messaging();
        test_producer_consumer();
        test_async_operations();
        test_performance();
        
        std::cout << "\n🎉 ALL TESTS PASSED! 🎉\n";
        std::cout << "The clean architecture is working perfectly!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}