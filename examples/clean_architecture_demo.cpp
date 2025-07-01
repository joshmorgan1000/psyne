/**
 * @file clean_architecture_demo.cpp
 * @brief Demonstration of the clean composition-based architecture
 * 
 * Shows how the new organized structure works:
 * - Substrates inherit from SubstrateBase (organized by category)
 * - Patterns inherit from PatternBase (organized by category)
 * - Messages are traits (organized by category)
 * - Channel composes them via Channel<MessageType, Substrate, Pattern>
 */

#include "psyne/channel/channel_v3.hpp"
#include "psyne/global/logger.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace psyne;

/**
 * @brief Demo using numeric message types
 */
void demo_numeric_messages() {
    std::cout << "\n=== Clean Architecture Demo ===\n";
    
    // Create channel with organized components
    using VectorChannel = Channel<message::Vector256, 
                                 substrate::InProcess<message::Vector256>, 
                                 pattern::SPSC<message::Vector256, substrate::InProcess<message::Vector256>>>;
    
    auto channel = std::make_shared<VectorChannel>();
    
    std::cout << "Channel Info:\n";
    std::cout << "  Substrate: " << channel->substrate_name() << "\n";
    std::cout << "  Pattern: " << channel->pattern_name() << "\n";
    std::cout << "  Zero-copy: " << channel->is_zero_copy() << "\n";
    std::cout << "  Needs locks: " << channel->needs_locks() << "\n";
    std::cout << "  Capacity: " << channel->capacity() << " messages\n\n";
    
    // Register listener
    channel->register_listener([](message::Vector256* msg) {
        std::cout << "Received vector " << msg->sequence_id 
                  << " with norm " << msg->as_eigen().norm() << "\n";
    });
    
    // Producer thread
    std::thread producer([&channel]() {
        for (int i = 0; i < 5; ++i) {
            // Message allocated directly in channel slab
            Message<message::Vector256, 
                   substrate::InProcess<message::Vector256>, 
                   pattern::SPSC<message::Vector256, substrate::InProcess<message::Vector256>>> msg(*channel);
            
            // Fill with data directly in final location
            msg->as_eigen().setRandom();
            msg->batch_idx = i;
            msg->sequence_id = 1000 + i;
            
            std::cout << "Produced vector " << i << "\n";
            
            // Send - zero copy, just updates pointers
            msg.send();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    
    producer.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

/**
 * @brief Demo async operations with organized substrates
 */
boost::asio::awaitable<void> demo_async_operations() {
    std::cout << "\n=== Async Operations Demo ===\n";
    
    boost::asio::io_context io_context;
    
    // Create channel with async-capable components
    using MatrixChannel = Channel<message::Matrix8x8,
                                 substrate::InProcess<message::Matrix8x8>,
                                 pattern::SPSC<message::Matrix8x8, substrate::InProcess<message::Matrix8x8>>>;
    
    auto channel = std::make_shared<MatrixChannel>();
    
    // Producer coroutine
    auto producer = [&]() -> boost::asio::awaitable<void> {
        for (int i = 0; i < 3; ++i) {
            Message<message::Matrix8x8,
                   substrate::InProcess<message::Matrix8x8>,
                   pattern::SPSC<message::Matrix8x8, substrate::InProcess<message::Matrix8x8>>> msg(*channel);
            
            msg->as_eigen().setIdentity();
            msg->as_eigen() *= (i + 1);
            msg->sequence_id = 2000 + i;
            
            std::cout << "Async producing matrix " << i << "\n";
            
            // Async send using substrate
            co_await msg.async_send();
            
            boost::asio::steady_timer timer(io_context);
            timer.expires_after(std::chrono::milliseconds(50));
            co_await timer.async_wait(boost::asio::use_awaitable);
        }
    };
    
    // Consumer coroutine
    auto consumer = [&]() -> boost::asio::awaitable<void> {
        for (int i = 0; i < 3; ++i) {
            // Async receive using pattern
            auto msg = co_await channel->async_receive(io_context, std::chrono::milliseconds(1000));
            if (msg) {
                std::cout << "Async received matrix " << msg->sequence_id 
                          << " with determinant " << msg->as_eigen().determinant() << "\n";
            } else {
                std::cout << "Async receive timeout\n";
            }
        }
    };
    
    // Run both coroutines
    co_await boost::asio::co_spawn(io_context, producer(), boost::asio::use_awaitable);
    co_await boost::asio::co_spawn(io_context, consumer(), boost::asio::use_awaitable);
}

void run_async_demo() {
    boost::asio::io_context io_context;
    
    boost::asio::co_spawn(io_context, demo_async_operations(), boost::asio::detached);
    
    io_context.run();
}

/**
 * @brief Demo convenience aliases
 */
void demo_convenience_aliases() {
    std::cout << "\n=== Convenience Aliases Demo ===\n";
    
    // Using convenience aliases
    auto fast_channel = make_fast_channel<message::Vector128>();
    
    std::cout << "Fast Channel Info:\n";
    std::cout << "  Type: FastChannel<Vector128>\n";
    std::cout << "  Substrate: " << fast_channel->substrate_name() << "\n";
    std::cout << "  Pattern: " << fast_channel->pattern_name() << "\n";
    std::cout << "  Zero-copy: " << fast_channel->is_zero_copy() << "\n";
    
    // Quick message test
    fast_channel->register_listener([](message::Vector128* msg) {
        std::cout << "Fast channel received vector with mean " 
                  << msg->as_eigen().mean() << "\n";
    });
    
    Message<message::Vector128,
           substrate::InProcess<message::Vector128>,
           pattern::SPSC<message::Vector128, substrate::InProcess<message::Vector128>>> msg(*fast_channel);
    
    msg->as_eigen().setConstant(42.0f);
    msg->sequence_id = 5000;
    
    msg.send();
    
    std::cout << "Convenience aliases work perfectly!\n";
}

int main() {
    LogManager::set_level(LogLevel::INFO);
    
    std::cout << "Psyne Clean Architecture Demo\n";
    std::cout << "============================\n";
    std::cout << "Demonstrating organized, composition-based design:\n";
    std::cout << "- Substrates inherit from SubstrateBase\n";
    std::cout << "- Patterns inherit from PatternBase\n";
    std::cout << "- Messages organized by category\n";
    std::cout << "- Channel composes via templates\n";
    
    try {
        demo_numeric_messages();
        run_async_demo();
        demo_convenience_aliases();
        
        std::cout << "\nClean architecture demo completed successfully!\n";
        std::cout << "Each component is properly organized and extensible.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}