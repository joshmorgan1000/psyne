#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <cstring>

using namespace psyne;

// Simple counter message
class CounterMessage : public Message<CounterMessage> {
public:
    static constexpr uint32_t message_type = 200;
    static constexpr size_t size = sizeof(uint64_t);
    
    template<typename Channel>
    explicit CounterMessage(Channel& channel) : Message<CounterMessage>(channel) {
        if (this->data_) {
            set_count(0);
        }
    }
    
    explicit CounterMessage(const void* data, size_t size) 
        : Message<CounterMessage>(data, size) {}
    
    static constexpr size_t calculate_size() { 
        return size; 
    }
    
    void set_count(uint64_t count) {
        if (data_) {
            *reinterpret_cast<uint64_t*>(data_) = count;
        }
    }
    
    uint64_t get_count() const {
        if (!data_) return 0;
        return *reinterpret_cast<const uint64_t*>(data_);
    }
    
    void before_send() {}
};

void demo_memory_channel() {
    std::cout << "\n=== Memory Channel Demo ===" << std::endl;
    
    // Create in-memory channel using factory
    auto channel = ChannelFactory::create<SPSCRingBuffer>(
        "memory://demo", 
        64 * 1024,
        ChannelType::SingleType
    );
    
    // Producer thread
    std::thread producer([&channel]() {
        for (uint64_t i = 1; i <= 5; ++i) {
            CounterMessage msg(*channel);
            msg.set_count(i);
            msg.send();
            std::cout << "Sent count: " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    
    // Consumer thread
    std::thread consumer([&channel]() {
        for (int i = 0; i < 5; ++i) {
            auto msg = channel->receive_single<CounterMessage>(
                std::chrono::milliseconds(1000)
            );
            if (msg) {
                std::cout << "Received count: " << msg->get_count() << std::endl;
            }
        }
    });
    
    producer.join();
    consumer.join();
}

void demo_ipc_channel() {
    std::cout << "\n=== IPC Channel Demo ===" << std::endl;
    std::cout << "Note: Run another instance of this program with --ipc-consumer flag" << std::endl;
    
    // Create IPC channel using factory
    auto channel = ChannelFactory::create<SPSCRingBuffer>(
        "ipc://factory_demo", 
        64 * 1024,
        ChannelType::SingleType,
        true  // create new
    );
    
    // Send a few messages
    for (uint64_t i = 100; i <= 103; ++i) {
        CounterMessage msg(*channel);
        msg.set_count(i);
        msg.send();
        std::cout << "Sent to IPC: " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void demo_ipc_consumer() {
    std::cout << "\n=== IPC Channel Consumer ===" << std::endl;
    
    // Open existing IPC channel
    auto channel = ChannelFactory::create<SPSCRingBuffer>(
        "ipc://factory_demo", 
        64 * 1024,
        ChannelType::SingleType,
        false  // open existing
    );
    
    // Receive messages
    std::cout << "Waiting for messages..." << std::endl;
    while (true) {
        auto msg = channel->receive_single<CounterMessage>(
            std::chrono::milliseconds(1000)
        );
        if (msg) {
            std::cout << "Received from IPC: " << msg->get_count() << std::endl;
        }
    }
}

void demo_tcp_channel() {
    std::cout << "\n=== TCP Channel Demo ===" << std::endl;
    std::cout << "Note: This creates a TCP server on port 9998" << std::endl;
    
    // Create TCP server channel using factory
    auto server = ChannelFactory::create<SPSCRingBuffer>(
        "tcp://0.0.0.0:9998", 
        64 * 1024,
        ChannelType::SingleType
    );
    
    std::cout << "TCP server created, waiting for connections..." << std::endl;
    
    // In a real application, you'd have client connections here
    // For demo purposes, just show that it was created successfully
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--ipc-consumer") {
            demo_ipc_consumer();
        } else {
            std::cout << "Channel Factory Demo" << std::endl;
            std::cout << "===================" << std::endl;
            
            // Demo different channel types
            demo_memory_channel();
            demo_ipc_channel();
            demo_tcp_channel();
            
            std::cout << "\nAll demos completed!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}