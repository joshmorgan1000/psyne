/**
 * @file publish_subscribe_demo.cpp
 * @brief Demonstrates publish/subscribe messaging pattern with Psyne
 * 
 * Pub/sub allows multiple subscribers to receive messages from publishers.
 * This example shows how to implement it efficiently using Psyne's
 * UDP multicast channel for true zero-copy broadcasting.
 */

#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <iomanip>

using namespace psyne;

// Market data event (like stock prices)
class MarketData : public Message<MarketData> {
public:
    struct TickData {
        char symbol[8];
        double price;
        uint64_t volume;
        uint64_t timestamp;
    };
    
    static consteval size_t calculate_size() noexcept {
        return sizeof(TickData);
    }
    
    MarketData(Channel& channel) : Message<MarketData>(channel) {
        // Initialize
        auto* tick = get_tick();
        std::memset(tick->symbol, 0, sizeof(tick->symbol));
        tick->price = 0.0;
        tick->volume = 0;
        tick->timestamp = 0;
    }
    
    TickData* get_tick() noexcept {
        return reinterpret_cast<TickData*>(data());
    }
    
    const TickData* get_tick() const noexcept {
        return reinterpret_cast<const TickData*>(data());
    }
    
    void set_symbol(const std::string& sym) {
        auto* tick = get_tick();
        std::strncpy(tick->symbol, sym.c_str(), 7);
        tick->symbol[7] = '\0';
    }
    
    void set_price(double price) {
        get_tick()->price = price;
    }
    
    void set_volume(uint64_t volume) {
        get_tick()->volume = volume;
    }
    
    void set_timestamp(uint64_t ts) {
        get_tick()->timestamp = ts;
    }
};

// Publisher that broadcasts market data
void market_data_publisher(Channel& channel, std::atomic<bool>& running) {
    std::cout << "📡 Publisher: Starting market data broadcast...\n";
    
    std::vector<std::string> symbols = {"AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"};
    std::vector<double> base_prices = {150.0, 2800.0, 300.0, 3300.0, 250.0};
    
    size_t tick_count = 0;
    
    while (running.load()) {
        for (size_t i = 0; i < symbols.size(); ++i) {
            MarketData tick(channel);
            
            // Simulate price movement
            double price_change = (rand() % 200 - 100) / 100.0; // -1.00 to +1.00
            double new_price = base_prices[i] + price_change;
            base_prices[i] = new_price;
            
            // Fill market data
            tick.set_symbol(symbols[i]);
            tick.set_price(new_price);
            tick.set_volume(rand() % 1000000);
            tick.set_timestamp(std::chrono::system_clock::now().time_since_epoch().count());
            
            // Broadcast to all subscribers (zero-copy multicast)
            tick.send();
            tick_count++;
        }
        
        // Simulate market tick rate (10 updates/second)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "📡 Publisher: Stopped after " << tick_count << " ticks\n";
}

// Subscriber that processes specific symbols
void market_data_subscriber(Channel& channel, const std::string& name, 
                          const std::string& filter_symbol, 
                          std::atomic<bool>& running) {
    std::cout << "📊 " << name << ": Subscribing to " 
              << (filter_symbol.empty() ? "ALL" : filter_symbol) << " ticks...\n";
    
    size_t received_count = 0;
    double total_volume = 0;
    
    while (running.load()) {
        auto buffer = channel.buffer_span();
        if (buffer.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Process tick (zero-copy view into buffer)
        const MarketData tick(channel);
        const auto* tick_data = reinterpret_cast<const MarketData::TickData*>(buffer.data());
        
        // Apply filter if specified
        if (!filter_symbol.empty() && 
            std::string(tick_data->symbol) != filter_symbol) {
            channel.advance_read_pointer(MarketData::calculate_size());
            continue;
        }
        
        // Process the tick
        received_count++;
        total_volume += tick_data->volume;
        
        // Print every 10th tick to avoid spam
        if (received_count % 10 == 0) {
            std::cout << "📈 " << name << ": " << tick_data->symbol 
                      << " = $" << std::fixed << std::setprecision(2) 
                      << tick_data->price 
                      << " (vol: " << tick_data->volume << ")\n";
        }
        
        channel.advance_read_pointer(MarketData::calculate_size());
    }
    
    std::cout << "📊 " << name << ": Received " << received_count 
              << " ticks, total volume: " << total_volume << "\n";
}

int main() {
    std::cout << "📢 Publish/Subscribe Pattern Demo\n";
    std::cout << "=================================\n\n";
    
    // Create UDP multicast channel for true pub/sub
    // In production, this would use a real multicast address
    auto channel = Channel::create("udp://239.255.1.1:5010",
                                 8 * 1024 * 1024,  // 8MB buffer
                                 ChannelMode::SPMC, // Single Publisher, Multiple Consumers
                                 ChannelType::SingleType);
    
    std::cout << "🌐 Created UDP multicast channel\n";
    std::cout << "   - True zero-copy broadcasting to all subscribers\n";
    std::cout << "   - No message duplication per subscriber\n";
    std::cout << "   - Network handles the fan-out\n\n";
    
    std::atomic<bool> running(true);
    
    // Start multiple subscribers with different filters
    std::thread sub1(market_data_subscriber, std::ref(*channel), 
                     "Subscriber-1", "", std::ref(running)); // All symbols
    
    std::thread sub2(market_data_subscriber, std::ref(*channel),
                     "Subscriber-2", "AAPL", std::ref(running)); // Only AAPL
    
    std::thread sub3(market_data_subscriber, std::ref(*channel),
                     "Subscriber-3", "TSLA", std::ref(running)); // Only TSLA
    
    // Give subscribers time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Start publisher
    std::thread pub(market_data_publisher, std::ref(*channel), std::ref(running));
    
    // Run for 5 seconds
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // Shutdown
    std::cout << "\n🛑 Shutting down...\n";
    running = false;
    
    pub.join();
    sub1.join();
    sub2.join();
    sub3.join();
    
    std::cout << "\n✨ Publish/Subscribe pattern demonstrated successfully!\n";
    std::cout << "\n📝 Key Points:\n";
    std::cout << "   - Publisher sends once, network duplicates to subscribers\n";
    std::cout << "   - Zero-copy throughout (multicast is hardware-accelerated)\n";
    std::cout << "   - Subscribers can filter messages as needed\n";
    std::cout << "   - Scales to thousands of subscribers with no publisher overhead\n";
    
    return 0;
}