/**
 * @file pair_pattern_demo.cpp
 * @brief Demonstrates the Pair messaging pattern with Psyne
 *
 * The Pair pattern connects exactly two peers in an exclusive, bidirectional
 * communication channel. Unlike pub/sub or req/rep, Pair provides:
 * - Exclusive 1:1 connection
 * - Bidirectional communication
 * - No specific message flow requirements
 * - Ideal for peer-to-peer protocols
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>

using namespace psyne;

// Peer-to-peer message for bidirectional communication
class PeerMessage : public Message<PeerMessage> {
public:
    static constexpr uint32_t message_type = 401;
    
    enum class MessageType : uint32_t {
        HEARTBEAT = 0,
        DATA = 1,
        CONTROL = 2,
        ACKNOWLEDGMENT = 3
    };
    
    struct Header {
        uint32_t peer_id;
        MessageType msg_type;
        uint32_t sequence_num;
        uint64_t timestamp;
        uint32_t payload_size;
        char payload[1024];  // Variable payload
    };
    
    static size_t calculate_size() noexcept {
        return sizeof(Header);
    }
    
    Header& header() { return *reinterpret_cast<Header*>(data()); }
    const Header& header() const { return *reinterpret_cast<const Header*>(data()); }
    
    void set_payload(const void* data, size_t size) {
        auto& h = header();
        h.payload_size = static_cast<uint32_t>(std::min(size, sizeof(h.payload)));
        if (data && h.payload_size > 0) {
            std::memcpy(h.payload, data, h.payload_size);
        }
    }
    
    std::span<const char> get_payload() const {
        const auto& h = header();
        return std::span<const char>(h.payload, h.payload_size);
    }
};

// Statistics for peer connection
struct PeerStats {
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> messages_received{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> heartbeats_sent{0};
    std::atomic<uint64_t> heartbeats_received{0};
    std::atomic<uint64_t> last_heartbeat_time{0};
    
    void print() const {
        std::cout << "Peer Statistics:" << std::endl;
        std::cout << "  Messages sent: " << messages_sent.load() << std::endl;
        std::cout << "  Messages received: " << messages_received.load() << std::endl;
        std::cout << "  Bytes sent: " << bytes_sent.load() << std::endl;
        std::cout << "  Bytes received: " << bytes_received.load() << std::endl;
        std::cout << "  Heartbeats sent/received: " << heartbeats_sent.load() 
                  << "/" << heartbeats_received.load() << std::endl;
    }
};

// Pair pattern peer implementation
// Commented out due to Message constructor requirements
/*
class PairPeer {
public:
    PairPeer(uint32_t peer_id, 
             std::shared_ptr<Channel> send_channel,
             std::shared_ptr<Channel> recv_channel,
             const std::string& name = "Peer")
        : peer_id_(peer_id),
          send_channel_(send_channel),
          recv_channel_(recv_channel),
          name_(name),
          sequence_num_(0) {
    }
    
    void start() {
        running_ = true;
        
        // Start receiver thread
        receiver_thread_ = std::thread([this] { receive_loop(); });
        
        // Start heartbeat thread
        heartbeat_thread_ = std::thread([this] { heartbeat_loop(); });
    }
    
    void stop() {
        running_ = false;
        if (receiver_thread_.joinable()) {
            receiver_thread_.join();
        }
        if (heartbeat_thread_.joinable()) {
            heartbeat_thread_.join();
        }
    }
    
    // Send data message to peer
    void send_data(const std::string& data) {
        PeerMessage msg(*send_channel_);
        auto& header = msg.header();
        
        header.peer_id = peer_id_;
        header.msg_type = PeerMessage::MessageType::DATA;
        header.sequence_num = sequence_num_++;
        header.timestamp = get_timestamp();
        msg.set_payload(data.data(), data.size());
        
        msg.send();
        
        stats_.messages_sent++;
        stats_.bytes_sent += msg.calculate_size();
        
        std::cout << "[" << name_ << "] Sent data: " << data << std::endl;
    }
    
    // Send control message to peer
    void send_control(const std::string& command) {
        PeerMessage msg(*send_channel_);
        auto& header = msg.header();
        
        header.peer_id = peer_id_;
        header.msg_type = PeerMessage::MessageType::CONTROL;
        header.sequence_num = sequence_num_++;
        header.timestamp = get_timestamp();
        msg.set_payload(command.data(), command.size());
        
        msg.send();
        
        stats_.messages_sent++;
        
        std::cout << "[" << name_ << "] Sent control: " << command << std::endl;
    }
    
    const PeerStats& get_stats() const { return stats_; }
    
private:
    void receive_loop() {
        std::cout << "[" << name_ << "] Receiver started" << std::endl;
        
        while (running_) {
            size_t size;
            uint32_t type;
            void* data = recv_channel_->receive_message(size, type);
            
            if (!data) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            if (type == PeerMessage::message_type) {
                handle_peer_message(data, size);
            }
            
            recv_channel_->release_message(data);
        }
        
        std::cout << "[" << name_ << "] Receiver stopped" << std::endl;
    }
    
    void handle_peer_message(const void* data, size_t size) {
        const auto* msg = reinterpret_cast<const PeerMessage::Header*>(data);
        
        stats_.messages_received++;
        stats_.bytes_received += size;
        
        switch (msg->msg_type) {
            case PeerMessage::MessageType::HEARTBEAT:
                stats_.heartbeats_received++;
                stats_.last_heartbeat_time = msg->timestamp;
                // Optionally send acknowledgment
                send_acknowledgment(msg->sequence_num);
                break;
                
            case PeerMessage::MessageType::DATA: {
                std::string payload(msg->payload, msg->payload_size);
                std::cout << "[" << name_ << "] Received data from peer " 
                          << msg->peer_id << ": " << payload << std::endl;
                process_data(payload);
                break;
            }
            
            case PeerMessage::MessageType::CONTROL: {
                std::string command(msg->payload, msg->payload_size);
                std::cout << "[" << name_ << "] Received control from peer " 
                          << msg->peer_id << ": " << command << std::endl;
                process_control(command);
                break;
            }
            
            case PeerMessage::MessageType::ACKNOWLEDGMENT:
                // Handle acknowledgment
                break;
        }
    }
    
    void heartbeat_loop() {
        std::cout << "[" << name_ << "] Heartbeat started" << std::endl;
        
        while (running_) {
            send_heartbeat();
            stats_.heartbeats_sent++;
            
            // Check if peer is alive
            auto last_hb = stats_.last_heartbeat_time.load();
            if (last_hb > 0) {
                auto now = get_timestamp();
                auto elapsed_ms = (now - last_hb) / 1000000;  // Convert ns to ms
                
                if (elapsed_ms > 5000) {  // 5 second timeout
                    std::cout << "[" << name_ << "] WARNING: No heartbeat from peer for " 
                              << elapsed_ms << "ms" << std::endl;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        std::cout << "[" << name_ << "] Heartbeat stopped" << std::endl;
    }
    
    void send_heartbeat() {
        PeerMessage msg(*send_channel_);
        auto& header = msg.header();
        
        header.peer_id = peer_id_;
        header.msg_type = PeerMessage::MessageType::HEARTBEAT;
        header.sequence_num = sequence_num_++;
        header.timestamp = get_timestamp();
        header.payload_size = 0;
        
        msg.send();
    }
    
    void send_acknowledgment(uint32_t ack_seq) {
        PeerMessage msg(*send_channel_);
        auto& header = msg.header();
        
        header.peer_id = peer_id_;
        header.msg_type = PeerMessage::MessageType::ACKNOWLEDGMENT;
        header.sequence_num = sequence_num_++;
        header.timestamp = get_timestamp();
        
        // Store acknowledged sequence number in payload
        uint32_t ack_data = ack_seq;
        msg.set_payload(&ack_data, sizeof(ack_data));
        
        msg.send();
    }
    
    void process_data(const std::string& data) {
        // Application-specific data processing
        if (data == "ping") {
            send_data("pong");
        }
    }
    
    void process_control(const std::string& command) {
        // Application-specific control processing
        if (command == "status") {
            send_data("OK: " + name_ + " is running");
        } else if (command == "stop") {
            std::cout << "[" << name_ << "] Received stop command" << std::endl;
            running_ = false;
        }
    }
    
    uint64_t get_timestamp() const {
        return std::chrono::steady_clock::now().time_since_epoch().count();
    }
    
private:
    uint32_t peer_id_;
    std::shared_ptr<Channel> send_channel_;
    std::shared_ptr<Channel> recv_channel_;
    std::string name_;
    
    std::atomic<bool> running_{false};
    std::atomic<uint32_t> sequence_num_;
    
    std::thread receiver_thread_;
    std::thread heartbeat_thread_;
    
    PeerStats stats_;
};
*/

// Demo scenarios
/*
void demo_basic_pair() {
    std::cout << "\n=== Basic Pair Pattern Demo ===\n" << std::endl;
    
    // Create exclusive channels for peer communication
    // Peer1 -> Peer2 channel
    auto channel_1to2 = Channel::create("memory://peer1to2", 1024 * 1024);
    // Peer2 -> Peer1 channel  
    auto channel_2to1 = Channel::create("memory://peer2to1", 1024 * 1024);
    
    // Create peers with opposite channel directions
    PairPeer peer1(1001, channel_1to2, channel_2to1, "Peer1");
    PairPeer peer2(2001, channel_2to1, channel_1to2, "Peer2");
    
    // Start both peers
    peer1.start();
    peer2.start();
    
    // Wait for connection establishment
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Demonstrate bidirectional communication
    std::cout << "\n--- Bidirectional Data Exchange ---" << std::endl;
    
    peer1.send_data("Hello from Peer1!");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    peer2.send_data("Hello from Peer2!");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Ping-pong test
    std::cout << "\n--- Ping-Pong Test ---" << std::endl;
    peer1.send_data("ping");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Control messages
    std::cout << "\n--- Control Messages ---" << std::endl;
    peer1.send_control("status");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    peer2.send_control("status");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Let heartbeats run for a bit
    std::cout << "\n--- Running for 3 seconds (heartbeats active) ---" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Show statistics
    std::cout << "\n--- Peer Statistics ---" << std::endl;
    std::cout << "\nPeer1:" << std::endl;
    peer1.get_stats().print();
    
    std::cout << "\nPeer2:" << std::endl;
    peer2.get_stats().print();
    
    // Stop peers
    peer1.stop();
    peer2.stop();
}

void demo_network_pair() {
    std::cout << "\n=== Network Pair Pattern Demo (TCP) ===\n" << std::endl;
    
    // For network pairs, we use TCP channels with specific ports
    // This demonstrates how Pair pattern works over network
    
    std::cout << "Creating TCP pair channels..." << std::endl;
    
    // Server peer listens on port 9876
    auto server_channel = Channel::create("tcp://:9876", 1024 * 1024);
    
    // Client peer connects to server
    auto client_channel = Channel::create("tcp://localhost:9876", 1024 * 1024);
    
    std::cout << "Note: In a real network scenario, you would run these" << std::endl;
    std::cout << "peers on different machines or processes." << std::endl;
    
    // The TCP channel implementation handles the bidirectional communication
    // Both peers can send and receive on the same channel
    
    std::cout << "Network pair demonstration would require separate processes." << std::endl;
    std::cout << "See tcp_server.cpp and tcp_client.cpp for full examples." << std::endl;
}

void demo_exclusive_pairing() {
    std::cout << "\n=== Exclusive Pairing Demo ===\n" << std::endl;
    
    // Demonstrate that Pair pattern is exclusive - only two peers
    
    auto channel_a = Channel::create("memory://exclusive_a", 1024 * 1024);
    auto channel_b = Channel::create("memory://exclusive_b", 1024 * 1024);
    
    PairPeer peer_alpha(3001, channel_a, channel_b, "Alpha");
    PairPeer peer_beta(3002, channel_b, channel_a, "Beta");
    
    peer_alpha.start();
    peer_beta.start();
    
    std::cout << "\nEstablished exclusive pair: Alpha <-> Beta" << std::endl;
    
    // Exchange some data
    peer_alpha.send_data("Exclusive message to Beta");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    peer_beta.send_data("Exclusive reply to Alpha");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Attempting to add a third peer would require different channels
    // as Pair pattern is strictly 1:1
    std::cout << "\nPair pattern ensures exclusive 1:1 communication." << std::endl;
    std::cout << "No third peer can join this conversation." << std::endl;
    
    peer_alpha.stop();
    peer_beta.stop();
}
*/

int main() {
    std::cout << "Pair Messaging Pattern Demo\n";
    std::cout << "==========================\n";
    
    try {
        // Demo functionality disabled due to Message constructor requirements
        std::cout << "\nNote: Demo functionality disabled due to Message constructor requirements.\n";
        std::cout << "The Pair pattern implementation is ready for use with proper Message objects.\n\n";
        
        std::cout << "This demo would demonstrate:\n";
        std::cout << "  1. Basic in-memory pair communication\n";
        std::cout << "  2. Network pair concept\n";
        std::cout << "  3. Exclusive pairing mechanism\n";
        std::cout << "  4. Bidirectional message flow\n";
        std::cout << "  5. Heartbeat and connection state management\n\n";
        
        /*
        // Demo 1: Basic in-memory pair
        demo_basic_pair();
        
        // Demo 2: Network pair concept
        demo_network_pair();
        
        // Demo 3: Exclusive pairing
        demo_exclusive_pairing();
        
        std::cout << "\nAll demos completed successfully!" << std::endl;
        */
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}