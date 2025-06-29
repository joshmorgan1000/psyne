/**
 * @file rudp.hpp
 * @brief Reliable UDP (RUDP) transport implementation
 * 
 * This provides TCP-like reliability over UDP, offering:
 * - Packet ordering and duplicate detection
 * - Automatic retransmission of lost packets
 * - Flow control and congestion control
 * - Lower latency than TCP for real-time applications
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#pragma once

#include <psyne/channel.hpp>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <queue>
#include <chrono>

namespace psyne {
namespace transport {

/**
 * @brief RUDP packet types
 */
enum class RUDPPacketType : uint8_t {
    DATA = 0,           ///< Data packet
    ACK = 1,            ///< Acknowledgment packet
    NACK = 2,           ///< Negative acknowledgment
    SYN = 3,            ///< Synchronization (connection establishment)
    FIN = 4,            ///< Finish (connection termination)
    HEARTBEAT = 5,      ///< Keep-alive heartbeat
    RESET = 6           ///< Reset connection
};

/**
 * @brief RUDP packet header
 */
struct RUDPHeader {
    uint32_t sequence_number;       ///< Packet sequence number
    uint32_t ack_number;           ///< Acknowledgment number
    uint16_t window_size;          ///< Receive window size
    uint8_t packet_type;           ///< Packet type (RUDPPacketType)
    uint8_t flags;                 ///< Control flags
    uint16_t checksum;             ///< Header and data checksum
    uint16_t payload_length;       ///< Payload length in bytes
    uint32_t timestamp;            ///< Timestamp for RTT calculation
} __attribute__((packed));

static_assert(sizeof(RUDPHeader) == 20, "RUDPHeader must be 20 bytes");

/**
 * @brief RUDP packet
 */
struct RUDPPacket {
    RUDPHeader header;
    std::vector<uint8_t> payload;
    std::chrono::steady_clock::time_point sent_time;
    uint8_t retransmit_count = 0;
    
    RUDPPacket() = default;
    RUDPPacket(RUDPPacketType type, uint32_t seq, const void* data = nullptr, size_t size = 0);
    
    void calculate_checksum();
    bool verify_checksum() const;
    std::vector<uint8_t> serialize() const;
    bool deserialize(const void* data, size_t size);
};

/**
 * @brief Connection state for RUDP
 */
enum class RUDPConnectionState {
    CLOSED,
    LISTEN,
    SYN_SENT,
    SYN_RECEIVED,
    ESTABLISHED,
    FIN_WAIT,
    CLOSE_WAIT,
    CLOSING,
    TIME_WAIT
};

/**
 * @brief RUDP connection statistics
 */
struct RUDPStats {
    uint64_t packets_sent = 0;
    uint64_t packets_received = 0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;
    uint64_t packets_retransmitted = 0;
    uint64_t packets_out_of_order = 0;
    uint64_t packets_duplicate = 0;
    uint64_t acks_sent = 0;
    uint64_t acks_received = 0;
    double rtt_ms = 0.0;
    double rtt_variance_ms = 0.0;
    uint32_t cwnd = 1;              ///< Congestion window
    uint32_t ssthresh = 65535;      ///< Slow start threshold
};

/**
 * @brief RUDP configuration
 */
struct RUDPConfig {
    uint32_t max_window_size = 8192;        ///< Maximum receive window
    uint32_t initial_timeout_ms = 1000;     ///< Initial retransmission timeout
    uint32_t max_retransmits = 5;           ///< Maximum retransmission attempts
    uint32_t heartbeat_interval_ms = 5000;  ///< Heartbeat interval
    uint32_t connection_timeout_ms = 30000; ///< Connection timeout
    bool enable_fast_retransmit = true;     ///< Enable fast retransmit on 3 duplicate ACKs
    bool enable_selective_ack = true;       ///< Enable selective acknowledgments
    bool enable_nagle = false;              ///< Enable Nagle's algorithm
};

/**
 * @brief Reliable UDP channel implementation
 */
class RUDPChannel : public Channel {
public:
    RUDPChannel(const std::string& local_address, uint16_t local_port,
                const RUDPConfig& config = {});
    ~RUDPChannel() override;
    
    // Channel interface
    size_t send(const void* data, size_t size, uint32_t type_id = 0) override;
    size_t receive(void* buffer, size_t buffer_size, uint32_t* type_id = nullptr) override;
    bool try_send(const void* data, size_t size, uint32_t type_id = 0) override;
    bool try_receive(void* buffer, size_t buffer_size, size_t* received_size = nullptr, 
                     uint32_t* type_id = nullptr) override;
    
    // RUDP-specific operations
    
    /**
     * @brief Connect to remote peer
     */
    bool connect(const std::string& remote_address, uint16_t remote_port);
    
    /**
     * @brief Listen for incoming connections
     */
    bool listen();
    
    /**
     * @brief Accept incoming connection
     */
    std::unique_ptr<RUDPChannel> accept();
    
    /**
     * @brief Close connection gracefully
     */
    void close();
    
    /**
     * @brief Check if connection is established
     */
    bool is_connected() const { return state_ == RUDPConnectionState::ESTABLISHED; }
    
    /**
     * @brief Get connection state
     */
    RUDPConnectionState get_state() const { return state_; }
    
    /**
     * @brief Get connection statistics
     */
    RUDPStats get_stats() const;
    
    /**
     * @brief Set configuration
     */
    void set_config(const RUDPConfig& config) { config_ = config; }
    
    /**
     * @brief Get current RTT estimate
     */
    double get_rtt_ms() const { return rtt_estimate_ms_; }
    
    /**
     * @brief Get congestion window size
     */
    uint32_t get_congestion_window() const { return congestion_window_; }
    
protected:
    // Core RUDP functionality
    bool send_packet(const RUDPPacket& packet);
    bool receive_packet(RUDPPacket& packet);
    void process_received_packet(const RUDPPacket& packet);
    
    // Connection management
    bool establish_connection(const std::string& remote_addr, uint16_t remote_port);
    void handle_syn_packet(const RUDPPacket& packet);
    void handle_ack_packet(const RUDPPacket& packet);
    void handle_data_packet(const RUDPPacket& packet);
    void handle_fin_packet(const RUDPPacket& packet);
    
    // Reliability mechanisms
    void send_ack(uint32_t ack_number);
    void send_nack(uint32_t nack_number);
    void retransmit_packet(uint32_t sequence_number);
    void handle_timeout();
    
    // Flow control and congestion control
    void update_congestion_window(bool packet_acked);
    void handle_congestion();
    uint32_t get_send_window() const;
    
    // RTT estimation
    void update_rtt(double sample_rtt_ms);
    uint32_t calculate_timeout() const;
    
    // Threading and event handling
    void sender_thread();
    void receiver_thread();
    void timeout_thread();
    void heartbeat_thread();
    
private:
    // Configuration
    RUDPConfig config_;
    
    // Connection state
    std::atomic<RUDPConnectionState> state_{RUDPConnectionState::CLOSED};
    std::string local_address_;
    uint16_t local_port_;
    std::string remote_address_;
    uint16_t remote_port_;
    
    // Sequence numbers
    std::atomic<uint32_t> next_sequence_number_{1};
    std::atomic<uint32_t> expected_sequence_number_{1};
    std::atomic<uint32_t> last_ack_sent_{0};
    
    // Send and receive windows
    std::atomic<uint32_t> send_window_size_;
    std::atomic<uint32_t> receive_window_size_;
    
    // Congestion control
    std::atomic<uint32_t> congestion_window_{1};
    std::atomic<uint32_t> slow_start_threshold_{65535};
    
    // RTT estimation
    std::atomic<double> rtt_estimate_ms_{1000.0};
    std::atomic<double> rtt_variance_ms_{500.0};
    
    // Packet management
    std::mutex send_buffer_mutex_;
    std::queue<RUDPPacket> send_buffer_;
    std::unordered_map<uint32_t, RUDPPacket> unacked_packets_;
    
    std::mutex receive_buffer_mutex_;
    std::queue<RUDPPacket> receive_buffer_;
    std::unordered_map<uint32_t, RUDPPacket> out_of_order_packets_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    RUDPStats stats_;
    
    // Threading
    std::thread sender_thread_;
    std::thread receiver_thread_;
    std::thread timeout_thread_;
    std::thread heartbeat_thread_;
    std::atomic<bool> running_{false};
    
    // Synchronization
    std::condition_variable send_cv_;
    std::condition_variable receive_cv_;
    
    // UDP socket (simplified - in real implementation would use actual UDP socket)
    std::unique_ptr<Channel> udp_channel_;
};

/**
 * @brief RUDP server for handling multiple connections
 */
class RUDPServer {
public:
    RUDPServer(uint16_t port, const RUDPConfig& config = {});
    ~RUDPServer();
    
    /**
     * @brief Start listening for connections
     */
    bool start();
    
    /**
     * @brief Stop server
     */
    void stop();
    
    /**
     * @brief Accept new connection
     */
    std::unique_ptr<RUDPChannel> accept();
    
    /**
     * @brief Get number of active connections
     */
    size_t get_connection_count() const;
    
    /**
     * @brief Set connection handler callback
     */
    void set_connection_handler(std::function<void(std::unique_ptr<RUDPChannel>)> handler);
    
private:
    uint16_t port_;
    RUDPConfig config_;
    std::atomic<bool> running_{false};
    
    std::mutex connections_mutex_;
    std::vector<std::unique_ptr<RUDPChannel>> connections_;
    
    std::thread acceptor_thread_;
    std::function<void(std::unique_ptr<RUDPChannel>)> connection_handler_;
    
    void acceptor_loop();
};

/**
 * @brief Factory functions for RUDP channels
 */

/**
 * @brief Create RUDP client channel
 */
std::unique_ptr<RUDPChannel> create_rudp_client(
    const std::string& remote_address,
    uint16_t remote_port,
    const RUDPConfig& config = {});

/**
 * @brief Create RUDP server
 */
std::unique_ptr<RUDPServer> create_rudp_server(
    uint16_t port,
    const RUDPConfig& config = {});

/**
 * @brief Utility functions
 */

/**
 * @brief Calculate checksum for RUDP packet
 */
uint16_t calculate_rudp_checksum(const void* data, size_t size);

/**
 * @brief Get current timestamp in milliseconds
 */
uint32_t get_timestamp_ms();

/**
 * @brief Convert RUDP state to string
 */
const char* rudp_state_to_string(RUDPConnectionState state);

} // namespace transport
} // namespace psyne