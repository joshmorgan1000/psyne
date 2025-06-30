/**
 * @file quic.hpp
 * @brief QUIC transport protocol implementation
 *
 * This provides a modern, secure, and efficient transport protocol offering:
 * - Built-in TLS 1.3 encryption
 * - Connection migration and 0-RTT resumption
 * - Stream multiplexing without head-of-line blocking
 * - Congestion control and loss recovery
 * - Used by HTTP/3 and modern applications
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include "../channel/channel_impl.hpp"
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace psyne {
namespace transport {

/**
 * @brief QUIC frame types
 */
enum class QUICFrameType : uint8_t {
    PADDING = 0x00,
    PING = 0x01,
    ACK = 0x02,
    RESET_STREAM = 0x04,
    STOP_SENDING = 0x05,
    CRYPTO = 0x06,
    NEW_TOKEN = 0x07,
    STREAM = 0x08, // Stream frames (0x08-0x0f with flags)
    MAX_DATA = 0x10,
    MAX_STREAM_DATA = 0x11,
    MAX_STREAMS = 0x12,
    DATA_BLOCKED = 0x14,
    STREAM_DATA_BLOCKED = 0x15,
    STREAMS_BLOCKED = 0x16,
    NEW_CONNECTION_ID = 0x18,
    RETIRE_CONNECTION_ID = 0x19,
    PATH_CHALLENGE = 0x1a,
    PATH_RESPONSE = 0x1b,
    CONNECTION_CLOSE = 0x1c
};

/**
 * @brief QUIC connection state
 */
enum class QUICConnectionState {
    INITIAL,
    HANDSHAKE,
    ESTABLISHED,
    CLOSING,
    CLOSED,
    DRAINING
};

/**
 * @brief QUIC encryption level
 */
enum class QUICEncryptionLevel { INITIAL, EARLY_DATA, HANDSHAKE, APPLICATION };

/**
 * @brief QUIC stream direction
 */
enum class QUICStreamDirection { BIDIRECTIONAL, UNIDIRECTIONAL };

/**
 * @brief QUIC stream state
 */
enum class QUICStreamState {
    OPEN,
    HALF_CLOSED_LOCAL,
    HALF_CLOSED_REMOTE,
    CLOSED,
    RESET
};

/**
 * @brief QUIC configuration
 */
struct QUICConfig {
    // Connection parameters
    uint64_t max_idle_timeout_ms = 30000;      ///< Maximum idle timeout
    uint64_t max_udp_payload_size = 1472;      ///< Maximum UDP payload size
    uint64_t initial_max_data = 1048576;       ///< Initial flow control limit
    uint64_t initial_max_stream_data = 262144; ///< Initial stream flow control
    uint32_t initial_max_streams_bidi = 100; ///< Initial bidirectional streams
    uint32_t initial_max_streams_uni = 100;  ///< Initial unidirectional streams

    // Performance tuning
    bool enable_0rtt = true;         ///< Enable 0-RTT resumption
    bool enable_migration = true;    ///< Enable connection migration
    uint32_t ack_delay_exponent = 3; ///< ACK delay exponent
    uint32_t max_ack_delay_ms = 25;  ///< Maximum ACK delay

    // Security
    std::string server_name;                 ///< SNI server name
    std::vector<std::string> alpn_protocols; ///< ALPN protocol list
    std::string cert_file;                   ///< Certificate file path
    std::string key_file;                    ///< Private key file path

    // Congestion control
    std::string congestion_control = "cubic"; ///< Congestion control algorithm
    uint64_t initial_congestion_window = 10;  ///< Initial congestion window
    uint64_t minimum_congestion_window = 2;   ///< Minimum congestion window
};

/**
 * @brief QUIC connection statistics
 */
struct QUICStats {
    uint64_t packets_sent = 0;
    uint64_t packets_received = 0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;
    uint64_t packets_lost = 0;
    uint64_t packets_retransmitted = 0;
    uint64_t streams_created = 0;
    uint64_t streams_closed = 0;
    double rtt_ms = 0.0;
    double rtt_variance_ms = 0.0;
    uint64_t congestion_window = 0;
    uint32_t connection_migrations = 0;
    bool using_0rtt = false;
};

/**
 * @brief QUIC stream for application data
 */
class QUICStream {
public:
    QUICStream(uint64_t stream_id, QUICStreamDirection direction);
    ~QUICStream();

    // Stream operations
    ssize_t send(const void *data, size_t size);
    ssize_t receive(void *buffer, size_t buffer_size);
    bool try_send(const void *data, size_t size);
    bool try_receive(void *buffer, size_t buffer_size,
                     size_t *received_size = nullptr);

    // Stream control
    void close();
    void reset(uint64_t error_code = 0);

    // Stream properties
    uint64_t id() const {
        return stream_id_;
    }
    QUICStreamDirection direction() const {
        return direction_;
    }
    QUICStreamState state() const {
        return state_;
    }
    bool is_readable() const;
    bool is_writable() const;

    // Flow control
    void set_max_stream_data(uint64_t max_data);
    uint64_t get_available_send_window() const;
    uint64_t get_available_receive_window() const;

private:
    uint64_t stream_id_;
    QUICStreamDirection direction_;
    std::atomic<QUICStreamState> state_{QUICStreamState::OPEN};

    // Flow control
    std::atomic<uint64_t> max_stream_data_send_{0};
    std::atomic<uint64_t> max_stream_data_recv_{0};
    std::atomic<uint64_t> stream_data_sent_{0};
    std::atomic<uint64_t> stream_data_received_{0};

    // Buffering
    std::mutex send_mutex_;
    std::mutex recv_mutex_;
    std::queue<std::vector<uint8_t>> send_queue_;
    std::queue<std::vector<uint8_t>> recv_queue_;

    // Stream finished indicators
    std::atomic<bool> send_finished_{false};
    std::atomic<bool> recv_finished_{false};
};

/**
 * @brief QUIC connection
 */
class QUICConnection {
public:
    QUICConnection(bool is_server, const QUICConfig &config = {});
    ~QUICConnection();

    // Connection management
    bool connect(const std::string &remote_address, uint16_t remote_port);
    bool listen(uint16_t local_port);
    void close(uint64_t error_code = 0, const std::string &reason = "");

    // Stream management
    std::shared_ptr<QUICStream> create_stream(
        QUICStreamDirection direction = QUICStreamDirection::BIDIRECTIONAL);
    std::shared_ptr<QUICStream> accept_stream();
    std::shared_ptr<QUICStream> get_stream(uint64_t stream_id);

    // Connection properties
    QUICConnectionState state() const {
        return state_;
    }
    
    // Allow QUICServer to set connection state (friend access)
    void set_state(QUICConnectionState new_state) {
        state_ = new_state;
    }
    bool is_connected() const {
        return state_ == QUICConnectionState::ESTABLISHED;
    }
    std::string remote_address() const {
        return remote_address_;
    }
    uint16_t remote_port() const {
        return remote_port_;
    }

    // Connection features
    bool supports_0rtt() const;
    bool supports_migration() const;
    void migrate_to_path(const std::string &new_local_address,
                         const std::string &new_remote_address);

    // Statistics
    QUICStats get_stats() const;

    // Event callbacks
    void set_stream_callback(
        std::function<void(std::shared_ptr<QUICStream>)> callback);
    void set_connection_close_callback(
        std::function<void(uint64_t error_code, const std::string &reason)>
            callback);

protected:
    // Core QUIC functionality
    bool send_packet(const std::vector<uint8_t> &packet);
    bool receive_packet(std::vector<uint8_t> &packet);
    void process_packet(const std::vector<uint8_t> &packet);

    // Handshake
    bool perform_handshake();
    void handle_crypto_frame(const std::vector<uint8_t> &crypto_data);
    void send_crypto_frame(const std::vector<uint8_t> &crypto_data);

    // Stream management
    void handle_stream_frame(uint64_t stream_id,
                             const std::vector<uint8_t> &data, bool fin);
    void send_stream_frame(uint64_t stream_id, const std::vector<uint8_t> &data,
                           bool fin = false);

    // Flow control
    void update_connection_flow_control(uint64_t consumed_bytes);
    void send_max_data_frame();
    void handle_max_data_frame(uint64_t max_data);

    // Loss detection and recovery
    void detect_lost_packets();
    void retransmit_lost_packets();
    void update_rtt(std::chrono::microseconds sample_rtt);

    // Congestion control
    void on_packet_sent(uint64_t bytes_sent);
    void on_packet_acked(uint64_t bytes_acked);
    void on_packet_lost(uint64_t bytes_lost);

    // Threading
    void connection_loop();

private:
    // Configuration
    QUICConfig config_;
    bool is_server_;

    // Connection state
    std::atomic<QUICConnectionState> state_{QUICConnectionState::INITIAL};
    std::string local_address_;
    uint16_t local_port_ = 0;
    std::string remote_address_;
    uint16_t remote_port_ = 0;

    // Connection IDs
    std::vector<uint8_t> local_connection_id_;
    std::vector<uint8_t> remote_connection_id_;

    // Streams
    std::mutex streams_mutex_;
    std::unordered_map<uint64_t, std::shared_ptr<QUICStream>> streams_;
    std::atomic<uint64_t> next_stream_id_{0};
    std::queue<std::shared_ptr<QUICStream>> incoming_streams_;

    // Flow control
    std::atomic<uint64_t> max_data_send_{0};
    std::atomic<uint64_t> max_data_recv_{0};
    std::atomic<uint64_t> data_sent_{0};
    std::atomic<uint64_t> data_received_{0};

    // Packet handling
    std::mutex send_mutex_;
    std::mutex recv_mutex_;
    std::queue<std::vector<uint8_t>> send_queue_;
    std::queue<std::vector<uint8_t>> recv_queue_;

    // Loss detection
    std::unordered_map<uint64_t, std::chrono::steady_clock::time_point>
        sent_packets_;
    std::atomic<uint64_t> largest_acked_packet_{0};

    // RTT measurement
    std::atomic<double> smoothed_rtt_ms_{333.0}; // Initial estimate
    std::atomic<double> rtt_variance_ms_{166.5}; // Initial variance
    std::atomic<double> latest_rtt_ms_{0.0};

    // Congestion control
    std::atomic<uint64_t> congestion_window_;
    std::atomic<uint64_t> bytes_in_flight_{0};
    std::atomic<uint64_t> slow_start_threshold_{UINT64_MAX};

    // Statistics
    mutable std::mutex stats_mutex_;
    QUICStats stats_;

    // Threading
    std::thread connection_thread_;
    std::atomic<bool> running_{false};

    // Callbacks
    std::function<void(std::shared_ptr<QUICStream>)> stream_callback_;
    std::function<void(uint64_t, const std::string &)> close_callback_;

    // Synchronization
    std::condition_variable connection_cv_;
    std::condition_variable stream_cv_;

    // Underlying transport (UDP channel)
    std::unique_ptr<Channel> udp_channel_;
};

/**
 * @brief QUIC client for creating connections
 */
class QUICClient {
public:
    QUICClient(const QUICConfig &config = {});
    ~QUICClient();
    
    /**
     * @brief Connect to a QUIC server
     */
    std::shared_ptr<QUICConnection> connect(const std::string &host, uint16_t port);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief QUIC server for handling multiple connections
 */
class QUICServer {
public:
    QUICServer(uint16_t port, const QUICConfig &config = {});
    ~QUICServer();

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
    std::shared_ptr<QUICConnection> accept();

    /**
     * @brief Get number of active connections
     */
    size_t get_connection_count() const;

    /**
     * @brief Set connection handler callback
     */
    void set_connection_handler(
        std::function<void(std::shared_ptr<QUICConnection>)> handler);

private:
    uint16_t port_;
    QUICConfig config_;
    std::atomic<bool> running_{false};

    mutable std::mutex connections_mutex_;
    std::vector<std::shared_ptr<QUICConnection>> connections_;

    std::thread acceptor_thread_;
    std::function<void(std::shared_ptr<QUICConnection>)> connection_handler_;

    void acceptor_loop();
};

/**
 * @brief QUIC channel adapter for psyne Channel interface
 */
class QUICChannel {
public:
    QUICChannel(std::shared_ptr<QUICConnection> connection);
    ~QUICChannel();

    // Channel interface
    size_t send(const void *data, size_t size, uint32_t type_id = 0);
    size_t receive(void *buffer, size_t buffer_size,
                   uint32_t *type_id = nullptr);
    bool try_send(const void *data, size_t size, uint32_t type_id = 0);
    bool try_receive(void *buffer, size_t buffer_size,
                     size_t *received_size = nullptr,
                     uint32_t *type_id = nullptr);

    // QUIC-specific access
    std::shared_ptr<QUICConnection> connection() const {
        return connection_;
    }
    std::shared_ptr<QUICStream> stream() const {
        return stream_;
    }

private:
    std::shared_ptr<QUICConnection> connection_;
    std::shared_ptr<QUICStream> stream_;
};

/**
 * @brief Factory functions for QUIC
 */

/**
 * @brief Create QUIC client connection
 */
std::shared_ptr<QUICConnection>
create_quic_client(const std::string &remote_address, uint16_t remote_port,
                   const QUICConfig &config = {});

/**
 * @brief Create QUIC server
 */
std::unique_ptr<QUICServer> create_quic_server(uint16_t port,
                                               const QUICConfig &config = {});

/**
 * @brief Create QUIC channel (high-level interface)
 */
std::unique_ptr<QUICChannel>
create_quic_channel(const std::string &remote_address, uint16_t remote_port,
                    const QUICConfig &config = {});

/**
 * @brief Utility functions
 */

/**
 * @brief Generate random connection ID
 */
std::vector<uint8_t> generate_connection_id(size_t length = 8);

/**
 * @brief Encode variable-length integer (QUIC format)
 */
std::vector<uint8_t> encode_varint(uint64_t value);

/**
 * @brief Decode variable-length integer (QUIC format)
 */
uint64_t decode_varint(const uint8_t *data, size_t *bytes_consumed = nullptr);

/**
 * @brief Get QUIC version string
 */
const char *get_quic_version();

/**
 * @brief Check if QUIC is supported
 */
bool is_quic_supported();

} // namespace transport
} // namespace psyne