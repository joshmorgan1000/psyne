/**
 * @file data_channel.hpp
 * @brief Enhanced WebRTC data channel implementation with reliable delivery
 *
 * Provides WebRTC data channels with various delivery modes, reliability
 * guarantees, and advanced features for real-time communication.
 *
 * Features:
 * - Multiple reliability modes (reliable, unreliable, partial reliability)
 * - Message ordering guarantees
 * - Flow control and congestion management
 * - Priority-based message queuing
 * - Binary and text message support
 * - Large message fragmentation
 * - Statistics and monitoring
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include "../webrtc_channel.hpp"
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace psyne {
namespace detail {
namespace webrtc {

/**
 * @brief Data channel delivery modes
 */
enum class DeliveryMode {
    RELIABLE_ORDERED,     ///< Guaranteed delivery in order (TCP-like)
    RELIABLE_UNORDERED,   ///< Guaranteed delivery, any order
    UNRELIABLE_ORDERED,   ///< Best effort with ordering (partial reliability)
    UNRELIABLE_UNORDERED, ///< Best effort, any order (UDP-like)
    PARTIAL_RELIABLE_RTX, ///< Limited retransmissions
    PARTIAL_RELIABLE_TTL  ///< Time-to-live based delivery
};

/**
 * @brief Message priority levels for QoS
 */
enum class MessagePriority {
    CRITICAL = 0, ///< Highest priority (control messages)
    HIGH = 1,     ///< High priority (user input, urgent data)
    NORMAL = 2,   ///< Normal priority (default)
    LOW = 3,      ///< Low priority (background data)
    BULK = 4      ///< Lowest priority (file transfers)
};

/**
 * @brief Data channel configuration
 */
struct DataChannelConfig {
    std::string label;
    DeliveryMode delivery_mode = DeliveryMode::RELIABLE_ORDERED;
    MessagePriority default_priority = MessagePriority::NORMAL;

    // Reliability parameters
    uint16_t max_retransmits = 0; // For partial reliability (0 = use default)
    std::chrono::milliseconds max_ttl{0}; // For TTL-based partial reliability

    // Flow control
    size_t send_buffer_size = 262144; // 256 KB default
    size_t recv_buffer_size = 262144; // 256 KB default
    size_t max_message_size = 65536;  // 64 KB default

    // Congestion control
    bool enable_congestion_control = true;
    size_t initial_congestion_window = 4096; // 4 KB
    size_t max_congestion_window = 65536;    // 64 KB

    // Performance tuning
    std::chrono::milliseconds send_timeout{5000}; // 5 seconds
    std::chrono::milliseconds ack_timeout{200};   // 200 ms
    bool enable_nagle = false;                    // Disable Nagle for real-time
    bool enable_fast_retransmit = true;
};

/**
 * @brief Message metadata for reliability and ordering
 */
struct MessageMetadata {
    uint64_t sequence_number = 0;
    uint64_t timestamp_us = 0;
    MessagePriority priority = MessagePriority::NORMAL;
    DeliveryMode delivery_mode = DeliveryMode::RELIABLE_ORDERED;

    // Reliability tracking
    uint16_t retransmit_count = 0;
    std::chrono::steady_clock::time_point send_time;
    std::chrono::steady_clock::time_point expiry_time;

    // Fragmentation
    bool is_fragmented = false;
    uint16_t fragment_id = 0;
    uint16_t fragment_index = 0;
    uint16_t total_fragments = 0;
};

/**
 * @brief Data channel message with metadata
 */
class DataChannelMessage {
public:
    DataChannelMessage() = default;
    DataChannelMessage(const void *data, size_t size,
                       MessagePriority priority = MessagePriority::NORMAL);
    DataChannelMessage(const std::string &text,
                       MessagePriority priority = MessagePriority::NORMAL);
    DataChannelMessage(std::vector<uint8_t> binary_data,
                       MessagePriority priority = MessagePriority::NORMAL);

    // Data access
    const uint8_t *data() const {
        return data_.data();
    }
    uint8_t *data() {
        return data_.data();
    }
    size_t size() const {
        return data_.size();
    }
    bool empty() const {
        return data_.empty();
    }

    // Message type
    bool is_text() const {
        return is_text_;
    }
    bool is_binary() const {
        return !is_text_;
    }

    // Text conversion
    std::string to_text() const;
    void from_text(const std::string &text);

    // Binary data
    const std::vector<uint8_t> &binary_data() const {
        return data_;
    }
    void set_binary_data(std::vector<uint8_t> data);

    // Metadata
    const MessageMetadata &metadata() const {
        return metadata_;
    }
    MessageMetadata &metadata() {
        return metadata_;
    }

    // Priority
    MessagePriority priority() const {
        return metadata_.priority;
    }
    void set_priority(MessagePriority priority) {
        metadata_.priority = priority;
    }

    // Serialization for wire protocol
    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t> &data);

private:
    std::vector<uint8_t> data_;
    bool is_text_ = false;
    MessageMetadata metadata_;
};

/**
 * @brief Statistics for data channel performance monitoring
 */
struct DataChannelStats {
    // Message counters
    uint64_t messages_sent = 0;
    uint64_t messages_received = 0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;

    // Reliability stats
    uint64_t messages_retransmitted = 0;
    uint64_t messages_lost = 0;
    uint64_t messages_duplicated = 0;
    uint64_t messages_out_of_order = 0;

    // Timing stats
    double avg_rtt_ms = 0.0;
    double min_rtt_ms = std::numeric_limits<double>::max();
    double max_rtt_ms = 0.0;

    // Buffer stats
    size_t send_buffer_bytes = 0;
    size_t recv_buffer_bytes = 0;
    size_t max_send_buffer_bytes = 0;
    size_t max_recv_buffer_bytes = 0;

    // Congestion control
    size_t congestion_window_bytes = 0;
    size_t slow_start_threshold = 0;
    uint64_t congestion_events = 0;

    // Error counters
    uint64_t send_errors = 0;
    uint64_t recv_errors = 0;
    uint64_t timeout_errors = 0;
    uint64_t buffer_full_errors = 0;
};

/**
 * @brief Advanced WebRTC data channel with reliability features
 */
class EnhancedDataChannel {
public:
    explicit EnhancedDataChannel(const DataChannelConfig &config);
    ~EnhancedDataChannel();

    // Channel lifecycle
    bool open();
    void close();
    bool is_open() const;
    bool is_connecting() const;

    // Configuration
    const DataChannelConfig &config() const {
        return config_;
    }
    void update_config(const DataChannelConfig &config);

    // Message sending
    bool send_message(const DataChannelMessage &message);
    bool send_text(const std::string &text,
                   MessagePriority priority = MessagePriority::NORMAL);
    bool send_binary(const void *data, size_t size,
                     MessagePriority priority = MessagePriority::NORMAL);
    bool send_binary(const std::vector<uint8_t> &data,
                     MessagePriority priority = MessagePriority::NORMAL);

    // Non-blocking send
    bool try_send_message(const DataChannelMessage &message);
    bool try_send_text(const std::string &text,
                       MessagePriority priority = MessagePriority::NORMAL);
    bool try_send_binary(const void *data, size_t size,
                         MessagePriority priority = MessagePriority::NORMAL);

    // Message receiving
    bool recv_message(DataChannelMessage &message);
    std::string recv_text();
    std::vector<uint8_t> recv_binary();

    // Non-blocking receive
    bool try_recv_message(DataChannelMessage &message);
    bool try_recv_text(std::string &text);
    bool try_recv_binary(std::vector<uint8_t> &data);

    // Batch operations
    bool send_batch(const std::vector<DataChannelMessage> &messages);
    std::vector<DataChannelMessage> recv_batch(size_t max_messages = 10);

    // Flow control
    size_t get_send_buffer_size() const;
    size_t get_recv_buffer_size() const;
    bool is_send_buffer_full() const;
    bool is_recv_buffer_empty() const;

    // Statistics and monitoring
    DataChannelStats get_stats() const;
    void reset_stats();

    // Quality of Service
    void set_priority_weights(
        const std::unordered_map<MessagePriority, double> &weights);
    std::unordered_map<MessagePriority, double> get_priority_weights() const;

    // Event callbacks
    std::function<void()> on_open;
    std::function<void()> on_close;
    std::function<void(const std::string &error)> on_error;
    std::function<void(const DataChannelMessage &)> on_message;
    std::function<void(const DataChannelStats &)> on_stats_update;

    // Advanced features
    bool ping(std::chrono::milliseconds timeout = std::chrono::milliseconds{
                  1000});
    double get_current_rtt() const;
    size_t get_optimal_message_size() const;
    void enable_bandwidth_adaptation(bool enable);

private:
    DataChannelConfig config_;
    std::atomic<bool> open_{false};
    std::atomic<bool> connecting_{false};
    std::atomic<bool> running_{false};

    // Sequence numbers for ordering and reliability
    std::atomic<uint64_t> next_send_seq_{1};
    std::atomic<uint64_t> next_expected_recv_seq_{1};

    // Message queues with priority support
    struct PriorityQueue {
        std::priority_queue<DataChannelMessage> queue;
        std::mutex mutex;
    };

    std::array<PriorityQueue, 5> send_queues_; // One for each priority level
    std::deque<DataChannelMessage> recv_queue_;
    mutable std::mutex recv_queue_mutex_;

    // Reliability tracking
    struct PendingMessage {
        DataChannelMessage message;
        std::chrono::steady_clock::time_point send_time;
        uint16_t retransmit_count = 0;
    };

    std::unordered_map<uint64_t, PendingMessage> pending_messages_;
    mutable std::mutex pending_messages_mutex_;

    // Fragment reassembly
    struct FragmentBuffer {
        std::unordered_map<uint16_t, DataChannelMessage> fragments;
        std::chrono::steady_clock::time_point first_fragment_time;
        uint16_t total_fragments = 0;
        uint16_t received_fragments = 0;
    };

    std::unordered_map<uint16_t, FragmentBuffer> fragment_buffers_;
    mutable std::mutex fragment_buffers_mutex_;

    // Flow control and congestion management
    std::atomic<size_t> congestion_window_;
    std::atomic<size_t> slow_start_threshold_;
    std::atomic<size_t> bytes_in_flight_;

    // RTT estimation
    std::deque<double> rtt_samples_;
    mutable std::mutex rtt_samples_mutex_;
    double srtt_ = 0.0;   // Smoothed RTT
    double rttvar_ = 0.0; // RTT variation

    // Threading
    std::thread sender_thread_;
    std::thread receiver_thread_;
    std::thread retransmit_thread_;
    std::thread stats_thread_;

    // Statistics
    mutable std::mutex stats_mutex_;
    DataChannelStats stats_;

    // Priority weights for QoS
    std::unordered_map<MessagePriority, double> priority_weights_;
    mutable std::mutex priority_weights_mutex_;

    // Private methods
    void run_sender();
    void run_receiver();
    void run_retransmit_timer();
    void run_stats_updater();

    // Message processing
    DataChannelMessage get_next_message_to_send();
    bool should_fragment_message(const DataChannelMessage &message) const;
    std::vector<DataChannelMessage>
    fragment_message(const DataChannelMessage &message);
    bool process_received_message(const DataChannelMessage &message);
    bool reassemble_fragments(const DataChannelMessage &fragment,
                              DataChannelMessage &complete_message);

    // Reliability
    void handle_acknowledgment(uint64_t sequence_number,
                               std::chrono::steady_clock::time_point ack_time);
    void retransmit_message(uint64_t sequence_number);
    bool should_retransmit(const PendingMessage &pending) const;

    // Congestion control
    void update_congestion_window(bool packet_lost);
    void handle_congestion_event();
    bool is_congestion_window_available(size_t message_size) const;

    // RTT calculation
    void update_rtt(double sample_rtt);
    std::chrono::milliseconds calculate_rto() const;

    // Buffer management
    bool is_send_buffer_space_available(size_t message_size) const;
    void cleanup_expired_fragments();
    void cleanup_old_pending_messages();

    // Statistics helpers
    void update_send_stats(const DataChannelMessage &message);
    void update_recv_stats(const DataChannelMessage &message);
    void update_error_stats(const std::string &error_type);

    // Utility methods
    uint64_t get_timestamp_us() const;
    uint16_t generate_fragment_id();
    double calculate_priority_weight(MessagePriority priority) const;
};

/**
 * @brief Data channel factory with pre-configured profiles
 */
class DataChannelFactory {
public:
    // Gaming profile - low latency, partial reliability
    static std::unique_ptr<EnhancedDataChannel>
    create_gaming_channel(const std::string &label = "gaming");

    // Streaming profile - high throughput, ordered delivery
    static std::unique_ptr<EnhancedDataChannel>
    create_streaming_channel(const std::string &label = "streaming");

    // File transfer profile - reliable, large messages
    static std::unique_ptr<EnhancedDataChannel>
    create_file_transfer_channel(const std::string &label = "files");

    // Chat profile - reliable text messaging
    static std::unique_ptr<EnhancedDataChannel>
    create_chat_channel(const std::string &label = "chat");

    // Control profile - critical, low latency, reliable
    static std::unique_ptr<EnhancedDataChannel>
    create_control_channel(const std::string &label = "control");

    // Custom profile
    static std::unique_ptr<EnhancedDataChannel>
    create_custom_channel(const DataChannelConfig &config);
};

/**
 * @brief Data channel multiplexer for managing multiple channels
 */
class DataChannelMultiplexer {
public:
    DataChannelMultiplexer() = default;
    ~DataChannelMultiplexer();

    // Channel management
    void add_channel(const std::string &name,
                     std::unique_ptr<EnhancedDataChannel> channel);
    void remove_channel(const std::string &name);
    EnhancedDataChannel *get_channel(const std::string &name);
    std::vector<std::string> get_channel_names() const;

    // Broadcast operations
    bool broadcast_text(const std::string &text,
                        MessagePriority priority = MessagePriority::NORMAL);
    bool broadcast_binary(const void *data, size_t size,
                          MessagePriority priority = MessagePriority::NORMAL);
    bool broadcast_message(const DataChannelMessage &message);

    // Selective operations
    bool send_to_channels(const std::vector<std::string> &channel_names,
                          const DataChannelMessage &message);

    // Aggregated statistics
    DataChannelStats get_aggregated_stats() const;
    std::unordered_map<std::string, DataChannelStats>
    get_per_channel_stats() const;

    // Event handling
    std::function<void(const std::string &channel_name,
                       const DataChannelMessage &)>
        on_message;
    std::function<void(const std::string &channel_name,
                       const std::string &error)>
        on_error;

private:
    std::unordered_map<std::string, std::unique_ptr<EnhancedDataChannel>>
        channels_;
    mutable std::mutex channels_mutex_;

    void setup_channel_callbacks(const std::string &name,
                                 EnhancedDataChannel *channel);
};

} // namespace webrtc
} // namespace detail
} // namespace psyne
