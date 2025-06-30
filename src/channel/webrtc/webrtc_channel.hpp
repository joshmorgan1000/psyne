#pragma once

#include "../compression/compression.hpp"
#include "../channel_impl.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace psyne {
namespace detail {

/**
 * @brief WebRTC connection states following the standard state machine
 */
enum class RTCPeerConnectionState {
    New,
    Connecting,
    Connected,
    Disconnected,
    Failed,
    Closed
};

/**
 * @brief ICE connection states
 */
enum class RTCIceConnectionState {
    New,
    Gathering,
    Checking,
    Connected,
    Completed,
    Failed,
    Disconnected,
    Closed
};

/**
 * @brief ICE candidate types
 */
enum class RTCIceCandidateType {
    Host,            // Local interface
    ServerReflexive, // Through STUN server
    PeerReflexive,   // Discovered during connectivity checks
    Relay            // Through TURN server
};

/**
 * @brief WebRTC data channel configuration
 */
struct RTCDataChannelConfig {
    std::string label = "psyne-channel";
    bool ordered = true;
    std::optional<uint16_t> max_packet_life_time;
    std::optional<uint16_t> max_retransmits;
    std::string protocol = "";
    bool negotiated = false;
    std::optional<uint16_t> id;
};

/**
 * @brief ICE candidate information
 */
struct RTCIceCandidate {
    std::string candidate;
    std::string sdp_mid;
    std::optional<uint16_t> sdp_mline_index;
    RTCIceCandidateType type;
    std::string foundation;
    uint32_t component;
    std::string transport;
    uint32_t priority;
    std::string address;
    uint16_t port;
    std::string related_address;
    uint16_t related_port;
};

/**
 * @brief Session Description Protocol (SDP) message
 */
struct RTCSessionDescription {
    enum Type { Offer, Answer, PrAnswer, Rollback } type;
    std::string sdp;
};

/**
 * @brief WebRTC signaling message types for coordination
 */
struct SignalingMessage {
    enum Type { Offer, Answer, IceCandidate, IceCandidateError, Close } type;

    std::string peer_id;
    std::string payload;
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief STUN server configuration
 */
struct STUNServerConfig {
    std::string host;
    uint16_t port = 3478;
    std::string username;
    std::string credential;
};

/**
 * @brief TURN server configuration
 */
struct TURNServerConfig {
    std::string host;
    uint16_t port = 3478;
    std::string username;
    std::string credential;
    std::string transport = "udp"; // udp, tcp, tls
};

/**
 * @brief WebRTC configuration
 */
struct WebRTCConfig {
    std::vector<STUNServerConfig> stun_servers;
    std::vector<TURNServerConfig> turn_servers;
    RTCDataChannelConfig data_channel_config;
    bool enable_ice_tcp = false;
    bool enable_ipv6 = true;
    uint32_t ice_candidate_pool_size = 0;
    std::chrono::milliseconds ice_gathering_timeout{5000};
    std::chrono::milliseconds connection_timeout{30000};
};

/**
 * @brief WebRTC data channel interface
 */
class RTCDataChannel {
public:
    virtual ~RTCDataChannel() = default;

    virtual void send(const void *data, size_t size) = 0;
    virtual bool is_open() const = 0;
    virtual void close() = 0;

    // Event callbacks
    std::function<void()> on_open;
    std::function<void(const void *data, size_t size)> on_message;
    std::function<void()> on_close;
    std::function<void(const std::string &error)> on_error;

protected:
    std::string label_;
    RTCDataChannelConfig config_;
};

/**
 * @brief WebRTC peer connection interface
 */
class RTCPeerConnection {
public:
    virtual ~RTCPeerConnection() = default;

    // Core WebRTC operations
    virtual void create_offer() = 0;
    virtual void create_answer() = 0;
    virtual void set_local_description(const RTCSessionDescription &desc) = 0;
    virtual void set_remote_description(const RTCSessionDescription &desc) = 0;
    virtual void add_ice_candidate(const RTCIceCandidate &candidate) = 0;

    // Data channel management
    virtual std::shared_ptr<RTCDataChannel>
    create_data_channel(const std::string &label,
                        const RTCDataChannelConfig &config = {}) = 0;

    // State management
    virtual RTCPeerConnectionState connection_state() const = 0;
    virtual RTCIceConnectionState ice_connection_state() const = 0;
    virtual void close() = 0;

    // Event callbacks
    std::function<void(const RTCSessionDescription &)> on_local_description;
    std::function<void(const RTCIceCandidate &)> on_ice_candidate;
    std::function<void(std::shared_ptr<RTCDataChannel>)> on_data_channel;
    std::function<void(RTCPeerConnectionState)> on_connection_state_change;
    std::function<void(RTCIceConnectionState)> on_ice_connection_state_change;
    std::function<void(const std::string &)> on_error;

protected:
    WebRTCConfig config_;
    RTCPeerConnectionState connection_state_ = RTCPeerConnectionState::New;
    RTCIceConnectionState ice_connection_state_ = RTCIceConnectionState::New;
};

/**
 * @brief Signaling transport interface for WebRTC coordination
 */
class SignalingTransport {
public:
    virtual ~SignalingTransport() = default;

    virtual void connect(const std::string &signaling_server_uri) = 0;
    virtual void send_message(const SignalingMessage &message) = 0;
    virtual void disconnect() = 0;
    virtual bool is_connected() const = 0;

    // Event callback
    std::function<void(const SignalingMessage &)> on_message;
    std::function<void()> on_connected;
    std::function<void()> on_disconnected;
    std::function<void(const std::string &)> on_error;
};

/**
 * @brief WebRTC Channel implementation for psyne
 *
 * Supports URI schemes:
 * - webrtc://peer-id - Direct peer connection (requires signaling server)
 * - webrtc://signaling-server/room-id - Room-based connection
 *
 * Examples:
 * - webrtc://player123 - Connect directly to peer "player123"
 * - webrtc://localhost:8080/game-room-1 - Join room "game-room-1" via
 * localhost:8080
 */
class WebRTCChannel : public ChannelImpl {
public:
    /**
     * @brief Create WebRTC channel as an offerer (initiator)
     */
    WebRTCChannel(
        const std::string &uri, size_t buffer_size, ChannelMode mode,
        ChannelType type, const WebRTCConfig &config,
        const std::string &signaling_server_uri,
        const compression::CompressionConfig &compression_config = {});

    /**
     * @brief Create WebRTC channel as an answerer (receiver)
     */
    WebRTCChannel(
        const std::string &uri, size_t buffer_size, ChannelMode mode,
        ChannelType type, const WebRTCConfig &config,
        std::shared_ptr<SignalingTransport> signaling_transport,
        const compression::CompressionConfig &compression_config = {});

    virtual ~WebRTCChannel();

    // ChannelImpl interface
    void *reserve_space(size_t size) override;
    void commit_message(void *handle) override;
    void *receive_message(size_t &size, uint32_t &type) override;
    void release_message(void *handle) override;
    
    // Zero-copy interface
    uint32_t reserve_write_slot(size_t size) override;
    void notify_message_ready(uint32_t offset, size_t size) override;
    RingBuffer& get_ring_buffer() override;
    const RingBuffer& get_ring_buffer() const override;
    void advance_read_pointer(size_t size) override;
    std::span<uint8_t> get_write_span(size_t size) noexcept;

    // WebRTC-specific methods
    void set_peer_id(const std::string &peer_id) {
        peer_id_ = peer_id;
    }
    std::string peer_id() const {
        return peer_id_;
    }

    RTCPeerConnectionState connection_state() const;
    RTCIceConnectionState ice_connection_state() const;
    bool is_connected() const;

    // Statistics and monitoring
    struct ConnectionStats {
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t packets_sent = 0;
        uint64_t packets_received = 0;
        uint32_t rtt_ms = 0;
        uint32_t jitter_ms = 0;
        double packet_loss_rate = 0.0;
        std::chrono::system_clock::time_point connected_at;
    };

    ConnectionStats get_connection_stats() const;

private:
    // Core components
    std::unique_ptr<RTCPeerConnection> peer_connection_;
    std::shared_ptr<RTCDataChannel> data_channel_;
    std::shared_ptr<SignalingTransport> signaling_transport_;

    // Configuration
    WebRTCConfig webrtc_config_;
    compression::CompressionManager compression_manager_;

    // Connection management
    std::string peer_id_;
    std::string signaling_server_uri_;
    bool is_offerer_ = false;
    std::atomic<bool> connected_{false};

    // Message queuing
    struct MessageHandle {
        std::vector<uint8_t> data;
        uint32_t type_id;
        size_t size;
    };

    std::queue<MessageHandle> incoming_messages_;
    std::queue<std::unique_ptr<uint8_t[]>> outgoing_buffers_;
    std::mutex message_mutex_;
    std::condition_variable message_cv_;

    // Threading
    std::thread signaling_thread_;
    std::thread connection_thread_;

    // Statistics
    mutable std::mutex stats_mutex_;
    ConnectionStats stats_;
    
    // For zero-copy API adaptation
    std::unordered_map<uint32_t, std::unique_ptr<uint8_t[]>> pending_writes_;
    std::atomic<uint32_t> next_write_id_{1};
    uint32_t current_write_id_{0};

    // Private methods
    void initialize_peer_connection();
    void setup_signaling();
    void setup_data_channel();
    void handle_signaling_message(const SignalingMessage &message);
    void handle_ice_candidate(const RTCIceCandidate &candidate);
    void handle_session_description(const RTCSessionDescription &desc);
    void on_data_channel_message(const void *data, size_t size);
    void on_data_channel_open();
    void on_data_channel_close();
    void on_connection_state_change(RTCPeerConnectionState state);
    void start_signaling_loop();
    void start_connection_monitoring();
    void parse_webrtc_uri(const std::string &uri);

    // Utility methods
    std::vector<uint8_t> serialize_message(const void *data, size_t size,
                                           uint32_t type_id);
    void deserialize_message(const void *data, size_t size,
                             MessageHandle &handle);
};

/**
 * @brief Factory function to create WebRTC channels
 */
std::unique_ptr<WebRTCChannel> create_webrtc_channel(
    const std::string &uri, size_t buffer_size,
    ChannelMode mode = ChannelMode::SPSC,
    ChannelType type = ChannelType::MultiType, const WebRTCConfig &config = {},
    const std::string &signaling_server_uri = "ws://localhost:8080",
    const compression::CompressionConfig &compression_config = {});

} // namespace detail
} // namespace psyne