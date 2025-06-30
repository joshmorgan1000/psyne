#include "webrtc_channel.hpp"
#include "../utils/checksum.hpp"
#include "webrtc/ice_agent.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>

namespace psyne {

// Minimal RingBuffer stub for WebRTC (which doesn't use ring buffers)
class RingBuffer {
public:
    virtual ~RingBuffer() = default;
    virtual uint8_t* base_ptr() { return nullptr; }
    virtual const uint8_t* base_ptr() const { return nullptr; }
    virtual uint32_t reserve_slot(size_t) { return Channel::BUFFER_FULL; }
    virtual void advance_write_pointer(size_t) {}
    virtual void advance_read_pointer(size_t) {}
    virtual size_t write_position() const { return 0; }
    virtual size_t read_position() const { return 0; }
    virtual size_t capacity() const { return 0; }
    virtual uint32_t reserve_write_space(size_t) { return Channel::BUFFER_FULL; }
    virtual std::span<const uint8_t> get_read_span(uint32_t, size_t) const { return {}; }
};

namespace detail {

/**
 * @brief Simple WebRTC signaling transport using WebSocket
 */
class WebSocketSignalingTransport : public SignalingTransport {
public:
    explicit WebSocketSignalingTransport(const std::string &uri)
        : server_uri_(uri) {}

    void connect(const std::string &signaling_server_uri) override {
        // For now, simplified implementation
        // In real implementation, would use WebSocket library
        connected_ = true;
        if (on_connected)
            on_connected();
    }

    void send_message(const SignalingMessage &message) override {
        // Simulate sending message through WebSocket
        std::cout << "Sending signaling message: "
                  << static_cast<int>(message.type) << " to peer "
                  << message.peer_id << std::endl;
    }

    void disconnect() override {
        connected_ = false;
        if (on_disconnected)
            on_disconnected();
    }

    bool is_connected() const override {
        return connected_;
    }

private:
    std::string server_uri_;
    std::atomic<bool> connected_{false};
};

/**
 * @brief Simple data channel implementation
 */
class SimpleDataChannel : public RTCDataChannel {
public:
    explicit SimpleDataChannel(const std::string &label,
                               const RTCDataChannelConfig &config)
        : label_(label), config_(config), open_(false) {}

    void send(const void *data, size_t size) override {
        if (!open_)
            return;

        // In real implementation, would send through SCTP/DTLS
        if (peer_channel_) {
            peer_channel_->receive_data(data, size);
        }
    }

    bool is_open() const override {
        return open_;
    }

    void close() override {
        open_ = false;
        if (on_close)
            on_close();
    }

    void open() {
        open_ = true;
        if (on_open)
            on_open();
    }

    void receive_data(const void *data, size_t size) {
        if (on_message) {
            on_message(data, size);
        }
    }

    void set_peer_channel(std::shared_ptr<SimpleDataChannel> peer) {
        peer_channel_ = peer;
    }

private:
    std::string label_;
    RTCDataChannelConfig config_;
    std::atomic<bool> open_;
    std::shared_ptr<SimpleDataChannel> peer_channel_;
};

/**
 * @brief Simple peer connection implementation for demonstration
 */
class SimplePeerConnection : public RTCPeerConnection {
public:
    explicit SimplePeerConnection(const WebRTCConfig &config)
        : config_(config) {}

    void create_offer() override {
        RTCSessionDescription offer;
        offer.type = RTCSessionDescription::Offer;
        offer.sdp = generate_simple_sdp(true);

        if (on_local_description) {
            on_local_description(offer);
        }
    }

    void create_answer() override {
        RTCSessionDescription answer;
        answer.type = RTCSessionDescription::Answer;
        answer.sdp = generate_simple_sdp(false);

        if (on_local_description) {
            on_local_description(answer);
        }
    }

    void set_local_description(const RTCSessionDescription &desc) override {
        local_description_ = desc;
        std::cout << "Set local description: " << desc.sdp.substr(0, 50)
                  << "..." << std::endl;
    }

    void set_remote_description(const RTCSessionDescription &desc) override {
        remote_description_ = desc;
        std::cout << "Set remote description: " << desc.sdp.substr(0, 50)
                  << "..." << std::endl;

        // Simulate connection establishment
        connection_state_ = RTCPeerConnectionState::Connected;
        if (on_connection_state_change) {
            on_connection_state_change(connection_state_);
        }
    }

    void add_ice_candidate(const RTCIceCandidate &candidate) override {
        ice_candidates_.push_back(candidate);
        std::cout << "Added ICE candidate: " << candidate.candidate
                  << std::endl;
    }

    std::shared_ptr<RTCDataChannel>
    create_data_channel(const std::string &label,
                        const RTCDataChannelConfig &config) override {
        auto channel = std::make_shared<SimpleDataChannel>(label, config);
        data_channels_.push_back(channel);

        return channel;
    }

    RTCPeerConnectionState connection_state() const override {
        return connection_state_;
    }

    RTCIceConnectionState ice_connection_state() const override {
        return ice_connection_state_;
    }

    void close() override {
        connection_state_ = RTCPeerConnectionState::Closed;
        ice_connection_state_ = RTCIceConnectionState::Closed;

        for (auto &channel : data_channels_) {
            if (auto simple_channel =
                    std::dynamic_pointer_cast<SimpleDataChannel>(channel)) {
                simple_channel->close();
            }
        }
    }

    void simulate_peer_connection(std::shared_ptr<SimplePeerConnection> peer) {
        peer_connection_ = peer;

        // Connect data channels
        if (!data_channels_.empty() && !peer->data_channels_.empty()) {
            auto our_channel =
                std::dynamic_pointer_cast<SimpleDataChannel>(data_channels_[0]);
            auto peer_channel = std::dynamic_pointer_cast<SimpleDataChannel>(
                peer->data_channels_[0]);

            if (our_channel && peer_channel) {
                our_channel->set_peer_channel(peer_channel);
                peer_channel->set_peer_channel(our_channel);

                our_channel->open();
                peer_channel->open();
            }
        }
    }

private:
    WebRTCConfig config_;
    std::optional<RTCSessionDescription> local_description_;
    std::optional<RTCSessionDescription> remote_description_;
    std::vector<RTCIceCandidate> ice_candidates_;
    std::vector<std::shared_ptr<RTCDataChannel>> data_channels_;
    std::shared_ptr<SimplePeerConnection> peer_connection_;

    std::string generate_simple_sdp(bool is_offer) {
        std::ostringstream sdp;
        sdp << "v=0\r\n";
        sdp << "o=- "
            << std::chrono::system_clock::now().time_since_epoch().count()
            << " 2 IN IP4 127.0.0.1\r\n";
        sdp << "s=-\r\n";
        sdp << "t=0 0\r\n";

        if (is_offer) {
            sdp << "a=group:BUNDLE data\r\n";
        }

        sdp << "m=application 9 DTLS/SCTP 5000\r\n";
        sdp << "c=IN IP4 0.0.0.0\r\n";
        sdp << "a=ice-ufrag:psyne\r\n";
        sdp << "a=ice-pwd:psynepassword\r\n";
        sdp << "a=fingerprint:sha-256 ";

        // Generate simple fingerprint
        for (int i = 0; i < 32; ++i) {
            sdp << std::hex << (i % 16);
            if (i < 31)
                sdp << ":";
        }
        sdp << "\r\n";

        sdp << "a=setup:" << (is_offer ? "actpass" : "active") << "\r\n";
        sdp << "a=mid:data\r\n";
        sdp << "a=sctpmap:5000 webrtc-datachannel 1024\r\n";

        return sdp.str();
    }
};

// WebRTCChannel implementation

detail::WebRTCChannel::WebRTCChannel(
    const std::string &uri, size_t buffer_size, ChannelMode mode,
    ChannelType type, const WebRTCConfig &config,
    const std::string &signaling_server_uri,
    const compression::CompressionConfig &compression_config)
    : ChannelImpl(uri, buffer_size, mode, type), webrtc_config_(config),
      signaling_server_uri_(signaling_server_uri), is_offerer_(true),
      compression_manager_(compression_config) {
    parse_webrtc_uri(uri);
    initialize_peer_connection();
    setup_signaling();
    setup_data_channel();
}

detail::WebRTCChannel::WebRTCChannel(
    const std::string &uri, size_t buffer_size, ChannelMode mode,
    ChannelType type, const WebRTCConfig &config,
    std::shared_ptr<SignalingTransport> signaling_transport,
    const compression::CompressionConfig &compression_config)
    : ChannelImpl(uri, buffer_size, mode, type), webrtc_config_(config),
      signaling_transport_(signaling_transport), is_offerer_(false),
      compression_manager_(compression_config) {
    parse_webrtc_uri(uri);
    initialize_peer_connection();
    setup_data_channel();
}

detail::WebRTCChannel::~WebRTCChannel() {
    stopped_.store(true);

    if (peer_connection_) {
        peer_connection_->close();
    }

    if (signaling_thread_.joinable()) {
        signaling_thread_.join();
    }

    if (connection_thread_.joinable()) {
        connection_thread_.join();
    }
}

void detail::WebRTCChannel::parse_webrtc_uri(const std::string &uri) {
    // Parse URI: webrtc://peer-id or webrtc://signaling-server/room-id
    std::regex uri_regex("webrtc://([^/]+)(?:/(.+))?");
    std::smatch matches;

    if (std::regex_match(uri, matches, uri_regex)) {
        if (matches.size() >= 2) {
            peer_id_ = matches[1].str();
        }
        if (matches.size() >= 3 && !matches[2].str().empty()) {
            // Room-based connection
            signaling_server_uri_ = "ws://" + matches[1].str();
            peer_id_ = matches[2].str();
        }
    }
}

void detail::WebRTCChannel::initialize_peer_connection() {
    peer_connection_ = std::make_unique<SimplePeerConnection>(webrtc_config_);

    // Set up event handlers
    peer_connection_->on_connection_state_change =
        [this](RTCPeerConnectionState state) {
            on_connection_state_change(state);
        };

    peer_connection_->on_ice_candidate =
        [this](const RTCIceCandidate &candidate) {
            handle_ice_candidate(candidate);
        };

    peer_connection_->on_local_description =
        [this](const RTCSessionDescription &desc) {
            handle_session_description(desc);
        };

    peer_connection_->on_data_channel =
        [this](std::shared_ptr<RTCDataChannel> channel) {
            data_channel_ = channel;
            setup_data_channel();
        };
}

void detail::WebRTCChannel::setup_signaling() {
    if (!signaling_transport_) {
        signaling_transport_ = std::make_shared<WebSocketSignalingTransport>(
            signaling_server_uri_);
    }

    signaling_transport_->on_message = [this](const SignalingMessage &message) {
        handle_signaling_message(message);
    };

    signaling_transport_->on_connected = [this]() {
        if (is_offerer_) {
            peer_connection_->create_offer();
        }
    };

    signaling_transport_->connect(signaling_server_uri_);
    start_signaling_loop();
}

void detail::WebRTCChannel::setup_data_channel() {
    if (!data_channel_ && is_offerer_) {
        data_channel_ = peer_connection_->create_data_channel(
            webrtc_config_.data_channel_config.label,
            webrtc_config_.data_channel_config);
    }

    if (data_channel_) {
        data_channel_->on_open = [this]() { on_data_channel_open(); };

        data_channel_->on_message = [this](const void *data, size_t size) {
            on_data_channel_message(data, size);
        };

        data_channel_->on_close = [this]() { on_data_channel_close(); };
    }
}

void detail::WebRTCChannel::on_data_channel_open() {
    connected_.store(true);
    start_connection_monitoring();
    std::cout << "WebRTC data channel opened for peer: " << peer_id_
              << std::endl;
}

void detail::WebRTCChannel::on_data_channel_close() {
    connected_.store(false);
    std::cout << "WebRTC data channel closed for peer: " << peer_id_
              << std::endl;
}

void detail::WebRTCChannel::on_data_channel_message(const void *data, size_t size) {
    MessageHandle handle;
    deserialize_message(data, size, handle);

    {
        std::lock_guard<std::mutex> lock(message_mutex_);
        incoming_messages_.push(std::move(handle));
    }
    message_cv_.notify_one();

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.bytes_received += size;
        stats_.packets_received++;
    }
}

void detail::WebRTCChannel::on_connection_state_change(RTCPeerConnectionState state) {
    std::cout << "WebRTC connection state changed to: "
              << static_cast<int>(state) << std::endl;
}

void detail::WebRTCChannel::handle_signaling_message(const SignalingMessage &message) {
    switch (message.type) {
    case SignalingMessage::Offer: {
        RTCSessionDescription desc;
        desc.type = RTCSessionDescription::Offer;
        desc.sdp = message.payload;
        peer_connection_->set_remote_description(desc);
        peer_connection_->create_answer();
        break;
    }
    case SignalingMessage::Answer: {
        RTCSessionDescription desc;
        desc.type = RTCSessionDescription::Answer;
        desc.sdp = message.payload;
        peer_connection_->set_remote_description(desc);
        break;
    }
    case SignalingMessage::IceCandidate: {
        // Parse ICE candidate from JSON payload
        RTCIceCandidate candidate;
        candidate.candidate = message.payload;
        peer_connection_->add_ice_candidate(candidate);
        break;
    }
    default:
        break;
    }
}

void detail::WebRTCChannel::handle_ice_candidate(const RTCIceCandidate &candidate) {
    SignalingMessage message;
    message.type = SignalingMessage::IceCandidate;
    message.peer_id = peer_id_;
    message.payload = candidate.candidate;
    message.timestamp = std::chrono::system_clock::now();

    signaling_transport_->send_message(message);
}

void detail::WebRTCChannel::handle_session_description(
    const RTCSessionDescription &desc) {
    SignalingMessage message;
    message.type = (desc.type == RTCSessionDescription::Offer)
                       ? SignalingMessage::Offer
                       : SignalingMessage::Answer;
    message.peer_id = peer_id_;
    message.payload = desc.sdp;
    message.timestamp = std::chrono::system_clock::now();

    signaling_transport_->send_message(message);
    peer_connection_->set_local_description(desc);
}

void detail::WebRTCChannel::start_signaling_loop() {
    signaling_thread_ = std::thread([this]() {
        while (!stopped_.load()) {
            // Signaling processing loop
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
}

void detail::WebRTCChannel::start_connection_monitoring() {
    connection_thread_ = std::thread([this]() {
        while (!stopped_.load() && connected_.load()) {
            // Monitor connection health
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });
}

// ChannelImpl interface implementation

// Zero-copy interface implementation
// Note: WebRTC requires serialization for network transport, so memcpy is unavoidable
// This is an acceptable exception to the zero-copy rule

uint32_t detail::WebRTCChannel::reserve_write_slot(size_t size) {
    if (!connected_.load()) {
        return Channel::BUFFER_FULL;
    }
    
    // WebRTC requires copying data for serialization anyway
    // We'll allocate a temporary buffer and return a pseudo-offset
    std::lock_guard<std::mutex> lock(message_mutex_);
    
    // Check if we have space in outgoing queue
    if (outgoing_buffers_.size() >= 100) { // Arbitrary limit
        return Channel::BUFFER_FULL;
    }
    
    // Allocate buffer for the message
    auto buffer = std::make_unique<uint8_t[]>(size);
    pending_writes_[next_write_id_] = std::move(buffer);
    
    return next_write_id_++; // Return write ID as "offset"
}

void detail::WebRTCChannel::notify_message_ready(uint32_t offset, size_t size) {
    if (!data_channel_ || !data_channel_->is_open()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(message_mutex_);
    
    // Find the pending write buffer
    auto it = pending_writes_.find(offset);
    if (it == pending_writes_.end()) {
        return; // Buffer not found
    }
    
    // Move buffer to outgoing queue
    outgoing_buffers_.push(std::move(it->second));
    pending_writes_.erase(it);
    
    // Process outgoing messages
    while (!outgoing_buffers_.empty() && data_channel_->is_open()) {
        auto& buffer = outgoing_buffers_.front();
        
        // Send via WebRTC data channel
        data_channel_->send(buffer.get(), size);
        
        // Update statistics
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            stats_.bytes_sent += size;
            stats_.packets_sent++;
        }
        
        outgoing_buffers_.pop();
    }
}

RingBuffer& detail::WebRTCChannel::get_ring_buffer() {
    // WebRTC doesn't use a ring buffer - it needs to serialize for network
    // Return a minimal stub to satisfy the interface
    static RingBuffer dummy_buffer;
    return dummy_buffer;
}

void detail::WebRTCChannel::advance_read_pointer(size_t /*size*/) {
    // No-op for WebRTC - messages are consumed immediately
}

// Deprecated legacy methods (required by ChannelImpl interface)
void* detail::WebRTCChannel::reserve_space(size_t size) {
    // Use the new API internally
    uint32_t offset = reserve_write_slot(size);
    if (offset == Channel::BUFFER_FULL) {
        return nullptr;
    }
    
    // Return the buffer pointer
    auto it = pending_writes_.find(offset);
    if (it != pending_writes_.end()) {
        return it->second.get();
    }
    return nullptr;
}

void detail::WebRTCChannel::commit_message(void* handle) {
    // Legacy commit - find the pending write by scanning
    std::lock_guard<std::mutex> lock(message_mutex_);
    
    for (auto& [offset, buffer] : pending_writes_) {
        if (buffer.get() == handle) {
            // Move to outgoing queue
            outgoing_buffers_.push(std::move(buffer));
            pending_writes_.erase(offset);
            
            // Process outgoing messages
            while (!outgoing_buffers_.empty() && data_channel_->is_open()) {
                auto& out_buffer = outgoing_buffers_.front();
                
                // Send via WebRTC data channel
                // For legacy API, we don't know the exact size, so send what we have
                data_channel_->send(out_buffer.get(), 1024); // Arbitrary size
                
                outgoing_buffers_.pop();
            }
            break;
        }
    }
}

void *detail::WebRTCChannel::receive_message(size_t &size, uint32_t &type) {
    std::unique_lock<std::mutex> lock(message_mutex_);

    if (incoming_messages_.empty()) {
        message_cv_.wait_for(lock, std::chrono::milliseconds(100));
    }

    if (incoming_messages_.empty()) {
        return nullptr;
    }

    auto &message = incoming_messages_.front();
    size = message.size;
    type = message.type_id;

    void *data = message.data.data();
    incoming_messages_.pop();

    return data;
}

void detail::WebRTCChannel::release_message(void *handle) {
    // For this simple implementation, messages are managed automatically
}

detail::RTCPeerConnectionState detail::WebRTCChannel::connection_state() const {
    return peer_connection_ ? peer_connection_->connection_state()
                            : RTCPeerConnectionState::New;
}

detail::RTCIceConnectionState detail::WebRTCChannel::ice_connection_state() const {
    return peer_connection_ ? peer_connection_->ice_connection_state()
                            : RTCIceConnectionState::New;
}

bool detail::WebRTCChannel::is_connected() const {
    return connected_.load();
}

detail::WebRTCChannel::ConnectionStats detail::WebRTCChannel::get_connection_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}


std::vector<uint8_t> detail::WebRTCChannel::serialize_message(const void *data,
                                                      size_t size,
                                                      uint32_t type_id) {
    std::vector<uint8_t> serialized;

    // Simple serialization: [size][type_id][data]
    serialized.reserve(sizeof(size) + sizeof(type_id) + size);

    serialized.insert(serialized.end(),
                      reinterpret_cast<const uint8_t *>(&size),
                      reinterpret_cast<const uint8_t *>(&size) + sizeof(size));

    serialized.insert(
        serialized.end(), reinterpret_cast<const uint8_t *>(&type_id),
        reinterpret_cast<const uint8_t *>(&type_id) + sizeof(type_id));

    serialized.insert(serialized.end(), static_cast<const uint8_t *>(data),
                      static_cast<const uint8_t *>(data) + size);

    return serialized;
}

void detail::WebRTCChannel::deserialize_message(const void *data, size_t total_size,
                                        MessageHandle &handle) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);

    if (total_size < sizeof(size_t) + sizeof(uint32_t)) {
        return;
    }

    size_t data_size = *reinterpret_cast<const size_t *>(bytes);
    handle.type_id =
        *reinterpret_cast<const uint32_t *>(bytes + sizeof(size_t));
    handle.size = data_size;

    const uint8_t *message_data = bytes + sizeof(size_t) + sizeof(uint32_t);
    handle.data.assign(message_data, message_data + data_size);
}

// Factory function

std::unique_ptr<WebRTCChannel> create_webrtc_channel(
    const std::string &uri, size_t buffer_size, ChannelMode mode,
    ChannelType type, const WebRTCConfig &config,
    const std::string &signaling_server_uri,
    const compression::CompressionConfig &compression_config) {
    return std::make_unique<WebRTCChannel>(uri, buffer_size, mode, type, config,
                                           signaling_server_uri,
                                           compression_config);
}

} // namespace detail
} // namespace psyne