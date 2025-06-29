/**
 * @file data_channel.cpp
 * @brief Implementation of enhanced WebRTC data channels
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include "data_channel.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>

namespace psyne {
namespace detail {
namespace webrtc {

// DataChannelMessage implementation

DataChannelMessage::DataChannelMessage(const void* data, size_t size, MessagePriority priority) 
    : data_(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size)
    , is_text_(false) {
    metadata_.priority = priority;
    metadata_.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

DataChannelMessage::DataChannelMessage(const std::string& text, MessagePriority priority)
    : data_(text.begin(), text.end())
    , is_text_(true) {
    metadata_.priority = priority;
    metadata_.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

DataChannelMessage::DataChannelMessage(std::vector<uint8_t> binary_data, MessagePriority priority)
    : data_(std::move(binary_data))
    , is_text_(false) {
    metadata_.priority = priority;
    metadata_.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string DataChannelMessage::to_text() const {
    return std::string(data_.begin(), data_.end());
}

void DataChannelMessage::from_text(const std::string& text) {
    data_.assign(text.begin(), text.end());
    is_text_ = true;
}

void DataChannelMessage::set_binary_data(std::vector<uint8_t> data) {
    data_ = std::move(data);
    is_text_ = false;
}

std::vector<uint8_t> DataChannelMessage::serialize() const {
    std::vector<uint8_t> result;
    
    // Simple serialization format:
    // [metadata_size:4][metadata][is_text:1][data_size:4][data]
    
    // Serialize metadata (simplified)
    std::vector<uint8_t> metadata_bytes;
    
    // Add sequence number
    uint64_t seq = metadata_.sequence_number;
    metadata_bytes.insert(metadata_bytes.end(), 
                         reinterpret_cast<const uint8_t*>(&seq),
                         reinterpret_cast<const uint8_t*>(&seq) + 8);
    
    // Add timestamp
    uint64_t ts = metadata_.timestamp_us;
    metadata_bytes.insert(metadata_bytes.end(),
                         reinterpret_cast<const uint8_t*>(&ts),
                         reinterpret_cast<const uint8_t*>(&ts) + 8);
    
    // Add priority
    uint8_t prio = static_cast<uint8_t>(metadata_.priority);
    metadata_bytes.push_back(prio);
    
    // Add delivery mode
    uint8_t mode = static_cast<uint8_t>(metadata_.delivery_mode);
    metadata_bytes.push_back(mode);
    
    // Add fragmentation info
    metadata_bytes.push_back(metadata_.is_fragmented ? 1 : 0);
    
    uint16_t frag_id = metadata_.fragment_id;
    metadata_bytes.insert(metadata_bytes.end(),
                         reinterpret_cast<const uint8_t*>(&frag_id),
                         reinterpret_cast<const uint8_t*>(&frag_id) + 2);
    
    uint16_t frag_idx = metadata_.fragment_index;
    metadata_bytes.insert(metadata_bytes.end(),
                         reinterpret_cast<const uint8_t*>(&frag_idx),
                         reinterpret_cast<const uint8_t*>(&frag_idx) + 2);
    
    uint16_t total_frags = metadata_.total_fragments;
    metadata_bytes.insert(metadata_bytes.end(),
                         reinterpret_cast<const uint8_t*>(&total_frags),
                         reinterpret_cast<const uint8_t*>(&total_frags) + 2);
    
    // Metadata size
    uint32_t metadata_size = metadata_bytes.size();
    result.insert(result.end(),
                  reinterpret_cast<const uint8_t*>(&metadata_size),
                  reinterpret_cast<const uint8_t*>(&metadata_size) + 4);
    
    // Metadata
    result.insert(result.end(), metadata_bytes.begin(), metadata_bytes.end());
    
    // Is text flag
    result.push_back(is_text_ ? 1 : 0);
    
    // Data size
    uint32_t data_size = data_.size();
    result.insert(result.end(),
                  reinterpret_cast<const uint8_t*>(&data_size),
                  reinterpret_cast<const uint8_t*>(&data_size) + 4);
    
    // Data
    result.insert(result.end(), data_.begin(), data_.end());
    
    return result;
}

bool DataChannelMessage::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < 9) { // Minimum size
        return false;
    }
    
    size_t offset = 0;
    
    // Read metadata size
    uint32_t metadata_size;
    std::memcpy(&metadata_size, data.data() + offset, 4);
    offset += 4;
    
    if (offset + metadata_size + 5 > data.size()) {
        return false;
    }
    
    // Read metadata
    if (metadata_size >= 25) { // Expected metadata size
        std::memcpy(&metadata_.sequence_number, data.data() + offset, 8);
        offset += 8;
        std::memcpy(&metadata_.timestamp_us, data.data() + offset, 8);
        offset += 8;
        metadata_.priority = static_cast<MessagePriority>(data[offset++]);
        metadata_.delivery_mode = static_cast<DeliveryMode>(data[offset++]);
        metadata_.is_fragmented = (data[offset++] != 0);
        std::memcpy(&metadata_.fragment_id, data.data() + offset, 2);
        offset += 2;
        std::memcpy(&metadata_.fragment_index, data.data() + offset, 2);
        offset += 2;
        std::memcpy(&metadata_.total_fragments, data.data() + offset, 2);
        offset += 2;
    } else {
        offset += metadata_size;
    }
    
    // Read is_text flag
    is_text_ = (data[offset++] != 0);
    
    // Read data size
    uint32_t data_size;
    std::memcpy(&data_size, data.data() + offset, 4);
    offset += 4;
    
    if (offset + data_size != data.size()) {
        return false;
    }
    
    // Read data
    data_.assign(data.begin() + offset, data.begin() + offset + data_size);
    
    return true;
}

// EnhancedDataChannel implementation

EnhancedDataChannel::EnhancedDataChannel(const DataChannelConfig& config) 
    : config_(config)
    , congestion_window_(config.initial_congestion_window)
    , slow_start_threshold_(config.max_congestion_window / 2) {
    
    // Initialize priority weights
    priority_weights_[MessagePriority::CRITICAL] = 4.0;
    priority_weights_[MessagePriority::HIGH] = 3.0;
    priority_weights_[MessagePriority::NORMAL] = 2.0;
    priority_weights_[MessagePriority::LOW] = 1.0;
    priority_weights_[MessagePriority::BULK] = 0.5;
}

EnhancedDataChannel::~EnhancedDataChannel() {
    close();
}

bool EnhancedDataChannel::open() {
    if (open_.load()) {
        return true;
    }
    
    connecting_.store(true);
    
    // Simulate connection process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    open_.store(true);
    connecting_.store(false);
    running_.store(true);
    
    // Start worker threads
    sender_thread_ = std::thread([this] { run_sender(); });
    receiver_thread_ = std::thread([this] { run_receiver(); });
    retransmit_thread_ = std::thread([this] { run_retransmit_timer(); });
    stats_thread_ = std::thread([this] { run_stats_updater(); });
    
    if (on_open) {
        on_open();
    }
    
    return true;
}

void EnhancedDataChannel::close() {
    if (!open_.load()) {
        return;
    }
    
    running_.store(false);
    open_.store(false);
    
    // Join threads
    if (sender_thread_.joinable()) {
        sender_thread_.join();
    }
    if (receiver_thread_.joinable()) {
        receiver_thread_.join();
    }
    if (retransmit_thread_.joinable()) {
        retransmit_thread_.join();
    }
    if (stats_thread_.joinable()) {
        stats_thread_.join();
    }
    
    if (on_close) {
        on_close();
    }
}

bool EnhancedDataChannel::is_open() const {
    return open_.load();
}

bool EnhancedDataChannel::is_connecting() const {
    return connecting_.load();
}

void EnhancedDataChannel::update_config(const DataChannelConfig& config) {
    config_ = config;
}

bool EnhancedDataChannel::send_message(const DataChannelMessage& message) {
    if (!open_.load()) {
        return false;
    }
    
    DataChannelMessage msg = message;
    msg.metadata().sequence_number = next_send_seq_.fetch_add(1);
    msg.metadata().delivery_mode = config_.delivery_mode;
    
    // Check if message needs fragmentation
    if (should_fragment_message(msg)) {
        auto fragments = fragment_message(msg);
        bool all_sent = true;
        
        for (const auto& fragment : fragments) {
            MessagePriority priority = fragment.priority();
            std::lock_guard<std::mutex> lock(send_queues_[static_cast<int>(priority)].mutex);
            
            if (send_queues_[static_cast<int>(priority)].queue.size() < 1000) {
                send_queues_[static_cast<int>(priority)].queue.push(fragment);
            } else {
                all_sent = false;
                update_error_stats("send_buffer_full");
            }
        }
        return all_sent;
    } else {
        MessagePriority priority = msg.priority();
        std::lock_guard<std::mutex> lock(send_queues_[static_cast<int>(priority)].mutex);
        
        if (send_queues_[static_cast<int>(priority)].queue.size() < 1000) {
            send_queues_[static_cast<int>(priority)].queue.push(msg);
            return true;
        } else {
            update_error_stats("send_buffer_full");
            return false;
        }
    }
}

bool EnhancedDataChannel::send_text(const std::string& text, MessagePriority priority) {
    DataChannelMessage msg(text, priority);
    return send_message(msg);
}

bool EnhancedDataChannel::send_binary(const void* data, size_t size, MessagePriority priority) {
    DataChannelMessage msg(data, size, priority);
    return send_message(msg);
}

bool EnhancedDataChannel::send_binary(const std::vector<uint8_t>& data, MessagePriority priority) {
    DataChannelMessage msg(data, priority);
    return send_message(msg);
}

bool EnhancedDataChannel::recv_message(DataChannelMessage& message) {
    std::lock_guard<std::mutex> lock(recv_queue_mutex_);
    
    if (!recv_queue_.empty()) {
        message = recv_queue_.front();
        recv_queue_.pop_front();
        return true;
    }
    
    return false;
}

std::string EnhancedDataChannel::recv_text() {
    DataChannelMessage msg;
    if (recv_message(msg) && msg.is_text()) {
        return msg.to_text();
    }
    return "";
}

std::vector<uint8_t> EnhancedDataChannel::recv_binary() {
    DataChannelMessage msg;
    if (recv_message(msg) && msg.is_binary()) {
        return msg.binary_data();
    }
    return {};
}

bool EnhancedDataChannel::try_send_message(const DataChannelMessage& message) {
    return send_message(message); // Same implementation for now
}

bool EnhancedDataChannel::try_recv_message(DataChannelMessage& message) {
    return recv_message(message); // Same implementation for now
}

DataChannelStats EnhancedDataChannel::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void EnhancedDataChannel::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = DataChannelStats{};
}

void EnhancedDataChannel::run_sender() {
    while (running_.load()) {
        DataChannelMessage msg = get_next_message_to_send();
        
        if (!msg.empty()) {
            // Simulate sending
            update_send_stats(msg);
            
            // Track for reliability if needed
            if (config_.delivery_mode == DeliveryMode::RELIABLE_ORDERED ||
                config_.delivery_mode == DeliveryMode::RELIABLE_UNORDERED) {
                
                std::lock_guard<std::mutex> lock(pending_messages_mutex_);
                PendingMessage pending;
                pending.message = msg;
                pending.send_time = std::chrono::steady_clock::now();
                pending_messages_[msg.metadata().sequence_number] = pending;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void EnhancedDataChannel::run_receiver() {
    while (running_.load()) {
        // Simulate receiving messages
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void EnhancedDataChannel::run_retransmit_timer() {
    while (running_.load()) {
        auto now = std::chrono::steady_clock::now();
        
        std::lock_guard<std::mutex> lock(pending_messages_mutex_);
        for (auto it = pending_messages_.begin(); it != pending_messages_.end();) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second.send_time);
            
            if (elapsed > config_.ack_timeout) {
                if (should_retransmit(it->second)) {
                    retransmit_message(it->first);
                    it->second.retransmit_count++;
                    it->second.send_time = now;
                    ++it;
                } else {
                    // Give up on this message
                    it = pending_messages_.erase(it);
                }
            } else {
                ++it;
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void EnhancedDataChannel::run_stats_updater() {
    while (running_.load()) {
        // Update statistics
        if (on_stats_update) {
            on_stats_update(get_stats());
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

DataChannelMessage EnhancedDataChannel::get_next_message_to_send() {
    // Priority-based message selection
    for (int priority = 0; priority < 5; ++priority) {
        std::lock_guard<std::mutex> lock(send_queues_[priority].mutex);
        
        if (!send_queues_[priority].queue.empty()) {
            auto msg = send_queues_[priority].queue.top();
            send_queues_[priority].queue.pop();
            return msg;
        }
    }
    
    return DataChannelMessage{};
}

bool EnhancedDataChannel::should_fragment_message(const DataChannelMessage& message) const {
    return message.size() > config_.max_message_size;
}

std::vector<DataChannelMessage> EnhancedDataChannel::fragment_message(const DataChannelMessage& message) {
    std::vector<DataChannelMessage> fragments;
    
    size_t fragment_size = config_.max_message_size - 100; // Leave room for headers
    uint16_t fragment_id = generate_fragment_id();
    uint16_t total_fragments = (message.size() + fragment_size - 1) / fragment_size;
    
    for (uint16_t i = 0; i < total_fragments; ++i) {
        size_t offset = i * fragment_size;
        size_t size = std::min(fragment_size, message.size() - offset);
        
        std::vector<uint8_t> fragment_data(message.data() + offset, message.data() + offset + size);
        DataChannelMessage fragment(fragment_data, message.priority());
        
        fragment.metadata().is_fragmented = true;
        fragment.metadata().fragment_id = fragment_id;
        fragment.metadata().fragment_index = i;
        fragment.metadata().total_fragments = total_fragments;
        
        fragments.push_back(fragment);
    }
    
    return fragments;
}

uint16_t EnhancedDataChannel::generate_fragment_id() {
    static std::atomic<uint16_t> fragment_id_counter{1};
    return fragment_id_counter.fetch_add(1);
}

bool EnhancedDataChannel::should_retransmit(const PendingMessage& pending) const {
    return pending.retransmit_count < 3; // Max 3 retransmissions
}

void EnhancedDataChannel::retransmit_message(uint64_t sequence_number) {
    // Simulate retransmission
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.messages_retransmitted++;
}

void EnhancedDataChannel::update_send_stats(const DataChannelMessage& message) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.messages_sent++;
    stats_.bytes_sent += message.size();
}

void EnhancedDataChannel::update_recv_stats(const DataChannelMessage& message) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.messages_received++;
    stats_.bytes_received += message.size();
}

void EnhancedDataChannel::update_error_stats(const std::string& error_type) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (error_type == "send_buffer_full") {
        stats_.buffer_full_errors++;
    } else if (error_type == "timeout") {
        stats_.timeout_errors++;
    } else {
        stats_.send_errors++;
    }
}

// DataChannelFactory implementation

std::unique_ptr<EnhancedDataChannel> DataChannelFactory::create_gaming_channel(const std::string& label) {
    DataChannelConfig config;
    config.label = label;
    config.delivery_mode = DeliveryMode::UNRELIABLE_ORDERED;
    config.default_priority = MessagePriority::HIGH;
    config.max_message_size = 1024;
    config.enable_nagle = false;
    config.ack_timeout = std::chrono::milliseconds(50);
    
    return std::make_unique<EnhancedDataChannel>(config);
}

std::unique_ptr<EnhancedDataChannel> DataChannelFactory::create_streaming_channel(const std::string& label) {
    DataChannelConfig config;
    config.label = label;
    config.delivery_mode = DeliveryMode::RELIABLE_ORDERED;
    config.default_priority = MessagePriority::NORMAL;
    config.max_message_size = 32768;
    config.send_buffer_size = 1048576; // 1MB
    
    return std::make_unique<EnhancedDataChannel>(config);
}

std::unique_ptr<EnhancedDataChannel> DataChannelFactory::create_file_transfer_channel(const std::string& label) {
    DataChannelConfig config;
    config.label = label;
    config.delivery_mode = DeliveryMode::RELIABLE_ORDERED;
    config.default_priority = MessagePriority::BULK;
    config.max_message_size = 65536;
    config.send_buffer_size = 2097152; // 2MB
    
    return std::make_unique<EnhancedDataChannel>(config);
}

std::unique_ptr<EnhancedDataChannel> DataChannelFactory::create_chat_channel(const std::string& label) {
    DataChannelConfig config;
    config.label = label;
    config.delivery_mode = DeliveryMode::RELIABLE_ORDERED;
    config.default_priority = MessagePriority::NORMAL;
    config.max_message_size = 4096;
    
    return std::make_unique<EnhancedDataChannel>(config);
}

std::unique_ptr<EnhancedDataChannel> DataChannelFactory::create_control_channel(const std::string& label) {
    DataChannelConfig config;
    config.label = label;
    config.delivery_mode = DeliveryMode::RELIABLE_ORDERED;
    config.default_priority = MessagePriority::CRITICAL;
    config.max_message_size = 512;
    config.ack_timeout = std::chrono::milliseconds(100);
    
    return std::make_unique<EnhancedDataChannel>(config);
}

std::unique_ptr<EnhancedDataChannel> DataChannelFactory::create_custom_channel(const DataChannelConfig& config) {
    return std::make_unique<EnhancedDataChannel>(config);
}

// DataChannelMultiplexer implementation

DataChannelMultiplexer::~DataChannelMultiplexer() {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    channels_.clear();
}

void DataChannelMultiplexer::add_channel(const std::string& name, std::unique_ptr<EnhancedDataChannel> channel) {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    
    setup_channel_callbacks(name, channel.get());
    channels_[name] = std::move(channel);
}

void DataChannelMultiplexer::remove_channel(const std::string& name) {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    channels_.erase(name);
}

EnhancedDataChannel* DataChannelMultiplexer::get_channel(const std::string& name) {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    
    auto it = channels_.find(name);
    return (it != channels_.end()) ? it->second.get() : nullptr;
}

std::vector<std::string> DataChannelMultiplexer::get_channel_names() const {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    
    std::vector<std::string> names;
    for (const auto& pair : channels_) {
        names.push_back(pair.first);
    }
    return names;
}

bool DataChannelMultiplexer::broadcast_text(const std::string& text, MessagePriority priority) {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    
    bool all_sent = true;
    for (const auto& pair : channels_) {
        if (!pair.second->send_text(text, priority)) {
            all_sent = false;
        }
    }
    return all_sent;
}

bool DataChannelMultiplexer::broadcast_binary(const void* data, size_t size, MessagePriority priority) {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    
    bool all_sent = true;
    for (const auto& pair : channels_) {
        if (!pair.second->send_binary(data, size, priority)) {
            all_sent = false;
        }
    }
    return all_sent;
}

bool DataChannelMultiplexer::broadcast_message(const DataChannelMessage& message) {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    
    bool all_sent = true;
    for (const auto& pair : channels_) {
        if (!pair.second->send_message(message)) {
            all_sent = false;
        }
    }
    return all_sent;
}

DataChannelStats DataChannelMultiplexer::get_aggregated_stats() const {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    
    DataChannelStats aggregated{};
    for (const auto& pair : channels_) {
        auto stats = pair.second->get_stats();
        aggregated.messages_sent += stats.messages_sent;
        aggregated.messages_received += stats.messages_received;
        aggregated.bytes_sent += stats.bytes_sent;
        aggregated.bytes_received += stats.bytes_received;
        // ... aggregate other stats
    }
    return aggregated;
}

void DataChannelMultiplexer::setup_channel_callbacks(const std::string& name, EnhancedDataChannel* channel) {
    channel->on_message = [this, name](const DataChannelMessage& msg) {
        if (on_message) {
            on_message(name, msg);
        }
    };
    
    channel->on_error = [this, name](const std::string& error) {
        if (on_error) {
            on_error(name, error);
        }
    };
}

} // namespace webrtc
} // namespace detail
} // namespace psyne