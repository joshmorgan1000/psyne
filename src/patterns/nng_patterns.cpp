/**
 * @file nng_patterns.cpp
 * @brief Implementation of Nanomsg/NNG-style messaging patterns
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include "nng_patterns.hpp"
#include <psyne/channel.hpp>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace psyne {
namespace patterns {
namespace nng {

// NNGMessage implementation

NNGMessage::NNGMessage(const void* data, size_t size) 
    : data_(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size) {}

NNGMessage::NNGMessage(const std::string& data) 
    : data_(data.begin(), data.end()) {}

NNGMessage::NNGMessage(std::vector<uint8_t> data) 
    : data_(std::move(data)) {}

void NNGMessage::set_header(const std::string& key, const std::string& value) {
    headers_[key] = value;
}

std::string NNGMessage::get_header(const std::string& key) const {
    auto it = headers_.find(key);
    return (it != headers_.end()) ? it->second : "";
}

bool NNGMessage::has_header(const std::string& key) const {
    return headers_.find(key) != headers_.end();
}

std::string NNGMessage::to_string() const {
    return std::string(data_.begin(), data_.end());
}

void NNGMessage::from_string(const std::string& str) {
    data_.assign(str.begin(), str.end());
}

std::vector<uint8_t> NNGMessage::serialize() const {
    std::vector<uint8_t> result;
    
    // Simple serialization format:
    // [hop_count:4][pipe_id:4][headers_size:4][headers][data_size:4][data]
    
    // Write hop count
    uint32_t hop_count = hop_count_;
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&hop_count), 
                  reinterpret_cast<uint8_t*>(&hop_count) + 4);
    
    // Write pipe ID
    uint32_t pipe_id = pipe_id_;
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&pipe_id),
                  reinterpret_cast<uint8_t*>(&pipe_id) + 4);
    
    // Serialize headers
    std::string headers_str;
    for (const auto& [key, value] : headers_) {
        headers_str += key + ":" + value + "\n";
    }
    
    uint32_t headers_size = static_cast<uint32_t>(headers_str.size());
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&headers_size),
                  reinterpret_cast<uint8_t*>(&headers_size) + 4);
    result.insert(result.end(), headers_str.begin(), headers_str.end());
    
    // Write data size and data
    uint32_t data_size = static_cast<uint32_t>(data_.size());
    result.insert(result.end(), reinterpret_cast<uint8_t*>(&data_size),
                  reinterpret_cast<uint8_t*>(&data_size) + 4);
    result.insert(result.end(), data_.begin(), data_.end());
    
    return result;
}

bool NNGMessage::deserialize(const std::vector<uint8_t>& data) {
    if (data.size() < 16) return false; // Minimum size for headers
    
    size_t offset = 0;
    
    // Read hop count
    if (offset + 4 > data.size()) return false;
    hop_count_ = *reinterpret_cast<const uint32_t*>(data.data() + offset);
    offset += 4;
    
    // Read pipe ID
    if (offset + 4 > data.size()) return false;
    pipe_id_ = *reinterpret_cast<const uint32_t*>(data.data() + offset);
    offset += 4;
    
    // Read headers
    if (offset + 4 > data.size()) return false;
    uint32_t headers_size = *reinterpret_cast<const uint32_t*>(data.data() + offset);
    offset += 4;
    
    if (offset + headers_size > data.size()) return false;
    std::string headers_str(data.begin() + offset, data.begin() + offset + headers_size);
    offset += headers_size;
    
    // Parse headers
    headers_.clear();
    std::istringstream headers_stream(headers_str);
    std::string line;
    while (std::getline(headers_stream, line)) {
        auto colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            headers_[key] = value;
        }
    }
    
    // Read data
    if (offset + 4 > data.size()) return false;
    uint32_t data_size = *reinterpret_cast<const uint32_t*>(data.data() + offset);
    offset += 4;
    
    if (offset + data_size > data.size()) return false;
    data_.assign(data.begin() + offset, data.begin() + offset + data_size);
    
    return true;
}

// NNGSocket implementation

NNGSocket::NNGSocket(Protocol protocol, SocketRole role, const SocketOptions& options)
    : protocol_(protocol), role_(role), options_(options) {
    
    worker_thread_ = std::thread(&NNGSocket::worker_loop, this);
}

NNGSocket::~NNGSocket() {
    close();
}

bool NNGSocket::listen(const std::string& url) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        // Parse URL and create appropriate channel
        // For simplicity, we'll use TCP for demonstration
        auto channel = std::make_unique<Channel>(url, options_.recv_buffer_size);
        
        if (add_channel(std::move(channel))) {
            listening_ = true;
            update_stats_send(0); // Mark connection event
            return true;
        }
    } catch (const std::exception& e) {
        update_stats_error();
        std::cerr << "Failed to listen on " << url << ": " << e.what() << std::endl;
    }
    
    return false;
}

bool NNGSocket::dial(const std::string& url) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        auto channel = std::make_unique<Channel>(url, options_.send_buffer_size);
        
        if (add_channel(std::move(channel))) {
            update_stats_send(0); // Mark connection event
            return true;
        }
    } catch (const std::exception& e) {
        update_stats_error();
        std::cerr << "Failed to dial " << url << ": " << e.what() << std::endl;
    }
    
    return false;
}

void NNGSocket::close() {
    running_ = false;
    cv_.notify_all();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    channels_.clear();
    listening_ = false;
}

bool NNGSocket::send(const NNGMessage& msg) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (!running_) return false;
    
    // Check send buffer size
    if (send_queue_.size() >= static_cast<size_t>(options_.send_buffer_size)) {
        if (options_.send_timeout.count() == 0) return false;
        
        auto timeout_time = std::chrono::steady_clock::now() + options_.send_timeout;
        if (!cv_.wait_until(lock, timeout_time, [this] { 
            return send_queue_.size() < static_cast<size_t>(options_.send_buffer_size) || !running_; 
        })) {
            update_stats_timeout();
            return false;
        }
        
        if (!running_) return false;
    }
    
    send_queue_.push(msg);
    cv_.notify_one();
    
    update_stats_send(msg.size());
    return true;
}

bool NNGSocket::send(const std::string& data) {
    return send(NNGMessage(data));
}

bool NNGSocket::send(const void* data, size_t size) {
    return send(NNGMessage(data, size));
}

bool NNGSocket::recv(NNGMessage& msg) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (!running_) return false;
    
    // Wait for message or timeout
    if (recv_queue_.empty()) {
        if (options_.recv_timeout.count() == 0) return false;
        
        auto timeout_time = std::chrono::steady_clock::now() + options_.recv_timeout;
        if (!cv_.wait_until(lock, timeout_time, [this] { 
            return !recv_queue_.empty() || !running_; 
        })) {
            update_stats_timeout();
            return false;
        }
        
        if (!running_ || recv_queue_.empty()) return false;
    }
    
    msg = std::move(recv_queue_.front());
    recv_queue_.pop();
    
    update_stats_recv(msg.size());
    return true;
}

std::string NNGSocket::recv_string() {
    NNGMessage msg;
    if (recv(msg)) {
        return msg.to_string();
    }
    return "";
}

bool NNGSocket::try_send(const NNGMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!running_ || send_queue_.size() >= static_cast<size_t>(options_.send_buffer_size)) {
        return false;
    }
    
    send_queue_.push(msg);
    cv_.notify_one();
    
    update_stats_send(msg.size());
    return true;
}

bool NNGSocket::try_recv(NNGMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!running_ || recv_queue_.empty()) {
        return false;
    }
    
    msg = std::move(recv_queue_.front());
    recv_queue_.pop();
    
    update_stats_recv(msg.size());
    return true;
}

void NNGSocket::set_option(const SocketOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    options_ = options;
}

SocketOptions NNGSocket::get_options() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return options_;
}

NNGSocket::Stats NNGSocket::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void NNGSocket::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Stats{};
}

void NNGSocket::worker_loop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Process outgoing messages
        while (!send_queue_.empty() && running_) {
            NNGMessage msg = std::move(send_queue_.front());
            send_queue_.pop();
            lock.unlock();
            
            route_outgoing_message(msg);
            
            lock.lock();
        }
        
        // Clean up disconnected channels
        remove_disconnected_channels();
        
        // Wait for work or timeout
        cv_.wait_for(lock, std::chrono::milliseconds(10));
    }
}

void NNGSocket::update_stats_send(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.messages_sent++;
    stats_.bytes_sent += bytes;
}

void NNGSocket::update_stats_recv(size_t bytes) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.messages_received++;
    stats_.bytes_received += bytes;
}

void NNGSocket::update_stats_error() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.errors++;
}

void NNGSocket::update_stats_timeout() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.timeouts++;
}

bool NNGSocket::add_channel(std::unique_ptr<Channel> channel) {
    if (!channel) return false;
    
    channels_.push_back(std::move(channel));
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.connect_events++;
    }
    return true;
}

void NNGSocket::remove_disconnected_channels() {
    // Remove channels that are no longer connected
    // This is a simplified implementation
    channels_.erase(
        std::remove_if(channels_.begin(), channels_.end(),
                      [](const std::unique_ptr<Channel>& channel) {
                          return !channel; // In real implementation, check if channel is connected
                      }),
        channels_.end());
}

// PipelinePush implementation

PipelinePush::PipelinePush(const SocketOptions& options)
    : NNGSocket(Protocol::PIPELINE, SocketRole::PUSH, options) {}

bool PipelinePush::send_work(const NNGMessage& work) {
    return send(work);
}

bool PipelinePush::send_work(const std::string& work_data) {
    return send(work_data);
}

size_t PipelinePush::active_workers() const {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    return worker_addresses_.size();
}

std::vector<std::string> PipelinePush::get_worker_addresses() const {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    return worker_addresses_;
}

bool PipelinePush::route_outgoing_message(const NNGMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (channels_.empty()) return false;
    
    // Load balance across available channels
    size_t channel_index = current_worker_.fetch_add(1) % channels_.size();
    auto& channel = channels_[channel_index];
    
    if (channel) {
        auto serialized = msg.serialize();
        return channel->send(serialized.data(), serialized.size()) > 0;
    }
    
    return false;
}

bool PipelinePush::handle_incoming_message(NNGMessage&& msg) {
    // Pipeline Push sockets don't receive messages
    return false;
}

// PipelinePull implementation

PipelinePull::PipelinePull(const SocketOptions& options)
    : NNGSocket(Protocol::PIPELINE, SocketRole::PULL, options) {}

bool PipelinePull::recv_work(NNGMessage& work) {
    return recv(work);
}

std::string PipelinePull::recv_work_string() {
    return recv_string();
}

bool PipelinePull::send_completion_ack(const std::string& work_id) {
    NNGMessage ack_msg("ACK:" + work_id);
    ack_msg.set_header("type", "completion_ack");
    return send(ack_msg);
}

bool PipelinePull::route_outgoing_message(const NNGMessage& msg) {
    // Pipeline Pull sockets typically don't send messages (except acks)
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!channels_.empty()) {
        auto& channel = channels_[0]; // Send to first available channel
        if (channel) {
            auto serialized = msg.serialize();
            return channel->send(serialized.data(), serialized.size()) > 0;
        }
    }
    
    return false;
}

bool PipelinePull::handle_incoming_message(NNGMessage&& msg) {
    std::lock_guard<std::mutex> lock(work_mutex_);
    work_queue_.push(std::move(msg));
    cv_.notify_one();
    return true;
}

// SurveySurveyor implementation

SurveySurveyor::SurveySurveyor(const SocketOptions& options)
    : NNGSocket(Protocol::SURVEY, SocketRole::SURVEYOR, options) {}

bool SurveySurveyor::send_survey(const NNGMessage& survey) {
    // Add survey ID for response correlation
    NNGMessage survey_msg = survey;
    uint32_t survey_id = survey_id_.fetch_add(1);
    survey_msg.set_header("survey_id", std::to_string(survey_id));
    survey_msg.set_header("type", "survey");
    
    collecting_responses_ = true;
    return send(survey_msg);
}

bool SurveySurveyor::send_survey(const std::string& question) {
    return send_survey(NNGMessage(question));
}

std::vector<NNGMessage> SurveySurveyor::collect_responses(std::chrono::milliseconds timeout) {
    std::vector<NNGMessage> responses;
    auto start_time = std::chrono::steady_clock::now();
    
    while (std::chrono::steady_clock::now() - start_time < timeout) {
        std::unique_lock<std::mutex> lock(responses_mutex_);
        
        if (responses_cv_.wait_for(lock, std::chrono::milliseconds(10), [this] { 
            return !responses_.empty(); 
        })) {
            while (!responses_.empty()) {
                responses.push_back(std::move(responses_.front()));
                responses_.pop();
            }
        }
    }
    
    collecting_responses_ = false;
    return responses;
}

std::vector<std::string> SurveySurveyor::collect_response_strings(std::chrono::milliseconds timeout) {
    auto responses = collect_responses(timeout);
    std::vector<std::string> string_responses;
    string_responses.reserve(responses.size());
    
    for (const auto& response : responses) {
        string_responses.push_back(response.to_string());
    }
    
    return string_responses;
}

SurveySurveyor::SurveyResult SurveySurveyor::conduct_survey(const NNGMessage& survey, std::chrono::milliseconds timeout) {
    auto start_time = std::chrono::steady_clock::now();
    
    SurveyResult result;
    result.respondents_contacted = channels_.size(); // Approximate
    
    if (!send_survey(survey)) {
        result.timed_out = true;
        return result;
    }
    
    result.responses = collect_responses(timeout);
    result.responses_received = result.responses.size();
    
    auto end_time = std::chrono::steady_clock::now();
    result.collection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.timed_out = (result.collection_time >= timeout);
    
    return result;
}

SurveySurveyor::SurveyResult SurveySurveyor::conduct_survey(const std::string& question, std::chrono::milliseconds timeout) {
    return conduct_survey(NNGMessage(question), timeout);
}

bool SurveySurveyor::route_outgoing_message(const NNGMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Broadcast survey to all connected respondents
    bool success = true;
    for (auto& channel : channels_) {
        if (channel) {
            auto serialized = msg.serialize();
            success &= (channel->send(serialized.data(), serialized.size()) > 0);
        }
    }
    
    return success;
}

bool SurveySurveyor::handle_incoming_message(NNGMessage&& msg) {
    if (collecting_responses_ && msg.has_header("type") && msg.get_header("type") == "survey_response") {
        std::lock_guard<std::mutex> lock(responses_mutex_);
        responses_.push(std::move(msg));
        responses_cv_.notify_one();
        return true;
    }
    
    return false;
}

// SurveyRespondent implementation

SurveyRespondent::SurveyRespondent(const SocketOptions& options)
    : NNGSocket(Protocol::SURVEY, SocketRole::RESPONDENT, options) {}

bool SurveyRespondent::recv_survey(NNGMessage& survey) {
    std::unique_lock<std::mutex> lock(surveys_mutex_);
    
    if (surveys_.empty()) {
        cv_.wait(lock, [this] { return !surveys_.empty() || !running_; });
    }
    
    if (!running_ || surveys_.empty()) return false;
    
    survey = std::move(surveys_.front());
    surveys_.pop();
    return true;
}

std::string SurveyRespondent::recv_survey_string() {
    NNGMessage survey;
    if (recv_survey(survey)) {
        return survey.to_string();
    }
    return "";
}

bool SurveyRespondent::send_response(const NNGMessage& response) {
    NNGMessage response_msg = response;
    response_msg.set_header("type", "survey_response");
    response_msg.set_header("survey_id", std::to_string(current_survey_id_));
    return send(response_msg);
}

bool SurveyRespondent::send_response(const std::string& response_data) {
    return send_response(NNGMessage(response_data));
}

void SurveyRespondent::set_survey_handler(SurveyHandler handler) {
    survey_handler_ = handler;
}

void SurveyRespondent::enable_auto_response(bool enable) {
    auto_response_enabled_ = enable;
}

bool SurveyRespondent::route_outgoing_message(const NNGMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!channels_.empty()) {
        auto& channel = channels_[0]; // Send response back through first channel
        if (channel) {
            auto serialized = msg.serialize();
            return channel->send(serialized.data(), serialized.size()) > 0;
        }
    }
    
    return false;
}

bool SurveyRespondent::handle_incoming_message(NNGMessage&& msg) {
    if (msg.has_header("type") && msg.get_header("type") == "survey") {
        if (msg.has_header("survey_id")) {
            current_survey_id_ = std::stoul(msg.get_header("survey_id"));
        }
        
        if (auto_response_enabled_ && survey_handler_) {
            // Automatically respond using the handler
            auto response = survey_handler_(msg);
            send_response(response);
        } else {
            // Queue for manual processing
            std::lock_guard<std::mutex> lock(surveys_mutex_);
            surveys_.push(std::move(msg));
            cv_.notify_one();
        }
        return true;
    }
    
    return false;
}

// BusSocket implementation

BusSocket::BusSocket(const SocketOptions& options)
    : NNGSocket(Protocol::BUS, SocketRole::BUS_NODE, options) {
    
    // Generate unique local ID
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    
    std::ostringstream oss;
    oss << std::hex << dis(gen);
    local_id_ = oss.str();
}

bool BusSocket::send_broadcast(const NNGMessage& msg) {
    NNGMessage broadcast_msg = msg;
    broadcast_msg.set_header("type", "broadcast");
    broadcast_msg.set_header("from", local_id_);
    broadcast_msg.set_header("msg_id", generate_message_id());
    broadcast_msg.set_hop_count(0);
    
    return send(broadcast_msg);
}

bool BusSocket::send_broadcast(const std::string& data) {
    return send_broadcast(NNGMessage(data));
}

bool BusSocket::send_to_peer(const std::string& peer_id, const NNGMessage& msg) {
    NNGMessage peer_msg = msg;
    peer_msg.set_header("type", "peer_message");
    peer_msg.set_header("from", local_id_);
    peer_msg.set_header("to", peer_id);
    peer_msg.set_header("msg_id", generate_message_id());
    peer_msg.set_hop_count(0);
    
    return send(peer_msg);
}

bool BusSocket::send_to_peer(const std::string& peer_id, const std::string& data) {
    return send_to_peer(peer_id, NNGMessage(data));
}

std::vector<std::string> BusSocket::get_connected_peers() const {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    std::vector<std::string> peer_list;
    
    for (const auto& [peer_id, timestamp] : peers_) {
        // Consider peers connected if we've heard from them recently
        auto now = std::chrono::steady_clock::now();
        if (now - timestamp < std::chrono::seconds(30)) {
            peer_list.push_back(peer_id);
        }
    }
    
    return peer_list;
}

bool BusSocket::is_peer_connected(const std::string& peer_id) const {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    auto it = peers_.find(peer_id);
    if (it == peers_.end()) return false;
    
    auto now = std::chrono::steady_clock::now();
    return (now - it->second) < std::chrono::seconds(30);
}

BusSocket::TopologyInfo BusSocket::get_topology() const {
    TopologyInfo info;
    info.local_id = local_id_;
    info.direct_peers = get_connected_peers();
    
    std::lock_guard<std::mutex> lock(peers_mutex_);
    info.reachable_peers = info.direct_peers; // Simplified
    info.peer_distances = peer_hop_counts_;
    
    return info;
}

void BusSocket::set_peer_event_handler(PeerEventHandler handler) {
    peer_event_handler_ = handler;
}

void BusSocket::set_message_handler(MessageHandler handler) {
    message_handler_ = handler;
}

bool BusSocket::route_outgoing_message(const NNGMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Forward to all connected peers
    bool success = true;
    for (auto& channel : channels_) {
        if (channel) {
            auto serialized = msg.serialize();
            success &= (channel->send(serialized.data(), serialized.size()) > 0);
        }
    }
    
    return success;
}

bool BusSocket::handle_incoming_message(NNGMessage&& msg) {
    // Check for message loops
    std::string msg_id = msg.get_header("msg_id");
    if (!msg_id.empty() && is_message_seen(msg_id)) {
        return false; // Drop duplicate message
    }
    
    // Update hop count
    uint32_t hops = msg.hop_count() + 1;
    if (hops > options_.max_ttl) {
        return false; // Drop message that has traveled too far
    }
    msg.set_hop_count(hops);
    
    std::string from_peer = msg.get_header("from");
    std::string msg_type = msg.get_header("type");
    
    // Update peer information
    if (!from_peer.empty()) {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        peers_[from_peer] = std::chrono::steady_clock::now();
        peer_hop_counts_[from_peer] = std::min(peer_hop_counts_[from_peer], hops);
        
        if (peer_event_handler_) {
            peer_event_handler_(from_peer, true);
        }
    }
    
    // Handle different message types
    if (msg_type == "broadcast") {
        // Forward broadcast to other peers (with increased hop count)
        route_outgoing_message(msg);
        
        if (message_handler_) {
            message_handler_(from_peer, msg);
        }
    } else if (msg_type == "peer_message") {
        std::string to_peer = msg.get_header("to");
        if (to_peer == local_id_) {
            // Message is for us
            if (message_handler_) {
                message_handler_(from_peer, msg);
            }
        } else {
            // Forward to destination (simplified routing)
            route_outgoing_message(msg);
        }
    }
    
    return true;
}

std::string BusSocket::generate_message_id() const {
    static std::atomic<uint64_t> counter{0};
    return local_id_ + "_" + std::to_string(counter.fetch_add(1));
}

bool BusSocket::is_message_seen(const std::string& msg_id) {
    std::lock_guard<std::mutex> lock(seen_messages_mutex_);
    
    if (seen_messages_.find(msg_id) != seen_messages_.end()) {
        return true;
    }
    
    seen_messages_.insert(msg_id);
    
    // Keep only recent messages (simple cleanup)
    if (seen_messages_.size() > 1000) {
        seen_messages_.clear();
    }
    
    return false;
}

void BusSocket::update_peer_topology(const NNGMessage& msg) {
    // Implementation for topology updates
}

void BusSocket::broadcast_topology_update() {
    // Implementation for broadcasting topology changes
}

// High-level pattern implementations

// WorkDistributor implementation

WorkDistributor::WorkDistributor(const std::string& bind_address, const SocketOptions& options) {
    push_socket_ = std::make_unique<PipelinePush>(options);
    if (!push_socket_->listen(bind_address)) {
        throw std::runtime_error("Failed to bind WorkDistributor to " + bind_address);
    }
}

WorkDistributor::~WorkDistributor() = default;

bool WorkDistributor::submit_work(const NNGMessage& work) {
    if (push_socket_->send_work(work)) {
        work_submitted_.fetch_add(1);
        return true;
    }
    return false;
}

bool WorkDistributor::submit_work(const std::string& work_data) {
    return submit_work(NNGMessage(work_data));
}

bool WorkDistributor::submit_work_batch(const std::vector<NNGMessage>& work_items) {
    bool success = true;
    for (const auto& work : work_items) {
        success &= submit_work(work);
    }
    return success;
}

bool WorkDistributor::submit_work_batch(const std::vector<std::string>& work_data) {
    bool success = true;
    for (const auto& work : work_data) {
        success &= submit_work(work);
    }
    return success;
}

size_t WorkDistributor::pending_work_count() const {
    // This would require more sophisticated tracking in a real implementation
    return 0;
}

size_t WorkDistributor::active_workers() const {
    return push_socket_->active_workers();
}

std::vector<std::string> WorkDistributor::get_worker_info() const {
    return push_socket_->get_worker_addresses();
}

// WorkProcessor implementation

WorkProcessor::WorkProcessor(const std::string& connect_address, WorkHandler handler, const SocketOptions& options)
    : work_handler_(handler) {
    
    pull_socket_ = std::make_unique<PipelinePull>(options);
    if (!pull_socket_->dial(connect_address)) {
        throw std::runtime_error("Failed to connect WorkProcessor to " + connect_address);
    }
}

WorkProcessor::~WorkProcessor() {
    stop_processing();
}

void WorkProcessor::start_processing() {
    if (!processing_) {
        processing_ = true;
        processing_thread_ = std::thread(&WorkProcessor::processing_loop, this);
    }
}

void WorkProcessor::stop_processing() {
    if (processing_) {
        processing_ = false;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }
}

bool WorkProcessor::is_processing() const {
    return processing_;
}

void WorkProcessor::processing_loop() {
    while (processing_) {
        NNGMessage work;
        if (pull_socket_->recv_work(work)) {
            if (work_handler_(work)) {
                work_processed_.fetch_add(1);
            } else {
                work_failed_.fetch_add(1);
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

// QueryEngine implementation

QueryEngine::QueryEngine(const std::string& bind_address, const SocketOptions& options) {
    surveyor_socket_ = std::make_unique<SurveySurveyor>(options);
    if (!surveyor_socket_->listen(bind_address)) {
        throw std::runtime_error("Failed to bind QueryEngine to " + bind_address);
    }
}

QueryEngine::~QueryEngine() = default;

std::vector<NNGMessage> QueryEngine::query_all(const NNGMessage& query, std::chrono::milliseconds timeout) {
    auto result = surveyor_socket_->conduct_survey(query, timeout);
    return result.responses;
}

std::vector<std::string> QueryEngine::query_all_string(const std::string& query, std::chrono::milliseconds timeout) {
    auto result = surveyor_socket_->conduct_survey(query, timeout);
    std::vector<std::string> string_responses;
    
    for (const auto& response : result.responses) {
        string_responses.push_back(response.to_string());
    }
    
    return string_responses;
}

// Template specializations would go here for aggregate_query

double QueryEngine::sum_query(const std::string& query, std::chrono::milliseconds timeout) {
    auto responses = query_all_string(query, timeout);
    double sum = 0.0;
    
    for (const auto& response : responses) {
        try {
            sum += std::stod(response);
        } catch (const std::exception&) {
            // Skip invalid responses
        }
    }
    
    return sum;
}

double QueryEngine::avg_query(const std::string& query, std::chrono::milliseconds timeout) {
    auto responses = query_all_string(query, timeout);
    if (responses.empty()) return 0.0;
    
    return sum_query(query, timeout) / responses.size();
}

size_t QueryEngine::count_query(const std::string& query, std::chrono::milliseconds timeout) {
    auto responses = query_all_string(query, timeout);
    return responses.size();
}

// QueryResponder implementation

QueryResponder::QueryResponder(const std::string& connect_address, QueryHandler handler, const SocketOptions& options)
    : query_handler_(handler) {
    
    respondent_socket_ = std::make_unique<SurveyRespondent>(options);
    if (!respondent_socket_->dial(connect_address)) {
        throw std::runtime_error("Failed to connect QueryResponder to " + connect_address);
    }
}

QueryResponder::~QueryResponder() {
    stop_responding();
}

void QueryResponder::start_responding() {
    if (!responding_) {
        responding_ = true;
        responding_thread_ = std::thread(&QueryResponder::responding_loop, this);
    }
}

void QueryResponder::stop_responding() {
    if (responding_) {
        responding_ = false;
        if (responding_thread_.joinable()) {
            responding_thread_.join();
        }
    }
}

bool QueryResponder::is_responding() const {
    return responding_;
}

void QueryResponder::responding_loop() {
    while (responding_) {
        NNGMessage query;
        if (respondent_socket_->recv_survey(query)) {
            auto response = query_handler_(query);
            respondent_socket_->send_response(response);
            queries_handled_.fetch_add(1);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

// MeshNode implementation

MeshNode::MeshNode(const std::string& node_id, const std::string& bind_address, const SocketOptions& options)
    : node_id_(node_id) {
    
    bus_socket_ = std::make_unique<BusSocket>(options);
    if (!bus_socket_->listen(bind_address)) {
        throw std::runtime_error("Failed to bind MeshNode to " + bind_address);
    }
    
    bus_socket_->set_message_handler([this](const std::string& from_node, const NNGMessage& msg) {
        if (message_handler_) {
            message_handler_(from_node, msg);
        }
    });
    
    message_thread_ = std::thread(&MeshNode::message_loop, this);
}

MeshNode::~MeshNode() {
    leave_mesh();
}

bool MeshNode::join_mesh(const std::string& peer_address) {
    return bus_socket_->dial(peer_address);
}

bool MeshNode::leave_mesh() {
    running_ = false;
    if (message_thread_.joinable()) {
        message_thread_.join();
    }
    bus_socket_->close();
    return true;
}

bool MeshNode::broadcast(const NNGMessage& msg) {
    return bus_socket_->send_broadcast(msg);
}

bool MeshNode::broadcast(const std::string& data) {
    return bus_socket_->send_broadcast(data);
}

bool MeshNode::send_to_node(const std::string& node_id, const NNGMessage& msg) {
    return bus_socket_->send_to_peer(node_id, msg);
}

bool MeshNode::send_to_node(const std::string& node_id, const std::string& data) {
    return bus_socket_->send_to_peer(node_id, data);
}

void MeshNode::set_message_handler(MessageHandler handler) {
    message_handler_ = handler;
}

std::vector<std::string> MeshNode::get_peers() const {
    return bus_socket_->get_connected_peers();
}

BusSocket::TopologyInfo MeshNode::get_topology() const {
    return bus_socket_->get_topology();
}

bool MeshNode::ping_node(const std::string& node_id, std::chrono::milliseconds timeout) {
    // Send ping and wait for response
    NNGMessage ping_msg("PING");
    ping_msg.set_header("type", "ping");
    ping_msg.set_header("timestamp", std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()));
    
    return send_to_node(node_id, ping_msg);
}

std::vector<std::string> MeshNode::discover_nodes(std::chrono::milliseconds timeout) {
    // Broadcast discovery message and collect responses
    NNGMessage discovery_msg("DISCOVERY_REQUEST");
    discovery_msg.set_header("type", "discovery");
    
    broadcast(discovery_msg);
    
    // Wait for responses (simplified implementation)
    std::this_thread::sleep_for(timeout);
    
    return get_peers();
}

void MeshNode::message_loop() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // Process any internal message handling
    }
}

// Factory functions

std::unique_ptr<PipelinePush> create_pipeline_push(const std::string& bind_address, const SocketOptions& options) {
    auto socket = std::make_unique<PipelinePush>(options);
    if (!socket->listen(bind_address)) {
        return nullptr;
    }
    return socket;
}

std::unique_ptr<PipelinePull> create_pipeline_pull(const std::string& connect_address, const SocketOptions& options) {
    auto socket = std::make_unique<PipelinePull>(options);
    if (!socket->dial(connect_address)) {
        return nullptr;
    }
    return socket;
}

std::unique_ptr<SurveySurveyor> create_survey_surveyor(const std::string& bind_address, const SocketOptions& options) {
    auto socket = std::make_unique<SurveySurveyor>(options);
    if (!socket->listen(bind_address)) {
        return nullptr;
    }
    return socket;
}

std::unique_ptr<SurveyRespondent> create_survey_respondent(const std::string& connect_address, const SocketOptions& options) {
    auto socket = std::make_unique<SurveyRespondent>(options);
    if (!socket->dial(connect_address)) {
        return nullptr;
    }
    return socket;
}

std::unique_ptr<BusSocket> create_bus_socket(const std::string& node_id, const std::string& bind_address, const SocketOptions& options) {
    auto socket = std::make_unique<BusSocket>(options);
    if (!socket->listen(bind_address)) {
        return nullptr;
    }
    return socket;
}

} // namespace nng
} // namespace patterns
} // namespace psyne