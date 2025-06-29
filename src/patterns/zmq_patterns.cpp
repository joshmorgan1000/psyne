/**
 * @file zmq_patterns.cpp
 * @brief ZeroMQ-style messaging patterns implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include "zmq_patterns.hpp"
#include <psyne/psyne.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>

namespace psyne {
namespace patterns {

// MessageFrame implementation

MessageFrame::MessageFrame(const void* data, size_t size, bool more)
    : data(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size)
    , more(more) {
}

MessageFrame::MessageFrame(std::vector<uint8_t> data, bool more)
    : data(std::move(data)), more(more) {
}

MessageFrame::MessageFrame(const std::string& str, bool more)
    : data(str.begin(), str.end()), more(more) {
}

std::string MessageFrame::to_string() const {
    return std::string(data.begin(), data.end());
}

// Message implementation

Message::Message(const std::string& data) {
    add_frame(data);
}

Message::Message(const std::vector<std::string>& parts) {
    for (size_t i = 0; i < parts.size(); ++i) {
        add_frame(parts[i], i < parts.size() - 1);
    }
}

void Message::add_frame(const MessageFrame& frame) {
    frames_.push_back(frame);
}

void Message::add_frame(const void* data, size_t size, bool more) {
    frames_.emplace_back(data, size, more);
}

void Message::add_frame(const std::string& data, bool more) {
    frames_.emplace_back(data, more);
}

std::string Message::to_string() const {
    std::string result;
    for (const auto& frame : frames_) {
        if (!result.empty()) result += " | ";
        result += frame.to_string();
    }
    return result;
}

// Socket base implementation

Socket::Socket(SocketType type, const std::string& transport)
    : type_(type), transport_(transport) {
}

Socket::~Socket() {
    close();
}

bool Socket::bind(const std::string& endpoint) {
    try {
        // Parse endpoint (e.g., "tcp://*:5555")
        size_t proto_end = endpoint.find("://");
        if (proto_end == std::string::npos) {
            throw std::invalid_argument("Invalid endpoint format");
        }
        
        std::string protocol = endpoint.substr(0, proto_end);
        std::string address = endpoint.substr(proto_end + 3);
        
        // Create appropriate channel
        std::unique_ptr<Channel> channel;
        
        if (protocol == "tcp") {
            size_t colon_pos = address.find_last_of(':');
            if (colon_pos != std::string::npos) {
                std::string port_str = address.substr(colon_pos + 1);
                uint16_t port = static_cast<uint16_t>(std::stoi(port_str));
                channel = create_channel("tcp://0.0.0.0:" + port_str, 1024 * 1024);
            }
        } else if (protocol == "ipc") {
            channel = create_channel("ipc://" + address, 1024 * 1024);
        } else if (protocol == "inproc") {
            channel = create_channel("memory://" + address, 1024 * 1024);
        }
        
        if (channel) {
            channels_.push_back(std::move(channel));
            connected_ = true;
            
            // Start worker thread
            if (!worker_thread_.joinable()) {
                worker_thread_ = std::thread(&Socket::worker_loop, this);
            }
            
            return true;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to bind to " << endpoint << ": " << e.what() << std::endl;
    }
    
    return false;
}

bool Socket::connect(const std::string& endpoint) {
    try {
        // Parse endpoint
        size_t proto_end = endpoint.find("://");
        if (proto_end == std::string::npos) {
            throw std::invalid_argument("Invalid endpoint format");
        }
        
        std::string protocol = endpoint.substr(0, proto_end);
        std::string address = endpoint.substr(proto_end + 3);
        
        // Create appropriate channel
        std::unique_ptr<Channel> channel;
        
        if (protocol == "tcp") {
            channel = create_channel("tcp://" + address, 1024 * 1024);
        } else if (protocol == "ipc") {
            channel = create_channel("ipc://" + address, 1024 * 1024);
        } else if (protocol == "inproc") {
            channel = create_channel("memory://" + address, 1024 * 1024);
        }
        
        if (channel) {
            channels_.push_back(std::move(channel));
            connected_ = true;
            
            // Start worker thread
            if (!worker_thread_.joinable()) {
                worker_thread_ = std::thread(&Socket::worker_loop, this);
            }
            
            return true;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to connect to " << endpoint << ": " << e.what() << std::endl;
    }
    
    return false;
}

void Socket::close() {
    running_ = false;
    connected_ = false;
    
    if (worker_thread_.joinable()) {
        cv_.notify_all();
        worker_thread_.join();
    }
    
    channels_.clear();
}

bool Socket::send(const Message& msg, int flags) {
    if (!connected_ || msg.empty()) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Add to send queue
    send_queue_.push(msg);
    cv_.notify_one();
    
    return true;
}

bool Socket::send(const std::string& data, int flags) {
    Message msg(data);
    return send(msg, flags);
}

bool Socket::recv(Message& msg, int flags) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait for message or timeout
    if (timeout_ms_ > 0) {
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms_),
                         [this] { return !recv_queue_.empty() || !running_; })) {
            return false; // Timeout
        }
    } else {
        cv_.wait(lock, [this] { return !recv_queue_.empty() || !running_; });
    }
    
    if (!running_ || recv_queue_.empty()) {
        return false;
    }
    
    msg = std::move(recv_queue_.front());
    recv_queue_.pop();
    
    return true;
}

std::string Socket::recv_string(int flags) {
    Message msg;
    if (recv(msg, flags) && !msg.frames().empty()) {
        return msg.frames()[0].to_string();
    }
    return "";
}

bool Socket::try_send(const Message& msg) {
    return send(msg, 0); // Non-blocking in our implementation
}

bool Socket::try_recv(Message& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (recv_queue_.empty()) {
        return false;
    }
    
    msg = std::move(recv_queue_.front());
    recv_queue_.pop();
    
    return true;
}

void Socket::set_identity(const std::string& identity) {
    identity_ = identity;
}

void Socket::set_subscribe(const std::string& filter) {
    if (std::find(subscriptions_.begin(), subscriptions_.end(), filter) == subscriptions_.end()) {
        subscriptions_.push_back(filter);
    }
}

void Socket::set_unsubscribe(const std::string& filter) {
    auto it = std::find(subscriptions_.begin(), subscriptions_.end(), filter);
    if (it != subscriptions_.end()) {
        subscriptions_.erase(it);
    }
}

void Socket::set_hwm(int hwm) {
    high_water_mark_ = hwm;
}

void Socket::set_linger(int linger_ms) {
    linger_ms_ = linger_ms;
}

void Socket::set_timeout(int timeout_ms) {
    timeout_ms_ = timeout_ms;
}

void Socket::worker_loop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Process send queue
        while (!send_queue_.empty() && running_) {
            Message msg = std::move(send_queue_.front());
            send_queue_.pop();
            
            lock.unlock();
            route_message(msg);
            lock.lock();
        }
        
        // Wait for more work
        cv_.wait_for(lock, std::chrono::milliseconds(10));
    }
}

// RequestSocket implementation

RequestSocket::RequestSocket() : Socket(SocketType::REQ) {
}

bool RequestSocket::send_request(const Message& request, Message& reply) {
    if (!send(request)) {
        return false;
    }
    
    waiting_for_reply_ = true;
    
    // Wait for reply
    std::unique_lock<std::mutex> lock(reply_mutex_);
    reply_cv_.wait(lock, [this] { return !waiting_for_reply_.load(); });
    
    reply = std::move(pending_reply_);
    return true;
}

bool RequestSocket::send_request(const std::string& request, std::string& reply) {
    Message req_msg(request);
    Message rep_msg;
    
    if (send_request(req_msg, rep_msg) && !rep_msg.frames().empty()) {
        reply = rep_msg.frames()[0].to_string();
        return true;
    }
    
    return false;
}

bool RequestSocket::route_message(const Message& msg) {
    // REQ socket sends to the first available channel
    if (!channels_.empty()) {
        auto serialized = msg.to_string();
        return channels_[0]->send(serialized.data(), serialized.size()) > 0;
    }
    return false;
}

bool RequestSocket::handle_received_message(Message&& msg) {
    if (waiting_for_reply_) {
        std::lock_guard<std::mutex> lock(reply_mutex_);
        pending_reply_ = std::move(msg);
        waiting_for_reply_ = false;
        reply_cv_.notify_one();
        return true;
    }
    return false;
}

// ReplySocket implementation

ReplySocket::ReplySocket() : Socket(SocketType::REP) {
}

bool ReplySocket::recv_request(Message& request) {
    return recv(request);
}

bool ReplySocket::send_reply(const Message& reply) {
    return send(reply);
}

bool ReplySocket::route_message(const Message& msg) {
    // REP socket sends to the client that sent the last request
    if (!channels_.empty()) {
        auto serialized = msg.to_string();
        return channels_[0]->send(serialized.data(), serialized.size()) > 0;
    }
    return false;
}

bool ReplySocket::handle_received_message(Message&& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    recv_queue_.push(std::move(msg));
    cv_.notify_one();
    return true;
}

// PublisherSocket implementation

PublisherSocket::PublisherSocket() : Socket(SocketType::PUB) {
}

bool PublisherSocket::publish(const std::string& topic, const Message& msg) {
    Message pub_msg;
    pub_msg.add_frame(topic, true);
    for (const auto& frame : msg.frames()) {
        pub_msg.add_frame(frame);
    }
    return send(pub_msg);
}

bool PublisherSocket::publish(const std::string& topic, const std::string& data) {
    Message msg(data);
    return publish(topic, msg);
}

bool PublisherSocket::route_message(const Message& msg) {
    // PUB socket sends to all connected channels
    bool success = true;
    for (auto& channel : channels_) {
        auto serialized = msg.to_string();
        if (channel->send(serialized.data(), serialized.size()) == 0) {
            success = false;
        }
    }
    return success;
}

bool PublisherSocket::handle_received_message(Message&& msg) {
    // PUB socket doesn't receive messages
    return false;
}

// SubscriberSocket implementation

SubscriberSocket::SubscriberSocket() : Socket(SocketType::SUB) {
}

void SubscriberSocket::subscribe(const std::string& topic) {
    set_subscribe(topic);
}

void SubscriberSocket::unsubscribe(const std::string& topic) {
    set_unsubscribe(topic);
}

bool SubscriberSocket::recv_topic(std::string& topic, Message& msg) {
    Message full_msg;
    if (recv(full_msg) && full_msg.frame_count() >= 2) {
        topic = full_msg.frames()[0].to_string();
        
        // Create message from remaining frames
        msg = Message();
        for (size_t i = 1; i < full_msg.frame_count(); ++i) {
            msg.add_frame(full_msg.frames()[i]);
        }
        
        return true;
    }
    return false;
}

bool SubscriberSocket::recv_topic(std::string& topic, std::string& data) {
    Message msg;
    if (recv_topic(topic, msg) && !msg.frames().empty()) {
        data = msg.frames()[0].to_string();
        return true;
    }
    return false;
}

bool SubscriberSocket::route_message(const Message& msg) {
    // SUB socket doesn't send messages
    return false;
}

bool SubscriberSocket::handle_received_message(Message&& msg) {
    // Check subscription filters
    if (msg.frame_count() >= 2) {
        std::string topic = msg.frames()[0].to_string();
        if (matches_subscription(topic)) {
            std::lock_guard<std::mutex> lock(mutex_);
            recv_queue_.push(std::move(msg));
            cv_.notify_one();
            return true;
        }
    }
    return false;
}

bool SubscriberSocket::matches_subscription(const std::string& topic) const {
    if (subscriptions_.empty()) {
        return true; // No filters = accept all
    }
    
    for (const auto& subscription : subscriptions_) {
        if (topic.substr(0, subscription.length()) == subscription) {
            return true;
        }
    }
    
    return false;
}

// PushSocket implementation

PushSocket::PushSocket() : Socket(SocketType::PUSH) {
}

bool PushSocket::route_message(const Message& msg) {
    // PUSH socket distributes to workers in round-robin fashion
    if (channels_.empty()) {
        return false;
    }
    
    size_t worker_idx = current_worker_++ % channels_.size();
    auto serialized = msg.to_string();
    return channels_[worker_idx]->send(serialized.data(), serialized.size()) > 0;
}

bool PushSocket::handle_received_message(Message&& msg) {
    // PUSH socket doesn't receive messages
    return false;
}

// PullSocket implementation

PullSocket::PullSocket() : Socket(SocketType::PULL) {
}

bool PullSocket::route_message(const Message& msg) {
    // PULL socket doesn't send messages
    return false;
}

bool PullSocket::handle_received_message(Message&& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    recv_queue_.push(std::move(msg));
    cv_.notify_one();
    return true;
}

// DealerSocket implementation

DealerSocket::DealerSocket() : Socket(SocketType::DEALER) {
}

bool DealerSocket::send_multipart(const std::vector<std::string>& parts) {
    Message msg(parts);
    return send(msg);
}

bool DealerSocket::recv_multipart(std::vector<std::string>& parts) {
    Message msg;
    if (recv(msg)) {
        parts.clear();
        for (const auto& frame : msg.frames()) {
            parts.push_back(frame.to_string());
        }
        return true;
    }
    return false;
}

bool DealerSocket::route_message(const Message& msg) {
    // DEALER socket sends to available channels (load balance)
    if (channels_.empty()) {
        return false;
    }
    
    // Simple load balancing - use a different channel each time
    static thread_local size_t channel_idx = 0;
    size_t idx = channel_idx++ % channels_.size();
    
    auto serialized = msg.to_string();
    return channels_[idx]->send(serialized.data(), serialized.size()) > 0;
}

bool DealerSocket::handle_received_message(Message&& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    recv_queue_.push(std::move(msg));
    cv_.notify_one();
    return true;
}

// RouterSocket implementation

RouterSocket::RouterSocket() : Socket(SocketType::ROUTER) {
}

bool RouterSocket::send_to_client(const std::string& client_id, const Message& msg) {
    Message routed_msg;
    routed_msg.add_frame(client_id, true);
    for (const auto& frame : msg.frames()) {
        routed_msg.add_frame(frame);
    }
    return send(routed_msg);
}

bool RouterSocket::recv_from_client(std::string& client_id, Message& msg) {
    Message full_msg;
    if (recv(full_msg) && full_msg.frame_count() >= 2) {
        client_id = full_msg.frames()[0].to_string();
        
        // Update client list
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            clients_[client_id] = std::chrono::steady_clock::now();
        }
        
        // Create message from remaining frames
        msg = Message();
        for (size_t i = 1; i < full_msg.frame_count(); ++i) {
            msg.add_frame(full_msg.frames()[i]);
        }
        
        return true;
    }
    return false;
}

std::vector<std::string> RouterSocket::get_connected_clients() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    std::vector<std::string> client_list;
    
    auto now = std::chrono::steady_clock::now();
    for (const auto& client : clients_) {
        // Consider clients active if seen in last 30 seconds
        if (now - client.second < std::chrono::seconds(30)) {
            client_list.push_back(client.first);
        }
    }
    
    return client_list;
}

bool RouterSocket::route_message(const Message& msg) {
    // ROUTER socket routes based on client ID in first frame
    if (msg.frame_count() >= 2 && !channels_.empty()) {
        auto serialized = msg.to_string();
        return channels_[0]->send(serialized.data(), serialized.size()) > 0;
    }
    return false;
}

bool RouterSocket::handle_received_message(Message&& msg) {
    // Add client identity to message if not present
    if (msg.frame_count() >= 1) {
        // Generate client ID if needed
        if (msg.frames()[0].to_string().empty()) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1000, 9999);
            std::string client_id = "client_" + std::to_string(dis(gen));
            
            Message routed_msg;
            routed_msg.add_frame(client_id, true);
            for (const auto& frame : msg.frames()) {
                routed_msg.add_frame(frame);
            }
            msg = std::move(routed_msg);
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        recv_queue_.push(std::move(msg));
        cv_.notify_one();
        return true;
    }
    return false;
}

// PairSocket implementation

PairSocket::PairSocket() : Socket(SocketType::PAIR) {
}

bool PairSocket::route_message(const Message& msg) {
    // PAIR socket sends to its exclusive peer
    if (!channels_.empty()) {
        auto serialized = msg.to_string();
        return channels_[0]->send(serialized.data(), serialized.size()) > 0;
    }
    return false;
}

bool PairSocket::handle_received_message(Message&& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    recv_queue_.push(std::move(msg));
    cv_.notify_one();
    return true;
}

// Factory functions

std::unique_ptr<Socket> create_socket(SocketType type, const std::string& transport) {
    switch (type) {
        case SocketType::REQ: return std::make_unique<RequestSocket>();
        case SocketType::REP: return std::make_unique<ReplySocket>();
        case SocketType::PUB: return std::make_unique<PublisherSocket>();
        case SocketType::SUB: return std::make_unique<SubscriberSocket>();
        case SocketType::PUSH: return std::make_unique<PushSocket>();
        case SocketType::PULL: return std::make_unique<PullSocket>();
        case SocketType::DEALER: return std::make_unique<DealerSocket>();
        case SocketType::ROUTER: return std::make_unique<RouterSocket>();
        case SocketType::PAIR: return std::make_unique<PairSocket>();
        default: return nullptr;
    }
}

std::unique_ptr<RequestSocket> create_request_socket(const std::string& transport) {
    return std::make_unique<RequestSocket>();
}

std::unique_ptr<ReplySocket> create_reply_socket(const std::string& transport) {
    return std::make_unique<ReplySocket>();
}

std::unique_ptr<PublisherSocket> create_publisher_socket(const std::string& transport) {
    return std::make_unique<PublisherSocket>();
}

std::unique_ptr<SubscriberSocket> create_subscriber_socket(const std::string& transport) {
    return std::make_unique<SubscriberSocket>();
}

std::unique_ptr<PushSocket> create_push_socket(const std::string& transport) {
    return std::make_unique<PushSocket>();
}

std::unique_ptr<PullSocket> create_pull_socket(const std::string& transport) {
    return std::make_unique<PullSocket>();
}

std::unique_ptr<DealerSocket> create_dealer_socket(const std::string& transport) {
    return std::make_unique<DealerSocket>();
}

std::unique_ptr<RouterSocket> create_router_socket(const std::string& transport) {
    return std::make_unique<RouterSocket>();
}

std::unique_ptr<PairSocket> create_pair_socket(const std::string& transport) {
    return std::make_unique<PairSocket>();
}

// Context implementation

Context::Context() {
}

Context::~Context() {
    terminate();
}

std::unique_ptr<Socket> Context::socket(SocketType type, const std::string& transport) {
    if (terminated_) {
        return nullptr;
    }
    
    auto sock = create_socket(type, transport);
    if (sock) {
        std::lock_guard<std::mutex> lock(mutex_);
        sockets_.push_back(std::unique_ptr<Socket>(sock.get()));
        return sock;
    }
    
    return nullptr;
}

bool Context::proxy(Socket& frontend, Socket& backend) {
    // Simple proxy implementation - forward messages between sockets
    // In a real implementation, this would be more sophisticated
    return false;
}

bool Context::device(Socket& socket1, Socket& socket2) {
    // Device implementation for connecting two sockets
    return false;
}

void Context::terminate() {
    terminated_ = true;
    std::lock_guard<std::mutex> lock(mutex_);
    sockets_.clear();
}

} // namespace patterns
} // namespace psyne