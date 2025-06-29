/**
 * @file zmq_patterns.hpp
 * @brief ZeroMQ-style messaging patterns for distributed systems
 * 
 * This provides high-level messaging patterns commonly used in distributed
 * systems, including request-reply, publish-subscribe, push-pull, and more.
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
#include <functional>

namespace psyne {
namespace patterns {

/**
 * @brief Socket types for different messaging patterns
 */
enum class SocketType {
    REQ,        ///< Request socket (sends request, waits for reply)
    REP,        ///< Reply socket (receives request, sends reply)
    DEALER,     ///< Dealer socket (asynchronous REQ)
    ROUTER,     ///< Router socket (asynchronous REP)
    PUB,        ///< Publisher socket (one-to-many)
    SUB,        ///< Subscriber socket (many-to-one)
    PUSH,       ///< Push socket (distributes to workers)
    PULL,       ///< Pull socket (receives from distributors)
    PAIR,       ///< Pair socket (bidirectional, exclusive)
    STREAM      ///< Stream socket (raw TCP-like)
};

/**
 * @brief Message frame for multi-part messages
 */
struct MessageFrame {
    std::vector<uint8_t> data;
    bool more = false;
    
    MessageFrame() = default;
    MessageFrame(const void* data, size_t size, bool more = false);
    MessageFrame(std::vector<uint8_t> data, bool more = false);
    MessageFrame(const std::string& str, bool more = false);
    
    std::string to_string() const;
    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
};

/**
 * @brief Multi-part message
 */
class Message {
public:
    Message() = default;
    Message(const std::string& data);
    Message(const std::vector<std::string>& parts);
    
    void add_frame(const MessageFrame& frame);
    void add_frame(const void* data, size_t size, bool more = false);
    void add_frame(const std::string& data, bool more = false);
    
    const std::vector<MessageFrame>& frames() const { return frames_; }
    std::vector<MessageFrame>& frames() { return frames_; }
    
    size_t frame_count() const { return frames_.size(); }
    bool empty() const { return frames_.empty(); }
    
    std::string to_string() const;
    
private:
    std::vector<MessageFrame> frames_;
};

/**
 * @brief Base class for ZMQ-style sockets
 */
class Socket {
public:
    Socket(SocketType type, const std::string& transport = "tcp");
    virtual ~Socket();
    
    // Connection management
    virtual bool bind(const std::string& endpoint);
    virtual bool connect(const std::string& endpoint);
    virtual void close();
    
    // Message operations
    virtual bool send(const Message& msg, int flags = 0);
    virtual bool send(const std::string& data, int flags = 0);
    virtual bool recv(Message& msg, int flags = 0);
    virtual std::string recv_string(int flags = 0);
    
    // Non-blocking operations
    virtual bool try_send(const Message& msg);
    virtual bool try_recv(Message& msg);
    
    // Socket options
    void set_identity(const std::string& identity);
    void set_subscribe(const std::string& filter);
    void set_unsubscribe(const std::string& filter);
    void set_hwm(int hwm);
    void set_linger(int linger_ms);
    void set_timeout(int timeout_ms);
    
    SocketType type() const { return type_; }
    bool is_connected() const { return connected_; }
    
protected:
    SocketType type_;
    std::string transport_;
    std::vector<std::unique_ptr<Channel>> channels_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> running_{true};
    
    // Socket options
    std::string identity_;
    std::vector<std::string> subscriptions_;
    int high_water_mark_ = 1000;
    int linger_ms_ = -1;
    int timeout_ms_ = -1;
    
    // Threading
    std::thread worker_thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    // Message queues
    std::queue<Message> send_queue_;
    std::queue<Message> recv_queue_;
    
    virtual void worker_loop();
    virtual bool route_message(const Message& msg) = 0;
    virtual bool handle_received_message(Message&& msg) = 0;
};

/**
 * @brief Request socket - synchronous request-reply client
 */
class RequestSocket : public Socket {
public:
    RequestSocket();
    
    bool send_request(const Message& request, Message& reply);
    bool send_request(const std::string& request, std::string& reply);
    
protected:
    bool route_message(const Message& msg) override;
    bool handle_received_message(Message&& msg) override;
    
private:
    std::atomic<bool> waiting_for_reply_{false};
    Message pending_reply_;
    std::mutex reply_mutex_;
    std::condition_variable reply_cv_;
};

/**
 * @brief Reply socket - synchronous request-reply server
 */
class ReplySocket : public Socket {
public:
    ReplySocket();
    
    bool recv_request(Message& request);
    bool send_reply(const Message& reply);
    
protected:
    bool route_message(const Message& msg) override;
    bool handle_received_message(Message&& msg) override;
    
private:
    std::queue<Message> request_queue_;
    std::string current_client_id_;
};

/**
 * @brief Publisher socket - one-to-many broadcast
 */
class PublisherSocket : public Socket {
public:
    PublisherSocket();
    
    bool publish(const std::string& topic, const Message& msg);
    bool publish(const std::string& topic, const std::string& data);
    
protected:
    bool route_message(const Message& msg) override;
    bool handle_received_message(Message&& msg) override;
};

/**
 * @brief Subscriber socket - many-to-one with filtering
 */
class SubscriberSocket : public Socket {
public:
    SubscriberSocket();
    
    void subscribe(const std::string& topic);
    void unsubscribe(const std::string& topic);
    
    bool recv_topic(std::string& topic, Message& msg);
    bool recv_topic(std::string& topic, std::string& data);
    
protected:
    bool route_message(const Message& msg) override;
    bool handle_received_message(Message&& msg) override;
    
private:
    bool matches_subscription(const std::string& topic) const;
};

/**
 * @brief Push socket - distributes work to pull sockets
 */
class PushSocket : public Socket {
public:
    PushSocket();
    
protected:
    bool route_message(const Message& msg) override;
    bool handle_received_message(Message&& msg) override;
    
private:
    std::atomic<size_t> current_worker_{0};
};

/**
 * @brief Pull socket - receives work from push sockets
 */
class PullSocket : public Socket {
public:
    PullSocket();
    
protected:
    bool route_message(const Message& msg) override;
    bool handle_received_message(Message&& msg) override;
};

/**
 * @brief Dealer socket - asynchronous request socket
 */
class DealerSocket : public Socket {
public:
    DealerSocket();
    
    bool send_multipart(const std::vector<std::string>& parts);
    bool recv_multipart(std::vector<std::string>& parts);
    
protected:
    bool route_message(const Message& msg) override;
    bool handle_received_message(Message&& msg) override;
};

/**
 * @brief Router socket - asynchronous reply socket with client tracking
 */
class RouterSocket : public Socket {
public:
    RouterSocket();
    
    bool send_to_client(const std::string& client_id, const Message& msg);
    bool recv_from_client(std::string& client_id, Message& msg);
    
    std::vector<std::string> get_connected_clients() const;
    
protected:
    bool route_message(const Message& msg) override;
    bool handle_received_message(Message&& msg) override;
    
private:
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> clients_;
    mutable std::mutex clients_mutex_;
};

/**
 * @brief Pair socket - exclusive bidirectional communication
 */
class PairSocket : public Socket {
public:
    PairSocket();
    
protected:
    bool route_message(const Message& msg) override;
    bool handle_received_message(Message&& msg) override;
    
private:
    std::string peer_identity_;
};

/**
 * @brief Factory functions for creating sockets
 */

std::unique_ptr<Socket> create_socket(SocketType type, const std::string& transport = "tcp");
std::unique_ptr<RequestSocket> create_request_socket(const std::string& transport = "tcp");
std::unique_ptr<ReplySocket> create_reply_socket(const std::string& transport = "tcp");
std::unique_ptr<PublisherSocket> create_publisher_socket(const std::string& transport = "tcp");
std::unique_ptr<SubscriberSocket> create_subscriber_socket(const std::string& transport = "tcp");
std::unique_ptr<PushSocket> create_push_socket(const std::string& transport = "tcp");
std::unique_ptr<PullSocket> create_pull_socket(const std::string& transport = "tcp");
std::unique_ptr<DealerSocket> create_dealer_socket(const std::string& transport = "tcp");
std::unique_ptr<RouterSocket> create_router_socket(const std::string& transport = "tcp");
std::unique_ptr<PairSocket> create_pair_socket(const std::string& transport = "tcp");

/**
 * @brief Context for managing multiple sockets and patterns
 */
class Context {
public:
    Context();
    ~Context();
    
    std::unique_ptr<Socket> socket(SocketType type, const std::string& transport = "tcp");
    
    // Common patterns
    bool proxy(Socket& frontend, Socket& backend);
    bool device(Socket& socket1, Socket& socket2);
    
    void terminate();
    
private:
    std::vector<std::unique_ptr<Socket>> sockets_;
    std::atomic<bool> terminated_{false};
    mutable std::mutex mutex_;
};

} // namespace patterns
} // namespace psyne