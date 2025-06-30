/**
 * @file nng_patterns.hpp
 * @brief Nanomsg/NNG-style messaging patterns for scalable protocols
 * 
 * This provides NNG-compatible messaging patterns that complement ZeroMQ patterns
 * with different semantics and use cases. NNG patterns emphasize simplicity,
 * built-in scalability, and protocol composability.
 * 
 * Key differences from ZMQ patterns:
 * - Simpler API with automatic connection management
 * - Built-in protocol semantics (timeouts, retries, load balancing)
 * - Better support for dynamic topologies
 * - Protocol-aware addressing and routing
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#pragma once

#include "../channel/channel_impl.hpp"
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
#include <chrono>
#include <random>
#include <unordered_set>

namespace psyne {
namespace patterns {
namespace nng {

/**
 * @brief NNG protocol types
 */
enum class Protocol {
    PIPELINE,       ///< One-way distributed processing pipeline
    SURVEY,         ///< Distributed query/response with collection semantics
    BUS,            ///< Multi-way peer-to-peer communication
    REQ_REP,        ///< Simple request-reply (1:1)
    PUB_SUB,        ///< Publish-subscribe (1:N)
    PAIR            ///< Exclusive pair communication
};

/**
 * @brief Socket roles for protocols
 */
enum class SocketRole {
    // Pipeline
    PUSH,           ///< Pipeline sender (distributes work)
    PULL,           ///< Pipeline receiver (processes work)
    
    // Survey 
    SURVEYOR,       ///< Survey initiator (asks questions)
    RESPONDENT,     ///< Survey responder (answers questions)
    
    // Bus
    BUS_NODE,       ///< Bus participant (peer)
    
    // Request-Reply
    REQ,            ///< Request socket
    REP,            ///< Reply socket
    
    // Publish-Subscribe  
    PUB,            ///< Publisher socket
    SUB,            ///< Subscriber socket
    
    // Pair
    PAIR0,          ///< Pair socket (version 0)
    PAIR1           ///< Pair socket (version 1 - with polyamorous mode)
};

/**
 * @brief Message with NNG semantics
 */
class NNGMessage {
public:
    NNGMessage() = default;
    NNGMessage(const void* data, size_t size);
    NNGMessage(const std::string& data);
    NNGMessage(std::vector<uint8_t> data);
    
    // Data access
    const uint8_t* data() const { return data_.data(); }
    uint8_t* data() { return data_.data(); }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }
    
    // Message properties
    void set_header(const std::string& key, const std::string& value);
    std::string get_header(const std::string& key) const;
    bool has_header(const std::string& key) const;
    
    // Routing information (used internally)
    void set_hop_count(uint32_t hops) { hop_count_ = hops; }
    uint32_t hop_count() const { return hop_count_; }
    
    void set_pipe_id(uint32_t pipe_id) { pipe_id_ = pipe_id; }
    uint32_t pipe_id() const { return pipe_id_; }
    
    // Convert to/from string
    std::string to_string() const;
    void from_string(const std::string& str);
    
    // Binary serialization
    std::vector<uint8_t> serialize() const;
    bool deserialize(const std::vector<uint8_t>& data);
    
private:
    std::vector<uint8_t> data_;
    std::unordered_map<std::string, std::string> headers_;
    uint32_t hop_count_ = 0;
    uint32_t pipe_id_ = 0;
};

/**
 * @brief NNG socket options
 */
struct SocketOptions {
    // Timeouts
    std::chrono::milliseconds send_timeout{1000};
    std::chrono::milliseconds recv_timeout{1000};
    std::chrono::milliseconds retry_time{100};
    
    // Buffering
    int send_buffer_size = 8192;
    int recv_buffer_size = 8192;
    
    // Survey-specific
    std::chrono::milliseconds survey_time{1000};  // How long to wait for survey responses
    
    // Pipeline-specific
    bool load_balance = true;  // Load balance across PULL sockets
    
    // Bus-specific
    uint32_t max_ttl = 8;      // Maximum hops for bus messages
    
    // General
    bool raw_mode = false;     // Bypass protocol semantics
    int max_reconnect_time_ms = 60000;  // Maximum time between reconnection attempts
};

/**
 * @brief Base class for NNG-style sockets
 */
class NNGSocket {
public:
    NNGSocket(Protocol protocol, SocketRole role, const SocketOptions& options = SocketOptions{});
    virtual ~NNGSocket();
    
    // Connection management (NNG-style)
    virtual bool listen(const std::string& url);
    virtual bool dial(const std::string& url);
    virtual void close();
    
    // Message operations
    virtual bool send(const NNGMessage& msg);
    virtual bool send(const std::string& data);
    virtual bool send(const void* data, size_t size);
    
    virtual bool recv(NNGMessage& msg);
    virtual std::string recv_string();
    
    // Non-blocking operations
    virtual bool try_send(const NNGMessage& msg);
    virtual bool try_recv(NNGMessage& msg);
    
    // Socket information
    Protocol protocol() const { return protocol_; }
    SocketRole role() const { return role_; }
    bool is_listening() const { return listening_; }
    
    // Socket options
    void set_option(const SocketOptions& options);
    SocketOptions get_options() const;
    
    // Statistics
    struct Stats {
        uint64_t messages_sent = 0;
        uint64_t messages_received = 0;
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        uint64_t connect_events = 0;
        uint64_t disconnect_events = 0;
        uint64_t timeouts = 0;
        uint64_t errors = 0;
    };
    
    Stats get_stats() const;
    void reset_stats();
    
protected:
    Protocol protocol_;
    SocketRole role_;
    SocketOptions options_;
    std::vector<std::unique_ptr<Channel>> channels_;
    std::atomic<bool> listening_{false};
    std::atomic<bool> running_{true};
    
    // Threading
    std::thread worker_thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    // Message queues
    std::queue<NNGMessage> send_queue_;
    std::queue<NNGMessage> recv_queue_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    Stats stats_;
    
    // Virtual interface for protocol-specific behavior
    virtual void worker_loop();
    virtual bool route_outgoing_message(const NNGMessage& msg) = 0;
    virtual bool handle_incoming_message(NNGMessage&& msg) = 0;
    
    // Helper methods
    void update_stats_send(size_t bytes);
    void update_stats_recv(size_t bytes);
    void update_stats_error();
    void update_stats_timeout();
    
    // Channel management
    bool add_channel(std::unique_ptr<Channel> channel);
    void remove_disconnected_channels();
};

/**
 * @brief Pipeline Push socket - distributes work to Pull sockets
 * 
 * The Pipeline pattern is designed for distributing work across multiple workers.
 * Push sockets send messages to Pull sockets in a load-balanced fashion.
 * Unlike ZMQ PUSH, NNG Pipeline has built-in load balancing and fault tolerance.
 */
class PipelinePush : public NNGSocket {
public:
    PipelinePush(const SocketOptions& options = SocketOptions{});
    
    // Pipeline-specific methods
    bool send_work(const NNGMessage& work);
    bool send_work(const std::string& work_data);
    
    // Load balancing information
    size_t active_workers() const;
    std::vector<std::string> get_worker_addresses() const;
    
protected:
    bool route_outgoing_message(const NNGMessage& msg) override;
    bool handle_incoming_message(NNGMessage&& msg) override;
    
private:
    std::atomic<size_t> current_worker_{0};
    std::vector<std::string> worker_addresses_;
    mutable std::mutex workers_mutex_;
};

/**
 * @brief Pipeline Pull socket - receives work from Push sockets
 * 
 * Pull sockets receive work items from Push sockets. Multiple Pull sockets
 * can connect to the same Push socket, creating a work distribution pattern.
 */
class PipelinePull : public NNGSocket {
public:
    PipelinePull(const SocketOptions& options = SocketOptions{});
    
    // Pipeline-specific methods
    bool recv_work(NNGMessage& work);
    std::string recv_work_string();
    
    // Processing feedback (optional)
    bool send_completion_ack(const std::string& work_id);
    
protected:
    bool route_outgoing_message(const NNGMessage& msg) override;
    bool handle_incoming_message(NNGMessage&& msg) override;
    
private:
    std::queue<NNGMessage> work_queue_;
    std::mutex work_mutex_;
};

/**
 * @brief Survey Surveyor socket - initiates distributed queries
 * 
 * The Survey pattern allows one surveyor to query multiple respondents
 * and collect their responses within a time window. It's useful for
 * distributed consensus, health checks, and data collection.
 */
class SurveySurveyor : public NNGSocket {
public:
    SurveySurveyor(const SocketOptions& options = SocketOptions{});
    
    // Survey operations
    bool send_survey(const NNGMessage& survey);
    bool send_survey(const std::string& question);
    
    // Response collection
    std::vector<NNGMessage> collect_responses(std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    std::vector<std::string> collect_response_strings(std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    
    // Comprehensive survey with automatic collection
    struct SurveyResult {
        std::vector<NNGMessage> responses;
        size_t respondents_contacted;
        size_t responses_received;
        std::chrono::milliseconds collection_time;
        bool timed_out;
    };
    
    SurveyResult conduct_survey(const NNGMessage& survey, std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    SurveyResult conduct_survey(const std::string& question, std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    
protected:
    bool route_outgoing_message(const NNGMessage& msg) override;
    bool handle_incoming_message(NNGMessage&& msg) override;
    
private:
    std::queue<NNGMessage> responses_;
    std::mutex responses_mutex_;
    std::condition_variable responses_cv_;
    std::atomic<uint32_t> survey_id_{0};
    std::atomic<bool> collecting_responses_{false};
};

/**
 * @brief Survey Respondent socket - answers distributed queries
 * 
 * Respondent sockets receive survey questions and can send back responses.
 * Multiple respondents can be connected to the same surveyor.
 */
class SurveyRespondent : public NNGSocket {
public:
    SurveyRespondent(const SocketOptions& options = SocketOptions{});
    
    // Survey handling
    bool recv_survey(NNGMessage& survey);
    std::string recv_survey_string();
    
    bool send_response(const NNGMessage& response);
    bool send_response(const std::string& response_data);
    
    // Automatic survey handling with callback
    using SurveyHandler = std::function<NNGMessage(const NNGMessage& survey)>;
    void set_survey_handler(SurveyHandler handler);
    void enable_auto_response(bool enable = true);
    
protected:
    bool route_outgoing_message(const NNGMessage& msg) override;
    bool handle_incoming_message(NNGMessage&& msg) override;
    
private:
    std::queue<NNGMessage> surveys_;
    std::mutex surveys_mutex_;
    SurveyHandler survey_handler_;
    std::atomic<bool> auto_response_enabled_{false};
    uint32_t current_survey_id_ = 0;
};

/**
 * @brief Bus socket - multi-way peer-to-peer communication
 * 
 * The Bus pattern creates a mesh network where every participant can
 * send messages to every other participant. Messages are automatically
 * routed with hop-count limiting to prevent loops.
 */
class BusSocket : public NNGSocket {
public:
    BusSocket(const SocketOptions& options = SocketOptions{});
    
    // Bus communication
    bool send_broadcast(const NNGMessage& msg);
    bool send_broadcast(const std::string& data);
    
    bool send_to_peer(const std::string& peer_id, const NNGMessage& msg);
    bool send_to_peer(const std::string& peer_id, const std::string& data);
    
    // Peer management
    std::vector<std::string> get_connected_peers() const;
    bool is_peer_connected(const std::string& peer_id) const;
    
    // Bus topology
    struct TopologyInfo {
        std::string local_id;
        std::vector<std::string> direct_peers;
        std::vector<std::string> reachable_peers;
        std::unordered_map<std::string, uint32_t> peer_distances;  // Hop counts
    };
    
    TopologyInfo get_topology() const;
    
    // Event callbacks
    using PeerEventHandler = std::function<void(const std::string& peer_id, bool connected)>;
    void set_peer_event_handler(PeerEventHandler handler);
    
    using MessageHandler = std::function<void(const std::string& from_peer, const NNGMessage& msg)>;
    void set_message_handler(MessageHandler handler);
    
protected:
    bool route_outgoing_message(const NNGMessage& msg) override;
    bool handle_incoming_message(NNGMessage&& msg) override;
    
private:
    std::string local_id_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> peers_;
    std::unordered_map<std::string, uint32_t> peer_hop_counts_;
    mutable std::mutex peers_mutex_;
    
    PeerEventHandler peer_event_handler_;
    MessageHandler message_handler_;
    
    // Message deduplication
    std::unordered_set<std::string> seen_messages_;
    std::mutex seen_messages_mutex_;
    
    // Helper methods
    std::string generate_message_id() const;
    bool is_message_seen(const std::string& msg_id);
    void update_peer_topology(const NNGMessage& msg);
    void broadcast_topology_update();
};

/**
 * @brief High-level pattern implementations
 */

/**
 * @brief Work distribution coordinator for Pipeline pattern
 */
class WorkDistributor {
public:
    WorkDistributor(const std::string& bind_address, const SocketOptions& options = SocketOptions{});
    ~WorkDistributor();
    
    // Work distribution
    bool submit_work(const NNGMessage& work);
    bool submit_work(const std::string& work_data);
    
    // Batch operations
    bool submit_work_batch(const std::vector<NNGMessage>& work_items);
    bool submit_work_batch(const std::vector<std::string>& work_data);
    
    // Status monitoring
    size_t pending_work_count() const;
    size_t active_workers() const;
    
    // Worker management
    std::vector<std::string> get_worker_info() const;
    
private:
    std::unique_ptr<PipelinePush> push_socket_;
    std::atomic<size_t> work_submitted_{0};
};

/**
 * @brief Work processor for Pipeline pattern
 */
class WorkProcessor {
public:
    using WorkHandler = std::function<bool(const NNGMessage& work)>;
    
    WorkProcessor(const std::string& connect_address, WorkHandler handler, 
                  const SocketOptions& options = SocketOptions{});
    ~WorkProcessor();
    
    // Processing control
    void start_processing();
    void stop_processing();
    bool is_processing() const;
    
    // Statistics
    size_t work_processed() const { return work_processed_; }
    size_t work_failed() const { return work_failed_; }
    
private:
    std::unique_ptr<PipelinePull> pull_socket_;
    WorkHandler work_handler_;
    std::atomic<bool> processing_{false};
    std::atomic<size_t> work_processed_{0};
    std::atomic<size_t> work_failed_{0};
    std::thread processing_thread_;
    
    void processing_loop();
};

/**
 * @brief Distributed query engine using Survey pattern
 */
class QueryEngine {
public:
    QueryEngine(const std::string& bind_address, const SocketOptions& options = SocketOptions{});
    ~QueryEngine();
    
    // Query operations
    std::vector<NNGMessage> query_all(const NNGMessage& query, 
                                      std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    std::vector<std::string> query_all_string(const std::string& query,
                                               std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    
    // Aggregation queries
    template<typename T>
    T aggregate_query(const std::string& query, T initial_value,
                      std::function<T(T, const NNGMessage&)> aggregator,
                      std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    
    // Common aggregations
    double sum_query(const std::string& query, std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    double avg_query(const std::string& query, std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    size_t count_query(const std::string& query, std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    
private:
    std::unique_ptr<SurveySurveyor> surveyor_socket_;
};

/**
 * @brief Query responder for Survey pattern
 */
class QueryResponder {
public:
    using QueryHandler = std::function<NNGMessage(const NNGMessage& query)>;
    
    QueryResponder(const std::string& connect_address, QueryHandler handler,
                   const SocketOptions& options = SocketOptions{});
    ~QueryResponder();
    
    // Response control
    void start_responding();
    void stop_responding();
    bool is_responding() const;
    
    // Statistics
    size_t queries_handled() const { return queries_handled_; }
    
private:
    std::unique_ptr<SurveyRespondent> respondent_socket_;
    QueryHandler query_handler_;
    std::atomic<bool> responding_{false};
    std::atomic<size_t> queries_handled_{0};
    std::thread responding_thread_;
    
    void responding_loop();
};

/**
 * @brief Mesh network node using Bus pattern
 */
class MeshNode {
public:
    MeshNode(const std::string& node_id, const std::string& bind_address,
             const SocketOptions& options = SocketOptions{});
    ~MeshNode();
    
    // Network operations
    bool join_mesh(const std::string& peer_address);
    bool leave_mesh();
    
    // Communication
    bool broadcast(const NNGMessage& msg);
    bool broadcast(const std::string& data);
    bool send_to_node(const std::string& node_id, const NNGMessage& msg);
    bool send_to_node(const std::string& node_id, const std::string& data);
    
    // Message handling
    using MessageHandler = std::function<void(const std::string& from_node, const NNGMessage& msg)>;
    void set_message_handler(MessageHandler handler);
    
    // Network monitoring
    std::vector<std::string> get_peers() const;
    BusSocket::TopologyInfo get_topology() const;
    
    // Network health
    bool ping_node(const std::string& node_id, std::chrono::milliseconds timeout = std::chrono::milliseconds{1000});
    std::vector<std::string> discover_nodes(std::chrono::milliseconds timeout = std::chrono::milliseconds{2000});
    
private:
    std::unique_ptr<BusSocket> bus_socket_;
    std::string node_id_;
    MessageHandler message_handler_;
    std::atomic<bool> running_{true};
    std::thread message_thread_;
    
    void message_loop();
};

/**
 * @brief Factory functions for creating NNG pattern objects
 */

std::unique_ptr<PipelinePush> create_pipeline_push(const std::string& bind_address, 
                                                   const SocketOptions& options = SocketOptions{});
std::unique_ptr<PipelinePull> create_pipeline_pull(const std::string& connect_address,
                                                   const SocketOptions& options = SocketOptions{});

std::unique_ptr<SurveySurveyor> create_survey_surveyor(const std::string& bind_address,
                                                       const SocketOptions& options = SocketOptions{});
std::unique_ptr<SurveyRespondent> create_survey_respondent(const std::string& connect_address,
                                                          const SocketOptions& options = SocketOptions{});

std::unique_ptr<BusSocket> create_bus_socket(const std::string& node_id, const std::string& bind_address,
                                            const SocketOptions& options = SocketOptions{});

} // namespace nng
} // namespace patterns
} // namespace psyne