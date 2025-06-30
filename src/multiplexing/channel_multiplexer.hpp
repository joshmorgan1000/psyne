#pragma once

#include <condition_variable>
#include <mutex>
#include <psyne/psyne.hpp>
#include <queue>
#include <thread>
#include <unordered_map>

namespace psyne {
namespace multiplexing {

/**
 * @brief Multiplexes multiple logical channels over a single physical channel
 *
 * This allows efficient use of resources by sharing a single high-bandwidth
 * channel among multiple logical streams. Each logical channel is identified
 * by a channel ID.
 */
class ChannelMultiplexer {
public:
    /**
     * @brief Construct a multiplexer
     * @param physical_channel The underlying physical channel
     * @param max_logical_channels Maximum number of logical channels
     */
    ChannelMultiplexer(std::shared_ptr<Channel> physical_channel,
                       size_t max_logical_channels = 256);
    ~ChannelMultiplexer();

    /**
     * @brief Create a logical channel
     * @param channel_id Unique identifier for the logical channel (0-255)
     * @return Logical channel interface
     */
    std::shared_ptr<Channel> CreateLogicalChannel(uint8_t channel_id);

    /**
     * @brief Start the multiplexer
     */
    void Start();

    /**
     * @brief Stop the multiplexer
     */
    void Stop();

    /**
     * @brief Get statistics
     */
    struct Stats {
        uint64_t messages_sent = 0;
        uint64_t messages_received = 0;
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        std::unordered_map<uint8_t, uint64_t> channel_message_counts;
    };
    Stats GetStats() const;

private:
    class LogicalChannel;
    struct MultiplexedMessage;

    void SendLoop();
    void ReceiveLoop();

    std::shared_ptr<Channel> physical_channel_;
    size_t max_logical_channels_;

    // Logical channels
    std::unordered_map<uint8_t, std::shared_ptr<LogicalChannel>>
        logical_channels_;
    std::mutex channels_mutex_;

    // Send queue
    std::queue<MultiplexedMessage> send_queue_;
    std::mutex send_mutex_;
    std::condition_variable send_cv_;

    // Threads
    std::thread send_thread_;
    std::thread receive_thread_;
    std::atomic<bool> running_{false};

    // Statistics
    mutable Stats stats_;
    mutable std::mutex stats_mutex_;
};

/**
 * @brief Demultiplexer for receiving multiplexed channels
 *
 * Companion class to ChannelMultiplexer for the receiving side.
 */
class ChannelDemultiplexer {
public:
    /**
     * @brief Construct a demultiplexer
     * @param physical_channel The underlying physical channel
     */
    explicit ChannelDemultiplexer(std::shared_ptr<Channel> physical_channel);
    ~ChannelDemultiplexer();

    /**
     * @brief Register a handler for a logical channel
     * @param channel_id Logical channel ID
     * @param handler Function to handle received messages
     */
    using MessageHandler =
        std::function<void(const void *data, size_t size, uint32_t type)>;
    void RegisterHandler(uint8_t channel_id, MessageHandler handler);

    /**
     * @brief Get a logical channel for receiving
     * @param channel_id Logical channel ID
     * @return Logical channel interface
     */
    std::shared_ptr<Channel> GetLogicalChannel(uint8_t channel_id);

    /**
     * @brief Start the demultiplexer
     */
    void Start();

    /**
     * @brief Stop the demultiplexer
     */
    void Stop();

private:
    class LogicalReceiveChannel;

    void ReceiveLoop();

    std::shared_ptr<Channel> physical_channel_;
    std::unordered_map<uint8_t, MessageHandler> handlers_;
    std::unordered_map<uint8_t, std::shared_ptr<LogicalReceiveChannel>>
        logical_channels_;
    std::mutex mutex_;
    std::thread receive_thread_;
    std::atomic<bool> running_{false};
};

/**
 * @brief Channel pool for efficient channel management
 *
 * Manages a pool of channels with automatic multiplexing when needed.
 */
class ChannelPool {
public:
    struct Config {
        size_t initial_channels = 4;
        size_t max_channels = 16;
        size_t multiplex_threshold =
            8; // Start multiplexing after this many channels
        size_t channel_buffer_size = 16 * 1024 * 1024;
        std::string channel_uri_prefix = "memory://pool_";
    };

    explicit ChannelPool(const Config &config = {});
    ~ChannelPool();

    /**
     * @brief Acquire a channel from the pool
     * @param timeout_ms Timeout in milliseconds (0 = no wait)
     * @return Channel or nullptr if none available
     */
    std::shared_ptr<Channel> Acquire(int timeout_ms = 0);

    /**
     * @brief Release a channel back to the pool
     * @param channel The channel to release
     */
    void Release(std::shared_ptr<Channel> channel);

    /**
     * @brief Get pool statistics
     */
    struct PoolStats {
        size_t total_channels;
        size_t available_channels;
        size_t multiplexed_channels;
        uint64_t acquisitions;
        uint64_t releases;
    };
    PoolStats GetStats() const;

private:
    Config config_;
    std::vector<std::shared_ptr<Channel>> channels_;
    std::queue<std::shared_ptr<Channel>> available_;
    std::vector<std::shared_ptr<ChannelMultiplexer>> multiplexers_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    mutable PoolStats stats_{};
};

/**
 * @brief Priority-based multiplexer
 *
 * Multiplexes channels with priority-based scheduling.
 */
class PriorityMultiplexer {
public:
    enum class Priority { Low = 0, Normal = 1, High = 2, Critical = 3 };

    PriorityMultiplexer(std::shared_ptr<Channel> physical_channel);
    ~PriorityMultiplexer();

    /**
     * @brief Create a prioritized logical channel
     * @param channel_id Unique channel ID
     * @param priority Channel priority
     * @param bandwidth_weight Relative bandwidth allocation (1-100)
     * @return Logical channel
     */
    std::shared_ptr<Channel>
    CreatePriorityChannel(uint8_t channel_id, Priority priority,
                          uint8_t bandwidth_weight = 10);

    void Start();
    void Stop();

private:
    struct PriorityQueueEntry {
        uint8_t channel_id;
        Priority priority;
        std::chrono::steady_clock::time_point timestamp;
        std::vector<uint8_t> data;
        uint32_t type;

        bool operator<(const PriorityQueueEntry &other) const {
            if (priority != other.priority) {
                return static_cast<int>(priority) <
                       static_cast<int>(other.priority);
            }
            return timestamp >
                   other.timestamp; // Earlier timestamp = higher priority
        }
    };

    void SchedulerLoop();

    std::shared_ptr<Channel> physical_channel_;
    std::priority_queue<PriorityQueueEntry> send_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread scheduler_thread_;
    std::atomic<bool> running_{false};

    // Bandwidth management
    std::unordered_map<uint8_t, uint8_t> bandwidth_weights_;
    std::unordered_map<uint8_t, uint64_t> bytes_sent_;
};

/**
 * @brief Virtual channel that spans multiple physical channels
 *
 * Provides a single logical channel interface that transparently
 * uses multiple underlying channels for increased bandwidth.
 */
class BondedChannel {
public:
    /**
     * @brief Create a bonded channel from multiple physical channels
     * @param channels The physical channels to bond
     * @param mode Load balancing mode
     */
    enum class BondingMode {
        RoundRobin,  // Distribute messages round-robin
        LeastLoaded, // Send to least loaded channel
        Broadcast,   // Send to all channels (for redundancy)
        Striped      // Stripe large messages across channels
    };

    BondedChannel(std::vector<std::shared_ptr<Channel>> channels,
                  BondingMode mode = BondingMode::RoundRobin);
    ~BondedChannel();

    /**
     * @brief Send a message across bonded channels
     * @param data Message data
     * @param size Message size
     * @param type Message type
     * @return true if sent successfully
     */
    bool Send(const void *data, size_t size, uint32_t type);

    /**
     * @brief Receive from any bonded channel
     * @param timeout_ms Timeout in milliseconds
     * @return Received message or nullptr
     */
    std::unique_ptr<std::vector<uint8_t>> Receive(uint32_t &type,
                                                  int timeout_ms = 0);

    /**
     * @brief Get aggregate statistics
     */
    struct BondStats {
        uint64_t total_sent = 0;
        uint64_t total_received = 0;
        std::vector<uint64_t> channel_sent;
        std::vector<uint64_t> channel_received;
        double aggregate_bandwidth_mbps = 0;
    };
    BondStats GetStats() const;

private:
    std::vector<std::shared_ptr<Channel>> channels_;
    BondingMode mode_;
    std::atomic<size_t> next_channel_{0};

    // Statistics
    mutable BondStats stats_;
    mutable std::mutex stats_mutex_;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace multiplexing
} // namespace psyne