#pragma once

/**
 * @file psyne.hpp
 * @brief Psyne - High-performance zero-copy messaging library optimized for AI/ML
 * 
 * This is the single public API header for the Psyne library.
 * Include this file to access all Psyne functionality.
 * 
 * @author Psyne Contributors
 * @version 0.1.1
 * @date 2024
 * 
 * @copyright Copyright (c) 2024 Psyne Contributors
 * 
 * @example simple_messaging.cpp
 * @example producer_consumer.cpp
 * @example channel_factory_demo.cpp
 */

// Psyne - Zero-copy messaging library
// This is the single public API header

#include <memory>
#include <string>
#include <functional>
#include <thread>
#include <chrono>
#include <optional>
#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <atomic>
#include <complex>

// Version information
#define PSYNE_VERSION_MAJOR 1
#define PSYNE_VERSION_MINOR 0
#define PSYNE_VERSION_PATCH 0

/**
 * @namespace psyne
 * @brief Main namespace for the Psyne messaging library
 */
namespace psyne {

/**
 * @brief Get the version string of the Psyne library
 * @return Version string in the format "major.minor.patch"
 */
constexpr const char* version() {
    return "1.0.0";
}

/**
 * @brief Get the version string of the Psyne library (runtime version)
 * @return Version string in the format "major.minor.patch"
 */
const char* get_version();

/**
 * @brief Print the Psyne banner to stdout
 * 
 * Displays the ASCII art logo and version information.
 */
void print_banner();

// ============================================================================
// Core Types
// ============================================================================

/**
 * @enum ChannelMode
 * @brief Defines the synchronization mode for a channel
 * 
 * Different modes provide different thread-safety guarantees and performance
 * characteristics. Choose the mode that matches your use case.
 */
enum class ChannelMode {
    SPSC,  ///< Single Producer, Single Consumer - Highest performance, lock-free
    SPMC,  ///< Single Producer, Multiple Consumer - One writer, many readers
    MPSC,  ///< Multiple Producer, Single Consumer - Many writers, one reader
    MPMC   ///< Multiple Producer, Multiple Consumer - Full multi-threading support
};

/**
 * @enum ChannelType
 * @brief Defines whether a channel supports single or multiple message types
 */
enum class ChannelType {
    SingleType,   ///< Optimized for single message type (no type metadata overhead)
    MultiType     ///< Supports multiple message types (small metadata overhead)
};

// Forward declarations
class Channel;
template<typename Derived> class Message;

// Implementation details - defined in src
namespace detail {
    class ChannelImpl;
}

// Debug metrics - optional for performance monitoring
namespace debug {

/**
 * @struct ChannelMetrics
 * @brief Lightweight performance metrics for channel debugging
 * 
 * Tracks basic counters for message throughput and blocking behavior.
 * Designed to have minimal overhead in the hot path.
 */
struct ChannelMetrics {
    // Core counters - non-atomic for SPSC, atomic for multi-threaded channels
    uint64_t messages_sent = 0;      ///< Number of messages sent
    uint64_t bytes_sent = 0;         ///< Total bytes sent
    uint64_t messages_received = 0;  ///< Number of messages received  
    uint64_t bytes_received = 0;     ///< Total bytes received
    uint64_t send_blocks = 0;        ///< Times send() blocked waiting for space
    uint64_t receive_blocks = 0;     ///< Times receive() blocked waiting for data
    
    ChannelMetrics() = default;
    ChannelMetrics(const ChannelMetrics&) = default;
    ChannelMetrics& operator=(const ChannelMetrics&) = default;
    ChannelMetrics(ChannelMetrics&&) = default;
    ChannelMetrics& operator=(ChannelMetrics&&) = default;
};

/**
 * @struct AtomicChannelMetrics
 * @brief Thread-safe version of ChannelMetrics for multi-threaded channels
 * 
 * Uses atomic counters for thread safety in MPSC/SPMC/MPMC modes.
 */
struct AtomicChannelMetrics {
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> messages_received{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> send_blocks{0};
    std::atomic<uint64_t> receive_blocks{0};
    
    AtomicChannelMetrics() = default;
    
    /**
     * @brief Get current metrics as non-atomic struct for calculations
     */
    ChannelMetrics current() const {
        ChannelMetrics result;
        result.messages_sent = messages_sent.load(std::memory_order_relaxed);
        result.bytes_sent = bytes_sent.load(std::memory_order_relaxed);
        result.messages_received = messages_received.load(std::memory_order_relaxed);
        result.bytes_received = bytes_received.load(std::memory_order_relaxed);
        result.send_blocks = send_blocks.load(std::memory_order_relaxed);
        result.receive_blocks = receive_blocks.load(std::memory_order_relaxed);
        return result;
    }
};

// Debug and introspection utilities
class ChannelInspector;
class PerformanceProfiler;
class MessageTracer;
class ChannelMonitor;
struct ChannelDiagnostics;
struct BufferUsage;
enum class ChannelHealth;

} // namespace debug

// Compression support - optional for reducing network bandwidth
namespace compression {

#ifndef PSYNE_COMPRESSION_TYPES_DEFINED
#define PSYNE_COMPRESSION_TYPES_DEFINED

/**
 * @enum CompressionType
 * @brief Supported compression algorithms
 */
enum class CompressionType : uint8_t {
    None = 0,      ///< No compression
    LZ4 = 1,       ///< Fast compression/decompression
    Zstd = 2,      ///< Better compression ratio
    Snappy = 3     ///< Google Snappy - balanced speed/ratio
};

/**
 * @struct CompressionConfig
 * @brief Configuration for compression behavior
 */
struct CompressionConfig {
    CompressionType type = CompressionType::None;
    int level = 1;                    ///< Compression level (algorithm dependent)
    size_t min_size_threshold = 128;  ///< Don't compress messages smaller than this
    bool enable_checksum = true;      ///< Add checksum for compressed data
};

#endif // PSYNE_COMPRESSION_TYPES_DEFINED

} // namespace compression

// ============================================================================
// Message API
// ============================================================================

/**
 * @class Message
 * @brief Base class for all zero-copy messages
 * @tparam Derived The derived message type (CRTP pattern)
 * 
 * Messages in Psyne are zero-copy views into pre-allocated channel buffers.
 * They support move-only semantics to ensure single ownership.
 * 
 * To create a custom message type:
 * @code
 * class MyMessage : public Message<MyMessage> {
 * public:
 *     static constexpr uint32_t message_type = 42;
 *     static size_t calculate_size() { return sizeof(MyData); }
 *     using Message<MyMessage>::Message;
 *     // ... custom methods ...
 * };
 * @endcode
 */
template<typename Derived>
class Message {
public:
    /**
     * @brief Create an outgoing message (allocates space in channel)
     * @param channel The channel to allocate the message in
     * @throws std::runtime_error if allocation fails
     */
    explicit Message(Channel& channel);
    
    /**
     * @brief Create an incoming message (view of existing data)
     * @param data Pointer to the message data
     * @param size Size of the message in bytes
     */
    explicit Message(const void* data, size_t size);
    
    /**
     * @brief Move constructor
     * @param other Message to move from
     */
    Message(Message&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     * @param other Message to move from
     * @return Reference to this message
     */
    Message& operator=(Message&& other) noexcept;
    
    // Deleted copy operations to ensure zero-copy semantics
    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;
    
    /**
     * @brief Destructor
     */
    virtual ~Message();
    
    /**
     * @brief Send the message through its channel
     * @throws std::runtime_error if message has no associated channel
     */
    void send();
    
    /**
     * @brief Check if the message is valid (has allocated data)
     * @return true if valid, false otherwise
     */
    bool is_valid() const { return data_ != nullptr; }
    
    /**
     * @brief Get the message type ID
     * @return Type ID of this message class
     */
    static constexpr uint32_t type() { return Derived::message_type; }
    
    /**
     * @brief Get the raw data pointer
     * @return Pointer to the message data
     */
    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    
    /**
     * @brief Get the size of the message data
     * @return Size in bytes
     */
    size_t size() const { return size_; }
    
protected:
    
    // Access to channel for derived classes
    Channel& channel() { return *channel_; }
    const Channel& channel() const { return *channel_; }
    
    // Called before sending
    virtual void before_send() {}
    
private:
    uint8_t* data_;
    size_t size_;
    Channel* channel_;
    void* handle_;  // Opaque handle to implementation
};

// ============================================================================
// Pre-defined Message Types
// ============================================================================

/**
 * @class FloatVector
 * @brief Dynamic-size array of floating-point values
 * 
 * A zero-copy message type for transmitting variable-length arrays of floats.
 * The vector can be resized up to the channel's buffer capacity.
 * 
 * @example
 * @code
 * FloatVector msg(channel);
 * msg.resize(100);
 * for (size_t i = 0; i < msg.size(); ++i) {
 *     msg[i] = std::sin(i * 0.1f);
 * }
 * msg.send();
 * @endcode
 */
class FloatVector : public Message<FloatVector> {
public:
    static constexpr uint32_t message_type = 1;
    
    using Message<FloatVector>::Message;
    
    // Calculate required size (must be provided by derived classes)
    static size_t calculate_size() {
        // Default size for dynamic messages - will be resized as needed
        return 1024;  
    }
    
    // Array access
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    
    // STL interface
    float* begin();
    float* end();
    const float* begin() const;
    const float* end() const;
    
    // Size management
    size_t size() const;
    size_t capacity() const;
    void resize(size_t new_size);
    
    // Assignment from initializer list
    FloatVector& operator=(std::initializer_list<float> values);
    
    // Initialize the message (called after allocation)
    void initialize();
};

/**
 * @class DoubleMatrix
 * @brief Dynamic 2D matrix of double-precision values
 * 
 * A zero-copy message type for transmitting 2D matrices of doubles.
 * Supports dynamic resizing and row-major storage.
 * 
 * @example
 * @code
 * DoubleMatrix msg(channel);
 * msg.set_dimensions(10, 10);
 * for (size_t i = 0; i < msg.rows(); ++i) {
 *     for (size_t j = 0; j < msg.cols(); ++j) {
 *         msg.at(i, j) = i * j;
 *     }
 * }
 * msg.send();
 * @endcode
 */
class DoubleMatrix : public Message<DoubleMatrix> {
public:
    static constexpr uint32_t message_type = 2;
    
    using Message<DoubleMatrix>::Message;
    
    // Calculate required size
    static size_t calculate_size() {
        // Default size for dynamic messages
        return 8192;  
    }
    
    // Dimension management
    void set_dimensions(size_t rows, size_t cols);
    size_t rows() const;
    size_t cols() const;
    
    // Element access
    double& at(size_t row, size_t col);
    const double& at(size_t row, size_t col) const;
    
    // Initialize the message (called after allocation)
    void initialize();
};

// ============================================================================
// Channel API
// ============================================================================

/**
 * @class Channel
 * @brief Abstract base class for all communication channels
 * 
 * Channels are the core abstraction in Psyne for inter-process and
 * inter-thread communication. They provide zero-copy message passing
 * with configurable synchronization modes.
 * 
 * Channels are created using the factory method with URI schemes:
 * - `memory://name` - In-process shared memory
 * - `ipc://name` - Inter-process communication
 * - `tcp://host:port` - Network communication
 * - `unix:///path` - Unix domain sockets
 * 
 * @see create_channel() for the recommended factory function
 */
class Channel {
public:
    /**
     * @brief Factory method to create channels
     * @param uri Channel URI (e.g., "memory://buffer1", "tcp://localhost:8080")
     * @param buffer_size Size of the internal buffer in bytes
     * @param mode Synchronization mode (SPSC, SPMC, MPSC, MPMC)
     * @param type SingleType or MultiType channel
     * @return Unique pointer to the created channel
     * @throws std::invalid_argument for invalid URI format
     * @throws std::runtime_error for creation failures
     */
    static std::unique_ptr<Channel> create(
        const std::string& uri,
        size_t buffer_size,
        ChannelMode mode = ChannelMode::SPSC,
        ChannelType type = ChannelType::MultiType,
        bool enable_metrics = false,
        const compression::CompressionConfig& compression_config = {}
    );
    
    /**
     * @brief Virtual destructor
     */
    virtual ~Channel() = default;
    
    /**
     * @brief Send a message through the channel
     * @tparam MessageType Type of message to send
     * @param msg Message to send (will be moved)
     * 
     * This is a convenience method that calls msg.send().
     */
    template<typename MessageType>
    void send(MessageType& msg) {
        msg.send();
    }
    
    /**
     * @brief Receive a message from the channel
     * @tparam MessageType Expected message type
     * @param timeout Maximum time to wait (0 for non-blocking)
     * @return Optional containing the message if received, empty otherwise
     * 
     * @note For MultiType channels, messages of wrong type are discarded
     */
    template<typename MessageType>
    std::optional<MessageType> receive(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()
    ) {
        size_t size;
        uint32_t type_id;
        
        // Get message from implementation using virtual method
        void* data = receive_raw_message(size, type_id);
        if (!data) {
            return std::nullopt;
        }
        
        // Verify type matches (for safety)
        if (type() == ChannelType::MultiType && type_id != MessageType::message_type) {
            release_raw_message(data);
            return std::nullopt;
        }
        
        // Create message view
        return MessageType(data, size);
    }
    
    // Event-driven listening
    template<typename MessageType>
    std::unique_ptr<std::thread> listen(
        std::function<void(MessageType&&)> handler
    ) {
        return std::make_unique<std::thread>([this, handler]() {
            while (!is_stopped()) {
                auto msg = receive<MessageType>(std::chrono::milliseconds(100));
                if (msg) {
                    handler(std::move(*msg));
                }
            }
        });
    }
    
    // Channel control
    virtual void stop() = 0;
    virtual bool is_stopped() const = 0;
    
    // Properties
    virtual const std::string& uri() const = 0;
    virtual ChannelType type() const = 0;
    virtual ChannelMode mode() const = 0;
    
    // Raw message operations for template implementation
    virtual void* receive_raw_message(size_t& size, uint32_t& type) = 0;
    virtual void release_raw_message(void* handle) = 0;
    
    // Debug metrics (optional)
    virtual bool has_metrics() const = 0;
    virtual debug::ChannelMetrics get_metrics() const = 0;
    virtual void reset_metrics() = 0;
    
    // Implementation access for Python bindings
    detail::ChannelImpl* get_impl() { return impl(); }
    const detail::ChannelImpl* get_impl() const { return impl(); }
    
protected:
    Channel() = default;
    
    // Implementation pointer - protected so Message can access
    virtual detail::ChannelImpl* impl() = 0;
    virtual const detail::ChannelImpl* impl() const = 0;
    
    // Friend declaration to allow Message to access impl()
    template<typename Derived>
    friend class Message;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create a channel from a URI
 * @param uri Channel URI specifying transport and location
 * @param buffer_size Size of the internal message buffer (default: 1MB)
 * @param mode Channel synchronization mode (default: SPSC)
 * @param type Single or multi-type channel (default: MultiType)
 * @param enable_metrics Enable debug metrics collection (default: false)
 * @return Unique pointer to the created channel
 * 
 * Supported URI schemes:
 * - `memory://name` - In-process memory channels
 * - `ipc://name` - Inter-process communication via shared memory
 * - `tcp://host:port` - TCP client (connects to host:port)
 * - `tcp://:port` - TCP server (listens on port)
 * - `unix:///path/to/sock` - Unix domain socket client
 * - `unix://@/path/to/sock` - Unix domain socket server
 * - `ws://host:port` - WebSocket client (connects to host:port)
 * - `ws://:port` - WebSocket server (listens on port)
 * - `wss://host:port` - Secure WebSocket client (TLS, not yet implemented)
 * 
 * @throws std::invalid_argument Invalid URI format
 * @throws std::runtime_error Channel creation failed
 * 
 * @example
 * @code
 * // Create an in-process channel
 * auto chan1 = psyne::create_channel("memory://buffer1");
 * 
 * // Create a TCP server with metrics enabled
 * auto server = psyne::create_channel("tcp://:8080", 1024*1024, 
 *                                     ChannelMode::SPSC, 
 *                                     ChannelType::MultiType, true);
 * 
 * // Check metrics
 * if (server->has_metrics()) {
 *     auto metrics = server->get_metrics();
 *     std::cout << "Messages sent: " << metrics.messages_sent << std::endl;
 * }
 * @endcode
 */
inline std::unique_ptr<Channel> create_channel(
    const std::string& uri,
    size_t buffer_size = 1024 * 1024,
    ChannelMode mode = ChannelMode::SPSC,
    ChannelType type = ChannelType::MultiType,
    bool enable_metrics = false,
    const compression::CompressionConfig& compression_config = {}
) {
    return Channel::create(uri, buffer_size, mode, type, enable_metrics, compression_config);
}

// ============================================================================
// Convenience Type Aliases
// ============================================================================

/**
 * @typedef ChannelPtr
 * @brief Unique pointer to a Channel
 */
using ChannelPtr = std::unique_ptr<Channel>;

// Pre-defined channel types (for backward compatibility)
using SPSCChannel = Channel;  // These are now created via factory
using SPMCChannel = Channel;
using MPSCChannel = Channel;
using MPMCChannel = Channel;

// ============================================================================
// Enhanced Message Types
// ============================================================================

/**
 * @class ByteVector
 * @brief Dynamic array of raw bytes
 * 
 * Simple byte array for binary data transport. Can be viewed as
 * any data type through casting or Eigen integration.
 */
class ByteVector : public Message<ByteVector> {
public:
    static constexpr uint32_t message_type = 10;
    
    using Message<ByteVector>::Message;
    
    static size_t calculate_size() { return 1024; }
    
    uint8_t& operator[](size_t index);
    const uint8_t& operator[](size_t index) const;
    uint8_t* begin();
    uint8_t* end();
    const uint8_t* begin() const;
    const uint8_t* end() const;
    uint8_t* data();
    const uint8_t* data() const;
    size_t size() const;
    size_t capacity() const;
    void resize(size_t new_size);
    void initialize();
    
    // Eigen integration helpers
    template<typename T>
    T* as() { return reinterpret_cast<T*>(data()); }
    
    template<typename T>
    const T* as() const { return reinterpret_cast<const T*>(data()); }
};

// ============================================================================
// Reliability Features
// ============================================================================

// Forward declarations
class AcknowledgmentManager;
class RetryManager;
class HeartbeatManager;
class ReplayBuffer;

/**
 * @struct ReliabilityConfig
 * @brief Configuration for channel reliability features
 * 
 * Controls acknowledgments, retries, heartbeats, and replay buffers
 * to ensure reliable message delivery even in unreliable networks.
 */
struct ReliabilityConfig {
    bool enable_acknowledgments = true;
    bool enable_retries = true;
    bool enable_heartbeat = true;
    bool enable_replay_buffer = true;
    
    // Acknowledgment settings
    std::chrono::milliseconds ack_timeout{1000};
    
    // Retry settings
    size_t max_retries = 3;
    std::chrono::milliseconds retry_delay{100};
    float retry_backoff_factor = 2.0f;
    
    // Heartbeat settings
    std::chrono::milliseconds heartbeat_interval{5000};
    std::chrono::milliseconds heartbeat_timeout{15000};
    
    // Replay buffer settings
    size_t replay_buffer_size = 1000;
};

/**
 * @class ReliabilityManager
 * @brief Manages all reliability features for a channel
 * 
 * Coordinates acknowledgments, retries, heartbeats, and replay buffers
 * to provide reliable messaging semantics. Can be attached to any channel.
 * 
 * @example
 * @code
 * auto channel = create_channel("tcp://localhost:8080");
 * ReliabilityConfig config;
 * config.max_retries = 5;
 * config.ack_timeout = std::chrono::seconds(2);
 * 
 * ReliabilityManager reliability(*channel, config);
 * reliability.start();
 * @endcode
 */
class ReliabilityManager {
public:
    explicit ReliabilityManager(Channel& channel, const ReliabilityConfig& config = {});
    ~ReliabilityManager();
    
    void start();
    void stop();
    bool is_running() const;
    
    // Component access
    AcknowledgmentManager* acknowledgment_manager();
    RetryManager* retry_manager();
    HeartbeatManager* heartbeat_manager();
    ReplayBuffer* replay_buffer();
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @class ReliableChannelGuard
 * @brief RAII wrapper for automatic reliability management
 * 
 * Automatically starts reliability features on construction and
 * stops them on destruction. Ensures proper cleanup even if
 * exceptions are thrown.
 */
class ReliableChannelGuard {
public:
    ReliableChannelGuard(Channel& channel, const ReliabilityConfig& config = {});
    ~ReliableChannelGuard();
    
    ReliableChannelGuard(const ReliableChannelGuard&) = delete;
    ReliableChannelGuard& operator=(const ReliableChannelGuard&) = delete;
    ReliableChannelGuard(ReliableChannelGuard&&) = default;
    ReliableChannelGuard& operator=(ReliableChannelGuard&&) = delete;
    
    Channel& channel() { return channel_; }
    ReliabilityManager& manager() { return *manager_; }
    
private:
    Channel& channel_;
    std::unique_ptr<ReliabilityManager> manager_;
};

/**
 * @brief Create a channel with reliability features enabled
 * @param uri Channel URI
 * @param buffer_size Internal buffer size
 * @param mode Channel synchronization mode
 * @param config Reliability configuration
 * @return Channel with active reliability manager
 * 
 * This is a convenience function that creates a channel and
 * automatically attaches a reliability manager with the specified
 * configuration.
 */
inline std::unique_ptr<Channel> create_reliable_channel(
    const std::string& uri,
    size_t buffer_size = 1024 * 1024,
    ChannelMode mode = ChannelMode::SPSC,
    const ReliabilityConfig& config = {}
) {
    auto channel = create_channel(uri, buffer_size, mode);
    auto manager = std::make_unique<ReliabilityManager>(*channel, config);
    manager->start();
    return channel;
}

// ============================================================================
// Performance Optimizations
// ============================================================================

/**
 * @struct PerformanceConfig
 * @brief Configuration for performance optimizations
 * 
 * Controls various performance features including huge pages,
 * CPU affinity, prefetching, and SIMD optimizations.
 */
struct PerformanceConfig {
    bool enable_huge_pages = true;
    bool enable_cpu_affinity = true;
    bool enable_prefetching = true;
    bool enable_simd = true;
    size_t cache_line_size = 64;
    std::vector<int> cpu_affinity_mask;
};

/**
 * @struct BenchmarkResult
 * @brief Results from performance benchmarking
 * 
 * Contains throughput, latency percentiles, and other metrics
 * from channel performance tests.
 */
struct BenchmarkResult {
    double throughput_mbps;
    double latency_us_p50;
    double latency_us_p99;
    double latency_us_p999;
    size_t messages_sent;
    std::chrono::nanoseconds duration;
};

/**
 * @class PerformanceManager
 * @brief Manages performance optimizations and benchmarking
 * 
 * Applies various performance optimizations to channels and buffers,
 * runs benchmarks, and provides performance recommendations.
 * 
 * @note This is a singleton - use PerformanceManager::instance()
 */
class PerformanceManager {
public:
    explicit PerformanceManager(const PerformanceConfig& config = {});
    ~PerformanceManager();
    
    void apply_optimizations(Channel& channel);
    void apply_optimizations(void* buffer, size_t size);
    
    BenchmarkResult benchmark_channel(Channel& channel, size_t message_size, size_t num_messages);
    
    std::string get_summary() const;
    std::vector<std::string> get_recommendations() const;
    
    static PerformanceManager& instance();
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @class PerformanceAllocator
 * @brief High-performance memory allocator
 * 
 * Provides aligned allocation with optional huge page support
 * for optimal cache and TLB performance.
 */
class PerformanceAllocator {
public:
    static void* allocate(size_t size, size_t alignment = 64);
    static void deallocate(void* ptr, size_t size);
};

/**
 * @class OptimizedMessage
 * @brief Wrapper that applies performance optimizations to messages
 * @tparam T The base message type to optimize
 * 
 * Automatically applies performance optimizations like prefetching
 * and cache alignment when messages are created.
 */
template<typename T>
class OptimizedMessage : public T {
public:
    using T::T;
    
    OptimizedMessage(Channel& channel) : T(channel) {
        PerformanceManager::instance().apply_optimizations(
            this->data(), this->size()
        );
    }
};

// Type aliases
template<typename T>
using OptimizedVector = OptimizedMessage<FloatVector>;

template<typename T>
using CacheAlignedVector = OptimizedMessage<FloatVector>;

/**
 * @brief Get the global performance manager instance
 * @return Reference to the singleton PerformanceManager
 */
inline PerformanceManager& get_performance_manager() {
    return PerformanceManager::instance();
}

/**
 * @brief Enable global performance optimizations
 * @param config Performance configuration options
 * 
 * Call this early in your application to enable performance
 * features for all subsequently created channels and messages.
 */
inline void enable_performance_optimizations(const PerformanceConfig& config = {}) {
    static PerformanceManager manager(config);
}

/**
 * @brief Disable global performance optimizations
 * 
 * @note Currently a no-op, optimizations persist once enabled
 */
inline void disable_performance_optimizations() {
    // No-op for now
}

/**
 * @brief Get a summary of current performance metrics
 * @return Human-readable performance summary
 */
std::string get_performance_summary();

/**
 * @brief Get performance tuning recommendations
 * @return List of recommendations based on current system state
 */
std::vector<std::string> get_performance_recommendations();

/**
 * @brief Run performance benchmarks on a channel
 * @param channel Channel to benchmark
 * @param message_size Size of test messages in bytes
 * @param num_messages Number of messages to send
 * @return Benchmark results including throughput and latency
 * 
 * @example
 * @code
 * auto channel = create_channel("memory://bench");
 * auto results = run_performance_benchmarks(*channel, 4096, 100000);
 * std::cout << "Throughput: " << results.throughput_mbps << " MB/s\n";
 * @endcode
 */
inline BenchmarkResult run_performance_benchmarks(
    Channel& channel,
    size_t message_size = 1024,
    size_t num_messages = 10000
) {
    return PerformanceManager::instance().benchmark_channel(
        channel, message_size, num_messages
    );
}

/**
 * @brief Create a message with performance optimizations applied
 * @tparam T Message type to create
 * @param channel Channel to allocate the message in
 * @return Optimized message instance
 */
template<typename T>
inline OptimizedMessage<T> create_optimized_message(Channel& channel) {
    return OptimizedMessage<T>(channel);
}

// ============================================================================
// Debug and Introspection API
// ============================================================================

/**
 * @brief Get comprehensive diagnostic information for a channel
 * @param channel The channel to inspect
 * @return Complete diagnostic report
 */
debug::ChannelDiagnostics inspect_channel(const Channel& channel);

/**
 * @brief Get a human-readable status report for a channel
 * @param channel The channel to inspect
 * @return Formatted status report
 */
std::string get_channel_status(const Channel& channel);

/**
 * @brief Visualize channel buffer usage
 * @param channel The channel to visualize
 * @param width Width of the visualization bar (default: 50)
 * @return ASCII art buffer visualization
 */
std::string visualize_channel_buffer(const Channel& channel, size_t width = 50);

/**
 * @brief Run health check on a channel
 * @param channel The channel to check
 * @return List of health issues found
 */
std::vector<std::string> check_channel_health(const Channel& channel);

/**
 * @brief Start performance profiling for a channel
 * @param channel The channel to profile
 * @return Performance profiler instance
 */
std::unique_ptr<debug::PerformanceProfiler> start_profiling(const Channel& channel);

/**
 * @brief Enable message tracing for a channel
 * @param channel_uri URI of the channel to trace
 */
void enable_message_tracing(const std::string& channel_uri);

/**
 * @brief Get message trace report for a channel
 * @param channel_uri URI of the channel
 * @return Formatted trace report
 */
std::string get_trace_report(const std::string& channel_uri);

/**
 * @brief Create a debugging dashboard showing all channels
 * @return ASCII art dashboard
 */
std::string create_debug_dashboard();

// ============================================================================
// UDP Multicast API
// ============================================================================

namespace multicast {

/**
 * @enum Role
 * @brief Role of the multicast endpoint
 */
enum class Role {
    Publisher,   ///< Sends multicast messages
    Subscriber   ///< Receives multicast messages
};

/**
 * @brief Create a UDP multicast publisher
 * @param multicast_address Multicast group address (e.g., "239.255.0.1")
 * @param port Port number
 * @param buffer_size Buffer size for messages (default: 1MB)
 * @param compression_config Optional compression configuration
 * @return Unique pointer to the channel
 */
std::unique_ptr<Channel> create_publisher(
    const std::string& multicast_address, uint16_t port,
    size_t buffer_size = 1024 * 1024,
    const compression::CompressionConfig& compression_config = {});

/**
 * @brief Create a UDP multicast subscriber
 * @param multicast_address Multicast group address (e.g., "239.255.0.1")
 * @param port Port number
 * @param buffer_size Buffer size for messages (default: 1MB)
 * @param interface_address Local interface to bind to (optional)
 * @return Unique pointer to the channel
 */
std::unique_ptr<Channel> create_subscriber(
    const std::string& multicast_address, uint16_t port,
    size_t buffer_size = 1024 * 1024,
    const std::string& interface_address = "");

/**
 * @brief Create a UDP multicast channel with specific role
 * @param multicast_address Multicast group address
 * @param port Port number
 * @param role Publisher or Subscriber
 * @param buffer_size Buffer size for messages
 * @param compression_config Compression configuration (publishers only)
 * @param interface_address Local interface address (subscribers only)
 * @return Unique pointer to the channel
 */
std::unique_ptr<Channel> create_multicast_channel(
    const std::string& multicast_address, uint16_t port,
    Role role,
    size_t buffer_size = 1024 * 1024,
    const compression::CompressionConfig& compression_config = {},
    const std::string& interface_address = "");

} // namespace multicast

// ============================================================================
// RDMA/InfiniBand Support  
// ============================================================================

namespace rdma {

// Forward declarations for RDMA types (defined in detail namespace)
// Note: RDMA types are actually in psyne::detail namespace

/**
 * @brief Create an RDMA server channel for high-performance computing
 * @param port Port to listen on
 * @param buffer_size Buffer size for RDMA operations (default: 1MB)
 * @return Unique pointer to the RDMA channel
 */
std::unique_ptr<Channel> create_server(uint16_t port, 
                                      size_t buffer_size = 1024 * 1024);

/**
 * @brief Create an RDMA client channel
 * @param host Remote host address
 * @param port Remote port
 * @param buffer_size Buffer size for operations (default: 1MB)
 * @return Unique pointer to the RDMA channel
 */
std::unique_ptr<Channel> create_client(const std::string& host, uint16_t port,
                                      size_t buffer_size = 1024 * 1024);

} // namespace rdma

// ============================================================================
// Enhanced Message Types
// ============================================================================

namespace types {

// Forward declarations
class Matrix4x4f;
class Matrix3x3f;
class Vector3f;
class Vector4f;
class Int8Vector;
class UInt8Vector;
class ComplexVectorF;
class MLTensorF;
class SparseMatrixF;

// Type aliases for convenience
using Matrix4f = Matrix4x4f;
using Matrix3f = Matrix3x3f;
using Vec3f = Vector3f;
using Vec4f = Vector4f;
using ComplexF = ComplexVectorF;
using MLTensor = MLTensorF;
using SparseMatrix = SparseMatrixF;
using QInt8 = Int8Vector;
using QUInt8 = UInt8Vector;



// ============================================================================
// Simplified Implementations for Other Types
// ============================================================================


/**
 * @class ComplexVectorF
 * @brief Vector of single-precision complex numbers
 */
class ComplexVectorF : public Message<ComplexVectorF> {
public:
    static constexpr uint32_t message_type = 107;
    using ComplexType = std::complex<float>;
    
    using Message<ComplexVectorF>::Message;
    
    static size_t calculate_size() {
        return 1024 * sizeof(ComplexType);
    }
    
    void initialize() {
        header().size = 0;
    }
    
    struct Header {
        uint32_t size;
        uint32_t padding;
    };
    
    Header& header() { return *reinterpret_cast<Header*>(Message<ComplexVectorF>::data()); }
    const Header& header() const { return *reinterpret_cast<const Header*>(Message<ComplexVectorF>::data()); }
    
    ComplexType* data() { 
        return reinterpret_cast<ComplexType*>(
            reinterpret_cast<uint8_t*>(Message<ComplexVectorF>::data()) + sizeof(Header)
        ); 
    }
    
    const ComplexType* data() const { 
        return reinterpret_cast<const ComplexType*>(
            reinterpret_cast<const uint8_t*>(Message<ComplexVectorF>::data()) + sizeof(Header)
        ); 
    }
    
    size_t size() const { return header().size; }
    
    void resize(size_t new_size) {
        header().size = static_cast<uint32_t>(new_size);
    }
    
    ComplexType& operator[](size_t index) {
        return data()[index];
    }
    
    const ComplexType& operator[](size_t index) const {
        return data()[index];
    }
    
    float power() const {
        float total = 0.0f;
        for (size_t i = 0; i < size(); ++i) {
            float real = data()[i].real();
            float imag = data()[i].imag();
            total += real * real + imag * imag;
        }
        return total;
    }
    
    void conjugate() {
        for (size_t i = 0; i < size(); ++i) {
            data()[i] = std::conj(data()[i]);
        }
    }
};

/**
 * @class MLTensorF
 * @brief Multi-dimensional tensor for machine learning
 */
class MLTensorF : public Message<MLTensorF> {
public:
    static constexpr uint32_t message_type = 108;
    static constexpr size_t MAX_DIMS = 8;
    
    enum class Layout {
        NCHW,
        NHWC,
        CHW,
        HWC,
        Custom
    };
    
    using Message<MLTensorF>::Message;
    
    static size_t calculate_size() {
        return 1024 * 1024; // 1MB default
    }
    
    void initialize() {
        header().num_dims = 0;
        header().layout = Layout::Custom;
        std::fill(header().shape, header().shape + MAX_DIMS, 0);
        header().total_elements = 0;
    }
    
    struct Header {
        uint32_t num_dims;
        Layout layout;
        uint32_t shape[MAX_DIMS];
        uint32_t total_elements;
        uint32_t padding[2];
    };
    
    Header& header() { return *reinterpret_cast<Header*>(Message<MLTensorF>::data()); }
    const Header& header() const { return *reinterpret_cast<const Header*>(Message<MLTensorF>::data()); }
    
    float* data() { 
        return reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(Message<MLTensorF>::data()) + sizeof(Header)
        ); 
    }
    
    const float* data() const { 
        return reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(Message<MLTensorF>::data()) + sizeof(Header)
        ); 
    }
    
    void set_shape(const std::vector<uint32_t>& new_shape, Layout layout = Layout::Custom) {
        header().num_dims = static_cast<uint32_t>(new_shape.size());
        header().layout = layout;
        
        std::copy(new_shape.begin(), new_shape.end(), header().shape);
        
        uint32_t total = 1;
        for (uint32_t dim : new_shape) {
            total *= dim;
        }
        header().total_elements = total;
    }
    
    std::vector<uint32_t> shape() const {
        return std::vector<uint32_t>(header().shape, header().shape + header().num_dims);
    }
    
    uint32_t total_elements() const { return header().total_elements; }
    Layout layout() const { return header().layout; }
};

/**
 * @class SparseMatrixF
 * @brief Sparse matrix in CSR format
 */
class SparseMatrixF : public Message<SparseMatrixF> {
public:
    static constexpr uint32_t message_type = 109;
    
    using Message<SparseMatrixF>::Message;
    
    static size_t calculate_size() {
        return 1024 * 1024; // 1MB default
    }
    
    void initialize() {
        header().rows = 0;
        header().cols = 0;
        header().nnz = 0;
    }
    
    struct Header {
        uint32_t rows;
        uint32_t cols;
        uint32_t nnz;
        uint32_t padding;
    };
    
    Header& header() { return *reinterpret_cast<Header*>(Message<SparseMatrixF>::data()); }
    const Header& header() const { return *reinterpret_cast<const Header*>(Message<SparseMatrixF>::data()); }
    
    float* values() { 
        return reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(Message<SparseMatrixF>::data()) + sizeof(Header)
        ); 
    }
    
    const float* values() const { 
        return reinterpret_cast<const float*>(
            reinterpret_cast<const uint8_t*>(Message<SparseMatrixF>::data()) + sizeof(Header)
        ); 
    }
    
    uint32_t* column_indices() {
        return reinterpret_cast<uint32_t*>(values() + header().nnz);
    }
    
    const uint32_t* column_indices() const {
        return reinterpret_cast<const uint32_t*>(values() + header().nnz);
    }
    
    uint32_t* row_pointers() {
        return column_indices() + header().nnz;
    }
    
    const uint32_t* row_pointers() const {
        return column_indices() + header().nnz;
    }
    
    uint32_t rows() const { return header().rows; }
    uint32_t cols() const { return header().cols; }
    uint32_t nnz() const { return header().nnz; }
    
    void set_structure(uint32_t rows, uint32_t cols, uint32_t nnz) {
        header().rows = rows;
        header().cols = cols;
        header().nnz = nnz;
        std::fill(row_pointers(), row_pointers() + rows + 1, 0);
    }
    
    void multiply_vector(const float* x, float* y) const {
        for (uint32_t row = 0; row < header().rows; ++row) {
            float sum = 0.0f;
            uint32_t start = row_pointers()[row];
            uint32_t end = row_pointers()[row + 1];
            
            for (uint32_t idx = start; idx < end; ++idx) {
                sum += values()[idx] * x[column_indices()[idx]];
            }
            
            y[row] = sum;
        }
    }
};

}

} // namespace psyne

// Optional Eigen integration
#ifdef PSYNE_ENABLE_EIGEN
#include <Eigen/Dense>

namespace psyne {

/**
 * @brief Create an Eigen::Map view of FloatVector data
 * @param vec The FloatVector to view
 * @param rows Number of rows (for matrix view)
 * @param cols Number of columns (for matrix view)
 * @return Eigen Map object for zero-copy access
 */
template<typename Derived = Eigen::Dynamic>
auto as_eigen_vector(FloatVector& vec) {
    return Eigen::Map<Eigen::VectorXf>(vec.begin(), vec.size());
}

template<typename Derived = Eigen::Dynamic>
auto as_eigen_vector(const FloatVector& vec) {
    return Eigen::Map<const Eigen::VectorXf>(vec.begin(), vec.size());
}

template<typename Derived = Eigen::Dynamic>
auto as_eigen_matrix(FloatVector& vec, size_t rows, size_t cols) {
    return Eigen::Map<Eigen::MatrixXf>(vec.begin(), rows, cols);
}

template<typename Derived = Eigen::Dynamic>
auto as_eigen_matrix(const FloatVector& vec, size_t rows, size_t cols) {
    return Eigen::Map<const Eigen::MatrixXf>(vec.begin(), rows, cols);
}

/**
 * @brief Create an Eigen::Map view of DoubleMatrix data
 */
auto as_eigen_matrix(DoubleMatrix& mat) {
    return Eigen::Map<Eigen::MatrixXd>(
        reinterpret_cast<double*>(mat.data()), 
        mat.rows(), 
        mat.cols()
    );
}

auto as_eigen_matrix(const DoubleMatrix& mat) {
    return Eigen::Map<const Eigen::MatrixXd>(
        reinterpret_cast<const double*>(mat.data()), 
        mat.rows(), 
        mat.cols()
    );
}

/**
 * @brief Create an Eigen view of ByteVector as any numeric type
 */
template<typename T>
auto as_eigen_vector(ByteVector& vec) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        vec.as<T>(), 
        vec.size() / sizeof(T)
    );
}

template<typename T>
auto as_eigen_matrix(ByteVector& vec, size_t rows, size_t cols) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        vec.as<T>(), 
        rows, 
        cols
    );
}

// Include enhanced message types to provide complete implementations
// for forward-declared classes like Matrix4x4f, Vector3f, Int8Vector
#include "../src/types/enhanced_types.hpp"

} // namespace psyne
#endif // PSYNE_ENABLE_EIGEN


/**
 * @mainpage Psyne Documentation
 * 
 * @section intro_sec Introduction
 * 
 * Psyne is a high-performance, zero-copy messaging library designed for AI/ML
 * applications. It provides efficient inter-process and inter-thread communication
 * with minimal overhead.
 * 
 * @section install_sec Installation
 * 
 * @code{.sh}
 * mkdir build && cd build
 * cmake ..
 * make -j8
 * sudo make install
 * @endcode
 * 
 * @section example_sec Quick Example
 * 
 * @code{.cpp}
 * #include <psyne/psyne.hpp>
 * 
 * int main() {
 *     // Create a channel
 *     auto chan = psyne::create_channel("ipc://shared1");
 *     
 *     // Send data
 *     psyne::FloatVector data(*chan);
 *     data.resize(1000);
 *     // ... fill data ...
 *     data.send();
 *     
 *     return 0;
 * }
 * @endcode
 * 
 * @section links_sec Links
 * - GitHub: https://github.com/joshmorgan1000/psyne
 * - Documentation: https://github.com/joshmorgan1000/psyne/docs
 */

