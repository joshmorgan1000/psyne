#pragma once

/**
 * @file psyne.hpp
 * @brief Psyne - High-performance zero-copy messaging library optimized for
 * AI/ML
 *
 * This is the single public API header for the Psyne library.
 * Include this file to access all Psyne functionality.
 *
 * @author Psyne Contributors
 * @version 1.2.0
 * @date 2025
 *
 * @copyright Copyright (c) 2025 Psyne Contributors
 *
 * @example simple_messaging.cpp
 * @example producer_consumer.cpp
 * @example channel_factory_demo.cpp
 */

// Psyne - Zero-copy messaging library
// This is the single public API header

#include <atomic>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>

// C++20 Modern Features (CORE_DESIGN.md requirements)
#include <span>
#include <concepts>
#include <ranges>
#include <type_traits>
#include <bit>

// Async support (optional)
#ifdef PSYNE_ASYNC_SUPPORT
#include <boost/asio/awaitable.hpp>
#include <boost/asio/executor.hpp>
#include <boost/asio/io_context.hpp>
#endif

// Version information
#define PSYNE_VERSION_MAJOR 1
#define PSYNE_VERSION_MINOR 2
#define PSYNE_VERSION_PATCH 2

/**
 * @namespace psyne
 * @brief Main namespace for the Psyne messaging library
 */
namespace psyne {

/**
 * @brief Get the version string of the Psyne library
 * @return Version string in the format "major.minor.patch"
 */
constexpr const char *version() {
    return "1.3.0";
}

/**
 * @brief Get the version string of the Psyne library (runtime version)
 * @return Version string in the format "major.minor.patch"
 */
const char *get_version();

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
    SPSC, ///< Single Producer, Single Consumer - Highest performance, lock-free
    SPMC, ///< Single Producer, Multiple Consumer - One writer, many readers
    MPSC, ///< Multiple Producer, Single Consumer - Many writers, one reader
    MPMC  ///< Multiple Producer, Multiple Consumer - Full multi-threading
          ///< support
};

/**
 * @enum ChannelType
 * @brief Defines whether a channel supports single or multiple message types
 */
enum class ChannelType {
    SingleType, ///< Optimized for single message type (no type metadata
                ///< overhead)
    MultiType   ///< Supports multiple message types (small metadata overhead)
};

// Forward declarations
class Channel;
class RingBuffer;
template <typename Derived>
class Message;

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
    uint64_t messages_sent = 0;     ///< Number of messages sent
    uint64_t bytes_sent = 0;        ///< Total bytes sent
    uint64_t messages_received = 0; ///< Number of messages received
    uint64_t bytes_received = 0;    ///< Total bytes received
    uint64_t send_blocks = 0;       ///< Times send() blocked waiting for space
    uint64_t receive_blocks = 0; ///< Times receive() blocked waiting for data

    ChannelMetrics() = default;
    ChannelMetrics(const ChannelMetrics &) = default;
    ChannelMetrics &operator=(const ChannelMetrics &) = default;
    ChannelMetrics(ChannelMetrics &&) = default;
    ChannelMetrics &operator=(ChannelMetrics &&) = default;
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
        result.messages_received =
            messages_received.load(std::memory_order_relaxed);
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
    None = 0,  ///< No compression
    LZ4 = 1,   ///< Fast compression/decompression
    Zstd = 2,  ///< Better compression ratio
    Snappy = 3 ///< Google Snappy - balanced speed/ratio
};

/**
 * @struct CompressionConfig
 * @brief Configuration for compression behavior
 */
struct CompressionConfig {
    CompressionType type = CompressionType::None;
    int level = 1; ///< Compression level (algorithm dependent)
    size_t min_size_threshold =
        128;                     ///< Don't compress messages smaller than this
    bool enable_checksum = true; ///< Add checksum for compressed data
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
template <typename Derived>
class Message {
public:
    /**
     * @brief Create an outgoing message (view into channel's ring buffer)
     * @param channel The channel whose ring buffer to view
     * @throws std::runtime_error if ring buffer is full
     */
    explicit Message(Channel &channel);  // Implementation after Channel definition

    /**
     * @brief Create an incoming message (view of existing data)
     * @param data Pointer to the message data
     * @param size Size of the message in bytes
     */
    explicit Message(const void *data, size_t size)
        : data_(const_cast<uint8_t *>(static_cast<const uint8_t *>(data))),
          size_(size), channel_(nullptr), handle_(nullptr), offset_(0) {}

    /**
     * @brief Move constructor
     * @param other Message to move from
     */
    Message(Message &&other) noexcept
        : data_(other.data_), size_(other.size_), channel_(other.channel_),
          handle_(other.handle_), offset_(other.offset_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.channel_ = nullptr;
        other.handle_ = nullptr;
        other.offset_ = 0;
    }

    /**
     * @brief Move assignment operator
     * @param other Message to move from
     * @return Reference to this message
     */
    Message &operator=(Message &&other) noexcept {
        if (this != &other) {
            data_ = other.data_;
            size_ = other.size_;
            channel_ = other.channel_;
            handle_ = other.handle_;
            offset_ = other.offset_;
            
            other.data_ = nullptr;
            other.size_ = 0;
            other.channel_ = nullptr;
            other.handle_ = nullptr;
            other.offset_ = 0;
        }
        return *this;
    }

    // Deleted copy operations to ensure zero-copy semantics
    Message(const Message &) = delete;
    Message &operator=(const Message &) = delete;

    /**
     * @brief Destructor
     */
    virtual ~Message() {
        // Nothing to do - data points into channel's ring buffer
        // Channel manages its own memory
    }

    /**
     * @brief Send the message through its channel
     * @throws std::runtime_error if message has no associated channel
     */
    void send();

    /**
     * @brief Check if the message is valid (has allocated data)
     * @return true if valid, false otherwise
     */
    bool is_valid() const {
        return data_ != nullptr;
    }

    /**
     * @brief Get pointer to message data
     * @return Pointer to message data in ring buffer
     */
    uint8_t *data() {
        return data_;
    }
    
    /**
     * @brief Get const pointer to message data
     * @return Const pointer to message data in ring buffer
     */
    const uint8_t *data() const {
        return data_;
    }

    /**
     * @brief Get the message type ID
     * @return Type ID of this message class
     */
    static constexpr uint32_t type() {
        return Derived::message_type;
    }


    /**
     * @brief Get the size of the message data
     * @return Size in bytes
     */
    size_t size() const {
        return size_;
    }

protected:
    /**
     * @brief Get reference to the channel that owns this message
     * @return Reference to the channel
     * 
     * This is used by derived classes that need to interact with the channel,
     * for example to check buffer sizes or channel capabilities.
     */
    Channel &channel() {
        return *channel_;
    }
    
    /**
     * @brief Get const reference to the channel that owns this message
     * @return Const reference to the channel
     */
    const Channel &channel() const {
        return *channel_;
    }

    /**
     * @brief Hook called before sending the message
     * 
     * Derived classes can override this to perform validation or
     * finalization before the message is sent. Default implementation
     * does nothing.
     */
    virtual void before_send() {}

    // Protected data members - accessible by derived classes
    uint8_t *data_;    ///< Pointer into channel's ring buffer (zero-copy)
    size_t size_;      ///< Size of this message in bytes
    Channel *channel_; ///< Channel that owns the ring buffer (null for received messages)
    void *handle_;     ///< Reserved for future use (legacy compatibility)
    uint32_t offset_;  ///< Offset in ring buffer for zero-copy operations
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

    /**
     * @brief Calculate the required buffer size for FloatVector
     * @return Size in bytes (64MB)
     * 
     * Returns the maximum size needed for a FloatVector message.
     * This is set to 64MB to accommodate massive GPU workloads
     * (up to 16,777,216 floats plus header).
     * 
     * @note This large size is designed for AI/ML tensor operations.
     *       For smaller messages, consider using a custom message type.
     */
    static size_t calculate_size() {
        // Large buffer for 16M+ floats (64MB)
        // 16,777,216 floats * 4 bytes + 8 bytes header = 67,108,872 bytes
        return 64 * 1024 * 1024; // 64MB buffer for massive GPU workloads
    }

    /**
     * @brief Array subscript operator for element access
     * @param index Element index
     * @return Reference to the element at the specified index
     * @throws std::out_of_range if index >= size()
     */
    float &operator[](size_t index);
    
    /**
     * @brief Const array subscript operator for element access
     * @param index Element index
     * @return Const reference to the element at the specified index
     * @throws std::out_of_range if index >= size()
     */
    const float &operator[](size_t index) const;

    /**
     * @brief Get iterator to the beginning of the vector
     * @return Pointer to the first element
     */
    float *begin();
    
    /**
     * @brief Get iterator to the end of the vector
     * @return Pointer to one past the last element
     */
    float *end();
    
    /**
     * @brief Get const iterator to the beginning of the vector
     * @return Const pointer to the first element
     */
    const float *begin() const;
    
    /**
     * @brief Get const iterator to the end of the vector
     * @return Const pointer to one past the last element
     */
    const float *end() const;

    /**
     * @brief Get the current number of elements
     * @return Number of float elements in the vector
     */
    size_t size() const;
    
    /**
     * @brief Get the maximum possible number of elements
     * @return Maximum capacity in number of floats
     * 
     * The capacity is determined by the buffer size minus the header.
     */
    size_t capacity() const;
    
    /**
     * @brief Resize the vector to contain new_size elements
     * @param new_size New number of elements
     * @throws std::runtime_error if new_size > capacity()
     * 
     * If new_size is greater than the current size, new elements
     * are uninitialized. If smaller, the vector is truncated.
     */
    void resize(size_t new_size);

    /**
     * @brief Assign values from an initializer list
     * @param values List of float values to assign
     * @return Reference to this vector
     * @throws std::runtime_error if values.size() > capacity()
     * 
     * Resizes the vector to match the initializer list size and
     * copies all values.
     */
    FloatVector &operator=(std::initializer_list<float> values);

    /**
     * @brief Initialize the message after allocation
     * 
     * Sets up the header and initializes the size to 0.
     * Called automatically by the Message constructor.
     */
    void initialize();

// Eigen integration for linear algebra operations
#ifdef PSYNE_ENABLE_EIGEN
    /**
     * @brief Get an Eigen view of this vector
     * @return Eigen::Map<Eigen::VectorXf> wrapping this vector's data
     * 
     * Provides zero-copy integration with Eigen for linear algebra operations.
     * The returned map directly references this vector's data.
     * 
     * @warning The map becomes invalid if the vector is resized or destroyed.
     */
    auto as_eigen() {
        return Eigen::Map<Eigen::VectorXf>(begin(), size());
    }

    /**
     * @brief Get a const Eigen view of this vector
     * @return Eigen::Map<const Eigen::VectorXf> wrapping this vector's data
     */
    auto as_eigen() const {
        return Eigen::Map<const Eigen::VectorXf>(begin(), size());
    }
#else
    // Dummy implementation for tests
    struct DummyEigenView {
        float *data_;
        size_t size_;
        DummyEigenView(float *data, size_t size) : data_(data), size_(size) {}
        size_t size() const {
            return size_;
        }
        float operator()(size_t index) const {
            return data_[index];
        }
    };

    DummyEigenView as_eigen() {
        return DummyEigenView(begin(), size());
    }

    DummyEigenView as_eigen() const {
        return DummyEigenView(const_cast<float *>(begin()), size());
    }
#endif
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
    double &at(size_t row, size_t col);
    const double &at(size_t row, size_t col) const;

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
     * @brief Default constructor
     */
    Channel() = default;

    /**
     * @brief Constructor with URI and buffer size (for backward compatibility)
     * @param uri Channel URI
     * @param buffer_size Size of the internal buffer
     * @param type Channel type (default: MultiType)
     */
    Channel(const std::string &uri, size_t buffer_size,
            ChannelType type = ChannelType::MultiType)
        : uri_(uri), buffer_size_(buffer_size), type_(type) {}

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
    static std::unique_ptr<Channel>
    create(const std::string &uri, size_t buffer_size,
           ChannelMode mode = ChannelMode::SPSC,
           ChannelType type = ChannelType::MultiType,
           bool enable_metrics = false,
           const compression::CompressionConfig &compression_config = {});

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
    template <typename MessageType>
    void send(MessageType &msg) {
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
    template <typename MessageType>
    std::optional<MessageType> receive(std::chrono::milliseconds /*timeout*/ =
                                           std::chrono::milliseconds::zero()) {
        size_t size;
        uint32_t type_id;

        // Get message from implementation using virtual method
        void *data = receive_raw_message(size, type_id);
        if (!data) {
            return std::nullopt;
        }

        // Verify type matches (for safety)
        if (type() == ChannelType::MultiType &&
            type_id != MessageType::message_type) {
            release_raw_message(data);
            return std::nullopt;
        }

        // Create message view from received data
        // For now, we use the data pointer directly since receive_raw_message
        // returns a pointer to the message data
        return MessageType(data, size);
    }

    /**
     * @brief Receive a single message from the channel (blocking)
     * @tparam MessageType Expected message type
     * @param timeout Maximum time to wait (default: infinite)
     * @return Optional containing the message if received, empty on timeout
     */
    template <typename MessageType>
    std::optional<MessageType> receive_single(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        return receive<MessageType>(timeout);
    }

    // Event-driven listening
    template <typename MessageType>
    std::unique_ptr<std::thread>
    listen(std::function<void(MessageType &&)> handler) {
        return std::make_unique<std::thread>([this, handler]() {
            while (!is_stopped()) {
                auto msg = receive<MessageType>(std::chrono::milliseconds(100));
                if (msg) {
                    handler(std::move(*msg));
                }
            }
        });
    }

    // Async/await support (requires boost.asio)
#ifdef PSYNE_ASYNC_SUPPORT
    /**
     * @brief Asynchronously receive a message (coroutine)
     * @tparam MessageType Type of message to receive
     * @param timeout Timeout duration (0 = no timeout)
     * @return Awaitable that yields optional message
     * @note Requires linking with boost_coroutine and boost_context
     */
    template <typename MessageType>
    boost::asio::awaitable<std::optional<MessageType>> async_receive(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());

    /**
     * @brief Asynchronously receive a single message
     * @tparam MessageType Type of message to receive
     * @param timeout Timeout duration
     * @return Awaitable that yields optional message
     */
    template <typename MessageType>
    boost::asio::awaitable<std::optional<MessageType>> async_receive_single(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(1000));

    /**
     * @brief Set shared io_context for async operations
     * @param io_ctx Shared io_context to use
     */
    virtual void
    set_io_context(std::shared_ptr<boost::asio::io_context> io_ctx) {
        (void)io_ctx; // Default does nothing
    }

    /**
     * @brief Get the executor for async operations
     * @return Executor or nullptr if async not supported
     */
    virtual boost::asio::io_context::executor_type *get_executor() {
        return nullptr;
    }
#endif // PSYNE_ASYNC_SUPPORT

    /**
     * @brief Stop the channel
     * 
     * Signals that the channel should stop processing messages.
     * This is used for graceful shutdown. After calling stop(),
     * is_stopped() will return true.
     * 
     * @note This does not immediately interrupt blocking operations.
     *       Receivers should check is_stopped() periodically.
     */
    virtual void stop() {
        stopped_ = true;
    }
    
    /**
     * @brief Check if the channel has been stopped
     * @return true if stop() has been called, false otherwise
     */
    virtual bool is_stopped() const {
        return stopped_;
    }

    /**
     * @brief Get the channel's URI
     * @return The URI string (e.g., "memory://buffer1", "tcp://localhost:8080")
     * 
     * The URI uniquely identifies the channel and specifies its transport type.
     */
    virtual const std::string &uri() const {
        return uri_;
    }
    
    /**
     * @brief Get the channel type
     * @return ChannelType::SingleType or ChannelType::MultiType
     * 
     * SingleType channels are optimized for one message type.
     * MultiType channels can handle multiple message types with runtime dispatch.
     */
    virtual ChannelType type() const {
        return type_;
    }
    
    /**
     * @brief Get the synchronization mode
     * @return The channel's ChannelMode (SPSC, SPMC, MPSC, or MPMC)
     * 
     * The mode determines thread-safety guarantees:
     * - SPSC: Single Producer, Single Consumer (lock-free)
     * - SPMC: Single Producer, Multiple Consumer
     * - MPSC: Multiple Producer, Single Consumer
     * - MPMC: Multiple Producer, Multiple Consumer
     */
    virtual ChannelMode mode() const {
        return mode_;
    }

    // Zero-copy ring buffer interface - the core of Psyne's design
    // These methods implement the true zero-copy message paradigm
    
    /**
     * @brief Reserve space in ring buffer and return offset
     * @param size Size of message to reserve
     * @return Offset within ring buffer, or BUFFER_FULL if no space
     */
    virtual uint32_t reserve_write_slot(size_t size) noexcept = 0;
    
    /**
     * @brief Notify receiver that message is ready at offset
     * @param offset Offset within ring buffer where message data starts
     * @param size Size of the message
     */
    virtual void notify_message_ready(uint32_t offset, size_t size) noexcept = 0;
    
    /**
     * @brief Get direct reference to underlying ring buffer
     * @return Reference to ring buffer for direct memory access
     */
    virtual RingBuffer& get_ring_buffer() noexcept = 0;
    virtual const RingBuffer& get_ring_buffer() const noexcept = 0;
    
    /**
     * @brief Consumer advances read pointer after processing message
     * @param size Size of message that was consumed
     */
    virtual void advance_read_pointer(size_t size) noexcept = 0;

    /**
     * @brief Get zero-copy view of available ring buffer space
     * @return Span over available buffer space for direct writing
     */
    virtual std::span<uint8_t> get_write_span(size_t size) noexcept {
        // Default implementation for backward compatibility
        return std::span<uint8_t>{};
    }

    /**
     * @brief Get channel buffer as zero-copy span
     * @return Span over entire ring buffer
     */
    virtual std::span<const uint8_t> buffer_span() const noexcept {
        return std::span<const uint8_t>{};
    }
    
    /**
     * @brief Buffer full constant for backpressure handling
     */
    static constexpr uint32_t BUFFER_FULL = 0xFFFFFFFF;
    
    // Legacy interface - DEPRECATED (violates zero-copy principles)
    [[deprecated("Use reserve_write_slot() instead - violates zero-copy design")]]
    virtual uint8_t *get_write_buffer(size_t /*size*/) {
        return nullptr;
    }
    [[deprecated("Data is committed when written - violates zero-copy design")]]
    virtual void commit_write(size_t /*size*/) {
        // DEPRECATED - violates CORE_DESIGN.md principles
    }
    
    // Legacy interface for compatibility
    virtual void send_raw_message(const void * /*data*/, size_t /*size*/,
                                  uint32_t /*type*/) {}
    virtual void *receive_raw_message(size_t & /*size*/, uint32_t & /*type*/) {
        return nullptr;
    }
    virtual void release_raw_message(void * /*handle*/) {}

    /**
     * @brief Check if this channel has metrics collection enabled
     * @return true if metrics are being collected, false otherwise
     * 
     * Metrics collection adds minimal overhead but provides valuable
     * performance insights including throughput, latency, and blocking statistics.
     */
    virtual bool has_metrics() const {
        return false;
    }
    
    /**
     * @brief Get current channel metrics
     * @return ChannelMetrics structure with performance statistics
     * 
     * Returns metrics including:
     * - messages_sent/received: Total message counts
     * - bytes_sent/received: Total data volume
     * - send_blocks/receive_blocks: Times operations blocked
     * 
     * @note Returns empty metrics if has_metrics() is false
     */
    virtual debug::ChannelMetrics get_metrics() const {
        return debug::ChannelMetrics{};
    }
    
    /**
     * @brief Reset all metrics counters to zero
     * 
     * Useful for benchmarking specific sections of code or
     * resetting statistics after warmup periods.
     */
    virtual void reset_metrics() {}

    /**
     * @brief Get pointer to implementation (for language bindings)
     * @return Pointer to internal implementation
     * 
     * This method is primarily used by Python bindings and other
     * language integrations to access the underlying implementation.
     * 
     * @warning Direct use of the implementation bypasses the public API
     *          and may break with future versions.
     */
    detail::ChannelImpl *get_impl() {
        return impl();
    }
    
    /**
     * @brief Get const pointer to implementation
     * @return Const pointer to internal implementation
     */
    const detail::ChannelImpl *get_impl() const {
        return impl();
    }

protected:
    // Member variables for basic implementation
    std::string uri_;                                ///< Channel URI (e.g., "memory://buffer1")
    size_t buffer_size_ = 0;                         ///< Size of internal ring buffer in bytes
    ChannelType type_ = ChannelType::MultiType;     ///< SingleType or MultiType channel
    ChannelMode mode_ = ChannelMode::SPSC;           ///< Synchronization mode (SPSC, SPMC, etc.)
    bool stopped_ = false;                           ///< Flag indicating channel is stopped

    /**
     * @brief Get pointer to the implementation
     * @return Pointer to implementation or nullptr for base class
     * 
     * Derived classes override this to return their specific implementation.
     * The base Channel class returns nullptr.
     */
    virtual detail::ChannelImpl *impl() {
        return nullptr;
    }
    
    /**
     * @brief Get const pointer to the implementation
     * @return Const pointer to implementation or nullptr for base class
     */
    virtual const detail::ChannelImpl *impl() const {
        return nullptr;
    }

    // Friend declaration to allow Message to access impl()
    template <typename Derived>
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
 * auto server = psyne::create_channel("tcp://:8080", 64*1024*1024,
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
inline std::unique_ptr<Channel>
create_channel(const std::string &uri, size_t buffer_size = 64 * 1024 * 1024,
               ChannelMode mode = ChannelMode::SPSC,
               ChannelType type = ChannelType::MultiType,
               bool enable_metrics = false,
               const compression::CompressionConfig &compression_config = {}) {
    return Channel::create(uri, buffer_size, mode, type, enable_metrics,
                           compression_config);
}

/**
 * @brief Create an IPC server (creates shared memory)
 * @param name IPC channel name (without ipc:// prefix)
 * @param buffer_size Buffer size (minimum 64MB)
 * @param mode Channel mode (default: SPSC)
 * @param type Channel type (default: MultiType)
 * @return Unique pointer to the channel
 * 
 * @note Only the server creates the shared memory segment. 
 *       Clients should use create_ipc_client() to attach.
 */
inline std::unique_ptr<Channel>
create_ipc_server(const std::string &name, size_t buffer_size = 64 * 1024 * 1024,
                  ChannelMode mode = ChannelMode::SPSC,
                  ChannelType type = ChannelType::MultiType) {
    if (buffer_size < 64 * 1024 * 1024) {
        buffer_size = 64 * 1024 * 1024; // Enforce 64MB minimum
    }
    return Channel::create("ipc://" + name, buffer_size, mode, type, false, {});
}

/**
 * @brief Create an IPC client (attaches to existing shared memory)
 * @param name IPC channel name (without ipc:// prefix)  
 * @param mode Channel mode (must match server)
 * @param type Channel type (must match server)
 * @return Unique pointer to the channel
 * 
 * @note Clients attach to shared memory created by the server.
 *       The buffer size is determined by the server.
 */
inline std::unique_ptr<Channel>
create_ipc_client(const std::string &name, ChannelMode mode = ChannelMode::SPSC,
                  ChannelType type = ChannelType::MultiType) {
    // Buffer size of 0 indicates client mode - attach to existing shared memory
    return Channel::create("ipc://" + name, 0, mode, type, false, {});
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
using SPSCChannel = Channel; // These are now created via factory
using SPMCChannel = Channel;
using MPSCChannel = Channel;
using MPMCChannel = Channel;

// Message template method implementations (must be after Channel definition)
template <typename Derived>
Message<Derived>::Message(Channel &channel)
    : data_(nullptr), size_(0), channel_(&channel), handle_(nullptr) {
    // Reserve space in the channel's ring buffer
    size_ = Derived::calculate_size();
    offset_ = channel.reserve_write_slot(size_);
    
    if (offset_ == Channel::BUFFER_FULL) {
        throw std::runtime_error("Channel buffer full");
    }
    
    // Get direct pointer to ring buffer memory at offset
    auto span = channel.get_write_span(size_);
    data_ = span.data();
    
    if (!data_) {
        throw std::runtime_error("Failed to get write buffer");
    }
    
    // Message is now a direct view into the ring buffer
    // User can write to it directly via the derived class methods
    // No copying, no allocation - pure zero-copy
}

template <typename Derived>
void Message<Derived>::send() {
    if (!channel_ || offset_ == Channel::BUFFER_FULL) {
        throw std::runtime_error("Cannot send invalid message");
    }
    
    before_send();
    // Notify channel that message is ready (zero-copy, just a notification)
    channel_->notify_message_ready(offset_, Derived::calculate_size());
    
    // Clear our view since the message has been sent
    data_ = nullptr;
    size_ = 0;
    offset_ = 0;
}

// ============================================================================
// Ring Buffer Implementation
// ============================================================================

/**
 * @struct WriteHandle
 * @brief Handle for writing data to ring buffer
 */
struct WriteHandle {
    void *data;
    size_t size;
    class SPSCRingBuffer *ring_buffer; // Forward declaration
    void commit();
};

/**
 * @struct ReadHandle
 * @brief Handle for reading data from ring buffer
 */
struct ReadHandle {
    const void *data;
    size_t size;
    class SPSCRingBuffer *ring_buffer; // Forward declaration

    /**
     * @brief Release the read handle and advance read position
     */
    void release();
};

/**
 * @class SPSCRingBuffer
 * @brief Single Producer Single Consumer ring buffer with circular message
 * storage
 *
 * Enhanced implementation that supports multiple concurrent messages with
 * proper circular buffer semantics and atomic operations for thread safety.
 */
class SPSCRingBuffer {
public:
    explicit SPSCRingBuffer(size_t capacity);

    std::optional<WriteHandle> reserve(size_t size);
    std::optional<ReadHandle> read();


private:
    size_t capacity_;
    std::unique_ptr<uint8_t[]> buffer_;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};

    // Temporary state for reservation
    size_t reserved_size_ = 0;
    size_t reserved_write_pos_ = 0;

    friend void WriteHandle::commit();
    friend void ReadHandle::release();
};

// ============================================================================
// Channel Factory
// ============================================================================

/**
 * @class ChannelFactory
 * @brief Factory for creating different types of channels
 */
class ChannelFactory {
public:
    template <typename RingBufferType>
    static std::unique_ptr<Channel>
    create(const std::string &uri, size_t buffer_size,
           ChannelType type = ChannelType::MultiType) {
        return std::make_unique<Channel>(uri, buffer_size, type);
    }

    static bool is_memory_uri(const std::string &uri) {
        return uri.substr(0, 9) == "memory://";
    }

    static bool is_ipc_uri(const std::string &uri) {
        return uri.substr(0, 6) == "ipc://";
    }

    static bool is_tcp_uri(const std::string &uri) {
        return uri.substr(0, 6) == "tcp://";
    }
};

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

    static size_t calculate_size() {
        return 1024;
    }

    uint8_t &operator[](size_t index);
    const uint8_t &operator[](size_t index) const;
    uint8_t *begin();
    uint8_t *end();
    const uint8_t *begin() const;
    const uint8_t *end() const;
    uint8_t *data();
    const uint8_t *data() const;
    size_t size() const;
    size_t capacity() const;
    void resize(size_t new_size);
    void initialize();

    // Eigen integration helpers
    template <typename T>
    T *as() {
        return reinterpret_cast<T *>(data());
    }

    template <typename T>
    const T *as() const {
        return reinterpret_cast<const T *>(data());
    }
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
    explicit ReliabilityManager(Channel &channel,
                                const ReliabilityConfig &config = {});
    ~ReliabilityManager();

    void start();
    void stop();
    bool is_running() const;

    // Component access
    AcknowledgmentManager *acknowledgment_manager();
    RetryManager *retry_manager();
    HeartbeatManager *heartbeat_manager();
    ReplayBuffer *replay_buffer();

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
    ReliableChannelGuard(Channel &channel,
                         const ReliabilityConfig &config = {});
    ~ReliableChannelGuard();

    ReliableChannelGuard(const ReliableChannelGuard &) = delete;
    ReliableChannelGuard &operator=(const ReliableChannelGuard &) = delete;
    ReliableChannelGuard(ReliableChannelGuard &&) = default;
    ReliableChannelGuard &operator=(ReliableChannelGuard &&) = delete;

    Channel &channel() {
        return channel_;
    }
    ReliabilityManager &manager() {
        return *manager_;
    }

private:
    Channel &channel_;
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
inline std::unique_ptr<Channel>
create_reliable_channel(const std::string &uri,
                        size_t buffer_size = 64 * 1024 * 1024,
                        ChannelMode mode = ChannelMode::SPSC,
                        const ReliabilityConfig &config = {}) {
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
    explicit PerformanceManager(const PerformanceConfig &config = {});
    ~PerformanceManager();

    void apply_optimizations(Channel &channel);
    void apply_optimizations(void *buffer, size_t size);

    BenchmarkResult benchmark_channel(Channel &channel, size_t message_size,
                                      size_t num_messages);

    std::string get_summary() const;
    std::vector<std::string> get_recommendations() const;

    static PerformanceManager &instance();

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
    static void *allocate(size_t size, size_t alignment = 64);
    static void deallocate(void *ptr, size_t size);
};

/**
 * @class OptimizedMessage
 * @brief Wrapper that applies performance optimizations to messages
 * @tparam T The base message type to optimize
 *
 * Automatically applies performance optimizations like prefetching
 * and cache alignment when messages are created.
 */
template <typename T>
class OptimizedMessage : public T {
public:
    using T::T;

    OptimizedMessage(Channel &channel) : T(channel) {
        PerformanceManager::instance().apply_optimizations(this->data(),
                                                           this->size());
    }
};

// Type aliases
template <typename T>
using OptimizedVector = OptimizedMessage<FloatVector>;

template <typename T>
using CacheAlignedVector = OptimizedMessage<FloatVector>;

/**
 * @brief Get the global performance manager instance
 * @return Reference to the singleton PerformanceManager
 */
inline PerformanceManager &get_performance_manager() {
    return PerformanceManager::instance();
}

/**
 * @brief Enable global performance optimizations
 * @param config Performance configuration options
 *
 * Call this early in your application to enable performance
 * features for all subsequently created channels and messages.
 */
inline void
enable_performance_optimizations(const PerformanceConfig &config = {}) {
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
inline BenchmarkResult run_performance_benchmarks(Channel &channel,
                                                  size_t message_size = 1024,
                                                  size_t num_messages = 10000) {
    return PerformanceManager::instance().benchmark_channel(
        channel, message_size, num_messages);
}

/**
 * @brief Create a message with performance optimizations applied
 * @tparam T Message type to create
 * @param channel Channel to allocate the message in
 * @return Optimized message instance
 */
template <typename T>
inline OptimizedMessage<T> create_optimized_message(Channel &channel) {
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
debug::ChannelDiagnostics inspect_channel(const Channel &channel);

/**
 * @brief Get a human-readable status report for a channel
 * @param channel The channel to inspect
 * @return Formatted status report
 */
std::string get_channel_status(const Channel &channel);

/**
 * @brief Visualize channel buffer usage
 * @param channel The channel to visualize
 * @param width Width of the visualization bar (default: 50)
 * @return ASCII art buffer visualization
 */
std::string visualize_channel_buffer(const Channel &channel, size_t width = 50);

/**
 * @brief Run health check on a channel
 * @param channel The channel to check
 * @return List of health issues found
 */
std::vector<std::string> check_channel_health(const Channel &channel);

/**
 * @brief Start performance profiling for a channel
 * @param channel The channel to profile
 * @return Performance profiler instance
 */
std::unique_ptr<debug::PerformanceProfiler>
start_profiling(const Channel &channel);

/**
 * @brief Enable message tracing for a channel
 * @param channel_uri URI of the channel to trace
 */
void enable_message_tracing(const std::string &channel_uri);

/**
 * @brief Get message trace report for a channel
 * @param channel_uri URI of the channel
 * @return Formatted trace report
 */
std::string get_trace_report(const std::string &channel_uri);

/**
 * @brief Create a debugging dashboard showing all channels
 * @return ASCII art dashboard
 */
std::string create_debug_dashboard();

// ============================================================================
// WebRTC API
// ============================================================================

namespace webrtc {

/**
 * @struct WebRTCConfig
 * @brief Configuration for WebRTC connections
 */
struct WebRTCConfig {
    std::vector<std::string> stun_servers = {"stun.l.google.com:19302",
                                             "stun1.l.google.com:19302"};
    std::vector<std::string> turn_servers;
    std::string data_channel_label = "psyne-channel";
    bool ordered = true;
    std::chrono::milliseconds ice_gathering_timeout{5000};
    std::chrono::milliseconds connection_timeout{30000};
};

/**
 * @brief Create a WebRTC channel for peer-to-peer communication
 * @param peer_id Target peer identifier
 * @param buffer_size Buffer size for messages (default: 64MB)
 * @param signaling_server_uri WebSocket signaling server URI
 * @param config WebRTC configuration
 * @return Unique pointer to the channel
 *
 * WebRTC channels enable direct peer-to-peer communication with NAT traversal
 * support. Perfect for real-time gaming, video calls, and low-latency
 * messaging.
 *
 * @example
 * @code
 * // Create WebRTC channel to peer "player2"
 * auto channel = psyne::webrtc::create_channel(
 *     "player2",
 *     64*1024*1024,
 *     "ws://signaling.example.com:8080"
 * );
 *
 * // Send game state update
 * psyne::FloatVector position(*channel);
 * position.resize(3);
 * position[0] = x; position[1] = y; position[2] = z;
 * position.send();
 *
 * // Receive opponent's position
 * auto opponent_pos = channel->receive<psyne::FloatVector>();
 * @endcode
 */
std::unique_ptr<Channel>
create_channel(const std::string &peer_id, size_t buffer_size = 64 * 1024 * 1024,
               const std::string &signaling_server_uri = "ws://localhost:8080",
               const WebRTCConfig &config = {});

/**
 * @brief Create a WebRTC gaming channel optimized for real-time games
 * @param peer_id Target peer identifier
 * @param game_room_id Game room or session identifier
 * @param signaling_server_uri WebSocket signaling server URI
 * @return Optimized channel for gaming applications
 *
 * Gaming channels are pre-configured with:
 * - Low-latency settings
 * - Unordered delivery for position updates
 * - Automatic NAT traversal
 * - Built-in latency monitoring
 */
std::unique_ptr<Channel> create_gaming_channel(
    const std::string &peer_id, const std::string &game_room_id,
    const std::string &signaling_server_uri = "ws://localhost:8080");

} // namespace webrtc

// ============================================================================
// UDP Multicast API
// ============================================================================

namespace multicast {

/**
 * @enum Role
 * @brief Role of the multicast endpoint
 */
enum class Role {
    Publisher, ///< Sends multicast messages
    Subscriber ///< Receives multicast messages
};

/**
 * @brief Create a UDP multicast publisher
 * @param multicast_address Multicast group address (e.g., "239.255.0.1")
 * @param port Port number
 * @param buffer_size Buffer size for messages (default: 64MB)
 * @param compression_config Optional compression configuration
 * @return Unique pointer to the channel
 */
std::unique_ptr<Channel>
create_publisher(const std::string &multicast_address, uint16_t port,
                 size_t buffer_size = 64 * 1024 * 1024,
                 const compression::CompressionConfig &compression_config = {});

/**
 * @brief Create a UDP multicast subscriber
 * @param multicast_address Multicast group address (e.g., "239.255.0.1")
 * @param port Port number
 * @param buffer_size Buffer size for messages (default: 64MB)
 * @param interface_address Local interface to bind to (optional)
 * @return Unique pointer to the channel
 */
std::unique_ptr<Channel>
create_subscriber(const std::string &multicast_address, uint16_t port,
                  size_t buffer_size = 64 * 1024 * 1024,
                  const std::string &interface_address = "");

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
    const std::string &multicast_address, uint16_t port, Role role,
    size_t buffer_size = 64 * 1024 * 1024,
    const compression::CompressionConfig &compression_config = {},
    const std::string &interface_address = "");

} // namespace multicast

// RDMA/InfiniBand support removed - no hardware available for testing

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

    Header &header() {
        return *reinterpret_cast<Header *>(Message<ComplexVectorF>::data());
    }
    const Header &header() const {
        return *reinterpret_cast<const Header *>(
            Message<ComplexVectorF>::data());
    }

    ComplexType *data() {
        return reinterpret_cast<ComplexType *>(
            reinterpret_cast<uint8_t *>(Message<ComplexVectorF>::data()) +
            sizeof(Header));
    }

    const ComplexType *data() const {
        return reinterpret_cast<const ComplexType *>(
            reinterpret_cast<const uint8_t *>(Message<ComplexVectorF>::data()) +
            sizeof(Header));
    }

    size_t size() const {
        return header().size;
    }

    void resize(size_t new_size) {
        header().size = static_cast<uint32_t>(new_size);
    }

    ComplexType &operator[](size_t index) {
        return data()[index];
    }

    const ComplexType &operator[](size_t index) const {
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

    enum class Layout { NCHW, NHWC, CHW, HWC, Custom };

    using Message<MLTensorF>::Message;

    static size_t calculate_size() {
        return 64 * 1024 * 1024; // 64MB default
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

    Header &header() {
        return *reinterpret_cast<Header *>(Message<MLTensorF>::data());
    }
    const Header &header() const {
        return *reinterpret_cast<const Header *>(Message<MLTensorF>::data());
    }

    float *data() {
        return reinterpret_cast<float *>(
            reinterpret_cast<uint8_t *>(Message<MLTensorF>::data()) +
            sizeof(Header));
    }

    const float *data() const {
        return reinterpret_cast<const float *>(
            reinterpret_cast<const uint8_t *>(Message<MLTensorF>::data()) +
            sizeof(Header));
    }

    void set_shape(const std::vector<uint32_t> &new_shape,
                   Layout layout = Layout::Custom) {
        header().num_dims = static_cast<uint32_t>(new_shape.size());
        header().layout = layout;

        // Zero-copy: manually copy shape data instead of using std::copy
        for (size_t i = 0; i < new_shape.size() && i < 4; ++i) {
            header().shape[i] = new_shape[i];
        }

        uint32_t total = 1;
        for (uint32_t dim : new_shape) {
            total *= dim;
        }
        header().total_elements = total;
    }

    std::vector<uint32_t> shape() const {
        return std::vector<uint32_t>(header().shape,
                                     header().shape + header().num_dims);
    }

    uint32_t total_elements() const {
        return header().total_elements;
    }
    Layout layout() const {
        return header().layout;
    }
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
        return 64 * 1024 * 1024; // 64MB default
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

    Header &header() {
        return *reinterpret_cast<Header *>(Message<SparseMatrixF>::data());
    }
    const Header &header() const {
        return *reinterpret_cast<const Header *>(
            Message<SparseMatrixF>::data());
    }

    float *values() {
        return reinterpret_cast<float *>(
            reinterpret_cast<uint8_t *>(Message<SparseMatrixF>::data()) +
            sizeof(Header));
    }

    const float *values() const {
        return reinterpret_cast<const float *>(
            reinterpret_cast<const uint8_t *>(Message<SparseMatrixF>::data()) +
            sizeof(Header));
    }

    uint32_t *column_indices() {
        return reinterpret_cast<uint32_t *>(values() + header().nnz);
    }

    const uint32_t *column_indices() const {
        return reinterpret_cast<const uint32_t *>(values() + header().nnz);
    }

    uint32_t *row_pointers() {
        return column_indices() + header().nnz;
    }

    const uint32_t *row_pointers() const {
        return column_indices() + header().nnz;
    }

    uint32_t rows() const {
        return header().rows;
    }
    uint32_t cols() const {
        return header().cols;
    }
    uint32_t nnz() const {
        return header().nnz;
    }

    void set_structure(uint32_t rows, uint32_t cols, uint32_t nnz) {
        header().rows = rows;
        header().cols = cols;
        header().nnz = nnz;
        std::fill(row_pointers(), row_pointers() + rows + 1, 0);
    }

    void multiply_vector(const float *x, float *y) const {
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
template <typename Derived = Eigen::Dynamic>
auto as_eigen_vector(FloatVector &vec) {
    return Eigen::Map<Eigen::VectorXf>(vec.begin(), vec.size());
}

template <typename Derived = Eigen::Dynamic>
auto as_eigen_vector(const FloatVector &vec) {
    return Eigen::Map<const Eigen::VectorXf>(vec.begin(), vec.size());
}

template <typename Derived = Eigen::Dynamic>
auto as_eigen_matrix(FloatVector &vec, size_t rows, size_t cols) {
    return Eigen::Map<Eigen::MatrixXf>(vec.begin(), rows, cols);
}

template <typename Derived = Eigen::Dynamic>
auto as_eigen_matrix(const FloatVector &vec, size_t rows, size_t cols) {
    return Eigen::Map<const Eigen::MatrixXf>(vec.begin(), rows, cols);
}

/**
 * @brief Create an Eigen::Map view of DoubleMatrix data
 */
auto as_eigen_matrix(DoubleMatrix &mat) {
    return Eigen::Map<Eigen::MatrixXd>(reinterpret_cast<double *>(mat.data()),
                                       mat.rows(), mat.cols());
}

auto as_eigen_matrix(const DoubleMatrix &mat) {
    return Eigen::Map<const Eigen::MatrixXd>(
        reinterpret_cast<const double *>(mat.data()), mat.rows(), mat.cols());
}

/**
 * @brief Create an Eigen view of ByteVector as any numeric type
 */
template <typename T>
auto as_eigen_vector(ByteVector &vec) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        vec.as<T>(), vec.size() / sizeof(T));
}

template <typename T>
auto as_eigen_matrix(ByteVector &vec, size_t rows, size_t cols) {
    return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        vec.as<T>(), rows, cols);
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
 * applications. It provides efficient inter-process and inter-thread
 * communication with minimal overhead.
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

// ============================================================================
// GPU Support
// ============================================================================

#ifdef PSYNE_GPU_ENABLED

namespace gpu {

/**
 * @brief Supported GPU backends
 */
enum class GPUBackend {
    None,  ///< No GPU support
    Metal, ///< Apple Metal
    CUDA,  ///< NVIDIA CUDA
    Vulkan ///< Vulkan
};

/**
 * @brief GPU buffer usage patterns
 */
enum class BufferUsage {
    Static,  ///< Buffer content rarely changes
    Dynamic, ///< Buffer content changes frequently
    Stream   ///< Buffer content changes every frame
};

/**
 * @brief Memory access modes
 */
enum class MemoryAccess {
    DeviceOnly, ///< GPU access only
    HostOnly,   ///< CPU access only
    Shared,     ///< Both CPU and GPU access (unified memory)
    Managed     ///< Automatic migration between CPU/GPU
};

// Forward declarations
class GPUBuffer;
class GPUBufferFactory;
class GPUContext;

/**
 * @brief Create a GPU context for the specified backend
 * @param backend The GPU backend to use (or None for auto-detection)
 * @return GPU context or nullptr if no suitable GPU found
 */
std::unique_ptr<GPUContext>
create_gpu_context(GPUBackend backend = GPUBackend::None);

/**
 * @brief Detect available GPU backends
 * @return Vector of available backends
 */
std::vector<GPUBackend> detect_gpu_backends();

/**
 * @brief Get the name of a GPU backend
 */
const char *gpu_backend_name(GPUBackend backend);

// GPU-aware vector types (forward declarations)
template <typename T>
class GPUVector;
using GPUFloatVector = GPUVector<float>;
using GPUDoubleVector = GPUVector<double>;
using GPUIntVector = GPUVector<int32_t>;

} // namespace gpu

#endif // PSYNE_GPU_ENABLED

// ============================================================================
// SIMD Support
// ============================================================================

namespace psyne {
namespace simd {

/**
 * @brief SIMD-accelerated tensor operations
 *
 * Provides hardware-accelerated implementations for common tensor operations
 * using AVX-512/AVX2 (x86) and NEON (ARM) intrinsics.
 */
template <typename T>
class TensorOps;

/**
 * @brief Layout transformation utilities for AI/ML tensors
 *
 * Optimized transformations between memory layouts (NCHW <-> NHWC)
 */
class LayoutTransform;

} // namespace simd

// ============================================================================
// Transport Support (RUDP)
// ============================================================================

namespace transport {

/**
 * @brief RUDP packet types
 */
enum class RUDPPacketType : uint8_t {
    DATA = 0,      ///< Data packet
    ACK = 1,       ///< Acknowledgment packet
    NACK = 2,      ///< Negative acknowledgment
    SYN = 3,       ///< Synchronization (connection establishment)
    FIN = 4,       ///< Finish (connection termination)
    HEARTBEAT = 5, ///< Keep-alive heartbeat
    RESET = 6      ///< Reset connection
};

/**
 * @brief RUDP connection statistics
 */
struct RUDPStats {
    uint64_t packets_sent = 0;
    uint64_t packets_received = 0;
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;
    uint64_t packets_retransmitted = 0;
    uint64_t packets_out_of_order = 0;
    uint64_t packets_duplicate = 0;
    uint64_t acks_sent = 0;
    uint64_t acks_received = 0;
    double rtt_ms = 0.0;
    double rtt_variance_ms = 0.0;
    uint32_t cwnd = 1;         ///< Congestion window
    uint32_t ssthresh = 65535; ///< Slow start threshold
};

/**
 * @brief RUDP configuration
 */
struct RUDPConfig {
    uint32_t max_window_size = 8192;        ///< Maximum receive window
    uint32_t initial_timeout_ms = 1000;     ///< Initial retransmission timeout
    uint32_t max_retransmits = 5;           ///< Maximum retransmission attempts
    uint32_t heartbeat_interval_ms = 5000;  ///< Heartbeat interval
    uint32_t connection_timeout_ms = 30000; ///< Connection timeout
    bool enable_fast_retransmit = true;     ///< Enable fast retransmit on 3 duplicate ACKs
    bool enable_selective_ack = true;       ///< Enable selective acknowledgments
    bool enable_nagle = false;              ///< Enable Nagle's algorithm
};

/**
 * @brief Connection state for RUDP
 */
enum class RUDPConnectionState {
    CLOSED,
    LISTEN,
    SYN_SENT,
    SYN_RECEIVED,
    ESTABLISHED,
    FIN_WAIT,
    CLOSE_WAIT,
    CLOSING,
    TIME_WAIT
};

// Forward declarations for RUDP classes
class RUDPChannel;
class RUDPServer;

/**
 * @brief Create RUDP client channel
 */
std::unique_ptr<RUDPChannel>
create_rudp_client(const std::string &remote_address, uint16_t remote_port,
                   const RUDPConfig &config = {});

/**
 * @brief Create RUDP server
 */
std::unique_ptr<RUDPServer> create_rudp_server(uint16_t port,
                                               const RUDPConfig &config = {});

} // namespace transport

// ============================================================================
// Async Support
// ============================================================================

#ifdef PSYNE_ASYNC_SUPPORT
namespace async {

// Forward declarations for async classes
template <typename ChannelType>
class AsyncChannel;

/**
 * @brief Thread pool for async handlers
 */
class PsynePool {
public:
    PsynePool(int min_threads, int max_threads);
    ~PsynePool();
    
    // Add methods as needed
    void submit_task(std::function<void()> task);
    void shutdown();
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Configuration for async message handlers
 */
struct AsyncHandlerConfig {
    int max_concurrent_handlers = 4;
    bool use_thread_pool = true;
    PsynePool* thread_pool = nullptr;
};

} // namespace async
#endif // PSYNE_ASYNC_SUPPORT

// ============================================================================
// Collective Operations
// ============================================================================

namespace collective {

/**
 * @brief Reduce operations for collective communication
 */
enum class ReduceOp {
    Sum,
    Min,
    Max,
    Avg
};

// Forward declarations
class CollectiveGroup;

template <typename T>
class Broadcast;

template <typename T>
class AllReduce;

template <typename T>
class Scatter;

template <typename T>
class Gather;

template <typename T>
class AllGather;

/**
 * @brief Create a collective communication group
 */
std::shared_ptr<CollectiveGroup> 
create_collective_group(int rank, const std::vector<std::string>& peer_uris, 
                       const std::string& topology = "ring");

} // namespace collective

// ============================================================================
// Memory Management
// ============================================================================

namespace memory {

/**
 * @brief Allocation flags for custom allocator
 */
enum class AllocFlags : uint32_t {
    None = 0,
    Zeroed = 1 << 0,       ///< Zero-initialize memory
    HugePage = 1 << 1,     ///< Use huge pages if available
    Aligned64 = 1 << 2,    ///< Align to 64-byte boundary
    Aligned128 = 1 << 3,   ///< Align to 128-byte boundary
    Aligned256 = 1 << 4,   ///< Align to 256-byte boundary
    NUMA = 1 << 5          ///< NUMA-aware allocation
};

// Allow bitwise operations on flags
inline AllocFlags operator|(AllocFlags a, AllocFlags b) {
    return static_cast<AllocFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

/**
 * @brief Block information for allocated memory
 */
struct BlockInfo {
    size_t size;
    bool is_huge_page;
    int numa_node;
    void* base_address;
};

/**
 * @brief Allocation statistics
 */
struct AllocatorStats {
    uint64_t total_allocated = 0;
    uint64_t total_freed = 0;
    uint64_t current_usage = 0;
    uint64_t peak_usage = 0;
    uint64_t allocation_count = 0;
    uint64_t free_count = 0;
    uint64_t huge_page_count = 0;
};

/**
 * @brief Custom memory allocator singleton
 */
class CustomAllocator {
public:
    static CustomAllocator& instance();
    
    void* allocate(size_t size, AllocFlags flags = AllocFlags::None);
    void deallocate(void* ptr);
    
    bool huge_pages_available() const;
    size_t get_huge_page_size() const;
    int get_numa_nodes() const;
    
    AllocatorStats stats() const;
    std::optional<BlockInfo> get_block_info(void* ptr) const;
    
private:
    CustomAllocator() = default;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief STL-compatible allocator using CustomAllocator
 */
template <typename T>
class StlCustomAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = StlCustomAllocator<U>;
    };

    StlCustomAllocator() = default;
    template <typename U>
    StlCustomAllocator(const StlCustomAllocator<U>&) {}

    pointer allocate(size_type n) {
        return static_cast<pointer>(
            CustomAllocator::instance().allocate(n * sizeof(T)));
    }

    void deallocate(pointer p, size_type) {
        CustomAllocator::instance().deallocate(p);
    }

    template <typename U>
    bool operator==(const StlCustomAllocator<U>&) const { return true; }
    
    template <typename U>
    bool operator!=(const StlCustomAllocator<U>&) const { return false; }
};

/**
 * @brief RAII wrapper for custom allocated memory
 */
class UniqueAlloc {
public:
    UniqueAlloc(size_t size, AllocFlags flags = AllocFlags::None);
    ~UniqueAlloc();
    
    // Move-only semantics
    UniqueAlloc(UniqueAlloc&& other) noexcept;
    UniqueAlloc& operator=(UniqueAlloc&& other) noexcept;
    UniqueAlloc(const UniqueAlloc&) = delete;
    UniqueAlloc& operator=(const UniqueAlloc&) = delete;
    
    void* get() const { return ptr_; }
    size_t size() const { return size_; }
    bool is_valid() const { return ptr_ != nullptr; }
    
    void reset();
    
private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

/**
 * @brief Allocate memory optimized for tensor operations
 */
void* allocate_tensor(size_t size);

/**
 * @brief Deallocate tensor memory
 */
void deallocate_tensor(void* ptr);

} // namespace memory

} // namespace psyne

// Include logging utilities
// #include "logging.hpp"  // TODO: Fix logging include path

// Include SIMD operations implementation
#include "../../src/simd/simd_ops.hpp"

// Include GPU support if enabled
#if defined(PSYNE_GPU_SUPPORT) || defined(PSYNE_CUDA_SUPPORT) || defined(PSYNE_METAL_SUPPORT) || defined(PSYNE_VULKAN_SUPPORT)
#include "../../src/gpu/gpu_buffer.hpp"
#include "../../src/gpu/gpu_message.hpp"
#endif
