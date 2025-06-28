#pragma once

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

// Version information
#define PSYNE_VERSION_MAJOR 0
#define PSYNE_VERSION_MINOR 1
#define PSYNE_VERSION_PATCH 0

namespace psyne {

// Version string
constexpr const char* version() {
    return "0.1.0";
}

// ============================================================================
// Core Types
// ============================================================================

// Channel synchronization modes
enum class ChannelMode {
    SPSC,  // Single Producer, Single Consumer
    SPMC,  // Single Producer, Multiple Consumer
    MPSC,  // Multiple Producer, Single Consumer
    MPMC   // Multiple Producer, Multiple Consumer
};

// Channel type modes
enum class ChannelType {
    SingleType,   // Optimized for single message type (no metadata)
    MultiType     // Supports multiple types (small overhead)
};

// Forward declarations
class Channel;
template<typename Derived> class Message;

// Implementation details - defined in src
namespace detail {
    class ChannelImpl;
}

// ============================================================================
// Message API
// ============================================================================

// Base message class - provides zero-copy view semantics
template<typename Derived>
class Message {
public:
    // Create outgoing message (allocates in channel)
    explicit Message(Channel& channel);
    
    // Create incoming message (view of existing data)
    explicit Message(const void* data, size_t size);
    
    // Move-only semantics
    Message(Message&& other) noexcept;
    Message& operator=(Message&& other) noexcept;
    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;
    
    virtual ~Message();
    
    // Send the message
    void send();
    
    // Check validity
    bool is_valid() const { return data_ != nullptr; }
    
    // Get message type ID
    static constexpr uint32_t type() { return Derived::message_type; }
    
protected:
    // Access to raw data for derived classes
    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    
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

// Dynamic-size array of floats
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

// 2D matrix of doubles
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

// Public Channel API
class Channel {
public:
    // Factory method to create channels
    static std::unique_ptr<Channel> create(
        const std::string& uri,
        size_t buffer_size,
        ChannelMode mode = ChannelMode::SPSC,
        ChannelType type = ChannelType::MultiType
    );
    
    virtual ~Channel() = default;
    
    // Send a message
    template<typename MessageType>
    void send(MessageType& msg) {
        msg.send();
    }
    
    // Receive methods
    template<typename MessageType>
    std::optional<MessageType> receive(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()
    );
    
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

// Factory function for creating channels from URIs
// Supported URI schemes:
//   - memory://name     : In-process memory channels
//   - ipc://name        : Inter-process communication via shared memory
//   - tcp://host:port   : TCP client (connects to host:port)
//   - tcp://:port       : TCP server (listens on port)
inline std::unique_ptr<Channel> create_channel(
    const std::string& uri,
    size_t buffer_size = 1024 * 1024,
    ChannelMode mode = ChannelMode::SPSC,
    ChannelType type = ChannelType::MultiType
) {
    return Channel::create(uri, buffer_size, mode, type);
}

// ============================================================================
// Convenience Type Aliases
// ============================================================================

using ChannelPtr = std::unique_ptr<Channel>;

// Pre-defined channel types (for backward compatibility)
using SPSCChannel = Channel;  // These are now created via factory
using SPMCChannel = Channel;
using MPSCChannel = Channel;
using MPMCChannel = Channel;

} // namespace psyne

// Enhanced Message Types
#include "types/fixed_matrices.hpp"
#include "types/quantized_vectors.hpp"

// GPU Support (optional - requires Metal/Vulkan/CUDA)
#ifdef PSYNE_GPU_SUPPORT
#include "gpu/gpu_buffer.hpp"
#include "gpu/gpu_message.hpp"
#endif

