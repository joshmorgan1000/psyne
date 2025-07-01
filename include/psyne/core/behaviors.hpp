#pragma once

#include "backpressure.hpp"
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

/**
 * @file behaviors.hpp
 * @brief Clear behavior definitions for Psyne architecture
 *
 * BEHAVIORS DEFINED:
 * - Substrate = Owns memory + transport protocol
 * - Message = Lens into substrate memory
 * - Channel = Bridge between substrate, message lens, and pattern
 * - Pattern = Producer/consumer coordination
 *
 * NO CIRCULAR DEPENDENCIES!
 */

namespace psyne::behaviors {

/**
 * @brief SUBSTRATE BEHAVIOR:
 * - Owns the memory slab
 * - Owns the transport protocol
 * - Provides memory to messages
 * - Handles sending data over transport
 */
class SubstrateBehavior {
public:
    virtual ~SubstrateBehavior() = default;

    // MEMORY OWNERSHIP
    virtual void *allocate_memory_slab(size_t size_bytes) = 0;
    virtual void deallocate_memory_slab(void *memory) = 0;

    // TRANSPORT PROTOCOL
    virtual void transport_send(void *data, size_t size) = 0;
    virtual void transport_receive(void *buffer, size_t buffer_size) = 0;

    // SUBSTRATE IDENTITY
    virtual const char *substrate_name() const = 0;
    virtual bool is_zero_copy() const = 0;
    virtual bool is_cross_process() const = 0;
};

/**
 * @brief MESSAGE BEHAVIOR:
 * - Lens into substrate memory
 * - Provides typed access to raw memory
 * - Does NOT own memory (substrate does!)
 * - Does NOT handle transport (substrate does!)
 */
enum class LensMode { Construct, ViewOnly };

template <typename T>
class MessageLens {
public:
    // Message is a LENS - takes a view into substrate memory
    explicit MessageLens(void *substrate_memory,
                         LensMode mode = LensMode::Construct)
        : memory_view_(static_cast<T *>(substrate_memory)) {
        if (mode == LensMode::Construct) {
            // Construct object in substrate's memory
            new (memory_view_) T{};
        }
        // If ViewOnly, just provide access to existing object
    }

    template <typename... Args>
    MessageLens(void *substrate_memory, Args &&...args)
        : memory_view_(static_cast<T *>(substrate_memory)) {
        // Construct with args in substrate's memory
        new (memory_view_) T(std::forward<Args>(args)...);
    }

    // LENS ACCESS - provides typed view of substrate memory
    T *operator->() {
        return memory_view_;
    }
    const T *operator->() const {
        return memory_view_;
    }
    T &operator*() {
        return *memory_view_;
    }
    const T &operator*() const {
        return *memory_view_;
    }

    // Raw memory access (for substrate transport)
    void *raw_memory() const {
        return memory_view_;
    }
    size_t size() const {
        return sizeof(T);
    }

private:
    T *memory_view_; // LENS into substrate memory (not owned!)
};

/**
 * @brief PATTERN BEHAVIOR:
 * - Coordinates producers and consumers
 * - Manages allocation order/timing
 * - Handles synchronization
 * - Does NOT own memory (substrate does!)
 * - Does NOT handle transport (substrate does!)
 */
class PatternBehavior {
public:
    virtual ~PatternBehavior() = default;

    // PRODUCER/CONSUMER COORDINATION
    virtual void *coordinate_allocation(void *slab_memory,
                                        size_t message_size) = 0;
    virtual void *coordinate_receive() = 0;

    // SYNCHRONIZATION
    virtual void producer_sync() = 0;
    virtual void consumer_sync() = 0;

    // PATTERN IDENTITY
    virtual const char *pattern_name() const = 0;
    virtual bool needs_locks() const = 0;
    virtual size_t max_producers() const = 0;
    virtual size_t max_consumers() const = 0;
};

/**
 * @brief CHANNEL BEHAVIOR:
 * - Bridge between substrate, message lens, and pattern
 * - Orchestrates the three components
 * - Does NOT own memory (delegates to substrate)
 * - Does NOT handle transport (delegates to substrate)
 * - Does NOT manage sync (delegates to pattern)
 */
template <typename MessageType, typename SubstrateType, typename PatternType>
class ChannelBridge {
public:
    explicit ChannelBridge(
        size_t slab_size = 32 * 1024 * 1024,
        std::unique_ptr<backpressure::PolicyBase> bp_policy = nullptr)
        : backpressure_policy_(std::move(bp_policy)) {
        // Initialize substrate (memory + transport owner)
        substrate_ = std::make_unique<SubstrateType>();

        // Substrate allocates and owns the memory
        slab_memory_ = substrate_->allocate_memory_slab(slab_size);
        if (!slab_memory_) {
            throw std::bad_alloc();
        }

        // Initialize pattern (coordination)
        pattern_ = std::make_unique<PatternType>();

        max_messages_ = slab_size / sizeof(MessageType);
    }

    ~ChannelBridge() {
        if (slab_memory_) {
            // Substrate owns memory, so substrate deallocates
            substrate_->deallocate_memory_slab(slab_memory_);
        }
    }

    // BRIDGE BEHAVIOR: Create message lens into substrate memory
    MessageLens<MessageType> create_message() {
        // Pattern coordinates allocation in substrate memory
        void *memory =
            pattern_->coordinate_allocation(slab_memory_, sizeof(MessageType));

        // Handle backpressure if allocation failed
        if (!memory && backpressure_policy_) {
            auto retry_fn = [this]() -> void * {
                return pattern_->coordinate_allocation(slab_memory_,
                                                       sizeof(MessageType));
            };
            memory = backpressure_policy_->handle_full(retry_fn);
        }

        if (!memory) {
            throw std::runtime_error(
                "Pattern allocation failed - channel full");
        }

        // Return lens into substrate memory
        return MessageLens<MessageType>(memory);
    }

    template <typename... Args>
    MessageLens<MessageType> create_message(Args &&...args) {
        void *memory =
            pattern_->coordinate_allocation(slab_memory_, sizeof(MessageType));

        // Handle backpressure if allocation failed
        if (!memory && backpressure_policy_) {
            auto retry_fn = [this]() -> void * {
                return pattern_->coordinate_allocation(slab_memory_,
                                                       sizeof(MessageType));
            };
            memory = backpressure_policy_->handle_full(retry_fn);
        }

        if (!memory) {
            throw std::runtime_error(
                "Pattern allocation failed - channel full");
        }

        return MessageLens<MessageType>(memory, std::forward<Args>(args)...);
    }

    // BRIDGE BEHAVIOR: Send via substrate transport
    void send_message(const MessageLens<MessageType> &message) {
        // Substrate handles transport
        substrate_->transport_send(message.raw_memory(), message.size());
    }

    // BRIDGE BEHAVIOR: Receive via pattern coordination
    std::optional<MessageLens<MessageType>> try_receive() {
        // Pattern coordinates receive
        void *memory = pattern_->coordinate_receive();
        if (!memory) {
            return std::nullopt;
        }

        return MessageLens<MessageType>(memory, LensMode::ViewOnly);
    }

    // BRIDGE INFO: Expose component capabilities
    const char *substrate_name() const {
        return substrate_->substrate_name();
    }
    const char *pattern_name() const {
        return pattern_->pattern_name();
    }
    bool is_zero_copy() const {
        return substrate_->is_zero_copy();
    }
    bool needs_locks() const {
        return pattern_->needs_locks();
    }

    // Backpressure management
    void
    set_backpressure_policy(std::unique_ptr<backpressure::PolicyBase> policy) {
        backpressure_policy_ = std::move(policy);
    }

    backpressure::PolicyBase *get_backpressure_policy() {
        return backpressure_policy_.get();
    }

private:
    std::unique_ptr<SubstrateType> substrate_; // Owns memory + transport
    std::unique_ptr<PatternType> pattern_;     // Coordinates access
    void *slab_memory_ = nullptr;              // Owned by substrate
    size_t max_messages_ = 0;
    std::unique_ptr<backpressure::PolicyBase>
        backpressure_policy_; // Optional backpressure handling
};

} // namespace psyne::behaviors

/**
 * @brief BEHAVIOR SUMMARY:
 *
 * Substrate = Memory Owner + Transport Owner
 * ├── allocate_memory_slab()
 * ├── deallocate_memory_slab()
 * ├── transport_send()
 * └── transport_receive()
 *
 * Message = Lens into Substrate Memory
 * ├── MessageLens(substrate_memory)
 * ├── operator->() typed access
 * └── raw_memory() for transport
 *
 * Pattern = Coordination + Synchronization
 * ├── coordinate_allocation()
 * ├── coordinate_receive()
 * ├── producer_sync()
 * └── consumer_sync()
 *
 * Channel = Bridge Between All Three
 * ├── create_message() → Pattern coordinates + Message lens
 * ├── send_message() → Substrate transport
 * └── try_receive() → Pattern coordinates + Message lens
 *
 * NO CIRCULAR DEPENDENCIES!
 * CLEAN SEPARATION OF CONCERNS!
 * INFINITE EXTENSIBILITY!
 */