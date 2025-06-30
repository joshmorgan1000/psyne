#pragma once

/**
 * @file message.hpp
 * @brief Core message interface for enhanced types
 *
 * This header provides the minimal interface needed for enhanced message types
 * to inherit from the Message base class without circular dependencies.
 *
 * Uses modern C++20 features:
 * - Concepts for type safety
 * - std::span for zero-copy data views
 * - constexpr/consteval for compile-time optimization
 */

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace psyne {

// Forward declarations
class Channel;
class RingBuffer;

// C++20 Concepts for type safety (following CORE_DESIGN.md line 11)

/**
 * @concept MessageType
 * @brief Concept defining requirements for zero-copy message types
 */
template <typename T>
concept MessageType = requires {
    // Must have static size calculation
    { T::calculate_size() } -> std::convertible_to<size_t>;
    // Must be default constructible with channel
    requires std::is_constructible_v<T, Channel &>;
    // Must be move-only (no copies for zero-copy)
    requires std::is_move_constructible_v<T>;
    requires !std::is_copy_constructible_v<T>;
};

/**
 * @concept FixedSizeMessage
 * @brief Concept for compile-time sized messages (maximum performance)
 */
template <typename T>
concept FixedSizeMessage = MessageType<T> && requires {
    // Size must be compile-time constant
    requires std::same_as<decltype(T::calculate_size()), const size_t>;
    // Size must be > 0
    requires(T::calculate_size() > 0);
};

/**
 * @concept DynamicSizeMessage
 * @brief Concept for runtime-sized messages (flexibility with performance cost)
 */
template <typename T>
concept DynamicSizeMessage = MessageType<T> && (!FixedSizeMessage<T>);

/**
 * @brief Base class for all zero-copy messages
 * @tparam Derived The derived message type (CRTP pattern)
 *
 * Messages are typed views over ring buffer memory. They do not own data,
 * but provide typed access to pre-allocated ring buffer slots.
 */
template <typename Derived>
class Message {
public:
    /**
     * @brief Create an outgoing message (reserves space in ring buffer)
     * @param channel The channel to reserve space in
     */
    explicit Message(Channel &channel);

    /**
     * @brief Create an incoming message (view of existing ring buffer data)
     * @param slab Pointer to ring buffer
     * @param offset Offset within ring buffer where message data starts
     */
    Message(RingBuffer *slab, uint32_t offset);

    /**
     * @brief Move constructor
     */
    Message(Message &&other) noexcept;

    /**
     * @brief Move assignment operator
     */
    Message &operator=(Message &&other) noexcept;

    /**
     * @brief Destructor
     */
    ~Message();

    // Disable copy construction and assignment
    Message(const Message &) = delete;
    Message &operator=(const Message &) = delete;

    /**
     * @brief Send the message (notify receiver of ready message)
     *
     * Data is already written directly to ring buffer by user.
     * This just sends a notification that message is ready at offset.
     */
    void send();

    /**
     * @brief Get the raw data pointer into ring buffer
     */
    uint8_t *data() noexcept {
        return slab_ ? slab_->base_ptr() + offset_ : nullptr;
    }
    const uint8_t *data() const noexcept {
        return slab_ ? slab_->base_ptr() + offset_ : nullptr;
    }

    /**
     * @brief Get zero-copy data view as std::span (C++20 feature)
     * @return Span over message data for zero-copy access
     */
    std::span<uint8_t> data_span() noexcept {
        auto *ptr = data();
        return ptr ? std::span<uint8_t>{ptr, size()} : std::span<uint8_t>{};
    }
    std::span<const uint8_t> data_span() const noexcept {
        auto *ptr = data();
        return ptr ? std::span<const uint8_t>{ptr, size()}
                   : std::span<const uint8_t>{};
    }

    /**
     * @brief Get typed data view as std::span
     * @tparam T Type to cast data to
     * @return Span over typed data
     */
    template <typename T>
    std::span<T> typed_data_span() noexcept {
        auto *ptr = reinterpret_cast<T *>(data());
        return ptr ? std::span<T>{ptr, size() / sizeof(T)} : std::span<T>{};
    }
    template <typename T>
    std::span<const T> typed_data_span() const noexcept {
        auto *ptr = reinterpret_cast<const T *>(data());
        return ptr ? std::span<const T>{ptr, size() / sizeof(T)}
                   : std::span<const T>{};
    }

    /**
     * @brief Get the message size
     */
    constexpr size_t size() const noexcept {
        return Derived::calculate_size();
    }

    /**
     * @brief Get the message size at compile time (for fixed-size messages)
     */
    static consteval size_t static_size() noexcept
        requires FixedSizeMessage<Derived>
    {
        return Derived::calculate_size();
    }

    /**
     * @brief Check if the message is valid (has valid slab and offset)
     */
    constexpr bool valid() const noexcept {
        return slab_ != nullptr && offset_ != BUFFER_FULL;
    }

    /**
     * @brief Check if message has data available
     */
    constexpr bool has_data() const noexcept {
        return valid() && size() > 0;
    }

    /**
     * @brief Get the offset within ring buffer
     */
    uint32_t offset() const {
        return offset_;
    }

    /**
     * @brief Buffer full constant for backpressure
     */
    static constexpr uint32_t BUFFER_FULL = 0xFFFFFFFF;

protected:
    RingBuffer *slab_; ///< Pointer to ring buffer
    uint32_t offset_;  ///< Offset within ring buffer where message starts
    Channel *channel_; ///< Channel for sending notifications

    friend class Channel;
};

} // namespace psyne