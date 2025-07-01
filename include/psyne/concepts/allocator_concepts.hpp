#pragma once

#include <concepts>
#include <type_traits>

namespace psyne::concepts {

/**
 * @brief Concept for allocators that messages can use
 */
template <typename A>
concept AllocatorType =
    requires(A allocator, size_t size, size_t alignment, void *ptr) {
        // Core allocation interface
        { allocator.allocate(size, alignment) } -> std::convertible_to<void *>;
        { allocator.deallocate(ptr) } -> std::same_as<void>;

        // Allocator info
        { allocator.name() } -> std::convertible_to<const char *>;
        { allocator.is_zero_copy() } -> std::same_as<bool>;
    };

/**
 * @brief Concept for messages that can be constructed with allocators
 */
template <typename T, typename A>
concept MessageWithAllocator = requires(A &allocator) {
    // Must be constructible with allocator
    T(allocator);

    // Must support variadic construction with allocator
    T(allocator, int{}, float{});

    // Must be movable and destructible
    std::is_move_constructible_v<T>;
    std::is_destructible_v<T>;

    // Must have compile-time size
    sizeof(T);
};

/**
 * @brief Concept for substrates (no circular dependency now!)
 */
template <typename S>
concept SubstrateType = requires(S substrate, size_t size_bytes) {
    // Memory slab management
    { substrate.allocate_slab(size_bytes) } -> std::convertible_to<void *>;
    {
        substrate.deallocate_slab(static_cast<void *>(nullptr))
    } -> std::same_as<void>;

    // Substrate capabilities
    { substrate.needs_serialization() } -> std::same_as<bool>;
    { substrate.is_zero_copy() } -> std::same_as<bool>;
    { substrate.is_cross_process() } -> std::same_as<bool>;
    { substrate.name() } -> std::convertible_to<const char *>;

    // Message sending (takes raw pointer, no circular dependency)
    {
        substrate.send_raw_message(static_cast<void *>(nullptr), size_t{})
    } -> std::same_as<void>;
};

/**
 * @brief Concept for patterns (no circular dependency!)
 */
template <typename P>
concept PatternType = requires(P pattern, void *slab, size_t max_messages) {
    // Core pattern operations (work with raw pointers)
    {
        pattern.try_allocate_raw(slab, max_messages, sizeof(int))
    } -> std::convertible_to<void *>;
    { pattern.try_receive_raw() } -> std::convertible_to<void *>;

    // Pattern capabilities
    { pattern.needs_locks() } -> std::same_as<bool>;
    { pattern.max_producers() } -> std::same_as<size_t>;
    { pattern.max_consumers() } -> std::same_as<size_t>;
    { pattern.name() } -> std::convertible_to<const char *>;

    // Pattern state
    { pattern.size() } -> std::same_as<size_t>;
    { pattern.empty() } -> std::same_as<bool>;
    { pattern.full() } -> std::same_as<bool>;
};

/**
 * @brief Complete channel configuration (no circular dependencies!)
 */
template <typename T, typename S, typename P, typename A>
concept ChannelConfiguration = MessageWithAllocator<T, A> && SubstrateType<S> &&
                               PatternType<P> && AllocatorType<A>;

} // namespace psyne::concepts