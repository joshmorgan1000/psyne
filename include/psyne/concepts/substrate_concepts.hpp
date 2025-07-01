#pragma once

#include <concepts>
#include <cstddef>

namespace psyne::concepts {

/**
 * @brief SUBSTRATE CONCEPT - Pure behavioral contract
 *
 * A substrate is anything that can:
 * 1. Own memory (allocate/deallocate slab)
 * 2. Own transport (send/receive data)
 * 3. Identify capabilities (zero-copy, cross-process, etc.)
 *
 * NO BASE CLASS NEEDED! Just satisfy the concept.
 */
template <typename S>
concept Substrate = requires(S substrate, void *ptr, size_t size) {
    // MEMORY OWNERSHIP BEHAVIORS
    { substrate.allocate_memory_slab(size) } -> std::convertible_to<void *>;
    { substrate.deallocate_memory_slab(ptr) } -> std::same_as<void>;

    // TRANSPORT BEHAVIORS
    { substrate.transport_send(ptr, size) } -> std::same_as<void>;
    { substrate.transport_receive(ptr, size) } -> std::same_as<void>;

    // IDENTITY BEHAVIORS
    { substrate.substrate_name() } -> std::convertible_to<const char *>;
    { substrate.is_zero_copy() } -> std::same_as<bool>;
    { substrate.is_cross_process() } -> std::same_as<bool>;
};

/**
 * @brief PATTERN CONCEPT - Pure behavioral contract
 *
 * A pattern is anything that can coordinate producer/consumer access
 */
template <typename P>
concept Pattern = requires(P pattern, void *slab, size_t size) {
    // COORDINATION BEHAVIORS
    {
        pattern.coordinate_allocation(slab, size, size)
    } -> std::convertible_to<void *>;
    { pattern.coordinate_receive() } -> std::convertible_to<void *>;

    // SYNCHRONIZATION BEHAVIORS
    { pattern.producer_sync() } -> std::same_as<void>;
    { pattern.consumer_sync() } -> std::same_as<void>;

    // IDENTITY BEHAVIORS
    { pattern.pattern_name() } -> std::convertible_to<const char *>;
    { pattern.needs_locks() } -> std::same_as<bool>;
    { pattern.max_producers() } -> std::same_as<size_t>;
    { pattern.max_consumers() } -> std::same_as<size_t>;
};

/**
 * @brief MESSAGE LENS CONCEPT - Pure behavioral contract
 *
 * A message lens is anything that provides typed access to substrate memory
 */
template <typename M, typename T>
concept MessageLens = requires(M lens) {
    // LENS ACCESS BEHAVIORS
    { lens.operator->() } -> std::convertible_to<T *>;
    { lens.operator*() } -> std::convertible_to<T &>;

    // MEMORY ACCESS BEHAVIORS (for substrate transport)
    { lens.raw_memory() } -> std::convertible_to<void *>;
    { lens.size() } -> std::same_as<size_t>;
};

/**
 * @brief COMPLETE CHANNEL CONCEPT - Everything together
 */
template <typename T, typename S, typename P>
concept ChannelConfiguration = Substrate<S> && Pattern<P>;

} // namespace psyne::concepts

/**
 * @brief CONCEPT-BASED EXAMPLES:
 *
 * // InfiniBand substrate - just satisfies the concept!
 * struct InfiniBandSubstrate {
 *     void* allocate_memory_slab(size_t size) { return ibv_alloc_dm(...); }
 *     void deallocate_memory_slab(void* ptr) { ibv_free_dm(ptr); }
 *     void transport_send(void* data, size_t size) { ibv_post_send(...); }
 *     void transport_receive(void* buf, size_t size) { ibv_post_recv(...); }
 *     const char* substrate_name() const { return "InfiniBand"; }
 *     bool is_zero_copy() const { return true; }
 *     bool is_cross_process() const { return true; }
 * };
 *
 * // CSV file substrate - completely different implementation!
 * struct CSVSubstrate {
 *     void* allocate_memory_slab(size_t size) { return malloc(size); }
 *     void deallocate_memory_slab(void* ptr) { free(ptr); }
 *     void transport_send(void* data, size_t size) { write_to_csv_file(data); }
 *     void transport_receive(void* buf, size_t size) { read_from_csv_file(buf);
 * } const char* substrate_name() const { return "CSV"; } bool is_zero_copy()
 * const { return false; } bool is_cross_process() const { return true; }
 * };
 *
 * // Both work identically:
 * Channel<MyMessage, InfiniBandSubstrate, SPSCPattern> ib_channel;
 * Channel<MyMessage, CSVSubstrate, SPSCPattern> csv_channel;
 *
 * PURE DUCK TYPING! NO INHERITANCE! MAXIMUM FLEXIBILITY!
 */