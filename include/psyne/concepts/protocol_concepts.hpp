#pragma once

#include <concepts>
#include <cstddef>
#include <vector>

namespace psyne::concepts {

/**
 * @brief PROTOCOL CONCEPT - Data transformation layer between substrates
 *
 * A protocol is the "buffer between two substrates" that can:
 * 1. Analyze and understand data semantically
 * 2. Make intelligent transformation decisions
 * 3. Encode/decode data for optimized transport
 * 4. Adapt behavior based on network/system conditions
 *
 * Examples: TDT compression, encryption, serialization, checksums
 *
 * NO BASE CLASS NEEDED! Just satisfy the concept.
 */
template <typename P>
concept Protocol = requires(P protocol, void *data, size_t size) {
    // DATA UNDERSTANDING BEHAVIORS
    { protocol.should_transform(data, size) } -> std::same_as<bool>;
    { protocol.analyze_data(data, size) } -> std::same_as<void>;

    // TRANSFORMATION BEHAVIORS
    {
        protocol.encode(data, size)
    } -> std::convertible_to<std::vector<uint8_t>>;
    {
        protocol.decode(std::declval<const std::vector<uint8_t> &>())
    } -> std::convertible_to<std::vector<uint8_t>>;

    // ADAPTATION BEHAVIORS
    {
        protocol.update_network_metrics(0.0, 0.0)
    } -> std::same_as<void>; // bandwidth, latency
    { protocol.update_system_metrics(0.0) } -> std::same_as<void>; // cpu usage

    // IDENTITY BEHAVIORS
    { protocol.protocol_name() } -> std::convertible_to<const char *>;
    { protocol.is_lossless() } -> std::same_as<bool>;
    { protocol.transformation_ratio() } -> std::same_as<double>;
    { protocol.processing_overhead_ms() } -> std::same_as<double>;
};

/**
 * @brief PROTOCOL STACK CONCEPT - Composable protocol layers
 *
 * Allows chaining multiple protocols:
 * TDT Compression -> AES Encryption -> Checksum -> Substrate
 */
template <typename PS>
concept ProtocolStack = requires(PS stack, void *data, size_t size) {
    // STACK BEHAVIORS
    { stack.push_protocol(/* some protocol */) } -> std::same_as<void>;
    {
        stack.encode_stack(data, size)
    } -> std::convertible_to<std::vector<uint8_t>>;
    {
        stack.decode_stack(std::declval<const std::vector<uint8_t> &>())
    } -> std::convertible_to<std::vector<uint8_t>>;

    // STACK IDENTITY
    { stack.stack_name() } -> std::convertible_to<const char *>;
    { stack.total_overhead_ms() } -> std::same_as<double>;
};

} // namespace psyne::concepts

/**
 * @brief PROTOCOL EXAMPLES:
 *
 * // TDT Compression Protocol
 * struct TDTProtocol {
 *     bool should_transform(void* data, size_t size) {
 *         return is_tensor_data(data, size) && size > 1024;
 *     }
 *     std::vector<uint8_t> encode(void* data, size_t size) {
 *         return tdt_compress(data, size);
 *     }
 *     std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded) {
 *         return tdt_decompress(encoded);
 *     }
 *     const char* protocol_name() const { return "TDT-Compression"; }
 *     bool is_lossless() const { return true; }
 *     double transformation_ratio() const { return compression_ratio_; }
 * };
 *
 * // AES Encryption Protocol
 * struct AESProtocol {
 *     bool should_transform(void* data, size_t size) { return true; }
 *     std::vector<uint8_t> encode(void* data, size_t size) {
 *         return aes_encrypt(data, size, key_);
 *     }
 *     std::vector<uint8_t> decode(const std::vector<uint8_t>& encrypted) {
 *         return aes_decrypt(encrypted, key_);
 *     }
 *     const char* protocol_name() const { return "AES-256"; }
 *     bool is_lossless() const { return true; }
 *     double transformation_ratio() const { return 1.0; } // No compression
 * };
 *
 * // Composable Usage:
 * Channel<TensorMsg, TCPSubstrate, SPSCPattern, TDTProtocol>
 * compressed_channel; Channel<TensorMsg, TCPSubstrate, SPSCPattern,
 * ProtocolStack<TDTProtocol, AESProtocol>> secure_channel;
 *
 * PROTOCOLS ARE THE INTELLIGENT BUFFER BETWEEN SUBSTRATES!
 */