#pragma once

/**
 * @file ipc_channel.hpp
 * @brief Inter-Process Communication channel implementation
 * @author Psyne Contributors
 * @date 2025
 * 
 * This file implements IPC channels using Boost.Interprocess shared memory.
 * IPC channels enable zero-copy message passing between processes on the same machine.
 */

#include "channel_impl.hpp"
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/deque.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <string>
#include <vector>
#include <span>

namespace psyne {
namespace detail {

namespace bip = boost::interprocess;

/**
 * @struct IPCMessage
 * @brief Message header for IPC communication
 * 
 * This structure represents the header of an IPC message. The actual message
 * data follows immediately after this header in memory.
 */
struct IPCMessage {
    uint32_t type;  ///< Message type identifier
    uint32_t size;  ///< Size of message data in bytes
    
    /**
     * @brief Get pointer to message data
     * @return Pointer to data immediately following the header
     */
    uint8_t *data() {
        return reinterpret_cast<uint8_t *>(this + 1);
    }
    
    /**
     * @brief Get const pointer to message data
     * @return Const pointer to data immediately following the header
     */
    const uint8_t *data() const {
        return reinterpret_cast<const uint8_t *>(this + 1);
    }
};

/**
 * @class IPCChannel
 * @brief Channel implementation for inter-process communication
 * 
 * IPCChannel uses Boost.Interprocess to create shared memory segments that
 * can be accessed by multiple processes. It implements a ring buffer in
 * shared memory with named synchronization primitives for thread-safe access.
 * 
 * The shared memory segment is named based on the channel URI, allowing
 * processes to connect by specifying the same URI.
 * 
 * @note The shared memory is created in /dev/shm/ on Linux systems
 * @note The creator process is responsible for cleanup
 */
class IPCChannel : public ChannelImpl {
public:
    IPCChannel(const std::string &uri, size_t buffer_size, ChannelMode mode,
               ChannelType type);
    ~IPCChannel();

    // Zero-copy interface
    uint32_t reserve_write_slot(size_t size) noexcept override;
    void notify_message_ready(uint32_t offset, size_t size) noexcept override;
    std::span<uint8_t> get_write_span(size_t size) noexcept override;
    std::span<const uint8_t> buffer_span() const noexcept override;
    void advance_read_pointer(size_t size) noexcept override;
    
    // Legacy interface (deprecated)
    [[deprecated("Use reserve_write_slot() instead")]]
    void *reserve_space(size_t size) override;
    [[deprecated("Data is committed when written")]]
    void commit_message(void *handle) override;
    void *receive_message(size_t &size, uint32_t &type) override;
    void release_message(void *handle) override;

private:
    // Shared memory segment name derived from URI
    std::string shm_name_;

    // Boost.Interprocess components
    std::unique_ptr<bip::managed_shared_memory> segment_;
    std::unique_ptr<bip::named_mutex> mutex_;
    std::unique_ptr<bip::named_condition> not_empty_;
    std::unique_ptr<bip::named_condition> not_full_;

    // Ring buffer in shared memory
    struct SharedData {
        std::atomic<bool> stopped;
        std::atomic<size_t> write_pos;
        std::atomic<size_t> read_pos;
        size_t buffer_size;
        size_t mask;
        // Buffer follows immediately after
        uint8_t *buffer() {
            return reinterpret_cast<uint8_t *>(this + 1);
        }
    };

    SharedData *shared_data_;
    uint8_t *buffer_;

    // Local state
    bool is_creator_;
    std::vector<uint8_t> receive_buffer_; // For received messages

    // Helper to align sizes
    static constexpr size_t align_up(size_t size) {
        return (size + 15) & ~15; // 16-byte alignment
    }
};

} // namespace detail
} // namespace psyne