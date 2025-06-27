#pragma once

#include "channel_impl.hpp"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/deque.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <string>
#include <vector>

namespace psyne {
namespace detail {

namespace bip = boost::interprocess;

// Simple IPC message structure
struct IPCMessage {
    uint32_t type;
    uint32_t size;
    // Data follows immediately after
    uint8_t* data() { return reinterpret_cast<uint8_t*>(this + 1); }
    const uint8_t* data() const { return reinterpret_cast<const uint8_t*>(this + 1); }
};

// IPC Channel using Boost.Interprocess shared memory
class IPCChannel : public ChannelImpl {
public:
    IPCChannel(const std::string& uri, size_t buffer_size,
               ChannelMode mode, ChannelType type);
    ~IPCChannel();
    
    void* reserve_space(size_t size) override;
    void commit_message(void* handle) override;
    void* receive_message(size_t& size, uint32_t& type) override;
    void release_message(void* handle) override;
    
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
        uint8_t* buffer() { return reinterpret_cast<uint8_t*>(this + 1); }
    };
    
    SharedData* shared_data_;
    uint8_t* buffer_;
    
    // Local state
    bool is_creator_;
    std::vector<uint8_t> receive_buffer_;  // For received messages
    
    // Helper to align sizes
    static constexpr size_t align_up(size_t size) {
        return (size + 15) & ~15;  // 16-byte alignment
    }
};

} // namespace detail
} // namespace psyne