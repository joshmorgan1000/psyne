#include "ipc_channel.hpp"
#include <boost/interprocess/exceptions.hpp>
#include <cstring>
#include <iostream> // For debug
#include <regex>

namespace psyne {
namespace detail {

IPCChannel::IPCChannel(const std::string &uri, size_t buffer_size,
                       ChannelMode mode, ChannelType type)
    : ChannelImpl(uri, buffer_size, mode, type), is_creator_(false) {
    // Extract shared memory name from URI (e.g., "ipc://my_channel" ->
    // "psyne_my_channel")
    std::regex uri_regex("^ipc://(.+)$");
    std::smatch match;
    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::invalid_argument("Invalid IPC URI: " + uri);
    }
    shm_name_ = "psyne_" + match[1].str();

    // Round up buffer size to power of 2
    size_t actual_size = 1;
    while (actual_size < buffer_size)
        actual_size <<= 1;

    try {
        // Try to create the shared memory segment
        try {
            segment_ = std::make_unique<bip::managed_shared_memory>(
                bip::create_only, shm_name_.c_str(),
                sizeof(SharedData) + actual_size + 1024 // Extra for metadata
            );
            is_creator_ = true;

            // Initialize shared data
            shared_data_ = segment_->construct<SharedData>("data")();
            shared_data_->stopped = false;
            shared_data_->write_pos = 0;
            shared_data_->read_pos = 0;
            shared_data_->buffer_size = actual_size;
            shared_data_->mask = actual_size - 1;

            // Create synchronization objects
            bip::named_mutex::remove((shm_name_ + "_mutex").c_str());
            bip::named_condition::remove((shm_name_ + "_not_empty").c_str());
            bip::named_condition::remove((shm_name_ + "_not_full").c_str());

        } catch (const bip::interprocess_exception &) {
            // Already exists, open it
            segment_ = std::make_unique<bip::managed_shared_memory>(
                bip::open_only, shm_name_.c_str());

            // Find existing shared data
            auto result = segment_->find<SharedData>("data");
            if (!result.first) {
                throw std::runtime_error("IPC channel data not found");
            }
            shared_data_ = result.first;
        }

        buffer_ = shared_data_->buffer();

        // Open or create synchronization objects
        mutex_ = std::make_unique<bip::named_mutex>(
            bip::open_or_create, (shm_name_ + "_mutex").c_str());
        not_empty_ = std::make_unique<bip::named_condition>(
            bip::open_or_create, (shm_name_ + "_not_empty").c_str());
        not_full_ = std::make_unique<bip::named_condition>(
            bip::open_or_create, (shm_name_ + "_not_full").c_str());

    } catch (const bip::interprocess_exception &e) {
        throw std::runtime_error(std::string("IPC channel creation failed: ") +
                                 e.what());
    }
}

IPCChannel::~IPCChannel() {
    if (is_creator_) {
        // Clean up shared memory and sync objects
        bip::shared_memory_object::remove(shm_name_.c_str());
        bip::named_mutex::remove((shm_name_ + "_mutex").c_str());
        bip::named_condition::remove((shm_name_ + "_not_empty").c_str());
        bip::named_condition::remove((shm_name_ + "_not_full").c_str());
    }
}

uint32_t IPCChannel::reserve_write_slot(size_t size) noexcept {
    if (!shared_data_) {
        return BUFFER_FULL;
    }
    
    bip::scoped_lock<bip::named_mutex> lock(*mutex_);
    
    size_t write_pos = shared_data_->write_pos.load();
    size_t read_pos = shared_data_->read_pos.load();
    size_t used = write_pos - read_pos;
    
    if (used + size > shared_data_->buffer_size) {
        return BUFFER_FULL; // Ring buffer full
    }
    
    // Return offset directly in shared memory ring buffer
    return static_cast<uint32_t>(write_pos & shared_data_->mask);
}

void IPCChannel::notify_message_ready(uint32_t offset, size_t size) noexcept {
    if (!shared_data_) {
        return;
    }
    
    // Message was written directly to ring buffer at offset
    // Just update write position and notify
    size_t total_size = align_up(size);
    
    bip::scoped_lock<bip::named_mutex> lock(*mutex_);
    shared_data_->write_pos += total_size;
    not_empty_->notify_one();
}

std::span<uint8_t> IPCChannel::get_write_span(size_t size) noexcept {
    uint32_t offset = reserve_write_slot(size);
    if (offset == BUFFER_FULL) {
        return {};
    }
    return {buffer_ + offset, size};
}

std::span<const uint8_t> IPCChannel::buffer_span() const noexcept {
    if (!shared_data_) {
        return {};
    }
    
    size_t read_pos = shared_data_->read_pos.load();
    size_t write_pos = shared_data_->write_pos.load();
    
    if (read_pos == write_pos) {
        return {}; // Empty
    }
    
    size_t read_offset = read_pos & shared_data_->mask;
    size_t available = write_pos - read_pos;
    
    return {buffer_ + read_offset, std::min(available, shared_data_->buffer_size - read_offset)};
}

void IPCChannel::advance_read_pointer(size_t size) noexcept {
    if (!shared_data_) {
        return;
    }
    
    bip::scoped_lock<bip::named_mutex> lock(*mutex_);
    shared_data_->read_pos += align_up(size);
    not_full_->notify_one();
}

// Legacy deprecated methods (kept for compatibility)
void* IPCChannel::reserve_space(size_t size) {
    uint32_t offset = reserve_write_slot(size);
    if (offset == BUFFER_FULL) {
        return nullptr;
    }
    return buffer_ + offset;
}

void IPCChannel::commit_message(void* handle) {
    // This is a no-op in zero-copy design
    // Message was already written directly to shared memory
}

void* IPCChannel::receive_message(size_t& size, uint32_t& type) {
    auto span = buffer_span();
    if (span.empty()) {
        return nullptr;
    }
    size = span.size();
    type = 1; // Default type
    return const_cast<uint8_t*>(span.data());
}

void IPCChannel::release_message(void* handle) {
    // In zero-copy design, caller must call advance_read_pointer()
}

// Note: stop() and is_stopped() are handled by the base class

} // namespace detail
} // namespace psyne