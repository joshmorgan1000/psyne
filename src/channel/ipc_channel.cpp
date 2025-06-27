#include "ipc_channel.hpp"
#include <boost/interprocess/exceptions.hpp>
#include <regex>
#include <cstring>
#include <iostream>  // For debug

namespace psyne {
namespace detail {

IPCChannel::IPCChannel(const std::string& uri, size_t buffer_size,
                       ChannelMode mode, ChannelType type)
    : ChannelImpl(uri, buffer_size, mode, type)
    , is_creator_(false) {
    
    // Extract shared memory name from URI (e.g., "ipc://my_channel" -> "psyne_my_channel")
    std::regex uri_regex("^ipc://(.+)$");
    std::smatch match;
    if (!std::regex_match(uri, match, uri_regex)) {
        throw std::invalid_argument("Invalid IPC URI: " + uri);
    }
    shm_name_ = "psyne_" + match[1].str();
    
    // Round up buffer size to power of 2
    size_t actual_size = 1;
    while (actual_size < buffer_size) actual_size <<= 1;
    
    try {
        // Try to create the shared memory segment
        try {
            segment_ = std::make_unique<bip::managed_shared_memory>(
                bip::create_only, shm_name_.c_str(), 
                sizeof(SharedData) + actual_size + 1024  // Extra for metadata
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
            
        } catch (const bip::interprocess_exception&) {
            // Already exists, open it
            segment_ = std::make_unique<bip::managed_shared_memory>(
                bip::open_only, shm_name_.c_str()
            );
            
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
            bip::open_or_create, (shm_name_ + "_mutex").c_str()
        );
        not_empty_ = std::make_unique<bip::named_condition>(
            bip::open_or_create, (shm_name_ + "_not_empty").c_str()
        );
        not_full_ = std::make_unique<bip::named_condition>(
            bip::open_or_create, (shm_name_ + "_not_full").c_str()
        );
        
    } catch (const bip::interprocess_exception& e) {
        throw std::runtime_error(std::string("IPC channel creation failed: ") + e.what());
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

void* IPCChannel::reserve_space(size_t size) {
    // For IPC, we allocate a local buffer for the message
    // and copy it to shared memory on commit
    receive_buffer_.resize(sizeof(IPCMessage) + size);
    auto* msg = reinterpret_cast<IPCMessage*>(receive_buffer_.data());
    msg->size = size;
    msg->type = 0;  // Will be set by the message
    return msg->data();
}

void IPCChannel::commit_message(void* handle) {
    // Calculate message location from handle
    auto* data_ptr = static_cast<uint8_t*>(handle);
    auto* msg = reinterpret_cast<IPCMessage*>(data_ptr - sizeof(IPCMessage));
    size_t total_size = align_up(sizeof(IPCMessage) + msg->size);
    
    std::cerr << "[IPC] Committing message, size=" << msg->size << ", total=" << total_size << "\n";
    
    bip::scoped_lock<bip::named_mutex> lock(*mutex_);
    
    // Wait for space if needed
    while (true) {
        size_t write_pos = shared_data_->write_pos.load();
        size_t read_pos = shared_data_->read_pos.load();
        size_t used = write_pos - read_pos;
        
        if (used + total_size <= shared_data_->buffer_size) {
            // Check for wrap-around
            size_t write_offset = write_pos & shared_data_->mask;
            if (write_offset + total_size <= shared_data_->buffer_size) {
                // Copy message to shared memory
                std::memcpy(buffer_ + write_offset, msg, sizeof(IPCMessage) + msg->size);
                shared_data_->write_pos = write_pos + total_size;
                std::cerr << "[IPC] Message committed at offset " << write_offset << "\n";
                not_empty_->notify_one();
                break;
            }
        }
        
        // Buffer full, wait
        not_full_->wait(lock);
    }
}

void* IPCChannel::receive_message(size_t& size, uint32_t& type) {
    bip::scoped_lock<bip::named_mutex> lock(*mutex_);
    
    size_t read_pos = shared_data_->read_pos.load();
    size_t write_pos = shared_data_->write_pos.load();
    
    std::cerr << "[IPC] Receive: read_pos=" << read_pos << ", write_pos=" << write_pos << "\n";
    
    if (read_pos == write_pos) {
        return nullptr;  // Empty
    }
    
    // Read message header
    size_t read_offset = read_pos & shared_data_->mask;
    auto* msg = reinterpret_cast<IPCMessage*>(buffer_ + read_offset);
    
    // Copy message to local buffer
    receive_buffer_.resize(sizeof(IPCMessage) + msg->size);
    std::memcpy(receive_buffer_.data(), msg, sizeof(IPCMessage) + msg->size);
    
    // Update read position
    shared_data_->read_pos = read_pos + align_up(sizeof(IPCMessage) + msg->size);
    not_full_->notify_one();
    
    // Return message data
    auto* local_msg = reinterpret_cast<IPCMessage*>(receive_buffer_.data());
    size = local_msg->size;
    type = local_msg->type;
    return local_msg->data();
}

void IPCChannel::release_message(void* handle) {
    // Nothing to do - message is in local buffer
}

// Note: stop() and is_stopped() are handled by the base class

} // namespace detail
} // namespace psyne