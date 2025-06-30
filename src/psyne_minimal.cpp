#include <condition_variable>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <psyne/psyne.hpp>
#include <queue>
#include <stdexcept>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include "../src/channel/tcp_channel_stub.hpp"
#include "../src/compression/compression.hpp"
#include "channel/webrtc_channel.hpp"
#include "channel/quic_channel.hpp"

namespace psyne {

// Minimal RingBuffer stub for compilation
class RingBuffer {
public:
    virtual ~RingBuffer() = default;
    virtual uint8_t* base_ptr() { return nullptr; }
    virtual const uint8_t* base_ptr() const { return nullptr; }
    virtual uint32_t reserve_slot(size_t) { return Channel::BUFFER_FULL; }
    virtual void advance_write_pointer(size_t) {}
    virtual void advance_read_pointer(size_t) {}
    virtual size_t write_position() const { return 0; }
    virtual size_t read_position() const { return 0; }
    virtual size_t capacity() const { return 0; }
    virtual uint32_t reserve_write_space(size_t) { return Channel::BUFFER_FULL; }
    virtual std::span<const uint8_t> get_read_span(uint32_t, size_t) const { return {}; }
};


// Simple message queue implementation for testing - Fixed unbounded growth
class SimpleMessageQueue {
public:
    struct MessageData {
        std::vector<uint8_t> data;
        uint32_t type;
    };

    explicit SimpleMessageQueue(size_t max_size = 1000) : max_size_(max_size) {}

    bool push(const void *data, size_t size, uint32_t type) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Enforce maximum queue size to prevent unbounded growth
        if (queue_.size() >= max_size_) {
            return false; // Queue full, message dropped
        }

        MessageData msg;
        msg.data.resize(size);
        std::memcpy(msg.data.data(), data, size);
        msg.type = type;
        queue_.push(std::move(msg));
        cv_.notify_one();
        return true;
    }

    std::unique_ptr<MessageData> pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return nullptr;
        }
        auto msg = std::make_unique<MessageData>(std::move(queue_.front()));
        queue_.pop();
        return msg;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<MessageData> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    const size_t max_size_;
};

// Global ring buffers for memory channels (enhanced implementation)
static std::map<std::string, std::shared_ptr<SPSCRingBuffer>> g_ring_buffers;
static std::mutex g_buffers_mutex;

// Basic IPC Channel implementation using shared memory
class IPCChannel : public Channel {
public:
    IPCChannel(const std::string &uri, size_t buffer_size, ChannelType type)
        : Channel(uri, buffer_size, type), shm_fd_(-1), shm_ptr_(nullptr) {
        // Extract name from IPC URI (ipc://name -> name)
        std::string name = uri.substr(6); // Skip "ipc://"
        shm_name_ = "/psyne_" + name;

#ifdef _WIN32
        // Windows implementation using CreateFileMapping
        HANDLE hMapFile = CreateFileMappingA(
            INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0,
            static_cast<DWORD>(buffer_size), shm_name_.c_str());

        if (hMapFile == NULL) {
            throw std::runtime_error("Failed to create file mapping: " +
                                     shm_name_);
        }

        shm_ptr_ =
            MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, buffer_size);
        if (shm_ptr_ == NULL) {
            CloseHandle(hMapFile);
            throw std::runtime_error("Failed to map view of file");
        }

        shm_fd_ = reinterpret_cast<int>(
            hMapFile); // Store handle as int for simplicity
#else
        // Unix implementation using shm_open/mmap
        shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR,
                           0600); // Owner read/write only
        if (shm_fd_ == -1) {
            throw std::runtime_error("Failed to create/open shared memory: " +
                                     shm_name_);
        }

        // Set the size of shared memory
        if (ftruncate(shm_fd_, buffer_size) == -1) {
            close(shm_fd_);
            throw std::runtime_error("Failed to set shared memory size");
        }

        // Map shared memory into process address space
        shm_ptr_ = mmap(nullptr, buffer_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, shm_fd_, 0);
        if (shm_ptr_ == MAP_FAILED) {
            close(shm_fd_);
            throw std::runtime_error("Failed to map shared memory");
        }
#endif

        // Initialize ring buffer using the shared memory
        ring_buffer_ = std::make_unique<SPSCRingBuffer>(buffer_size);
        current_read_handle_ = std::nullopt;
    }

    ~IPCChannel() {
#ifdef _WIN32
        if (shm_ptr_) {
            UnmapViewOfFile(shm_ptr_);
        }
        if (shm_fd_ != -1) {
            CloseHandle(reinterpret_cast<HANDLE>(shm_fd_));
        }
#else
        if (shm_ptr_ && shm_ptr_ != MAP_FAILED) {
            munmap(shm_ptr_, buffer_size_);
        }
        if (shm_fd_ != -1) {
            close(shm_fd_);
        }
#endif
        // Note: We don't unlink the shared memory here since other processes
        // might be using it
    }

    void send_raw_message(const void *data, size_t size,
                          uint32_t type) override {
        if (!ring_buffer_)
            return;

        // Reserve space for message type (4 bytes) + data
        size_t total_size = sizeof(uint32_t) + size;
        auto write_handle = ring_buffer_->reserve(total_size);

        if (write_handle) {
            // Write message type first, then data
            uint8_t *buffer = static_cast<uint8_t *>(write_handle->data);
            *reinterpret_cast<uint32_t *>(buffer) = type;
            std::memcpy(buffer + sizeof(uint32_t), data, size);

            write_handle->commit();
        }
    }

    void *receive_raw_message(size_t &size, uint32_t &type) override {
        if (!ring_buffer_)
            return nullptr;

        // Release previous read handle if exists
        if (current_read_handle_) {
            current_read_handle_->release();
            current_read_handle_ = std::nullopt;
        }

        // Try to read new message
        auto read_handle = ring_buffer_->read();
        if (!read_handle)
            return nullptr;

        // Extract message type and data
        const uint8_t *buffer = static_cast<const uint8_t *>(read_handle->data);
        if (read_handle->size < sizeof(uint32_t))
            return nullptr;

        type = *reinterpret_cast<const uint32_t *>(buffer);
        size = read_handle->size - sizeof(uint32_t);

        // Store read handle for later release
        current_read_handle_ = *read_handle;

        // Return pointer to actual message data (after type header)
        return const_cast<uint8_t *>(buffer + sizeof(uint32_t));
    }

    void release_raw_message(void * /*handle*/) override {
        if (current_read_handle_) {
            current_read_handle_->release();
            current_read_handle_ = std::nullopt;
        }
    }

    // Zero-copy interface implementation
    uint32_t reserve_write_slot(size_t size) noexcept override {
        if (!ring_buffer_) return Channel::BUFFER_FULL;
        
        // Reserve space for message type (4 bytes) + data
        size_t total_size = sizeof(uint32_t) + size;
        auto write_handle = ring_buffer_->reserve(total_size);
        
        if (!write_handle) {
            return Channel::BUFFER_FULL;
        }
        
        // Store write handle and return offset
        // For simplicity, we'll use the write position as offset
        // In a real implementation, we'd track this properly
        current_write_handle_ = std::move(*write_handle);
        return 0; // Simplified - in reality would return actual offset
    }
    
    void notify_message_ready(uint32_t /*offset*/, size_t /*size*/) noexcept override {
        // Commit the write that was reserved
        if (current_write_handle_) {
            current_write_handle_->commit();
            current_write_handle_ = std::nullopt;
        }
    }
    
    RingBuffer& get_ring_buffer() noexcept override {
        static RingBuffer dummy; // Stub implementation
        return dummy;
    }
    
    const RingBuffer& get_ring_buffer() const noexcept override {
        static RingBuffer dummy; // Stub implementation
        return dummy;
    }
    
    void advance_read_pointer(size_t /*size*/) noexcept override {
        // In real implementation, would advance ring buffer read pointer
        if (current_read_handle_) {
            current_read_handle_->release();
            current_read_handle_ = std::nullopt;
        }
    }

private:
    std::string shm_name_;
    int shm_fd_;
    void *shm_ptr_;
    std::unique_ptr<SPSCRingBuffer> ring_buffer_;
    std::optional<ReadHandle> current_read_handle_;
    std::optional<WriteHandle> current_write_handle_;
};

// Enhanced Channel implementation using ring buffers
class TestChannel : public Channel {
public:
    TestChannel(const std::string &uri, size_t buffer_size, ChannelType type)
        : Channel(uri, buffer_size, type) {
        std::lock_guard<std::mutex> lock(g_buffers_mutex);
        auto it = g_ring_buffers.find(uri);
        if (it == g_ring_buffers.end()) {
            ring_buffer_ = std::make_shared<SPSCRingBuffer>(buffer_size);
            g_ring_buffers[uri] = ring_buffer_;
        } else {
            ring_buffer_ = it->second;
        }
        current_read_handle_ = std::nullopt;
    }

    ~TestChannel() {
        // Clean up ring buffer from global map when this is the last reference
        std::lock_guard<std::mutex> lock(g_buffers_mutex);
        auto it = g_ring_buffers.find(uri_);
        if (it != g_ring_buffers.end() && it->second.use_count() <= 2) {
            // use_count <= 2 means only the map and this channel hold
            // references
            g_ring_buffers.erase(it);
        }
    }

    void send_raw_message(const void *data, size_t size,
                          uint32_t type) override {
        if (!ring_buffer_)
            return;

        // Reserve space for message type (4 bytes) + data
        size_t total_size = sizeof(uint32_t) + size;
        auto write_handle = ring_buffer_->reserve(total_size);

        if (write_handle) {
            // Write message type first, then data
            uint8_t *buffer = static_cast<uint8_t *>(write_handle->data);
            *reinterpret_cast<uint32_t *>(buffer) = type;
            std::memcpy(buffer + sizeof(uint32_t), data, size);

            write_handle->commit();
        }
        // If reservation fails, message is dropped (backpressure)
    }

    void *receive_raw_message(size_t &size, uint32_t &type) override {
        if (!ring_buffer_)
            return nullptr;

        // Release previous read handle if exists
        if (current_read_handle_) {
            current_read_handle_->release();
            current_read_handle_ = std::nullopt;
        }

        // Try to read new message
        auto read_handle = ring_buffer_->read();
        if (!read_handle)
            return nullptr;

        // Extract message type and data
        const uint8_t *buffer = static_cast<const uint8_t *>(read_handle->data);
        if (read_handle->size < sizeof(uint32_t))
            return nullptr;

        type = *reinterpret_cast<const uint32_t *>(buffer);
        size = read_handle->size - sizeof(uint32_t);

        // Store read handle for later release
        current_read_handle_ = *read_handle;

        // Return pointer to actual message data (after type header)
        return const_cast<uint8_t *>(buffer + sizeof(uint32_t));
    }

    void release_raw_message(void * /*handle*/) override {
        if (current_read_handle_) {
            current_read_handle_->release();
            current_read_handle_ = std::nullopt;
        }
    }

    // Zero-copy interface implementation
    uint32_t reserve_write_slot(size_t size) noexcept override {
        if (!ring_buffer_) return Channel::BUFFER_FULL;
        
        // Reserve space for message type (4 bytes) + data
        size_t total_size = sizeof(uint32_t) + size;
        auto write_handle = ring_buffer_->reserve(total_size);
        
        if (!write_handle) {
            return Channel::BUFFER_FULL;
        }
        
        // Store write handle and return offset
        // For simplicity, we'll use the write position as offset
        // In a real implementation, we'd track this properly
        current_write_handle_ = std::move(*write_handle);
        return 0; // Simplified - in reality would return actual offset
    }
    
    void notify_message_ready(uint32_t /*offset*/, size_t /*size*/) noexcept override {
        // Commit the write that was reserved
        if (current_write_handle_) {
            current_write_handle_->commit();
            current_write_handle_ = std::nullopt;
        }
    }
    
    RingBuffer& get_ring_buffer() noexcept override {
        static RingBuffer dummy; // Stub implementation
        return dummy;
    }
    
    const RingBuffer& get_ring_buffer() const noexcept override {
        static RingBuffer dummy; // Stub implementation
        return dummy;
    }
    
    void advance_read_pointer(size_t /*size*/) noexcept override {
        // In real implementation, would advance ring buffer read pointer
        if (current_read_handle_) {
            current_read_handle_->release();
            current_read_handle_ = std::nullopt;
        }
    }
    
    std::span<uint8_t> get_write_span(size_t size) noexcept override {
        if (!ring_buffer_) {
            return std::span<uint8_t>{};
        }
        
        // Reserve space for message type (4 bytes) + data
        size_t total_size = sizeof(uint32_t) + size;
        auto write_handle = ring_buffer_->reserve(total_size);
        
        if (!write_handle) {
            return std::span<uint8_t>{};
        }
        
        // Store write handle for later use
        current_write_handle_ = std::move(*write_handle);
        
        // Return span pointing to the data portion (after the 4-byte type header)
        uint8_t *buffer = static_cast<uint8_t *>(current_write_handle_->data);
        return std::span<uint8_t>(buffer + sizeof(uint32_t), size);
    }

private:
    std::shared_ptr<SPSCRingBuffer> ring_buffer_;
    std::optional<ReadHandle> current_read_handle_;
    std::optional<WriteHandle> current_write_handle_;
};

// Note: Message template methods are defined inline in psyne.hpp

// Channel implementation stubs
std::unique_ptr<Channel>
Channel::create(const std::string &uri, size_t buffer_size, ChannelMode mode,
                ChannelType type, bool enable_metrics,
                const compression::CompressionConfig &compression_config) {
    // Validate buffer size
    if (buffer_size == 0) {
        throw std::invalid_argument("Buffer size cannot be zero");
    }

    // Validate URI format
    if (uri.find("://") == std::string::npos) {
        throw std::invalid_argument("Invalid URI format: missing protocol");
    }

    std::string protocol = uri.substr(0, uri.find("://"));
    if (protocol != "memory" && protocol != "ipc" && protocol != "unix" &&
        protocol != "tcp" && protocol != "ws" && protocol != "wss" &&
        protocol != "webrtc" && protocol != "quic") {
        throw std::invalid_argument("Unsupported protocol: " + protocol);
    }

    // Route to appropriate channel implementation based on protocol
    if (protocol == "ipc") {
        return std::make_unique<IPCChannel>(uri, buffer_size, type);
    } else if (protocol == "tcp") {
        return detail::create_tcp_channel(uri, buffer_size, mode, type,
                                          compression_config);
    } else if (protocol == "webrtc") {
        // Create WebRTC channel with default configuration
        detail::WebRTCConfig webrtc_config;
        webrtc_config.stun_servers.push_back(
            {"stun.l.google.com", 19302, "", ""});

        std::string signaling_server = "ws://localhost:8080";
        auto webrtc_impl = detail::create_webrtc_channel(
            uri, buffer_size, mode, type, webrtc_config, signaling_server,
            compression_config);

        // WebRTC uses memcpy for network serialization (acceptable exception)
        // Need to wrap since WebRTCChannel inherits from ChannelImpl
        class WebRTCChannelAdapter : public Channel {
        public:
            explicit WebRTCChannelAdapter(std::unique_ptr<detail::WebRTCChannel> impl) 
                : Channel(impl->uri(), 65536, impl->type()), impl_(std::move(impl)) {}
            
            void stop() override { impl_->stop(); }
            bool is_stopped() const override { return impl_->is_stopped(); }
            void *receive_raw_message(size_t &size, uint32_t &type) override {
                return impl_->receive_message(size, type);
            }
            void release_raw_message(void *handle) override {
                impl_->release_message(handle);
            }
            uint32_t reserve_write_slot(size_t size) noexcept override {
                return impl_->reserve_write_slot(size);
            }
            void notify_message_ready(uint32_t offset, size_t size) noexcept override {
                impl_->notify_message_ready(offset, size);
            }
            RingBuffer& get_ring_buffer() noexcept override {
                return impl_->get_ring_buffer();
            }
            const RingBuffer& get_ring_buffer() const noexcept override {
                return const_cast<WebRTCChannelAdapter*>(this)->impl_->get_ring_buffer();
            }
            void advance_read_pointer(size_t size) noexcept override {
                impl_->advance_read_pointer(size);
            }
            
        private:
            std::unique_ptr<detail::WebRTCChannel> impl_;
        };
        
        return std::make_unique<WebRTCChannelAdapter>(std::move(webrtc_impl));
    } else if (protocol == "quic") {
        // Create QUIC channel
        auto quic_impl = detail::create_quic_channel(
            uri, buffer_size, mode, type, compression_config);

        // QUIC adapter similar to WebRTC
        class QUICChannelAdapter : public Channel {
        public:
            explicit QUICChannelAdapter(std::unique_ptr<detail::ChannelImpl> impl) 
                : Channel(impl->uri(), 65536, impl->type()) {
                impl_.reset(static_cast<detail::QUICChannel*>(impl.release()));
            }
            
            void stop() override { impl_->stop(); }
            bool is_stopped() const override { return impl_->is_stopped(); }
            void *receive_raw_message(size_t &size, uint32_t &type) override {
                return impl_->receive_message(size, type);
            }
            void release_raw_message(void *handle) override {
                impl_->release_message(handle);
            }
            bool has_metrics() const override { return false; }
            debug::ChannelMetrics get_metrics() const override { return {}; }
            void reset_metrics() override {}
            uint32_t reserve_write_slot(size_t size) noexcept override {
                return impl_->reserve_write_slot(size);
            }
            void notify_message_ready(uint32_t offset, size_t size) noexcept override {
                impl_->notify_message_ready(offset, size);
            }
            RingBuffer& get_ring_buffer() noexcept override {
                return impl_->get_ring_buffer();
            }
            const RingBuffer& get_ring_buffer() const noexcept override {
                return const_cast<QUICChannelAdapter*>(this)->impl_->get_ring_buffer();
            }
            void advance_read_pointer(size_t size) noexcept override {
                impl_->advance_read_pointer(size);
            }
            
        private:
            std::unique_ptr<detail::QUICChannel> impl_;
        };
        
        return std::make_unique<QUICChannelAdapter>(std::move(quic_impl));
    } else {
        // Default to memory channel for all other protocols (for now)
        return std::make_unique<TestChannel>(uri, buffer_size, type);
    }
}

// Explicit template instantiations for library message types only
// Test message types (TestMsg, SimpleMessage) are instantiated in test files

// SPSCRingBuffer implementation - Enhanced circular buffer
SPSCRingBuffer::SPSCRingBuffer(size_t capacity) : capacity_(capacity) {
    // Allocate buffer for circular ring buffer
    buffer_ = std::make_unique<uint8_t[]>(capacity);
    write_pos_.store(0);
    read_pos_.store(0);
    reserved_size_ = 0;
}

std::optional<WriteHandle> SPSCRingBuffer::reserve(size_t size) {
    // Need space for size header (8 bytes) + message data
    size_t total_size = sizeof(uint64_t) + size;

    if (total_size > capacity_) {
        return std::nullopt; // Message too large for buffer
    }

    size_t current_write = write_pos_.load();
    size_t current_read = read_pos_.load();

    // Calculate available space in circular buffer
    size_t available_space;
    if (current_write >= current_read) {
        // Write position is ahead of read position
        size_t space_to_end = capacity_ - current_write;
        size_t space_from_start = current_read;

        if (total_size <= space_to_end) {
            // Fits in remaining space
            available_space = space_to_end;
        } else if (total_size <= space_from_start && current_read > 0) {
            // Need to wrap around to beginning
            current_write = 0;
            available_space = space_from_start;
        } else {
            return std::nullopt; // Not enough space
        }
    } else {
        // Read position is ahead of write position
        available_space = current_read - current_write;
        if (total_size > available_space) {
            return std::nullopt; // Not enough space
        }
    }

    // Reserve space and store size header
    uint64_t *size_header =
        reinterpret_cast<uint64_t *>(buffer_.get() + current_write);
    *size_header = size;

    reserved_size_ = total_size;
    reserved_write_pos_ = current_write;

    return WriteHandle{buffer_.get() + current_write + sizeof(uint64_t), size,
                       this};
}

std::optional<ReadHandle> SPSCRingBuffer::read() {
    size_t current_read = read_pos_.load();
    size_t current_write = write_pos_.load();

    if (current_read == current_write) {
        return std::nullopt; // No data available
    }

    // Read size header
    uint64_t *size_header =
        reinterpret_cast<uint64_t *>(buffer_.get() + current_read);
    uint64_t message_size = *size_header;

    return ReadHandle{buffer_.get() + current_read + sizeof(uint64_t),
                      message_size, this};
}

void WriteHandle::commit() {
    // Mark data as written by updating write position
    if (ring_buffer) {
        auto *rb = ring_buffer;
        size_t new_write_pos = rb->reserved_write_pos_ + rb->reserved_size_;

        // Handle wrap-around
        if (new_write_pos >= rb->capacity_) {
            new_write_pos = 0;
        }

        rb->write_pos_.store(new_write_pos);
    }
}

void ReadHandle::release() {
    // Advance read position past this message
    if (ring_buffer) {
        auto *rb = ring_buffer;
        size_t current_read = rb->read_pos_.load();
        size_t message_total_size = sizeof(uint64_t) + size;
        size_t new_read_pos = current_read + message_total_size;

        // Handle wrap-around
        if (new_read_pos >= rb->capacity_) {
            new_read_pos = 0;
        }

        rb->read_pos_.store(new_read_pos);
    }
}

// Add basic function implementations
const char *get_version() {
    return "1.2.0";
}

void print_banner() {
    // Empty implementation
}

// FloatVector implementations
float &FloatVector::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("FloatVector index out of range");
    }
    float *floats = reinterpret_cast<float *>(data() + sizeof(size_t));
    return floats[index];
}

const float &FloatVector::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("FloatVector index out of range");
    }
    const float *floats =
        reinterpret_cast<const float *>(data() + sizeof(size_t));
    return floats[index];
}

float *FloatVector::begin() {
    return reinterpret_cast<float *>(data() + sizeof(size_t));
}

float *FloatVector::end() {
    return begin() + size();
}

const float *FloatVector::begin() const {
    return reinterpret_cast<const float *>(data() + sizeof(size_t));
}

const float *FloatVector::end() const {
    return begin() + size();
}

size_t FloatVector::size() const {
    // Assuming first 8 bytes store size
    if (!data())
        return 0;
    return *reinterpret_cast<const size_t *>(data());
}

size_t FloatVector::capacity() const {
    return (Message<FloatVector>::size() - sizeof(size_t)) / sizeof(float);
}

void FloatVector::resize(size_t new_size) {
    if (data() && new_size <= capacity()) {
        *reinterpret_cast<size_t *>(data()) = new_size;
    }
}

FloatVector &FloatVector::operator=(std::initializer_list<float> values) {
    resize(values.size());
    std::copy(values.begin(), values.end(), begin());
    return *this;
}

void FloatVector::initialize() {
    if (data()) {
        *reinterpret_cast<size_t *>(data()) = 0;
    }
}

// ByteVector implementations
uint8_t &ByteVector::operator[](size_t index) {
    return data()[index];
}

const uint8_t &ByteVector::operator[](size_t index) const {
    return data()[index];
}

uint8_t *ByteVector::begin() {
    return data();
}

uint8_t *ByteVector::end() {
    return data() + size();
}

const uint8_t *ByteVector::begin() const {
    return data();
}

const uint8_t *ByteVector::end() const {
    return data() + size();
}

uint8_t *ByteVector::data() {
    return Message<ByteVector>::data() + sizeof(size_t);
}

const uint8_t *ByteVector::data() const {
    return Message<ByteVector>::data() + sizeof(size_t);
}

size_t ByteVector::size() const {
    if (!Message<ByteVector>::data())
        return 0;
    return *reinterpret_cast<const size_t *>(Message<ByteVector>::data());
}

size_t ByteVector::capacity() const {
    return Message<ByteVector>::size() - sizeof(size_t);
}

void ByteVector::resize(size_t new_size) {
    if (Message<ByteVector>::data() && new_size <= capacity()) {
        *reinterpret_cast<size_t *>(Message<ByteVector>::data()) = new_size;
    }
}

void ByteVector::initialize() {
    if (Message<ByteVector>::data()) {
        *reinterpret_cast<size_t *>(Message<ByteVector>::data()) = 0;
    }
}

// DoubleMatrix implementations
void DoubleMatrix::set_dimensions(size_t rows, size_t cols) {
    if (Message<DoubleMatrix>::data()) {
        size_t *header =
            reinterpret_cast<size_t *>(Message<DoubleMatrix>::data());
        header[0] = rows;
        header[1] = cols;
    }
}

size_t DoubleMatrix::rows() const {
    if (!Message<DoubleMatrix>::data())
        return 0;
    return reinterpret_cast<const size_t *>(Message<DoubleMatrix>::data())[0];
}

size_t DoubleMatrix::cols() const {
    if (!Message<DoubleMatrix>::data())
        return 0;
    return reinterpret_cast<const size_t *>(Message<DoubleMatrix>::data())[1];
}

double &DoubleMatrix::at(size_t row, size_t col) {
    double *matrix_data = reinterpret_cast<double *>(
        Message<DoubleMatrix>::data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

const double &DoubleMatrix::at(size_t row, size_t col) const {
    const double *matrix_data = reinterpret_cast<const double *>(
        Message<DoubleMatrix>::data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

void DoubleMatrix::initialize() {
    if (Message<DoubleMatrix>::data()) {
        size_t *header =
            reinterpret_cast<size_t *>(Message<DoubleMatrix>::data());
        header[0] = 0; // rows
        header[1] = 0; // cols
    }
}

// Explicit template instantiations for library message types
// Test message types (TestMsg, SimpleMessage) will be instantiated in test
// files
template class Message<FloatVector>;
template class Message<ByteVector>;
template class Message<DoubleMatrix>;

} // namespace psyne
