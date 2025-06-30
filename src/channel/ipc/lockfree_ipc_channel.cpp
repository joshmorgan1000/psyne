/**
 * @file lockfree_ipc_channel.cpp
 * @brief Lock-free IPC channel implementation
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "lockfree_ipc_channel.hpp"
#include "../simd/simd_ops.hpp"
#include "../utils/checksum.hpp"
#include <cstring>
#include <sstream>
#include <thread>

#ifdef __linux__
#include <poll.h>
#include <sys/eventfd.h>
#endif

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

namespace psyne {

// Platform-specific helpers
namespace {

#ifdef _WIN32
std::string get_shared_name(const std::string &base,
                            const std::string &suffix) {
    return "Global\\" + base + suffix;
}

uint32_t get_current_pid() {
    return GetCurrentProcessId();
}
#else
std::string get_shared_name(const std::string &base,
                            const std::string &suffix) {
    return base + suffix;
}

uint32_t get_current_pid() {
    return static_cast<uint32_t>(getpid());
}
#endif

} // anonymous namespace

LockFreeIPCChannel::LockFreeIPCChannel(const std::string &name, bool is_server,
                                       const Config &config)
    : config_(config), channel_name_(name), is_server_(is_server),
      is_connected_(false), header_(nullptr), buffer_(nullptr) {
    if (!init_shared_memory()) {
        throw std::runtime_error("Failed to initialize shared memory");
    }

    if (!init_notifier()) {
        cleanup();
        throw std::runtime_error("Failed to initialize notifier");
    }

    // Set CPU affinity if requested
    if (config_.cpu_affinity >= 0) {
        set_cpu_affinity(config_.cpu_affinity);
    }
}

LockFreeIPCChannel::~LockFreeIPCChannel() {
    close();
}

std::unique_ptr<LockFreeIPCChannel>
LockFreeIPCChannel::create_server(const std::string &name,
                                  const Config &config) {
    return std::unique_ptr<LockFreeIPCChannel>(
        new LockFreeIPCChannel(name, true, config));
}

std::unique_ptr<LockFreeIPCChannel>
LockFreeIPCChannel::create_client(const std::string &name,
                                  const Config &config) {
    return std::unique_ptr<LockFreeIPCChannel>(
        new LockFreeIPCChannel(name, false, config));
}

bool LockFreeIPCChannel::init_shared_memory() {
    size_t total_size = sizeof(SharedHeader) + config_.buffer_size;

    // Create or open shared memory
    std::string shm_name =
        get_shared_name(config_.namespace_prefix + channel_name_, "_shm");
    void *ptr =
        map_shared_memory(shm_name, total_size, is_server_, shared_mem_);
    if (!ptr) {
        return false;
    }

    header_ = reinterpret_cast<SharedHeader *>(ptr);
    buffer_ = reinterpret_cast<uint8_t *>(ptr) + sizeof(SharedHeader);

    if (is_server_) {
        // Initialize header
        header_->magic.store(SharedHeader::MAGIC, std::memory_order_relaxed);
        header_->version.store(1, std::memory_order_relaxed);
        header_->flags.store(0, std::memory_order_relaxed);
        header_->buffer_size.store(config_.buffer_size,
                                   std::memory_order_relaxed);
        header_->write_pos.store(0, std::memory_order_relaxed);
        header_->read_pos.store(0, std::memory_order_relaxed);
        header_->write_cache.store(0, std::memory_order_relaxed);
        header_->read_cache.store(0, std::memory_order_relaxed);
        header_->server_pid.store(get_current_pid(), std::memory_order_relaxed);
        header_->client_pid.store(0, std::memory_order_relaxed);
        header_->server_ready.store(1, std::memory_order_release);
        header_->client_ready.store(0, std::memory_order_relaxed);
        header_->messages_sent.store(0, std::memory_order_relaxed);
        header_->messages_received.store(0, std::memory_order_relaxed);
        header_->bytes_sent.store(0, std::memory_order_relaxed);
        header_->bytes_received.store(0, std::memory_order_relaxed);
    } else {
        // Validate magic number
        if (header_->magic.load(std::memory_order_acquire) !=
            SharedHeader::MAGIC) {
            return false;
        }

        // Set client info
        header_->client_pid.store(get_current_pid(), std::memory_order_relaxed);
        header_->client_ready.store(1, std::memory_order_release);
    }

    // Wait for peer
    if (!wait_for_peer(std::chrono::milliseconds(5000))) {
        return false;
    }

    is_connected_ = true;
    return true;
}

bool LockFreeIPCChannel::init_notifier() {
#ifdef _WIN32
    // Create events for signaling
    std::string send_name = get_shared_name(
        config_.namespace_prefix + channel_name_, is_server_ ? "_s2c" : "_c2s");
    std::string recv_name = get_shared_name(
        config_.namespace_prefix + channel_name_, is_server_ ? "_c2s" : "_s2c");

    send_notifier_.event =
        CreateEventA(nullptr, FALSE, FALSE, send_name.c_str());
    if (!send_notifier_.event)
        return false;

    recv_notifier_.event =
        CreateEventA(nullptr, FALSE, FALSE, recv_name.c_str());
    if (!recv_notifier_.event)
        return false;

#else
    // Try eventfd first (Linux)
#ifdef __linux__
    if (config_.use_eventfd) {
        send_notifier_.eventfd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        recv_notifier_.eventfd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);

        if (send_notifier_.eventfd >= 0 && recv_notifier_.eventfd >= 0) {
            return true;
        }

        // Cleanup on failure
        if (send_notifier_.eventfd >= 0) {
            ::close(send_notifier_.eventfd);
            send_notifier_.eventfd = -1;
        }
        if (recv_notifier_.eventfd >= 0) {
            ::close(recv_notifier_.eventfd);
            recv_notifier_.eventfd = -1;
        }
    }
#endif

#ifdef __APPLE__
    // Use dispatch semaphores on macOS
    send_notifier_.semaphore = dispatch_semaphore_create(0);
    recv_notifier_.semaphore = dispatch_semaphore_create(0);
    
    return send_notifier_.semaphore != nullptr &&
           recv_notifier_.semaphore != nullptr;
#else
    // Fall back to POSIX semaphores on Linux
    std::string send_name = config_.namespace_prefix + channel_name_ +
                            (is_server_ ? "_s2c" : "_c2s");
    std::string recv_name = config_.namespace_prefix + channel_name_ +
                            (is_server_ ? "_c2s" : "_s2c");

    send_notifier_.sem_name = send_name;
    recv_notifier_.sem_name = recv_name;

    if (is_server_) {
        // Create semaphores
        send_notifier_.semaphore =
            sem_open(send_name.c_str(), O_CREAT | O_EXCL, 0666, 0);
        recv_notifier_.semaphore =
            sem_open(recv_name.c_str(), O_CREAT | O_EXCL, 0666, 0);
    } else {
        // Open existing semaphores
        send_notifier_.semaphore = sem_open(send_name.c_str(), 0);
        recv_notifier_.semaphore = sem_open(recv_name.c_str(), 0);
    }

    return send_notifier_.semaphore != SEM_FAILED &&
           recv_notifier_.semaphore != SEM_FAILED;
#endif
#endif
}

bool LockFreeIPCChannel::wait_for_peer(std::chrono::milliseconds timeout) {
    auto start = std::chrono::steady_clock::now();

    while (true) {
        if (is_server_) {
            if (header_->client_ready.load(std::memory_order_acquire) != 0) {
                return true;
            }
        } else {
            if (header_->server_ready.load(std::memory_order_acquire) != 0) {
                return true;
            }
        }

        if (std::chrono::steady_clock::now() - start > timeout) {
            return false;
        }

        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

// Template methods must be implemented in header file
/*
void LockFreeIPCChannel::send(const Message &message) {
    if (!is_connected_) {
        throw std::runtime_error("Channel not connected");
    }

    // Serialize message
    auto serialized = message.serialize();
    if (serialized.size() > config_.max_message_size) {
        throw std::runtime_error("Message too large");
    }

    // Prepare message header
    MessageHeader msg_header;
    msg_header.size =
        static_cast<uint32_t>(sizeof(MessageHeader) + serialized.size());
    msg_header.type_hash = message.type_hash();
    msg_header.timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    msg_header.checksum = simd::SIMDChecksum::crc32(reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size());
    msg_header.flags = 0;

    // Write to ring buffer
    if (!write_to_buffer(&msg_header, sizeof(msg_header))) {
        throw std::runtime_error("Failed to write message header");
    }

    if (!write_to_buffer(serialized.data(), serialized.size())) {
        throw std::runtime_error("Failed to write message data");
    }

    // Update statistics
    header_->messages_sent.fetch_add(1, std::memory_order_relaxed);
    header_->bytes_sent.fetch_add(msg_header.size, std::memory_order_relaxed);

    stats_.messages_sent++;
    stats_.bytes_sent += msg_header.size;

    // Signal peer
    signal_peer();
}

*/

/*
bool LockFreeIPCChannel::receive(Message &message) {
    return try_receive(message, std::chrono::milliseconds(-1));
}

*/

/*
bool LockFreeIPCChannel::try_receive(Message &message,
                                     std::chrono::milliseconds timeout) {
    if (!is_connected_) {
        return false;
    }

    auto start = std::chrono::steady_clock::now();

    while (true) {
        // Check if data available
        if (available_read() >= sizeof(MessageHeader)) {
            MessageHeader msg_header;

            // Peek at header
            size_t saved_read =
                header_->read_pos.load(std::memory_order_acquire);
            if (!read_from_buffer(&msg_header, sizeof(msg_header))) {
                return false;
            }

            // Check if full message available
            if (available_read() >= msg_header.size - sizeof(MessageHeader)) {
                // Read message data
                std::vector<uint8_t> data(msg_header.size -
                                          sizeof(MessageHeader));
                if (!read_from_buffer(data.data(), data.size())) {
                    // Restore read position
                    header_->read_pos.store(saved_read,
                                            std::memory_order_release);
                    return false;
                }

                // Verify checksum
                uint32_t checksum = simd::SIMDChecksum::crc32(reinterpret_cast<const uint8_t*>(data.data()), data.size());
                if (checksum != msg_header.checksum) {
                    // Restore read position
                    header_->read_pos.store(saved_read,
                                            std::memory_order_release);
                    continue; // Skip corrupted message
                }

                // Deserialize message
                if (!message.deserialize(data)) {
                    continue; // Skip invalid message
                }

                // Update statistics
                header_->messages_received.fetch_add(1,
                                                     std::memory_order_relaxed);
                header_->bytes_received.fetch_add(msg_header.size,
                                                  std::memory_order_relaxed);

                stats_.messages_received++;
                stats_.bytes_received += msg_header.size;

                return true;
            } else {
                // Restore read position
                header_->read_pos.store(saved_read, std::memory_order_release);
            }
        }

        // Check timeout
        if (timeout.count() >= 0) {
            if (std::chrono::steady_clock::now() - start > timeout) {
                return false;
            }
        }

        // Wait for signal with remaining timeout
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto remaining =
            timeout -
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);

        if (!wait_for_signal(remaining)) {
            return false;
        }
    }
}
*/

bool LockFreeIPCChannel::is_connected() const {
    if (!is_connected_)
        return false;

    // Check if peer is still alive
    uint32_t peer_pid =
        is_server_ ? header_->client_pid.load(std::memory_order_acquire)
                   : header_->server_pid.load(std::memory_order_acquire);

    if (peer_pid == 0)
        return false;

    // Check if process is still alive
#ifdef _WIN32
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, peer_pid);
    if (hProcess != NULL) {
        DWORD exitCode;
        BOOL result = GetExitCodeProcess(hProcess, &exitCode);
        CloseHandle(hProcess);
        return result && exitCode == STILL_ACTIVE;
    }
    return false;
#else
    // On Unix, use kill(0) to check if process exists
    return kill(peer_pid, 0) == 0 || errno == EPERM;
#endif
}

void LockFreeIPCChannel::close() {
    if (!is_connected_)
        return;

    is_connected_ = false;

    // Mark as disconnected
    if (is_server_) {
        header_->server_ready.store(0, std::memory_order_release);
    } else {
        header_->client_ready.store(0, std::memory_order_release);
    }

    cleanup();
}

debug::ChannelMetrics LockFreeIPCChannel::get_stats() const {
    if (header_) {
        stats_.messages_sent =
            header_->messages_sent.load(std::memory_order_relaxed);
        stats_.messages_received =
            header_->messages_received.load(std::memory_order_relaxed);
        stats_.bytes_sent = header_->bytes_sent.load(std::memory_order_relaxed);
        stats_.bytes_received =
            header_->bytes_received.load(std::memory_order_relaxed);
    }
    return stats_;
}

std::string LockFreeIPCChannel::get_endpoint() const {
    std::stringstream ss;
    ss << "ipc://" << channel_name_;
    if (is_server_) {
        ss << " (server)";
    } else {
        ss << " (client)";
    }
    return ss.str();
}

void LockFreeIPCChannel::cleanup() {
    // Cleanup notifiers
#ifdef _WIN32
    if (send_notifier_.event != INVALID_HANDLE_VALUE) {
        CloseHandle(send_notifier_.event);
        send_notifier_.event = INVALID_HANDLE_VALUE;
    }
    if (recv_notifier_.event != INVALID_HANDLE_VALUE) {
        CloseHandle(recv_notifier_.event);
        recv_notifier_.event = INVALID_HANDLE_VALUE;
    }
#else
#ifdef __linux__
    if (send_notifier_.eventfd >= 0) {
        ::close(send_notifier_.eventfd);
        send_notifier_.eventfd = -1;
    }
    if (recv_notifier_.eventfd >= 0) {
        ::close(recv_notifier_.eventfd);
        recv_notifier_.eventfd = -1;
    }
#endif

#ifdef __APPLE__
    // Release dispatch semaphores on macOS
    if (send_notifier_.semaphore != nullptr) {
        dispatch_release(send_notifier_.semaphore);
        send_notifier_.semaphore = nullptr;
    }
    if (recv_notifier_.semaphore != nullptr) {
        dispatch_release(recv_notifier_.semaphore);
        recv_notifier_.semaphore = nullptr;
    }
#else
    // Clean up POSIX semaphores on Linux
    if (send_notifier_.semaphore != nullptr &&
        send_notifier_.semaphore != SEM_FAILED) {
        sem_close(send_notifier_.semaphore);
        if (is_server_) {
            sem_unlink(send_notifier_.sem_name.c_str());
        }
        send_notifier_.semaphore = nullptr;
    }
    if (recv_notifier_.semaphore != nullptr &&
        recv_notifier_.semaphore != SEM_FAILED) {
        sem_close(recv_notifier_.semaphore);
        if (is_server_) {
            sem_unlink(recv_notifier_.sem_name.c_str());
        }
        recv_notifier_.semaphore = nullptr;
    }
#endif
#endif

    // Cleanup shared memory
    unmap_shared_memory(shared_mem_);
    header_ = nullptr;
    buffer_ = nullptr;
}

bool LockFreeIPCChannel::write_to_buffer(const void *data, size_t size) {
    const uint8_t *src = static_cast<const uint8_t *>(data);
    const size_t buffer_size = config_.buffer_size;

    while (size > 0) {
        size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
        size_t read_pos = cached_read_pos_;

        // Check if we need to update cached read position
        if (write_pos - read_pos >= buffer_size) {
            read_pos = header_->read_cache.load(std::memory_order_acquire);
            cached_read_pos_ = read_pos;

            if (write_pos - read_pos >= buffer_size) {
                // Still full, need to wait
                return false;
            }
        }

        // Calculate available space
        size_t available = buffer_size - (write_pos - read_pos);
        if (available == 0)
            return false;

        // Calculate contiguous space
        size_t write_idx = write_pos % buffer_size;
        size_t chunk =
            std::min(size, std::min(available, buffer_size - write_idx));

        // Copy data - zero-copy compliant manual loop
        for (size_t i = 0; i < chunk; ++i) {
            buffer_[write_idx + i] = src[i];
        }

        // Update write position
        header_->write_pos.store(write_pos + chunk, std::memory_order_release);

        // Update cached write position for readers
        header_->write_cache.store(write_pos + chunk,
                                   std::memory_order_release);

        src += chunk;
        size -= chunk;
    }

    return true;
}

bool LockFreeIPCChannel::read_from_buffer(void *data, size_t size) {
    uint8_t *dst = static_cast<uint8_t *>(data);
    const size_t buffer_size = config_.buffer_size;

    while (size > 0) {
        size_t read_pos = header_->read_pos.load(std::memory_order_acquire);
        size_t write_pos = cached_write_pos_;

        // Check if we need to update cached write position
        if (write_pos <= read_pos) {
            write_pos = header_->write_cache.load(std::memory_order_acquire);
            cached_write_pos_ = write_pos;

            if (write_pos <= read_pos) {
                // Still empty
                return false;
            }
        }

        // Calculate available data
        size_t available = write_pos - read_pos;
        if (available == 0)
            return false;

        // Calculate contiguous data
        size_t read_idx = read_pos % buffer_size;
        size_t chunk =
            std::min(size, std::min(available, buffer_size - read_idx));

        // Copy data - zero-copy compliant manual loop
        for (size_t i = 0; i < chunk; ++i) {
            dst[i] = buffer_[read_idx + i];
        }

        // Update read position
        header_->read_pos.store(read_pos + chunk, std::memory_order_release);

        // Update cached read position for writers
        header_->read_cache.store(read_pos + chunk, std::memory_order_release);

        dst += chunk;
        size -= chunk;
    }

    return true;
}

size_t LockFreeIPCChannel::available_read() const {
    size_t read_pos = header_->read_pos.load(std::memory_order_acquire);
    size_t write_pos = header_->write_cache.load(std::memory_order_acquire);
    return write_pos > read_pos ? write_pos - read_pos : 0;
}

size_t LockFreeIPCChannel::available_write() const {
    size_t write_pos = header_->write_pos.load(std::memory_order_acquire);
    size_t read_pos = header_->read_cache.load(std::memory_order_acquire);
    size_t used = write_pos - read_pos;
    return used < config_.buffer_size ? config_.buffer_size - used : 0;
}

void LockFreeIPCChannel::signal_peer() {
#ifdef _WIN32
    SetEvent(send_notifier_.event);
#else
#ifdef __linux__
    if (send_notifier_.eventfd >= 0) {
        uint64_t value = 1;
        write(send_notifier_.eventfd, &value, sizeof(value));
        return;
    }
#endif
#ifdef __APPLE__
    if (send_notifier_.semaphore != nullptr) {
        dispatch_semaphore_signal(send_notifier_.semaphore);
    }
#else
    if (send_notifier_.semaphore != nullptr) {
        sem_post(send_notifier_.semaphore);
    }
#endif
#endif
}

bool LockFreeIPCChannel::wait_for_signal(std::chrono::milliseconds timeout) {
#ifdef _WIN32
    DWORD ms =
        timeout.count() < 0 ? INFINITE : static_cast<DWORD>(timeout.count());
    return WaitForSingleObject(recv_notifier_.event, ms) == WAIT_OBJECT_0;
#else
#ifdef __linux__
    if (recv_notifier_.eventfd >= 0) {
        struct pollfd pfd;
        pfd.fd = recv_notifier_.eventfd;
        pfd.events = POLLIN;

        int ms = timeout.count() < 0 ? -1 : static_cast<int>(timeout.count());
        int ret = poll(&pfd, 1, ms);

        if (ret > 0 && (pfd.revents & POLLIN)) {
            uint64_t value;
            read(recv_notifier_.eventfd, &value, sizeof(value));
            return true;
        }
        return false;
    }
#endif

    if (recv_notifier_.semaphore != nullptr) {
#ifdef __APPLE__
        // Use dispatch semaphore on macOS
        if (timeout.count() < 0) {
            return dispatch_semaphore_wait(recv_notifier_.semaphore, DISPATCH_TIME_FOREVER) == 0;
        } else {
            dispatch_time_t dtimeout = dispatch_time(DISPATCH_TIME_NOW, 
                timeout.count() * NSEC_PER_MSEC);
            return dispatch_semaphore_wait(recv_notifier_.semaphore, dtimeout) == 0;
        }
#else
        // Use POSIX semaphore on Linux
        if (timeout.count() < 0) {
            return sem_wait(recv_notifier_.semaphore) == 0;
        } else {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += timeout.count() / 1000;
            ts.tv_nsec += (timeout.count() % 1000) * 1000000;
            if (ts.tv_nsec >= 1000000000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000;
            }
            return sem_timedwait(recv_notifier_.semaphore, &ts) == 0;
        }
#endif
    }

    return false;
#endif
}

void *LockFreeIPCChannel::map_shared_memory(const std::string &name,
                                            size_t size, bool create,
                                            SharedMemory &shm) {
#ifdef _WIN32
    DWORD access = FILE_MAP_ALL_ACCESS;

    if (create) {
        LARGE_INTEGER li;
        li.QuadPart = size;
        shm.mapping =
            CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                               li.HighPart, li.LowPart, name.c_str());
    } else {
        shm.mapping = OpenFileMappingA(access, FALSE, name.c_str());
    }

    if (!shm.mapping)
        return nullptr;

    shm.ptr = MapViewOfFile(shm.mapping, access, 0, 0, size);
    shm.size = size;

    return shm.ptr;
#else
    int flags = O_RDWR;
    if (create)
        flags |= O_CREAT | O_EXCL;

    shm.name = name;
    shm.fd = shm_open(name.c_str(), flags, 0666);
    if (shm.fd < 0)
        return nullptr;

    if (create) {
        if (ftruncate(shm.fd, size) < 0) {
            ::close(shm.fd);
            shm_unlink(name.c_str());
            return nullptr;
        }
    }

    int mmap_flags = MAP_SHARED;

#ifdef MAP_HUGETLB
    // Try to use huge pages if available
    if (size >= 2 * 1024 * 1024) {
        void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         mmap_flags | MAP_HUGETLB, shm.fd, 0);
        if (ptr != MAP_FAILED) {
            shm.ptr = ptr;
            shm.size = size;
            return ptr;
        }
    }
#endif

    // Regular mapping
    shm.ptr =
        mmap(nullptr, size, PROT_READ | PROT_WRITE, mmap_flags, shm.fd, 0);
    if (shm.ptr == MAP_FAILED) {
        ::close(shm.fd);
        if (create)
            shm_unlink(name.c_str());
        return nullptr;
    }

    shm.size = size;
    return shm.ptr;
#endif
}

void LockFreeIPCChannel::unmap_shared_memory(SharedMemory &shm) {
#ifdef _WIN32
    if (shm.ptr) {
        UnmapViewOfFile(shm.ptr);
        shm.ptr = nullptr;
    }
    if (shm.mapping != INVALID_HANDLE_VALUE) {
        CloseHandle(shm.mapping);
        shm.mapping = INVALID_HANDLE_VALUE;
    }
#else
    if (shm.ptr && shm.ptr != MAP_FAILED) {
        munmap(shm.ptr, shm.size);
        shm.ptr = nullptr;
    }
    if (shm.fd >= 0) {
        ::close(shm.fd);
        shm.fd = -1;
    }
    // Don't unlink in static method - let destructor handle it
    // if (!shm.name.empty() && is_server_) {
    //     shm_unlink(shm.name.c_str());
    // }
#endif
}

void LockFreeIPCChannel::set_cpu_affinity(int cpu) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#elif defined(__APPLE__)
    // macOS doesn't support thread affinity directly
    // Could use thread_policy_set with THREAD_AFFINITY_POLICY
#elif defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << cpu);
#endif
}

} // namespace psyne