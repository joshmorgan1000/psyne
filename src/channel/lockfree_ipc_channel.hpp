/**
 * @file lockfree_ipc_channel.hpp
 * @brief Lock-free IPC channel implementation
 *
 * High-performance inter-process communication using shared memory
 * with lock-free ring buffers and memory-mapped files.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include "../memory/ring_buffer_impl.hpp"
#include "channel_impl.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace psyne {

/**
 * @brief Lock-free IPC channel using shared memory
 *
 * Features:
 * - True zero-copy through shared memory mapping
 * - Lock-free SPSC/MPSC ring buffers
 * - Eventfd/semaphore signaling for low-latency wakeup
 * - Robust cleanup and crash recovery
 */
class LockFreeIPCChannel : public Channel {
public:
    /**
     * @brief IPC configuration
     */
    struct Config {
        size_t buffer_size;
        bool use_huge_pages;
        bool use_eventfd;
        size_t max_message_size;
        std::string namespace_prefix;
        int cpu_affinity;
        
        Config() : buffer_size(64 * 1024 * 1024), // 64MB default
                   use_huge_pages(true),           // Use 2MB huge pages
                   use_eventfd(true),              // Use eventfd for signaling (Linux)
                   max_message_size(16 * 1024 * 1024), // 16MB max message
                   namespace_prefix("/psyne_ipc_"),
                   cpu_affinity(-1) {} // CPU to pin to (-1 for none)
    };

    /**
     * @brief Create server-side IPC channel
     */
    static std::unique_ptr<LockFreeIPCChannel>
    create_server(const std::string &name, const Config &config = {});

    /**
     * @brief Create client-side IPC channel
     */
    static std::unique_ptr<LockFreeIPCChannel>
    create_client(const std::string &name, const Config &config = {});

    ~LockFreeIPCChannel() override;

    // Channel interface
    template <typename T>
    void send(const Message<T> &message);
    
    template <typename T>
    bool receive(Message<T> &message);
    
    template <typename T>
    bool try_receive(Message<T> &message,
                     std::chrono::milliseconds timeout);
                     
    bool is_connected() const;
    void close();
    debug::ChannelMetrics get_stats() const;
    std::string get_endpoint() const;

protected:
    LockFreeIPCChannel(const std::string &name, bool is_server,
                       const Config &config);

private:
    // Shared memory layout
    struct SharedHeader {
        // Magic number for validation
        static constexpr uint64_t MAGIC = 0x50534E45495043ULL; // "PSYNEIPC"

        std::atomic<uint64_t> magic;
        std::atomic<uint32_t> version;
        std::atomic<uint32_t> flags;

        // Ring buffer metadata
        std::atomic<size_t> buffer_size;
        std::atomic<size_t> write_pos;
        std::atomic<size_t> read_pos;
        std::atomic<size_t> write_cache; // Cached write position
        std::atomic<size_t> read_cache;  // Cached read position

        // Connection state
        std::atomic<uint32_t> server_pid;
        std::atomic<uint32_t> client_pid;
        std::atomic<uint64_t> server_ready;
        std::atomic<uint64_t> client_ready;

        // Statistics
        std::atomic<uint64_t> messages_sent;
        std::atomic<uint64_t> messages_received;
        std::atomic<uint64_t> bytes_sent;
        std::atomic<uint64_t> bytes_received;

        // Padding to cache line
        char padding[64 - (sizeof(std::atomic<uint64_t>) * 4) % 64];
    };

    // Message header in ring buffer
    struct MessageHeader {
        uint32_t size;      // Total message size including header
        uint32_t type_hash; // Type hash for validation
        uint64_t timestamp; // Timestamp for ordering
        uint32_t checksum;  // Optional checksum
        uint32_t flags;     // Message flags
    };

    // Platform-specific shared memory handle
    struct SharedMemory {
        void *ptr = nullptr;
        size_t size = 0;
#ifdef _WIN32
        HANDLE mapping = INVALID_HANDLE_VALUE;
#else
        int fd = -1;
        std::string name;
#endif
    };

    // Platform-specific notification mechanism
    struct Notifier {
#ifdef _WIN32
        HANDLE event = INVALID_HANDLE_VALUE;
#elif defined(__APPLE__)
        int eventfd = -1;           // For kqueue/pipes
        dispatch_semaphore_t semaphore = nullptr; // Dispatch semaphore for macOS
        std::string sem_name;
#else
        int eventfd = -1;           // For Linux eventfd
        sem_t *semaphore = nullptr; // Fallback to POSIX semaphore
        std::string sem_name;
#endif
    };

    // Internal methods
    bool init_shared_memory();
    bool init_notifier();
    bool wait_for_peer(std::chrono::milliseconds timeout);
    void cleanup();

    // Ring buffer operations
    bool write_to_buffer(const void *data, size_t size);
    bool read_from_buffer(void *data, size_t size);
    size_t available_read() const;
    size_t available_write() const;

    // Signaling
    void signal_peer();
    bool wait_for_signal(std::chrono::milliseconds timeout);

    // Memory mapping helpers
    static void *map_shared_memory(const std::string &name, size_t size,
                                   bool create, SharedMemory &shm);
    static void unmap_shared_memory(SharedMemory &shm);

    // Member variables
    Config config_;
    std::string channel_name_;
    bool is_server_;
    bool is_connected_;

    SharedMemory shared_mem_;
    SharedHeader *header_;
    uint8_t *buffer_;

    Notifier send_notifier_;
    Notifier recv_notifier_;

    // Local caches to reduce atomic operations
    mutable size_t cached_read_pos_ = 0;
    mutable size_t cached_write_pos_ = 0;

    // Statistics
    mutable debug::ChannelMetrics stats_;

    // CPU affinity
    void set_cpu_affinity(int cpu);
};

} // namespace psyne