#pragma once

/**
 * @file channel.hpp
 * @brief Unified channel system - replaces all previous channel implementations
 * 
 * This single file replaces:
 * - channel_base.hpp, spsc_channel.hpp, mpsc_channel.hpp, spmc_channel.hpp, mpmc_channel.hpp
 * - ipc_channel.hpp, tcp_channel.hpp, gpu_channel.hpp, debug_channel.hpp
 * - All corresponding .cpp files
 * 
 * Everything is now compile-time optimized via Channel<MessageType, Substrate, Pattern>
 */

#include "core/message.hpp"
#include "global/logger.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <vector>
#include <type_traits>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <boost/asio.hpp>

#ifdef PSYNE_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace psyne {

// Forward declarations
template<typename T, class Substrate, class Pattern> class Channel;
template<typename T, class Substrate, class Pattern> class Message;

/**
 * @brief Substrate implementations for different transport mechanisms
 */
namespace substrate {

    /**
     * @brief Pure in-process memory slab - fastest possible
     */
    class InProcess {
    public:
        template<typename T>
        static T* allocate_slab(size_t size_bytes) {
            return static_cast<T*>(std::aligned_alloc(alignof(T), size_bytes));
        }
        
        template<typename T>
        static void deallocate_slab(T* ptr) {
            std::free(ptr);
        }
        
        template<typename T>
        static void send_message(T* msg_ptr, std::vector<std::function<void(T*)>>& listeners) {
            // Direct notification - zero latency
            for (auto& listener : listeners) {
                listener(msg_ptr);
            }
        }
        
        static constexpr bool needs_serialization = false;
        static constexpr bool is_zero_copy = true;
        static constexpr bool cross_process = false;
    };

    /**
     * @brief IPC via shared memory - zero-copy across processes  
     */
    class IPC {
    public:
        template<typename T>
        static T* allocate_slab(size_t size_bytes) {
            // TODO: Implement POSIX shared memory
            return static_cast<T*>(std::aligned_alloc(alignof(T), size_bytes));
        }
        
        template<typename T>
        static void deallocate_slab(T* ptr) {
            std::free(ptr);
        }
        
        template<typename T>
        static void send_message(T* msg_ptr, std::vector<std::function<void(T*)>>& listeners) {
            // TODO: Signal across process boundaries
            for (auto& listener : listeners) {
                listener(msg_ptr);
            }
        }
        
        static constexpr bool needs_serialization = false;
        static constexpr bool is_zero_copy = true;
        static constexpr bool cross_process = true;
    };

    /**
     * @brief TCP network transport with boost::asio
     */
    class TCP {
    private:
        static inline boost::asio::io_context io_context_;
        static inline std::unique_ptr<boost::asio::ip::tcp::acceptor> acceptor_;
        static inline std::unique_ptr<boost::asio::ip::tcp::socket> socket_;
        static inline std::thread io_thread_;
        static inline bool initialized_ = false;
        static inline std::mutex init_mutex_;
        
    public:
        template<typename T>
        static T* allocate_slab(size_t size_bytes) {
            return static_cast<T*>(std::aligned_alloc(alignof(T), size_bytes));
        }
        
        template<typename T>
        static void deallocate_slab(T* ptr) {
            std::free(ptr);
        }
        
        template<typename T>
        static void send_message(T* msg_ptr, std::vector<std::function<void(T*)>>& listeners) {
            // Initialize TCP if needed
            if (!initialized_) {
                std::lock_guard<std::mutex> lock(init_mutex_);
                if (!initialized_) {
                    initialize_tcp();
                }
            }
            
            if (socket_ && socket_->is_open()) {
                try {
                    // Serialize message data
                    boost::asio::async_write(*socket_,
                        boost::asio::buffer(msg_ptr, sizeof(T)),
                        [](boost::system::error_code ec, std::size_t bytes_transferred) {
                            if (ec) {
                                LOG_ERROR("TCP send error: {}", ec.message());
                            } else {
                                LOG_DEBUG("TCP sent {} bytes", bytes_transferred);
                            }
                        });
                } catch (const std::exception& e) {
                    LOG_ERROR("TCP send exception: {}", e.what());
                }
            }
            
            // Also notify local listeners
            for (auto& listener : listeners) {
                listener(msg_ptr);
            }
        }
        
        static void initialize_tcp(const std::string& host = "localhost", uint16_t port = 8080) {
            try {
                // Create acceptor for server mode
                acceptor_ = std::make_unique<boost::asio::ip::tcp::acceptor>(
                    io_context_, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
                
                // Start accepting connections
                start_accept();
                
                // Start io_context in separate thread
                io_thread_ = std::thread([]() {
                    io_context_.run();
                });
                
                initialized_ = true;
                LOG_INFO("TCP substrate initialized on port {}", port);
                
            } catch (const std::exception& e) {
                LOG_ERROR("TCP initialization failed: {}", e.what());
            }
        }
        
        static void start_accept() {
            socket_ = std::make_unique<boost::asio::ip::tcp::socket>(io_context_);
            
            acceptor_->async_accept(*socket_,
                [](boost::system::error_code ec) {
                    if (!ec) {
                        LOG_INFO("TCP client connected");
                        start_receive();
                        start_accept(); // Accept next connection
                    } else {
                        LOG_ERROR("TCP accept error: {}", ec.message());
                    }
                });
        }
        
        template<typename T>
        static void start_receive() {
            auto buffer = std::make_shared<std::array<char, 4096>>();
            
            socket_->async_read_some(boost::asio::buffer(*buffer),
                [buffer](boost::system::error_code ec, std::size_t bytes_transferred) {
                    if (!ec) {
                        LOG_DEBUG("TCP received {} bytes", bytes_transferred);
                        // TODO: Deserialize and process received data
                        start_receive<T>(); // Continue receiving
                    } else {
                        LOG_ERROR("TCP receive error: {}", ec.message());
                    }
                });
        }
        
        static void shutdown() {
            if (initialized_) {
                io_context_.stop();
                if (io_thread_.joinable()) {
                    io_thread_.join();
                }
                socket_.reset();
                acceptor_.reset();
                initialized_ = false;
            }
        }
        
        static constexpr bool needs_serialization = true;
        static constexpr bool is_zero_copy = false;
        static constexpr bool cross_process = true;
    };

    /**
     * @brief GPU memory substrate - unified memory or GPU-accessible
     */
    class GPU {
    public:
        template<typename T>
        static T* allocate_slab(size_t size_bytes) {
#ifdef PSYNE_CUDA_ENABLED
            T* ptr = nullptr;
            // Try to allocate unified memory first (accessible from both CPU and GPU)
            cudaError_t err = cudaMallocManaged(&ptr, size_bytes);
            if (err == cudaSuccess) {
                return ptr;
            }
            
            // Fall back to host memory if unified memory fails
            err = cudaMallocHost(&ptr, size_bytes);
            if (err == cudaSuccess) {
                return ptr;
            }
            
            // Last resort: regular aligned allocation
            LOG_WARN("CUDA memory allocation failed, falling back to host memory");
#endif
            return static_cast<T*>(std::aligned_alloc(alignof(T), size_bytes));
        }
        
        template<typename T>
        static void deallocate_slab(T* ptr) {
#ifdef PSYNE_CUDA_ENABLED
            // Check if this is CUDA memory
            cudaPointerAttributes attrs;
            cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
            if (err == cudaSuccess) {
                if (attrs.type == cudaMemoryTypeManaged || 
                    attrs.type == cudaMemoryTypeHost) {
                    cudaFree(ptr);
                    return;
                }
            }
#endif
            std::free(ptr);
        }
        
        template<typename T>
        static void send_message(T* msg_ptr, std::vector<std::function<void(T*)>>& listeners) {
#ifdef PSYNE_CUDA_ENABLED
            // Ensure data is synchronized to host if needed
            cudaDeviceSynchronize();
#endif
            // Direct notification - GPU memory is accessible from CPU
            for (auto& listener : listeners) {
                listener(msg_ptr);
            }
        }
        
        static constexpr bool needs_serialization = false;
        static constexpr bool is_zero_copy = true;
        static constexpr bool cross_process = false;
    };

    /**
     * @brief Debug substrate - adds instrumentation and logging
     */
    class Debug {
    private:
        static inline std::atomic<size_t> allocation_count_{0};
        static inline std::atomic<size_t> send_count_{0};
        static inline std::atomic<size_t> total_bytes_allocated_{0};
        
    public:
        template<typename T>
        static T* allocate_slab(size_t size_bytes) {
            T* ptr = static_cast<T*>(std::aligned_alloc(alignof(T), size_bytes));
            if (ptr) {
                allocation_count_.fetch_add(1);
                total_bytes_allocated_.fetch_add(size_bytes);
                LOG_DEBUG("Debug: Allocated slab {} bytes at {:p} (total allocations: {}, total bytes: {})", 
                         size_bytes, static_cast<void*>(ptr), 
                         allocation_count_.load(), total_bytes_allocated_.load());
            } else {
                LOG_ERROR("Debug: Failed to allocate slab {} bytes", size_bytes);
            }
            return ptr;
        }
        
        template<typename T>
        static void deallocate_slab(T* ptr) {
            if (ptr) {
                LOG_DEBUG("Debug: Deallocating slab at {:p}", static_cast<void*>(ptr));
                std::free(ptr);
                allocation_count_.fetch_sub(1);
            }
        }
        
        template<typename T>
        static void send_message(T* msg_ptr, std::vector<std::function<void(T*)>>& listeners) {
            size_t count = send_count_.fetch_add(1) + 1;
            LOG_DEBUG("Debug: Sending message #{} at {:p} to {} listeners", 
                     count, static_cast<void*>(msg_ptr), listeners.size());
            
            auto start = std::chrono::high_resolution_clock::now();
            
            for (size_t i = 0; i < listeners.size(); ++i) {
                LOG_DEBUG("Debug: Notifying listener {}", i);
                listeners[i](msg_ptr);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            LOG_DEBUG("Debug: Message notification took {} μs", duration.count());
        }
        
        static constexpr bool needs_serialization = false;
        static constexpr bool is_zero_copy = true;
        static constexpr bool cross_process = false;
        
        // Debug-specific statistics
        static size_t get_allocation_count() { return allocation_count_.load(); }
        static size_t get_send_count() { return send_count_.load(); }
        static size_t get_total_bytes_allocated() { return total_bytes_allocated_.load(); }
        static void reset_stats() {
            allocation_count_ = 0;
            send_count_ = 0;
            total_bytes_allocated_ = 0;
        }
    };

} // namespace substrate

/**
 * @brief Pattern implementations for different producer/consumer configurations
 */
namespace pattern {

    /**
     * @brief Single Producer Single Consumer - lock-free ring buffer
     */
    class SPSC {
    private:
        static constexpr size_t CACHE_LINE_SIZE = 64;
        
    public:
        template<typename T, class Substrate>
        class Implementation {
        public:
            explicit Implementation(size_t ring_size) : ring_size_(ring_size) {
                if ((ring_size & (ring_size - 1)) != 0) {
                    throw std::invalid_argument("Ring size must be power of 2");
                }
                ring_mask_ = ring_size - 1;
                ring_.resize(ring_size);
            }
            
            T* try_allocate(T* slab, size_t max_messages) {
                size_t head = head_.load(std::memory_order_relaxed);
                size_t next_head = head + 1;
                
                if (next_head - tail_.load(std::memory_order_acquire) > ring_size_) {
                    return nullptr; // Ring full
                }
                
                size_t slab_pos = head % max_messages;
                T* slot = &slab[slab_pos];
                
                // Store in ring for consumer
                ring_[head & ring_mask_].store(slot, std::memory_order_relaxed);
                head_.store(next_head, std::memory_order_release);
                
                return slot;
            }
            
            T* try_receive() {
                size_t tail = tail_.load(std::memory_order_relaxed);
                
                if (tail >= head_.load(std::memory_order_acquire)) {
                    return nullptr; // Ring empty
                }
                
                T* msg = ring_[tail & ring_mask_].load(std::memory_order_relaxed);
                if (!msg) return nullptr;
                
                ring_[tail & ring_mask_].store(nullptr, std::memory_order_relaxed);
                tail_.store(tail + 1, std::memory_order_release);
                
                return msg;
            }
            
        private:
            size_t ring_size_;
            size_t ring_mask_;
            std::vector<std::atomic<T*>> ring_;
            
            alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
            alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};
        };
        
        static constexpr bool needs_locks = false;
        static constexpr size_t default_ring_size = 1024;
    };

    /**
     * @brief Multiple Producer Single Consumer - per-producer slots
     */
    class MPSC {
    private:
        static constexpr size_t CACHE_LINE_SIZE = 64;
        static constexpr size_t MAX_PRODUCERS = 64;
        
    public:
        template<typename T>
        struct ProducerSlot {
            alignas(CACHE_LINE_SIZE) std::atomic<size_t> head{0};
            alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail{0};
            std::vector<std::atomic<T*>> ring;
            size_t ring_mask;
            
            ProducerSlot(size_t ring_size) : ring_mask(ring_size - 1) {
                ring.resize(ring_size);
            }
        };
        
        template<typename T, class Substrate>
        class Implementation {
        public:
            explicit Implementation(size_t ring_size) : ring_size_(ring_size) {
                if ((ring_size & (ring_size - 1)) != 0) {
                    throw std::invalid_argument("Ring size must be power of 2");
                }
                
                // Pre-allocate producer slots
                producer_slots_.reserve(MAX_PRODUCERS);
                
                LOG_INFO("MPSC pattern implementation created with ring size {}", ring_size);
            }
            
            size_t register_producer() {
                std::lock_guard<std::mutex> lock(producer_mutex_);
                if (producer_slots_.size() >= MAX_PRODUCERS) {
                    throw std::runtime_error("Maximum number of producers exceeded");
                }
                
                size_t producer_id = producer_slots_.size();
                producer_slots_.emplace_back(std::make_unique<ProducerSlot<T>>(ring_size_));
                
                LOG_DEBUG("Registered producer {} (total: {})", producer_id, producer_slots_.size());
                return producer_id;
            }
            
            T* try_allocate(T* slab, size_t max_messages, size_t producer_id = 0) {
                if (producer_id >= producer_slots_.size()) {
                    // Auto-register producer if needed
                    std::lock_guard<std::mutex> lock(producer_mutex_);
                    while (producer_slots_.size() <= producer_id) {
                        producer_slots_.emplace_back(std::make_unique<ProducerSlot<T>>(ring_size_));
                    }
                }
                
                auto& slot = *producer_slots_[producer_id];
                size_t head = slot.head.load(std::memory_order_relaxed);
                size_t next_head = head + 1;
                
                // Check if this producer's slot is full
                if (next_head - slot.tail.load(std::memory_order_acquire) > ring_size_) {
                    return nullptr;
                }
                
                // Allocate from slab using global position
                size_t global_pos = next_pos_.fetch_add(1) % max_messages;
                T* msg_ptr = &slab[global_pos];
                
                // Store in producer's ring
                slot.ring[head & slot.ring_mask].store(msg_ptr, std::memory_order_relaxed);
                slot.head.store(next_head, std::memory_order_release);
                
                return msg_ptr;
            }
            
            T* try_receive() {
                // Round-robin through producer slots
                size_t start_producer = last_producer_checked_.load();
                
                for (size_t i = 0; i < producer_slots_.size(); ++i) {
                    size_t producer_id = (start_producer + i) % producer_slots_.size();
                    auto& slot = *producer_slots_[producer_id];
                    
                    size_t tail = slot.tail.load(std::memory_order_relaxed);
                    if (tail < slot.head.load(std::memory_order_acquire)) {
                        T* msg = slot.ring[tail & slot.ring_mask].load(std::memory_order_relaxed);
                        if (msg) {
                            slot.ring[tail & slot.ring_mask].store(nullptr, std::memory_order_relaxed);
                            slot.tail.store(tail + 1, std::memory_order_release);
                            
                            last_producer_checked_.store((producer_id + 1) % producer_slots_.size());
                            return msg;
                        }
                    }
                }
                
                return nullptr; // No messages from any producer
            }
            
            size_t get_producer_count() const {
                std::lock_guard<std::mutex> lock(producer_mutex_);
                return producer_slots_.size();
            }
            
        private:
            size_t ring_size_;
            std::vector<std::unique_ptr<ProducerSlot<T>>> producer_slots_;
            mutable std::mutex producer_mutex_;
            
            std::atomic<size_t> next_pos_{0};
            std::atomic<size_t> last_producer_checked_{0};
        };
        
        static constexpr bool needs_locks = false; // Lock-free per-producer
        static constexpr size_t default_ring_size = 1024;
    };

    /**
     * @brief Single Producer Multiple Consumer - broadcast semantics with async support
     */
    class SPMC {
    private:
        static constexpr size_t CACHE_LINE_SIZE = 64;
        
    public:
        template<typename T, class Substrate>
        class Implementation {
        public:
            explicit Implementation(size_t ring_size) 
                : ring_size_(ring_size), io_context_(nullptr) {
                if ((ring_size & (ring_size - 1)) != 0) {
                    throw std::invalid_argument("Ring size must be power of 2");
                }
                ring_mask_ = ring_size - 1;
                ring_.resize(ring_size);
                consumer_positions_.resize(8, 0); // Support up to 8 consumers initially
                LOG_INFO("SPMC pattern implementation created with ring size {}", ring_size);
            }
            
            // Enable async support
            void set_io_context(boost::asio::io_context* io_ctx) {
                io_context_ = io_ctx;
            }
            
            T* try_allocate(T* slab, size_t max_messages) {
                size_t head = head_.load(std::memory_order_relaxed);
                size_t next_head = head + 1;
                
                // Check if any consumer is too far behind
                size_t min_consumer_pos = get_min_consumer_position();
                if (next_head - min_consumer_pos > ring_size_) {
                    return nullptr; // Ring would overflow slowest consumer
                }
                
                size_t slab_pos = head % max_messages;
                T* slot = &slab[slab_pos];
                
                // Store in ring for broadcast to all consumers
                ring_[head & ring_mask_].store(slot, std::memory_order_relaxed);
                
                // Atomically update head and notify waiting consumers
                head_.store(next_head, std::memory_order_release);
                notify_consumers();
                
                return slot;
            }
            
            T* try_receive(size_t consumer_id = 0) {
                if (consumer_id >= consumer_positions_.size()) {
                    consumer_positions_.resize(consumer_id + 1, 0);
                }
                
                size_t consumer_pos = consumer_positions_[consumer_id].load(std::memory_order_relaxed);
                size_t current_head = head_.load(std::memory_order_acquire);
                
                if (consumer_pos >= current_head) {
                    return nullptr; // No new messages for this consumer
                }
                
                T* msg = ring_[consumer_pos & ring_mask_].load(std::memory_order_relaxed);
                if (msg) {
                    consumer_positions_[consumer_id].store(consumer_pos + 1, std::memory_order_release);
                }
                
                return msg;
            }
            
            // Async receive with boost::asio awaitable
            boost::asio::awaitable<T*> async_receive(size_t consumer_id = 0) {
                if (!io_context_) {
                    throw std::runtime_error("async_receive requires io_context");
                }
                
                while (true) {
                    T* msg = try_receive(consumer_id);
                    if (msg) {
                        co_return msg;
                    }
                    
                    // Wait for notification
                    boost::asio::steady_timer timer(*io_context_);
                    timer.expires_after(std::chrono::milliseconds(1));
                    co_await timer.async_wait(boost::asio::use_awaitable);
                }
            }
            
            size_t register_consumer() {
                return consumer_count_.fetch_add(1);
            }
            
        private:
            size_t ring_size_;
            size_t ring_mask_;
            std::vector<std::atomic<T*>> ring_;
            
            alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
            std::vector<std::atomic<size_t>> consumer_positions_;
            std::atomic<size_t> consumer_count_{0};
            
            boost::asio::io_context* io_context_;
            
            size_t get_min_consumer_position() const {
                size_t min_pos = SIZE_MAX;
                for (const auto& pos : consumer_positions_) {
                    min_pos = std::min(min_pos, pos.load(std::memory_order_relaxed));
                }
                return min_pos == SIZE_MAX ? 0 : min_pos;
            }
            
            void notify_consumers() {
                // In a real implementation, this would notify waiting async consumers
                // For now, they poll with short timeouts
            }
        };
        
        static constexpr bool needs_locks = false;
        static constexpr size_t default_ring_size = 1024;
    };

    /**
     * @brief Multiple Producer Multiple Consumer - work stealing with locks
     */
    class MPMC {
    private:
        static constexpr size_t CACHE_LINE_SIZE = 64;
        
    public:
        template<typename T, class Substrate>
        class Implementation {
        public:
            explicit Implementation(size_t ring_size) : ring_size_(ring_size) {
                if ((ring_size & (ring_size - 1)) != 0) {
                    throw std::invalid_argument("Ring size must be power of 2");
                }
                ring_mask_ = ring_size - 1;
                ring_.resize(ring_size);
                
                LOG_INFO("MPMC pattern implementation created with ring size {}", ring_size);
            }
            
            T* try_allocate(T* slab, size_t max_messages) {
                std::unique_lock<std::mutex> lock(producer_mutex_, std::try_to_lock);
                if (!lock.owns_lock()) {
                    return nullptr; // Another producer is allocating
                }
                
                size_t head = head_.load(std::memory_order_relaxed);
                size_t next_head = head + 1;
                
                // Check if ring would be full
                if (next_head - tail_.load(std::memory_order_acquire) > ring_size_) {
                    return nullptr;
                }
                
                // Allocate from slab
                size_t slab_pos = head % max_messages;
                T* slot = &slab[slab_pos];
                
                // Store in ring buffer
                ring_[head & ring_mask_].store(slot, std::memory_order_relaxed);
                head_.store(next_head, std::memory_order_release);
                
                // Notify waiting consumers
                consumer_cv_.notify_one();
                
                return slot;
            }
            
            T* try_receive() {
                std::unique_lock<std::mutex> lock(consumer_mutex_, std::try_to_lock);
                if (!lock.owns_lock()) {
                    return nullptr; // Another consumer is receiving
                }
                
                size_t tail = tail_.load(std::memory_order_relaxed);
                if (tail >= head_.load(std::memory_order_acquire)) {
                    return nullptr; // Ring empty
                }
                
                T* msg = ring_[tail & ring_mask_].load(std::memory_order_relaxed);
                if (!msg) {
                    return nullptr;
                }
                
                ring_[tail & ring_mask_].store(nullptr, std::memory_order_relaxed);
                tail_.store(tail + 1, std::memory_order_release);
                
                return msg;
            }
            
            // Blocking receive for MPMC
            T* receive_blocking(std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
                std::unique_lock<std::mutex> lock(consumer_mutex_);
                
                auto deadline = std::chrono::steady_clock::now() + timeout;
                
                while (true) {
                    size_t tail = tail_.load(std::memory_order_relaxed);
                    if (tail < head_.load(std::memory_order_acquire)) {
                        T* msg = ring_[tail & ring_mask_].load(std::memory_order_relaxed);
                        if (msg) {
                            ring_[tail & ring_mask_].store(nullptr, std::memory_order_relaxed);
                            tail_.store(tail + 1, std::memory_order_release);
                            return msg;
                        }
                    }
                    
                    // Wait for notification or timeout
                    if (consumer_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                        return nullptr;
                    }
                }
            }
            
            size_t size() const {
                return head_.load(std::memory_order_acquire) - 
                       tail_.load(std::memory_order_acquire);
            }
            
            bool empty() const {
                return size() == 0;
            }
            
            bool full() const {
                return size() >= ring_size_;
            }
            
        private:
            size_t ring_size_;
            size_t ring_mask_;
            std::vector<std::atomic<T*>> ring_;
            
            alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
            alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};
            
            // MPMC requires synchronization
            std::mutex producer_mutex_;
            std::mutex consumer_mutex_;
            std::condition_variable consumer_cv_;
        };
        
        static constexpr bool needs_locks = true;
        static constexpr size_t default_ring_size = 1024;
    };

} // namespace pattern

/**
 * @brief Unified channel template - replaces ALL previous channel classes
 * 
 * @tparam T Message type (e.g., Float32VectorMessage<64>)
 * @tparam S Substrate (InProcess, IPC, TCP)
 * @tparam P Pattern (SPSC, MPSC, SPMC, MPMC)
 */
template<typename T, 
         class S = substrate::InProcess, 
         class P = pattern::SPSC>
class Channel {
public:
    static_assert(std::is_trivially_copyable_v<T>, 
                 "Message types must be trivially copyable");

    using Substrate = S;
    using Pattern = P;
    using PatternImpl = typename P::template Implementation<T, S>;

    /**
     * @brief Create channel with memory slab
     */
    explicit Channel(uint16_t size_factor = 0, size_t ring_size = P::default_ring_size) 
        : size_factor_(size_factor), pattern_impl_(ring_size) {
        
        // Allocate typed memory slab
        size_t slab_size = (size_factor + 1) * 32 * 1024 * 1024;
        slab_ = S::template allocate_slab<T>(slab_size);
        if (!slab_) {
            throw std::bad_alloc();
        }
        
        max_messages_ = slab_size / sizeof(T);
        
        LOG_INFO("Created Channel<{}, {}, {}> with {}MB slab ({} messages)", 
                 typeid(T).name(), substrate_name(), pattern_name(),
                 slab_size / (1024 * 1024), max_messages_);
    }

    ~Channel() {
        if (slab_) {
            S::template deallocate_slab<T>(slab_);
        }
    }

    // Non-copyable
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    /**
     * @brief Allocate message using pattern-specific logic
     */
    T* allocate_next() {
        return pattern_impl_.try_allocate(slab_, max_messages_);
    }

    /**
     * @brief Receive message using pattern-specific logic
     */
    T* receive_next() {
        return pattern_impl_.try_receive();
    }

    /**
     * @brief Send message using substrate-specific logic
     */
    void send_message(T* msg_ptr) {
        S::template send_message<T>(msg_ptr, listeners_);
    }

    /**
     * @brief Register message listener
     */
    void register_listener(std::function<void(T*)> listener) {
        listeners_.push_back(std::move(listener));
    }

    /**
     * @brief Compile-time traits
     */
    static constexpr bool is_zero_copy() { return S::is_zero_copy; }
    static constexpr bool needs_serialization() { return S::needs_serialization; }
    static constexpr bool is_cross_process() { return S::cross_process; }
    static constexpr bool needs_locks() { return P::needs_locks; }

    /**
     * @brief Runtime statistics
     */
    size_t capacity() const { return max_messages_; }
    T* slab() const { return slab_; }

private:
    uint16_t size_factor_;
    T* slab_ = nullptr;
    size_t max_messages_ = 0;
    PatternImpl pattern_impl_;
    std::vector<std::function<void(T*)>> listeners_;

    const char* substrate_name() const {
        if constexpr (std::is_same_v<S, substrate::InProcess>) return "InProcess";
        else if constexpr (std::is_same_v<S, substrate::IPC>) return "IPC";
        else if constexpr (std::is_same_v<S, substrate::TCP>) return "TCP";
        else if constexpr (std::is_same_v<S, substrate::GPU>) return "GPU";
        else if constexpr (std::is_same_v<S, substrate::Debug>) return "Debug";
        else return "Unknown";
    }

    const char* pattern_name() const {
        if constexpr (std::is_same_v<P, pattern::SPSC>) return "SPSC";
        else if constexpr (std::is_same_v<P, pattern::MPSC>) return "MPSC";
        else if constexpr (std::is_same_v<P, pattern::SPMC>) return "SPMC";
        else if constexpr (std::is_same_v<P, pattern::MPMC>) return "MPMC";
        else return "Unknown";
    }
};

/**
 * @brief Unified message class - replaces all previous message classes
 */
template<typename T, class S = substrate::InProcess, class P = pattern::SPSC>
class Message {
public:
    /**
     * @brief Construct message directly in channel slab
     */
    explicit Message(Channel<T, S, P>& channel) : channel_(channel) {
        data_ = channel_.allocate_next();
        if (!data_) {
            throw std::runtime_error("Channel full - cannot allocate message");
        }
        new (data_) T{};
    }

    template<typename... Args>
    Message(Channel<T, S, P>& channel, Args&&... args) : channel_(channel) {
        data_ = channel_.allocate_next();
        if (!data_) {
            throw std::runtime_error("Channel full - cannot allocate message");
        }
        new (data_) T(std::forward<Args>(args)...);
    }

    // Move-only
    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;
    
    Message(Message&& other) noexcept 
        : channel_(other.channel_), data_(other.data_) {
        other.data_ = nullptr;
    }

    T* operator->() { return data_; }
    const T* operator->() const { return data_; }
    T& operator*() { return *data_; }
    const T& operator*() const { return *data_; }

    /**
     * @brief Send message - just updates pointers, NO COPY
     */
    void send() {
        if (data_) {
            channel_.send_message(data_);
            data_ = nullptr;
        }
    }

    bool valid() const { return data_ != nullptr; }

private:
    Channel<T, S, P>& channel_;
    T* data_ = nullptr;
};

// Convenience aliases replacing all the old channel types
template<typename T> using InProcessSPSC = Channel<T, substrate::InProcess, pattern::SPSC>;
template<typename T> using InProcessMPSC = Channel<T, substrate::InProcess, pattern::MPSC>;
template<typename T> using InProcessSPMC = Channel<T, substrate::InProcess, pattern::SPMC>;
template<typename T> using InProcessMPMC = Channel<T, substrate::InProcess, pattern::MPMC>;

template<typename T> using IPCChannel = Channel<T, substrate::IPC, pattern::SPSC>;
template<typename T> using TCPChannel = Channel<T, substrate::TCP, pattern::SPSC>;
template<typename T> using GPUChannel = Channel<T, substrate::GPU, pattern::SPSC>;
template<typename T> using DebugChannel = Channel<T, substrate::Debug, pattern::SPSC>;

// GPU pattern combinations
template<typename T> using GPUSPSC = Channel<T, substrate::GPU, pattern::SPSC>;
template<typename T> using GPUMPSC = Channel<T, substrate::GPU, pattern::MPSC>;
template<typename T> using GPUSPMC = Channel<T, substrate::GPU, pattern::SPMC>;
template<typename T> using GPUMPMC = Channel<T, substrate::GPU, pattern::MPMC>;

// Debug pattern combinations  
template<typename T> using DebugSPSC = Channel<T, substrate::Debug, pattern::SPSC>;
template<typename T> using DebugMPSC = Channel<T, substrate::Debug, pattern::MPSC>;
template<typename T> using DebugSPMC = Channel<T, substrate::Debug, pattern::SPMC>;
template<typename T> using DebugMPMC = Channel<T, substrate::Debug, pattern::MPMC>;

// Factory functions
template<typename T>
std::shared_ptr<InProcessSPSC<T>> make_spsc_channel(uint16_t size_factor = 0) {
    return std::make_shared<InProcessSPSC<T>>(size_factor);
}

template<typename T>
std::shared_ptr<InProcessMPSC<T>> make_mpsc_channel(uint16_t size_factor = 0) {
    return std::make_shared<InProcessMPSC<T>>(size_factor);
}

template<typename T>
std::shared_ptr<InProcessSPMC<T>> make_spmc_channel(uint16_t size_factor = 0) {
    return std::make_shared<InProcessSPMC<T>>(size_factor);
}

template<typename T>
std::shared_ptr<InProcessMPMC<T>> make_mpmc_channel(uint16_t size_factor = 0) {
    return std::make_shared<InProcessMPMC<T>>(size_factor);
}

template<typename T>
std::shared_ptr<GPUChannel<T>> make_gpu_channel(uint16_t size_factor = 0) {
    return std::make_shared<GPUChannel<T>>(size_factor);
}

template<typename T>
std::shared_ptr<DebugChannel<T>> make_debug_channel(uint16_t size_factor = 0) {
    return std::make_shared<DebugChannel<T>>(size_factor);
}

} // namespace psyne