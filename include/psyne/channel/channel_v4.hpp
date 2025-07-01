#pragma once

/**
 * @file channel_v4.hpp
 * @brief Clean architecture without circular dependencies
 * 
 * Flow:
 * 1. Channel<MessageType, Substrate, Pattern> creates Allocator using Substrate
 * 2. Messages take Allocator& (not Channel&) - breaks circular dependency
 * 3. Substrate and Pattern work with raw pointers - no message type dependency
 * 4. Clean separation of concerns!
 */

#include "core/allocator.hpp"
#include "concepts/allocator_concepts.hpp"
#include "global/logger.hpp"

#include <memory>
#include <functional>
#include <vector>

namespace psyne {

/**
 * @brief Clean substrate interface (no message type dependency)
 */
class SubstrateBase {
public:
    virtual ~SubstrateBase() = default;
    
    /**
     * @brief Allocate memory slab
     */
    virtual void* allocate_slab(size_t size_bytes) = 0;
    
    /**
     * @brief Deallocate memory slab
     */
    virtual void deallocate_slab(void* ptr) = 0;
    
    /**
     * @brief Send raw message data
     */
    virtual void send_raw_message(void* msg_ptr, size_t msg_size) = 0;
    
    /**
     * @brief Substrate capabilities
     */
    virtual bool needs_serialization() const = 0;
    virtual bool is_zero_copy() const = 0;
    virtual bool is_cross_process() const = 0;
    virtual const char* name() const = 0;
};

/**
 * @brief Clean pattern interface (no message type dependency)  
 */
class PatternBase {
public:
    virtual ~PatternBase() = default;
    
    /**
     * @brief Try to allocate space for a message
     */
    virtual void* try_allocate_raw(void* slab, size_t max_messages, size_t message_size) = 0;
    
    /**
     * @brief Try to receive a message
     */
    virtual void* try_receive_raw() = 0;
    
    /**
     * @brief Pattern capabilities
     */
    virtual bool needs_locks() const = 0;
    virtual size_t max_producers() const = 0;
    virtual size_t max_consumers() const = 0;
    virtual const char* name() const = 0;
    
    /**
     * @brief Pattern state
     */
    virtual size_t size() const = 0;
    virtual bool empty() const = 0;
    virtual bool full() const = 0;
};

/**
 * @brief Clean channel without circular dependencies
 */
template<typename T, typename S, typename P>
    requires concepts::SubstrateType<S> && concepts::PatternType<P>
class Channel {
public:
    /**
     * @brief Create channel with substrate and pattern
     */
    explicit Channel(uint16_t size_factor = 0) : size_factor_(size_factor) {
        // Allocate memory slab using substrate
        size_t slab_size = (size_factor + 1) * 32 * 1024 * 1024;
        slab_memory_ = substrate_.allocate_slab(slab_size);
        if (!slab_memory_) {
            throw std::bad_alloc();
        }
        
        max_messages_ = slab_size / sizeof(T);
        
        // Create allocator that messages will use
        allocator_ = std::make_unique<SlabAllocator<T>>(
            static_cast<T*>(slab_memory_), slab_size, substrate_.name());
        
        // Initialize pattern
        pattern_ = std::make_unique<P>();
        
        LOG_INFO("Created Channel<{}, {}, {}> with {}MB slab ({} messages)", 
                 typeid(T).name(), substrate_.name(), pattern_->name(),
                 slab_size / (1024 * 1024), max_messages_);
    }
    
    ~Channel() {
        if (slab_memory_) {
            substrate_.deallocate_slab(slab_memory_);
        }
    }
    
    // Non-copyable
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;
    
    /**
     * @brief Get allocator for message construction
     */
    AllocatorBase& allocator() { return *allocator_; }
    
    /**
     * @brief Send message using substrate
     */
    void send_message(T* msg_ptr) {
        substrate_.send_raw_message(msg_ptr, sizeof(T));
        
        // Notify local listeners
        for (auto& listener : listeners_) {
            listener(msg_ptr);
        }
    }
    
    /**
     * @brief Try to receive message using pattern
     */
    T* try_receive() {
        return static_cast<T*>(pattern_->try_receive_raw());
    }
    
    /**
     * @brief Register message listener
     */
    void register_listener(std::function<void(T*)> listener) {
        listeners_.push_back(std::move(listener));
    }
    
    /**
     * @brief Channel info
     */
    bool needs_serialization() const { return substrate_.needs_serialization(); }
    bool is_zero_copy() const { return substrate_.is_zero_copy(); }
    bool is_cross_process() const { return substrate_.is_cross_process(); }
    bool needs_locks() const { return pattern_->needs_locks(); }
    
    size_t capacity() const { return max_messages_; }
    size_t size() const { return pattern_->size(); }
    bool empty() const { return pattern_->empty(); }
    bool full() const { return pattern_->full(); }
    
    const char* substrate_name() const { return substrate_.name(); }
    const char* pattern_name() const { return pattern_->name(); }

private:
    uint16_t size_factor_;
    void* slab_memory_ = nullptr;
    size_t max_messages_ = 0;
    
    S substrate_;
    std::unique_ptr<P> pattern_;
    std::unique_ptr<AllocatorBase> allocator_;
    std::vector<std::function<void(T*)>> listeners_;
};

/**
 * @brief Message that takes allocator (no circular dependency!)
 */
template<typename T>
    requires concepts::MessageWithAllocator<T, AllocatorBase>
class Message {
public:
    /**
     * @brief Construct message using allocator
     */
    explicit Message(AllocatorBase& allocator) : allocator_(allocator) {
        void* memory = allocator_.allocate(sizeof(T), alignof(T));
        if (!memory) {
            throw std::runtime_error("Allocator out of memory");
        }
        
        data_ = static_cast<T*>(memory);
        new (data_) T(allocator_); // Message constructor takes allocator!
    }
    
    /**
     * @brief Construct with args
     */
    template<typename... Args>
    Message(AllocatorBase& allocator, Args&&... args) : allocator_(allocator) {
        void* memory = allocator_.allocate(sizeof(T), alignof(T));
        if (!memory) {
            throw std::runtime_error("Allocator out of memory");
        }
        
        data_ = static_cast<T*>(memory);
        new (data_) T(allocator_, std::forward<Args>(args)...);
    }
    
    // Move-only
    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;
    
    Message(Message&& other) noexcept 
        : allocator_(other.allocator_), data_(other.data_), channel_(other.channel_) {
        other.data_ = nullptr;
        other.channel_ = nullptr;
    }
    
    ~Message() {
        if (data_) {
            data_->~T();
            allocator_.deallocate(data_);
        }
    }
    
    /**
     * @brief Set channel for sending (called by channel)
     */
    template<typename S, typename P>
    void set_channel(Channel<T, S, P>* channel) {
        channel_ = reinterpret_cast<void*>(channel);
    }
    
    /**
     * @brief Access message data
     */
    T* operator->() { return data_; }
    const T* operator->() const { return data_; }
    T& operator*() { return *data_; }
    const T& operator*() const { return *data_; }
    
    /**
     * @brief Send message
     */
    void send() {
        if (data_ && channel_) {
            auto* typed_channel = reinterpret_cast<Channel<T, void, void>*>(channel_);
            typed_channel->send_message(data_);
            data_ = nullptr; // Message is sent
            channel_ = nullptr;
        }
    }
    
    bool valid() const { return data_ != nullptr; }

private:
    AllocatorBase& allocator_;
    T* data_ = nullptr;
    void* channel_ = nullptr; // Type-erased channel pointer
};

// Clean substrate implementations
namespace substrate {

class InProcess : public SubstrateBase {
public:
    void* allocate_slab(size_t size_bytes) override {
        return std::aligned_alloc(64, size_bytes);
    }
    
    void deallocate_slab(void* ptr) override {
        std::free(ptr);
    }
    
    void send_raw_message(void* msg_ptr, size_t msg_size) override {
        // For in-process, sending is just notification (handled by channel)
    }
    
    bool needs_serialization() const override { return false; }
    bool is_zero_copy() const override { return true; }
    bool is_cross_process() const override { return false; }
    const char* name() const override { return "InProcess"; }
};

class CSV : public SubstrateBase {
public:
    void* allocate_slab(size_t size_bytes) override {
        return std::aligned_alloc(64, size_bytes);
    }
    
    void deallocate_slab(void* ptr) override {
        std::free(ptr);
    }
    
    void send_raw_message(void* msg_ptr, size_t msg_size) override {
        // Convert message to CSV and write to file
        LOG_INFO("CSV: Writing message to CSV file");
        // serialize_to_csv(msg_ptr, msg_size);
    }
    
    bool needs_serialization() const override { return true; }
    bool is_zero_copy() const override { return false; }
    bool is_cross_process() const override { return true; }
    const char* name() const override { return "CSV"; }
};

} // namespace substrate

// Clean pattern implementations
namespace pattern {

class SPSC : public PatternBase {
public:
    void* try_allocate_raw(void* slab, size_t max_messages, size_t message_size) override {
        // Simple SPSC allocation logic
        size_t pos = next_pos_++ % max_messages;
        return static_cast<char*>(slab) + (pos * message_size);
    }
    
    void* try_receive_raw() override {
        // TODO: Implement proper SPSC receive
        return nullptr;
    }
    
    bool needs_locks() const override { return false; }
    size_t max_producers() const override { return 1; }
    size_t max_consumers() const override { return 1; }
    const char* name() const override { return "SPSC"; }
    
    size_t size() const override { return 0; }
    bool empty() const override { return true; }
    bool full() const override { return false; }

private:
    std::atomic<size_t> next_pos_{0};
};

} // namespace pattern

} // namespace psyne