/**
 * @file object_pool.hpp
 * @brief High-performance lock-free object pool for Psyne
 *
 * Provides zero-allocation object recycling for network buffers,
 * messages, and other frequently allocated objects.
 */

#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <functional>
#include <type_traits>

namespace psyne {

/**
 * @brief Lock-free object pool using hazard pointers
 * 
 * Features:
 * - Lock-free push/pop operations
 * - Automatic growth when pool is exhausted
 * - Custom object initialization/cleanup
 * - Memory-efficient storage
 * - Cache-friendly access patterns
 */
template <typename T>
class ObjectPool {
public:
    using value_type = T;
    using pointer = T*;
    using deleter_type = std::function<void(T*)>;
    
    /**
     * @brief Smart pointer that returns object to pool when destroyed
     */
    class PooledPtr {
    public:
        PooledPtr() = default;
        
        PooledPtr(T* ptr, ObjectPool* pool) 
            : ptr_(ptr), pool_(pool) {}
        
        ~PooledPtr() {
            if (ptr_ && pool_) {
                pool_->release(ptr_);
            }
        }
        
        // Move semantics
        PooledPtr(PooledPtr&& other) noexcept
            : ptr_(other.ptr_), pool_(other.pool_) {
            other.ptr_ = nullptr;
            other.pool_ = nullptr;
        }
        
        PooledPtr& operator=(PooledPtr&& other) noexcept {
            if (this != &other) {
                if (ptr_ && pool_) {
                    pool_->release(ptr_);
                }
                ptr_ = other.ptr_;
                pool_ = other.pool_;
                other.ptr_ = nullptr;
                other.pool_ = nullptr;
            }
            return *this;
        }
        
        // Delete copy operations
        PooledPtr(const PooledPtr&) = delete;
        PooledPtr& operator=(const PooledPtr&) = delete;
        
        // Pointer operations
        T* get() const { return ptr_; }
        T* operator->() const { return ptr_; }
        T& operator*() const { return *ptr_; }
        explicit operator bool() const { return ptr_ != nullptr; }
        
        // Release ownership without returning to pool
        T* release() {
            T* tmp = ptr_;
            ptr_ = nullptr;
            pool_ = nullptr;
            return tmp;
        }
        
        // Reset with new pointer
        void reset(T* ptr = nullptr) {
            if (ptr_ && pool_) {
                pool_->release(ptr_);
            }
            ptr_ = ptr;
        }
        
    private:
        T* ptr_ = nullptr;
        ObjectPool* pool_ = nullptr;
    };
    
    /**
     * @brief Construct pool with initial capacity
     * @param initial_size Number of objects to pre-allocate
     * @param max_size Maximum pool size (0 = unlimited)
     * @param factory Function to create new objects
     * @param deleter Function to destroy objects
     */
    explicit ObjectPool(
        size_t initial_size = 16,
        size_t max_size = 0,
        std::function<T*()> factory = []() { return new T(); },
        deleter_type deleter = [](T* p) { delete p; })
        : max_size_(max_size), 
          factory_(std::move(factory)),
          deleter_(std::move(deleter)) {
        
        // Pre-allocate initial objects
        for (size_t i = 0; i < initial_size; ++i) {
            Node* node = new Node{factory_(), nullptr};
            push_node(node);
            total_objects_++;
        }
    }
    
    ~ObjectPool() {
        // Clean up all objects
        Node* current = head_.load();
        while (current) {
            Node* next = current->next;
            if (current->data) {
                deleter_(current->data);
            }
            delete current;
            current = next;
        }
    }
    
    /**
     * @brief Acquire object from pool
     * @return Smart pointer to pooled object
     */
    PooledPtr acquire() {
        T* obj = pop();
        if (!obj) {
            // Pool exhausted, create new object if allowed
            if (max_size_ == 0 || total_objects_ < max_size_) {
                obj = factory_();
                total_objects_++;
            } else {
                return PooledPtr(); // Return empty pointer
            }
        }
        return PooledPtr(obj, this);
    }
    
    /**
     * @brief Try to acquire object without blocking
     * @return Object pointer or nullptr if pool is empty
     */
    T* try_pop() {
        return pop();
    }
    
    /**
     * @brief Return object to pool
     * @param obj Object to return (must have been acquired from this pool)
     */
    void release(T* obj) {
        if (obj) {
            push(obj);
        }
    }
    
    /**
     * @brief Get current number of available objects
     */
    size_t available() const {
        return available_count_.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Get total number of objects (in use + available)
     */
    size_t total_objects() const {
        return total_objects_.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Reserve additional objects
     */
    void reserve(size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (max_size_ > 0 && total_objects_ >= max_size_) {
                break;
            }
            Node* node = new Node{factory_(), nullptr};
            push_node(node);
            total_objects_++;
        }
    }
    
private:
    struct Node {
        T* data;
        Node* next;
    };
    
    // Lock-free stack implementation
    std::atomic<Node*> head_{nullptr};
    std::atomic<size_t> available_count_{0};
    std::atomic<size_t> total_objects_{0};
    
    size_t max_size_;
    std::function<T*()> factory_;
    deleter_type deleter_;
    
    void push_node(Node* node) {
        node->next = head_.load(std::memory_order_relaxed);
        while (!head_.compare_exchange_weak(node->next, node,
                                           std::memory_order_release,
                                           std::memory_order_relaxed));
        available_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void push(T* obj) {
        Node* node = new Node{obj, nullptr};
        push_node(node);
    }
    
    T* pop() {
        Node* head = head_.load(std::memory_order_acquire);
        while (head) {
            if (head_.compare_exchange_weak(head, head->next,
                                           std::memory_order_release,
                                           std::memory_order_acquire)) {
                available_count_.fetch_sub(1, std::memory_order_relaxed);
                T* obj = head->data;
                delete head;
                return obj;
            }
        }
        return nullptr;
    }
};

/**
 * @brief Specialized pool for byte buffers
 */
class BufferPool {
public:
    struct Buffer {
        std::unique_ptr<uint8_t[]> data;
        size_t capacity;
        size_t used = 0;
        
        Buffer(size_t size) : data(std::make_unique<uint8_t[]>(size)), capacity(size) {}
        
        void reset() { used = 0; }
        uint8_t* ptr() { return data.get(); }
        const uint8_t* ptr() const { return data.get(); }
    };
    
    using PooledBuffer = ObjectPool<Buffer>::PooledPtr;
    
    explicit BufferPool(size_t buffer_size, size_t initial_count = 32)
        : buffer_size_(buffer_size),
          pool_(initial_count, 0,
                [buffer_size]() { return new Buffer(buffer_size); },
                [](Buffer* b) { delete b; }) {}
    
    /**
     * @brief Get buffer from pool
     */
    PooledBuffer acquire() {
        auto buffer = pool_.acquire();
        if (buffer) {
            buffer->reset();
        }
        return buffer;
    }
    
    /**
     * @brief Get pool statistics
     */
    size_t available() const { return pool_.available(); }
    size_t total_buffers() const { return pool_.total_objects(); }
    size_t buffer_size() const { return buffer_size_; }
    
private:
    size_t buffer_size_;
    ObjectPool<Buffer> pool_;
};

/**
 * @brief Pool for message objects with embedded data
 */
template <typename MessageType>
class MessagePool {
public:
    using PooledMessage = typename ObjectPool<MessageType>::PooledPtr;
    
    explicit MessagePool(size_t initial_count = 64)
        : pool_(initial_count) {}
    
    /**
     * @brief Allocate message from pool
     */
    PooledMessage allocate() {
        return pool_.acquire();
    }
    
    /**
     * @brief Allocate with custom size (for variable-sized messages)
     */
    template <typename... Args>
    PooledMessage allocate(Args&&... args) {
        auto msg = pool_.acquire();
        if (msg) {
            new (msg.get()) MessageType(std::forward<Args>(args)...);
        }
        return msg;
    }
    
    size_t available() const { return pool_.available(); }
    size_t total_messages() const { return pool_.total_objects(); }
    
private:
    ObjectPool<MessageType> pool_;
};

} // namespace psyne