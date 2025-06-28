#pragma once

/**
 * @file message.hpp
 * @brief Core message interface for enhanced types
 * 
 * This header provides the minimal interface needed for enhanced message types
 * to inherit from the Message base class without circular dependencies.
 */

#include <cstdint>
#include <cstddef>

namespace psyne {

// Forward declarations
class Channel;

/**
 * @brief Base class for all zero-copy messages
 * @tparam Derived The derived message type (CRTP pattern)
 */
template<typename Derived>
class Message {
public:
    /**
     * @brief Create an outgoing message (allocates space in channel)
     * @param channel The channel to allocate the message in
     */
    explicit Message(Channel& channel);
    
    /**
     * @brief Create an incoming message (view of existing data)
     * @param data Pointer to existing message data
     * @param size Size of the message data
     */
    Message(void* data, size_t size);
    
    /**
     * @brief Move constructor
     */
    Message(Message&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     */
    Message& operator=(Message&& other) noexcept;
    
    /**
     * @brief Destructor
     */
    ~Message();
    
    // Disable copy construction and assignment
    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;
    
    /**
     * @brief Send the message through the channel
     */
    void send();
    
    /**
     * @brief Get the raw data pointer
     */
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    /**
     * @brief Get the message size
     */
    size_t size() const { return size_; }
    
    /**
     * @brief Check if the message is valid
     */
    bool valid() const { return data_ != nullptr; }

protected:
    void* data_;
    size_t size_;
    Channel* channel_;
    void* handle_;
    
    friend class Channel;
};

} // namespace psyne