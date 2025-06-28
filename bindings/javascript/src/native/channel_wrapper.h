/**
 * @file channel_wrapper.h
 * @brief N-API wrapper for Psyne Channel class
 * 
 * Provides JavaScript bindings for the Psyne Channel functionality,
 * including message sending, receiving, and event handling.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

#pragma once

#include <napi.h>
#include <psyne/psyne.hpp>
#include <memory>
#include <thread>
#include <atomic>

namespace psyne_js {

/**
 * @class ChannelWrapper
 * @brief N-API wrapper class for psyne::Channel
 * 
 * This class wraps the C++ Channel object and exposes it to JavaScript
 * with proper async handling and event emission capabilities.
 */
class ChannelWrapper : public Napi::ObjectWrap<ChannelWrapper> {
public:
    /**
     * @brief Initialize the wrapper class in N-API environment
     * @param env N-API environment
     * @param exports Module exports object
     */
    static void Init(Napi::Env env, Napi::Object& exports);
    
    /**
     * @brief Create new JavaScript instance wrapping a C++ channel
     * @param env N-API environment
     * @param channel Unique pointer to C++ channel
     * @return Napi::Object representing the wrapped channel
     */
    static Napi::Object NewInstance(Napi::Env env, std::unique_ptr<psyne::Channel> channel);
    
    /**
     * @brief Constructor called from JavaScript
     * @param info N-API callback info
     */
    ChannelWrapper(const Napi::CallbackInfo& info);
    
    /**
     * @brief Destructor - ensures proper cleanup
     */
    ~ChannelWrapper();

private:
    // JavaScript-exposed methods
    
    /**
     * @brief Send a message through the channel
     * @param info Contains message data and type information
     * @return Promise that resolves when message is sent
     */
    Napi::Value Send(const Napi::CallbackInfo& info);
    
    /**
     * @brief Receive a message from the channel
     * @param info Optional timeout and message type filter
     * @return Promise that resolves with received message or null
     */
    Napi::Value Receive(const Napi::CallbackInfo& info);
    
    /**
     * @brief Start listening for messages with a callback
     * @param info Contains callback function and optional message type filter
     * @return Undefined
     */
    Napi::Value Listen(const Napi::CallbackInfo& info);
    
    /**
     * @brief Stop listening for messages
     * @param info N-API callback info (no parameters)
     * @return Undefined
     */
    Napi::Value StopListening(const Napi::CallbackInfo& info);
    
    /**
     * @brief Stop the channel
     * @param info N-API callback info (no parameters)
     * @return Undefined
     */
    Napi::Value Stop(const Napi::CallbackInfo& info);
    
    /**
     * @brief Check if channel is stopped
     * @param info N-API callback info (no parameters)
     * @return Boolean indicating if channel is stopped
     */
    Napi::Value IsStopped(const Napi::CallbackInfo& info);
    
    /**
     * @brief Get channel URI
     * @param info N-API callback info (no parameters)
     * @return String containing channel URI
     */
    Napi::Value GetUri(const Napi::CallbackInfo& info);
    
    /**
     * @brief Get channel mode
     * @param info N-API callback info (no parameters)
     * @return Number representing channel mode enum
     */
    Napi::Value GetMode(const Napi::CallbackInfo& info);
    
    /**
     * @brief Get channel type
     * @param info N-API callback info (no parameters)
     * @return Number representing channel type enum
     */
    Napi::Value GetType(const Napi::CallbackInfo& info);
    
    /**
     * @brief Check if metrics are enabled
     * @param info N-API callback info (no parameters)
     * @return Boolean indicating if metrics are enabled
     */
    Napi::Value HasMetrics(const Napi::CallbackInfo& info);
    
    /**
     * @brief Get current metrics
     * @param info N-API callback info (no parameters)
     * @return Object containing metrics data
     */
    Napi::Value GetMetrics(const Napi::CallbackInfo& info);
    
    /**
     * @brief Reset metrics counters
     * @param info N-API callback info (no parameters)
     * @return Undefined
     */
    Napi::Value ResetMetrics(const Napi::CallbackInfo& info);

private:
    // Internal helper methods
    
    /**
     * @brief Worker thread function for message listening
     */
    void ListenWorker();
    
    /**
     * @brief Convert C++ message to JavaScript object
     * @param env N-API environment
     * @param data Message data pointer
     * @param size Message size in bytes
     * @param type_id Message type ID
     * @return JavaScript object representing the message
     */
    Napi::Object MessageToJS(Napi::Env env, const void* data, size_t size, uint32_t type_id);
    
    /**
     * @brief Convert JavaScript object to C++ message data
     * @param env N-API environment
     * @param obj JavaScript message object
     * @param data Output data buffer
     * @param size Output size
     * @param type_id Output type ID
     * @return True if conversion successful
     */
    bool MessageFromJS(Napi::Env env, const Napi::Object& obj, 
                      std::vector<uint8_t>& data, uint32_t& type_id);

private:
    std::unique_ptr<psyne::Channel> channel_;
    std::unique_ptr<std::thread> listen_thread_;
    std::atomic<bool> should_stop_listening_{false};
    Napi::ThreadSafeFunction message_callback_;
    
    static Napi::FunctionReference constructor_;
};