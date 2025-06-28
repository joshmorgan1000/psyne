/**
 * @file psyne_addon.cpp
 * @brief Main Node.js addon entry point for Psyne JavaScript bindings
 * 
 * This file implements the N-API bindings for the Psyne C++ library,
 * providing JavaScript access to high-performance messaging functionality.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

#include <napi.h>
#include <psyne/psyne.hpp>
#include "channel_wrapper.h"
#include "message_wrapper.h"
#include "metrics_wrapper.h"
#include "compression_wrapper.h"

namespace psyne_js {

/**
 * @brief Get the version of the Psyne library
 * @param info N-API callback info
 * @return String containing version information
 */
Napi::Value GetVersion(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return Napi::String::New(env, psyne::get_version());
}

/**
 * @brief Print the Psyne banner to console
 * @param info N-API callback info
 * @return Undefined
 */
Napi::Value PrintBanner(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    psyne::print_banner();
    return env.Undefined();
}

/**
 * @brief Create a channel from URI
 * @param info N-API callback info containing:
 *   - uri: string - Channel URI
 *   - bufferSize: number - Buffer size in bytes (optional, default 1MB)
 *   - mode: number - Channel mode enum (optional, default SPSC)
 *   - type: number - Channel type enum (optional, default MultiType)
 *   - enableMetrics: boolean - Enable metrics collection (optional, default false)
 *   - compressionConfig: object - Compression configuration (optional)
 * @return ChannelWrapper instance
 */
Napi::Value CreateChannel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    // Validate arguments
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "First argument must be a string URI")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    
    std::string uri = info[0].As<Napi::String>().Utf8Value();
    
    // Parse optional arguments
    size_t buffer_size = 1024 * 1024; // Default 1MB
    psyne::ChannelMode mode = psyne::ChannelMode::SPSC;
    psyne::ChannelType type = psyne::ChannelType::MultiType;
    bool enable_metrics = false;
    psyne::compression::CompressionConfig compression_config;
    
    if (info.Length() > 1 && info[1].IsNumber()) {
        buffer_size = info[1].As<Napi::Number>().Uint32Value();
    }
    
    if (info.Length() > 2 && info[2].IsNumber()) {
        uint32_t mode_val = info[2].As<Napi::Number>().Uint32Value();
        mode = static_cast<psyne::ChannelMode>(mode_val);
    }
    
    if (info.Length() > 3 && info[3].IsNumber()) {
        uint32_t type_val = info[3].As<Napi::Number>().Uint32Value();
        type = static_cast<psyne::ChannelType>(type_val);
    }
    
    if (info.Length() > 4 && info[4].IsBoolean()) {
        enable_metrics = info[4].As<Napi::Boolean>().Value();
    }
    
    if (info.Length() > 5 && info[5].IsObject()) {
        auto config_obj = info[5].As<Napi::Object>();
        compression_config = CompressionWrapper::FromObject(env, config_obj);
    }
    
    try {
        // Create the channel using Psyne factory
        auto channel = psyne::create_channel(
            uri, buffer_size, mode, type, enable_metrics, compression_config
        );
        
        // Wrap in JavaScript object
        return ChannelWrapper::NewInstance(env, std::move(channel));
        
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

/**
 * @brief Create a reliable channel with error recovery features
 * @param info N-API callback info with same parameters as CreateChannel
 * @return ChannelWrapper instance with reliability features enabled
 */
Napi::Value CreateReliableChannel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "First argument must be a string URI")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    
    std::string uri = info[0].As<Napi::String>().Utf8Value();
    size_t buffer_size = 1024 * 1024;
    psyne::ChannelMode mode = psyne::ChannelMode::SPSC;
    psyne::ReliabilityConfig reliability_config;
    
    if (info.Length() > 1 && info[1].IsNumber()) {
        buffer_size = info[1].As<Napi::Number>().Uint32Value();
    }
    
    if (info.Length() > 2 && info[2].IsNumber()) {
        uint32_t mode_val = info[2].As<Napi::Number>().Uint32Value();
        mode = static_cast<psyne::ChannelMode>(mode_val);
    }
    
    // Parse reliability config from object if provided
    if (info.Length() > 3 && info[3].IsObject()) {
        auto config_obj = info[3].As<Napi::Object>();
        
        if (config_obj.Has("enableAcknowledgments")) {
            reliability_config.enable_acknowledgments = 
                config_obj.Get("enableAcknowledgments").As<Napi::Boolean>().Value();
        }
        
        if (config_obj.Has("maxRetries")) {
            reliability_config.max_retries = 
                config_obj.Get("maxRetries").As<Napi::Number>().Uint32Value();
        }
        
        if (config_obj.Has("ackTimeout")) {
            auto timeout_ms = config_obj.Get("ackTimeout").As<Napi::Number>().Uint32Value();
            reliability_config.ack_timeout = std::chrono::milliseconds(timeout_ms);
        }
    }
    
    try {
        auto channel = psyne::create_reliable_channel(uri, buffer_size, mode, reliability_config);
        return ChannelWrapper::NewInstance(env, std::move(channel));
        
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

/**
 * @brief Initialize the addon and export functions/classes
 * @param env N-API environment
 * @param exports Module exports object
 * @return Module exports
 */
Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // Export utility functions
    exports.Set("getVersion", Napi::Function::New(env, GetVersion));
    exports.Set("printBanner", Napi::Function::New(env, PrintBanner));
    exports.Set("createChannel", Napi::Function::New(env, CreateChannel));
    exports.Set("createReliableChannel", Napi::Function::New(env, CreateReliableChannel));
    
    // Export enums
    auto channel_mode = Napi::Object::New(env);
    channel_mode.Set("SPSC", Napi::Number::New(env, static_cast<uint32_t>(psyne::ChannelMode::SPSC)));
    channel_mode.Set("SPMC", Napi::Number::New(env, static_cast<uint32_t>(psyne::ChannelMode::SPMC)));
    channel_mode.Set("MPSC", Napi::Number::New(env, static_cast<uint32_t>(psyne::ChannelMode::MPSC)));
    channel_mode.Set("MPMC", Napi::Number::New(env, static_cast<uint32_t>(psyne::ChannelMode::MPMC)));
    exports.Set("ChannelMode", channel_mode);
    
    auto channel_type = Napi::Object::New(env);
    channel_type.Set("SingleType", Napi::Number::New(env, static_cast<uint32_t>(psyne::ChannelType::SingleType)));
    channel_type.Set("MultiType", Napi::Number::New(env, static_cast<uint32_t>(psyne::ChannelType::MultiType)));
    exports.Set("ChannelType", channel_type);
    
    auto compression_type = Napi::Object::New(env);
    compression_type.Set("None", Napi::Number::New(env, static_cast<uint32_t>(psyne::compression::CompressionType::None)));
    compression_type.Set("LZ4", Napi::Number::New(env, static_cast<uint32_t>(psyne::compression::CompressionType::LZ4)));
    compression_type.Set("Zstd", Napi::Number::New(env, static_cast<uint32_t>(psyne::compression::CompressionType::Zstd)));
    compression_type.Set("Snappy", Napi::Number::New(env, static_cast<uint32_t>(psyne::compression::CompressionType::Snappy)));
    exports.Set("CompressionType", compression_type);
    
    // Initialize wrapper classes
    ChannelWrapper::Init(env, exports);
    MessageWrapper::Init(env, exports);
    MetricsWrapper::Init(env, exports);
    CompressionWrapper::Init(env, exports);
    
    return exports;
}

} // namespace psyne_js

// Register the addon
NODE_API_MODULE(psyne_native, psyne_js::Init)