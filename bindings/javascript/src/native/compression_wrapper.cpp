/**
 * @file compression_wrapper.cpp
 * @brief Implementation of N-API wrapper for Psyne Compression functionality
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

#include "compression_wrapper.h"

namespace psyne_js {

void CompressionWrapper::Init(Napi::Env env, Napi::Object& exports) {
    auto compression = Napi::Object::New(env);
    
    // Export compression utility functions if needed
    exports.Set("Compression", compression);
}

psyne::compression::CompressionConfig CompressionWrapper::FromObject(Napi::Env env, const Napi::Object& obj) {
    psyne::compression::CompressionConfig config;
    
    if (obj.Has("type")) {
        auto type_val = obj.Get("type").As<Napi::Number>().Uint32Value();
        config.type = static_cast<psyne::compression::CompressionType>(type_val);
    }
    
    if (obj.Has("level")) {
        config.level = obj.Get("level").As<Napi::Number>().Int32Value();
    }
    
    if (obj.Has("minSizeThreshold")) {
        config.min_size_threshold = obj.Get("minSizeThreshold").As<Napi::Number>().Uint32Value();
    }
    
    if (obj.Has("enableChecksum")) {
        config.enable_checksum = obj.Get("enableChecksum").As<Napi::Boolean>().Value();
    }
    
    return config;
}

Napi::Object CompressionWrapper::ToObject(Napi::Env env, const psyne::compression::CompressionConfig& config) {
    Napi::Object result = Napi::Object::New(env);
    
    result.Set("type", Napi::Number::New(env, static_cast<uint32_t>(config.type)));
    result.Set("level", Napi::Number::New(env, config.level));
    result.Set("minSizeThreshold", Napi::Number::New(env, config.min_size_threshold));
    result.Set("enableChecksum", Napi::Boolean::New(env, config.enable_checksum));
    
    return result;
}

} // namespace psyne_js