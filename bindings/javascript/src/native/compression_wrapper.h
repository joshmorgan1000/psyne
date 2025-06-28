/**
 * @file compression_wrapper.h
 * @brief N-API wrapper for Psyne Compression functionality
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

#pragma once

#include <napi.h>
#include <psyne/psyne.hpp>

namespace psyne_js {

/**
 * @class CompressionWrapper
 * @brief N-API wrapper for Psyne compression configuration
 */
class CompressionWrapper {
public:
    /**
     * @brief Initialize compression wrapper functionality
     * @param env N-API environment
     * @param exports Module exports object
     */
    static void Init(Napi::Env env, Napi::Object& exports);
    
    /**
     * @brief Convert JavaScript object to C++ CompressionConfig
     * @param env N-API environment
     * @param obj JavaScript configuration object
     * @return C++ CompressionConfig
     */
    static psyne::compression::CompressionConfig FromObject(Napi::Env env, const Napi::Object& obj);
    
    /**
     * @brief Convert C++ CompressionConfig to JavaScript object
     * @param env N-API environment
     * @param config C++ compression config
     * @return JavaScript object
     */
    static Napi::Object ToObject(Napi::Env env, const psyne::compression::CompressionConfig& config);
};

} // namespace psyne_js