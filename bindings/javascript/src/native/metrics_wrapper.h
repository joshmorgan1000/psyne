/**
 * @file metrics_wrapper.h
 * @brief N-API wrapper for Psyne Metrics functionality
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

#pragma once

#include <napi.h>
#include <psyne/psyne.hpp>

namespace psyne_js {

/**
 * @class MetricsWrapper
 * @brief N-API wrapper for Psyne metrics and debugging functionality
 */
class MetricsWrapper {
public:
    /**
     * @brief Initialize metrics wrapper functionality
     * @param env N-API environment
     * @param exports Module exports object
     */
    static void Init(Napi::Env env, Napi::Object& exports);
    
    /**
     * @brief Convert C++ ChannelMetrics to JavaScript object
     * @param env N-API environment
     * @param metrics C++ metrics object
     * @return JavaScript object with metrics data
     */
    static Napi::Object ToObject(Napi::Env env, const psyne::debug::ChannelMetrics& metrics);
};

} // namespace psyne_js