/**
 * @file message_wrapper.h
 * @brief N-API wrapper for Psyne Message classes
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

#pragma once

#include <napi.h>
#include <psyne/psyne.hpp>

namespace psyne_js {

/**
 * @class MessageWrapper
 * @brief N-API wrapper for various Psyne message types
 */
class MessageWrapper {
public:
    /**
     * @brief Initialize message wrapper functionality
     * @param env N-API environment
     * @param exports Module exports object
     */
    static void Init(Napi::Env env, Napi::Object& exports);
    
    /**
     * @brief Create a FloatVector message
     * @param info N-API callback info
     * @return JavaScript object representing the message
     */
    static Napi::Value CreateFloatVector(const Napi::CallbackInfo& info);
    
    /**
     * @brief Create a DoubleMatrix message
     * @param info N-API callback info
     * @return JavaScript object representing the message
     */
    static Napi::Value CreateDoubleMatrix(const Napi::CallbackInfo& info);
    
    /**
     * @brief Create a ByteVector message
     * @param info N-API callback info
     * @return JavaScript object representing the message
     */
    static Napi::Value CreateByteVector(const Napi::CallbackInfo& info);
};

} // namespace psyne_js