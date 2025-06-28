/**
 * @file metrics_wrapper.cpp
 * @brief Implementation of N-API wrapper for Psyne Metrics functionality
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

#include "metrics_wrapper.h"

namespace psyne_js {

void MetricsWrapper::Init(Napi::Env env, Napi::Object& exports) {
    // Add any metrics-specific functionality here
    // Currently handled in ChannelWrapper::GetMetrics()
}

Napi::Object MetricsWrapper::ToObject(Napi::Env env, const psyne::debug::ChannelMetrics& metrics) {
    Napi::Object result = Napi::Object::New(env);
    
    result.Set("messagesSent", Napi::Number::New(env, metrics.messages_sent));
    result.Set("bytesSent", Napi::Number::New(env, metrics.bytes_sent));
    result.Set("messagesReceived", Napi::Number::New(env, metrics.messages_received));
    result.Set("bytesReceived", Napi::Number::New(env, metrics.bytes_received));
    result.Set("sendBlocks", Napi::Number::New(env, metrics.send_blocks));
    result.Set("receiveBlocks", Napi::Number::New(env, metrics.receive_blocks));
    
    return result;
}

} // namespace psyne_js