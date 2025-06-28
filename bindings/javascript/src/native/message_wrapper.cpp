/**
 * @file message_wrapper.cpp
 * @brief Implementation of N-API wrapper for Psyne Message classes
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

#include "message_wrapper.h"
#include "channel_wrapper.h"

namespace psyne_js {

void MessageWrapper::Init(Napi::Env env, Napi::Object& exports) {
    auto messages = Napi::Object::New(env);
    
    messages.Set("createFloatVector", Napi::Function::New(env, CreateFloatVector));
    messages.Set("createDoubleMatrix", Napi::Function::New(env, CreateDoubleMatrix));
    messages.Set("createByteVector", Napi::Function::New(env, CreateByteVector));
    
    exports.Set("Messages", messages);
}

Napi::Value MessageWrapper::CreateFloatVector(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Channel argument required").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    // Extract channel from wrapper
    auto channel_wrapper = Napi::ObjectWrap<ChannelWrapper>::Unwrap(info[0].As<Napi::Object>());
    // Note: In real implementation, we'd need access to the channel
    
    Napi::Object result = Napi::Object::New(env);
    result.Set("type", Napi::String::New(env, "floatVector"));
    result.Set("typeId", Napi::Number::New(env, 1));
    
    return result;
}

Napi::Value MessageWrapper::CreateDoubleMatrix(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Channel argument required").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    Napi::Object result = Napi::Object::New(env);
    result.Set("type", Napi::String::New(env, "doubleMatrix"));
    result.Set("typeId", Napi::Number::New(env, 2));
    
    return result;
}

Napi::Value MessageWrapper::CreateByteVector(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Channel argument required").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    Napi::Object result = Napi::Object::New(env);
    result.Set("type", Napi::String::New(env, "byteVector"));
    result.Set("typeId", Napi::Number::New(env, 10));
    
    return result;
}

} // namespace psyne_js