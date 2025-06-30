/**
 * @file channel_wrapper.cpp
 * @brief Implementation of N-API wrapper for Psyne Channel class
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

#include "channel_wrapper.h"
#include "message_wrapper.h"
#include <vector>
#include <chrono>
#include <future>

namespace psyne_js {

Napi::FunctionReference ChannelWrapper::constructor_;

void ChannelWrapper::Init(Napi::Env env, Napi::Object& exports) {
    Napi::Function func = DefineClass(env, "Channel", {
        InstanceMethod("send", &ChannelWrapper::Send),
        InstanceMethod("receive", &ChannelWrapper::Receive),
        InstanceMethod("listen", &ChannelWrapper::Listen),
        InstanceMethod("stopListening", &ChannelWrapper::StopListening),
        InstanceMethod("stop", &ChannelWrapper::Stop),
        InstanceMethod("isStopped", &ChannelWrapper::IsStopped),
        InstanceMethod("getUri", &ChannelWrapper::GetUri),
        InstanceMethod("getMode", &ChannelWrapper::GetMode),
        InstanceMethod("getType", &ChannelWrapper::GetType),
        InstanceMethod("hasMetrics", &ChannelWrapper::HasMetrics),
        InstanceMethod("getMetrics", &ChannelWrapper::GetMetrics),
        InstanceMethod("resetMetrics", &ChannelWrapper::ResetMetrics),
    });
    
    constructor_ = Napi::Persistent(func);
    constructor_.SuppressDestruct();
    exports.Set("Channel", func);
}

Napi::Object ChannelWrapper::NewInstance(Napi::Env env, std::unique_ptr<psyne::Channel> channel) {
    Napi::Object obj = constructor_.New({});
    ChannelWrapper* wrapper = Napi::ObjectWrap<ChannelWrapper>::Unwrap(obj);
    wrapper->channel_ = std::move(channel);
    return obj;
}

ChannelWrapper::ChannelWrapper(const Napi::CallbackInfo& info) 
    : Napi::ObjectWrap<ChannelWrapper>(info) {
    // Constructor is called by NewInstance, channel_ is set there
}

ChannelWrapper::~ChannelWrapper() {
    if (listen_thread_ && listen_thread_->joinable()) {
        should_stop_listening_ = true;
        listen_thread_->join();
    }
    
    if (message_callback_) {
        message_callback_.Release();
    }
}

Napi::Value ChannelWrapper::Send(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1) {
        Napi::TypeError::New(env, "Message data is required").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    // Create a promise for async operation
    auto deferred = Napi::Promise::Deferred::New(env);
    
    try {
        // Handle different message formats
        if (info[0].IsBuffer()) {
            // Raw buffer data
            auto buffer = info[0].As<Napi::Buffer<uint8_t>>();
            uint32_t type_id = info.Length() > 1 && info[1].IsNumber() ? 
                info[1].As<Napi::Number>().Uint32Value() : 0;
            
            // Create ByteVector message
            psyne::ByteVector msg(*channel_);
            msg.resize(buffer.Length());
            std::memcpy(msg.data(), buffer.Data(), buffer.Length());
            msg.send();
            
            deferred.Resolve(env.Undefined());
            
        } else if (info[0].IsTypedArray()) {
            // Typed array (Float32Array, Int32Array, etc.)
            auto typed_array = info[0].As<Napi::TypedArray>();
            auto array_buffer = typed_array.ArrayBuffer();
            auto data = static_cast<uint8_t*>(array_buffer.Data()) + typed_array.ByteOffset();
            size_t length = typed_array.ByteLength();
            
            psyne::ByteVector msg(*channel_);
            msg.resize(length);
            std::memcpy(msg.data(), data, length);
            msg.send();
            
            deferred.Resolve(env.Undefined());
            
        } else if (info[0].IsArray()) {
            // JavaScript array - convert to FloatVector
            auto js_array = info[0].As<Napi::Array>();
            psyne::FloatVector msg(*channel_);
            msg.resize(js_array.Length());
            
            for (uint32_t i = 0; i < js_array.Length(); ++i) {
                auto value = js_array.Get(i);
                if (value.IsNumber()) {
                    msg[i] = value.As<Napi::Number>().FloatValue();
                } else {
                    msg[i] = 0.0f;
                }
            }
            
            msg.send();
            deferred.Resolve(env.Undefined());
            
        } else if (info[0].IsObject()) {
            // Structured message object
            auto obj = info[0].As<Napi::Object>();
            
            if (obj.Has("type") && obj.Has("data")) {
                auto type_str = obj.Get("type").As<Napi::String>().Utf8Value();
                auto data_val = obj.Get("data");
                
                if (type_str == "floatVector" && data_val.IsArray()) {
                    auto data_array = data_val.As<Napi::Array>();
                    psyne::FloatVector msg(*channel_);
                    msg.resize(data_array.Length());
                    
                    for (uint32_t i = 0; i < data_array.Length(); ++i) {
                        auto val = data_array.Get(i);
                        msg[i] = val.IsNumber() ? val.As<Napi::Number>().FloatValue() : 0.0f;
                    }
                    
                    msg.send();
                    deferred.Resolve(env.Undefined());
                    
                } else if (type_str == "doubleMatrix" && data_val.IsObject()) {
                    auto data_obj = data_val.As<Napi::Object>();
                    auto rows = data_obj.Get("rows").As<Napi::Number>().Uint32Value();
                    auto cols = data_obj.Get("cols").As<Napi::Number>().Uint32Value();
                    auto values = data_obj.Get("values").As<Napi::Array>();
                    
                    psyne::DoubleMatrix msg(*channel_);
                    msg.set_dimensions(rows, cols);
                    
                    for (uint32_t i = 0; i < rows; ++i) {
                        for (uint32_t j = 0; j < cols; ++j) {
                            auto idx = i * cols + j;
                            if (idx < values.Length()) {
                                auto val = values.Get(idx);
                                msg.at(i, j) = val.IsNumber() ? val.As<Napi::Number>().DoubleValue() : 0.0;
                            }
                        }
                    }
                    
                    msg.send();
                    deferred.Resolve(env.Undefined());
                    
                } else {
                    deferred.Reject(Napi::Error::New(env, "Unsupported message type").Value());
                }
            } else {
                deferred.Reject(Napi::Error::New(env, "Message object must have 'type' and 'data' properties").Value());
            }
        } else {
            deferred.Reject(Napi::TypeError::New(env, "Unsupported message format").Value());
        }
        
    } catch (const std::exception& e) {
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }
    
    return deferred.Promise();
}

Napi::Value ChannelWrapper::Receive(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    // Parse timeout (optional)
    std::chrono::milliseconds timeout(0); // Non-blocking by default
    if (info.Length() > 0 && info[0].IsNumber()) {
        timeout = std::chrono::milliseconds(info[0].As<Napi::Number>().Uint32Value());
    }
    
    // Create promise for async operation
    auto deferred = Napi::Promise::Deferred::New(env);
    
    // Run receive operation in thread pool to avoid blocking
    auto async_work = new Napi::AsyncWorker(env, "ChannelReceive");
    async_work->Queue();
    
    try {
        // Try to receive a message
        size_t size;
        uint32_t type_id;
        void* data = channel_->receive_raw_message(size, type_id);
        
        if (data) {
            // Convert to JavaScript object based on type
            Napi::Object result = MessageToJS(env, data, size, type_id);
            channel_->release_raw_message(data);
            deferred.Resolve(result);
        } else {
            // No message available
            deferred.Resolve(env.Null());
        }
        
    } catch (const std::exception& e) {
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }
    
    return deferred.Promise();
}

Napi::Value ChannelWrapper::Listen(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsFunction()) {
        Napi::TypeError::New(env, "Callback function is required").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    
    // Stop any existing listener
    if (listen_thread_) {
        should_stop_listening_ = true;
        if (listen_thread_->joinable()) {
            listen_thread_->join();
        }
    }
    
    // Create thread-safe function for callback
    message_callback_ = Napi::ThreadSafeFunction::New(
        env,
        info[0].As<Napi::Function>(),
        "MessageListener",
        0, // Unlimited queue
        1, // Only one thread
        [](Napi::Env) {} // Finalizer
    );
    
    // Start listening thread
    should_stop_listening_ = false;
    listen_thread_ = std::make_unique<std::thread>([this]() {
        ListenWorker();
    });
    
    return env.Undefined();
}

Napi::Value ChannelWrapper::StopListening(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (listen_thread_) {
        should_stop_listening_ = true;
        if (listen_thread_->joinable()) {
            listen_thread_->join();
        }
        listen_thread_.reset();
    }
    
    if (message_callback_) {
        message_callback_.Release();
    }
    
    return env.Undefined();
}

Napi::Value ChannelWrapper::Stop(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    StopListening(info);
    
    if (channel_) {
        channel_->stop();
    }
    
    return env.Undefined();
}

Napi::Value ChannelWrapper::IsStopped(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (!channel_) {
        return Napi::Boolean::New(env, true);
    }
    
    return Napi::Boolean::New(env, channel_->is_stopped());
}

Napi::Value ChannelWrapper::GetUri(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (!channel_) {
        return env.Null();
    }
    
    return Napi::String::New(env, channel_->uri());
}

Napi::Value ChannelWrapper::GetMode(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (!channel_) {
        return env.Null();
    }
    
    return Napi::Number::New(env, static_cast<uint32_t>(channel_->mode()));
}

Napi::Value ChannelWrapper::GetType(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (!channel_) {
        return env.Null();
    }
    
    return Napi::Number::New(env, static_cast<uint32_t>(channel_->type()));
}

Napi::Value ChannelWrapper::HasMetrics(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (!channel_) {
        return Napi::Boolean::New(env, false);
    }
    
    return Napi::Boolean::New(env, channel_->has_metrics());
}

Napi::Value ChannelWrapper::GetMetrics(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (!channel_ || !channel_->has_metrics()) {
        return env.Null();
    }
    
    auto metrics = channel_->get_metrics();
    
    Napi::Object result = Napi::Object::New(env);
    result.Set("messagesSent", Napi::Number::New(env, metrics.messages_sent));
    result.Set("bytesSent", Napi::Number::New(env, metrics.bytes_sent));
    result.Set("messagesReceived", Napi::Number::New(env, metrics.messages_received));
    result.Set("bytesReceived", Napi::Number::New(env, metrics.bytes_received));
    result.Set("sendBlocks", Napi::Number::New(env, metrics.send_blocks));
    result.Set("receiveBlocks", Napi::Number::New(env, metrics.receive_blocks));
    
    return result;
}

Napi::Value ChannelWrapper::ResetMetrics(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    
    if (channel_) {
        channel_->reset_metrics();
    }
    
    return env.Undefined();
}

void ChannelWrapper::ListenWorker() {
    while (!should_stop_listening_) {
        try {
            size_t size;
            uint32_t type_id;
            void* data = channel_->receive_raw_message(size, type_id);
            
            if (data && !should_stop_listening_) {
                // Call JavaScript callback with message
                auto callback = [this, data, size, type_id](Napi::Env env, Napi::Function jsCallback) {
                    Napi::Object message = MessageToJS(env, data, size, type_id);
                    channel_->release_raw_message(data);
                    jsCallback.Call({message});
                };
                
                message_callback_.BlockingCall(callback);
            } else if (data) {
                // Release message if we're stopping
                channel_->release_raw_message(data);
            }
            
            // Small delay to prevent busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
        } catch (const std::exception& e) {
            // Handle errors in listener thread
            auto callback = [e](Napi::Env env, Napi::Function jsCallback) {
                Napi::Error error = Napi::Error::New(env, e.what());
                jsCallback.Call({env.Null(), error.Value()});
            };
            
            if (message_callback_) {
                message_callback_.BlockingCall(callback);
            }
            break;
        }
    }
}

Napi::Object ChannelWrapper::MessageToJS(Napi::Env env, const void* data, size_t size, uint32_t type_id) {
    Napi::Object result = Napi::Object::New(env);
    result.Set("typeId", Napi::Number::New(env, type_id));
    result.Set("size", Napi::Number::New(env, size));
    
    // Convert based on known message types
    switch (type_id) {
        case 1: { // FloatVector
            auto float_data = static_cast<const float*>(data);
            size_t count = size / sizeof(float);
            
            Napi::Array array = Napi::Array::New(env, count);
            for (size_t i = 0; i < count; ++i) {
                array.Set(i, Napi::Number::New(env, float_data[i]));
            }
            
            result.Set("type", Napi::String::New(env, "floatVector"));
            result.Set("data", array);
            break;
        }
        
        case 2: { // DoubleMatrix
            // Parse matrix header and data
            result.Set("type", Napi::String::New(env, "doubleMatrix"));
            // Matrix parsing deferred to v2.0 bindings update
            break;
        }
        
        case 10: { // ByteVector
            auto buffer = Napi::Buffer<uint8_t>::Copy(env, static_cast<const uint8_t*>(data), size);
            result.Set("type", Napi::String::New(env, "byteVector"));
            result.Set("data", buffer);
            break;
        }
        
        default: {
            // Generic binary data
            auto buffer = Napi::Buffer<uint8_t>::Copy(env, static_cast<const uint8_t*>(data), size);
            result.Set("type", Napi::String::New(env, "binary"));
            result.Set("data", buffer);
            break;
        }
    }
    
    return result;
}

} // namespace psyne_js