#include <cstring>
#include <memory>
#include <mutex>
#include <psyne/psyne.hpp>
#include <psyne/psyne_c_api.h>
#include <thread>
#include <unordered_map>

using namespace psyne;

// Internal structures
struct psyne_channel {
    std::shared_ptr<Channel> cpp_channel;
    std::string uri;
    psyne_receive_callback_t callback = nullptr;
    void *callback_user_data = nullptr;
    std::unique_ptr<std::thread> callback_thread;
    std::atomic<bool> callback_running{false};
};

struct psyne_message {
    psyne_channel *channel;
    void *data;
    size_t size;
    uint32_t type;
    bool is_send; // true for send, false for receive

    // For send messages
    std::unique_ptr<ByteVector> send_msg;
};

// Global state
static std::mutex g_channels_mutex;
static std::unordered_map<psyne_channel *, std::unique_ptr<psyne_channel>>
    g_channels;
static bool g_initialized = false;

// Error descriptions
static const char *error_strings[] = {"Success",
                                      "Invalid argument",
                                      "Out of memory",
                                      "Channel full",
                                      "No message available",
                                      "Channel stopped",
                                      "Unsupported operation",
                                      "I/O error",
                                      "Timeout",
                                      "Unknown error"};

// Helper to convert C++ exceptions to error codes
static psyne_error_t handle_exception() {
    try {
        throw;
    } catch (const std::invalid_argument &) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        return PSYNE_ERROR_OUT_OF_MEMORY;
    } catch (const std::runtime_error &e) {
        // Parse error message for specific errors
        std::string msg = e.what();
        if (msg.find("full") != std::string::npos) {
            return PSYNE_ERROR_CHANNEL_FULL;
        } else if (msg.find("stopped") != std::string::npos) {
            return PSYNE_ERROR_CHANNEL_STOPPED;
        }
        return PSYNE_ERROR_IO;
    } catch (...) {
        return PSYNE_ERROR_UNKNOWN;
    }
}

// Library management
psyne_error_t psyne_init(void) {
    g_initialized = true;
    return PSYNE_OK;
}

void psyne_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_channels_mutex);

    // Stop all callback threads
    for (auto &[ptr, channel] : g_channels) {
        if (channel->callback_running) {
            channel->callback_running = false;
            if (channel->callback_thread &&
                channel->callback_thread->joinable()) {
                channel->callback_thread->join();
            }
        }
    }

    g_channels.clear();
    g_initialized = false;
}

const char *psyne_version(void) {
    return "1.2.0";
}

const char *psyne_error_string(psyne_error_t error) {
    int index = -error;
    if (index >= 0 &&
        index < sizeof(error_strings) / sizeof(error_strings[0])) {
        return error_strings[index];
    }
    return error_strings[sizeof(error_strings) / sizeof(error_strings[0]) - 1];
}

// Channel management
psyne_error_t psyne_channel_create(const char *uri, size_t buffer_size,
                                   psyne_channel_mode_t mode,
                                   psyne_channel_type_t type,
                                   psyne_channel_t **channel) {
    if (!uri || !channel) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        // Convert enums
        ChannelMode cpp_mode = static_cast<ChannelMode>(mode);
        ChannelType cpp_type = static_cast<ChannelType>(type);

        // Create C++ channel
        auto cpp_channel = create_channel(uri, buffer_size, cpp_mode, cpp_type);

        // Create C wrapper
        auto wrapper = std::make_unique<psyne_channel>();
        wrapper->cpp_channel = std::move(cpp_channel);
        wrapper->uri = uri;

        // Store in global map
        psyne_channel *raw_ptr = wrapper.get();
        {
            std::lock_guard<std::mutex> lock(g_channels_mutex);
            g_channels[raw_ptr] = std::move(wrapper);
        }

        *channel = raw_ptr;
        return PSYNE_OK;

    } catch (...) {
        return handle_exception();
    }
}

psyne_error_t psyne_channel_create_compressed(
    const char *uri, size_t buffer_size, psyne_channel_mode_t mode,
    psyne_channel_type_t type, const psyne_compression_config_t *compression,
    psyne_channel_t **channel) {
    if (!uri || !channel) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        // Convert enums
        ChannelMode cpp_mode = static_cast<ChannelMode>(mode);
        ChannelType cpp_type = static_cast<ChannelType>(type);

        // Create compression config
        compression::CompressionConfig cpp_compression;
        if (compression) {
            cpp_compression.type =
                static_cast<compression::CompressionType>(compression->type);
            cpp_compression.level = compression->level;
            cpp_compression.min_size_threshold =
                compression->min_size_threshold;
            cpp_compression.enable_checksum = compression->enable_checksum;
        }

        // Create C++ channel
        auto cpp_channel = create_channel(uri, buffer_size, cpp_mode, cpp_type,
                                          false, cpp_compression);

        // Create C wrapper
        auto wrapper = std::make_unique<psyne_channel>();
        wrapper->cpp_channel = std::move(cpp_channel);
        wrapper->uri = uri;

        // Store in global map
        psyne_channel *raw_ptr = wrapper.get();
        {
            std::lock_guard<std::mutex> lock(g_channels_mutex);
            g_channels[raw_ptr] = std::move(wrapper);
        }

        *channel = raw_ptr;
        return PSYNE_OK;

    } catch (...) {
        return handle_exception();
    }
}

void psyne_channel_destroy(psyne_channel_t *channel) {
    if (!channel)
        return;

    std::lock_guard<std::mutex> lock(g_channels_mutex);

    // Stop callback thread if running
    auto it = g_channels.find(channel);
    if (it != g_channels.end()) {
        if (it->second->callback_running) {
            it->second->callback_running = false;
            if (it->second->callback_thread &&
                it->second->callback_thread->joinable()) {
                it->second->callback_thread->join();
            }
        }
        g_channels.erase(it);
    }
}

psyne_error_t psyne_channel_stop(psyne_channel_t *channel) {
    if (!channel) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        channel->cpp_channel->stop();
        return PSYNE_OK;
    } catch (...) {
        return handle_exception();
    }
}

psyne_error_t psyne_channel_is_stopped(psyne_channel_t *channel,
                                       bool *stopped) {
    if (!channel || !stopped) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        *stopped = channel->cpp_channel->is_stopped();
        return PSYNE_OK;
    } catch (...) {
        return handle_exception();
    }
}

psyne_error_t psyne_channel_get_uri(psyne_channel_t *channel, char *uri,
                                    size_t uri_size) {
    if (!channel || !uri || uri_size == 0) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        size_t len = channel->uri.length();
        if (len >= uri_size) {
            return PSYNE_ERROR_INVALID_ARGUMENT;
        }

        std::strcpy(uri, channel->uri.c_str());
        return PSYNE_OK;
    } catch (...) {
        return handle_exception();
    }
}

psyne_error_t psyne_channel_get_metrics(psyne_channel_t *channel,
                                        psyne_metrics_t *metrics) {
    if (!channel || !metrics) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        if (!channel->cpp_channel->has_metrics()) {
            return PSYNE_ERROR_UNSUPPORTED;
        }

        auto cpp_metrics = channel->cpp_channel->get_metrics();
        metrics->messages_sent = cpp_metrics.messages_sent;
        metrics->bytes_sent = cpp_metrics.bytes_sent;
        metrics->messages_received = cpp_metrics.messages_received;
        metrics->bytes_received = cpp_metrics.bytes_received;
        metrics->send_blocks = cpp_metrics.send_blocks;
        metrics->receive_blocks = cpp_metrics.receive_blocks;

        return PSYNE_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Message operations
psyne_error_t psyne_message_reserve(psyne_channel_t *channel, size_t size,
                                    psyne_message_t **message) {
    if (!channel || !message) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        // Create message wrapper
        auto msg_wrapper = std::make_unique<psyne_message>();
        msg_wrapper->channel = channel;
        msg_wrapper->size = size;
        msg_wrapper->is_send = true;

        // Create ByteVector message
        msg_wrapper->send_msg =
            std::make_unique<ByteVector>(*channel->cpp_channel);
        msg_wrapper->send_msg->resize(size);
        msg_wrapper->data = msg_wrapper->send_msg->data();

        *message = msg_wrapper.release();
        return PSYNE_OK;
    } catch (...) {
        return handle_exception();
    }
}

psyne_error_t psyne_message_get_data(psyne_message_t *message, void **data,
                                     size_t *size) {
    if (!message || !data || !size) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    *data = message->data;
    *size = message->size;
    return PSYNE_OK;
}

psyne_error_t psyne_message_send(psyne_message_t *message, uint32_t type) {
    if (!message || !message->is_send) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        message->type = type;
        message->channel->cpp_channel->send(*message->send_msg);

        // Message is consumed after send
        delete message;
        return PSYNE_OK;
    } catch (...) {
        delete message;
        return handle_exception();
    }
}

void psyne_message_cancel(psyne_message_t *message) {
    if (message && message->is_send) {
        delete message;
    }
}

psyne_error_t psyne_message_receive(psyne_channel_t *channel,
                                    psyne_message_t **message, uint32_t *type) {
    return psyne_message_receive_timeout(channel, 0, message, type);
}

psyne_error_t psyne_message_receive_timeout(psyne_channel_t *channel,
                                            uint32_t timeout_ms,
                                            psyne_message_t **message,
                                            uint32_t *type) {
    if (!channel || !message || !type) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto msg = channel->cpp_channel->receive<ByteVector>(
            std::chrono::milliseconds(timeout_ms));

        if (!msg) {
            return timeout_ms > 0 ? PSYNE_ERROR_TIMEOUT
                                  : PSYNE_ERROR_NO_MESSAGE;
        }

        // Create message wrapper
        auto msg_wrapper = std::make_unique<psyne_message>();
        msg_wrapper->channel = channel;
        msg_wrapper->size = msg->size();
        msg_wrapper->type = 0; // ByteVector doesn't carry type info
        msg_wrapper->is_send = false;

        // Allocate memory for received data
        msg_wrapper->data = new uint8_t[msg_wrapper->size];
        std::memcpy(msg_wrapper->data, msg->data(), msg_wrapper->size);

        *message = msg_wrapper.release();
        *type = (*message)->type;
        return PSYNE_OK;
    } catch (...) {
        return handle_exception();
    }
}

void psyne_message_release(psyne_message_t *message) {
    if (message && !message->is_send) {
        delete[] static_cast<uint8_t *>(message->data);
        delete message;
    }
}

// Utility functions
psyne_error_t psyne_send_data(psyne_channel_t *channel, const void *data,
                              size_t size, uint32_t type) {
    if (!channel || !data) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        ByteVector msg(*channel->cpp_channel);
        msg.resize(size);
        std::memcpy(msg.data(), data, size);
        channel->cpp_channel->send(msg);
        return PSYNE_OK;
    } catch (...) {
        return handle_exception();
    }
}

psyne_error_t psyne_receive_data(psyne_channel_t *channel, void *buffer,
                                 size_t buffer_size, size_t *received_size,
                                 uint32_t *type, uint32_t timeout_ms) {
    if (!channel || !buffer || !received_size || !type) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto msg = channel->cpp_channel->receive<ByteVector>(
            std::chrono::milliseconds(timeout_ms));

        if (!msg) {
            return timeout_ms > 0 ? PSYNE_ERROR_TIMEOUT
                                  : PSYNE_ERROR_NO_MESSAGE;
        }

        if (msg->size() > buffer_size) {
            return PSYNE_ERROR_INVALID_ARGUMENT;
        }

        std::memcpy(buffer, msg->data(), msg->size());
        *received_size = msg->size();
        *type = 0; // ByteVector doesn't carry type info

        return PSYNE_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Advanced features
psyne_error_t
psyne_channel_set_receive_callback(psyne_channel_t *channel,
                                   psyne_receive_callback_t callback,
                                   void *user_data) {
    if (!channel) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    // Stop existing callback thread
    if (channel->callback_running) {
        channel->callback_running = false;
        if (channel->callback_thread && channel->callback_thread->joinable()) {
            channel->callback_thread->join();
        }
        channel->callback_thread.reset();
    }

    channel->callback = callback;
    channel->callback_user_data = user_data;

    // Start new callback thread if callback is set
    if (callback) {
        channel->callback_running = true;
        channel->callback_thread = std::make_unique<std::thread>([channel]() {
            while (channel->callback_running &&
                   !channel->cpp_channel->is_stopped()) {
                auto msg = channel->cpp_channel->receive<ByteVector>(
                    std::chrono::milliseconds(100));

                if (msg) {
                    // Create temporary message wrapper
                    psyne_message temp_msg;
                    temp_msg.channel = channel;
                    temp_msg.data = msg->data();
                    temp_msg.size = msg->size();
                    temp_msg.type = 0;
                    temp_msg.is_send = false;

                    // Call user callback
                    channel->callback(&temp_msg, temp_msg.type,
                                      channel->callback_user_data);
                }
            }
        });
    }

    return PSYNE_OK;
}

// Additional functions for completeness
psyne_error_t psyne_channel_enable_metrics(psyne_channel_t *channel,
                                           bool enable) {
    if (!channel) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    // Metrics are enabled at creation time in current implementation
    return enable ? PSYNE_OK : PSYNE_ERROR_UNSUPPORTED;
}

psyne_error_t psyne_channel_reset_metrics(psyne_channel_t *channel) {
    if (!channel) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    try {
        channel->cpp_channel->reset_metrics();
        return PSYNE_OK;
    } catch (...) {
        return handle_exception();
    }
}

psyne_error_t psyne_channel_get_buffer_size(psyne_channel_t *channel,
                                            size_t *size) {
    if (!channel || !size) {
        return PSYNE_ERROR_INVALID_ARGUMENT;
    }

    // This would require adding a method to the C++ API
    return PSYNE_ERROR_UNSUPPORTED;
}