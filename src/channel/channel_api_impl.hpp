#pragma once

// Template implementations for Channel methods
// This file should be included by source files that need the template
// implementations

#include "channel_impl.hpp"
#include <psyne/psyne.hpp>

namespace psyne {

template <typename MessageType>
void Channel::send(MessageType &msg) {
    msg.send();
}

template <typename MessageType>
std::optional<MessageType> Channel::receive(std::chrono::milliseconds timeout) {
    size_t size;
    uint32_t type;

    auto *data = impl()->receive_message(size, type);
    if (!data) {
        return std::nullopt;
    }

    // Verify type matches
    if (type != MessageType::message_type) {
        impl()->release_message(data);
        return std::nullopt;
    }

    return MessageType(data, size);
}

template <typename MessageType>
std::unique_ptr<std::thread>
Channel::listen(std::function<void(MessageType &&)> handler) {
    return std::make_unique<std::thread>([this, handler]() {
        while (!is_stopped()) {
            auto msg = receive<MessageType>(std::chrono::milliseconds(100));
            if (msg) {
                handler(std::move(*msg));
            }
        }
    });
}

} // namespace psyne