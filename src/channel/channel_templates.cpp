// Template implementations for Channel methods
// This file provides explicit instantiations for common message types

#include <psyne/psyne.hpp>
#include "channel_impl.hpp"

namespace psyne {

template<typename MessageType>
std::optional<MessageType> Channel::receive(std::chrono::milliseconds timeout) {
    size_t size;
    uint32_t type_id;
    
    // Get message from implementation
    void* data = impl()->receive_message(size, type_id);
    if (!data) {
        return std::nullopt;
    }
    
    // Verify type matches (for safety)
    if (type() == ChannelType::MultiType && type_id != MessageType::message_type) {
        impl()->release_message(data);
        return std::nullopt;
    }
    
    // Create message view
    return MessageType(data, size);
}

// Explicit instantiations for common types
template std::optional<FloatVector> Channel::receive<FloatVector>(std::chrono::milliseconds);
template std::optional<DoubleMatrix> Channel::receive<DoubleMatrix>(std::chrono::milliseconds);

} // namespace psyne