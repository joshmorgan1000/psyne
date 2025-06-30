#include <psyne/psyne.hpp>

// Custom message types for routing demo
class StatusMessage : public psyne::Message<StatusMessage> {
public:
    static constexpr uint32_t message_type = 100;
    using Message<StatusMessage>::Message;

    static size_t calculate_size() {
        return 256;
    }
    void initialize() {}
};

class AlertMessage : public psyne::Message<AlertMessage> {
public:
    static constexpr uint32_t message_type = 200;
    using Message<AlertMessage>::Message;

    static size_t calculate_size() {
        return sizeof(int) + 256;
    }
    void initialize() {}
};

// Explicit template instantiations
namespace psyne {
template class Message<StatusMessage>;
template class Message<AlertMessage>;
} // namespace psyne