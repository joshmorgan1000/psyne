#include <psyne/psyne.hpp>

// Custom message types for routing demo
class StatusMessage : public psyne::Message<StatusMessage> {
public:
    static constexpr uint32_t message_type = 100;
    using Message<StatusMessage>::Message;

    struct Data {
        uint32_t node_id;
        uint32_t timestamp;
        float cpu_usage;
        float memory_usage;
        char hostname[240]; // Padding to make 256 bytes total
    };

    static size_t calculate_size() {
        return sizeof(Data);
    }
    
    void initialize() {
        auto& data = *reinterpret_cast<Data*>(Message::data());
        data.node_id = 0;
        data.timestamp = 0;
        data.cpu_usage = 0.0f;
        data.memory_usage = 0.0f;
        std::memset(data.hostname, 0, sizeof(data.hostname));
    }
    
    Data& status_data() { return *reinterpret_cast<Data*>(Message::data()); }
    const Data& status_data() const { return *reinterpret_cast<const Data*>(Message::data()); }
};

class AlertMessage : public psyne::Message<AlertMessage> {
public:
    static constexpr uint32_t message_type = 200;
    using Message<AlertMessage>::Message;

    struct Data {
        uint32_t alert_id;
        uint32_t severity; // 1=low, 2=medium, 3=high, 4=critical
        uint32_t timestamp;
        uint32_t source_node;
        char message[240]; // Alert message text
    };

    static size_t calculate_size() {
        return sizeof(Data);
    }
    
    void initialize() {
        auto& data = *reinterpret_cast<Data*>(Message::data());
        data.alert_id = 0;
        data.severity = 1;
        data.timestamp = 0;
        data.source_node = 0;
        std::memset(data.message, 0, sizeof(data.message));
    }
    
    Data& alert_data() { return *reinterpret_cast<Data*>(Message::data()); }
    const Data& alert_data() const { return *reinterpret_cast<const Data*>(Message::data()); }
};

// Explicit template instantiations
namespace psyne {
template class Message<StatusMessage>;
template class Message<AlertMessage>;
} // namespace psyne