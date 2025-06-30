#include <cassert>
#include <iostream>
#include <psyne/psyne.hpp>

using namespace psyne;

// Test message type
class TestMsg : public Message<TestMsg> {
public:
    static constexpr uint32_t message_type = 500;
    static constexpr size_t size = 64;

    template <typename Channel>
    explicit TestMsg(Channel &channel) : Message<TestMsg>(channel) {}

    explicit TestMsg(const void *data, size_t sz)
        : Message<TestMsg>(data, sz) {}

    static constexpr size_t calculate_size() {
        return size;
    }

    void set_value(uint64_t val) {
        if (data_) {
            *reinterpret_cast<uint64_t *>(data_) = val;
        }
    }

    uint64_t get_value() const {
        if (!data_)
            return 0;
        return *reinterpret_cast<const uint64_t *>(data_);
    }

    void before_send() {}
};

// Explicit template instantiation for TestMsg
template class psyne::Message<TestMsg>;

int main() {
    std::cout << "Simple synchronous test..." << std::endl;

    auto channel_ptr = Channel::create(
        "memory://simple", 1024, ChannelMode::SPSC, ChannelType::SingleType);
    auto &channel = *channel_ptr;

    // Send 3 messages
    for (int i = 0; i < 3; ++i) {
        TestMsg msg(channel);
        msg.set_value(i * 10);
        std::cout << "Sending: " << msg.get_value() << std::endl;
        msg.send();
    }

    // Receive 3 messages
    for (int i = 0; i < 3; ++i) {
        auto msg = channel.receive_single<TestMsg>();
        if (msg) {
            std::cout << "Received: " << msg->get_value() << std::endl;
            assert(msg->get_value() == static_cast<uint64_t>(i * 10));
        } else {
            std::cout << "No message!" << std::endl;
            assert(false);
        }
    }

    std::cout << "✓ Test passed!" << std::endl;
    return 0;
}