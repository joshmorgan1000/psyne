#include <algorithm>
#include <atomic>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <thread>

using namespace psyne;

// Custom message type for control commands
class ControlMessage : public Message<ControlMessage> {
public:
    static constexpr uint32_t message_type = 1001;

    using Message::Message;

    enum Command : uint32_t {
        Start = 1,
        Stop = 2,
        Reset = 3,
        SetParameter = 4
    };

    void set_command(Command cmd, float param = 0.0f) {
        if (!is_valid())
            return;
        // Write directly to buffer
        auto* header = reinterpret_cast<Header*>(data());
        header->cmd = cmd;
        header->parameter = param;
    }

    Command command() const {
        if (!is_valid())
            return Start;
        auto* header = reinterpret_cast<const Header*>(data());
        return header->cmd;
    }
    
    float parameter() const {
        if (!is_valid())
            return 0.0f;
        auto* header = reinterpret_cast<const Header*>(data());
        return header->parameter;
    }

    static constexpr size_t calculate_size() {
        return sizeof(Header);
    }

private:
    struct Header {
        Command cmd;
        float parameter;
        uint8_t padding[8]; // Align to 16 bytes
    };
};

// Simulate a system that processes different message types
class DataProcessor {
public:
    DataProcessor(Channel* channel)
        : channel_(channel), running_(false), sample_rate_(100.0f) {}

    void start() {
        running_ = true;
        processor_thread_ = std::thread([this]() {
            process_messages();
        });
        std::cout << "[Processor] Started listening for messages\n";
    }

    void stop() {
        running_ = false;
        if (processor_thread_.joinable()) {
            processor_thread_.join();
        }
        std::cout << "[Processor] Stopped\n";
    }

private:
    void handle_sensor_data(FloatVector &&data) {
        if (!processing_enabled_)
            return;

        // Calculate statistics
        float sum = 0.0f, min = std::numeric_limits<float>::max(),
              max = std::numeric_limits<float>::min();

        for (float val : data) {
            sum += val;
            min = std::min(min, val);
            max = std::max(max, val);
        }

        float mean = sum / data.size();

        sensor_count_++;
        if (sensor_count_ % 50 == 0) {
            std::cout << "[Processor] Sensor data: " << data.size()
                      << " samples, mean=" << mean << ", range=[" << min << ", "
                      << max << "]\n";
        }
    }

    void handle_matrix(DoubleMatrix &&matrix) {
        if (!processing_enabled_)
            return;

        // Calculate matrix norm
        double norm = 0.0;
        for (size_t i = 0; i < matrix.rows(); ++i) {
            for (size_t j = 0; j < matrix.cols(); ++j) {
                double val = matrix.at(i, j);
                norm += val * val;
            }
        }
        norm = std::sqrt(norm);

        matrix_count_++;
        std::cout << "[Processor] Matrix " << matrix.rows() << "x"
                  << matrix.cols() << ", Frobenius norm = " << norm << "\n";
    }

    void handle_control(ControlMessage &&msg) {
        switch (msg.command()) {
        case ControlMessage::Start:
            processing_enabled_ = true;
            std::cout << "[Processor] Processing STARTED\n";
            break;

        case ControlMessage::Stop:
            processing_enabled_ = false;
            std::cout << "[Processor] Processing STOPPED\n";
            break;

        case ControlMessage::Reset:
            sensor_count_ = 0;
            matrix_count_ = 0;
            std::cout << "[Processor] Counters RESET\n";
            break;

        case ControlMessage::SetParameter:
            sample_rate_ = msg.parameter();
            std::cout << "[Processor] Sample rate set to " << sample_rate_
                      << " Hz\n";
            break;
        }
    }

    void process_messages() {
        while (running_) {
            // Try to receive any message
            size_t msg_size;
            uint32_t msg_type;
            void* msg_data = channel_->receive_raw_message(msg_size, msg_type);
            
            if (!msg_data) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            // Process based on type
            switch (msg_type) {
            case FloatVector::message_type: {
                FloatVector vec(msg_data, msg_size);
                handle_sensor_data(std::move(vec));
                break;
            }
            case DoubleMatrix::message_type: {
                DoubleMatrix mat(msg_data, msg_size);
                handle_matrix(std::move(mat));
                break;
            }
            case ControlMessage::message_type: {
                ControlMessage ctrl(msg_data, msg_size);
                handle_control(std::move(ctrl));
                break;
            }
            default:
                std::cout << "[Processor] Unknown message type: " << msg_type << "\n";
            }
            
            // Release message back to channel
            channel_->release_raw_message(msg_data);
        }
    }

    Channel* channel_;
    std::thread processor_thread_;
    std::atomic<bool> running_;
    bool processing_enabled_ = false;
    float sample_rate_;
    size_t sensor_count_ = 0;
    size_t matrix_count_ = 0;
};

// Generate different types of messages
void message_generator(Channel* channel, std::atomic<bool> &running) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::cout << "[Generator] Starting message generation\n";

    // Send initial control message to start processing
    ControlMessage start_msg(*channel);
    start_msg.set_command(ControlMessage::Start);
    start_msg.send();

    size_t iteration = 0;

    while (running.load()) {
        iteration++;

        // Send sensor data frequently
        if (iteration % 10 == 0) {
            try {
                FloatVector sensor_data(*channel);
                sensor_data.resize(16);
                for (size_t i = 0; i < 16; ++i) {
                    sensor_data[i] = dist(gen);
                }
                sensor_data.send();
            } catch (const std::runtime_error& e) {
                // Buffer full, skip this message
            }
        }

        // Send matrix data occasionally
        if (iteration % 50 == 0) {
            try {
                DoubleMatrix matrix(*channel);
                matrix.set_dimensions(4, 4);
                for (size_t i = 0; i < 4; ++i) {
                    for (size_t j = 0; j < 4; ++j) {
                        matrix.at(i, j) = dist(gen);
                    }
                }
                matrix.send();
            } catch (const std::runtime_error& e) {
                // Buffer full, skip this message
            }
        }

        // Send control messages rarely
        if (iteration % 200 == 0) {
            try {
                ControlMessage ctrl(*channel);
                if (iteration % 400 == 0) {
                    ctrl.set_command(ControlMessage::Reset);
                } else {
                    ctrl.set_command(ControlMessage::SetParameter,
                                     50.0f + iteration);
                }
                ctrl.send();
            } catch (const std::runtime_error& e) {
                // Buffer full, skip this message
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Send stop message
    try {
        ControlMessage stop_msg(*channel);
        stop_msg.set_command(ControlMessage::Stop);
        stop_msg.send();
    } catch (const std::runtime_error& e) {
        // Ignore if buffer is full at shutdown
    }

    std::cout << "[Generator] Stopped\n";
}

int main() {
    std::cout << "Psyne Multi-Type Channel Example\n";
    std::cout << "================================\n\n";

    // Create a multi-type channel with MPMC support
    auto channel = create_channel("memory://control", 50 * 1024 * 1024,
                                  ChannelMode::MPMC, ChannelType::MultiType);

    std::cout << "Created multi-type channel with 50MB buffer\n";
    std::cout << "Channel supports multiple message types with 8-byte overhead "
                 "per message\n\n";

    // Create the data processor
    DataProcessor processor(channel.get());
    processor.start();

    // Start message generation
    std::atomic<bool> running{true};
    std::thread generator(message_generator, channel.get(),
                          std::ref(running));

    // Run for 5 seconds
    std::cout << "Running for 5 seconds...\n\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Stop everything
    std::cout << "\nStopping...\n";
    running = false;

    generator.join();
    std::this_thread::sleep_for(
        std::chrono::milliseconds(100)); // Let last messages process
    processor.stop();

    std::cout << "\nExample completed successfully!\n";
    return 0;
}
