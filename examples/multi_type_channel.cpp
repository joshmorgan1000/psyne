#include <psyne/psyne.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <random>

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
        if (!is_valid()) return;
        cmd_ = cmd;
        parameter_ = param;
    }
    
    Command command() const { return cmd_; }
    float parameter() const { return parameter_; }
    
    static constexpr size_t calculate_size() {
        return sizeof(Command) + sizeof(float) + 16; // Some padding
    }
    
private:
    friend class Message<ControlMessage>;
    
    void initialize_storage(void* ptr) {
        cmd_ = Start;
        parameter_ = 0.0f;
    }
    
    void initialize_view(void* ptr) {
        // In a real implementation, we'd deserialize from ptr
        // For this example, we'll use member variables
    }
    
    Command cmd_ = Start;
    float parameter_ = 0.0f;
};

// Simulate a system that processes different message types
class DataProcessor {
public:
    DataProcessor(MPMCChannel& channel) 
        : channel_(channel), running_(false), sample_rate_(100.0f) {}
    
    void start() {
        running_ = true;
        
        // Start the event listener with handlers for each message type
        listener_ = channel_.listen({
            // Handle sensor data
            Channel<MPMCRingBuffer>::make_handler<FloatVector>(
                [this](FloatVector&& data) { handle_sensor_data(std::move(data)); }
            ),
            
            // Handle matrix operations
            Channel<MPMCRingBuffer>::make_handler<DoubleMatrix>(
                [this](DoubleMatrix&& matrix) { handle_matrix(std::move(matrix)); }
            ),
            
            // Handle control messages
            Channel<MPMCRingBuffer>::make_handler<ControlMessage>(
                [this](ControlMessage&& msg) { handle_control(std::move(msg)); }
            )
        });
        
        std::cout << "[Processor] Started listening for messages\n";
    }
    
    void stop() {
        running_ = false;
        channel_.stop();
        if (listener_) {
            listener_->join();
        }
        std::cout << "[Processor] Stopped\n";
    }
    
private:
    void handle_sensor_data(FloatVector&& data) {
        if (!processing_enabled_) return;
        
        // Calculate statistics
        float sum = 0.0f, min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
        
        for (float val : data) {
            sum += val;
            min = std::min(min, val);
            max = std::max(max, val);
        }
        
        float mean = sum / data.size();
        
        sensor_count_++;
        if (sensor_count_ % 50 == 0) {
            std::cout << "[Processor] Sensor data: " << data.size() 
                      << " samples, mean=" << mean 
                      << ", range=[" << min << ", " << max << "]\n";
        }
    }
    
    void handle_matrix(DoubleMatrix&& matrix) {
        if (!processing_enabled_) return;
        
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
        std::cout << "[Processor] Matrix " << matrix.rows() << "x" << matrix.cols() 
                  << ", Frobenius norm = " << norm << "\n";
    }
    
    void handle_control(ControlMessage&& msg) {
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
                std::cout << "[Processor] Sample rate set to " << sample_rate_ << " Hz\n";
                break;
        }
    }
    
    MPMCChannel& channel_;
    std::unique_ptr<std::thread> listener_;
    std::atomic<bool> running_;
    bool processing_enabled_ = false;
    float sample_rate_;
    size_t sensor_count_ = 0;
    size_t matrix_count_ = 0;
};

// Generate different types of messages
void message_generator(MPMCChannel& channel, std::atomic<bool>& running) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::cout << "[Generator] Starting message generation\n";
    
    // Send initial control message to start processing
    ControlMessage start_msg(channel);
    start_msg.set_command(ControlMessage::Start);
    channel.send(start_msg);
    
    size_t iteration = 0;
    
    while (running.load()) {
        iteration++;
        
        // Send sensor data frequently
        if (iteration % 10 == 0) {
            FloatVector sensor_data(channel);
            if (sensor_data.is_valid()) {
                sensor_data.resize(16);
                for (size_t i = 0; i < 16; ++i) {
                    sensor_data[i] = dist(gen);
                }
                channel.send(sensor_data);
            }
        }
        
        // Send matrix data occasionally
        if (iteration % 50 == 0) {
            DoubleMatrix matrix(channel);
            if (matrix.is_valid()) {
                matrix.set_dimensions(4, 4);
                for (size_t i = 0; i < 4; ++i) {
                    for (size_t j = 0; j < 4; ++j) {
                        matrix.at(i, j) = dist(gen);
                    }
                }
                channel.send(matrix);
            }
        }
        
        // Send control messages rarely
        if (iteration % 200 == 0) {
            ControlMessage ctrl(channel);
            if (ctrl.is_valid()) {
                if (iteration % 400 == 0) {
                    ctrl.set_command(ControlMessage::Reset);
                } else {
                    ctrl.set_command(ControlMessage::SetParameter, 50.0f + iteration);
                }
                channel.send(ctrl);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Send stop message
    ControlMessage stop_msg(channel);
    stop_msg.set_command(ControlMessage::Stop);
    channel.send(stop_msg);
    
    std::cout << "[Generator] Stopped\n";
}

int main() {
    std::cout << "Psyne Multi-Type Channel Example\n";
    std::cout << "================================\n\n";
    
    // Create a multi-type channel with MPMC support
    MPMCChannel channel("local://control", 50 * 1024 * 1024, ChannelType::MultiType);
    
    std::cout << "Created multi-type channel with 50MB buffer\n";
    std::cout << "Channel supports multiple message types with 8-byte overhead per message\n\n";
    
    // Create the data processor
    DataProcessor processor(channel);
    processor.start();
    
    // Start message generation
    std::atomic<bool> running{true};
    std::thread generator(message_generator, std::ref(channel), std::ref(running));
    
    // Run for 5 seconds
    std::cout << "Running for 5 seconds...\n\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // Stop everything
    std::cout << "\nStopping...\n";
    running = false;
    
    generator.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Let last messages process
    processor.stop();
    
    std::cout << "\nExample completed successfully!\n";
    return 0;
}