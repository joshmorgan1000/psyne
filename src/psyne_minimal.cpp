#include <psyne/psyne.hpp>
#include <cstring>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <map>

namespace psyne {

// Simple message queue implementation for testing
class SimpleMessageQueue {
public:
    struct MessageData {
        std::vector<uint8_t> data;
        uint32_t type;
    };
    
    void push(const void* data, size_t size, uint32_t type) {
        std::lock_guard<std::mutex> lock(mutex_);
        MessageData msg;
        msg.data.resize(size);
        std::memcpy(msg.data.data(), data, size);
        msg.type = type;
        queue_.push(std::move(msg));
        cv_.notify_one();
    }
    
    std::unique_ptr<MessageData> pop(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            return nullptr;
        }
        auto msg = std::make_unique<MessageData>(std::move(queue_.front()));
        queue_.pop();
        return msg;
    }
    
private:
    std::queue<MessageData> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

// Global message queues for channels (simple implementation for testing)
static std::map<std::string, std::shared_ptr<SimpleMessageQueue>> g_message_queues;
static std::mutex g_queues_mutex;

// Extended Channel implementation with basic message passing
class TestChannel : public Channel {
public:
    TestChannel(const std::string& uri, size_t buffer_size, ChannelType type) 
        : Channel(uri, buffer_size, type) {
        std::lock_guard<std::mutex> lock(g_queues_mutex);
        auto it = g_message_queues.find(uri);
        if (it == g_message_queues.end()) {
            queue_ = std::make_shared<SimpleMessageQueue>();
            g_message_queues[uri] = queue_;
        } else {
            queue_ = it->second;
        }
        current_message_ = nullptr;
    }
    
    void send_raw_message(const void* data, size_t size, uint32_t type) override {
        if (queue_) {
            queue_->push(data, size, type);
        }
    }
    
    void* receive_raw_message(size_t& size, uint32_t& type) override {
        if (!queue_) return nullptr;
        
        current_message_ = queue_->pop(std::chrono::milliseconds(1000));
        if (!current_message_) return nullptr;
        
        size = current_message_->data.size();
        type = current_message_->type;
        return current_message_->data.data();
    }
    
    void release_raw_message(void* handle) override {
        current_message_.reset();
    }
    
private:
    std::shared_ptr<SimpleMessageQueue> queue_;
    std::unique_ptr<SimpleMessageQueue::MessageData> current_message_;
};

// Note: Message template methods are defined inline in psyne.hpp

// Channel implementation stubs
std::unique_ptr<Channel> Channel::create(
    const std::string& uri,
    size_t buffer_size,
    ChannelMode mode,
    ChannelType type,
    bool enable_metrics,
    const compression::CompressionConfig& compression_config
) {
    return std::make_unique<TestChannel>(uri, buffer_size, type);
}

// Explicit template instantiations for library message types only
// Test message types (TestMsg, SimpleMessage) are instantiated in test files

// SPSCRingBuffer implementation
SPSCRingBuffer* SPSCRingBuffer::current_instance_ = nullptr;

SPSCRingBuffer::SPSCRingBuffer(size_t capacity) : capacity_(capacity) {
    current_instance_ = this;
}

std::optional<WriteHandle> SPSCRingBuffer::reserve(size_t size) {
    if (size <= capacity_) {
        // Allocate memory for testing
        auto* buffer = new uint8_t[size];
        std::memset(buffer, 0, size);
        return WriteHandle{buffer, size};
    }
    return std::nullopt;
}

std::optional<ReadHandle> SPSCRingBuffer::read() {
    // For testing, return last written data
    if (last_write_data_) {
        return ReadHandle{last_write_data_, last_write_size_};
    }
    return std::nullopt;
}

void WriteHandle::commit() {
    // Store for testing
    if (auto* rb = SPSCRingBuffer::current_instance_) {
        rb->last_write_data_ = data;
        rb->last_write_size_ = size;
    }
}

// Add basic function implementations
const char* get_version() {
    return "1.0.0";
}

void print_banner() {
    // Empty implementation
}

// FloatVector implementations
float& FloatVector::operator[](size_t index) {
    float* floats = reinterpret_cast<float*>(data() + sizeof(size_t));
    return floats[index];
}

const float& FloatVector::operator[](size_t index) const {
    const float* floats = reinterpret_cast<const float*>(data() + sizeof(size_t));
    return floats[index];
}

float* FloatVector::begin() {
    return reinterpret_cast<float*>(data() + sizeof(size_t));
}

float* FloatVector::end() {
    return begin() + size();
}

const float* FloatVector::begin() const {
    return reinterpret_cast<const float*>(data() + sizeof(size_t));
}

const float* FloatVector::end() const {
    return begin() + size();
}

size_t FloatVector::size() const {
    // Assuming first 8 bytes store size
    if (!data()) return 0;
    return *reinterpret_cast<const size_t*>(data());
}

size_t FloatVector::capacity() const {
    return (Message<FloatVector>::size() - sizeof(size_t)) / sizeof(float);
}

void FloatVector::resize(size_t new_size) {
    if (data() && new_size <= capacity()) {
        *reinterpret_cast<size_t*>(data()) = new_size;
    }
}

FloatVector& FloatVector::operator=(std::initializer_list<float> values) {
    resize(values.size());
    std::copy(values.begin(), values.end(), begin());
    return *this;
}

void FloatVector::initialize() {
    if (data()) {
        *reinterpret_cast<size_t*>(data()) = 0;
    }
}

// ByteVector implementations
uint8_t& ByteVector::operator[](size_t index) {
    return data()[index];
}

const uint8_t& ByteVector::operator[](size_t index) const {
    return data()[index];
}

uint8_t* ByteVector::begin() {
    return data();
}

uint8_t* ByteVector::end() {
    return data() + size();
}

const uint8_t* ByteVector::begin() const {
    return data();
}

const uint8_t* ByteVector::end() const {
    return data() + size();
}

uint8_t* ByteVector::data() {
    return Message<ByteVector>::data() + sizeof(size_t);
}

const uint8_t* ByteVector::data() const {
    return Message<ByteVector>::data() + sizeof(size_t);
}

size_t ByteVector::size() const {
    if (!Message<ByteVector>::data()) return 0;
    return *reinterpret_cast<const size_t*>(Message<ByteVector>::data());
}

size_t ByteVector::capacity() const {
    return Message<ByteVector>::size() - sizeof(size_t);
}

void ByteVector::resize(size_t new_size) {
    if (Message<ByteVector>::data() && new_size <= capacity()) {
        *reinterpret_cast<size_t*>(Message<ByteVector>::data()) = new_size;
    }
}

void ByteVector::initialize() {
    if (Message<ByteVector>::data()) {
        *reinterpret_cast<size_t*>(Message<ByteVector>::data()) = 0;
    }
}

// DoubleMatrix implementations
void DoubleMatrix::set_dimensions(size_t rows, size_t cols) {
    if (Message<DoubleMatrix>::data()) {
        size_t* header = reinterpret_cast<size_t*>(Message<DoubleMatrix>::data());
        header[0] = rows;
        header[1] = cols;
    }
}

size_t DoubleMatrix::rows() const {
    if (!Message<DoubleMatrix>::data()) return 0;
    return reinterpret_cast<const size_t*>(Message<DoubleMatrix>::data())[0];
}

size_t DoubleMatrix::cols() const {
    if (!Message<DoubleMatrix>::data()) return 0;
    return reinterpret_cast<const size_t*>(Message<DoubleMatrix>::data())[1];
}

double& DoubleMatrix::at(size_t row, size_t col) {
    double* matrix_data = reinterpret_cast<double*>(Message<DoubleMatrix>::data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

const double& DoubleMatrix::at(size_t row, size_t col) const {
    const double* matrix_data = reinterpret_cast<const double*>(Message<DoubleMatrix>::data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

void DoubleMatrix::initialize() {
    if (Message<DoubleMatrix>::data()) {
        size_t* header = reinterpret_cast<size_t*>(Message<DoubleMatrix>::data());
        header[0] = 0; // rows
        header[1] = 0; // cols
    }
}

// Explicit template instantiations for library message types
// Test message types (TestMsg, SimpleMessage) will be instantiated in test files
template class Message<FloatVector>;
template class Message<ByteVector>;
template class Message<DoubleMatrix>;

} // namespace psyne