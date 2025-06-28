#include <psyne/psyne.hpp>

namespace psyne {

// Basic implementation stubs for Message class
template<typename Derived>
Message<Derived>::Message(Channel& channel) 
    : data_(nullptr), size_(0), channel_(&channel), handle_(nullptr) {
    // Basic allocation stub - for real implementation would allocate from channel buffer
    size_ = Derived::calculate_size();
    data_ = new uint8_t[size_];
    std::memset(data_, 0, size_);
}

template<typename Derived>
Message<Derived>::Message(const void* data, size_t size) 
    : data_(const_cast<uint8_t*>(static_cast<const uint8_t*>(data))), 
      size_(size), channel_(nullptr), handle_(nullptr) {}

template<typename Derived>
Message<Derived>::Message(Message&& other) noexcept 
    : data_(other.data_), size_(other.size_), channel_(other.channel_), handle_(other.handle_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.channel_ = nullptr;
    other.handle_ = nullptr;
}

template<typename Derived>
Message<Derived>& Message<Derived>::operator=(Message&& other) noexcept {
    if (this != &other) {
        if (data_ && !channel_) delete[] data_;  // Only delete if we allocated it
        data_ = other.data_;
        size_ = other.size_;
        channel_ = other.channel_;
        handle_ = other.handle_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.channel_ = nullptr;
        other.handle_ = nullptr;
    }
    return *this;
}

template<typename Derived>
Message<Derived>::~Message() {
    if (data_ && !channel_) {
        delete[] data_;  // Only delete if we allocated it
    }
}

template<typename Derived>
void Message<Derived>::send() {
    if (channel_) {
        before_send();
        // Stub implementation - in real implementation would put message on channel
    }
}

// Channel implementation stubs
std::unique_ptr<Channel> Channel::create(
    const std::string& uri,
    size_t buffer_size,
    ChannelMode mode,
    ChannelType type,
    bool enable_metrics,
    const compression::CompressionConfig& compression_config
) {
    return std::make_unique<Channel>(uri, buffer_size, type);
}

// Note: Explicit template instantiations are done in the test files where the classes are defined

// Add basic function implementations
const char* get_version() {
    return "1.0.0";
}

void print_banner() {
    // Empty implementation
}

// FloatVector implementations
float& FloatVector::operator[](size_t index) {
    float* floats = reinterpret_cast<float*>(data());
    return floats[index];
}

const float& FloatVector::operator[](size_t index) const {
    const float* floats = reinterpret_cast<const float*>(data());
    return floats[index];
}

float* FloatVector::begin() {
    return reinterpret_cast<float*>(data());
}

float* FloatVector::end() {
    return begin() + size();
}

const float* FloatVector::begin() const {
    return reinterpret_cast<const float*>(data());
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

} // namespace psyne