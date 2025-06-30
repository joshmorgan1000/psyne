#include "../channel/channel_impl.hpp"
#include "../memory/ring_buffer_impl.hpp" // For SlabHeader
#include <cstring>
#include <psyne/psyne.hpp>
#include <stdexcept>

namespace psyne {

// Message base class implementation
template <typename Derived>
Message<Derived>::Message(Channel &channel)
    : slab_(&channel.get_ring_buffer()), offset_(0), channel_(&channel) {
    // Reserve space in ring buffer and get offset - no allocation!
    offset_ = channel.reserve_write_slot(Derived::calculate_size());

    if (offset_ == BUFFER_FULL) {
        throw std::runtime_error("Ring buffer full - cannot reserve space");
    }

    // Message is now a typed view over ring buffer at offset
    // User writes directly to slab memory via data() method

    // Let derived class initialize their data structure in ring buffer
    static_cast<Derived *>(this)->initialize();
}

template <typename Derived>
Message<Derived>::Message(RingBuffer *slab, uint32_t offset)
    : slab_(slab), offset_(offset), channel_(nullptr) {
    // For incoming messages, create view into ring buffer at offset
    // Message is just a typed view over existing ring buffer data
}

template <typename Derived>
Message<Derived>::Message(Message &&other) noexcept
    : slab_(other.slab_), offset_(other.offset_), channel_(other.channel_) {
    other.slab_ = nullptr;
    other.offset_ = BUFFER_FULL;
    other.channel_ = nullptr;
}

template <typename Derived>
Message<Derived> &Message<Derived>::operator=(Message &&other) noexcept {
    if (this != &other) {
        slab_ = other.slab_;
        offset_ = other.offset_;
        channel_ = other.channel_;

        other.slab_ = nullptr;
        other.offset_ = BUFFER_FULL;
        other.channel_ = nullptr;
    }
    return *this;
}

template <typename Derived>
Message<Derived>::~Message() {
    // Message destructor - no cleanup needed since we don't own data
    // Data lives in ring buffer, message is just a view
}

template <typename Derived>
void Message<Derived>::send() {
    if (!channel_ || offset_ == BUFFER_FULL) {
        throw std::runtime_error("Cannot send invalid message");
    }

    // Data is already written by user directly to ring buffer
    // Just notify receiver that there's a message ready at this offset
    channel_->notify_message_ready(offset_, Derived::calculate_size());

    // Message object can be destroyed - data lives in ring buffer
    // No pointer nulling needed - message is just a view
}

// FloatVector implementation
void FloatVector::initialize() {
    // Initialize size to 0
    if (data()) {
        *reinterpret_cast<size_t *>(data()) = 0;
    }
}

float &FloatVector::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("FloatVector index out of range");
    }
    return begin()[index];
}

const float &FloatVector::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("FloatVector index out of range");
    }
    return begin()[index];
}

float *FloatVector::begin() {
    return reinterpret_cast<float *>(data() + sizeof(size_t));
}

float *FloatVector::end() {
    return begin() + size();
}

const float *FloatVector::begin() const {
    return reinterpret_cast<const float *>(data() + sizeof(size_t));
}

const float *FloatVector::end() const {
    return begin() + size();
}

size_t FloatVector::size() const {
    if (!data())
        return 0;
    size_t stored_size = *reinterpret_cast<const size_t *>(data());
    // Sanity check - if size is unreasonably large, it's probably uninitialized
    if (stored_size > capacity()) {
        return 0; // Return 0 for invalid size
    }
    return stored_size;
}

size_t FloatVector::capacity() const {
    if (!data())
        return 0;
    // Calculate based on total size minus header
    return (Message<FloatVector>::size() - sizeof(size_t)) / sizeof(float);
}

void FloatVector::resize(size_t new_size) {
    if (new_size > capacity()) {
        throw std::runtime_error("Cannot resize beyond capacity");
    }
    if (data()) {
        *reinterpret_cast<size_t *>(data()) = new_size;
    }
}

FloatVector &FloatVector::operator=(std::initializer_list<float> values) {
    resize(values.size());
    // Zero-copy optimization: manual copy from initializer list
    size_t i = 0;
    for (float value : values) {
        begin()[i++] = value;
    }
    return *this;
}

// DoubleMatrix implementation
void DoubleMatrix::initialize() {
    // Initialize dimensions to 0x0
    if (data()) {
        size_t *header = reinterpret_cast<size_t *>(data());
        header[0] = 0; // rows
        header[1] = 0; // cols
    }
}

void DoubleMatrix::set_dimensions(size_t rows, size_t cols) {
    if (!data())
        return;

    size_t *header = reinterpret_cast<size_t *>(data());
    header[0] = rows;
    header[1] = cols;
}

size_t DoubleMatrix::rows() const {
    if (!data())
        return 0;
    return reinterpret_cast<const size_t *>(data())[0];
}

size_t DoubleMatrix::cols() const {
    if (!data())
        return 0;
    return reinterpret_cast<const size_t *>(data())[1];
}

double &DoubleMatrix::at(size_t row, size_t col) {
    if (row >= rows() || col >= cols()) {
        throw std::out_of_range("DoubleMatrix index out of range");
    }
    double *matrix_data =
        reinterpret_cast<double *>(data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

const double &DoubleMatrix::at(size_t row, size_t col) const {
    if (row >= rows() || col >= cols()) {
        throw std::out_of_range("DoubleMatrix index out of range");
    }
    const double *matrix_data =
        reinterpret_cast<const double *>(data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

// ByteVector implementation
uint8_t &ByteVector::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("ByteVector index out of range");
    }
    return data()[index];
}

const uint8_t &ByteVector::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("ByteVector index out of range");
    }
    return data()[index];
}

uint8_t *ByteVector::begin() {
    return data();
}

uint8_t *ByteVector::end() {
    return data() + size();
}

const uint8_t *ByteVector::begin() const {
    return data();
}

const uint8_t *ByteVector::end() const {
    return data() + size();
}

uint8_t *ByteVector::data() {
    // Return pointer after size header
    return Message<ByteVector>::data() + sizeof(size_t);
}

const uint8_t *ByteVector::data() const {
    // Return pointer after size header
    return Message<ByteVector>::data() + sizeof(size_t);
}

size_t ByteVector::size() const {
    if (!Message<ByteVector>::data())
        return 0;
    size_t stored_size =
        *reinterpret_cast<const size_t *>(Message<ByteVector>::data());
    // Sanity check - if size is unreasonably large, it's probably uninitialized
    if (stored_size > capacity()) {
        return 0; // Return 0 for invalid size
    }
    return stored_size;
}

size_t ByteVector::capacity() const {
    if (!Message<ByteVector>::data())
        return 0;
    // Calculate based on total size minus header
    return Message<ByteVector>::size() - sizeof(size_t);
}

void ByteVector::resize(size_t new_size) {
    if (new_size > capacity()) {
        throw std::runtime_error(
            "Cannot resize ByteVector beyond allocated capacity");
    }
    if (Message<ByteVector>::data()) {
        *reinterpret_cast<size_t *>(Message<ByteVector>::data()) = new_size;
    }
}

void ByteVector::initialize() {
    // Initialize size to 0
    if (Message<ByteVector>::data()) {
        *reinterpret_cast<size_t *>(Message<ByteVector>::data()) = 0;
    }
}

// Explicit template instantiations for basic message types
template class Message<FloatVector>;
template class Message<DoubleMatrix>;
template class Message<ByteVector>;

} // namespace psyne