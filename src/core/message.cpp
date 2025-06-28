#include <psyne/psyne.hpp>
#include "../channel/channel_impl.hpp"
#include "../memory/ring_buffer_impl.hpp"  // For SlabHeader
#include <cstring>
#include <stdexcept>

namespace psyne {

// Message base class implementation
template<typename Derived>
Message<Derived>::Message(Channel& channel) 
    : data_(nullptr)
    , size_(0)
    , channel_(&channel)
    , handle_(nullptr) {
    // Calculate size needed
    size_t required_size = Derived::calculate_size();
    
    // Reserve space in channel
    auto* impl = channel.impl();
    handle_ = impl->reserve_space(required_size);
    
    if (handle_) {
        // Get the data pointer from handle
        // The handle is actually a SlabHeader pointer
        auto* header = static_cast<detail::SlabHeader*>(handle_);
        data_ = static_cast<uint8_t*>(header->data());
        size_ = required_size;
        
        // Let derived class initialize their data
        static_cast<Derived*>(this)->initialize();
    }
}

template<typename Derived>
Message<Derived>::Message(const void* data, size_t size)
    : data_(const_cast<uint8_t*>(static_cast<const uint8_t*>(data)))
    , size_(size)
    , channel_(nullptr)
    , handle_(nullptr) {
    // For incoming messages, data points directly to the message payload
    // (after any headers)
}

template<typename Derived>
Message<Derived>::Message(Message&& other) noexcept
    : data_(other.data_)
    , size_(other.size_)
    , channel_(other.channel_)
    , handle_(other.handle_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.channel_ = nullptr;
    other.handle_ = nullptr;
}

template<typename Derived>
Message<Derived>& Message<Derived>::operator=(Message&& other) noexcept {
    if (this != &other) {
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
    // If we have a handle and haven't sent, we need to release it
    // This is simplified - real implementation would handle this properly
}

template<typename Derived>
void Message<Derived>::send() {
    if (!channel_ || !handle_) {
        throw std::runtime_error("Cannot send message without channel");
    }
    
    before_send();
    
    auto* impl = channel_->impl();
    impl->commit_message(handle_);
    
    // Clear our state after sending
    data_ = nullptr;
    size_ = 0;
    handle_ = nullptr;
}

// FloatVector implementation
void FloatVector::initialize() {
    // Initialize size to 0
    if (data()) {
        *reinterpret_cast<size_t*>(data()) = 0;
    }
}

float& FloatVector::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("FloatVector index out of range");
    }
    return begin()[index];
}

const float& FloatVector::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("FloatVector index out of range");
    }
    return begin()[index];
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
    if (!data()) return 0;
    size_t stored_size = *reinterpret_cast<const size_t*>(data());
    // Sanity check - if size is unreasonably large, it's probably uninitialized
    if (stored_size > capacity()) {
        return 0;  // Return 0 for invalid size
    }
    return stored_size;
}

size_t FloatVector::capacity() const {
    if (!data()) return 0;
    // Calculate based on total size minus header
    return (Message<FloatVector>::size() - sizeof(size_t)) / sizeof(float);
}

void FloatVector::resize(size_t new_size) {
    if (new_size > capacity()) {
        throw std::runtime_error("Cannot resize beyond capacity");
    }
    if (data()) {
        *reinterpret_cast<size_t*>(data()) = new_size;
    }
}

FloatVector& FloatVector::operator=(std::initializer_list<float> values) {
    resize(values.size());
    std::copy(values.begin(), values.end(), begin());
    return *this;
}

// DoubleMatrix implementation
void DoubleMatrix::initialize() {
    // Initialize dimensions to 0x0
    if (data()) {
        size_t* header = reinterpret_cast<size_t*>(data());
        header[0] = 0;  // rows
        header[1] = 0;  // cols
    }
}

void DoubleMatrix::set_dimensions(size_t rows, size_t cols) {
    if (!data()) return;
    
    size_t* header = reinterpret_cast<size_t*>(data());
    header[0] = rows;
    header[1] = cols;
}

size_t DoubleMatrix::rows() const {
    if (!data()) return 0;
    return reinterpret_cast<const size_t*>(data())[0];
}

size_t DoubleMatrix::cols() const {
    if (!data()) return 0;
    return reinterpret_cast<const size_t*>(data())[1];
}

double& DoubleMatrix::at(size_t row, size_t col) {
    if (row >= rows() || col >= cols()) {
        throw std::out_of_range("DoubleMatrix index out of range");
    }
    double* matrix_data = reinterpret_cast<double*>(data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

const double& DoubleMatrix::at(size_t row, size_t col) const {
    if (row >= rows() || col >= cols()) {
        throw std::out_of_range("DoubleMatrix index out of range");
    }
    const double* matrix_data = reinterpret_cast<const double*>(data() + 2 * sizeof(size_t));
    return matrix_data[row * cols() + col];
}

// Fixed-size matrix types
namespace types {
    class Matrix4x4f;
    class Matrix3x3f;
    class Matrix2x2f;
    class Vector4f;
    class Vector3f;
    class Int8Vector;
    class UInt8Vector;
    template<typename T> class ComplexVector;
    template<typename T> class MLTensor;
    template<typename T> class SparseMatrix;
}

// Explicit template instantiations
template class Message<FloatVector>;
template class Message<DoubleMatrix>;
template class Message<types::Matrix4x4f>;
template class Message<types::Matrix3x3f>;
template class Message<types::Matrix2x2f>;
template class Message<types::Vector4f>;
template class Message<types::Vector3f>;
template class Message<types::Int8Vector>;
template class Message<types::UInt8Vector>;
template class Message<types::ComplexVector<float>>;
template class Message<types::ComplexVector<double>>;
template class Message<types::MLTensor<float>>;
template class Message<types::MLTensor<double>>;
template class Message<types::MLTensor<int32_t>>;
template class Message<types::MLTensor<int8_t>>;
template class Message<types::MLTensor<uint8_t>>;
template class Message<types::SparseMatrix<float>>;
template class Message<types::SparseMatrix<double>>;
template class Message<types::SparseMatrix<int32_t>>;

#ifdef PSYNE_GPU_SUPPORT
// Forward declarations for GPU types
namespace gpu {
    class GPUFloatVector;
    class GPUMatrix;
    class GPUTensor;
}

// GPU message template instantiations
template class Message<gpu::GPUFloatVector>;
template class Message<gpu::GPUMatrix>;
template class Message<gpu::GPUTensor>;
#endif

} // namespace psyne