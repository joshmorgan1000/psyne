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

// Explicit instantiations for common types
template std::optional<FloatVector> Channel::receive<FloatVector>(std::chrono::milliseconds);
template std::optional<DoubleMatrix> Channel::receive<DoubleMatrix>(std::chrono::milliseconds);

// Fixed-size matrix template instantiations
template std::optional<types::Matrix4x4f> Channel::receive<types::Matrix4x4f>(std::chrono::milliseconds);
template std::optional<types::Matrix3x3f> Channel::receive<types::Matrix3x3f>(std::chrono::milliseconds);
template std::optional<types::Matrix2x2f> Channel::receive<types::Matrix2x2f>(std::chrono::milliseconds);
template std::optional<types::Vector4f> Channel::receive<types::Vector4f>(std::chrono::milliseconds);
template std::optional<types::Vector3f> Channel::receive<types::Vector3f>(std::chrono::milliseconds);

template void Channel::send<types::Matrix4x4f>(types::Matrix4x4f&);
template void Channel::send<types::Matrix3x3f>(types::Matrix3x3f&);
template void Channel::send<types::Matrix2x2f>(types::Matrix2x2f&);
template void Channel::send<types::Vector4f>(types::Vector4f&);
template void Channel::send<types::Vector3f>(types::Vector3f&);

// Quantized vector template instantiations
template std::optional<types::Int8Vector> Channel::receive<types::Int8Vector>(std::chrono::milliseconds);
template std::optional<types::UInt8Vector> Channel::receive<types::UInt8Vector>(std::chrono::milliseconds);
template void Channel::send<types::Int8Vector>(types::Int8Vector&);
template void Channel::send<types::UInt8Vector>(types::UInt8Vector&);

// Complex vector template instantiations
template std::optional<types::ComplexVector<float>> Channel::receive<types::ComplexVector<float>>(std::chrono::milliseconds);
template std::optional<types::ComplexVector<double>> Channel::receive<types::ComplexVector<double>>(std::chrono::milliseconds);
template void Channel::send<types::ComplexVector<float>>(types::ComplexVector<float>&);
template void Channel::send<types::ComplexVector<double>>(types::ComplexVector<double>&);

// ML tensor template instantiations
template std::optional<types::MLTensor<float>> Channel::receive<types::MLTensor<float>>(std::chrono::milliseconds);
template std::optional<types::MLTensor<double>> Channel::receive<types::MLTensor<double>>(std::chrono::milliseconds);
template std::optional<types::MLTensor<int32_t>> Channel::receive<types::MLTensor<int32_t>>(std::chrono::milliseconds);
template std::optional<types::MLTensor<int8_t>> Channel::receive<types::MLTensor<int8_t>>(std::chrono::milliseconds);
template std::optional<types::MLTensor<uint8_t>> Channel::receive<types::MLTensor<uint8_t>>(std::chrono::milliseconds);
template void Channel::send<types::MLTensor<float>>(types::MLTensor<float>&);
template void Channel::send<types::MLTensor<double>>(types::MLTensor<double>&);
template void Channel::send<types::MLTensor<int32_t>>(types::MLTensor<int32_t>&);
template void Channel::send<types::MLTensor<int8_t>>(types::MLTensor<int8_t>&);
template void Channel::send<types::MLTensor<uint8_t>>(types::MLTensor<uint8_t>&);

// Sparse matrix template instantiations
template std::optional<types::SparseMatrix<float>> Channel::receive<types::SparseMatrix<float>>(std::chrono::milliseconds);
template std::optional<types::SparseMatrix<double>> Channel::receive<types::SparseMatrix<double>>(std::chrono::milliseconds);
template std::optional<types::SparseMatrix<int32_t>> Channel::receive<types::SparseMatrix<int32_t>>(std::chrono::milliseconds);
template void Channel::send<types::SparseMatrix<float>>(types::SparseMatrix<float>&);
template void Channel::send<types::SparseMatrix<double>>(types::SparseMatrix<double>&);
template void Channel::send<types::SparseMatrix<int32_t>>(types::SparseMatrix<int32_t>&);

#ifdef PSYNE_GPU_SUPPORT
// Forward declarations for GPU types
namespace gpu {
    class GPUFloatVector;
    class GPUMatrix;
    class GPUTensor;
}

// GPU message template instantiations
template std::optional<gpu::GPUFloatVector> Channel::receive<gpu::GPUFloatVector>(std::chrono::milliseconds);
template std::optional<gpu::GPUMatrix> Channel::receive<gpu::GPUMatrix>(std::chrono::milliseconds);
template std::optional<gpu::GPUTensor> Channel::receive<gpu::GPUTensor>(std::chrono::milliseconds);

template void Channel::send<gpu::GPUFloatVector>(gpu::GPUFloatVector&);
template void Channel::send<gpu::GPUMatrix>(gpu::GPUMatrix&);
template void Channel::send<gpu::GPUTensor>(gpu::GPUTensor&);
#endif

} // namespace psyne