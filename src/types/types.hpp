#pragma once

/**
 * @file types.hpp
 * @brief Complete enhanced message types for Psyne
 * 
 * This header includes all enhanced message types:
 * - Fixed-size matrices and vectors
 * - Quantized types for ML inference
 * - Complex number vectors
 * - ML tensors with layout support
 * - Sparse matrices
 */

#include "enhanced_types.hpp"
#include "advanced_types.hpp"

namespace psyne {
namespace types {

// Type aliases for convenience
using Matrix4f = Matrix4x4f;
using Matrix3f = Matrix3x3f;
using Vec3f = Vector3f;
using Vec4f = Vector4f;

using ComplexF = ComplexVectorF;
using MLTensor = MLTensorF;
using SparseMatrix = SparseMatrixF;

// Quantized type aliases
using QInt8 = Int8Vector;
using QUInt8 = UInt8Vector;

} // namespace types
} // namespace psyne