#include "../../include/psyne/types/fixed_matrices.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>

namespace psyne {
namespace types {

// Matrix4x4f implementation
void Matrix4x4f::initialize() {
    // Initialize to identity matrix
    if (this->data()) {
        std::memset(this->data(), 0, calculate_size());
        auto eigen_mat = as_eigen();
        eigen_mat.setIdentity();
    }
}

float& Matrix4x4f::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix4x4f index out of range");
    }
    return data()[index(row, col)];
}

const float& Matrix4x4f::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix4x4f index out of range");
    }
    return data()[index(row, col)];
}

float& Matrix4x4f::at(size_t row, size_t col) {
    return (*this)(row, col);
}

const float& Matrix4x4f::at(size_t row, size_t col) const {
    return (*this)(row, col);
}

float* Matrix4x4f::data() {
    return reinterpret_cast<float*>(this->Message<Matrix4x4f>::data());
}

const float* Matrix4x4f::data() const {
    return reinterpret_cast<const float*>(this->Message<Matrix4x4f>::data());
}

// Mathematical operations that return new matrices are not supported
// in zero-copy messaging due to move-only semantics

Matrix4x4f& Matrix4x4f::operator*=(const Matrix4x4f& other) {
    as_eigen() *= other.as_eigen();
    return *this;
}


Matrix4x4f& Matrix4x4f::operator+=(const Matrix4x4f& other) {
    as_eigen() += other.as_eigen();
    return *this;
}


Matrix4x4f& Matrix4x4f::operator-=(const Matrix4x4f& other) {
    as_eigen() -= other.as_eigen();
    return *this;
}


Matrix4x4f& Matrix4x4f::operator*=(float scalar) {
    as_eigen() *= scalar;
    return *this;
}


float Matrix4x4f::determinant() const {
    return as_eigen().determinant();
}


float Matrix4x4f::trace() const {
    return as_eigen().trace();
}

Matrix4x4f Matrix4x4f::identity() {
    // This is a placeholder - in real usage, you'd need a channel
    throw std::runtime_error("Static factory methods need channel context");
}

Eigen::Map<Eigen::Matrix4f> Matrix4x4f::as_eigen() {
    return Eigen::Map<Eigen::Matrix4f>(data());
}

Eigen::Map<const Eigen::Matrix4f> Matrix4x4f::as_eigen() const {
    return Eigen::Map<const Eigen::Matrix4f>(data());
}

// Graphics operations (static factory methods - simplified for now)
Matrix4x4f Matrix4x4f::translation(float x, float y, float z) {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix4x4f Matrix4x4f::rotation_x(float angle_radians) {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix4x4f Matrix4x4f::rotation_y(float angle_radians) {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix4x4f Matrix4x4f::rotation_z(float angle_radians) {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix4x4f Matrix4x4f::scale(float x, float y, float z) {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix4x4f Matrix4x4f::perspective(float fov, float aspect, float near, float far) {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix4x4f Matrix4x4f::orthographic(float left, float right, float bottom, float top, float near, float far) {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix4x4f Matrix4x4f::look_at(const float eye[3], const float center[3], const float up[3]) {
    throw std::runtime_error("Static factory methods need channel context");
}

// Matrix3x3f implementation
void Matrix3x3f::initialize() {
    if (this->data()) {
        std::memset(this->data(), 0, calculate_size());
        as_eigen().setIdentity();
    }
}

float& Matrix3x3f::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix3x3f index out of range");
    }
    return data()[index(row, col)];
}

const float& Matrix3x3f::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix3x3f index out of range");
    }
    return data()[index(row, col)];
}

float& Matrix3x3f::at(size_t row, size_t col) {
    return (*this)(row, col);
}

const float& Matrix3x3f::at(size_t row, size_t col) const {
    return (*this)(row, col);
}

float* Matrix3x3f::data() {
    return reinterpret_cast<float*>(this->Message<Matrix3x3f>::data());
}

const float* Matrix3x3f::data() const {
    return reinterpret_cast<const float*>(this->Message<Matrix3x3f>::data());
}


Matrix3x3f& Matrix3x3f::operator*=(const Matrix3x3f& other) {
    as_eigen() *= other.as_eigen();
    return *this;
}


Matrix3x3f& Matrix3x3f::operator+=(const Matrix3x3f& other) {
    as_eigen() += other.as_eigen();
    return *this;
}


Matrix3x3f& Matrix3x3f::operator-=(const Matrix3x3f& other) {
    as_eigen() -= other.as_eigen();
    return *this;
}


Matrix3x3f& Matrix3x3f::operator*=(float scalar) {
    as_eigen() *= scalar;
    return *this;
}


float Matrix3x3f::determinant() const {
    return as_eigen().determinant();
}


float Matrix3x3f::trace() const {
    return as_eigen().trace();
}

Matrix3x3f Matrix3x3f::identity() {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix3x3f Matrix3x3f::translation(float x, float y) {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix3x3f Matrix3x3f::rotation(float angle_radians) {
    throw std::runtime_error("Static factory methods need channel context");
}

Matrix3x3f Matrix3x3f::scale(float x, float y) {
    throw std::runtime_error("Static factory methods need channel context");
}

Eigen::Map<Eigen::Matrix3f> Matrix3x3f::as_eigen() {
    return Eigen::Map<Eigen::Matrix3f>(data());
}

Eigen::Map<const Eigen::Matrix3f> Matrix3x3f::as_eigen() const {
    return Eigen::Map<const Eigen::Matrix3f>(data());
}

// Matrix2x2f implementation
void Matrix2x2f::initialize() {
    if (this->data()) {
        std::memset(this->data(), 0, calculate_size());
        as_eigen().setIdentity();
    }
}

float& Matrix2x2f::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix2x2f index out of range");
    }
    return data()[index(row, col)];
}

const float& Matrix2x2f::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix2x2f index out of range");
    }
    return data()[index(row, col)];
}

float& Matrix2x2f::at(size_t row, size_t col) {
    return (*this)(row, col);
}

const float& Matrix2x2f::at(size_t row, size_t col) const {
    return (*this)(row, col);
}

float* Matrix2x2f::data() {
    return reinterpret_cast<float*>(this->Message<Matrix2x2f>::data());
}

const float* Matrix2x2f::data() const {
    return reinterpret_cast<const float*>(this->Message<Matrix2x2f>::data());
}

Eigen::Map<Eigen::Matrix2f> Matrix2x2f::as_eigen() {
    return Eigen::Map<Eigen::Matrix2f>(data());
}

Eigen::Map<const Eigen::Matrix2f> Matrix2x2f::as_eigen() const {
    return Eigen::Map<const Eigen::Matrix2f>(data());
}

// Similar implementations for Matrix2x2f operations...

Matrix2x2f& Matrix2x2f::operator*=(const Matrix2x2f& other) {
    as_eigen() *= other.as_eigen();
    return *this;
}

float Matrix2x2f::determinant() const {
    return as_eigen().determinant();
}


// Vector4f implementation
void Vector4f::initialize() {
    if (this->data()) {
        std::memset(this->data(), 0, calculate_size());
    }
}

float& Vector4f::operator[](size_t index) {
    if (index >= size_value) {
        throw std::out_of_range("Vector4f index out of range");
    }
    return data()[index];
}

const float& Vector4f::operator[](size_t index) const {
    if (index >= size_value) {
        throw std::out_of_range("Vector4f index out of range");
    }
    return data()[index];
}

float& Vector4f::at(size_t index) {
    return (*this)[index];
}

const float& Vector4f::at(size_t index) const {
    return (*this)[index];
}

float* Vector4f::data() {
    return reinterpret_cast<float*>(this->Message<Vector4f>::data());
}

const float* Vector4f::data() const {
    return reinterpret_cast<const float*>(this->Message<Vector4f>::data());
}


Vector4f& Vector4f::operator+=(const Vector4f& other) {
    as_eigen() += other.as_eigen();
    return *this;
}


Vector4f& Vector4f::operator-=(const Vector4f& other) {
    as_eigen() -= other.as_eigen();
    return *this;
}


Vector4f& Vector4f::operator*=(float scalar) {
    as_eigen() *= scalar;
    return *this;
}

float Vector4f::dot(const Vector4f& other) const {
    return as_eigen().dot(other.as_eigen());
}

float Vector4f::length() const {
    return as_eigen().norm();
}

float Vector4f::length_squared() const {
    return as_eigen().squaredNorm();
}


void Vector4f::normalize() {
    as_eigen().normalize();
}

Eigen::Map<Eigen::Vector4f> Vector4f::as_eigen() {
    return Eigen::Map<Eigen::Vector4f>(data());
}

Eigen::Map<const Eigen::Vector4f> Vector4f::as_eigen() const {
    return Eigen::Map<const Eigen::Vector4f>(data());
}

// Vector3f implementation
void Vector3f::initialize() {
    if (this->data()) {
        std::memset(this->data(), 0, calculate_size());
    }
}

float& Vector3f::operator[](size_t index) {
    if (index >= size_value) {
        throw std::out_of_range("Vector3f index out of range");
    }
    return data()[index];
}

const float& Vector3f::operator[](size_t index) const {
    if (index >= size_value) {
        throw std::out_of_range("Vector3f index out of range");
    }
    return data()[index];
}

float& Vector3f::at(size_t index) {
    return (*this)[index];
}

const float& Vector3f::at(size_t index) const {
    return (*this)[index];
}

float* Vector3f::data() {
    return reinterpret_cast<float*>(this->Message<Vector3f>::data());
}

const float* Vector3f::data() const {
    return reinterpret_cast<const float*>(this->Message<Vector3f>::data());
}


Vector3f& Vector3f::operator+=(const Vector3f& other) {
    as_eigen() += other.as_eigen();
    return *this;
}


Vector3f& Vector3f::operator-=(const Vector3f& other) {
    as_eigen() -= other.as_eigen();
    return *this;
}


Vector3f& Vector3f::operator*=(float scalar) {
    as_eigen() *= scalar;
    return *this;
}

float Vector3f::dot(const Vector3f& other) const {
    return as_eigen().dot(other.as_eigen());
}


float Vector3f::length() const {
    return as_eigen().norm();
}

float Vector3f::length_squared() const {
    return as_eigen().squaredNorm();
}


void Vector3f::normalize() {
    as_eigen().normalize();
}

void Vector3f::cross_assign(const Vector3f& other) {
    as_eigen() = as_eigen().cross(other.as_eigen());
}

Eigen::Map<Eigen::Vector3f> Vector3f::as_eigen() {
    return Eigen::Map<Eigen::Vector3f>(data());
}

Eigen::Map<const Eigen::Vector3f> Vector3f::as_eigen() const {
    return Eigen::Map<const Eigen::Vector3f>(data());
}

} // namespace types
} // namespace psyne