#pragma once

#include "../psyne.hpp"
#include <Eigen/Dense>
#include <array>

namespace psyne {
namespace types {

// Fixed-size 4x4 float matrix (common in graphics)
class Matrix4x4f : public Message<Matrix4x4f> {
public:
    static constexpr uint32_t message_type = 20;
    static constexpr size_t rows = 4;
    static constexpr size_t cols = 4;
    static constexpr size_t total_elements = rows * cols;
    
    using Message<Matrix4x4f>::Message;
    
    // Calculate required size (fixed)
    static size_t calculate_size() {
        return total_elements * sizeof(float);
    }
    
    // Element access
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    
    // Raw data access
    float* data();
    const float* data() const;
    
    // In-place matrix operations (compatible with zero-copy messaging)
    Matrix4x4f& operator*=(const Matrix4x4f& other);
    Matrix4x4f& operator+=(const Matrix4x4f& other);
    Matrix4x4f& operator-=(const Matrix4x4f& other);
    Matrix4x4f& operator*=(float scalar);
    
    // Read-only operations that return scalars
    float determinant() const;
    float trace() const;
    
    // Graphics-specific operations
    static Matrix4x4f identity();
    static Matrix4x4f translation(float x, float y, float z);
    static Matrix4x4f rotation_x(float angle_radians);
    static Matrix4x4f rotation_y(float angle_radians);
    static Matrix4x4f rotation_z(float angle_radians);
    static Matrix4x4f scale(float x, float y, float z);
    static Matrix4x4f perspective(float fov, float aspect, float near, float far);
    static Matrix4x4f orthographic(float left, float right, float bottom, float top, float near, float far);
    static Matrix4x4f look_at(const float eye[3], const float center[3], const float up[3]);
    
    // Eigen integration
    Eigen::Map<Eigen::Matrix4f> as_eigen();
    Eigen::Map<const Eigen::Matrix4f> as_eigen() const;
    
    // Initialization
    void initialize();
    void before_send() override {}
    
private:
    size_t index(size_t row, size_t col) const { return row * cols + col; }
};

// Fixed-size 3x3 float matrix (rotations, 2D transforms)
class Matrix3x3f : public Message<Matrix3x3f> {
public:
    static constexpr uint32_t message_type = 21;
    static constexpr size_t rows = 3;
    static constexpr size_t cols = 3;
    static constexpr size_t total_elements = rows * cols;
    
    using Message<Matrix3x3f>::Message;
    
    static size_t calculate_size() {
        return total_elements * sizeof(float);
    }
    
    // Element access
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    
    float* data();
    const float* data() const;
    
    // In-place matrix operations
    Matrix3x3f& operator*=(const Matrix3x3f& other);
    Matrix3x3f& operator+=(const Matrix3x3f& other);
    Matrix3x3f& operator-=(const Matrix3x3f& other);
    Matrix3x3f& operator*=(float scalar);
    
    // Read-only operations
    float determinant() const;
    float trace() const;
    
    // 2D graphics operations
    static Matrix3x3f identity();
    static Matrix3x3f translation(float x, float y);
    static Matrix3x3f rotation(float angle_radians);
    static Matrix3x3f scale(float x, float y);
    
    // Eigen integration
    Eigen::Map<Eigen::Matrix3f> as_eigen();
    Eigen::Map<const Eigen::Matrix3f> as_eigen() const;
    
    void initialize();
    void before_send() override {}
    
private:
    size_t index(size_t row, size_t col) const { return row * cols + col; }
};

// Fixed-size 2x2 float matrix (simple 2D operations)
class Matrix2x2f : public Message<Matrix2x2f> {
public:
    static constexpr uint32_t message_type = 22;
    static constexpr size_t rows = 2;
    static constexpr size_t cols = 2;
    static constexpr size_t total_elements = rows * cols;
    
    using Message<Matrix2x2f>::Message;
    
    static size_t calculate_size() {
        return total_elements * sizeof(float);
    }
    
    // Element access
    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    
    float* data();
    const float* data() const;
    
    // In-place matrix operations
    Matrix2x2f& operator*=(const Matrix2x2f& other);
    Matrix2x2f& operator+=(const Matrix2x2f& other);
    Matrix2x2f& operator-=(const Matrix2x2f& other);
    Matrix2x2f& operator*=(float scalar);
    
    // Read-only operations
    float determinant() const;
    float trace() const;
    
    static Matrix2x2f identity();
    static Matrix2x2f rotation(float angle_radians);
    static Matrix2x2f scale(float x, float y);
    
    // Eigen integration
    Eigen::Map<Eigen::Matrix2f> as_eigen();
    Eigen::Map<const Eigen::Matrix2f> as_eigen() const;
    
    void initialize();
    void before_send() override {}
    
private:
    size_t index(size_t row, size_t col) const { return row * cols + col; }
};

// Fixed-size vectors for common graphics use cases
class Vector4f : public Message<Vector4f> {
public:
    static constexpr uint32_t message_type = 23;
    static constexpr size_t size_value = 4;
    
    using Message<Vector4f>::Message;
    
    static size_t calculate_size() {
        return size_value * sizeof(float);
    }
    
    // Element access
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float& at(size_t index);
    const float& at(size_t index) const;
    
    // Named accessors for graphics
    float& x() { return (*this)[0]; }
    float& y() { return (*this)[1]; }
    float& z() { return (*this)[2]; }
    float& w() { return (*this)[3]; }
    const float& x() const { return (*this)[0]; }
    const float& y() const { return (*this)[1]; }
    const float& z() const { return (*this)[2]; }
    const float& w() const { return (*this)[3]; }
    
    float* data();
    const float* data() const;
    size_t size() const { return size_value; }
    
    // In-place vector operations
    Vector4f& operator+=(const Vector4f& other);
    Vector4f& operator-=(const Vector4f& other);
    Vector4f& operator*=(float scalar);
    
    // Read-only operations
    float dot(const Vector4f& other) const;
    float length() const;
    float length_squared() const;
    void normalize();
    
    // Eigen integration
    Eigen::Map<Eigen::Vector4f> as_eigen();
    Eigen::Map<const Eigen::Vector4f> as_eigen() const;
    
    void initialize();
    void before_send() override {}
};

class Vector3f : public Message<Vector3f> {
public:
    static constexpr uint32_t message_type = 24;
    static constexpr size_t size_value = 3;
    
    using Message<Vector3f>::Message;
    
    static size_t calculate_size() {
        return size_value * sizeof(float);
    }
    
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float& at(size_t index);
    const float& at(size_t index) const;
    
    float& x() { return (*this)[0]; }
    float& y() { return (*this)[1]; }
    float& z() { return (*this)[2]; }
    const float& x() const { return (*this)[0]; }
    const float& y() const { return (*this)[1]; }
    const float& z() const { return (*this)[2]; }
    
    float* data();
    const float* data() const;
    size_t size() const { return size_value; }
    
    // In-place vector operations
    Vector3f& operator+=(const Vector3f& other);
    Vector3f& operator-=(const Vector3f& other);
    Vector3f& operator*=(float scalar);
    
    // Read-only operations
    float dot(const Vector3f& other) const;
    float length() const;
    float length_squared() const;
    void normalize();
    
    // Cross product - modifies this vector
    void cross_assign(const Vector3f& other);
    
    Eigen::Map<Eigen::Vector3f> as_eigen();
    Eigen::Map<const Eigen::Vector3f> as_eigen() const;
    
    void initialize();
    void before_send() override {}
};

} // namespace types
} // namespace psyne