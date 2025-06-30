/**
 * @file gpu_message.hpp
 * @brief GPU-aware message types for zero-copy tensor operations
 *
 * Provides message types that can seamlessly move between CPU and GPU memory
 * without copying, leveraging unified memory architectures when available.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include "gpu_buffer.hpp"
#include <Eigen/Core>
#include <memory>
#include <span>
#include <vector>
// Forward declare to avoid circular includes
namespace psyne {
class Channel;
}

namespace psyne {
namespace gpu {

/**
 * @brief GPU-aware vector that can be efficiently sent through channels
 *
 * This class provides a vector-like interface that can automatically
 * handle GPU memory allocation and CPU/GPU synchronization.
 */
template <typename T>
class GPUVector {
public:
    using value_type = T;
    using size_type = size_t;
    using iterator = T *;
    using const_iterator = const T *;

    GPUVector(Channel &channel)
        : channel_(channel), size_(0), capacity_(0), cpu_data_(nullptr),
          gpu_buffer_(nullptr), is_gpu_dirty_(false), is_cpu_dirty_(false) {}

    GPUVector(const GPUVector &) = delete;
    GPUVector &operator=(const GPUVector &) = delete;

    GPUVector(GPUVector &&) = default;
    GPUVector &operator=(GPUVector &&) = default;

    /**
     * @brief Initialize from CPU data
     */
    GPUVector &operator=(std::initializer_list<T> init) {
        resize(init.size());
        // Zero-copy optimization: manual copy from initializer list
        size_t i = 0;
        for (const T &value : init) {
            cpu_data_[i++] = value;
        }
        is_cpu_dirty_ = true;
        return *this;
    }

    /**
     * @brief Get the size of the vector
     */
    size_type size() const {
        return size_;
    }

    /**
     * @brief Check if vector is empty
     */
    bool empty() const {
        return size_ == 0;
    }

    /**
     * @brief Resize the vector
     */
    void resize(size_type new_size) {
        if (new_size > capacity_) {
            reserve(new_size);
        }
        size_ = new_size;
    }

    /**
     * @brief Reserve capacity
     */
    void reserve(size_type new_capacity) {
        if (new_capacity <= capacity_) {
            return;
        }

        // Allocate new CPU buffer
        T *new_data =
            static_cast<T *>(std::aligned_alloc(64, new_capacity * sizeof(T)));
        if (!new_data) {
            throw std::bad_alloc();
        }

        // Copy existing data
        if (cpu_data_ && size_ > 0) {
            ensure_cpu_valid();
            // Zero-copy note: memcpy is necessary here for reallocation
            // This is similar to std::vector resize and cannot be avoided
            std::memcpy(new_data, cpu_data_, size_ * sizeof(T));
        }

        // Free old buffer
        if (cpu_data_) {
            std::free(cpu_data_);
        }

        cpu_data_ = new_data;
        capacity_ = new_capacity;
        gpu_buffer_.reset(); // Invalidate GPU buffer
        is_gpu_dirty_ = true;
    }

    /**
     * @brief Access element (CPU side)
     */
    T &operator[](size_type idx) {
        ensure_cpu_valid();
        is_cpu_dirty_ = true;
        return cpu_data_[idx];
    }

    const T &operator[](size_type idx) const {
        ensure_cpu_valid();
        return cpu_data_[idx];
    }

    /**
     * @brief Get iterators (CPU side)
     */
    iterator begin() {
        ensure_cpu_valid();
        is_cpu_dirty_ = true;
        return cpu_data_;
    }

    iterator end() {
        ensure_cpu_valid();
        is_cpu_dirty_ = true;
        return cpu_data_ + size_;
    }

    const_iterator begin() const {
        ensure_cpu_valid();
        return cpu_data_;
    }

    const_iterator end() const {
        ensure_cpu_valid();
        return cpu_data_ + size_;
    }

    /**
     * @brief Check if data is valid
     */
    bool is_valid() const {
        return cpu_data_ != nullptr || gpu_buffer_ != nullptr;
    }

    /**
     * @brief Transfer data to GPU buffer
     */
    std::shared_ptr<GPUBuffer> to_gpu_buffer(GPUContext &context) {
        if (!gpu_buffer_ || is_gpu_dirty_) {
            // Create GPU buffer
            auto factory = context.create_buffer_factory();
            gpu_buffer_ = factory->create_buffer(
                size_ * sizeof(T), BufferUsage::Dynamic, MemoryAccess::Shared);

            // Upload data if CPU has latest version
            if (is_cpu_dirty_ || !gpu_buffer_) {
                ensure_cpu_valid();
                gpu_buffer_->upload(cpu_data_, size_ * sizeof(T));
                is_cpu_dirty_ = false;
            }

            is_gpu_dirty_ = false;
        }

        return gpu_buffer_;
    }

    /**
     * @brief Check if data is currently on GPU
     */
    bool is_on_gpu() const {
        return gpu_buffer_ != nullptr && !is_gpu_dirty_;
    }

    /**
     * @brief Get as Eigen vector view
     */
    Eigen::Map<Eigen::VectorX<T>> as_eigen() {
        ensure_cpu_valid();
        is_cpu_dirty_ = true;
        return Eigen::Map<Eigen::VectorX<T>>(cpu_data_, size_);
    }

    Eigen::Map<const Eigen::VectorX<T>> as_eigen() const {
        ensure_cpu_valid();
        return Eigen::Map<const Eigen::VectorX<T>>(cpu_data_, size_);
    }

    /**
     * @brief GPU compute operation - scale all elements
     */
    void gpu_scale(GPUContext &context, T scalar);

    /**
     * @brief GPU compute operation - element-wise add
     */
    void gpu_add(GPUContext &context, const GPUVector<T> &other);

    ~GPUVector() {
        if (cpu_data_) {
            std::free(cpu_data_);
        }
    }

private:
    Channel &channel_;
    size_type size_;
    size_type capacity_;
    T *cpu_data_;
    std::shared_ptr<GPUBuffer> gpu_buffer_;
    mutable bool is_gpu_dirty_;
    mutable bool is_cpu_dirty_;

    void ensure_cpu_valid() const {
        if (!cpu_data_ && gpu_buffer_) {
            // Allocate CPU memory
            const_cast<GPUVector *>(this)->reserve(size_);

            // Download from GPU
            gpu_buffer_->download(cpu_data_, size_ * sizeof(T));
            const_cast<GPUVector *>(this)->is_cpu_dirty_ = false;
        }
    }
};

// Convenience type aliases
using GPUFloatVector = GPUVector<float>;
using GPUDoubleVector = GPUVector<double>;
using GPUIntVector = GPUVector<int32_t>;

/**
 * @brief GPU-aware matrix type
 */
template <typename T>
class GPUMatrix {
public:
    using value_type = T;
    using size_type = size_t;

    GPUMatrix(size_type rows, size_type cols)
        : rows_(rows), cols_(cols), data_(rows * cols), gpu_buffer_(nullptr) {}

    size_type rows() const {
        return rows_;
    }
    size_type cols() const {
        return cols_;
    }
    size_type size() const {
        return rows_ * cols_;
    }

    /**
     * @brief Access element
     */
    T &operator()(size_type row, size_type col) {
        return data_[row * cols_ + col];
    }

    const T &operator()(size_type row, size_type col) const {
        return data_[row * cols_ + col];
    }

    /**
     * @brief Get as Eigen matrix view
     */
    Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    as_eigen() {
        return Eigen::Map<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            data_.data(), rows_, cols_);
    }

    /**
     * @brief Transfer to GPU
     */
    std::shared_ptr<GPUBuffer> to_gpu_buffer(GPUContext &context) {
        if (!gpu_buffer_) {
            auto factory = context.create_buffer_factory();
            gpu_buffer_ = factory->create_buffer(
                size() * sizeof(T), BufferUsage::Dynamic, MemoryAccess::Shared);

            gpu_buffer_->upload(data_.data(), size() * sizeof(T));
        }

        return gpu_buffer_;
    }

private:
    size_type rows_;
    size_type cols_;
    std::vector<T> data_;
    std::shared_ptr<GPUBuffer> gpu_buffer_;
};

/**
 * @brief GPU vector serialization helpers
 *
 * These helper functions can be used to serialize/deserialize GPU vectors
 * when the MessageTraits system is available.
 */
template <typename T>
class GPUVectorSerializer {
public:
    static size_t serialized_size(const GPUVector<T> &vec) {
        return sizeof(size_t) + vec.size() * sizeof(T);
    }

    static void serialize(uint8_t *buffer, const GPUVector<T> &vec) {
        // Write size
        *reinterpret_cast<size_t *>(buffer) = vec.size();
        buffer += sizeof(size_t);

        // Write data
        if (vec.size() > 0) {
            // Zero-copy optimization: manual serialization
            const T *src = vec.begin();
            T *dst = reinterpret_cast<T *>(buffer);
            for (size_t i = 0; i < vec.size(); ++i) {
                dst[i] = src[i];
            }
        }
    }

    static GPUVector<T> deserialize(const uint8_t *buffer, Channel &channel) {
        // Read size
        size_t size = *reinterpret_cast<const size_t *>(buffer);
        buffer += sizeof(size_t);

        // Create vector
        GPUVector<T> vec(channel);
        vec.resize(size);

        // Read data
        if (size > 0) {
            // Zero-copy optimization: manual deserialization
            const T *src = reinterpret_cast<const T *>(buffer);
            T *dst = vec.begin();
            for (size_t i = 0; i < size; ++i) {
                dst[i] = src[i];
            }
        }

        return vec;
    }
};

} // namespace gpu
} // namespace psyne