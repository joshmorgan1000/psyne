#pragma once

#include "../psyne.hpp"
#include <vector>
#include <array>

namespace psyne {
namespace types {

// Sparse matrix implementation using Compressed Sparse Row (CSR) format
// Optimized for scientific computing and ML workloads
template<typename T>
class SparseMatrix : public Message<SparseMatrix<T>> {
public:
    using value_type = T;
    using index_type = uint32_t;
    static constexpr uint32_t message_type = 34; // Will be specialized
    
    using Message<SparseMatrix<T>>::Message;
    
    static size_t calculate_size() {
        return 8192; // Default size for sparse matrices
    }
    
    // Matrix dimensions and structure
    void set_dimensions(size_t rows, size_t cols);
    void reserve_nnz(size_t max_non_zeros); // Reserve space for non-zero elements
    size_t rows() const;
    size_t cols() const;
    size_t nnz() const; // Number of non-zero elements
    size_t capacity() const; // Maximum non-zero elements that can be stored
    
    // CSR format access
    const T* values() const; // Non-zero values array
    const index_type* col_indices() const; // Column indices for each value
    const index_type* row_ptrs() const; // Row pointers (size = rows + 1)
    
    T* values(); // Mutable access
    index_type* col_indices();
    index_type* row_ptrs();
    
    // Element access and modification
    T at(size_t row, size_t col) const; // Returns 0 if element doesn't exist
    void set(size_t row, size_t col, T value); // Set/insert element
    void add(size_t row, size_t col, T value); // Add to existing element
    
    // Efficient row access
    struct RowView {
        const T* values;
        const index_type* indices;
        size_t size;
        
        // Iterator support for range-based loops
        struct iterator {
            const T* val_ptr;
            const index_type* idx_ptr;
            
            std::pair<index_type, T> operator*() const { return {*idx_ptr, *val_ptr}; }
            iterator& operator++() { ++val_ptr; ++idx_ptr; return *this; }
            bool operator!=(const iterator& other) const { return val_ptr != other.val_ptr; }
        };
        
        iterator begin() const { return {values, indices}; }
        iterator end() const { return {values + size, indices + size}; }
    };
    
    RowView row(size_t row_idx) const;
    
    // Matrix operations (in-place when possible)
    SparseMatrix& operator*=(T scalar);
    SparseMatrix& operator+=(T scalar); // Add scalar to all non-zero elements
    
    // Sparse-specific operations
    void transpose_inplace(); // Transpose the matrix in-place
    T frobenius_norm() const; // ||A||_F
    T trace() const; // Sum of diagonal elements
    
    // Conversion utilities
    void from_dense(const T* dense_data, size_t rows, size_t cols, T threshold = T(1e-12));
    void to_dense(T* dense_data) const; // Output must be pre-allocated
    
    // Sparse matrix-vector multiplication
    void matvec(const T* x, T* y) const; // y = A * x
    void matvec_transpose(const T* x, T* y) const; // y = A^T * x
    
    // Matrix analysis
    size_t count_nonzeros_in_row(size_t row) const;
    size_t count_nonzeros_in_col(size_t col) const;
    std::vector<size_t> nonzero_pattern() const; // Returns rows with non-zeros
    
    // Memory management
    void compress(); // Remove explicit zeros and compress storage
    void clear(); // Clear all data but keep dimensions
    
    void initialize();
    void before_send() override {}

private:
    // Header layout:
    // [rows: size_t][cols: size_t][nnz: size_t][capacity: size_t]
    // [row_ptrs: index_type * (rows+1)][col_indices: index_type * nnz][values: T * nnz]
    
    size_t get_header_size() const;
    uint8_t* get_data_start();
    const uint8_t* get_data_start() const;
    
    // Helper methods
    size_t find_element(size_t row, size_t col) const; // Returns index in values array, or SIZE_MAX if not found
    void sort_row(size_t row); // Sort elements in a row by column index
    void ensure_capacity(size_t required_nnz);
};

// Specializations for common types
using SparseMatrixF = SparseMatrix<float>;
using SparseMatrixD = SparseMatrix<double>;
using SparseMatrixI32 = SparseMatrix<int32_t>;

// Specialized message type IDs
template<> constexpr uint32_t SparseMatrix<float>::message_type = 34;
template<> constexpr uint32_t SparseMatrix<double>::message_type = 35;
template<> constexpr uint32_t SparseMatrix<int32_t>::message_type = 36;

} // namespace types
} // namespace psyne