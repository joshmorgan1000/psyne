#include "../../include/psyne/types/sparse_matrix.hpp"
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace psyne {
namespace types {

template<typename T>
void SparseMatrix<T>::initialize() {
    if (this->Message<SparseMatrix<T>>::data()) {
        // Initialize with 0x0 matrix, 0 nnz, 0 capacity
        uint8_t* ptr = this->Message<SparseMatrix<T>>::data();
        *reinterpret_cast<size_t*>(ptr) = 0; // rows
        ptr += sizeof(size_t);
        *reinterpret_cast<size_t*>(ptr) = 0; // cols
        ptr += sizeof(size_t);
        *reinterpret_cast<size_t*>(ptr) = 0; // nnz
        ptr += sizeof(size_t);
        *reinterpret_cast<size_t*>(ptr) = 0; // capacity
    }
}

template<typename T>
size_t SparseMatrix<T>::get_header_size() const {
    size_t rows_val = rows();
    return 4 * sizeof(size_t) + // rows, cols, nnz, capacity
           (rows_val + 1) * sizeof(index_type); // row_ptrs
}

template<typename T>
uint8_t* SparseMatrix<T>::get_data_start() {
    return this->Message<SparseMatrix<T>>::data();
}

template<typename T>
const uint8_t* SparseMatrix<T>::get_data_start() const {
    return this->Message<SparseMatrix<T>>::data();
}

template<typename T>
void SparseMatrix<T>::set_dimensions(size_t rows, size_t cols) {
    if (!this->Message<SparseMatrix<T>>::data()) return;
    
    uint8_t* ptr = get_data_start();
    *reinterpret_cast<size_t*>(ptr) = rows;
    ptr += sizeof(size_t);
    *reinterpret_cast<size_t*>(ptr) = cols;
    ptr += sizeof(size_t);
    // Keep existing nnz and capacity
    ptr += 2 * sizeof(size_t);
    
    // Initialize row_ptrs to all zeros
    index_type* row_ptrs_data = reinterpret_cast<index_type*>(ptr);
    for (size_t i = 0; i <= rows; ++i) {
        row_ptrs_data[i] = 0;
    }
}

template<typename T>
void SparseMatrix<T>::reserve_nnz(size_t max_non_zeros) {
    if (!this->Message<SparseMatrix<T>>::data()) return;
    
    size_t required_size = get_header_size() + 
                          max_non_zeros * sizeof(index_type) + // col_indices
                          max_non_zeros * sizeof(T); // values
    
    if (required_size > this->Message<SparseMatrix<T>>::size()) {
        throw std::runtime_error("Sparse matrix capacity exceeds message buffer size");
    }
    
    uint8_t* ptr = get_data_start() + 3 * sizeof(size_t);
    *reinterpret_cast<size_t*>(ptr) = max_non_zeros; // capacity
}

template<typename T>
size_t SparseMatrix<T>::rows() const {
    if (!get_data_start()) return 0;
    return *reinterpret_cast<const size_t*>(get_data_start());
}

template<typename T>
size_t SparseMatrix<T>::cols() const {
    if (!get_data_start()) return 0;
    return *reinterpret_cast<const size_t*>(get_data_start() + sizeof(size_t));
}

template<typename T>
size_t SparseMatrix<T>::nnz() const {
    if (!get_data_start()) return 0;
    return *reinterpret_cast<const size_t*>(get_data_start() + 2 * sizeof(size_t));
}

template<typename T>
size_t SparseMatrix<T>::capacity() const {
    if (!get_data_start()) return 0;
    return *reinterpret_cast<const size_t*>(get_data_start() + 3 * sizeof(size_t));
}

template<typename T>
const typename SparseMatrix<T>::index_type* SparseMatrix<T>::row_ptrs() const {
    if (!get_data_start()) return nullptr;
    return reinterpret_cast<const index_type*>(get_data_start() + 4 * sizeof(size_t));
}

template<typename T>
typename SparseMatrix<T>::index_type* SparseMatrix<T>::row_ptrs() {
    return const_cast<index_type*>(const_cast<const SparseMatrix<T>*>(this)->row_ptrs());
}

template<typename T>
const typename SparseMatrix<T>::index_type* SparseMatrix<T>::col_indices() const {
    if (!get_data_start()) return nullptr;
    const uint8_t* ptr = get_data_start() + 4 * sizeof(size_t) + (rows() + 1) * sizeof(index_type);
    return reinterpret_cast<const index_type*>(ptr);
}

template<typename T>
typename SparseMatrix<T>::index_type* SparseMatrix<T>::col_indices() {
    return const_cast<index_type*>(const_cast<const SparseMatrix<T>*>(this)->col_indices());
}

template<typename T>
const T* SparseMatrix<T>::values() const {
    if (!get_data_start()) return nullptr;
    const uint8_t* ptr = get_data_start() + 4 * sizeof(size_t) + 
                        (rows() + 1) * sizeof(index_type) + 
                        capacity() * sizeof(index_type);
    return reinterpret_cast<const T*>(ptr);
}

template<typename T>
T* SparseMatrix<T>::values() {
    return const_cast<T*>(const_cast<const SparseMatrix<T>*>(this)->values());
}

template<typename T>
T SparseMatrix<T>::at(size_t row, size_t col) const {
    if (row >= rows() || col >= cols()) {
        throw std::out_of_range("SparseMatrix index out of range");
    }
    
    size_t idx = find_element(row, col);
    return idx != SIZE_MAX ? values()[idx] : T(0);
}

template<typename T>
size_t SparseMatrix<T>::find_element(size_t row, size_t col) const {
    const index_type* row_ptrs_data = row_ptrs();
    const index_type* col_indices_data = col_indices();
    
    index_type start = row_ptrs_data[row];
    index_type end = row_ptrs_data[row + 1];
    
    // Binary search for column index
    auto it = std::lower_bound(col_indices_data + start, col_indices_data + end, col);
    
    if (it != col_indices_data + end && *it == col) {
        return std::distance(col_indices_data, it);
    }
    
    return SIZE_MAX; // Not found
}

template<typename T>
void SparseMatrix<T>::set(size_t row, size_t col, T value) {
    if (row >= rows() || col >= cols()) {
        throw std::out_of_range("SparseMatrix index out of range");
    }
    
    size_t idx = find_element(row, col);
    
    if (idx != SIZE_MAX) {
        // Element exists, update value
        values()[idx] = value;
        return;
    }
    
    // Element doesn't exist, need to insert
    if (nnz() >= capacity()) {
        throw std::runtime_error("SparseMatrix capacity exceeded");
    }
    
    // This is a simplified insertion - in practice would need to shift elements
    // For now, just append to the end and mark as needing sorting
    index_type* row_ptrs_data = row_ptrs();
    index_type* col_indices_data = col_indices();
    T* values_data = values();
    
    size_t current_nnz = nnz();
    
    // Insert at end of current row's data
    size_t insert_pos = row_ptrs_data[row + 1];
    
    // Shift elements in higher rows
    for (size_t i = insert_pos; i < current_nnz; ++i) {
        col_indices_data[i + 1] = col_indices_data[i];
        values_data[i + 1] = values_data[i];
    }
    
    // Insert new element
    col_indices_data[insert_pos] = col;
    values_data[insert_pos] = value;
    
    // Update row pointers
    for (size_t i = row + 1; i <= rows(); ++i) {
        row_ptrs_data[i]++;
    }
    
    // Update nnz
    uint8_t* ptr = get_data_start() + 2 * sizeof(size_t);
    *reinterpret_cast<size_t*>(ptr) = current_nnz + 1;
    
    // Sort the row to maintain CSR invariant
    sort_row(row);
}

template<typename T>
void SparseMatrix<T>::sort_row(size_t row) {
    const index_type* row_ptrs_data = row_ptrs();
    index_type* col_indices_data = col_indices();
    T* values_data = values();
    
    index_type start = row_ptrs_data[row];
    index_type end = row_ptrs_data[row + 1];
    
    // Sort by column index, keeping values aligned
    std::vector<std::pair<index_type, T>> row_data;
    for (index_type i = start; i < end; ++i) {
        row_data.emplace_back(col_indices_data[i], values_data[i]);
    }
    
    std::sort(row_data.begin(), row_data.end());
    
    for (size_t i = 0; i < row_data.size(); ++i) {
        col_indices_data[start + i] = row_data[i].first;
        values_data[start + i] = row_data[i].second;
    }
}

template<typename T>
typename SparseMatrix<T>::RowView SparseMatrix<T>::row(size_t row_idx) const {
    if (row_idx >= rows()) {
        throw std::out_of_range("Row index out of range");
    }
    
    const index_type* row_ptrs_data = row_ptrs();
    index_type start = row_ptrs_data[row_idx];
    index_type end = row_ptrs_data[row_idx + 1];
    
    return RowView{
        values() + start,
        col_indices() + start,
        static_cast<size_t>(end - start)
    };
}

template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator*=(T scalar) {
    T* values_data = values();
    size_t nnz_val = nnz();
    
    for (size_t i = 0; i < nnz_val; ++i) {
        values_data[i] *= scalar;
    }
    
    return *this;
}

template<typename T>
void SparseMatrix<T>::matvec(const T* x, T* y) const {
    size_t rows_val = rows();
    
    // Initialize output
    for (size_t i = 0; i < rows_val; ++i) {
        y[i] = T(0);
    }
    
    const index_type* row_ptrs_data = row_ptrs();
    const index_type* col_indices_data = col_indices();
    const T* values_data = values();
    
    for (size_t i = 0; i < rows_val; ++i) {
        for (index_type j = row_ptrs_data[i]; j < row_ptrs_data[i + 1]; ++j) {
            y[i] += values_data[j] * x[col_indices_data[j]];
        }
    }
}

template<typename T>
T SparseMatrix<T>::frobenius_norm() const {
    T sum_squares = T(0);
    const T* values_data = values();
    size_t nnz_val = nnz();
    
    for (size_t i = 0; i < nnz_val; ++i) {
        sum_squares += values_data[i] * values_data[i];
    }
    
    return std::sqrt(sum_squares);
}

template<typename T>
T SparseMatrix<T>::trace() const {
    T trace_sum = T(0);
    size_t min_dim = std::min(rows(), cols());
    
    for (size_t i = 0; i < min_dim; ++i) {
        size_t idx = find_element(i, i);
        if (idx != SIZE_MAX) {
            trace_sum += values()[idx];
        }
    }
    
    return trace_sum;
}

template<typename T>
void SparseMatrix<T>::from_dense(const T* dense_data, size_t rows_val, size_t cols_val, T threshold) {
    set_dimensions(rows_val, cols_val);
    
    // Count non-zeros first
    size_t nnz_count = 0;
    for (size_t i = 0; i < rows_val; ++i) {
        for (size_t j = 0; j < cols_val; ++j) {
            if (std::abs(dense_data[i * cols_val + j]) > threshold) {
                nnz_count++;
            }
        }
    }
    
    reserve_nnz(nnz_count);
    
    // Fill sparse format
    index_type* row_ptrs_data = row_ptrs();
    index_type* col_indices_data = col_indices();
    T* values_data = values();
    
    size_t current_nnz = 0;
    row_ptrs_data[0] = 0;
    
    for (size_t i = 0; i < rows_val; ++i) {
        for (size_t j = 0; j < cols_val; ++j) {
            T val = dense_data[i * cols_val + j];
            if (std::abs(val) > threshold) {
                col_indices_data[current_nnz] = j;
                values_data[current_nnz] = val;
                current_nnz++;
            }
        }
        row_ptrs_data[i + 1] = current_nnz;
    }
    
    // Update nnz
    uint8_t* ptr = get_data_start() + 2 * sizeof(size_t);
    *reinterpret_cast<size_t*>(ptr) = current_nnz;
}

template<typename T>
void SparseMatrix<T>::clear() {
    // Reset nnz but keep dimensions and capacity
    uint8_t* ptr = get_data_start() + 2 * sizeof(size_t);
    *reinterpret_cast<size_t*>(ptr) = 0;
    
    // Clear row pointers
    index_type* row_ptrs_data = row_ptrs();
    for (size_t i = 0; i <= rows(); ++i) {
        row_ptrs_data[i] = 0;
    }
}

// Explicit template instantiations for common types
template class SparseMatrix<float>;
template class SparseMatrix<double>;
template class SparseMatrix<int32_t>;

} // namespace types
} // namespace psyne