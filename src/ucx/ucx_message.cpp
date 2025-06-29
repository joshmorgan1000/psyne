/**
 * @file ucx_message.cpp
 * @brief UCX message implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#if defined(PSYNE_UCX_SUPPORT)

#include "ucx_message.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace psyne {
namespace ucx {

// UCXMessage base implementation - most functionality is in headers due to templates

// Template instantiations for common types
template class UCXVector<float>;
template class UCXVector<double>;
template class UCXVector<int32_t>;
template class UCXVector<int64_t>;
template class UCXVector<uint32_t>;
template class UCXVector<uint64_t>;

template class UCXGPUVector<float>;
template class UCXGPUVector<double>;
template class UCXGPUVector<int32_t>;
template class UCXGPUVector<int64_t>;

} // namespace ucx
} // namespace psyne

#else // !PSYNE_UCX_SUPPORT

// Stub implementation when UCX is not available
namespace psyne {
namespace ucx {

// Empty namespace for compatibility

} // namespace ucx
} // namespace psyne

#endif // PSYNE_UCX_SUPPORT