/**
 * @file ucx_channel.cpp
 * @brief UCX channel implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#if defined(PSYNE_UCX_SUPPORT)

#include "ucx_channel.hpp"
#include <iostream>
#include <chrono>
#include <cstring>
#include <algorithm>

// UCX includes
extern "C" {
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>
}

namespace psyne {
namespace ucx {

// UCXMemoryRegion implementation

UCXMemoryRegion::UCXMemoryRegion(ucp_context_t* context, void* addr, size_t length, MemoryType type)
    : addr_(addr), length_(length), type_(type), context_(context) {
    
    ucp_mem_map_params_t params = {};
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                       UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                       UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address = addr;
    params.length = length;
    params.flags = UCP_MEM_MAP_NONBLOCK;
    
    // Set memory type flags based on type
    switch (type) {
        case MemoryType::CUDA:
            params.flags |= UCP_MEM_MAP_ALLOCATE;
            params.field_mask |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
            params.memory_type = UCS_MEMORY_TYPE_CUDA;
            break;
        case MemoryType::ROCM:
            params.flags |= UCP_MEM_MAP_ALLOCATE;
            params.field_mask |= UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
            params.memory_type = UCS_MEMORY_TYPE_ROCM;
            break;
        case MemoryType::HOST_PINNED:
            params.flags |= UCP_MEM_MAP_ALLOCATE;
            break;
        case MemoryType::HOST:
        default:
            // Default host memory
            break;
    }
    
    ucs_status_t status = ucp_mem_map(context, &params, &mem_handle_);
    if (status != UCS_OK) {
        throw std::runtime_error("Failed to map UCX memory region: " + std::string(ucs_status_string(status)));
    }
    
    // Pack remote key for RMA operations
    void* rkey_buffer;
    size_t rkey_size;
    status = ucp_rkey_pack(context, mem_handle_, &rkey_buffer, &rkey_size);
    if (status != UCS_OK) {
        ucp_mem_unmap(context, mem_handle_);
        throw std::runtime_error("Failed to pack UCX remote key: " + std::string(ucs_status_string(status)));
    }
    
    // Store the packed key (simplified - real implementation would handle this better)
    rkey_ = rkey_buffer;
    
    std::cout << "UCX memory region registered: " << addr << " size: " << length 
              << " type: " << static_cast<int>(type) << std::endl;
}

UCXMemoryRegion::~UCXMemoryRegion() {
    if (mem_handle_) {
        ucp_mem_unmap(context_, mem_handle_);
    }
    if (rkey_) {
        ucp_rkey_buffer_release(rkey_);
    }
}

std::vector<uint8_t> UCXMemoryRegion::pack_rkey() const {
    // In a real implementation, this would properly serialize the remote key
    std::vector<uint8_t> packed_key(sizeof(ucp_rkey_h));
    std::memcpy(packed_key.data(), &rkey_, sizeof(ucp_rkey_h));
    return packed_key;
}

// UCXEndpoint implementation

UCXEndpoint::UCXEndpoint(ucp_worker_t* worker, const std::string& peer_address)
    : worker_(worker), endpoint_(nullptr), connected_(false), peer_address_(peer_address) {
    
    // In a real implementation, this would parse the peer address and create the endpoint
    ucp_ep_params_t ep_params = {};
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    
    // For demo purposes, create endpoint (real implementation would handle address resolution)
    ucs_status_t status = ucp_ep_create(worker, &ep_params, &endpoint_);
    if (status == UCS_OK) {
        connected_ = true;
        std::cout << "UCX endpoint created for peer: " << peer_address << std::endl;
    } else {
        std::cerr << "Failed to create UCX endpoint for " << peer_address 
                  << ": " << ucs_status_string(status) << std::endl;
        throw std::runtime_error("Failed to create UCX endpoint: " + std::string(ucs_status_string(status)));
    }
}

UCXEndpoint::~UCXEndpoint() {
    close();
}

bool UCXEndpoint::send(const void* buffer, size_t length, ucp_tag_t tag) {
    if (!connected_) {
        return false;
    }
    
    ucp_request_t* request = ucp_tag_send_nb(endpoint_, buffer, length, ucp_dt_make_contig(1), tag, nullptr);
    
    if (UCS_PTR_IS_ERR(request)) {
        ucs_status_t error_status = UCS_PTR_STATUS(request);
        std::cerr << "UCX send failed for tag " << tag << ": " << ucs_status_string(error_status) << std::endl;
        
        // Update error statistics
        // stats_.send_errors++; // Would need to add error tracking
        return false;
    }
    
    if (UCS_PTR_IS_PTR(request)) {
        // Wait for completion
        ucs_status_t status;
        do {
            progress();
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        
        ucp_request_free(request);
        return status == UCS_OK;
    }
    
    // Immediate completion
    return true;
}

bool UCXEndpoint::receive(void* buffer, size_t length, ucp_tag_t tag, size_t* received_length) {
    if (!connected_) {
        return false;
    }
    
    ucp_tag_recv_info_t recv_info;
    ucp_request_t* request = ucp_tag_recv_nb(worker_, buffer, length, ucp_dt_make_contig(1), 
                                            tag, ~0UL, nullptr);
    
    if (UCS_PTR_IS_ERR(request)) {
        ucs_status_t error_status = UCS_PTR_STATUS(request);
        std::cerr << "UCX receive failed for tag " << tag << ": " << ucs_status_string(error_status) << std::endl;
        
        // Update error statistics  
        // stats_.receive_errors++; // Would need to add error tracking
        return false;
    }
    
    if (UCS_PTR_IS_PTR(request)) {
        // Wait for completion
        ucs_status_t status;
        do {
            progress();
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        
        if (status == UCS_OK) {
            ucp_tag_recv_info_t info;
            ucp_request_query(request, UCP_REQUEST_ATTR_FIELD_INFO, &info);
            if (received_length) {
                *received_length = info.length;
            }
        }
        
        ucp_request_free(request);
        return status == UCS_OK;
    }
    
    // Immediate completion
    if (received_length) {
        *received_length = length;
    }
    return true;
}

ucp_request_t* UCXEndpoint::send_nb(const void* buffer, size_t length, ucp_tag_t tag) {
    if (!connected_) {
        return nullptr;
    }
    
    return ucp_tag_send_nb(endpoint_, buffer, length, ucp_dt_make_contig(1), tag, nullptr);
}

ucp_request_t* UCXEndpoint::receive_nb(void* buffer, size_t length, ucp_tag_t tag) {
    if (!connected_) {
        return nullptr;
    }
    
    return ucp_tag_recv_nb(worker_, buffer, length, ucp_dt_make_contig(1), tag, ~0UL, nullptr);
}

bool UCXEndpoint::put(const void* local_buffer, size_t length, uint64_t remote_addr, ucp_rkey_h rkey) {
    if (!connected_) {
        return false;
    }
    
    ucp_request_t* request = ucp_put_nb(endpoint_, local_buffer, length, remote_addr, rkey, nullptr);
    
    if (UCS_PTR_IS_ERR(request)) {
        return false;
    }
    
    if (UCS_PTR_IS_PTR(request)) {
        // Wait for completion
        ucs_status_t status;
        do {
            progress();
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        
        ucp_request_free(request);
        return status == UCS_OK;
    }
    
    return true;
}

bool UCXEndpoint::get(void* local_buffer, size_t length, uint64_t remote_addr, ucp_rkey_h rkey) {
    if (!connected_) {
        return false;
    }
    
    ucp_request_t* request = ucp_get_nb(endpoint_, local_buffer, length, remote_addr, rkey, nullptr);
    
    if (UCS_PTR_IS_ERR(request)) {
        return false;
    }
    
    if (UCS_PTR_IS_PTR(request)) {
        // Wait for completion
        ucs_status_t status;
        do {
            progress();
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        
        ucp_request_free(request);
        return status == UCS_OK;
    }
    
    return true;
}

bool UCXEndpoint::atomic_add(uint64_t* result, uint64_t operand, uint64_t remote_addr, ucp_rkey_h rkey) {
    // UCX atomic operations (implementation would depend on UCX version)
    // For now, return false as not all UCX builds support atomics
    return false;
}

bool UCXEndpoint::atomic_cas(uint64_t* result, uint64_t compare, uint64_t swap, uint64_t remote_addr, ucp_rkey_h rkey) {
    // UCX atomic operations (implementation would depend on UCX version)
    return false;
}

bool UCXEndpoint::stream_send(const void* buffer, size_t length) {
    if (!connected_) {
        return false;
    }
    
    ucp_request_t* request = ucp_stream_send_nb(endpoint_, buffer, length, ucp_dt_make_contig(1), nullptr, 0);
    
    if (UCS_PTR_IS_ERR(request)) {
        return false;
    }
    
    if (UCS_PTR_IS_PTR(request)) {
        // Wait for completion
        ucs_status_t status;
        do {
            progress();
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        
        ucp_request_free(request);
        return status == UCS_OK;
    }
    
    return true;
}

bool UCXEndpoint::stream_receive(void* buffer, size_t length, size_t* received_length) {
    if (!connected_) {
        return false;
    }
    
    size_t recv_length;
    ucp_request_t* request = ucp_stream_recv_nb(endpoint_, buffer, length, ucp_dt_make_contig(1), 
                                               nullptr, 0, &recv_length);
    
    if (UCS_PTR_IS_ERR(request)) {
        return false;
    }
    
    if (UCS_PTR_IS_PTR(request)) {
        // Wait for completion
        ucs_status_t status;
        do {
            progress();
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        
        ucp_request_free(request);
        if (received_length && status == UCS_OK) {
            *received_length = recv_length;
        }
        return status == UCS_OK;
    }
    
    // Immediate completion
    if (received_length) {
        *received_length = recv_length;
    }
    return true;
}

void UCXEndpoint::close() {
    if (endpoint_ && connected_) {
        ucp_ep_close_nb(endpoint_, UCP_EP_CLOSE_MODE_FLUSH);
        endpoint_ = nullptr;
        connected_ = false;
    }
}

void UCXEndpoint::progress() {
    if (worker_) {
        ucp_worker_progress(worker_);
    }
}

// UCXWorker implementation

UCXWorker::UCXWorker(ucp_context_t* context, bool enable_wakeup)
    : context_(context), worker_(nullptr), wakeup_enabled_(enable_wakeup) {
    
    ucp_worker_params_t worker_params = {};
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    
    if (enable_wakeup) {
        worker_params.field_mask |= UCP_WORKER_PARAM_FIELD_EVENTS;
        worker_params.events = UCP_WAKEUP_EDGE;
    }
    
    ucs_status_t status = ucp_worker_create(context, &worker_params, &worker_);
    if (status != UCS_OK) {
        throw std::runtime_error("Failed to create UCX worker: " + std::string(ucs_status_string(status)));
    }
    
    std::cout << "UCX worker created successfully" << std::endl;
}

UCXWorker::~UCXWorker() {
    {
        std::lock_guard<std::mutex> lock(endpoints_mutex_);
        endpoints_.clear();
    }
    
    if (worker_) {
        ucp_worker_destroy(worker_);
    }
}

std::shared_ptr<UCXEndpoint> UCXWorker::create_endpoint(const std::string& peer_address) {
    std::lock_guard<std::mutex> lock(endpoints_mutex_);
    
    auto it = endpoints_.find(peer_address);
    if (it != endpoints_.end()) {
        return it->second;
    }
    
    try {
        auto endpoint = std::make_shared<UCXEndpoint>(worker_, peer_address);
        endpoints_[peer_address] = endpoint;
        return endpoint;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create endpoint for " << peer_address << ": " << e.what() << std::endl;
        return nullptr;
    }
}

void UCXWorker::remove_endpoint(const std::string& peer_address) {
    std::lock_guard<std::mutex> lock(endpoints_mutex_);
    endpoints_.erase(peer_address);
}

std::vector<uint8_t> UCXWorker::get_address() const {
    ucp_address_t* address;
    size_t address_length;
    
    ucs_status_t status = ucp_worker_get_address(worker_, &address, &address_length);
    if (status != UCS_OK) {
        return {};
    }
    
    std::vector<uint8_t> addr_data(address_length);
    std::memcpy(addr_data.data(), address, address_length);
    
    ucp_worker_release_address(worker_, address);
    return addr_data;
}

void UCXWorker::progress() {
    if (worker_) {
        ucp_worker_progress(worker_);
    }
}

void UCXWorker::wait_for_events(int timeout_ms) {
    if (worker_ && wakeup_enabled_) {
        ucp_worker_wait(worker_);
    } else if (timeout_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
    }
}

void UCXWorker::signal() {
    if (worker_ && wakeup_enabled_) {
        ucp_worker_signal(worker_);
    }
}

bool UCXWorker::test_request(ucp_request_t* request) {
    if (!request || UCS_PTR_IS_ERR(request)) {
        return true;
    }
    
    if (!UCS_PTR_IS_PTR(request)) {
        return true; // Immediate completion
    }
    
    ucs_status_t status = ucp_request_check_status(request);
    return status != UCS_INPROGRESS;
}

void UCXWorker::wait_request(ucp_request_t* request) {
    if (!request || UCS_PTR_IS_ERR(request) || !UCS_PTR_IS_PTR(request)) {
        return;
    }
    
    ucs_status_t status;
    do {
        progress();
        status = ucp_request_check_status(request);
    } while (status == UCS_INPROGRESS);
}

void UCXWorker::cancel_request(ucp_request_t* request) {
    if (request && UCS_PTR_IS_PTR(request)) {
        ucp_request_cancel(worker_, request);
    }
}

// UCXChannel implementation

UCXChannel::UCXChannel(const std::string& name, TransportMode mode, size_t buffer_size, bool enable_gpu)
    : Channel(name, buffer_size, ChannelMode::Blocking, ChannelType::Reliable)
    , context_(nullptr)
    , transport_mode_(mode)
    , gpu_enabled_(enable_gpu)
    , running_(true) {
    
    if (!init_ucx_context(mode, enable_gpu)) {
        std::cerr << "Failed to initialize UCX context for channel: " << name 
                  << " mode: " << static_cast<int>(mode) 
                  << " GPU: " << (enable_gpu ? "enabled" : "disabled") << std::endl;
        throw std::runtime_error("Failed to initialize UCX context for channel: " + name);
    }
    
    try {
        worker_ = std::make_unique<UCXWorker>(context_);
    } catch (const std::exception& e) {
        cleanup_resources();
        throw std::runtime_error("Failed to create UCX worker: " + std::string(e.what()));
    }
    
    // Start progress thread
    progress_thread_ = std::thread(&UCXChannel::progress_worker, this);
    
    std::cout << "UCX channel created: " << name << " mode: " << static_cast<int>(mode) 
              << " GPU: " << (enable_gpu ? "enabled" : "disabled") << std::endl;
}

UCXChannel::~UCXChannel() {
    running_ = false;
    
    if (progress_thread_.joinable()) {
        progress_thread_.join();
    }
    
    cleanup_resources();
}

size_t UCXChannel::send(const void* data, size_t size, uint32_t type_id) {
    if (peers_.empty()) {
        return 0;
    }
    
    auto start_time = get_timestamp_us();
    
    // For simplicity, send to first peer (real implementation would handle multiple peers)
    auto& first_peer = peers_.begin()->second;
    bool success = first_peer->send(data, size, type_id);
    
    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.messages_sent++;
        stats_.bytes_sent += size;
        
        auto end_time = get_timestamp_us();
        double latency = static_cast<double>(end_time - start_time);
        stats_.avg_latency_us = (stats_.avg_latency_us * (stats_.messages_sent - 1) + latency) / stats_.messages_sent;
        
        return size;
    }
    
    return 0;
}

size_t UCXChannel::receive(void* buffer, size_t buffer_size, uint32_t* type_id) {
    if (peers_.empty()) {
        return 0;
    }
    
    auto start_time = get_timestamp_us();
    
    // For simplicity, receive from first peer
    auto& first_peer = peers_.begin()->second;
    size_t received_length;
    bool success = first_peer->receive(buffer, buffer_size, 0, &received_length);
    
    if (success) {
        if (type_id) {
            *type_id = 0; // Would be set from tag in real implementation
        }
        
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.messages_received++;
        stats_.bytes_received += received_length;
        
        auto end_time = get_timestamp_us();
        double latency = static_cast<double>(end_time - start_time);
        stats_.avg_latency_us = (stats_.avg_latency_us * stats_.messages_received + latency) / (stats_.messages_received + 1);
        
        return received_length;
    }
    
    return 0;
}

bool UCXChannel::try_send(const void* data, size_t size, uint32_t type_id) {
    return send(data, size, type_id) > 0;
}

bool UCXChannel::try_receive(void* buffer, size_t buffer_size, size_t* received_size, uint32_t* type_id) {
    size_t received = receive(buffer, buffer_size, type_id);
    if (received_size) *received_size = received;
    return received > 0;
}

std::shared_ptr<UCXMemoryRegion> UCXChannel::register_memory(void* addr, size_t length, MemoryType type) {
    try {
        auto region = std::make_shared<UCXMemoryRegion>(context_, addr, length, type);
        memory_regions_.push_back(region);
        return region;
    } catch (const std::exception& e) {
        std::cerr << "Failed to register UCX memory: " << e.what() << std::endl;
        return nullptr;
    }
}

bool UCXChannel::send_memory_region(const UCXMemoryRegion& region, size_t offset, size_t length) {
    if (peers_.empty()) {
        return false;
    }
    
    size_t send_length = (length == 0) ? region.length() - offset : length;
    void* send_addr = static_cast<char*>(region.addr()) + offset;
    
    auto& first_peer = peers_.begin()->second;
    bool success = first_peer->send(send_addr, send_length);
    
    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.zero_copy_sends++;
    }
    
    return success;
}

bool UCXChannel::receive_memory_region(const UCXMemoryRegion& region, size_t offset, size_t length) {
    if (peers_.empty()) {
        return false;
    }
    
    size_t recv_length = (length == 0) ? region.length() - offset : length;
    void* recv_addr = static_cast<char*>(region.addr()) + offset;
    
    auto& first_peer = peers_.begin()->second;
    size_t received;
    bool success = first_peer->receive(recv_addr, recv_length, 0, &received);
    
    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.zero_copy_receives++;
    }
    
    return success;
}

bool UCXChannel::rma_put(const void* local_buffer, size_t length, const std::string& peer,
                         uint64_t remote_addr, const std::vector<uint8_t>& remote_key) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    auto it = peers_.find(peer);
    if (it == peers_.end()) {
        return false;
    }
    
    // Unpack remote key (simplified)
    ucp_rkey_h rkey = nullptr;
    if (remote_key.size() >= sizeof(ucp_rkey_h)) {
        std::memcpy(&rkey, remote_key.data(), sizeof(ucp_rkey_h));
    }
    
    bool success = it->second->put(local_buffer, length, remote_addr, rkey);
    
    if (success) {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.rma_operations++;
    }
    
    return success;
}

bool UCXChannel::rma_get(void* local_buffer, size_t length, const std::string& peer,
                         uint64_t remote_addr, const std::vector<uint8_t>& remote_key) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    auto it = peers_.find(peer);
    if (it == peers_.end()) {
        return false;
    }
    
    // Unpack remote key (simplified)
    ucp_rkey_h rkey = nullptr;
    if (remote_key.size() >= sizeof(ucp_rkey_h)) {
        std::memcpy(&rkey, remote_key.data(), sizeof(ucp_rkey_h));
    }
    
    bool success = it->second->get(local_buffer, length, remote_addr, rkey);
    
    if (success) {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.rma_operations++;
    }
    
    return success;
}

bool UCXChannel::atomic_fadd(uint64_t* result, uint64_t operand, const std::string& peer,
                             uint64_t remote_addr, const std::vector<uint8_t>& remote_key) {
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    auto it = peers_.find(peer);
    if (it == peers_.end()) {
        return false;
    }
    
    // Unpack remote key (simplified)
    ucp_rkey_h rkey = nullptr;
    if (remote_key.size() >= sizeof(ucp_rkey_h)) {
        std::memcpy(&rkey, remote_key.data(), sizeof(ucp_rkey_h));
    }
    
    bool success = it->second->atomic_add(result, operand, remote_addr, rkey);
    
    if (success) {
        std::lock_guard<std::mutex> stats_lock(stats_mutex_);
        stats_.atomic_operations++;
    }
    
    return success;
}

bool UCXChannel::connect(const std::string& peer_address) {
    if (!worker_) {
        std::cerr << "UCX worker not initialized, cannot connect to: " << peer_address << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(peers_mutex_);
    
    try {
        auto endpoint = worker_->create_endpoint(peer_address);
        if (endpoint) {
            peers_[peer_address] = endpoint;
            std::cout << "UCX connected to peer: " << peer_address << std::endl;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during UCX connection to " << peer_address << ": " << e.what() << std::endl;
        return false;
    }
    
    std::cerr << "Failed to create UCX endpoint for peer: " << peer_address << std::endl;
    return false;
}

bool UCXChannel::listen(const std::string& address) {
    // UCX listening would typically involve creating a listener endpoint
    // For simplicity, we'll just log this
    std::cout << "UCX listening on: " << address << std::endl;
    return true;
}

UCXCapabilities UCXChannel::get_capabilities() const {
    return UCXContextManager::instance().get_system_capabilities();
}

UCXChannel::UCXStats UCXChannel::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void UCXChannel::progress() {
    if (worker_) {
        worker_->progress();
    }
}

bool UCXChannel::init_ucx_context(TransportMode mode, bool enable_gpu) {
    context_ = UCXContextManager::instance().get_context(mode, enable_gpu);
    return context_ != nullptr;
}

void UCXChannel::cleanup_resources() {
    memory_regions_.clear();
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        peers_.clear();
    }
    worker_.reset();
    
    if (context_) {
        UCXContextManager::instance().release_context(context_);
        context_ = nullptr;
    }
}

void UCXChannel::progress_worker() {
    while (running_) {
        progress();
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

ucp_config_t* UCXChannel::create_ucx_config(TransportMode mode, bool enable_gpu) {
    ucp_config_t* config;
    ucs_status_t status = ucp_config_read(nullptr, nullptr, &config);
    if (status != UCS_OK) {
        return nullptr;
    }
    
    // Configure based on transport mode
    switch (mode) {
        case TransportMode::TCP_ONLY:
            ucp_config_modify(config, "TLS", "tcp");
            break;
        case TransportMode::RDMA_ONLY:
            ucp_config_modify(config, "TLS", "rc_verbs,ud_verbs");
            break;
        case TransportMode::SHM_ONLY:
            ucp_config_modify(config, "TLS", "shm");
            break;
        case TransportMode::MULTI_RAIL:
            ucp_config_modify(config, "TLS", "all");
            ucp_config_modify(config, "NET_DEVICES", "all");
            break;
        case TransportMode::GPU_DIRECT:
            if (enable_gpu) {
                ucp_config_modify(config, "TLS", "rc_verbs,cuda_copy,cuda_ipc");
            }
            break;
        case TransportMode::AUTO:
        default:
            // Use default UCX transport selection
            break;
    }
    
    return config;
}

void UCXChannel::update_transport_stats() {
    // Implementation would query UCX for transport statistics
}

uint64_t UCXChannel::get_timestamp_us() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// UCXContextManager implementation

UCXContextManager& UCXContextManager::instance() {
    static UCXContextManager instance;
    return instance;
}

UCXContextManager::~UCXContextManager() {
    for (auto& ctx_info : contexts_) {
        if (ctx_info.context) {
            ucp_cleanup(ctx_info.context);
        }
    }
}

ucp_context_t* UCXContextManager::get_context(TransportMode mode, bool enable_gpu) {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    
    // Look for existing context with same parameters
    for (auto& ctx_info : contexts_) {
        if (ctx_info.mode == mode && ctx_info.gpu_enabled == enable_gpu) {
            ctx_info.reference_count++;
            return ctx_info.context;
        }
    }
    
    // Create new context
    ucp_params_t ucp_params = {};
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_STREAM | UCP_FEATURE_RMA;
    
    if (enable_gpu) {
        ucp_params.features |= UCP_FEATURE_AMO;  // Atomic operations for GPU
    }
    
    ucp_config_t* config = nullptr;
    // Configuration would be set up here based on mode
    
    ucp_context_t* context;
    ucs_status_t status = ucp_init(&ucp_params, config, &context);
    
    if (config) {
        ucp_config_release(config);
    }
    
    if (status != UCS_OK) {
        std::cerr << "Failed to initialize UCX context: " << ucs_status_string(status) << std::endl;
        return nullptr;
    }
    
    contexts_.push_back({context, mode, enable_gpu, 1});
    return context;
}

void UCXContextManager::release_context(ucp_context_t* context) {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    
    for (auto it = contexts_.begin(); it != contexts_.end(); ++it) {
        if (it->context == context) {
            it->reference_count--;
            if (it->reference_count == 0) {
                ucp_cleanup(context);
                contexts_.erase(it);
            }
            break;
        }
    }
}

UCXCapabilities UCXContextManager::get_system_capabilities() {
    if (capabilities_cached_) {
        return cached_capabilities_;
    }
    
    // Discover UCX capabilities
    UCXCapabilities caps;
    caps.available_transports = discover_transports();
    caps.supports_tag_matching = true;
    caps.supports_stream_api = true;
    caps.supports_rma = true;
    caps.supports_atomic = true;
    caps.supports_gpu_memory = true;
    caps.supports_multi_rail = true;
    caps.max_message_size = 1024 * 1024 * 1024;  // 1GB
    caps.max_iov_count = 1024;
    
    cached_capabilities_ = caps;
    capabilities_cached_ = true;
    
    return caps;
}

std::vector<TransportInfo> UCXContextManager::discover_transports() {
    std::vector<TransportInfo> transports;
    
    // Mock transport discovery (real implementation would query UCX)
    transports.push_back({"TCP", "eth0", 1000.0, 50.0, false, false, false, true});
    transports.push_back({"InfiniBand", "mlx5_0", 100000.0, 1.0, true, true, true, false});
    transports.push_back({"Shared Memory", "shm", 50000.0, 0.1, false, false, false, false});
    
    return transports;
}

std::string UCXContextManager::select_optimal_transport(bool require_gpu, bool require_rma) {
    auto transports = discover_transports();
    
    for (const auto& transport : transports) {
        if (require_gpu && !transport.supports_gpu) continue;
        if (require_rma && !transport.supports_rma) continue;
        
        // Prefer high-performance transports
        if (transport.name == "InfiniBand") return "rc_verbs";
        if (transport.name == "Shared Memory") return "shm";
    }
    
    return "tcp";  // Fallback
}

// Factory functions

std::unique_ptr<UCXChannel> create_ucx_server(const std::string& address, TransportMode mode,
                                              size_t buffer_size, bool enable_gpu) {
    auto channel = std::make_unique<UCXChannel>("ucx_server", mode, buffer_size, enable_gpu);
    if (channel->listen(address)) {
        return channel;
    }
    return nullptr;
}

std::unique_ptr<UCXChannel> create_ucx_client(const std::string& server_address, TransportMode mode,
                                              size_t buffer_size, bool enable_gpu) {
    auto channel = std::make_unique<UCXChannel>("ucx_client", mode, buffer_size, enable_gpu);
    if (channel->connect(server_address)) {
        return channel;
    }
    return nullptr;
}

std::unique_ptr<UCXChannel> create_auto_ucx_channel(const std::string& peer_address,
                                                    bool require_gpu, bool require_rma,
                                                    size_t buffer_size) {
    TransportMode mode = TransportMode::AUTO;
    if (require_gpu) {
        mode = TransportMode::GPU_DIRECT;
    }
    
    return std::make_unique<UCXChannel>("ucx_auto", mode, buffer_size, require_gpu);
}

std::unique_ptr<UCXChannel> create_ml_ucx_channel(const std::string& peer_address,
                                                  bool enable_cuda, size_t buffer_size) {
    TransportMode mode = enable_cuda ? TransportMode::GPU_DIRECT : TransportMode::MULTI_RAIL;
    return std::make_unique<UCXChannel>("ucx_ml", mode, buffer_size, enable_cuda);
}

} // namespace ucx
} // namespace psyne

#else // !PSYNE_UCX_SUPPORT

// Stub implementation when UCX is not available
namespace psyne {
namespace ucx {

UCXContextManager& UCXContextManager::instance() {
    static UCXContextManager instance;
    return instance;
}

UCXCapabilities UCXContextManager::get_system_capabilities() {
    return UCXCapabilities{};
}

std::vector<TransportInfo> UCXContextManager::discover_transports() {
    return {};
}

std::string UCXContextManager::select_optimal_transport(bool require_gpu, bool require_rma) {
    return "none";
}

} // namespace ucx
} // namespace psyne

#endif // PSYNE_UCX_SUPPORT
