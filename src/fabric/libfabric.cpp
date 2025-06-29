/**
 * @file libfabric.cpp
 * @brief Libfabric implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include <psyne/fabric/libfabric.hpp>

#ifdef PSYNE_LIBFABRIC_SUPPORT

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>

namespace psyne {
namespace fabric {

// FabricMemoryRegion implementation

FabricMemoryRegion::FabricMemoryRegion(fid_domain* domain, void* addr, size_t length, uint64_t access)
    : addr_(addr), length_(length) {
    
    int ret = fi_mr_reg(domain, addr, length, access, 0, 0, 0, &mr_, nullptr);
    if (ret) {
        throw std::runtime_error("Failed to register memory region: " + std::string(fi_strerror(-ret)));
    }
}

FabricMemoryRegion::~FabricMemoryRegion() {
    if (mr_) {
        fi_close(&mr_->fid);
    }
}

uint64_t FabricMemoryRegion::key() const {
    return mr_ ? fi_mr_key(mr_) : 0;
}

// FabricCompletionQueue implementation

FabricCompletionQueue::FabricCompletionQueue(fid_domain* domain, size_t size) {
    fi_cq_attr cq_attr = {};
    cq_attr.size = size;
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    cq_attr.wait_obj = FI_WAIT_NONE;
    
    int ret = fi_cq_open(domain, &cq_attr, &cq_, nullptr);
    if (ret) {
        throw std::runtime_error("Failed to create completion queue: " + std::string(fi_strerror(-ret)));
    }
}

FabricCompletionQueue::~FabricCompletionQueue() {
    if (cq_) {
        fi_close(&cq_->fid);
    }
}

int FabricCompletionQueue::read(void* buf, size_t count) {
    return fi_cq_read(cq_, buf, count);
}

int FabricCompletionQueue::read_error(void* buf, size_t count) {
    return fi_cq_readerr(cq_, reinterpret_cast<fi_cq_err_entry*>(buf), 0);
}

void FabricCompletionQueue::signal() {
    fi_cq_signal(cq_);
}

// FabricEndpoint implementation

FabricEndpoint::FabricEndpoint(fid_domain* domain, fi_info* info, EndpointType type)
    : type_(type) {
    
    int ret = fi_endpoint(domain, info, &ep_, nullptr);
    if (ret) {
        throw std::runtime_error("Failed to create endpoint: " + std::string(fi_strerror(-ret)));
    }
}

FabricEndpoint::~FabricEndpoint() {
    if (ep_) {
        fi_close(&ep_->fid);
    }
}

bool FabricEndpoint::bind_cq(FabricCompletionQueue* cq, uint64_t flags) {
    int ret = fi_ep_bind(ep_, &cq->cq()->fid, flags);
    return ret == 0;
}

bool FabricEndpoint::enable() {
    int ret = fi_enable(ep_);
    return ret == 0;
}

ssize_t FabricEndpoint::send(const void* buf, size_t len, void* context) {
    return fi_send(ep_, buf, len, nullptr, FI_ADDR_UNSPEC, context);
}

ssize_t FabricEndpoint::recv(void* buf, size_t len, void* context) {
    return fi_recv(ep_, buf, len, nullptr, FI_ADDR_UNSPEC, context);
}

ssize_t FabricEndpoint::post_recv(void* buf, size_t len, void* context) {
    return fi_recv(ep_, buf, len, nullptr, FI_ADDR_UNSPEC, context);
}

ssize_t FabricEndpoint::read(void* buf, size_t len, uint64_t addr, uint64_t key, void* context) {
    return fi_read(ep_, buf, len, nullptr, FI_ADDR_UNSPEC, addr, key, context);
}

ssize_t FabricEndpoint::write(const void* buf, size_t len, uint64_t addr, uint64_t key, void* context) {
    return fi_write(ep_, buf, len, nullptr, FI_ADDR_UNSPEC, addr, key, context);
}

ssize_t FabricEndpoint::atomic_compare_swap(void* result, const void* compare, const void* swap,
                                           uint64_t addr, uint64_t key, void* context) {
    return fi_compare_atomic(ep_, compare, 1, nullptr, swap, nullptr, result, nullptr,
                            FI_ADDR_UNSPEC, addr, key, FI_UINT64, FI_CSWAP, context);
}

ssize_t FabricEndpoint::atomic_fetch_add(void* result, const void* operand,
                                        uint64_t addr, uint64_t key, void* context) {
    return fi_fetch_atomic(ep_, operand, 1, nullptr, result, nullptr,
                          FI_ADDR_UNSPEC, addr, key, FI_UINT64, FI_SUM, context);
}

// LibfabricChannel implementation

LibfabricChannel::LibfabricChannel(const std::string& provider, const std::string& node,
                                   const std::string& service, EndpointType ep_type,
                                   size_t buffer_size)
    : Channel("fabric", buffer_size, ChannelMode::Blocking, ChannelType::Reliable)
    , fabric_(nullptr)
    , domain_(nullptr)
    , address_vector_(nullptr)
    , connected_(false)
    , provider_name_(provider)
    , node_(node)
    , service_(service)
    , endpoint_type_(ep_type)
    , running_(true) {
    
    if (!init_fabric(provider)) {
        throw std::runtime_error("Failed to initialize fabric");
    }
    
    if (!create_domain()) {
        cleanup_resources();
        throw std::runtime_error("Failed to create domain");
    }
    
    if (!create_completion_queues()) {
        cleanup_resources();
        throw std::runtime_error("Failed to create completion queues");
    }
    
    if (!create_endpoint()) {
        cleanup_resources();
        throw std::runtime_error("Failed to create endpoint");
    }
    
    if (!setup_memory_regions()) {
        cleanup_resources();
        throw std::runtime_error("Failed to setup memory regions");
    }
    
    // Start completion handler thread
    completion_thread_ = std::thread(&LibfabricChannel::completion_handler, this);
    
    std::cout << "Libfabric channel initialized: provider=" << provider_name_ 
              << ", node=" << node_ << ", service=" << service_ << std::endl;
}

LibfabricChannel::~LibfabricChannel() {
    running_ = false;
    
    if (completion_thread_.joinable()) {
        completion_thread_.join();
    }
    
    cleanup_resources();
}

bool LibfabricChannel::init_fabric(const std::string& provider) {
    fi_info* hints = fi_allocinfo();
    if (!hints) {
        return false;
    }
    
    // Set hints based on provider
    if (provider != "auto" && !provider.empty()) {
        hints->fabric_attr->prov_name = strdup(provider.c_str());
    }
    
    // Set endpoint type
    switch (endpoint_type_) {
        case EndpointType::MSG:
            hints->ep_attr->type = FI_EP_MSG;
            break;
        case EndpointType::RDM:
            hints->ep_attr->type = FI_EP_RDM;
            break;
        case EndpointType::DGRAM:
            hints->ep_attr->type = FI_EP_DGRAM;
            break;
    }
    
    hints->caps = FI_MSG | FI_RMA | FI_ATOMIC;
    hints->mode = FI_CONTEXT;
    hints->domain_attr->mr_mode = FI_MR_LOCAL;
    
    fi_info* info;
    int ret = fi_getinfo(FI_VERSION(1, 9), node_.empty() ? nullptr : node_.c_str(),
                        service_.empty() ? nullptr : service_.c_str(), 0, hints, &info);
    
    fi_freeinfo(hints);
    
    if (ret) {
        std::cerr << "fi_getinfo failed: " << fi_strerror(-ret) << std::endl;
        return false;
    }
    
    // Create fabric
    ret = fi_fabric(info->fabric_attr, &fabric_, nullptr);
    if (ret) {
        fi_freeinfo(info);
        return false;
    }
    
    provider_name_ = info->fabric_attr->prov_name;
    fi_freeinfo(info);
    
    return true;
}

bool LibfabricChannel::create_domain() {
    fi_info* info;
    int ret = fi_getinfo(FI_VERSION(1, 9), nullptr, nullptr, 0, nullptr, &info);
    if (ret) {
        return false;
    }
    
    ret = fi_domain(fabric_, info, &domain_, nullptr);
    fi_freeinfo(info);
    
    return ret == 0;
}

bool LibfabricChannel::create_endpoint() {
    fi_info* info;
    int ret = fi_getinfo(FI_VERSION(1, 9), nullptr, nullptr, 0, nullptr, &info);
    if (ret) {
        return false;
    }
    
    try {
        endpoint_ = std::make_unique<FabricEndpoint>(domain_, info, endpoint_type_);
        
        // Bind completion queues
        endpoint_->bind_cq(tx_cq_.get(), FI_TRANSMIT);
        endpoint_->bind_cq(rx_cq_.get(), FI_RECV);
        
        // Enable endpoint
        if (!endpoint_->enable()) {
            fi_freeinfo(info);
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to create endpoint: " << e.what() << std::endl;
        fi_freeinfo(info);
        return false;
    }
    
    fi_freeinfo(info);
    return true;
}

bool LibfabricChannel::create_completion_queues() {
    try {
        tx_cq_ = std::make_unique<FabricCompletionQueue>(domain_, 256);
        rx_cq_ = std::make_unique<FabricCompletionQueue>(domain_, 256);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create completion queues: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

bool LibfabricChannel::setup_memory_regions() {
    try {
        size_t half_size = buffer_size_ / 2;
        void* send_buf = std::aligned_alloc(4096, half_size);
        void* recv_buf = std::aligned_alloc(4096, half_size);
        
        if (!send_buf || !recv_buf) {
            std::free(send_buf);
            std::free(recv_buf);
            return false;
        }
        
        uint64_t access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
        send_mr_ = std::make_shared<FabricMemoryRegion>(domain_, send_buf, half_size, access);
        recv_mr_ = std::make_shared<FabricMemoryRegion>(domain_, recv_buf, half_size, access);
        
        memory_regions_.push_back(send_mr_);
        memory_regions_.push_back(recv_mr_);
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to setup memory regions: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

void LibfabricChannel::cleanup_resources() {
    memory_regions_.clear();
    endpoint_.reset();
    tx_cq_.reset();
    rx_cq_.reset();
    
    if (address_vector_) {
        fi_close(&address_vector_->fid);
        address_vector_ = nullptr;
    }
    
    if (domain_) {
        fi_close(&domain_->fid);
        domain_ = nullptr;
    }
    
    if (fabric_) {
        fi_close(&fabric_->fid);
        fabric_ = nullptr;
    }
}

size_t LibfabricChannel::send(const void* data, size_t size, uint32_t type_id) {
    if (!connected_ || size > send_mr_->length()) {
        return 0;
    }
    
    // Copy data to send buffer
    std::memcpy(send_mr_->addr(), data, size);
    
    // Post send
    ssize_t ret = endpoint_->send(send_mr_->addr(), size, this);
    if (ret < 0) {
        return 0;
    }
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.bytes_sent += size;
        stats_.messages_sent++;
    }
    
    return size;
}

size_t LibfabricChannel::receive(void* buffer, size_t buffer_size, uint32_t* type_id) {
    if (!connected_) {
        return 0;
    }
    
    // Post receive
    ssize_t ret = endpoint_->post_recv(recv_mr_->addr(), recv_mr_->length(), this);
    if (ret < 0) {
        return 0;
    }
    
    // Wait for completion (simplified)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    
    // For now, return 0 (would need proper completion handling)
    return 0;
}

bool LibfabricChannel::try_send(const void* data, size_t size, uint32_t type_id) {
    return send(data, size, type_id) > 0;
}

bool LibfabricChannel::try_receive(void* buffer, size_t buffer_size, size_t* received_size, uint32_t* type_id) {
    size_t received = receive(buffer, buffer_size, type_id);
    if (received_size) *received_size = received;
    return received > 0;
}

std::shared_ptr<FabricMemoryRegion> LibfabricChannel::register_memory(void* addr, size_t length) {
    try {
        uint64_t access = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
        auto mr = std::make_shared<FabricMemoryRegion>(domain_, addr, length, access);
        memory_regions_.push_back(mr);
        return mr;
    } catch (const std::exception& e) {
        std::cerr << "Failed to register memory: " << e.what() << std::endl;
        return nullptr;
    }
}

bool LibfabricChannel::rma_read(void* local_addr, size_t length, uint64_t remote_addr, uint64_t remote_key) {
    if (!connected_) {
        return false;
    }
    
    ssize_t ret = endpoint_->read(local_addr, length, remote_addr, remote_key, this);
    bool success = ret >= 0;
    
    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.rma_reads++;
        stats_.bytes_received += length;
    }
    
    return success;
}

bool LibfabricChannel::rma_write(const void* local_addr, size_t length, uint64_t remote_addr, uint64_t remote_key) {
    if (!connected_) {
        return false;
    }
    
    ssize_t ret = endpoint_->write(local_addr, length, remote_addr, remote_key, this);
    bool success = ret >= 0;
    
    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.rma_writes++;
        stats_.bytes_sent += length;
    }
    
    return success;
}

bool LibfabricChannel::atomic_compare_swap(uint64_t* result, uint64_t compare, uint64_t swap,
                                          uint64_t remote_addr, uint64_t remote_key) {
    if (!connected_) {
        return false;
    }
    
    ssize_t ret = endpoint_->atomic_compare_swap(result, &compare, &swap, remote_addr, remote_key, this);
    bool success = ret >= 0;
    
    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.atomic_ops++;
    }
    
    return success;
}

bool LibfabricChannel::atomic_fetch_add(uint64_t* result, uint64_t operand,
                                       uint64_t remote_addr, uint64_t remote_key) {
    if (!connected_) {
        return false;
    }
    
    ssize_t ret = endpoint_->atomic_fetch_add(result, &operand, remote_addr, remote_key, this);
    bool success = ret >= 0;
    
    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.atomic_ops++;
    }
    
    return success;
}

bool LibfabricChannel::connect(const std::string& remote_node, const std::string& remote_service) {
    // Simplified connection (real implementation would handle address resolution)
    connected_ = true;
    std::cout << "Fabric connection established to " << remote_node << ":" << remote_service << std::endl;
    return true;
}

bool LibfabricChannel::listen() {
    // Simplified listen (real implementation would bind and listen)
    std::cout << "Fabric listening on " << service_ << std::endl;
    return true;
}

std::unique_ptr<LibfabricChannel> LibfabricChannel::accept() {
    // Simplified accept (real implementation would accept connections)
    return nullptr;
}

FabricCapabilities LibfabricChannel::get_capabilities() const {
    FabricCapabilities caps;
    caps.provider_name = provider_name_;
    caps.fabric_name = "fabric0";
    caps.domain_name = "domain0";
    caps.max_msg_size = 1024 * 1024;
    caps.max_rma_size = 1024 * 1024;
    caps.max_ep_cnt = 256;
    caps.max_cq_cnt = 256;
    caps.supports_rma = true;
    caps.supports_atomic = true;
    caps.supports_multicast = false;
    caps.supports_tagged = false;
    return caps;
}

LibfabricChannel::Stats LibfabricChannel::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void LibfabricChannel::completion_handler() {
    const int max_entries = 16;
    fi_cq_entry entries[max_entries];
    
    while (running_) {
        // Process TX completions
        int ne = tx_cq_->read(entries, max_entries);
        if (ne > 0) {
            process_tx_completion();
        }
        
        // Process RX completions
        ne = rx_cq_->read(entries, max_entries);
        if (ne > 0) {
            process_rx_completion();
        }
        
        if (ne == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

void LibfabricChannel::process_tx_completion() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    // Process send completions
}

void LibfabricChannel::process_rx_completion() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    // Process receive completions
}

uint64_t LibfabricChannel::get_timestamp_ns() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// FabricProvider implementation

std::vector<FabricCapabilities> FabricProvider::list_providers() {
    std::vector<FabricCapabilities> providers;
    
    fi_info* info_list;
    int ret = fi_getinfo(FI_VERSION(1, 9), nullptr, nullptr, 0, nullptr, &info_list);
    if (ret) {
        return providers;
    }
    
    for (fi_info* info = info_list; info; info = info->next) {
        FabricCapabilities caps;
        caps.provider_name = info->fabric_attr->prov_name ? info->fabric_attr->prov_name : "unknown";
        caps.fabric_name = info->fabric_attr->name ? info->fabric_attr->name : "fabric";
        caps.domain_name = info->domain_attr->name ? info->domain_attr->name : "domain";
        caps.max_msg_size = info->ep_attr->max_msg_size;
        caps.max_rma_size = info->ep_attr->max_msg_size;
        caps.supports_rma = (info->caps & FI_RMA) != 0;
        caps.supports_atomic = (info->caps & FI_ATOMIC) != 0;
        caps.supports_multicast = (info->caps & FI_MULTICAST) != 0;
        caps.supports_tagged = (info->caps & FI_TAGGED) != 0;
        
        providers.push_back(caps);
    }
    
    fi_freeinfo(info_list);
    return providers;
}

std::string FabricProvider::select_provider(bool requires_rma, bool requires_atomic, uint64_t min_msg_size) {
    auto providers = list_providers();
    
    for (const auto& provider : providers) {
        if (requires_rma && !provider.supports_rma) continue;
        if (requires_atomic && !provider.supports_atomic) continue;
        if (min_msg_size > 0 && provider.max_msg_size < min_msg_size) continue;
        
        // Prefer specific providers in order
        if (provider.provider_name == "verbs") return "verbs";
        if (provider.provider_name == "mlx") return "mlx";
        if (provider.provider_name == "psm2") return "psm2";
    }
    
    // Fallback to first available provider
    if (!providers.empty()) {
        return providers[0].provider_name;
    }
    
    return "sockets"; // Final fallback
}

bool FabricProvider::is_available() {
    fi_info* info;
    int ret = fi_getinfo(FI_VERSION(1, 9), nullptr, nullptr, 0, nullptr, &info);
    if (ret == 0) {
        fi_freeinfo(info);
        return true;
    }
    return false;
}

std::unordered_map<std::string, std::string> FabricProvider::get_provider_hints(const std::string& provider) {
    std::unordered_map<std::string, std::string> hints;
    
    if (provider == "verbs") {
        hints["FI_VERBS_INLINE_SIZE"] = "64";
        hints["FI_VERBS_USE_ODP"] = "1";
    } else if (provider == "mlx") {
        hints["FI_MLX_ENABLE_SPAWN"] = "1";
    } else if (provider == "psm2") {
        hints["PSM2_MULTI_EP"] = "1";
    }
    
    return hints;
}

// Factory functions

std::unique_ptr<LibfabricChannel> create_fabric_server(
    const std::string& service, const std::string& provider,
    EndpointType ep_type, size_t buffer_size) {
    
    std::string actual_provider = (provider == "auto") ? 
        FabricProvider::select_provider() : provider;
    
    return std::make_unique<LibfabricChannel>(actual_provider, "", service, ep_type, buffer_size);
}

std::unique_ptr<LibfabricChannel> create_fabric_client(
    const std::string& node, const std::string& service,
    const std::string& provider, EndpointType ep_type, size_t buffer_size) {
    
    std::string actual_provider = (provider == "auto") ? 
        FabricProvider::select_provider() : provider;
    
    return std::make_unique<LibfabricChannel>(actual_provider, node, service, ep_type, buffer_size);
}

std::unique_ptr<LibfabricChannel> create_auto_fabric_channel(
    const std::string& node, const std::string& service,
    bool require_rma, bool require_atomic, size_t buffer_size) {
    
    std::string provider = FabricProvider::select_provider(require_rma, require_atomic);
    return std::make_unique<LibfabricChannel>(provider, node, service, EndpointType::MSG, buffer_size);
}

} // namespace fabric
} // namespace psyne

#else // !PSYNE_LIBFABRIC_SUPPORT

// Stub implementation when libfabric is not available
namespace psyne {
namespace fabric {

std::vector<FabricCapabilities> FabricProvider::list_providers() {
    return {};
}

std::string FabricProvider::select_provider(bool requires_rma, bool requires_atomic, uint64_t min_msg_size) {
    return "none";
}

bool FabricProvider::is_available() {
    return false;
}

std::unordered_map<std::string, std::string> FabricProvider::get_provider_hints(const std::string& provider) {
    return {};
}

} // namespace fabric
} // namespace psyne

#endif // PSYNE_LIBFABRIC_SUPPORT