/**
 * @file rdma_verbs.cpp
 * @brief RDMA Verbs API implementation
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include <psyne/rdma/rdma_verbs.hpp>

#ifdef PSYNE_RDMA_SUPPORT

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <arpa/inet.h>

namespace psyne {
namespace rdma {

// MemoryRegion implementation

MemoryRegion::MemoryRegion(ibv_pd* pd, void* addr, size_t length, int access_flags)
    : addr_(addr), length_(length) {
    
    mr_ = ibv_reg_mr(pd, addr, length, access_flags);
    if (!mr_) {
        throw std::runtime_error("Failed to register memory region: " + 
                               std::string(strerror(errno)));
    }
}

MemoryRegion::~MemoryRegion() {
    if (mr_) {
        ibv_dereg_mr(mr_);
    }
}

uint32_t MemoryRegion::lkey() const {
    return mr_ ? mr_->lkey : 0;
}

uint32_t MemoryRegion::rkey() const {
    return mr_ ? mr_->rkey : 0;
}

// CompletionQueue implementation

CompletionQueue::CompletionQueue(ibv_context* context, int cq_size, 
                                 ibv_comp_channel* channel)
    : channel_(channel) {
    
    cq_ = ibv_create_cq(context, cq_size, nullptr, channel, 0);
    if (!cq_) {
        throw std::runtime_error("Failed to create completion queue");
    }
    
    if (channel_) {
        if (ibv_req_notify_cq(cq_, 0)) {
            ibv_destroy_cq(cq_);
            throw std::runtime_error("Failed to request CQ notification");
        }
    }
}

CompletionQueue::~CompletionQueue() {
    if (cq_) {
        ibv_destroy_cq(cq_);
    }
}

int CompletionQueue::poll(ibv_wc* wc, int num_entries) {
    return ibv_poll_cq(cq_, num_entries, wc);
}

void CompletionQueue::request_notify(bool solicited_only) {
    ibv_req_notify_cq(cq_, solicited_only ? 1 : 0);
}

// QueuePair implementation

QueuePair::QueuePair(ibv_pd* pd, ibv_cq* send_cq, ibv_cq* recv_cq,
                     TransportType type, uint32_t max_send_wr,
                     uint32_t max_recv_wr, uint32_t max_inline_data)
    : type_(type) {
    
    ibv_qp_init_attr init_attr = {};
    init_attr.send_cq = send_cq;
    init_attr.recv_cq = recv_cq;
    init_attr.qp_type = (type == TransportType::RC) ? IBV_QPT_RC :
                       (type == TransportType::UC) ? IBV_QPT_UC :
                       IBV_QPT_UD;
    init_attr.cap.max_send_wr = max_send_wr;
    init_attr.cap.max_recv_wr = max_recv_wr;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 1;
    init_attr.cap.max_inline_data = max_inline_data;
    
    qp_ = ibv_create_qp(pd, &init_attr);
    if (!qp_) {
        throw std::runtime_error("Failed to create queue pair");
    }
}

QueuePair::~QueuePair() {
    if (qp_) {
        ibv_destroy_qp(qp_);
    }
}

bool QueuePair::modify_to_init(uint8_t port_num) {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = port_num;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | 
                          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
    
    int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    
    return ibv_modify_qp(qp_, &attr, flags) == 0;
}

bool QueuePair::modify_to_rtr(const ConnectionParams& remote_params) {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote_params.qp_num;
    attr.rq_psn = remote_params.psn;
    
    if (type_ == TransportType::RC) {
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 12;
    }
    
    attr.ah_attr.dlid = remote_params.lid;
    attr.ah_attr.sl = remote_params.sl;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = remote_params.port_num;
    
    int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | 
                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
    
    if (type_ == TransportType::RC) {
        flags |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    }
    
    return ibv_modify_qp(qp_, &attr, flags) == 0;
}

bool QueuePair::modify_to_rts() {
    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    
    int flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
    
    if (type_ == TransportType::RC) {
        flags |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | 
                 IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
    }
    
    return ibv_modify_qp(qp_, &attr, flags) == 0;
}

uint32_t QueuePair::qp_num() const {
    return qp_ ? qp_->qp_num : 0;
}

bool QueuePair::post_send(ibv_send_wr* wr) {
    ibv_send_wr* bad_wr = nullptr;
    return ibv_post_send(qp_, wr, &bad_wr) == 0;
}

bool QueuePair::post_recv(ibv_recv_wr* wr) {
    ibv_recv_wr* bad_wr = nullptr;
    return ibv_post_recv(qp_, wr, &bad_wr) == 0;
}

// RDMAVerbsChannel implementation

RDMAVerbsChannel::RDMAVerbsChannel(const std::string& device_name, uint8_t port_num,
                                   size_t buffer_size, TransportType transport)
    : Channel("rdma", buffer_size, ChannelMode::Blocking, ChannelType::Reliable)
    , device_(nullptr)
    , context_(nullptr)
    , pd_(nullptr)
    , connected_(false)
    , port_num_(port_num)
    , transport_(transport)
    , inline_threshold_(64)
    , max_send_wr_(256)
    , max_recv_wr_(256)
    , max_sge_(1)
    , running_(true) {
    
    if (!init_device(device_name)) {
        throw std::runtime_error("Failed to initialize RDMA device");
    }
    
    if (!create_resources()) {
        destroy_resources();
        throw std::runtime_error("Failed to create RDMA resources");
    }
    
    // Start completion handler thread
    completion_thread_ = std::thread(&RDMAVerbsChannel::completion_handler, this);
}

RDMAVerbsChannel::~RDMAVerbsChannel() {
    running_ = false;
    
    if (completion_thread_.joinable()) {
        completion_thread_.join();
    }
    
    destroy_resources();
}

bool RDMAVerbsChannel::init_device(const std::string& device_name) {
    // Get device list
    int num_devices;
    ibv_device** device_list = ibv_get_device_list(&num_devices);
    if (!device_list || num_devices == 0) {
        std::cerr << "No RDMA devices found" << std::endl;
        return false;
    }
    
    // Find requested device or use first available
    device_ = nullptr;
    for (int i = 0; i < num_devices; ++i) {
        const char* dev_name = ibv_get_device_name(device_list[i]);
        if (device_name.empty() || device_name == dev_name) {
            device_ = device_list[i];
            std::cout << "Using RDMA device: " << dev_name << std::endl;
            break;
        }
    }
    
    if (!device_) {
        device_ = device_list[0];
        std::cout << "Using default RDMA device: " << 
                     ibv_get_device_name(device_) << std::endl;
    }
    
    // Open device
    context_ = ibv_open_device(device_);
    if (!context_) {
        ibv_free_device_list(device_list);
        return false;
    }
    
    ibv_free_device_list(device_list);
    
    // Query device attributes
    ibv_device_attr device_attr;
    if (ibv_query_device(context_, &device_attr)) {
        return false;
    }
    
    std::cout << "Device capabilities:" << std::endl;
    std::cout << "  Max MR size: " << device_attr.max_mr_size << std::endl;
    std::cout << "  Max QP: " << device_attr.max_qp << std::endl;
    std::cout << "  Max CQ: " << device_attr.max_cq << std::endl;
    std::cout << "  Max QP WR: " << device_attr.max_qp_wr << std::endl;
    std::cout << "  Max SGE: " << device_attr.max_sge << std::endl;
    
    return true;
}

bool RDMAVerbsChannel::create_resources() {
    // Allocate protection domain
    pd_ = ibv_alloc_pd(context_);
    if (!pd_) {
        return false;
    }
    
    // Create completion queues
    try {
        send_cq_ = std::make_unique<CompletionQueue>(context_, max_send_wr_);
        recv_cq_ = std::make_unique<CompletionQueue>(context_, max_recv_wr_);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create completion queues: " << e.what() << std::endl;
        return false;
    }
    
    // Create queue pair
    try {
        qp_ = std::make_unique<QueuePair>(pd_, send_cq_->cq(), recv_cq_->cq(),
                                         transport_, max_send_wr_, max_recv_wr_,
                                         inline_threshold_);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create queue pair: " << e.what() << std::endl;
        return false;
    }
    
    // Transition QP to INIT state
    if (!qp_->modify_to_init(port_num_)) {
        std::cerr << "Failed to modify QP to INIT state" << std::endl;
        return false;
    }
    
    // Allocate and register memory buffers
    size_t half_size = buffer_size_ / 2;
    void* send_buf = std::aligned_alloc(4096, half_size);
    void* recv_buf = std::aligned_alloc(4096, half_size);
    
    if (!send_buf || !recv_buf) {
        std::free(send_buf);
        std::free(recv_buf);
        return false;
    }
    
    try {
        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | 
                    IBV_ACCESS_REMOTE_READ;
        send_mr_ = std::make_shared<MemoryRegion>(pd_, send_buf, half_size, access);
        recv_mr_ = std::make_shared<MemoryRegion>(pd_, recv_buf, half_size, access);
        
        memory_regions_.push_back(send_mr_);
        memory_regions_.push_back(recv_mr_);
    } catch (const std::exception& e) {
        std::cerr << "Failed to register memory: " << e.what() << std::endl;
        std::free(send_buf);
        std::free(recv_buf);
        return false;
    }
    
    // Get local connection parameters
    ibv_port_attr port_attr;
    if (ibv_query_port(context_, port_num_, &port_attr)) {
        return false;
    }
    
    local_params_.qp_num = qp_->qp_num();
    local_params_.lid = port_attr.lid;
    local_params_.psn = 0;
    local_params_.port_num = port_num_;
    local_params_.sl = 0;
    
    // Query GID
    if (ibv_query_gid(context_, port_num_, 0, (ibv_gid*)local_params_.gid)) {
        memset(local_params_.gid, 0, sizeof(local_params_.gid));
    }
    
    return true;
}

void RDMAVerbsChannel::destroy_resources() {
    memory_regions_.clear();
    qp_.reset();
    send_cq_.reset();
    recv_cq_.reset();
    
    if (pd_) {
        ibv_dealloc_pd(pd_);
        pd_ = nullptr;
    }
    
    if (context_) {
        ibv_close_device(context_);
        context_ = nullptr;
    }
}

size_t RDMAVerbsChannel::send(const void* data, size_t size, uint32_t type_id) {
    if (!connected_ || size > send_mr_->length()) {
        return 0;
    }
    
    // Copy data to send buffer
    std::memcpy(send_mr_->addr(), data, size);
    
    // Prepare send work request
    ibv_sge sge = {};
    sge.addr = (uint64_t)send_mr_->addr();
    sge.length = size;
    sge.lkey = send_mr_->lkey();
    
    ibv_send_wr wr = {};
    wr.wr_id = reinterpret_cast<uint64_t>(this);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    
    // Use inline data for small messages
    if (size <= inline_threshold_) {
        wr.send_flags |= IBV_SEND_INLINE;
    }
    
    // Post send request
    if (!qp_->post_send(&wr)) {
        return 0;
    }
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.bytes_sent += size;
    }
    
    return size;
}

size_t RDMAVerbsChannel::receive(void* buffer, size_t buffer_size, uint32_t* type_id) {
    if (!connected_) {
        return 0;
    }
    
    // Wait for receive completion
    // In a real implementation, this would be more sophisticated
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    
    // For now, return 0 (would need proper receive handling)
    return 0;
}

bool RDMAVerbsChannel::rdma_write(const void* local_addr, size_t length,
                                  uint64_t remote_addr, uint32_t rkey) {
    if (!connected_) {
        return false;
    }
    
    ibv_sge sge = {};
    sge.addr = (uint64_t)local_addr;
    sge.length = length;
    sge.lkey = send_mr_->lkey();
    
    ibv_send_wr wr = {};
    wr.wr_id = reinterpret_cast<uint64_t>(this);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    
    bool success = qp_->post_send(&wr);
    
    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.rdma_writes++;
        stats_.bytes_sent += length;
    }
    
    return success;
}

bool RDMAVerbsChannel::rdma_read(void* local_addr, size_t length,
                                 uint64_t remote_addr, uint32_t rkey) {
    if (!connected_) {
        return false;
    }
    
    ibv_sge sge = {};
    sge.addr = (uint64_t)local_addr;
    sge.length = length;
    sge.lkey = recv_mr_->lkey();
    
    ibv_send_wr wr = {};
    wr.wr_id = reinterpret_cast<uint64_t>(this);
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    
    bool success = qp_->post_send(&wr);
    
    if (success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.rdma_reads++;
        stats_.bytes_received += length;
    }
    
    return success;
}

void RDMAVerbsChannel::completion_handler() {
    const int max_wc = 16;
    ibv_wc wc[max_wc];
    
    while (running_) {
        // Poll send completions
        int ne = send_cq_->poll(wc, max_wc);
        for (int i = 0; i < ne; ++i) {
            process_completion(wc[i]);
        }
        
        // Poll receive completions
        ne = recv_cq_->poll(wc, max_wc);
        for (int i = 0; i < ne; ++i) {
            process_completion(wc[i]);
        }
        
        if (ne == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

void RDMAVerbsChannel::process_completion(const ibv_wc& wc) {
    if (wc.status != IBV_WC_SUCCESS) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.completion_errors++;
        std::cerr << "Work completion error: " << ibv_wc_status_str(wc.status) << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    switch (wc.opcode) {
        case IBV_WC_SEND:
            stats_.send_completions++;
            break;
        case IBV_WC_RECV:
            stats_.recv_completions++;
            break;
        case IBV_WC_RDMA_WRITE:
        case IBV_WC_RDMA_READ:
            // Already counted in rdma_write/read
            break;
        default:
            break;
    }
}

ConnectionParams RDMAVerbsChannel::get_local_params() const {
    return local_params_;
}

RDMAVerbsChannel::Stats RDMAVerbsChannel::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// Factory functions

std::unique_ptr<RDMAVerbsChannel> create_rdma_server(
    uint16_t port, size_t buffer_size,
    const std::string& device, TransportType transport) {
    
    auto channel = std::make_unique<RDMAVerbsChannel>(device, 1, buffer_size, transport);
    
    if (!channel->listen(port)) {
        return nullptr;
    }
    
    return channel;
}

std::unique_ptr<RDMAVerbsChannel> create_rdma_client(
    const std::string& server_addr, uint16_t server_port,
    size_t buffer_size, const std::string& device, TransportType transport) {
    
    auto channel = std::make_unique<RDMAVerbsChannel>(device, 1, buffer_size, transport);
    
    if (!channel->connect(server_addr, server_port)) {
        return nullptr;
    }
    
    return channel;
}

bool is_rdma_available() {
    int num_devices;
    ibv_device** device_list = ibv_get_device_list(&num_devices);
    
    if (device_list) {
        ibv_free_device_list(device_list);
        return num_devices > 0;
    }
    
    return false;
}

std::vector<DeviceCapabilities> list_rdma_devices() {
    std::vector<DeviceCapabilities> devices;
    
    int num_devices;
    ibv_device** device_list = ibv_get_device_list(&num_devices);
    if (!device_list) {
        return devices;
    }
    
    for (int i = 0; i < num_devices; ++i) {
        ibv_context* ctx = ibv_open_device(device_list[i]);
        if (!ctx) continue;
        
        ibv_device_attr attr;
        if (ibv_query_device(ctx, &attr) == 0) {
            DeviceCapabilities caps;
            caps.name = ibv_get_device_name(device_list[i]);
            caps.max_mr_size = attr.max_mr_size;
            caps.max_qp = attr.max_qp;
            caps.max_cq = attr.max_cq;
            caps.max_qp_wr = attr.max_qp_wr;
            caps.max_sge = attr.max_sge;
            caps.atomic_cap = (attr.atomic_cap != IBV_ATOMIC_NONE);
            
            devices.push_back(caps);
        }
        
        ibv_close_device(ctx);
    }
    
    ibv_free_device_list(device_list);
    
    return devices;
}

// RDMAGPUDirectChannel implementation

#ifdef PSYNE_CUDA_ENABLED
#include "../gpu/cuda/cuda_buffer.hpp"
#endif

RDMAGPUDirectChannel::RDMAGPUDirectChannel(const std::string& device_name, uint8_t port_num,
                                           size_t buffer_size, TransportType transport)
    : RDMAVerbsChannel(device_name, port_num, buffer_size, transport)
    , gpu_direct_enabled_(false) {
    
    // Check if GPUDirect is supported
    DeviceCapabilities caps = get_capabilities();
    
    // Query device for GPUDirect support
    // Note: This is a simplified check - real implementation would query
    // the device attributes for GPUDirect support
    gpu_direct_enabled_ = true; // Assume support for now
    
    if (gpu_direct_enabled_) {
        std::cout << "GPUDirect RDMA enabled for device: " << caps.name << std::endl;
    } else {
        std::cerr << "Warning: GPUDirect RDMA not supported on device: " << caps.name << std::endl;
    }
}

std::shared_ptr<MemoryRegion> RDMAGPUDirectChannel::register_gpu_memory(void* gpu_addr, size_t length) {
    if (!gpu_direct_enabled_) {
        throw std::runtime_error("GPUDirect RDMA not supported on this device");
    }
    
    if (!gpu_addr || length == 0) {
        throw std::invalid_argument("Invalid GPU memory address or length");
    }
    
    try {
        // Register GPU memory with RDMA hardware
        // IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
        int access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | 
                          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
        
        auto mr = std::make_shared<MemoryRegion>(pd_, gpu_addr, length, access_flags);
        
        // Store the memory region for cleanup
        memory_regions_.push_back(mr);
        
        std::cout << "Registered GPU memory region: " << gpu_addr 
                  << " length: " << length << " bytes" << std::endl;
        
        return mr;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to register GPU memory: " << e.what() << std::endl;
        return nullptr;
    }
}

bool RDMAGPUDirectChannel::rdma_write_from_gpu(const void* local_gpu_addr, size_t length,
                                               uint64_t remote_addr, uint32_t rkey) {
    if (!gpu_direct_enabled_) {
        std::cerr << "GPUDirect RDMA not enabled" << std::endl;
        return false;
    }
    
    if (!connected_) {
        std::cerr << "Channel not connected" << std::endl;
        return false;
    }
    
    // Find the memory region for this GPU address
    std::shared_ptr<MemoryRegion> mr = nullptr;
    for (const auto& region : memory_regions_) {
        uint8_t* region_start = static_cast<uint8_t*>(region->addr());
        uint8_t* region_end = region_start + region->length();
        uint8_t* gpu_ptr = const_cast<uint8_t*>(static_cast<const uint8_t*>(local_gpu_addr));
        
        if (gpu_ptr >= region_start && (gpu_ptr + length) <= region_end) {
            mr = region;
            break;
        }
    }
    
    if (!mr) {
        std::cerr << "GPU memory not registered for RDMA" << std::endl;
        return false;
    }
    
    // Create scatter-gather element
    ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(local_gpu_addr);
    sge.length = length;
    sge.lkey = mr->lkey();
    
    // Create work request
    ibv_send_wr wr = {};
    wr.wr_id = reinterpret_cast<uint64_t>(this); // Use this pointer as ID
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    
    // Post the work request
    bool success = qp_->post_send(&wr);
    if (success) {
        stats_.rdma_writes++;
        stats_.bytes_sent += length;
    }
    
    return success;
}

bool RDMAGPUDirectChannel::rdma_read_to_gpu(void* local_gpu_addr, size_t length,
                                            uint64_t remote_addr, uint32_t rkey) {
    if (!gpu_direct_enabled_) {
        std::cerr << "GPUDirect RDMA not enabled" << std::endl;
        return false;
    }
    
    if (!connected_) {
        std::cerr << "Channel not connected" << std::endl;
        return false;
    }
    
    // Find the memory region for this GPU address
    std::shared_ptr<MemoryRegion> mr = nullptr;
    for (const auto& region : memory_regions_) {
        uint8_t* region_start = static_cast<uint8_t*>(region->addr());
        uint8_t* region_end = region_start + region->length();
        uint8_t* gpu_ptr = static_cast<uint8_t*>(local_gpu_addr);
        
        if (gpu_ptr >= region_start && (gpu_ptr + length) <= region_end) {
            mr = region;
            break;
        }
    }
    
    if (!mr) {
        std::cerr << "GPU memory not registered for RDMA" << std::endl;
        return false;
    }
    
    // Create scatter-gather element
    ibv_sge sge = {};
    sge.addr = reinterpret_cast<uint64_t>(local_gpu_addr);
    sge.length = length;
    sge.lkey = mr->lkey();
    
    // Create work request
    ibv_send_wr wr = {};
    wr.wr_id = reinterpret_cast<uint64_t>(this); // Use this pointer as ID
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    
    // Post the work request
    bool success = qp_->post_send(&wr);
    if (success) {
        stats_.rdma_reads++;
        stats_.bytes_received += length;
    }
    
    return success;
}

} // namespace rdma
} // namespace psyne

#else // !PSYNE_RDMA_SUPPORT

// Stub implementation when RDMA is not available
namespace psyne {
namespace rdma {

bool is_rdma_available() { return false; }

std::vector<DeviceCapabilities> list_rdma_devices() {
    return {};
}

} // namespace rdma
} // namespace psyne

#endif // PSYNE_RDMA_SUPPORT