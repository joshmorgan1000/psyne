/**
 * @file udp_multicast_simple.cpp
 * @brief Simple UDP multicast implementation that delegates to memory channels
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include "udp_multicast_simple.hpp"
#include "../memory/ring_buffer_impl.hpp"
#include <iostream>
#include <thread>
#include <cstring>

#ifdef __linux__
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

namespace psyne {
namespace detail {

SimpleUDPMulticastChannel::SimpleUDPMulticastChannel(const std::string& multicast_address, 
                                                     uint16_t port, 
                                                     Role role,
                                                     size_t buffer_size)
    : multicast_address_(multicast_address), port_(port), role_(role) {
    
    // Create internal memory channel for local delivery
    std::string channel_name = "memory://udp_multicast_" + multicast_address + "_" + std::to_string(port);
    memory_channel_ = Channel::get_or_create<ByteVector>(channel_name, buffer_size);
    
    // Setup UDP multicast socket (basic implementation)
    setup_udp_multicast();
}

SimpleUDPMulticastChannel::~SimpleUDPMulticastChannel() {
    // Stop network thread
    running_ = false;
    if (network_thread_.joinable()) {
        network_thread_.join();
    }
    
    // Close socket
#ifdef _WIN32
    if (socket_ != INVALID_SOCKET) {
        closesocket(socket_);
        WSACleanup();
    }
#else
    if (socket_ >= 0) {
        close(socket_);
    }
#endif
    
    // Memory channel will be cleaned up automatically
}

void SimpleUDPMulticastChannel::setup_udp_multicast() {
#ifdef __linux__
    // Basic UDP multicast setup for Linux
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "Failed to create UDP socket" << std::endl;
        return;
    }
    
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    
    if (role_ == Role::Publisher) {
        // Publisher setup
        addr.sin_addr.s_addr = INADDR_ANY;
        
        // Enable broadcast
        int broadcast_enable = 1;
        setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast_enable, sizeof(broadcast_enable));
        
        std::cout << "UDP Multicast Publisher setup for " << multicast_address_ << ":" << port_ << std::endl;
    } else {
        // Subscriber setup
        addr.sin_addr.s_addr = INADDR_ANY;
        
        if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "Failed to bind UDP socket" << std::endl;
            close(sock);
            return;
        }
        
        // Join multicast group
        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(multicast_address_.c_str());
        mreq.imr_interface.s_addr = INADDR_ANY;
        
        if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
            std::cerr << "Failed to join multicast group" << std::endl;
        } else {
            std::cout << "UDP Multicast Subscriber joined " << multicast_address_ << ":" << port_ << std::endl;
        }
    }
    
    // Store socket for network operations
    socket_ = sock;
    
    // Start network thread
    running_ = true;
    if (role_ == Role::Publisher) {
        network_thread_ = std::thread([this] { network_send_loop(); });
    } else {
        network_thread_ = std::thread([this] { network_receive_loop(); });
    }
    
#elif defined(_WIN32)
    // Basic Windows UDP multicast setup
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
        return;
    }
    
    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock == INVALID_SOCKET) {
        std::cerr << "Failed to create UDP socket" << std::endl;
        WSACleanup();
        return;
    }
    
    // Basic setup similar to Linux
    std::cout << "UDP Multicast setup for Windows: " << multicast_address_ << ":" << port_ << std::endl;
    
    socket_ = sock;
    
    // Start network thread
    running_ = true;
    if (role_ == Role::Publisher) {
        network_thread_ = std::thread([this] { network_send_loop(); });
    } else {
        network_thread_ = std::thread([this] { network_receive_loop(); });
    }
    
#else
    std::cout << "UDP Multicast not implemented for this platform" << std::endl;
#endif
}

std::string SimpleUDPMulticastChannel::uri() const {
    return "udp://" + multicast_address_ + ":" + std::to_string(port_);
}

ChannelMode SimpleUDPMulticastChannel::mode() const {
    return ChannelMode::SPMC; // Single publisher, multiple consumers
}

ChannelType SimpleUDPMulticastChannel::type() const {
    return ChannelType::MultiType;
}

void SimpleUDPMulticastChannel::stop() {
    if (memory_channel_) {
        memory_channel_->stop();
    }
}

bool SimpleUDPMulticastChannel::is_stopped() const {
    return memory_channel_ ? memory_channel_->is_stopped() : true;
}

void* SimpleUDPMulticastChannel::receive_message(size_t& size, uint32_t& type) {
    if (memory_channel_) {
        return memory_channel_->receive_message(size, type);
    }
    return nullptr;
}

void SimpleUDPMulticastChannel::release_message(void* handle) {
    if (memory_channel_) {
        memory_channel_->release_message(handle);
    }
}

bool SimpleUDPMulticastChannel::has_metrics() const {
    return memory_channel_ ? memory_channel_->has_metrics() : false;
}

debug::ChannelMetrics SimpleUDPMulticastChannel::get_metrics() const {
    return memory_channel_ ? memory_channel_->get_metrics() : debug::ChannelMetrics{};
}

void SimpleUDPMulticastChannel::reset_metrics() {
    if (memory_channel_) {
        memory_channel_->reset_metrics();
    }
}

void SimpleUDPMulticastChannel::network_send_loop() {
    if (role_ != Role::Publisher) return;
    
    struct sockaddr_in dest_addr;
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(port_);
    inet_pton(AF_INET, multicast_address_.c_str(), &dest_addr.sin_addr);
    
    while (running_) {
        // Check memory channel for messages to send
        size_t size;
        uint32_t type;
        void* data = memory_channel_->receive_message(size, type);
        
        if (data) {
            // Send over network
            ssize_t sent = sendto(socket_, data, size, 0,
                                 (struct sockaddr*)&dest_addr, sizeof(dest_addr));
            if (sent < 0) {
                std::cerr << "UDP multicast send error" << std::endl;
            }
            memory_channel_->release_message(data);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void SimpleUDPMulticastChannel::network_receive_loop() {
    if (role_ != Role::Subscriber) return;
    
    std::vector<uint8_t> recv_buffer(65536); // Max UDP packet size
    struct sockaddr_in src_addr;
    socklen_t addr_len = sizeof(src_addr);
    
    while (running_) {
        ssize_t received = recvfrom(socket_, recv_buffer.data(), recv_buffer.size(),
                                   0, (struct sockaddr*)&src_addr, &addr_len);
        
        if (received > 0) {
            // Forward to memory channel for local delivery
            auto slot = memory_channel_->reserve_write_slot(received);
            if (slot != BUFFER_FULL) {
                auto span = memory_channel_->get_write_span(received);
                std::memcpy(span.data(), recv_buffer.data(), received);
                memory_channel_->notify_message_ready(slot, received);
            }
        } else if (received < 0) {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                std::cerr << "UDP multicast receive error" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

} // namespace detail
} // namespace psyne