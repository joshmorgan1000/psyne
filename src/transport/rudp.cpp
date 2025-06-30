/**
 * @file rudp.cpp
 * @brief Implementation of Reliable UDP (RUDP) transport
 */

#include <chrono>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

namespace psyne {
namespace transport {

// Implementation for RUDPChannel::Impl
class RUDPChannel::Impl {
public:
    Impl(const std::string &remote_address, uint16_t remote_port,
         const RUDPConfig &config)
        : remote_address_(remote_address), remote_port_(remote_port),
          config_(config), buffer_(1024 * 1024), write_pos_(0), read_pos_(0) {
        state_ = RUDPConnectionState::ESTABLISHED;
    }

    uint32_t reserve_write_slot(size_t size) noexcept {
        // Simple linear buffer for demo
        if (write_pos_ + size > buffer_.size()) {
            return 0xFFFFFFFF; // BUFFER_FULL
        }
        uint32_t offset = write_pos_;
        write_pos_ += size;
        return offset;
    }

    void notify_message_ready(uint32_t offset, size_t size) noexcept {
        // No-op in simple implementation
        (void)offset;
        (void)size;
    }

    RingBuffer &get_ring_buffer() noexcept {
        // Return a dummy reference - this is a hack for the demo
        static SPSCRingBuffer dummy_buffer(1024);
        return dummy_buffer;
    }

    const RingBuffer &get_ring_buffer() const noexcept {
        // Return a dummy reference - this is a hack for the demo
        static SPSCRingBuffer dummy_buffer(1024);
        return dummy_buffer;
    }

    void advance_read_pointer(size_t size) noexcept {
        read_pos_ += size;
    }

    void close() {
        state_ = RUDPConnectionState::CLOSED;
    }

    bool is_open() const {
        return state_ == RUDPConnectionState::ESTABLISHED;
    }

    std::span<uint8_t> get_write_span(size_t size) noexcept {
        auto offset = reserve_write_slot(size);
        if (offset == RingBuffer::BUFFER_FULL) {
            return std::span<uint8_t>{};
        }
        return std::span<uint8_t>(ring_buffer_->base_ptr() + offset, size);
    }

private:
    std::string remote_address_;
    uint16_t remote_port_;
    RUDPConfig config_;
    RUDPConnectionState state_ = RUDPConnectionState::CLOSED;
    std::unique_ptr<RingBuffer> ring_buffer_;
    RUDPStats stats_;
};

// Implementation for RUDPServer::Impl
class RUDPServer::Impl {
public:
    Impl(uint16_t port, const RUDPConfig &config)
        : port_(port), config_(config) {}

    std::unique_ptr<RUDPChannel> accept() {
        // Simplified implementation - return a new channel
        return std::make_unique<RUDPChannel>("127.0.0.1", port_, config_);
    }

    void close() {
        running_ = false;
    }

    bool is_running() const {
        return running_;
    }

private:
    uint16_t port_;
    RUDPConfig config_;
    bool running_ = true;
};

// RUDPChannel implementation
RUDPChannel::RUDPChannel(const std::string &remote_address,
                         uint16_t remote_port, const RUDPConfig &config)
    : pimpl_(std::make_unique<Impl>(remote_address, remote_port, config)) {}

uint32_t RUDPChannel::reserve_write_slot(size_t size) noexcept {
    return pimpl_->reserve_write_slot(size);
}

void RUDPChannel::notify_message_ready(uint32_t offset, size_t size) noexcept {
    pimpl_->notify_message_ready(offset, size);
}

RingBuffer &RUDPChannel::get_ring_buffer() noexcept {
    return pimpl_->get_ring_buffer();
}

const RingBuffer &RUDPChannel::get_ring_buffer() const noexcept {
    return pimpl_->get_ring_buffer();
}

void RUDPChannel::advance_read_pointer(size_t size) noexcept {
    pimpl_->advance_read_pointer(size);
}

void RUDPChannel::close() {
    pimpl_->close();
}

bool RUDPChannel::is_open() const {
    return pimpl_->is_open();
}

std::span<uint8_t> RUDPChannel::get_write_span(size_t size) noexcept {
    return pimpl_->get_write_span(size);
}

// RUDPServer implementation
RUDPServer::RUDPServer(uint16_t port, const RUDPConfig &config)
    : pimpl_(std::make_unique<Impl>(port, config)) {}

std::unique_ptr<RUDPChannel> RUDPServer::accept() {
    return pimpl_->accept();
}

void RUDPServer::close() {
    pimpl_->close();
}

bool RUDPServer::is_running() const {
    return pimpl_->is_running();
}

// Factory functions
std::unique_ptr<RUDPChannel>
create_rudp_client(const std::string &remote_address, uint16_t remote_port,
                   const RUDPConfig &config) {
    return std::make_unique<RUDPChannel>(remote_address, remote_port, config);
}

std::unique_ptr<RUDPServer> create_rudp_server(uint16_t port,
                                               const RUDPConfig &config) {
    return std::make_unique<RUDPServer>(port, config);
}

} // namespace transport
} // namespace psyne