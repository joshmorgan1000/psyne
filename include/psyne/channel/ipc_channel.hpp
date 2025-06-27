#pragma once

#include "channel.hpp"
#include "tcp_protocol.hpp"
#include "../memory/ring_buffer.hpp"
#include <boost/asio.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <memory>
#include <thread>
#include <iostream>
#include <array>
#include <mutex>
#include <condition_variable>
#include <boost/system/error_code.hpp>

namespace psyne {

namespace asio = boost::asio;
namespace ipc = boost::interprocess;

template<typename RingBufferType>
class IPCChannel : public Channel<RingBufferType> {
public:
    IPCChannel(const std::string& name, size_t buffer_size, bool create)
        : Channel<RingBufferType>("ipc://" + name, 0)  // Don't allocate local buffer
        , io_context_()
        , work_guard_(asio::make_work_guard(io_context_))
        , name_(name)
        , buffer_size_(buffer_size)
        , is_creator_(create) {
        
        try {
            if (create) {
                create_shared_memory();
            } else {
                open_shared_memory();
            }
            
            // Start the IO thread
            io_thread_ = std::thread([this]() {
                io_context_.run();
            });
        } catch (...) {
            cleanup();
            throw;
        }
    }
    
    ~IPCChannel() {
        stop();
        cleanup();
    }
    
    // Override to post notification to IO context
    void notify() override {
        asio::post(io_context_, [this]() {
            notify_semaphore_->post();
        });
    }
    
    // Override to return shared ring buffer
    RingBufferType* ring_buffer() override { 
        return shared_ring_buffer_; 
    }
    
    void stop() {
        work_guard_.reset();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }
    
    // Async wait for data using Boost.Asio timer
    boost::asio::awaitable<bool> async_wait_for_data() {
        // Try non-blocking first
        if (wait_semaphore_ && wait_semaphore_->try_wait()) {
            co_return true;
        }
        
        // Otherwise poll with timer
        asio::steady_timer timer(io_context_);
        while (true) {
            timer.expires_after(std::chrono::milliseconds(10));
            co_await timer.async_wait(asio::use_awaitable);
            
            if (wait_semaphore_ && wait_semaphore_->try_wait()) {
                co_return true;
            }
        }
    }
    
protected:
    // Override to use semaphore waiting with Boost.Asio
    bool wait_for_data(std::chrono::milliseconds timeout) override {
        if (!wait_semaphore_) return false;
        
        try {
            if (timeout == std::chrono::milliseconds::zero()) {
                // Non-blocking
                return wait_semaphore_->try_wait();
            } else {
                // Blocking with timeout
                auto deadline = std::chrono::steady_clock::now() + timeout;
                return wait_semaphore_->timed_wait(deadline);
            }
        } catch (...) {
            return false;
        }
    }
    
private:
    void create_shared_memory() {
        // Remove any existing shared memory and semaphores
        ipc::shared_memory_object::remove(("psyne_shm_" + name_).c_str());
        ipc::named_semaphore::remove(("psyne_sem_wait_" + name_).c_str());
        ipc::named_semaphore::remove(("psyne_sem_notify_" + name_).c_str());
        
        // Create shared memory
        shm_ = std::make_unique<ipc::shared_memory_object>(
            ipc::create_only,
            ("psyne_shm_" + name_).c_str(),
            ipc::read_write
        );
        
        // Set size
        shm_->truncate(buffer_size_);
        
        // Map the shared memory
        region_ = std::make_unique<ipc::mapped_region>(*shm_, ipc::read_write);
        
        // Create semaphores
        wait_semaphore_ = std::make_unique<ipc::named_semaphore>(
            ipc::create_only,
            ("psyne_sem_wait_" + name_).c_str(),
            0
        );
        
        notify_semaphore_ = std::make_unique<ipc::named_semaphore>(
            ipc::create_only,
            ("psyne_sem_notify_" + name_).c_str(),
            0
        );
        
        // Initialize ring buffer in shared memory
        void* addr = region_->get_address();
        shared_ring_buffer_ = new (addr) RingBufferType(buffer_size_ - sizeof(RingBufferType));
    }
    
    void open_shared_memory() {
        // Open existing shared memory
        shm_ = std::make_unique<ipc::shared_memory_object>(
            ipc::open_only,
            ("psyne_shm_" + name_).c_str(),
            ipc::read_write
        );
        
        // Map the shared memory
        region_ = std::make_unique<ipc::mapped_region>(*shm_, ipc::read_write);
        
        // Open semaphores
        wait_semaphore_ = std::make_unique<ipc::named_semaphore>(
            ipc::open_only,
            ("psyne_sem_wait_" + name_).c_str()
        );
        
        notify_semaphore_ = std::make_unique<ipc::named_semaphore>(
            ipc::open_only,
            ("psyne_sem_notify_" + name_).c_str()
        );
        
        // Get existing ring buffer
        void* addr = region_->get_address();
        shared_ring_buffer_ = static_cast<RingBufferType*>(addr);
    }
    
    void cleanup() {
        shared_ring_buffer_ = nullptr;
        
        // Clean up in reverse order
        notify_semaphore_.reset();
        wait_semaphore_.reset();
        region_.reset();
        shm_.reset();
        
        if (is_creator_) {
            // Remove shared resources
            ipc::shared_memory_object::remove(("psyne_shm_" + name_).c_str());
            ipc::named_semaphore::remove(("psyne_sem_wait_" + name_).c_str());
            ipc::named_semaphore::remove(("psyne_sem_notify_" + name_).c_str());
        }
    }
    
    // Boost.Asio context
    asio::io_context io_context_;
    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
    std::thread io_thread_;
    
    // IPC resources
    std::string name_;
    size_t buffer_size_;
    bool is_creator_;
    
    std::unique_ptr<ipc::shared_memory_object> shm_;
    std::unique_ptr<ipc::mapped_region> region_;
    std::unique_ptr<ipc::named_semaphore> wait_semaphore_;
    std::unique_ptr<ipc::named_semaphore> notify_semaphore_;
    
    RingBufferType* shared_ring_buffer_ = nullptr;
};

// TCP Channel using Boost.Asio
template<typename RingBufferType>
class TCPChannel : public Channel<RingBufferType> {
public:
    TCPChannel(const std::string& host, uint16_t port, size_t buffer_size, bool is_server)
        : Channel<RingBufferType>("tcp://" + host + ":" + std::to_string(port), buffer_size)
        , io_context_()
        , work_guard_(asio::make_work_guard(io_context_))
        , is_server_(is_server)
        , connected_(false) {
        
        if (is_server) {
            start_server(port);
        } else {
            connect_client(host, port);
        }
        
        // Start IO thread
        io_thread_ = std::thread([this]() {
            io_context_.run();
        });
        
        // Start sender thread for zero-copy sending
        sender_thread_ = std::thread([this]() {
            sender_loop();
        });
    }
    
    ~TCPChannel() {
        stop();
    }
    
    void notify() override {
        // Wake up the sender thread
        send_cv_.notify_one();
    }
    
    void stop() {
        connected_.store(false);
        stop_sender_.store(true);
        send_cv_.notify_all();
        
        work_guard_.reset();
        if (acceptor_) acceptor_->close();
        if (socket_) socket_->close();
        
        if (sender_thread_.joinable()) {
            sender_thread_.join();
        }
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }
    
    // Get io_context for coroutine integration
    asio::io_context& get_io_context() { return io_context_; }
    
    // Send data from ring buffer over TCP
    void send_from_buffer(const void* data, size_t size) {
        if (!connected_.load() || !socket_) return;
        
        // Create frame header
        TCPFrameHeader header = TCPFramer::create_header(data, size);
        
        // Use scatter-gather I/O to send header + data
        std::array<asio::const_buffer, 2> buffers = {
            asio::buffer(&header, sizeof(header)),
            asio::buffer(data, size)
        };
        
        asio::async_write(*socket_, buffers,
            [this](boost::system::error_code ec, std::size_t) {
                if (ec) {
                    handle_error(ec);
                }
            });
    }
    
private:
    void sender_loop() {
        auto* rb = this->ring_buffer();
        if (!rb) return;
        
        while (!stop_sender_.load()) {
            // Wait for notification or timeout
            std::unique_lock<std::mutex> lock(send_mutex_);
            send_cv_.wait_for(lock, std::chrono::milliseconds(100), [this, rb]() {
                return stop_sender_.load() || !rb->empty();
            });
            
            // Send all available messages
            while (connected_.load() && socket_) {
                auto handle = rb->read();
                if (!handle) break;
                
                // Send directly without extra copy
                send_from_buffer(handle->data, handle->size);
                // handle commits on destruction
            }
        }
    }
    
    void start_server(uint16_t port) {
        acceptor_ = std::make_unique<asio::ip::tcp::acceptor>(io_context_, 
            asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port));
        
        // Set socket options
        acceptor_->set_option(asio::socket_base::reuse_address(true));
        
        // Accept connections
        accept_connection();
    }
    
    void connect_client(const std::string& host, uint16_t port) {
        socket_ = std::make_unique<asio::ip::tcp::socket>(io_context_);
        
        asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(host, std::to_string(port));
        
        asio::async_connect(*socket_, endpoints,
            [this](boost::system::error_code ec, asio::ip::tcp::endpoint) {
                if (!ec) {
                    on_connected();
                } else {
                    handle_error(ec);
                }
            });
    }
    
    void accept_connection() {
        acceptor_->async_accept(
            [this](boost::system::error_code ec, asio::ip::tcp::socket socket) {
                if (!ec) {
                    socket_ = std::make_unique<asio::ip::tcp::socket>(std::move(socket));
                    on_connected();
                }
                accept_connection(); // Continue accepting
            });
    }
    
    void on_connected() {
        // Set TCP_NODELAY for low latency
        socket_->set_option(asio::ip::tcp::no_delay(true));
        
        // Set socket buffer sizes if needed
        socket_->set_option(asio::socket_base::send_buffer_size(1024 * 1024));
        socket_->set_option(asio::socket_base::receive_buffer_size(1024 * 1024));
        
        connected_.store(true);
        start_read_header();
    }
    
    void start_read_header() {
        if (!connected_.load() || !socket_) return;
        
        // Read frame header
        asio::async_read(*socket_,
            asio::buffer(&read_header_, sizeof(TCPFrameHeader)),
            [this](boost::system::error_code ec, std::size_t) {
                if (!ec) {
                    // Convert from network byte order
                    read_header_.to_host();
                    
                    // Validate frame size
                    if (read_header_.length == 0) {
                        // Notification frame
                        start_read_header();
                    } else if (read_header_.length <= TCPFramer::MAX_FRAME_SIZE) {
                        start_read_body();
                    } else {
                        // Invalid frame size
                        handle_error(asio::error::message_size);
                    }
                } else {
                    handle_error(ec);
                }
            });
    }
    
    void start_read_body() {
        // Allocate space in ring buffer
        auto* rb = this->ring_buffer();
        if (!rb) return;
        
        auto write_handle = rb->reserve(read_header_.length);
        if (!write_handle) {
            // Buffer full, drop message
            // TODO: Add flow control
            read_buffer_.resize(read_header_.length);
            asio::async_read(*socket_,
                asio::buffer(read_buffer_),
                [this](boost::system::error_code ec, std::size_t) {
                    if (!ec) {
                        start_read_header();
                    } else {
                        handle_error(ec);
                    }
                });
            return;
        }
        
        // Read directly into ring buffer
        current_write_handle_ = write_handle;
        asio::async_read(*socket_,
            asio::buffer(write_handle->data, read_header_.length),
            [this](boost::system::error_code ec, std::size_t) {
                if (!ec) {
                    // Verify checksum
                    if (TCPFramer::verify_frame(read_header_, 
                                               current_write_handle_->data, 
                                               read_header_.length)) {
                        // Commit the message
                        current_write_handle_->commit();
                        this->notify();  // Wake up receivers
                    }
                    
                    start_read_header();
                } else {
                    handle_error(ec);
                }
            });
    }
    
    void handle_error(boost::system::error_code ec) {
        connected_.store(false);
        
        // TODO: Add reconnection logic
        if (ec != boost::asio::error::eof && 
            ec != boost::asio::error::connection_reset) {
            std::cerr << "TCP error: " << ec.message() << std::endl;
        }
    }
    
    asio::io_context io_context_;
    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
    std::thread io_thread_;
    std::thread sender_thread_;
    
    bool is_server_;
    std::atomic<bool> connected_;
    std::atomic<bool> stop_sender_{false};
    std::unique_ptr<asio::ip::tcp::acceptor> acceptor_;
    std::unique_ptr<asio::ip::tcp::socket> socket_;
    
    // Sender thread synchronization
    std::mutex send_mutex_;
    std::condition_variable send_cv_;
    
    // Read state
    TCPFrameHeader read_header_;
    std::vector<uint8_t> read_buffer_;  // For dropped messages
    std::optional<typename RingBufferType::WriteHandle> current_write_handle_;
};

}  // namespace psyne