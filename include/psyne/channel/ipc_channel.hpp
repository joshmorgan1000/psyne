#pragma once

#include "channel.hpp"
#include "../memory/ring_buffer.hpp"
#include <boost/asio.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <memory>
#include <thread>

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
        , is_server_(is_server) {
        
        if (is_server) {
            start_server(port);
        } else {
            connect_client(host, port);
        }
        
        // Start IO thread
        io_thread_ = std::thread([this]() {
            io_context_.run();
        });
    }
    
    ~TCPChannel() {
        stop();
    }
    
    void notify() override {
        // For TCP, we would send a notification packet
        // This is a simplified version
    }
    
    void stop() {
        work_guard_.reset();
        if (acceptor_) acceptor_->close();
        if (socket_) socket_->close();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }
    
private:
    void start_server(uint16_t port) {
        acceptor_ = std::make_unique<asio::ip::tcp::acceptor>(io_context_, 
            asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port));
        
        // Accept connections
        accept_connection();
    }
    
    void connect_client(const std::string& host, uint16_t port) {
        socket_ = std::make_unique<asio::ip::tcp::socket>(io_context_);
        
        asio::ip::tcp::resolver resolver(io_context_);
        auto endpoints = resolver.resolve(host, std::to_string(port));
        
        asio::async_connect(*socket_, endpoints,
            [this](std::error_code ec, asio::ip::tcp::endpoint) {
                if (!ec) {
                    start_read();
                }
            });
    }
    
    void accept_connection() {
        acceptor_->async_accept(
            [this](std::error_code ec, asio::ip::tcp::socket socket) {
                if (!ec) {
                    socket_ = std::make_unique<asio::ip::tcp::socket>(std::move(socket));
                    start_read();
                }
                accept_connection(); // Continue accepting
            });
    }
    
    void start_read() {
        // Implement async read for messages
        // This would read from socket and write to ring buffer
    }
    
    asio::io_context io_context_;
    asio::executor_work_guard<asio::io_context::executor_type> work_guard_;
    std::thread io_thread_;
    
    bool is_server_;
    std::unique_ptr<asio::ip::tcp::acceptor> acceptor_;
    std::unique_ptr<asio::ip::tcp::socket> socket_;
};

}  // namespace psyne