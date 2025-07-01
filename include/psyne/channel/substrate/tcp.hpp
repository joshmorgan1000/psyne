#pragma once

#include "substrate_base.hpp"
#include "global/logger.hpp"
#include <boost/asio.hpp>
#include <thread>
#include <mutex>
#include <memory>

namespace psyne::substrate {

/**
 * @brief TCP network substrate with boost::asio
 */
template<typename T>
class TCP : public SubstrateBase<T> {
private:
    static inline boost::asio::io_context io_context_;
    static inline std::unique_ptr<boost::asio::ip::tcp::acceptor> acceptor_;
    static inline std::unique_ptr<boost::asio::ip::tcp::socket> socket_;
    static inline std::thread io_thread_;
    static inline bool initialized_ = false;
    static inline std::mutex init_mutex_;
    
public:
    T* allocate_slab(size_t size_bytes) override {
        return static_cast<T*>(std::aligned_alloc(alignof(T), size_bytes));
    }
    
    void deallocate_slab(T* ptr) override {
        std::free(ptr);
    }
    
    void send_message(T* msg_ptr, std::vector<std::function<void(T*)>>& listeners) override {
        // Initialize TCP if needed
        if (!initialized_) {
            std::lock_guard<std::mutex> lock(init_mutex_);
            if (!initialized_) {
                initialize_tcp();
            }
        }
        
        if (socket_ && socket_->is_open()) {
            try {
                // Synchronous send for sync API
                boost::asio::write(*socket_,
                    boost::asio::buffer(msg_ptr, sizeof(T)));
                LOG_DEBUG("TCP sent {} bytes", sizeof(T));
            } catch (const std::exception& e) {
                LOG_ERROR("TCP send exception: {}", e.what());
            }
        }
        
        // Also notify local listeners
        for (auto& listener : listeners) {
            listener(msg_ptr);
        }
    }
    
    boost::asio::awaitable<void> async_send_message(T* msg_ptr, 
                                                   std::vector<std::function<void(T*)>>& listeners) override {
        // Initialize TCP if needed
        if (!initialized_) {
            std::lock_guard<std::mutex> lock(init_mutex_);
            if (!initialized_) {
                initialize_tcp();
            }
        }
        
        if (socket_ && socket_->is_open()) {
            try {
                // Async send
                co_await boost::asio::async_write(*socket_,
                    boost::asio::buffer(msg_ptr, sizeof(T)),
                    boost::asio::use_awaitable);
                LOG_DEBUG("TCP async sent {} bytes", sizeof(T));
            } catch (const std::exception& e) {
                LOG_ERROR("TCP async send exception: {}", e.what());
            }
        }
        
        // Also notify local listeners
        for (auto& listener : listeners) {
            listener(msg_ptr);
        }
        
        co_return;
    }
    
    void initialize() override {
        initialize_tcp();
    }
    
    void shutdown() override {
        if (initialized_) {
            io_context_.stop();
            if (io_thread_.joinable()) {
                io_thread_.join();
            }
            socket_.reset();
            acceptor_.reset();
            initialized_ = false;
        }
    }
    
    bool needs_serialization() const override { return true; }
    bool is_zero_copy() const override { return false; }
    bool is_cross_process() const override { return true; }
    const char* name() const override { return "TCP"; }
    
private:
    void initialize_tcp(const std::string& host = "localhost", uint16_t port = 8080) {
        try {
            acceptor_ = std::make_unique<boost::asio::ip::tcp::acceptor>(
                io_context_, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
            
            start_accept();
            
            io_thread_ = std::thread([]() {
                io_context_.run();
            });
            
            initialized_ = true;
            LOG_INFO("TCP substrate initialized on port {}", port);
            
        } catch (const std::exception& e) {
            LOG_ERROR("TCP initialization failed: {}", e.what());
        }
    }
    
    void start_accept() {
        socket_ = std::make_unique<boost::asio::ip::tcp::socket>(io_context_);
        
        acceptor_->async_accept(*socket_,
            [this](boost::system::error_code ec) {
                if (!ec) {
                    LOG_INFO("TCP client connected");
                    start_receive();
                    start_accept(); // Accept next connection
                } else {
                    LOG_ERROR("TCP accept error: {}", ec.message());
                }
            });
    }
    
    void start_receive() {
        auto buffer = std::make_shared<std::array<char, 4096>>();
        
        socket_->async_read_some(boost::asio::buffer(*buffer),
            [this, buffer](boost::system::error_code ec, std::size_t bytes_transferred) {
                if (!ec) {
                    LOG_DEBUG("TCP received {} bytes", bytes_transferred);
                    // TODO: Deserialize and process received data
                    start_receive(); // Continue receiving
                } else {
                    LOG_ERROR("TCP receive error: {}", ec.message());
                }
            });
    }
};

} // namespace psyne::substrate