/**
 * @file tcp_simple.hpp
 * @brief Simple TCP substrate for v2.0 architecture
 *
 * Pure concept-based TCP substrate that implements the SubstrateBehavior
 * interface for network communication over TCP sockets.
 */

#pragma once

#include "../../core/behaviors.hpp"
#include <atomic>
#include <boost/asio.hpp>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace psyne::substrate {

/**
 * @brief Simple TCP substrate for cross-network messaging
 */
class SimpleTCP : public psyne::behaviors::SubstrateBehavior {
public:
    /**
     * @brief Create TCP substrate
     * @param host Remote host (for client) or bind address (for server)
     * @param port Port number
     * @param is_server If true, acts as server. If false, acts as client.
     */
    explicit SimpleTCP(const std::string &host = "localhost",
                       uint16_t port = 8080, bool is_server = false)
        : host_(host), port_(port), is_server_(is_server) {
        initialize_tcp();
    }

    ~SimpleTCP() {
        shutdown();
    }

    /**
     * @brief Allocate memory slab (local memory for TCP)
     */
    void *allocate_memory_slab(size_t size_bytes) override {
        void *ptr = std::aligned_alloc(64, size_bytes);
        if (!ptr) {
            throw std::bad_alloc();
        }
        slab_size_ = size_bytes;
        return ptr;
    }

    /**
     * @brief Deallocate memory slab
     */
    void deallocate_memory_slab(void *memory) override {
        if (memory) {
            std::free(memory);
        }
    }

    /**
     * @brief Send data over TCP
     */
    void transport_send(void *data, size_t size) override {
        if (!is_connected()) {
            throw std::runtime_error("TCP: Not connected");
        }

        try {
            std::lock_guard<std::mutex> lock(socket_mutex_);
            if (socket_ && socket_->is_open()) {
                // Send size first, then data
                uint32_t size_header = static_cast<uint32_t>(size);
                boost::asio::write(
                    *socket_,
                    boost::asio::buffer(&size_header, sizeof(size_header)));
                boost::asio::write(*socket_, boost::asio::buffer(data, size));

                bytes_sent_ += size + sizeof(size_header);
                packets_sent_++;
            }
        } catch (const std::exception &e) {
            connected_.store(false);
            throw std::runtime_error("TCP send failed: " +
                                     std::string(e.what()));
        }
    }

    /**
     * @brief Receive data over TCP
     */
    void transport_receive(void *buffer, size_t buffer_size) override {
        if (!is_connected()) {
            throw std::runtime_error("TCP: Not connected");
        }

        try {
            std::lock_guard<std::mutex> lock(socket_mutex_);
            if (socket_ && socket_->is_open()) {
                // Read size header with validation
                uint32_t size_header = 0;
                boost::asio::read(
                    *socket_,
                    boost::asio::buffer(&size_header, sizeof(size_header)));

                // Validate size header to prevent malicious input
                if (size_header == 0) {
                    throw std::runtime_error("TCP: Received empty message");
                }

                if (size_header > buffer_size) {
                    // Don't attempt to read data that won't fit
                    // Disconnect to prevent protocol desync
                    connected_.store(false);
                    socket_->close();
                    throw std::runtime_error(
                        "TCP: Received message too large (" +
                        std::to_string(size_header) + " > " +
                        std::to_string(buffer_size) + ")");
                }

                // Sanity check for extremely large sizes (potential attack)
                constexpr size_t MAX_REASONABLE_SIZE =
                    100 * 1024 * 1024; // 100MB
                if (size_header > MAX_REASONABLE_SIZE) {
                    connected_.store(false);
                    socket_->close();
                    throw std::runtime_error(
                        "TCP: Suspicious message size detected");
                }

                // Read data
                boost::asio::read(*socket_,
                                  boost::asio::buffer(buffer, size_header));

                bytes_received_ += size_header + sizeof(size_header);
                packets_received_++;
            }
        } catch (const std::exception &e) {
            connected_.store(false);
            throw std::runtime_error("TCP receive failed: " +
                                     std::string(e.what()));
        }
    }

    /**
     * @brief Try to receive without blocking
     */
    bool try_transport_receive(void *buffer, size_t buffer_size,
                               size_t &received_size) {
        if (!is_connected()) {
            return false;
        }

        try {
            std::lock_guard<std::mutex> lock(socket_mutex_);
            if (socket_ && socket_->is_open()) {
                // Check if data is available
                boost::system::error_code ec;
                size_t available = socket_->available(ec);
                if (ec || available < sizeof(uint32_t)) {
                    return false;
                }

                // Read size header
                uint32_t size_header;
                boost::asio::read(
                    *socket_,
                    boost::asio::buffer(&size_header, sizeof(size_header)));

                if (size_header > buffer_size) {
                    throw std::runtime_error(
                        "TCP: Received message too large for buffer");
                }

                // Read data
                boost::asio::read(*socket_,
                                  boost::asio::buffer(buffer, size_header));

                received_size = size_header;
                bytes_received_ += size_header + sizeof(size_header);
                packets_received_++;
                return true;
            }
        } catch (const std::exception &) {
            connected_.store(false);
        }

        return false;
    }

    /**
     * @brief Substrate identity
     */
    const char *substrate_name() const override {
        return "SimpleTCP";
    }
    bool is_zero_copy() const override {
        return false;
    } // Network involves copying
    bool is_cross_process() const override {
        return true;
    } // Network is cross-process

    /**
     * @brief Connection status
     */
    bool is_connected() const {
        return connected_.load();
    }

    /**
     * @brief Wait for connection (useful for testing)
     */
    bool wait_for_connection(
        std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
        auto start = std::chrono::steady_clock::now();
        while (!is_connected() &&
               (std::chrono::steady_clock::now() - start) < timeout) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return is_connected();
    }

    /**
     * @brief Get statistics
     */
    size_t get_bytes_sent() const {
        return bytes_sent_;
    }
    size_t get_bytes_received() const {
        return bytes_received_;
    }
    size_t get_packets_sent() const {
        return packets_sent_;
    }
    size_t get_packets_received() const {
        return packets_received_;
    }

    /**
     * @brief Get connection info
     */
    const std::string &get_host() const {
        return host_;
    }
    uint16_t get_port() const {
        return port_;
    }
    bool is_server_mode() const {
        return is_server_;
    }

private:
    void initialize_tcp() {
        try {
            io_context_ = std::make_unique<boost::asio::io_context>();

            if (is_server_) {
                start_server();
            } else {
                start_client();
            }

            // Start IO thread
            io_thread_ = std::thread([this]() { io_context_->run(); });

        } catch (const std::exception &e) {
            throw std::runtime_error("TCP initialization failed: " +
                                     std::string(e.what()));
        }
    }

    void start_server() {
        acceptor_ = std::make_unique<boost::asio::ip::tcp::acceptor>(
            *io_context_,
            boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port_));

        start_accept();
    }

    void start_client() {
        socket_ = std::make_unique<boost::asio::ip::tcp::socket>(*io_context_);
        start_connect();
    }

    void start_accept() {
        socket_ = std::make_unique<boost::asio::ip::tcp::socket>(*io_context_);

        acceptor_->async_accept(*socket_, [this](boost::system::error_code ec) {
            if (!ec) {
                connected_.store(true);
                // Don't accept more connections for simplicity
            } else {
                // Try accepting again
                start_accept();
            }
        });
    }

    void start_connect() {
        boost::asio::ip::tcp::resolver resolver(*io_context_);
        auto endpoints = resolver.resolve(host_, std::to_string(port_));

        boost::asio::async_connect(*socket_, endpoints,
                                   [this](boost::system::error_code ec,
                                          boost::asio::ip::tcp::endpoint) {
                                       if (!ec) {
                                           connected_.store(true);
                                       } else {
                                           // Retry connection
                                           std::this_thread::sleep_for(
                                               std::chrono::milliseconds(100));
                                           start_connect();
                                       }
                                   });
    }

    void shutdown() {
        connected_.store(false);

        if (io_context_) {
            io_context_->stop();
        }

        if (io_thread_.joinable()) {
            io_thread_.join();
        }

        std::lock_guard<std::mutex> lock(socket_mutex_);
        socket_.reset();
        acceptor_.reset();
        io_context_.reset();
    }

private:
    std::string host_;
    uint16_t port_;
    bool is_server_;
    size_t slab_size_ = 0;

    // Network components
    std::unique_ptr<boost::asio::io_context> io_context_;
    std::unique_ptr<boost::asio::ip::tcp::socket> socket_;
    std::unique_ptr<boost::asio::ip::tcp::acceptor> acceptor_;
    std::thread io_thread_;

    // Connection state
    std::atomic<bool> connected_{false};
    mutable std::mutex socket_mutex_;

    // Statistics
    std::atomic<size_t> bytes_sent_{0};
    std::atomic<size_t> bytes_received_{0};
    std::atomic<size_t> packets_sent_{0};
    std::atomic<size_t> packets_received_{0};
};

} // namespace psyne::substrate