# TCP Socket Implementation Guide

## Overview

This guide provides a complete implementation for TCP client/server communication with both synchronous and asynchronous patterns. TCP provides reliable, ordered, error-checked delivery of data between applications.

## Key Concepts

1. **Server**: Listens on a port, accepts incoming connections
2. **Client**: Connects to a server at a specific address and port
3. **Socket**: Endpoint for communication
4. **Blocking vs Non-blocking**: Synchronous vs asynchronous I/O

## Complete Implementation

### 1. Cross-Platform Socket Wrapper

```cpp
#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <memory>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET socket_t;
    typedef int socklen_t;
    #define INVALID_SOCKET_VALUE INVALID_SOCKET
    #define SOCKET_ERROR_VALUE SOCKET_ERROR
    #define CLOSE_SOCKET closesocket
    #define SOCKET_ERRNO WSAGetLastError()
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>
    typedef int socket_t;
    #define INVALID_SOCKET_VALUE -1
    #define SOCKET_ERROR_VALUE -1
    #define CLOSE_SOCKET close
    #define SOCKET_ERRNO errno
#endif

class SocketInitializer {
public:
    SocketInitializer() {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
#endif
    }
    
    ~SocketInitializer() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
    
    // Singleton
    static void ensure_initialized() {
        static SocketInitializer instance;
    }
};

class Socket {
protected:
    socket_t sock_;
    bool is_blocking_;
    
public:
    Socket() : sock_(INVALID_SOCKET_VALUE), is_blocking_(true) {
        SocketInitializer::ensure_initialized();
    }
    
    explicit Socket(socket_t sock) : sock_(sock), is_blocking_(true) {
        SocketInitializer::ensure_initialized();
    }
    
    virtual ~Socket() {
        close();
    }
    
    // Move operations
    Socket(Socket&& other) noexcept : sock_(other.sock_), is_blocking_(other.is_blocking_) {
        other.sock_ = INVALID_SOCKET_VALUE;
    }
    
    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) {
            close();
            sock_ = other.sock_;
            is_blocking_ = other.is_blocking_;
            other.sock_ = INVALID_SOCKET_VALUE;
        }
        return *this;
    }
    
    // No copy
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    
    void close() {
        if (sock_ != INVALID_SOCKET_VALUE) {
            CLOSE_SOCKET(sock_);
            sock_ = INVALID_SOCKET_VALUE;
        }
    }
    
    bool is_valid() const {
        return sock_ != INVALID_SOCKET_VALUE;
    }
    
    void set_blocking(bool blocking) {
#ifdef _WIN32
        unsigned long mode = blocking ? 0 : 1;
        ioctlsocket(sock_, FIONBIO, &mode);
#else
        int flags = fcntl(sock_, F_GETFL, 0);
        if (blocking) {
            fcntl(sock_, F_SETFL, flags & ~O_NONBLOCK);
        } else {
            fcntl(sock_, F_SETFL, flags | O_NONBLOCK);
        }
#endif
        is_blocking_ = blocking;
    }
    
    void set_nodelay(bool nodelay) {
        int flag = nodelay ? 1 : 0;
        setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, 
                   reinterpret_cast<const char*>(&flag), sizeof(flag));
    }
    
    void set_reuse_addr(bool reuse) {
        int flag = reuse ? 1 : 0;
        setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&flag), sizeof(flag));
    }
    
    void set_recv_timeout(int milliseconds) {
#ifdef _WIN32
        DWORD timeout = milliseconds;
#else
        struct timeval timeout;
        timeout.tv_sec = milliseconds / 1000;
        timeout.tv_usec = (milliseconds % 1000) * 1000;
#endif
        setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO,
                   reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    }
    
    void set_send_timeout(int milliseconds) {
#ifdef _WIN32
        DWORD timeout = milliseconds;
#else
        struct timeval timeout;
        timeout.tv_sec = milliseconds / 1000;
        timeout.tv_usec = (milliseconds % 1000) * 1000;
#endif
        setsockopt(sock_, SOL_SOCKET, SO_SNDTIMEO,
                   reinterpret_cast<const char*>(&timeout), sizeof(timeout));
    }
    
    // Send data
    ssize_t send(const void* data, size_t size) {
        return ::send(sock_, static_cast<const char*>(data), 
                      static_cast<int>(size), 0);
    }
    
    ssize_t send(const std::vector<uint8_t>& data) {
        return send(data.data(), data.size());
    }
    
    ssize_t send(const std::string& data) {
        return send(data.data(), data.size());
    }
    
    // Send all data (blocking)
    bool send_all(const void* data, size_t size) {
        const char* ptr = static_cast<const char*>(data);
        size_t remaining = size;
        
        while (remaining > 0) {
            ssize_t sent = send(ptr, remaining);
            if (sent <= 0) {
                return false;
            }
            ptr += sent;
            remaining -= sent;
        }
        return true;
    }
    
    // Receive data
    ssize_t recv(void* buffer, size_t size) {
        return ::recv(sock_, static_cast<char*>(buffer), 
                      static_cast<int>(size), 0);
    }
    
    std::vector<uint8_t> recv(size_t max_size) {
        std::vector<uint8_t> buffer(max_size);
        ssize_t received = recv(buffer.data(), max_size);
        if (received > 0) {
            buffer.resize(received);
        } else {
            buffer.clear();
        }
        return buffer;
    }
    
    // Receive exact amount (blocking)
    bool recv_all(void* buffer, size_t size) {
        char* ptr = static_cast<char*>(buffer);
        size_t remaining = size;
        
        while (remaining > 0) {
            ssize_t received = recv(ptr, remaining);
            if (received <= 0) {
                return false;
            }
            ptr += received;
            remaining -= received;
        }
        return true;
    }
    
    socket_t native_handle() const { return sock_; }
};
```

### 2. TCP Server Implementation

```cpp
class TCPServer : public Socket {
private:
    uint16_t port_;
    int backlog_;
    
public:
    explicit TCPServer(uint16_t port, int backlog = 10) 
        : port_(port), backlog_(backlog) {
        
        // Create socket
        sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock_ == INVALID_SOCKET_VALUE) {
            throw std::runtime_error("Failed to create socket");
        }
        
        // Enable address reuse
        set_reuse_addr(true);
        
        // Bind to port
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;
        
        if (bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR_VALUE) {
            close();
            throw std::runtime_error("Failed to bind to port " + std::to_string(port));
        }
        
        // Start listening
        if (listen(sock_, backlog) == SOCKET_ERROR_VALUE) {
            close();
            throw std::runtime_error("Failed to listen on socket");
        }
    }
    
    // Accept a client connection
    std::unique_ptr<Socket> accept() {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        
        socket_t client_sock = ::accept(sock_, 
            reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        
        if (client_sock == INVALID_SOCKET_VALUE) {
            if (!is_blocking_ && (SOCKET_ERRNO == EWOULDBLOCK || SOCKET_ERRNO == EAGAIN)) {
                return nullptr; // No connection available
            }
            throw std::runtime_error("Failed to accept connection");
        }
        
        return std::make_unique<Socket>(client_sock);
    }
    
    // Get client address from accepted socket
    static std::string get_peer_address(const Socket& client) {
        sockaddr_in addr{};
        socklen_t len = sizeof(addr);
        getpeername(client.native_handle(), 
                    reinterpret_cast<sockaddr*>(&addr), &len);
        
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
        return std::string(ip) + ":" + std::to_string(ntohs(addr.sin_port));
    }
    
    uint16_t port() const { return port_; }
};
```

### 3. TCP Client Implementation

```cpp
class TCPClient : public Socket {
private:
    std::string host_;
    uint16_t port_;
    
public:
    TCPClient() : host_(""), port_(0) {}
    
    bool connect(const std::string& host, uint16_t port, int timeout_ms = 5000) {
        host_ = host;
        port_ = port;
        
        // Create socket
        sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock_ == INVALID_SOCKET_VALUE) {
            return false;
        }
        
        // Set non-blocking for timeout support
        set_blocking(false);
        
        // Resolve host
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
            // Try to resolve hostname
            struct addrinfo hints{}, *result;
            hints.ai_family = AF_INET;
            hints.ai_socktype = SOCK_STREAM;
            
            if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), 
                            &hints, &result) != 0) {
                close();
                return false;
            }
            
            addr = *reinterpret_cast<sockaddr_in*>(result->ai_addr);
            freeaddrinfo(result);
        }
        
        // Connect
        int ret = ::connect(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        
        if (ret == SOCKET_ERROR_VALUE) {
            if (SOCKET_ERRNO != EINPROGRESS && SOCKET_ERRNO != EWOULDBLOCK) {
                close();
                return false;
            }
            
            // Wait for connection with timeout
            fd_set write_fds;
            FD_ZERO(&write_fds);
            FD_SET(sock_, &write_fds);
            
            timeval tv;
            tv.tv_sec = timeout_ms / 1000;
            tv.tv_usec = (timeout_ms % 1000) * 1000;
            
            ret = select(sock_ + 1, nullptr, &write_fds, nullptr, &tv);
            
            if (ret <= 0) {
                close();
                return false;
            }
            
            // Check if connected successfully
            int error;
            socklen_t len = sizeof(error);
            if (getsockopt(sock_, SOL_SOCKET, SO_ERROR, 
                           reinterpret_cast<char*>(&error), &len) != 0 || error != 0) {
                close();
                return false;
            }
        }
        
        // Set back to blocking mode
        set_blocking(true);
        
        return true;
    }
    
    bool is_connected() const {
        if (!is_valid()) return false;
        
        // Try to peek at the socket
        char buffer;
        int result = ::recv(sock_, &buffer, 1, MSG_PEEK);
        
        if (result == 0) {
            return false; // Connection closed
        } else if (result < 0) {
            if (SOCKET_ERRNO == EWOULDBLOCK || SOCKET_ERRNO == EAGAIN) {
                return true; // No data available, but connected
            }
            return false; // Error
        }
        return true; // Data available
    }
    
    const std::string& host() const { return host_; }
    uint16_t port() const { return port_; }
};
```

### 4. Message Framing

```cpp
// Protocol for sending/receiving complete messages
class MessageProtocol {
public:
    // Send a message with 4-byte length header
    static bool send_message(Socket& sock, const std::vector<uint8_t>& data) {
        uint32_t size = htonl(static_cast<uint32_t>(data.size()));
        
        // Send size
        if (!sock.send_all(&size, sizeof(size))) {
            return false;
        }
        
        // Send data
        return sock.send_all(data.data(), data.size());
    }
    
    static bool send_message(Socket& sock, const std::string& data) {
        std::vector<uint8_t> vec(data.begin(), data.end());
        return send_message(sock, vec);
    }
    
    // Receive a message with 4-byte length header
    static std::vector<uint8_t> recv_message(Socket& sock) {
        uint32_t size;
        
        // Receive size
        if (!sock.recv_all(&size, sizeof(size))) {
            return {};
        }
        
        size = ntohl(size);
        
        // Sanity check
        if (size > 100 * 1024 * 1024) { // 100MB max
            return {};
        }
        
        // Receive data
        std::vector<uint8_t> data(size);
        if (!sock.recv_all(data.data(), size)) {
            return {};
        }
        
        return data;
    }
    
    static std::string recv_string(Socket& sock) {
        auto data = recv_message(sock);
        return std::string(data.begin(), data.end());
    }
};
```

### 5. Asynchronous I/O with select()

```cpp
class AsyncTCPServer {
private:
    TCPServer server_;
    std::vector<std::unique_ptr<Socket>> clients_;
    std::function<void(Socket&, const std::vector<uint8_t>&)> message_handler_;
    
public:
    AsyncTCPServer(uint16_t port) : server_(port) {
        server_.set_blocking(false);
    }
    
    void set_message_handler(std::function<void(Socket&, const std::vector<uint8_t>&)> handler) {
        message_handler_ = handler;
    }
    
    void run() {
        while (true) {
            fd_set read_fds;
            FD_ZERO(&read_fds);
            
            // Add server socket
            FD_SET(server_.native_handle(), &read_fds);
            socket_t max_fd = server_.native_handle();
            
            // Add client sockets
            for (auto& client : clients_) {
                FD_SET(client->native_handle(), &read_fds);
                max_fd = std::max(max_fd, client->native_handle());
            }
            
            // Wait for activity
            timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 100000; // 100ms
            
            int activity = select(max_fd + 1, &read_fds, nullptr, nullptr, &tv);
            
            if (activity < 0) {
                break; // Error
            }
            
            // Check for new connections
            if (FD_ISSET(server_.native_handle(), &read_fds)) {
                auto client = server_.accept();
                if (client) {
                    client->set_blocking(false);
                    clients_.push_back(std::move(client));
                }
            }
            
            // Check clients for data
            for (auto it = clients_.begin(); it != clients_.end();) {
                if (FD_ISSET((*it)->native_handle(), &read_fds)) {
                    auto data = MessageProtocol::recv_message(**it);
                    
                    if (data.empty()) {
                        // Client disconnected
                        it = clients_.erase(it);
                        continue;
                    }
                    
                    // Handle message
                    if (message_handler_) {
                        message_handler_(**it, data);
                    }
                }
                ++it;
            }
        }
    }
    
    void broadcast(const std::vector<uint8_t>& data) {
        for (auto it = clients_.begin(); it != clients_.end();) {
            if (!MessageProtocol::send_message(**it, data)) {
                it = clients_.erase(it);
            } else {
                ++it;
            }
        }
    }
};
```

### 6. Thread Pool Server

```cpp
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

class ThreadPoolTCPServer {
private:
    TCPServer server_;
    std::vector<std::thread> workers_;
    std::queue<std::unique_ptr<Socket>> client_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool running_;
    std::function<void(Socket&)> client_handler_;
    
    void worker_thread() {
        while (running_) {
            std::unique_ptr<Socket> client;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { 
                    return !client_queue_.empty() || !running_; 
                });
                
                if (!running_) break;
                
                client = std::move(client_queue_.front());
                client_queue_.pop();
            }
            
            // Handle client
            if (client_handler_) {
                client_handler_(*client);
            }
        }
    }
    
public:
    ThreadPoolTCPServer(uint16_t port, size_t num_threads = 4) 
        : server_(port), running_(true) {
        
        // Start worker threads
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back(&ThreadPoolTCPServer::worker_thread, this);
        }
    }
    
    ~ThreadPoolTCPServer() {
        stop();
    }
    
    void set_client_handler(std::function<void(Socket&)> handler) {
        client_handler_ = handler;
    }
    
    void run() {
        while (running_) {
            auto client = server_.accept();
            if (client) {
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    client_queue_.push(std::move(client));
                }
                queue_cv_.notify_one();
            }
        }
    }
    
    void stop() {
        running_ = false;
        queue_cv_.notify_all();
        
        for (auto& thread : workers_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
};
```

### 7. Usage Examples

```cpp
// Example 1: Simple echo server
void echo_server() {
    TCPServer server(8080);
    std::cout << "Echo server listening on port 8080" << std::endl;
    
    while (true) {
        auto client = server.accept();
        std::cout << "Client connected: " 
                  << TCPServer::get_peer_address(*client) << std::endl;
        
        // Echo until client disconnects
        std::vector<uint8_t> buffer(1024);
        while (true) {
            ssize_t received = client->recv(buffer.data(), buffer.size());
            if (received <= 0) break;
            
            client->send_all(buffer.data(), received);
        }
        
        std::cout << "Client disconnected" << std::endl;
    }
}

// Example 2: Simple client
void echo_client() {
    TCPClient client;
    
    if (!client.connect("localhost", 8080)) {
        std::cerr << "Failed to connect" << std::endl;
        return;
    }
    
    std::string message = "Hello, server!";
    client.send(message);
    
    auto response = client.recv(1024);
    std::cout << "Received: " 
              << std::string(response.begin(), response.end()) << std::endl;
}

// Example 3: Message-based protocol
void message_server() {
    ThreadPoolTCPServer server(8080, 4);
    
    server.set_client_handler([](Socket& client) {
        while (true) {
            auto message = MessageProtocol::recv_string(client);
            if (message.empty()) break;
            
            std::cout << "Received: " << message << std::endl;
            
            // Send response
            std::string response = "Echo: " + message;
            MessageProtocol::send_message(client, response);
        }
    });
    
    server.run();
}

// Example 4: Async server with broadcast
void chat_server() {
    AsyncTCPServer server(8080);
    
    server.set_message_handler([&server](Socket& client, const std::vector<uint8_t>& data) {
        // Broadcast message to all clients
        server.broadcast(data);
    });
    
    std::cout << "Chat server running on port 8080" << std::endl;
    server.run();
}

// Example 5: Integration with Psyne
class PsyneTCPChannel {
private:
    TCPClient client_;
    std::thread receiver_thread_;
    std::function<void(const std::vector<uint8_t>&)> message_callback_;
    bool running_;
    
    void receiver_loop() {
        while (running_ && client_.is_connected()) {
            auto data = MessageProtocol::recv_message(client_);
            if (!data.empty() && message_callback_) {
                message_callback_(data);
            }
        }
    }
    
public:
    bool connect(const std::string& host, uint16_t port) {
        if (!client_.connect(host, port)) {
            return false;
        }
        
        running_ = true;
        receiver_thread_ = std::thread(&PsyneTCPChannel::receiver_loop, this);
        return true;
    }
    
    void send_tensor(const float* data, size_t count) {
        std::vector<uint8_t> buffer(count * sizeof(float));
        std::memcpy(buffer.data(), data, buffer.size());
        MessageProtocol::send_message(client_, buffer);
    }
    
    void set_receive_callback(std::function<void(const std::vector<uint8_t>&)> callback) {
        message_callback_ = callback;
    }
    
    ~PsyneTCPChannel() {
        running_ = false;
        if (receiver_thread_.joinable()) {
            receiver_thread_.join();
        }
    }
};
```

## Important Considerations

### 1. Error Handling
- Always check return values
- Handle EINTR for interrupted system calls
- Gracefully handle disconnections

### 2. Buffer Management
- Use message framing for reliable communication
- Don't assume one send() = one recv()
- TCP is a stream protocol, not message-based

### 3. Performance
- Use TCP_NODELAY for low-latency applications
- Consider non-blocking I/O for scalability
- Buffer sizes affect throughput

### 4. Security
- Never trust data from network
- Validate all input
- Consider TLS for encryption

### 5. Platform Differences
- Windows requires WSAStartup
- Socket handles are different types
- Error codes differ

## Common Pitfalls

1. **Not handling partial sends/receives**
2. **Assuming message boundaries**
3. **Not setting SO_REUSEADDR on server**
4. **Blocking in single-threaded servers**
5. **Not handling endianness for binary data**

## Debugging Tips

1. Use Wireshark to inspect traffic
2. `netstat -an` to see connections
3. `telnet` for basic testing
4. Check firewall rules
5. Verify port availability