/**
 * @file network_saturation_test.cpp
 * @brief Network saturation test for distributed benchmarking
 *
 * Designed to test Psyne TCP channels between two machines,
 * saturating network hardware and measuring real-world performance.
 */

#include "psyne/psyne.hpp"
#include "psyne/memory/object_pool.hpp"
#include "logger.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <thread>
#include <vector>

using namespace psyne;
using namespace std::chrono;

// Buffer pool for efficient memory management
static thread_local std::unique_ptr<BufferPool> message_buffer_pool;

// Initialize buffer pool for this thread
void init_thread_buffer_pool(size_t buffer_size) {
    if (!message_buffer_pool) {
        message_buffer_pool = std::make_unique<BufferPool>(buffer_size, 64); // 64 initial buffers
    }
}

// Get buffer pool for current thread
BufferPool& get_buffer_pool() {
    return *message_buffer_pool;
}

// Test configuration
struct NetworkTestConfig {
    bool is_server;
    std::string remote_host;
    uint16_t port;
    size_t num_connections;      // Number of parallel TCP connections
    size_t message_size;         // Size of each message
    size_t messages_per_conn;    // Messages per connection
    size_t batch_size;           // Messages to send before yielding
    bool bidirectional;          // Test both directions simultaneously
    bool zero_copy;              // Use zero-copy optimizations
    int send_buffer_size;        // TCP send buffer size (0 = default)
    int recv_buffer_size;        // TCP receive buffer size (0 = default)
    bool tcp_nodelay;            // Disable Nagle's algorithm
};

// Per-connection statistics
struct ConnectionStats {
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> messages_received{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> errors{0};
    std::vector<uint64_t> latencies;
    high_resolution_clock::time_point start_time;
    high_resolution_clock::time_point end_time;
    
    ConnectionStats() {
        latencies.reserve(100000); // Pre-allocate for performance
    }
};

// Message structure for network testing
struct NetworkTestMessage {
    uint64_t sequence;
    uint64_t timestamp_ns;
    uint32_t connection_id;
    uint32_t batch_id;
    uint8_t pattern_start;
    uint8_t pattern_inc;
    uint16_t payload_size;
    uint8_t payload[]; // Variable size payload
    
    // Verify payload pattern
    bool verify_pattern() const {
        uint8_t expected = pattern_start;
        for (uint16_t i = 0; i < payload_size; ++i) {
            if (payload[i] != expected) {
                return false;
            }
            expected += pattern_inc;
        }
        return true;
    }
    
    // Fill payload with pattern
    void fill_pattern() {
        uint8_t value = pattern_start;
        for (uint16_t i = 0; i < payload_size; ++i) {
            payload[i] = value;
            value += pattern_inc;
        }
    }
};

// Worker function for server mode
void server_connection_worker(uint32_t conn_id,
                            const NetworkTestConfig& config,
                            ConnectionStats& stats) {
    try {
        // Initialize buffer pool for this thread
        size_t max_message_size = sizeof(NetworkTestMessage) + config.message_size;
        init_thread_buffer_pool(max_message_size);
        // Create TCP channel for this connection
        ChannelConfig chan_config;
        chan_config.name = "network_test_" + std::to_string(conn_id);
        chan_config.size_mb = 32; // 32MB per connection
        chan_config.mode = ChannelMode::SPSC;
        chan_config.transport = ChannelTransport::TCP;
        chan_config.is_server = true;
        chan_config.remote_port = config.port + conn_id;
        chan_config.blocking = true;
        
        // Set TCP options if specified
        if (config.send_buffer_size > 0) {
            chan_config.tcp_send_buffer_size = config.send_buffer_size;
        }
        if (config.recv_buffer_size > 0) {
            chan_config.tcp_recv_buffer_size = config.recv_buffer_size;
        }
        chan_config.tcp_nodelay = config.tcp_nodelay;
        
        auto channel = Channel<NetworkTestMessage>::create(chan_config);
        
        log_info("[Server ", conn_id, "] Waiting for connection on port ",
                  (config.port + conn_id));
        
        // Wait for client connection
        if (auto tcp_chan = std::dynamic_pointer_cast<TCPChannel>(channel)) {
            tcp_chan->wait_for_connection();
        }
        
        log_info("[Server ", conn_id, "] Connected!");
        stats.start_time = high_resolution_clock::now();
        
        // Receive messages
        for (size_t i = 0; i < config.messages_per_conn; ++i) {
            auto msg = channel->receive();
            
            // Verify message integrity
            if (msg->connection_id != conn_id) {
                log_error("[Server ", conn_id, "] Wrong connection ID: expected ",
                          conn_id, ", got ", msg->connection_id);
                stats.errors++;
            }
            
            if (!msg->verify_pattern()) {
                log_error("[Server ", conn_id, "] Payload corruption detected!");
                stats.errors++;
            }
            
            // Calculate latency
            uint64_t now = duration_cast<nanoseconds>(
                high_resolution_clock::now().time_since_epoch()).count();
            uint64_t latency = now - msg->timestamp_ns;
            stats.latencies.push_back(latency);
            
            stats.messages_received++;
            stats.bytes_received += sizeof(NetworkTestMessage) + msg->payload_size;
            
            // Echo back if bidirectional using buffer pool
            if (config.bidirectional) {
                // Use buffer pool for efficient reply allocation
                size_t reply_size = sizeof(NetworkTestMessage) + msg->payload_size;
                auto buffer = get_buffer_pool().acquire();
                
                if (buffer && buffer->capacity >= reply_size) {
                    // Construct reply in pooled buffer
                    auto* reply_msg = reinterpret_cast<NetworkTestMessage*>(buffer->ptr());
                    reply_msg->sequence = msg->sequence;
                    reply_msg->timestamp_ns = now; // Update timestamp
                    reply_msg->connection_id = conn_id;
                    reply_msg->batch_id = msg->batch_id;
                    reply_msg->pattern_start = msg->pattern_start;
                    reply_msg->pattern_inc = msg->pattern_inc;
                    reply_msg->payload_size = msg->payload_size;
                    std::memcpy(reply_msg->payload, msg->payload, msg->payload_size);
                    
                    buffer->used = reply_size;
                    
                    // Allocate channel message and copy from buffer
                    auto reply = channel->allocate(reply_size);
                    std::memcpy(reply.get(), buffer->ptr(), reply_size);
                    
                    reply.send();
                    stats.messages_sent++;
                    stats.bytes_sent += reply_size;
                    
                    // Buffer automatically returns to pool when 'buffer' goes out of scope
                } else {
                    log_error("[Server ", conn_id, "] Failed to get buffer for reply");
                    stats.errors++;
                }
            }
        }
        
        stats.end_time = high_resolution_clock::now();
        
    } catch (const std::exception& e) {
        log_error("[Server ", conn_id, "] Error: ", e.what());
        stats.errors++;
    }
}

// Worker function for client mode
void client_connection_worker(uint32_t conn_id,
                            const NetworkTestConfig& config,
                            ConnectionStats& stats) {
    try {
        // Initialize buffer pool for this thread
        size_t max_message_size = sizeof(NetworkTestMessage) + config.message_size;
        init_thread_buffer_pool(max_message_size);
        // Create TCP channel for this connection
        ChannelConfig chan_config;
        chan_config.name = "network_test_client_" + std::to_string(conn_id);
        chan_config.size_mb = 32; // 32MB per connection
        chan_config.mode = ChannelMode::SPSC;
        chan_config.transport = ChannelTransport::TCP;
        chan_config.is_server = false;
        chan_config.remote_host = config.remote_host;
        chan_config.remote_port = config.port + conn_id;
        chan_config.blocking = true;
        
        // Set TCP options
        if (config.send_buffer_size > 0) {
            chan_config.tcp_send_buffer_size = config.send_buffer_size;
        }
        if (config.recv_buffer_size > 0) {
            chan_config.tcp_recv_buffer_size = config.recv_buffer_size;
        }
        chan_config.tcp_nodelay = config.tcp_nodelay;
        
        auto channel = Channel<NetworkTestMessage>::create(chan_config);
        
        log_info("[Client ", conn_id, "] Connecting to ",
                  config.remote_host, ":", (config.port + conn_id));
        
        // Wait for connection
        if (auto tcp_chan = std::dynamic_pointer_cast<TCPChannel>(channel)) {
            tcp_chan->wait_for_connection();
        }
        
        log_info("[Client ", conn_id, "] Connected!");
        stats.start_time = high_resolution_clock::now();
        
        // Send messages in batches using buffer pool for construction
        size_t total_size = sizeof(NetworkTestMessage) + config.message_size;
        size_t batch_count = 0;
        
        for (size_t i = 0; i < config.messages_per_conn; ++i) {
            // Use buffer pool to construct message efficiently
            auto buffer = get_buffer_pool().acquire();
            
            if (buffer && buffer->capacity >= total_size) {
                // Construct message in pooled buffer
                auto* temp_msg = reinterpret_cast<NetworkTestMessage*>(buffer->ptr());
                temp_msg->sequence = i;
                temp_msg->timestamp_ns = duration_cast<nanoseconds>(
                    high_resolution_clock::now().time_since_epoch()).count();
                temp_msg->connection_id = conn_id;
                temp_msg->batch_id = batch_count;
                temp_msg->pattern_start = (uint8_t)(i & 0xFF);
                temp_msg->pattern_inc = (uint8_t)(conn_id & 0xFF);
                temp_msg->payload_size = config.message_size;
                temp_msg->fill_pattern();
                
                buffer->used = total_size;
                
                // Allocate channel message and copy from buffer
                auto msg = channel->allocate(total_size);
                std::memcpy(msg.get(), buffer->ptr(), total_size);
                
                msg.send();
                stats.messages_sent++;
                stats.bytes_sent += total_size;
                
                // Buffer automatically returns to pool when 'buffer' goes out of scope
            } else {
                log_error("[Client ", conn_id, "] Failed to get buffer for message construction");
                stats.errors++;
                break;
            }
            
            // Yield after batch to allow other connections
            if (++batch_count >= config.batch_size) {
                batch_count = 0;
                std::this_thread::yield();
            }
        }
        
        // Receive echoes if bidirectional
        if (config.bidirectional) {
            for (size_t i = 0; i < config.messages_per_conn; ++i) {
                auto msg = channel->receive();
                
                // Verify echo
                if (msg->connection_id != conn_id || msg->sequence != i) {
                    log_error("[Client ", conn_id, "] Echo mismatch!");
                    stats.errors++;
                }
                
                // Calculate round-trip latency
                uint64_t now = duration_cast<nanoseconds>(
                    high_resolution_clock::now().time_since_epoch()).count();
                uint64_t rtt = now - msg->timestamp_ns;
                stats.latencies.push_back(rtt);
                
                stats.messages_received++;
                stats.bytes_received += sizeof(NetworkTestMessage) + msg->payload_size;
            }
        }
        
        stats.end_time = high_resolution_clock::now();
        
    } catch (const std::exception& e) {
        log_error("[Client ", conn_id, "] Error: ", e.what());
        stats.errors++;
    }
}

// Print statistics summary
void print_statistics(const std::vector<ConnectionStats>& all_stats,
                     const NetworkTestConfig& config) {
    // Aggregate statistics
    uint64_t total_messages_sent = 0;
    uint64_t total_messages_received = 0;
    uint64_t total_bytes_sent = 0;
    uint64_t total_bytes_received = 0;
    uint64_t total_errors = 0;
    std::vector<uint64_t> all_latencies;
    
    auto earliest_start = high_resolution_clock::time_point::max();
    auto latest_end = high_resolution_clock::time_point::min();
    
    for (const auto& stats : all_stats) {
        total_messages_sent += stats.messages_sent;
        total_messages_received += stats.messages_received;
        total_bytes_sent += stats.bytes_sent;
        total_bytes_received += stats.bytes_received;
        total_errors += stats.errors;
        
        all_latencies.insert(all_latencies.end(),
                           stats.latencies.begin(),
                           stats.latencies.end());
        
        if (stats.start_time < earliest_start) {
            earliest_start = stats.start_time;
        }
        if (stats.end_time > latest_end) {
            latest_end = stats.end_time;
        }
    }
    
    auto duration = duration_cast<milliseconds>(latest_end - earliest_start).count();
    
    log_info("========== Network Saturation Test Results ==========");
    log_info("Configuration:");
    log_info("  Mode: ", (config.is_server ? "Server" : "Client"));
    log_info("  Connections: ", config.num_connections);
    log_info("  Message size: ", config.message_size, " bytes");
    log_info("  Messages per connection: ", config.messages_per_conn);
    log_info("  Total messages: ", config.num_connections * config.messages_per_conn);
    log_info("  Bidirectional: ", (config.bidirectional ? "Yes" : "No"));
    
    log_info("Performance:");
    log_info("  Duration: ", duration, " ms");
    log_info("  Messages sent: ", total_messages_sent);
    log_info("  Messages received: ", total_messages_received);
    log_info("  Errors: ", total_errors);
    
    if (duration > 0) {
        double send_rate = (total_messages_sent * 1000.0) / duration;
        double recv_rate = (total_messages_received * 1000.0) / duration;
        double send_bandwidth = (total_bytes_sent / 1024.0 / 1024.0) / (duration / 1000.0);
        double recv_bandwidth = (total_bytes_received / 1024.0 / 1024.0) / (duration / 1000.0);
        
        log_info("  Send rate: ", std::fixed, std::setprecision(2), send_rate, " msg/s");
        log_info("  Receive rate: ", std::fixed, std::setprecision(2), recv_rate, " msg/s");
        log_info("  Send bandwidth: ", std::fixed, std::setprecision(2), send_bandwidth, " MB/s");
        log_info("  Receive bandwidth: ", std::fixed, std::setprecision(2), recv_bandwidth, " MB/s");
        log_info("  Total bandwidth: ", std::fixed, std::setprecision(2), (send_bandwidth + recv_bandwidth), " MB/s");
    }
    
    // Latency statistics
    if (!all_latencies.empty()) {
        std::sort(all_latencies.begin(), all_latencies.end());
        
        log_info("Latency (microseconds):");
        log_info("  Min: ", all_latencies.front() / 1000.0, " μs");
        log_info("  Median: ", all_latencies[all_latencies.size() / 2] / 1000.0, " μs");
        log_info("  90th: ", all_latencies[all_latencies.size() * 90 / 100] / 1000.0, " μs");
        log_info("  99th: ", all_latencies[all_latencies.size() * 99 / 100] / 1000.0, " μs");
        log_info("  Max: ", all_latencies.back() / 1000.0, " μs");
        
        double avg_latency = std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0) 
                           / all_latencies.size() / 1000.0;
        log_info("  Average: ", avg_latency, " μs");
    }
    
    log_info("===================================================");
}

void print_usage(const char* program) {
    log_info("Usage: ", program, " [server|client] [options]");
    log_info("Options:");
    log_info("  --host <hostname>      Remote host (client mode, default: localhost)");
    log_info("  --port <port>          Base port number (default: 10000)");
    log_info("  --connections <num>    Number of parallel connections (default: 4)");
    log_info("  --size <bytes>         Message payload size (default: 1024)");
    log_info("  --count <num>          Messages per connection (default: 100000)");
    log_info("  --batch <size>         Batch size for sending (default: 1000)");
    log_info("  --bidirectional        Enable bidirectional test (echo mode)");
    log_info("  --nodelay              Disable Nagle's algorithm");
    log_info("  --sendbuf <size>       TCP send buffer size in KB");
    log_info("  --recvbuf <size>       TCP receive buffer size in KB");
    log_info("Examples:");
    log_info("  Server: ", program, " server --port 10000 --connections 8");
    log_info("  Client: ", program, " client --host 192.168.1.100 --port 10000 --connections 8");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    NetworkTestConfig config;
    config.is_server = (std::string(argv[1]) == "server");
    config.remote_host = "localhost";
    config.port = 10000;
    config.num_connections = 4;
    config.message_size = 1024;
    config.messages_per_conn = 100000;
    config.batch_size = 1000;
    config.bidirectional = false;
    config.zero_copy = true;
    config.send_buffer_size = 0;
    config.recv_buffer_size = 0;
    config.tcp_nodelay = false;
    
    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            config.remote_host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        } else if (arg == "--connections" && i + 1 < argc) {
            config.num_connections = std::stoul(argv[++i]);
        } else if (arg == "--size" && i + 1 < argc) {
            config.message_size = std::stoul(argv[++i]);
        } else if (arg == "--count" && i + 1 < argc) {
            config.messages_per_conn = std::stoul(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            config.batch_size = std::stoul(argv[++i]);
        } else if (arg == "--bidirectional") {
            config.bidirectional = true;
        } else if (arg == "--nodelay") {
            config.tcp_nodelay = true;
        } else if (arg == "--sendbuf" && i + 1 < argc) {
            config.send_buffer_size = std::stoi(argv[++i]) * 1024;
        } else if (arg == "--recvbuf" && i + 1 < argc) {
            config.recv_buffer_size = std::stoi(argv[++i]) * 1024;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    log_info("Psyne Network Saturation Test");
    log_info("=============================");
    
    // Create statistics for each connection
    std::vector<ConnectionStats> all_stats(config.num_connections);
    std::vector<std::thread> workers;
    
    // Launch worker threads
    if (config.is_server) {
        log_info("Starting server with ", config.num_connections,
                  " connections on ports ", config.port, "-",
                  (config.port + config.num_connections - 1));
        
        for (size_t i = 0; i < config.num_connections; ++i) {
            workers.emplace_back(server_connection_worker, i, 
                               std::ref(config), std::ref(all_stats[i]));
        }
    } else {
        log_info("Starting client with ", config.num_connections,
                  " connections to ", config.remote_host);
        
        for (size_t i = 0; i < config.num_connections; ++i) {
            workers.emplace_back(client_connection_worker, i, 
                               std::ref(config), std::ref(all_stats[i]));
        }
    }
    
    // Wait for all workers to complete
    for (auto& worker : workers) {
        worker.join();
    }
    
    // Print statistics
    print_statistics(all_stats, config);
    
    return 0;
}