#pragma once

#include <psyne/psyne.hpp>
#include <grpcpp/grpcpp.h>
#include <memory>
#include <functional>
#include <unordered_map>

namespace psyne {
namespace grpc_compat {

/**
 * @brief Adapter to expose Psyne channels as gRPC services
 * 
 * This adapter allows Psyne-based services to be accessed via gRPC,
 * enabling interoperability with the gRPC ecosystem.
 */
class GrpcAdapter {
public:
    using MessageHandler = std::function<void(const void* request, size_t request_size,
                                             void* response, size_t& response_size)>;

    /**
     * @brief Create a gRPC adapter for a Psyne channel
     * @param channel The Psyne channel to adapt
     * @param server_address Address to bind the gRPC server (e.g., "0.0.0.0:50051")
     */
    GrpcAdapter(std::shared_ptr<Channel> channel, const std::string& server_address);
    ~GrpcAdapter();

    /**
     * @brief Register a unary RPC method
     * @param method_name Full method name (e.g., "/service.Package/Method")
     * @param handler Function to handle the RPC
     */
    void RegisterUnaryMethod(const std::string& method_name, MessageHandler handler);

    /**
     * @brief Register a server streaming RPC method
     * @param method_name Full method name
     * @param handler Function to handle the RPC (called multiple times for streaming)
     */
    void RegisterServerStreamingMethod(const std::string& method_name, MessageHandler handler);

    /**
     * @brief Start the gRPC server
     */
    void Start();

    /**
     * @brief Stop the gRPC server
     */
    void Stop();

    /**
     * @brief Wait for the server to shut down
     */
    void Wait();

private:
    class AsyncService;
    class CallData;

    std::shared_ptr<Channel> channel_;
    std::string server_address_;
    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    std::unique_ptr<AsyncService> service_;
    std::unordered_map<std::string, MessageHandler> handlers_;
    std::thread server_thread_;
    std::atomic<bool> shutdown_{false};

    void HandleRpcs();
};

/**
 * @brief Client adapter to access gRPC services via Psyne channels
 */
class GrpcClient {
public:
    /**
     * @brief Create a gRPC client that forwards to a Psyne channel
     * @param channel The Psyne channel for communication
     * @param target gRPC server address (e.g., "localhost:50051")
     */
    GrpcClient(std::shared_ptr<Channel> channel, const std::string& target);
    ~GrpcClient();

    /**
     * @brief Make a unary RPC call
     * @param method Full method name (e.g., "/service.Package/Method")
     * @param request Request message data
     * @param request_size Size of request data
     * @param response Buffer for response data
     * @param response_size Size of response buffer (updated with actual size)
     * @return gRPC status
     */
    grpc::Status CallUnary(const std::string& method,
                          const void* request, size_t request_size,
                          void* response, size_t& response_size);

    /**
     * @brief Make a server streaming RPC call
     * @param method Full method name
     * @param request Request message data
     * @param request_size Size of request data
     * @param stream_handler Called for each response in the stream
     * @return gRPC status
     */
    grpc::Status CallServerStreaming(const std::string& method,
                                    const void* request, size_t request_size,
                                    std::function<void(const void*, size_t)> stream_handler);

private:
    std::shared_ptr<Channel> channel_;
    std::shared_ptr<grpc::Channel> grpc_channel_;
    std::unique_ptr<grpc::GenericStub> stub_;
};

/**
 * @brief Protocol buffer message wrapper for Psyne
 * 
 * Allows sending protobuf messages over Psyne channels with zero-copy
 * when possible.
 */
template<typename ProtoMessage>
class ProtobufMessage : public Message<ProtobufMessage<ProtoMessage>> {
public:
    static constexpr uint32_t message_type = 1000; // Base type for protobuf messages
    
    using Message<ProtobufMessage<ProtoMessage>>::Message;
    
    static size_t calculate_size() {
        // Default size - will be resized as needed
        return 4096;
    }
    
    /**
     * @brief Serialize a protobuf message into this Psyne message
     * @param proto The protobuf message to serialize
     * @return true if successful, false if buffer too small
     */
    bool SerializeFrom(const ProtoMessage& proto) {
        size_t size = proto.ByteSizeLong();
        if (size > this->capacity()) {
            return false;
        }
        
        this->resize(size);
        return proto.SerializeToArray(this->data(), size);
    }
    
    /**
     * @brief Parse this Psyne message as a protobuf
     * @param proto Output protobuf message
     * @return true if successful
     */
    bool ParseTo(ProtoMessage& proto) const {
        return proto.ParseFromArray(this->data(), this->size());
    }
    
    void initialize() {
        // No special initialization needed
    }
};

/**
 * @brief Create a gRPC service that forwards to Psyne channels
 * 
 * This is a convenience function that sets up a complete gRPC service
 * backed by Psyne channels for high-performance communication.
 * 
 * @param service_name Name of the gRPC service
 * @param server_address Address to bind (e.g., "0.0.0.0:50051")
 * @param request_channel Channel for receiving requests
 * @param response_channel Channel for sending responses
 * @return Configured GrpcAdapter ready to start
 */
std::unique_ptr<GrpcAdapter> CreateGrpcService(
    const std::string& service_name,
    const std::string& server_address,
    std::shared_ptr<Channel> request_channel,
    std::shared_ptr<Channel> response_channel);

/**
 * @brief Bridge between gRPC and Psyne for bidirectional communication
 * 
 * This class provides a high-level interface for creating services that
 * can be accessed both via gRPC and native Psyne channels.
 */
class GrpcBridge {
public:
    struct Config {
        std::string grpc_address = "0.0.0.0:50051";
        size_t channel_buffer_size = 16 * 1024 * 1024;
        bool enable_compression = true;
        bool enable_reflection = true; // gRPC reflection for debugging
    };
    
    GrpcBridge(const Config& config = {});
    ~GrpcBridge();
    
    /**
     * @brief Add a service to the bridge
     * @param service_name Fully qualified service name
     * @param impl Service implementation
     */
    template<typename ServiceImpl>
    void AddService(const std::string& service_name, 
                   std::shared_ptr<ServiceImpl> impl);
    
    /**
     * @brief Get the Psyne channel for a service
     * @param service_name Service name
     * @return Channel for direct Psyne communication
     */
    std::shared_ptr<Channel> GetServiceChannel(const std::string& service_name);
    
    /**
     * @brief Start the bridge (both gRPC and Psyne endpoints)
     */
    void Start();
    
    /**
     * @brief Stop the bridge
     */
    void Stop();
    
    /**
     * @brief Wait for shutdown
     */
    void Wait();
    
private:
    Config config_;
    std::unique_ptr<grpc::Server> grpc_server_;
    std::unordered_map<std::string, std::shared_ptr<Channel>> service_channels_;
    std::unordered_map<std::string, std::unique_ptr<GrpcAdapter>> adapters_;
};

} // namespace grpc_compat
} // namespace psyne