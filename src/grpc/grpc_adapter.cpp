#include "grpc_adapter.hpp"
#include <grpcpp/generic/async_generic_service.h>
#include <grpcpp/support/byte_buffer.h>
#include <iostream>

namespace psyne {
namespace grpc_compat {

// Internal async service implementation
class GrpcAdapter::AsyncService : public grpc::AsyncGenericService {
public:
    AsyncService() = default;
};

// Call data for handling RPCs
class GrpcAdapter::CallData {
public:
    CallData(AsyncService* service, grpc::ServerCompletionQueue* cq,
             std::unordered_map<std::string, MessageHandler>& handlers,
             std::shared_ptr<Channel> channel)
        : service_(service), cq_(cq), handlers_(handlers), channel_(channel),
          responder_(&ctx_), status_(CREATE) {
        Proceed();
    }
    
    void Proceed() {
        if (status_ == CREATE) {
            status_ = PROCESS;
            service_->RequestCall(&ctx_, &stream_, cq_, cq_, this);
        } else if (status_ == PROCESS) {
            // Spawn a new CallData instance to serve new clients
            new CallData(service_, cq_, handlers_, channel_);
            
            // Get the method name
            std::string method = ctx_.method();
            
            // Find handler
            auto it = handlers_.find(method);
            if (it == handlers_.end()) {
                status_ = FINISH;
                responder_.Finish(grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                                              "Method not found"), this);
                return;
            }
            
            // Read request
            stream_.Read(&request_buffer_, this);
            status_ = READ;
        } else if (status_ == READ) {
            // Convert ByteBuffer to raw bytes
            std::vector<grpc::Slice> slices;
            request_buffer_.Dump(&slices);
            
            size_t total_size = 0;
            for (const auto& slice : slices) {
                total_size += slice.size();
            }
            
            std::vector<uint8_t> request_data(total_size);
            size_t offset = 0;
            for (const auto& slice : slices) {
                std::memcpy(request_data.data() + offset, slice.begin(), slice.size());
                offset += slice.size();
            }
            
            // Process request
            std::vector<uint8_t> response_data(64 * 1024); // 64KB initial size
            size_t response_size = response_data.size();
            
            auto it = handlers_.find(ctx_.method());
            it->second(request_data.data(), request_data.size(),
                      response_data.data(), response_size);
            
            // Create response buffer
            grpc::Slice response_slice(response_data.data(), response_size);
            grpc::ByteBuffer response_buffer(&response_slice, 1);
            
            status_ = WRITE;
            stream_.Write(response_buffer, this);
        } else if (status_ == WRITE) {
            status_ = FINISH;
            stream_.Finish(grpc::Status::OK, this);
        } else {
            GPR_ASSERT(status_ == FINISH);
            delete this;
        }
    }
    
private:
    AsyncService* service_;
    grpc::ServerCompletionQueue* cq_;
    std::unordered_map<std::string, MessageHandler>& handlers_;
    std::shared_ptr<Channel> channel_;
    
    grpc::ServerContext ctx_;
    grpc::GenericServerAsyncReaderWriter stream_;
    grpc::ServerAsyncResponseWriter<grpc::ByteBuffer> responder_;
    
    grpc::ByteBuffer request_buffer_;
    
    enum CallStatus { CREATE, PROCESS, READ, WRITE, FINISH };
    CallStatus status_;
};

// GrpcAdapter implementation
GrpcAdapter::GrpcAdapter(std::shared_ptr<Channel> channel, const std::string& server_address)
    : channel_(channel), server_address_(server_address) {
    service_ = std::make_unique<AsyncService>();
}

GrpcAdapter::~GrpcAdapter() {
    Stop();
}

void GrpcAdapter::RegisterUnaryMethod(const std::string& method_name, MessageHandler handler) {
    handlers_[method_name] = handler;
}

void GrpcAdapter::RegisterServerStreamingMethod(const std::string& method_name, MessageHandler handler) {
    // For now, treat the same as unary - full implementation would handle streaming
    handlers_[method_name] = handler;
}

void GrpcAdapter::Start() {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
    builder.RegisterAsyncGenericService(service_.get());
    cq_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();
    
    std::cout << "gRPC adapter listening on " << server_address_ << std::endl;
    
    // Start the RPC handling thread
    server_thread_ = std::thread(&GrpcAdapter::HandleRpcs, this);
}

void GrpcAdapter::Stop() {
    shutdown_ = true;
    if (server_) {
        server_->Shutdown();
    }
    if (cq_) {
        cq_->Shutdown();
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void GrpcAdapter::Wait() {
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void GrpcAdapter::HandleRpcs() {
    // Spawn a new CallData instance to serve new clients
    new CallData(service_.get(), cq_.get(), handlers_, channel_);
    
    void* tag;
    bool ok;
    while (true) {
        GPR_ASSERT(cq_->Next(&tag, &ok));
        if (!ok || shutdown_) {
            break;
        }
        static_cast<CallData*>(tag)->Proceed();
    }
}

// GrpcClient implementation
GrpcClient::GrpcClient(std::shared_ptr<Channel> channel, const std::string& target)
    : channel_(channel) {
    grpc_channel_ = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    stub_ = std::make_unique<grpc::GenericStub>(grpc_channel_);
}

GrpcClient::~GrpcClient() = default;

grpc::Status GrpcClient::CallUnary(const std::string& method,
                                  const void* request, size_t request_size,
                                  void* response, size_t& response_size) {
    grpc::ClientContext context;
    
    // Create request buffer
    grpc::Slice request_slice(request, request_size);
    grpc::ByteBuffer request_buffer(&request_slice, 1);
    
    // Make the call
    grpc::ByteBuffer response_buffer;
    grpc::Status status = stub_->UnaryCall(&context, method, 
                                          request_buffer, &response_buffer);
    
    if (status.ok()) {
        // Extract response
        std::vector<grpc::Slice> slices;
        response_buffer.Dump(&slices);
        
        size_t total_size = 0;
        for (const auto& slice : slices) {
            total_size += slice.size();
        }
        
        if (total_size <= response_size) {
            size_t offset = 0;
            for (const auto& slice : slices) {
                std::memcpy(static_cast<uint8_t*>(response) + offset, 
                           slice.begin(), slice.size());
                offset += slice.size();
            }
            response_size = total_size;
        } else {
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                               "Response buffer too small");
        }
    }
    
    return status;
}

grpc::Status GrpcClient::CallServerStreaming(const std::string& method,
                                           const void* request, size_t request_size,
                                           std::function<void(const void*, size_t)> stream_handler) {
    grpc::ClientContext context;
    
    // Create request buffer
    grpc::Slice request_slice(request, request_size);
    grpc::ByteBuffer request_buffer(&request_slice, 1);
    
    // Make the streaming call
    auto reader = stub_->PrepareCall(&context, method, request_buffer);
    reader->StartCall();
    
    grpc::ByteBuffer response_buffer;
    while (reader->Read(&response_buffer)) {
        // Extract response
        std::vector<grpc::Slice> slices;
        response_buffer.Dump(&slices);
        
        std::vector<uint8_t> response_data;
        for (const auto& slice : slices) {
            response_data.insert(response_data.end(), slice.begin(), slice.end());
        }
        
        stream_handler(response_data.data(), response_data.size());
    }
    
    grpc::Status status = reader->Finish();
    return status;
}

// Convenience function implementation
std::unique_ptr<GrpcAdapter> CreateGrpcService(
    const std::string& service_name,
    const std::string& server_address,
    std::shared_ptr<Channel> request_channel,
    std::shared_ptr<Channel> response_channel) {
    
    auto adapter = std::make_unique<GrpcAdapter>(request_channel, server_address);
    
    // Register a generic handler that forwards to Psyne channels
    auto handler = [request_channel, response_channel](
        const void* request, size_t request_size,
        void* response, size_t& response_size) {
        
        // Forward request to Psyne channel
        ByteVector psyne_request(*request_channel);
        psyne_request.resize(request_size);
        std::memcpy(psyne_request.data(), request, request_size);
        request_channel->send(psyne_request);
        
        // Wait for response
        auto psyne_response = response_channel->receive<ByteVector>(
            std::chrono::milliseconds(5000));
        
        if (psyne_response && psyne_response->size() <= response_size) {
            std::memcpy(response, psyne_response->data(), psyne_response->size());
            response_size = psyne_response->size();
        } else {
            response_size = 0;
        }
    };
    
    // Register for all methods under this service
    adapter->RegisterUnaryMethod("/" + service_name + "/*", handler);
    
    return adapter;
}

// GrpcBridge implementation
GrpcBridge::GrpcBridge(const Config& config) : config_(config) {}

GrpcBridge::~GrpcBridge() {
    Stop();
}

std::shared_ptr<Channel> GrpcBridge::GetServiceChannel(const std::string& service_name) {
    auto it = service_channels_.find(service_name);
    if (it != service_channels_.end()) {
        return it->second;
    }
    
    // Create new channel for this service
    auto channel = create_channel("memory://grpc_" + service_name, 
                                 config_.channel_buffer_size);
    service_channels_[service_name] = channel;
    return channel;
}

void GrpcBridge::Start() {
    // Start all adapters
    for (auto& [name, adapter] : adapters_) {
        adapter->Start();
    }
    
    // Build and start gRPC server if needed
    if (!adapters_.empty() && !grpc_server_) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort(config_.grpc_address, 
                                grpc::InsecureServerCredentials());
        
        if (config_.enable_reflection) {
            // TODO: Add reflection service
        }
        
        grpc_server_ = builder.BuildAndStart();
        std::cout << "GrpcBridge listening on " << config_.grpc_address << std::endl;
    }
}

void GrpcBridge::Stop() {
    // Stop all adapters
    for (auto& [name, adapter] : adapters_) {
        adapter->Stop();
    }
    
    if (grpc_server_) {
        grpc_server_->Shutdown();
    }
}

void GrpcBridge::Wait() {
    if (grpc_server_) {
        grpc_server_->Wait();
    }
}

} // namespace grpc_compat
} // namespace psyne