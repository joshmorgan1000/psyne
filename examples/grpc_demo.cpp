#include "../src/grpc/grpc_adapter.hpp"
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;
using namespace psyne::grpc_compat;

// Example service implementation
class CalculatorService {
public:
    CalculatorService(std::shared_ptr<Channel> channel) : channel_(channel) {}

    void Run() {
        std::cout << "Calculator service running..." << std::endl;

        while (!channel_->is_stopped()) {
            // Receive request
            auto request =
                channel_->receive<ByteVector>(std::chrono::milliseconds(100));
            if (!request)
                continue;

            // Parse request (simple protocol: 4 bytes op, 4 bytes a, 4 bytes b)
            if (request->size() >= 12) {
                uint32_t op = *reinterpret_cast<uint32_t *>(request->data());
                float a = *reinterpret_cast<float *>(request->data() + 4);
                float b = *reinterpret_cast<float *>(request->data() + 8);

                float result = 0;
                switch (op) {
                case 0:
                    result = a + b;
                    break;
                case 1:
                    result = a - b;
                    break;
                case 2:
                    result = a * b;
                    break;
                case 3:
                    result = (b != 0) ? a / b : 0;
                    break;
                }

                // Send response
                ByteVector response(*channel_);
                response.resize(sizeof(float));
                *reinterpret_cast<float *>(response.data()) = result;
                channel_->send(response);

                std::cout << "Processed: " << a << " op(" << op << ") " << b
                          << " = " << result << std::endl;
            }
        }
    }

private:
    std::shared_ptr<Channel> channel_;
};

void run_grpc_server() {
    // Create Psyne channels for the service
    auto request_channel =
        create_channel("memory://calc_requests", 1024 * 1024);
    auto response_channel =
        create_channel("memory://calc_responses", 1024 * 1024);

    // Create calculator service
    CalculatorService calc_service(request_channel);
    std::thread service_thread([&calc_service]() { calc_service.Run(); });

    // Create gRPC adapter
    auto adapter = CreateGrpcService("Calculator", "0.0.0.0:50051",
                                     request_channel, response_channel);

    // Define the RPC handler
    auto handler = [response_channel](const void *request, size_t request_size,
                                      void *response, size_t &response_size) {
        // Wait for service to process and send response
        auto psyne_response = response_channel->receive<ByteVector>(
            std::chrono::milliseconds(1000));

        if (psyne_response && psyne_response->size() <= response_size) {
            std::memcpy(response, psyne_response->data(),
                        psyne_response->size());
            response_size = psyne_response->size();
        }
    };

    adapter->RegisterUnaryMethod("/Calculator/Calculate", handler);

    // Start the gRPC server
    adapter->Start();
    std::cout << "gRPC server started on port 50051" << std::endl;

    // Wait for shutdown
    adapter->Wait();

    request_channel->stop();
    service_thread.join();
}

void run_native_client() {
    std::cout << "Running native Psyne client..." << std::endl;

    // Connect directly to the Psyne channels
    auto request_channel =
        create_channel("memory://calc_requests", 1024 * 1024);
    auto response_channel =
        create_channel("memory://calc_responses", 1024 * 1024);

    // Make some calculations
    for (int i = 0; i < 5; ++i) {
        ByteVector request(*request_channel);
        request.resize(12);

        uint32_t op = i % 4; // Cycle through operations
        float a = 10.0f + i;
        float b = 3.0f;

        *reinterpret_cast<uint32_t *>(request.data()) = op;
        *reinterpret_cast<float *>(request.data() + 4) = a;
        *reinterpret_cast<float *>(request.data() + 8) = b;

        request_channel->send(request);

        auto response = response_channel->receive<ByteVector>(
            std::chrono::milliseconds(1000));

        if (response && response->size() >= sizeof(float)) {
            float result = *reinterpret_cast<float *>(response->data());
            std::cout << "Native client: " << a << " op(" << op << ") " << b
                      << " = " << result << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void run_grpc_client() {
    std::cout << "Running gRPC client..." << std::endl;

    // Create a dummy channel (not used for gRPC client)
    auto dummy_channel = create_channel("memory://dummy", 1024);

    // Create gRPC client
    GrpcClient client(dummy_channel, "localhost:50051");

    // Make some calculations via gRPC
    for (int i = 0; i < 5; ++i) {
        // Prepare request
        uint8_t request_data[12];
        uint32_t op = i % 4;
        float a = 20.0f + i;
        float b = 4.0f;

        *reinterpret_cast<uint32_t *>(request_data) = op;
        *reinterpret_cast<float *>(request_data + 4) = a;
        *reinterpret_cast<float *>(request_data + 8) = b;

        // Make RPC call
        uint8_t response_data[256];
        size_t response_size = sizeof(response_data);

        auto status = client.CallUnary("/Calculator/Calculate", request_data,
                                       sizeof(request_data), response_data,
                                       response_size);

        if (status.ok() && response_size >= sizeof(float)) {
            float result = *reinterpret_cast<float *>(response_data);
            std::cout << "gRPC client: " << a << " op(" << op << ") " << b
                      << " = " << result << std::endl;
        } else {
            std::cout << "gRPC call failed: " << status.error_message()
                      << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0]
                  << " [server|native-client|grpc-client]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    try {
        if (mode == "server") {
            run_grpc_server();
        } else if (mode == "native-client") {
            run_native_client();
        } else if (mode == "grpc-client") {
            run_grpc_client();
        } else {
            std::cerr << "Invalid mode. Use 'server', 'native-client', or "
                         "'grpc-client'"
                      << std::endl;
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}