#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;

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

void run_tcp_server() {
    std::cout << "Starting TCP server (simulating gRPC)..." << std::endl;
    
    // Create TCP server channel
    auto server_channel = create_channel("tcp://:50051", 1024 * 1024);
    
    // Create calculator service (convert unique_ptr to shared_ptr)
    std::shared_ptr<Channel> shared_channel = std::move(server_channel);
    CalculatorService calc_service(shared_channel);
    std::thread service_thread([&calc_service]() { calc_service.Run(); });
    
    std::cout << "TCP server started on port 50051" << std::endl;
    std::cout << "Press Enter to stop..." << std::endl;
    
    // Wait for user input to stop
    std::cin.get();
    
    server_channel->stop();
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

void run_tcp_client() {
    std::cout << "Running TCP client (simulating gRPC)..." << std::endl;
    
    // Create TCP client channel
    auto client_channel = create_channel("tcp://localhost:50051", 1024 * 1024);
    
    // Give server time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // Make some calculations via TCP
    for (int i = 0; i < 5; ++i) {
        ByteVector request(*client_channel);
        request.resize(12);
        
        uint32_t op = i % 4;
        float a = 20.0f + i;
        float b = 4.0f;
        
        *reinterpret_cast<uint32_t *>(request.data()) = op;
        *reinterpret_cast<float *>(request.data() + 4) = a;
        *reinterpret_cast<float *>(request.data() + 8) = b;
        
        request.send();
        
        auto response = client_channel->receive<ByteVector>(
            std::chrono::milliseconds(1000));
            
        if (response && response->size() >= sizeof(float)) {
            float result = *reinterpret_cast<float *>(response->data());
            std::cout << "TCP client: " << a << " op(" << op << ") " << b
                      << " = " << result << std::endl;
        } else {
            std::cout << "TCP call failed or timed out" << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0]
                  << " [server|native-client|tcp-client]" << std::endl;
        std::cerr << "Note: This demo shows TCP-based RPC, simulating gRPC behavior" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    try {
        if (mode == "server") {
            run_tcp_server();
        } else if (mode == "native-client") {
            run_native_client();
        } else if (mode == "tcp-client") {
            run_tcp_client();
        } else {
            std::cerr << "Invalid mode. Use 'server', 'native-client', or "
                         "'tcp-client'"
                      << std::endl;
            return 1;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}