#include <psyne/psyne.hpp>
#include <cassert>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

// Test performance features and reliability
int main() {
    std::cout << "Running Performance and Reliability Tests..." << std::endl;
    
    try {
        // Test 1: Performance Manager
        {
            auto& perf_manager = psyne::PerformanceManager::instance();
            auto summary = perf_manager.get_summary();
            assert(!summary.empty());
            
            auto recommendations = perf_manager.get_recommendations();
            assert(!recommendations.empty());
            
            std::cout << "✓ Performance Manager creation and summary" << std::endl;
        }
        
        // Test 2: Reliability Manager
        {
            auto channel = psyne::create_channel("memory://reliability_test", 1024 * 1024);
            
            psyne::ReliabilityConfig config;
            config.max_retries = 3;
            config.ack_timeout = std::chrono::milliseconds(100);
            
            psyne::ReliabilityManager reliability(*channel, config);
            
            assert(!reliability.is_running());
            reliability.start();
            assert(reliability.is_running());
            
            // Test component access
            assert(reliability.acknowledgment_manager() != nullptr);
            assert(reliability.retry_manager() != nullptr);
            assert(reliability.heartbeat_manager() != nullptr);
            assert(reliability.replay_buffer() != nullptr);
            
            reliability.stop();
            assert(!reliability.is_running());
            
            std::cout << "✓ Reliability Manager lifecycle and components" << std::endl;
        }
        
        // Test 3: Benchmark functionality
        {
            auto channel = psyne::create_channel("memory://benchmark_test", 2 * 1024 * 1024);
            auto& perf_manager = psyne::PerformanceManager::instance();
            
            // Run a basic benchmark
            auto result = perf_manager.benchmark_channel(*channel, 1024, 1000);
            
            // Should have measured something
            assert(result.duration.count() > 0);
            std::cout << "✓ Channel benchmarking" << std::endl;
        }
        
        // Test 4: Performance optimization application
        {
            auto channel = psyne::create_channel("memory://optimization_test", 1024 * 1024);
            auto& perf_manager = psyne::PerformanceManager::instance();
            
            // Apply optimizations (should not throw)
            perf_manager.apply_optimizations(*channel);
            
            // Apply buffer optimizations
            std::vector<uint8_t> buffer(4096, 0);
            perf_manager.apply_optimizations(buffer.data(), buffer.size());
            
            std::cout << "✓ Performance optimization application" << std::endl;
        }
        
        // Test 5: Reliable channel guard
        {
            auto channel = psyne::create_channel("memory://guard_test", 1024 * 1024);
            
            psyne::ReliabilityConfig config;
            config.enable_acknowledgments = true;
            config.max_retries = 2;
            
            // Test reliable channel guard creation
            auto reliable_channel = psyne::create_reliable_channel(*channel, config);
            assert(reliable_channel != nullptr);
            
            std::cout << "✓ Reliable channel guard creation" << std::endl;
        }
        
        // Test 6: Version information
        {
            const char* version = psyne::version();
            assert(version != nullptr);
            assert(strlen(version) > 0);
            
            // Should be version 1.0.0
            std::string version_str(version);
            assert(version_str == "1.0.0");
            
            std::cout << "✓ Version information correct: " << version << std::endl;
        }
        
        std::cout << "All Performance and Reliability Tests Passed! ✅" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}