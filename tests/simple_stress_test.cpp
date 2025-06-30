#include "test_fixtures.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#include <mach/task_info.h>
#endif

using namespace psyne;

size_t get_memory_usage() {
#ifdef __APPLE__
    task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kerr = task_info(mach_task_self(), TASK_BASIC_INFO,
                                   (task_info_t)&info, &size);
    return (kerr == KERN_SUCCESS) ? info.resident_size : 0;
#else
    return 0;
#endif
}

int main() {
    std::cout << "=== Psyne Stress Test for Memory Growth ===" << std::endl;
    
    size_t initial_memory = get_memory_usage();
    std::cout << "Initial memory: " << initial_memory << " bytes" << std::endl;
    
    // Run continuous operations and monitor memory
    for (int cycle = 0; cycle < 10; ++cycle) {
        // Create and destroy channels rapidly
        for (int i = 0; i < 100; ++i) {
            auto channel = Channel::create("memory://stress_" + std::to_string(i), 1024 * 1024);
            
            // Send and receive some messages
            for (int j = 0; j < 10; ++j) {
                auto slot = channel->reserve_write_slot(100);
                if (slot != 0xFFFFFFFF) {
                    auto span = channel->get_write_span(100);
                    span[0] = static_cast<uint8_t>(j);
                    channel->notify_message_ready(slot, 100);
                }
                
                size_t size;
                uint32_t type;
                if (void* msg_data = channel->receive_raw_message(size, type)) {
                    channel->release_raw_message(msg_data);
                }
            }
        }
        
        size_t current_memory = get_memory_usage();
        double growth = ((double)(current_memory - initial_memory) / initial_memory) * 100.0;
        
        std::cout << "Cycle " << cycle + 1 << ": " << current_memory 
                  << " bytes (+" << (current_memory - initial_memory) 
                  << ", " << std::fixed << std::setprecision(1) << growth << "%)" << std::endl;
        
        // Check for runaway growth
        if (growth > 50.0) {
            std::cerr << "FAIL: Excessive memory growth detected!" << std::endl;
            return 1;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "\n=== Stress Test PASSED ===" << std::endl;
    std::cout << "Memory growth remained within acceptable bounds" << std::endl;
    
    return 0;
}