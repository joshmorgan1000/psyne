/**
 * @file metal_simple_demo.cpp
 * @brief Simple Metal GPU demo using Objective-C runtime
 *
 * This demo shows Metal integration without requiring Metal-cpp headers
 */

#ifdef __APPLE__

#include <cmath>
#include <iostream>
#include <objc/message.h>
#include <objc/runtime.h>
#include <vector>

// Simplified Metal wrapper using Objective-C runtime
class SimpleMetalContext {
public:
    SimpleMetalContext() : device(nullptr), queue(nullptr) {
        // Get MTLCreateSystemDefaultDevice function
        id (*MTLCreateSystemDefaultDevice)(void) =
            (id (*)(void))dlsym(RTLD_DEFAULT, "MTLCreateSystemDefaultDevice");
        if (!MTLCreateSystemDefaultDevice) {
            throw std::runtime_error("Metal framework not available");
        }

        // Create device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("No Metal device found");
        }

        // Create command queue
        queue = ((id (*)(id, SEL))objc_msgSend)(
            device, sel_registerName("newCommandQueue"));
        if (!queue) {
            throw std::runtime_error("Failed to create command queue");
        }

        // Get device name
        id nameStr =
            ((id (*)(id, SEL))objc_msgSend)(device, sel_registerName("name"));
        const char *name = ((const char *(*)(id, SEL))objc_msgSend)(
            nameStr, sel_registerName("UTF8String"));

        std::cout << "Metal device: " << (name ? name : "Unknown") << std::endl;

        // Check for unified memory
        BOOL hasUnifiedMemory = ((BOOL (*)(id, SEL))objc_msgSend)(
            device, sel_registerName("hasUnifiedMemory"));
        std::cout << "Unified memory: " << (hasUnifiedMemory ? "Yes" : "No")
                  << std::endl;
    }

    ~SimpleMetalContext() {
        if (queue)
            ((void (*)(id, SEL))objc_msgSend)(queue,
                                              sel_registerName("release"));
        if (device)
            ((void (*)(id, SEL))objc_msgSend)(device,
                                              sel_registerName("release"));
    }

    id createBuffer(const void *data, size_t length) {
        // Create buffer with shared storage mode (CPU/GPU accessible)
        const NSUInteger MTLResourceStorageModeShared = 0;
        const NSUInteger MTLResourceCPUCacheModeDefaultCache = 0;
        NSUInteger options = (MTLResourceStorageModeShared << 0) |
                             (MTLResourceCPUCacheModeDefaultCache << 4);

        id buffer = ((
            id (*)(id, SEL, const void *, NSUInteger, NSUInteger))objc_msgSend)(
            device, sel_registerName("newBufferWithBytes:length:options:"),
            data, length, options);

        return buffer;
    }

    void *getBufferContents(id buffer) {
        return ((void *(*)(id, SEL))objc_msgSend)(buffer,
                                                  sel_registerName("contents"));
    }

    size_t getBufferLength(id buffer) {
        return ((NSUInteger (*)(id, SEL))objc_msgSend)(
            buffer, sel_registerName("length"));
    }

private:
    id device;
    id queue;
};

int main() {
    std::cout << "Simple Metal Demo\n";
    std::cout << "=================\n\n";

    try {
        // Create Metal context
        SimpleMetalContext metal;

        // Create test data
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f};
        std::cout << "\nOriginal data: ";
        for (float v : data)
            std::cout << v << " ";
        std::cout << std::endl;

        // Create Metal buffer
        id buffer =
            metal.createBuffer(data.data(), data.size() * sizeof(float));
        if (!buffer) {
            throw std::runtime_error("Failed to create Metal buffer");
        }

        std::cout << "\nMetal buffer created: " << metal.getBufferLength(buffer)
                  << " bytes" << std::endl;

        // Get direct pointer to buffer memory (zero-copy on unified memory)
        float *gpu_ptr = static_cast<float *>(metal.getBufferContents(buffer));

        // Modify data directly in GPU memory
        std::cout << "\nModifying data in GPU memory..." << std::endl;
        for (size_t i = 0; i < data.size(); ++i) {
            gpu_ptr[i] *= 2.0f;
        }

        // Read back results (no copy needed!)
        std::cout << "\nModified data: ";
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << gpu_ptr[i] << " ";
        }
        std::cout << std::endl;

        // Demonstrate unified memory benefit
        std::cout << "\n✅ Zero-copy demonstration:" << std::endl;
        std::cout << "   - CPU wrote directly to GPU memory" << std::endl;
        std::cout << "   - No explicit transfers needed" << std::endl;
        std::cout << "   - Same memory visible to both CPU and GPU"
                  << std::endl;

        // Clean up
        ((void (*)(id, SEL))objc_msgSend)(buffer, sel_registerName("release"));

        std::cout << "\nMetal demo completed successfully!" << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#else

#include <iostream>

int main() {
    std::cout << "This demo requires macOS with Metal support." << std::endl;
    return 1;
}

#endif // __APPLE__