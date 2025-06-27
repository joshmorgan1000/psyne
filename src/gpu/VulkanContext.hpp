#pragma once

#include "PipelineManager.hpp"
#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vulkan/vulkan.h>

namespace psyne {

/**
 * @brief A singleton class that manages the Vulkan context.
 *
 * This class is responsible for creating and managing the Vulkan instance, device, and other
 * Vulkan objects.
 */
class VulkanContext {
  public:
    /**
     * @brief Get the singleton instance of the VulkanContext.
     *
     * @return VulkanContext&
     */
    static VulkanContext& getInstance(uint32_t deviceIndex = 0) {
        static std::mutex mtx;
        static std::unordered_map<uint32_t, std::unique_ptr<VulkanContext>> insts;
        std::lock_guard<std::mutex> lock(mtx);
        auto it = insts.find(deviceIndex);
        if (it == insts.end()) {
            auto ptr = std::unique_ptr<VulkanContext>(new VulkanContext(deviceIndex));
            it = insts.emplace(deviceIndex, std::move(ptr)).first;
        }
        return *it->second;
    }

    /**
     * @brief Check if a GPU was found/initialized. If the system this is running on does not have
     *        a GPU, this will return false.
     */
    bool isGPUCapable() const { return gpuCapable_; }

    /**
     * @brief Returns the last initialization error message, if any.
     */
    const std::string& getInitError() const { return initError_; }

    /**
     * @brief Get the Vulkan instance handle. The instance handle is the root object for all
     *        Vulkan operations.
     */
    VkInstance getInstanceHandle() const { return instance_; }

    /**
     * @brief Get the Vulkan device handle. The device handle is the primary object for
     *        interacting with the GPU.
     */
    VkDevice getDevice() const { return device_; }

    /**
     * @brief Get the physical device handle. The physical device handle represents the
     *        physical GPU.
     */
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice_; }

    /**
     * @brief Get the compute queue handle. The compute queue is used for dispatching compute
     *        work to the GPU.
     */
    VkQueue getComputeQueue() const { return queue_; }

    /**
     * @brief Get the command pool handle. The command pool is used for creating command buffers
     *        for the GPU. Command buffers are used to record and submit work to the GPU.
     */
    VkCommandPool getCommandPool() const { return commandPool_; }

    /**
     * @brief Get the PipelineManager. The PipelineManager is responsible for creating and managing
     *        compute pipelines.
     */
    PipelineManager& pipelineManager() { return *pipelineManager_; }

    /**
     * @brief Destroy the VulkanContext singleton instance.
     */
    ~VulkanContext() { cleanup(); }

    /**
     * @brief Allocate a global GPU arena. We do this once at the beginning so we can directly
     *        map the memory and use it for all subsequent GPU operations. This is a performance
     *        optimization that means in many cases objects can be created and destroyed without
     *        ever leaving the GPU.
     *
     * @param size The size of the GPU arena to allocate.
     * @param device_memory The device memory handle for the allocated arena.
     * @return uint8_t* A pointer to the mapped GPU memory.
     */
    uint8_t* allocateGPUArena(size_t size) {
        if (!initialized_)
            initialize();
        if (mapped == nullptr) {
            VkBufferCreateInfo arenatBufCI{};
            arenatBufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            arenatBufCI.size = static_cast<VkDeviceSize>(size);
            arenatBufCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            arenatBufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            arenatBufCI.queueFamilyIndexCount = 0;
            arenatBufCI.pQueueFamilyIndices = nullptr;
            arenaBuf = VK_NULL_HANDLE;
            if (vkCreateBuffer(device_, &arenatBufCI, nullptr, &arenaBuf) != VK_SUCCESS) {
                throw std::runtime_error("VulkanContext: create arena buffer failed.");
            }
            // Get memory requirements
            VkMemoryRequirements arenaReq;
            vkGetBufferMemoryRequirements(device_, arenaBuf, &arenaReq);
            // Find memory type
            VkPhysicalDeviceMemoryProperties memArenaProps;
            vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memArenaProps);
            // Allocate Memory fo inBuf
            VkMemoryAllocateInfo arenaAI{};
            arenaAI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            arenaAI.allocationSize = arenaReq.size;
            uint32_t memoryTypeIndex = UINT32_MAX;
            for (uint32_t i = 0; i < memArenaProps.memoryTypeCount; i++) {
                if ((arenaReq.memoryTypeBits & (1 << i)) &&
                    (memArenaProps.memoryTypes[i].propertyFlags &
                     (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    memoryTypeIndex = i;
                    break;
                }
            }
            if (memoryTypeIndex == UINT32_MAX) {
                vkDestroyBuffer(device_, arenaBuf, nullptr);
                arenaBuf = VK_NULL_HANDLE;
                throw std::runtime_error(
                    "VulkanContext: failed to find suitable memory type for arena buffer.");
            }
            arenaAI.memoryTypeIndex = memoryTypeIndex;
            if (vkAllocateMemory(device_, &arenaAI, nullptr, &arenaMem) != VK_SUCCESS) {
                vkDestroyBuffer(device_, arenaBuf, nullptr);
                arenaBuf = VK_NULL_HANDLE;
                throw std::runtime_error("VulkanContext: allocate input memory failed.");
            }
            if (vkBindBufferMemory(device_, arenaBuf, arenaMem, 0) != VK_SUCCESS) {
                vkFreeMemory(device_, arenaMem, nullptr);
                vkDestroyBuffer(device_, arenaBuf, nullptr);
                arenaBuf = VK_NULL_HANDLE;
                arenaMem = VK_NULL_HANDLE;
                throw std::runtime_error("VulkanContext: bind arena buffer memory failed.");
            }
            if (vkMapMemory(device_, arenaMem, 0, static_cast<VkDeviceSize>(size), 0, &mapped) !=
                VK_SUCCESS) {
                vkFreeMemory(device_, arenaMem, nullptr);
                vkDestroyBuffer(device_, arenaBuf, nullptr);
                arenaBuf = VK_NULL_HANDLE;
                arenaMem = VK_NULL_HANDLE;
                throw std::runtime_error("VulkanContext: map arena memory failed.");
            }
            device_memory_ = arenaMem;
        }
        return reinterpret_cast<uint8_t*>(mapped);
    }

  private:
    VulkanContext(uint32_t index)
        : instance_(VK_NULL_HANDLE), physicalDevice_(VK_NULL_HANDLE), device_(VK_NULL_HANDLE),
          deviceIndex_(index), queue_(VK_NULL_HANDLE), commandPool_(VK_NULL_HANDLE),
          pipelineManager_(nullptr), gpuCapable_(false), initialized_(false) {
        initialize();
    }

    /**
     * @brief Initialize the Vulkan context.
     */
    void initialize() {
        if (initialized_)
            return;
        registerAllShaders();
        createInstance();

        initError_.clear();
        uint32_t devCount = 0;
        vkEnumeratePhysicalDevices(instance_, &devCount, nullptr);
        if (devCount == 0) {
            initError_ = "No Vulkan-compatible GPU found.";
            std::cerr << "VulkanContext: " << initError_ << '\n';
            gpuCapable_ = false;
            initialized_ = true;
            return;
        }

        pickPhysicalDevice();
        if (physicalDevice_ == VK_NULL_HANDLE) {
            initError_ = "Failed to find a suitable Vulkan device.";
            std::cerr << "VulkanContext: " << initError_ << '\n';
            gpuCapable_ = false;
            initialized_ = true;
            return;
        }

        createLogicalDevice();
        createCommandPool();

        // Create managers for pipelines & memory
        pipelineManager_ = std::unique_ptr<PipelineManager>(new PipelineManager(device_));

        gpuCapable_ = true;
        initialized_ = true;
    }

    /**
     * @brief Cleanup the Vulkan context.
     */
    void cleanup() {
        if (!initialized_)
            return;
        if (mapped != nullptr) {
            vkUnmapMemory(device_, device_memory_);
            vkDestroyBuffer(device_, arenaBuf, nullptr);
            vkFreeMemory(device_, arenaMem, nullptr);
            mapped = nullptr;
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device_);
        }
        if (pipelineManager_ != nullptr) { // Check if pipelineManager_ is not null
            pipelineManager_->destroyAll();
        }
        pipelineManager_.reset();

        if (commandPool_) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
            commandPool_ = VK_NULL_HANDLE;
        }
        if (device_) {
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (instance_) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
        initialized_ = false;
        gpuCapable_ = false;
    }

    /**
     * @brief Singleton initialization function.
     */
    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "MyVulkanCompute";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 2, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        // Example: you could enable certain extensions if needed
        std::vector<const char*> extensions;
#ifdef __APPLE__
        // On macOS with MoltenVK, you often need "VK_KHR_portability_enumeration"
        extensions.push_back("VK_KHR_portability_enumeration");
#endif

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &appInfo;
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
#ifdef __APPLE__
        ci.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        VkResult res = vkCreateInstance(&ci, nullptr, &instance_);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("VulkanContext: Failed to create Vulkan instance.");
        }
    }

    /**
     * @brief Pick the physical device to use.
     */
    void pickPhysicalDevice() {
        uint32_t devCount = 0;
        vkEnumeratePhysicalDevices(instance_, &devCount, nullptr);
        std::vector<VkPhysicalDevice> devices(devCount);
        vkEnumeratePhysicalDevices(instance_, &devCount, devices.data());

        if (deviceIndex_ < devCount) {
            physicalDevice_ = devices[deviceIndex_];
            return;
        }

        size_t bestScore = 0;
        int bestIndex = -1;

        // We'll pick whichever device has the most device-local memory
        for (int i = 0; i < static_cast<int>(devCount); i++) {
            VkPhysicalDevice pd = devices[i];

            // Check if there's a valid compute queue
            uint32_t qCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, nullptr);
            std::vector<VkQueueFamilyProperties> qProps(qCount);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qCount, qProps.data());
            bool hasCompute = false;
            for (auto& q : qProps) {
                if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    hasCompute = true;
                    break;
                }
            }
            if (!hasCompute)
                continue;

            // Score by total device-local memory
            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
            uint64_t localMem = 0;
            for (uint32_t m = 0; m < memProps.memoryHeapCount; m++) {
                if (memProps.memoryHeaps[m].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    localMem += memProps.memoryHeaps[m].size;
                }
            }
            if (localMem > bestScore) {
                bestScore = localMem;
                bestIndex = i;
            }
        }

        if (bestIndex < 0) {
            return;
        }
        physicalDevice_ = devices[bestIndex];
        // Could log device name, etc., if desired
    }

    /**
     * @brief Create the logical device.
     */
    void createLogicalDevice() {
        // Find a compute-only or compute-capable queue
        uint32_t queueFamilyIndex = findComputeQueueFamily(physicalDevice_);

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queueFamilyIndex;
        qci.queueCount = 1;
        qci.pQueuePriorities = &queuePriority;

        // If you need additional device extensions, add them here
        std::vector<const char*> devExtensions;
#ifdef __APPLE__
        devExtensions.push_back("VK_KHR_portability_subset");
#endif

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(devExtensions.size());
        dci.ppEnabledExtensionNames = devExtensions.empty() ? nullptr : devExtensions.data();

        VkResult res = vkCreateDevice(physicalDevice_, &dci, nullptr, &device_);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("VulkanContext: Failed to create logical device.");
        }

        // Retrieve the queue handle
        vkGetDeviceQueue(device_, queueFamilyIndex, 0, &queue_);
    }

    /**
     * @brief Create the command pool.
     */
    void createCommandPool() {
        uint32_t queueFamilyIndex = findComputeQueueFamily(physicalDevice_);

        VkCommandPoolCreateInfo cpCI{};
        cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpCI.queueFamilyIndex = queueFamilyIndex;
        // We can also set VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT if needed
        cpCI.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        if (vkCreateCommandPool(device_, &cpCI, nullptr, &commandPool_) != VK_SUCCESS) {
            throw std::runtime_error("VulkanContext: Failed to create command pool.");
        }
    }

    /**
     * @brief Find a compute-capable queue family.
     */
    uint32_t findComputeQueueFamily(VkPhysicalDevice pd) const {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
        std::vector<VkQueueFamilyProperties> qProps(count);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, qProps.data());

        int best = -1;
        for (uint32_t i = 0; i < count; i++) {
            if (qProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                // Prefer a pure compute queue if possible
                if ((qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
                    return i;
                }
                if (best < 0)
                    best = i;
            }
        }
        if (best < 0) {
            throw std::runtime_error("VulkanContext: No compute-capable queue family found.");
        }
        return static_cast<uint32_t>(best);
    }

    bool gpuCapable_{false};
    bool initialized_{false};
    std::string initError_{};

    VkInstance instance_{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    uint32_t deviceIndex_{0};
    VkQueue queue_{VK_NULL_HANDLE};
    VkCommandPool commandPool_{VK_NULL_HANDLE};
    VkDeviceMemory device_memory_;
    VkBuffer arenaBuf = VK_NULL_HANDLE;
    VkDeviceMemory arenaMem = VK_NULL_HANDLE;
    std::unique_ptr<PipelineManager> pipelineManager_;

    void* mapped = nullptr;
};

/**
 * @brief Returns the amount of free memory on the GPU.
 */
inline static VkDeviceSize getFreeMem(const VulkanContext& ctx, float mem_percent,
                                      size_t max_size = (4ull * 1024 * 1024 * 1024)) {
    if (!ctx.isGPUCapable()) {
        return 1;
    }

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(ctx.getPhysicalDevice(), &memProps);

    // Find the largest heap that has VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
    // On Apple, you often get a single unified heap. But let's do a fallback:
    uint64_t bestHeapSize = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            bestHeapSize = std::max(bestHeapSize, memProps.memoryHeaps[i].size);
        }
    }

    // If none found, just pick the largest heap as fallback
    if (bestHeapSize == 0) {
        for (uint32_t i = 0; i < memProps.memoryHeapCount; i++) {
            bestHeapSize = std::max(bestHeapSize, memProps.memoryHeaps[i].size);
        }
    }

    return static_cast<VkDeviceSize>(
        std::max(static_cast<size_t>(static_cast<size_t>(bestHeapSize) * mem_percent), max_size));
};

} // namespace psyne