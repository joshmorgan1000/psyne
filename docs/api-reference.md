# Psyne API Reference

## Core Types

### Channel<RingBufferType>

The main communication primitive. Template parameter determines the synchronization model.

```cpp
template<typename RingBufferType>
class Channel {
public:
    // Constructor
    Channel(const std::string& uri, size_t buffer_size, 
            ChannelType type = ChannelType::MultiType);
    
    // Message creation (zero allocation)
    template<typename MessageType>
    MessageType create_message();
    
    // Send a message
    template<typename MessageType>
    void send(MessageType& msg);
    
    // Receive for single-type channels
    template<typename MessageType>
    std::optional<MessageType> receive_single(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    
    // Receive for multi-type channels
    std::optional<std::pair<uint32_t, void*>> receive_multi(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    
    // Type-safe receive for multi-type channels
    template<typename MessageType>
    std::optional<MessageType> receive_as(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    
    // Event-driven listening
    template<typename MessageType>
    std::unique_ptr<std::thread> listen(
        std::function<void(MessageType&&)> handler);
    
    // Multi-type event listening
    std::unique_ptr<std::thread> listen(
        std::unordered_map<uint32_t, std::function<void(void*)>> handlers);
    
    // Control
    void stop();
    void notify();
    
    // Properties
    RingBufferType* ring_buffer();
    const std::string& uri() const;
    ChannelType type() const;
};
```

### Pre-defined Channel Types

```cpp
using SPSCChannel = Channel<SPSCRingBuffer>;  // Single Producer, Single Consumer
using SPMCChannel = Channel<SPMCRingBuffer>;  // Single Producer, Multi Consumer
using MPSCChannel = Channel<MPSCRingBuffer>;  // Multi Producer, Single Consumer
using MPMCChannel = Channel<MPMCRingBuffer>;  // Multi Producer, Multi Consumer
```

### Message<Derived>

Base class for all message types. Provides zero-copy view semantics.

```cpp
template<typename Derived>
class Message {
public:
    // Create outgoing message (allocates in channel)
    template<typename Channel>
    explicit Message(Channel& channel);
    
    // Create incoming message (view of existing data)
    explicit Message(void* data, size_t size);
    
    // Send the message
    void send();
    
    // Check validity
    bool is_valid() const;
    
    // Get message type ID
    static constexpr uint32_t type();
};
```

### Pre-defined Message Types

#### FloatVector

Dynamic-size array of floats.

```cpp
class FloatVector : public Message<FloatVector> {
public:
    static constexpr uint32_t message_type = 1;
    
    // Assignment from initializer list
    FloatVector& operator=(std::initializer_list<float> values);
    
    // Array access
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    
    // STL interface
    float* begin();
    float* end();
    const float* begin() const;
    const float* end() const;
    
    // Size management
    size_t size() const;
    size_t capacity() const;
    void resize(size_t new_size);
};
```

#### DoubleMatrix

2D matrix of doubles.

```cpp
class DoubleMatrix : public Message<DoubleMatrix> {
public:
    static constexpr uint32_t message_type = 2;
    
    // Dimension management
    void set_dimensions(size_t rows, size_t cols);
    size_t rows() const;
    size_t cols() const;
    
    // Element access
    double& at(size_t row, size_t col);
    const double& at(size_t row, size_t col) const;
};
```

### Dynamic Memory Management

#### DynamicSlabAllocator

Automatically grows memory slabs based on usage patterns.

```cpp
class DynamicSlabAllocator {
public:
    struct Config {
        size_t initial_slab_size = 1024 * 1024;      // 1MB initial
        size_t max_slab_size = 128 * 1024 * 1024;    // 128MB max per slab
        size_t growth_factor = 2;                     // Double size on growth
        double growth_threshold = 0.75;               // Grow when 75% full
        double shrink_threshold = 0.25;               // Shrink when <25% used
        std::chrono::seconds cleanup_interval = 60s;  // Cleanup check interval
    };
    
    explicit DynamicSlabAllocator(const Config& config = {});
    
    // Allocate memory, growing if necessary
    void* allocate(size_t size);
    
    // Get current statistics
    Stats get_stats() const;
    
    // Force a cleanup check
    void cleanup();
    
    // Get recommended ring buffer size based on usage
    size_t get_recommended_buffer_size() const;
};
```

#### DynamicRingBuffer<ProducerType, ConsumerType>

Ring buffer that automatically resizes based on usage patterns.

```cpp
template<typename ProducerType, typename ConsumerType>
class DynamicRingBuffer {
public:
    struct Config {
        size_t initial_size = 64 * 1024;             // 64KB initial
        size_t min_size = 4 * 1024;                  // 4KB minimum
        size_t max_size = 128 * 1024 * 1024;         // 128MB maximum
        double resize_up_threshold = 0.9;            // Resize up when 90% full
        double resize_down_threshold = 0.1;          // Resize down when <10% used
        size_t resize_factor = 2;                    // Double/halve on resize
        std::chrono::seconds resize_check_interval = 5s; // Check interval
        size_t high_water_mark_window = 1000;        // Usage history window
    };
    
    explicit DynamicRingBuffer(const Config& config = {});
    
    // Same interface as RingBuffer
    std::optional<WriteHandle> reserve(size_t size);
    std::optional<ReadHandle> read();
    bool empty() const;
    
    // Get current statistics
    Stats get_stats() const;
    
    // Force a resize check
    void check_resize();
};

// Pre-defined dynamic buffer types
using DynamicSPSCRingBuffer = DynamicRingBuffer<SingleProducer, SingleConsumer>;
using DynamicSPMCRingBuffer = DynamicRingBuffer<SingleProducer, MultiConsumer>;
using DynamicMPSCRingBuffer = DynamicRingBuffer<MultiProducer, SingleConsumer>;
using DynamicMPMCRingBuffer = DynamicRingBuffer<MultiProducer, MultiConsumer>;
```

### VariantView<T>

Zero-copy view of typed data within a message.

```cpp
template<typename T>
class VariantView {
public:
    // Data access
    T* data();
    const T* data() const;
    
    // Size information
    size_t size() const;
    bool empty() const;
    
    // Array access
    T& operator[](size_t idx);
    const T& operator[](size_t idx) const;
    
    // Bounds-checked access
    T& at(size_t idx);
    const T& at(size_t idx) const;
    
    // STL interface
    std::span<T> as_span();
    std::span<const T> as_span() const;
    
    T* begin();
    T* end();
    const T* begin() const;
    const T* end() const;
    
    // Type queries
    bool is_scalar() const;
    bool is_array() const;
    bool is_gpu_buffer() const;
    bool is_readonly() const;
};
```

## Enumerations

### ChannelType

```cpp
enum class ChannelType {
    SingleType,   // Optimized for single message type (no metadata)
    MultiType     // Supports multiple types (8-byte overhead)
};
```

### ChannelMode

```cpp
enum class ChannelMode {
    SPSC,  // Single Producer, Single Consumer
    SPMC,  // Single Producer, Multi Consumer
    MPSC,  // Multi Producer, Single Consumer
    MPMC   // Multi Producer, Multi Consumer
};
```

### VariantType

```cpp
enum class VariantType : uint8_t {
    None = 0,
    Float32 = 1,
    Float64 = 2,
    Int8 = 3,
    Int16 = 4,
    Int32 = 5,
    Int64 = 6,
    Uint8 = 7,
    Uint16 = 8,
    Uint32 = 9,
    Uint64 = 10,
    Float32Array = 11,
    Float64Array = 12,
    Int8Array = 13,
    Int32Array = 14,
    Custom = 255
};
```

### VariantFlags

```cpp
enum class VariantFlags : uint8_t {
    None = 0,
    GpuBuffer = 1 << 0,
    Readonly = 1 << 1,
    Quantized = 1 << 2,
    Compressed = 1 << 3
};
```

## Custom Message Types

To define your own message type:

```cpp
class MyMessage : public Message<MyMessage> {
public:
    static constexpr uint32_t message_type = 1001;  // Choose unique ID >= 1000
    
    using Message::Message;  // Inherit constructors
    
    // Add your data access methods
    void set_data(/* ... */);
    auto get_data() const;
    
    // Required: Calculate size at compile time
    static constexpr size_t calculate_size() {
        return /* your calculation */;
    }
    
private:
    friend class Message<MyMessage>;
    
    // Initialize storage for outgoing messages
    void initialize_storage(void* ptr) {
        // Set up your data layout
    }
    
    // Initialize view for incoming messages
    void initialize_view(void* ptr) {
        // Set up your view pointers
    }
};
```

## Memory Management

### SlabAllocator

Low-level memory allocator (usually not used directly).

```cpp
class SlabAllocator {
public:
    explicit SlabAllocator(size_t slab_size);
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    
    size_t available() const;
    size_t capacity() const;
    
    void* base();
    const void* base() const;
};
```

### Ring Buffer Types

Lock-free ring buffers with different synchronization guarantees.

```cpp
// All ring buffers share this interface
class RingBuffer {
public:
    struct WriteHandle {
        void* data;
        size_t size;
        void commit();
    };
    
    struct ReadHandle {
        const void* data;
        size_t size;
    };
    
    std::optional<WriteHandle> reserve(size_t size);
    std::optional<ReadHandle> read();
    
    bool empty() const;
    size_t capacity() const;
    void* base();
};
```

## Error Handling

Psyne uses exceptions for initialization errors and return types for runtime errors:

- Constructor failures throw `std::runtime_error` or `std::bad_alloc`
- Message creation returns invalid messages (check with `is_valid()`)
- Receive operations return `std::optional<T>`
- Send operations on invalid messages are no-ops

## Thread Safety

- SPSC: Wait-free for single producer/consumer
- MPSC: Lock-free for multiple producers
- SPMC: Lock-free for multiple consumers  
- MPMC: Lock-free with CAS operations
- All message types are move-only (no accidental sharing)