# Psyne Core Design Principles

**THESE PRINCIPLES ARE FUNDAMENTAL AND MUST NEVER BE VIOLATED**

Overall, this follows the philosophy *zero copy at all costs*, as well as "If ZeroMQ were to be rewritten today in modern C++20, what woudl they have done differently?"

Which includes:

- No memory copies *unless there is literally no other way*
- Coroutines intead of state machines
- Concepts for type safety
- std::span and ranges for zero-copy
- Modern memory management (RAII, custom allocators, etc.)
- Native networking TS/ASIO integration
- Compile type configuration
- Better error handling
- Structured bindings and pattern matching
- Better testing with mocks

## 🎯 Core Philosophy: Zero-Copy Message Views

Psyne is built on one fundamental principle: **Messages are views, not objects**.

### The Ring Buffer is King

1. **Channel = Ring Buffer** 
   - Each channel owns exactly one ring buffer (slab)
   - Ring buffer is pre-allocated memory (Starting at 64MB, but will grow if a high water mark is reached)
   - Ring buffers implement ZERO atomics for SPSC (the fastest case):
      - **SPSC**: Producer owns head, consumer owns tail, no atomics needed
      - **MPSC**: Multiple producers need atomic head CAS 
      - **SPMC**: Multiple consumers need atomic tail CAS
      - **MPMC**: Both head and tail need atomic CAS
      
2. **Message Creation = Getting a View**
   - `Message<T> msg(channel)` does NOT allocate memory
   - It gets a pointer to the CURRENT WRITE POSITION in the ring buffer
   - The message IS the ring buffer data at that position
   - **No malloc, no new[], no allocation, no copying**

3. **User Writes Directly to Ring Buffer**
   - User calls `msg.resize(100)` → writes directly to ring buffer
   - User calls `msg[5] = 42.0f` → writes directly to ring buffer  
   - User calls `msg.set_position(x, y, z)` → writes directly to ring buffer
   - **The message object is just a typed view/schema over raw memory**

4. **Send = Advance Pointer + Notify**
   - `msg.send()` simply advances the ring buffer write pointer
   - **No copying, no serialization, no allocation**
   - Boost.Asio notifies listener with (slab_pointer + message_offset)
   - Receiver gets direct pointer arithmetic to the message data
   
5. **Fixed-Size = Zero Metadata**
   - Fixed schemas (e.g., Vector64f) need NO headers, NO size fields
   - Message offset = head_position, message size = compile-time constant
   - Pure data: `receiver_ptr = slab_base + (head * message_size)`

### Message Types = Memory Schemas

```cpp
// This is WRONG - allocating and copying
FloatVector msg;
msg.resize(1000);
for (int i = 0; i < 1000; i++) msg[i] = i * 2.0f;
channel.send(msg);  // COPIES 4KB of data!

// This is RIGHT - direct memory writing
FloatVector msg(channel);  // Just a view into ring buffer
msg.resize(1000);          // Sets size in ring buffer
for (int i = 0; i < 1000; i++) msg[i] = i * 2.0f;  // Writes to ring buffer
msg.send();                // Just advances write pointer
```

### Performance Trade-offs: Fixed vs Dynamic

**🏎️ Fixed-Size = Maximum Performance**

For maximum performance, users should pre-define their message schemas:

```cpp
// 64-dimensional float vector = exactly 256 bytes
class Vector64f : public Message<Vector64f> {
public:
    static constexpr size_t calculate_size() { return 64 * sizeof(float); }
    
    float& operator[](size_t i) { 
        return reinterpret_cast<float*>(data())[i]; 
    }
};

// Usage:
Vector64f msg(channel);  // Points to next 256 bytes in ring buffer
msg[0] = 1.0f;          // Writes directly to ring buffer[write_pos]
msg[1] = 2.0f;          // Writes directly to ring buffer[write_pos + 4]
msg.send();             // Advances write_pos += 256
```

**No metadata, no size fields, no headers - pure data performance!**

**📊 Dynamic-Size = Flexibility with Performance Tax**

If users need variable-size messages, they can use dynamic schemas but pay the performance cost:

```cpp
// Variable-size message - requires size metadata
class DynamicFloatVector : public Message<DynamicFloatVector> {
    struct Header {
        size_t count;  // Number of floats (metadata overhead)
    };
    
public:
    static constexpr size_t calculate_size() { 
        return 64 * 1024;  // Max possible size (wastes memory)
    }
    
    void resize(size_t new_count) {
        header().count = new_count;  // Store size in metadata
    }
    
    float& operator[](size_t i) {
        return reinterpret_cast<float*>(data() + sizeof(Header))[i];
    }
};
```

**Performance Trade-offs:**
- ✅ **Flexible**: Can handle variable-size data
- ❌ **Metadata overhead**: 8 bytes for size field  
- ❌ **Memory waste**: Ring buffer slots sized for worst case
- ❌ **Cache misses**: Size checks, indirection through header
- ❌ **Branching**: Conditional logic for size validation

**Rule: Use fixed-size when possible, dynamic only when necessary.**

## 🎯 Compile-Time Configuration = Maximum Performance

**The fast path must be the default path.** All critical performance parameters should be compile-time constants:

```cpp
// ⚡ FAST PATH (default) - everything compile-time optimized
Channel<Vector64f, SPSC, 1GB> channel;  // Fixed schema, pattern, buffer size
Vector64f msg(channel);                  // Zero-copy view into ring buffer
msg[0] = 1.0f;                          // Direct memory write
msg.send();                             // Just pointer advance + notification

// 🐌 SLOW PATH - clearly marked as performance-degrading  
Channel<DynamicMessage, MPMC, DYNAMIC_SIZE> slow_channel;  // All the overhead
```

**Default Template Parameters:**
```cpp
template<typename MessageType, 
         ChannelPattern Pattern = SPSC,           // Fastest: no atomics
         size_t BufferSize = recommended_size_v<MessageType>,  // Workload-appropriate
         BufferType Type = FIXED_SIZE>            // Fastest: no dynamic allocation
class Channel;

// Separate class for flexibility (clearly performance-degrading)
template<typename MessageType>
class DynamicChannel;  // All the slow features enabled
```

## 🌐 Global Slab Registry & Transport Abstraction

**String addresses are the universal coordination mechanism:**

```cpp
// Same API, different transports based on address resolution
auto channel1 = Channel::get_or_create<GPUFloat32Vector>("tensor_pipeline");     // In-process
auto channel2 = Channel::get_or_create<GPUFloat32Vector>("ipc://tensor_pipeline"); // Inter-process  
auto channel3 = Channel::get_or_create<GPUFloat32Vector>("tcp://host2:5010/tensor_pipeline"); // Network
```

**Transport Selection:**
- **In-process**: Same address space → direct memory access
- **Inter-process**: Shared memory mapping → `/dev/shm/psyne_tensor_pipeline` 
- **Inter-host**: Network coordination → matching GPU buffers + TCP connection

**Multi-Transport Listeners:**
```cpp
// Single logical endpoint accepting from all transport types
auto listener = MultiTransportListener::create<GPUFloat32Vector>("tensor_results");

// Internally creates:
// 1. In-process channel for same-process senders
// 2. Inter-process shared memory for local senders  
// 3. TCP listener for remote senders
// All feeding same GPU buffer

while (auto msg = listener.receive()) {
    // Unified API - same regardless of transport
    process_tensor(msg);  // Zero-copy GPU access
}
```

**Network Transport Strategy:**
- **Sender**: Network layer streams directly from ring buffer (zero-copy from user perspective)
- **Receiver**: Network writes directly to matching buffer type (GPU buffers write to GPU-visible memory)
- **Configuration**: Channel type determines buffer type on both ends

```cpp
// Host 1: Creates 1GB GPU buffer, streams over TCP (default port 5010)
Channel<GPUFloat32Vector, SPSC, 1GB, GPU_BUFFER> sender("tcp://host2:5010/pipeline");

// Host 2: Creates matching 1GB GPU buffer, receives TCP data directly  
Channel<GPUFloat32Vector, SPSC, 1GB, GPU_BUFFER> receiver("tcp://0.0.0.0:5010/pipeline");

// Multiple channels on different ports
Channel<TensorBatch, SPSC, 2GB, GPU_BUFFER> layer1("tcp://host2:5010/layer1");
Channel<TensorBatch, SPSC, 2GB, GPU_BUFFER> layer2("tcp://host2:5011/layer2"); 
Channel<ResultVector, SPSC, 512MB, GPU_BUFFER> results("tcp://host2:5012/results");
```

## 🔄 Backpressure: Head-Hits-Tail

**Simple, deterministic backpressure when ring buffer fills:**

```cpp
uint32_t reserve_write_slot(size_t size) {
    uint64_t current_head = head_.load();
    uint64_t current_tail = tail_.load();
    
    if ((current_head + size - current_tail) >= buffer_size_) {
        return BUFFER_FULL;  // Metrics track this, user adjusts buffer sizes
    }
    return current_head & mask_;
}
```

**Users monitor metrics to tune compile-time buffer sizes for their workloads.**

## 🚫 Anti-Patterns (Never Do This)

### ❌ Memory Allocation in Message Constructors
```cpp
// WRONG - violates zero-copy
Message(Channel& channel) {
    data_ = new uint8_t[size];  // NO!
    memcpy(data_, source, size); // NO!
}
```

### ❌ Copying Data on Send
```cpp
// WRONG - violates zero-copy  
void send() {
    channel.copy_message(data_, size_);  // NO! This method shouldn't exist!
    // or
    channel.commit_write(size_);        // NO! Data is committed when written!
    data_ = nullptr;                    // NO! Message is just a view, no pointers to null!
}
```

### ❌ Dynamic Allocation for Message Data
```cpp
// WRONG - defeats the purpose
std::vector<float> data;  // NO!
data.resize(1000);        // NO!
```

## ✅ Correct Implementation Patterns

### ✅ Message Constructor (Zero-Copy View)
```cpp
template<typename Derived>
Message<Derived>::Message(Channel& channel) 
    : slab_(&channel.get_ring_buffer()), offset_(0), channel_(&channel) {
    
    // Reserve space in ring buffer and get offset - no allocation!
    offset_ = channel.reserve_write_slot(Derived::calculate_size());
    
    if (offset_ == BUFFER_FULL) throw std::runtime_error("Ring buffer full");
    
    // Message is now a typed view over ring buffer at offset
    // User writes directly to slab memory via data() method
}

uint8_t* data() { 
    return slab_->base_ptr() + offset_; 
}
```

### ✅ Send Implementation (Notification Only)
```cpp
template<typename Derived>  
void Message<Derived>::send() {
    // Data is already written by user directly to ring buffer
    // Just notify receiver that there's a message ready at this offset
    channel_->notify_message_ready(offset_, Derived::calculate_size());
    
    // Message object can be destroyed - data lives in ring buffer
    // No pointer nulling needed - message is just a view
}
```

### ✅ Channel Interface (Ring Buffer Operations)
```cpp
class Channel {
    virtual uint32_t reserve_write_slot(size_t size) = 0;     // Reserve space, return offset
    virtual void notify_message_ready(uint32_t offset, size_t size) = 0; // Send notification
    virtual RingBuffer& get_ring_buffer() = 0;               // Get slab reference
    virtual void advance_read_pointer(size_t size) = 0;      // Consumer advances tail
    
    // NO copy_message() method - violates zero-copy!
    // NO commit_write() method - data is committed when written!
};

// Notification structure sent via Boost.Asio
struct MessageNotification {
    uint32_t offset;    // Where in ring buffer
    uint32_t size;      // Message size (or omit for fixed-size)
};
```

## 🎯 Performance Implications

### Why This Matters

1. **CPU Cache Efficiency**: Data written once, read once, never moved
2. **Memory Bandwidth**: Eliminates all memory copies  
3. **CPU Cycles**: No malloc/free overhead
4. **Latency**: Sub-microsecond message passing
5. **Throughput**: 100GB/s+ sustained data rates

### Benchmark Targets
- **Latency**: < 1 microsecond for local IPC
- **Throughput**: > 100 GB/s for large messages  
- **CPU**: < 5% CPU usage at 1M messages/second
- **Memory**: Zero allocations during message passing

## 🔧 Implementation Notes

### Ring Buffer Requirements
- **SPSC Lock-free**: Zero atomics, producer owns head, consumer owns tail
- **Cache-aligned**: 64-byte alignment, head/tail in separate cache lines
- **Power-of-2 sizes**: For efficient modulo operations (bitwise AND)
- **Memory barriers**: Only for MPSC/SPMC/MPMC patterns
- **Boost.Asio integration**: Event notification with slab pointer + offset

### Message Type Requirements
- **Fixed size calculation**: `static constexpr size_t calculate_size()`
- **Direct data access**: Provide typed accessors over `data()` pointer
- **No dynamic allocation**: Never use `new`, `malloc`, `std::vector`, etc.
- **POD-compatible**: Prefer plain-old-data layouts when possible

### Channel Implementation Requirements  
- **Ring buffer per channel**: Each channel has dedicated memory
- **Configurable sizes**: 1KB to 1GB ring buffers
- **Transport-agnostic**: Memory, IPC, TCP, etc. all use same interface
- **Error handling**: Graceful handling of full buffers

## 🏆 The Original v1.0.0 SPSC Design

The record-breaking throughput numbers were achieved with this atomic-free design:

```cpp
// SPSC Ring Buffer - Zero Atomics
struct SPSCRingBuffer {
    uint64_t head;          // Producer-owned (cache line 0)
    uint8_t padding1[56];   // Cache line separation  
    uint64_t tail;          // Consumer-owned (cache line 1)  
    uint8_t padding2[56];   // Cache line separation
    uint8_t* slab;          // The actual data buffer
    size_t mask;            // size - 1 (for power-of-2 modulo)
};

// Producer (write side) - NO ATOMICS
void write_message(Vector64f& msg) {
    uint64_t pos = head;
    Vector64f* slot = (Vector64f*)(slab + (pos & mask) * sizeof(Vector64f));
    *slot = msg;           // Direct memory write (256 bytes)
    head++;                // Advance head (no atomic needed)
    asio_notify(slab, pos); // Notify consumer via Boost.Asio
}

// Consumer (read side) - NO ATOMICS  
Vector64f* read_message() {
    uint64_t pos = tail;
    if (pos >= head) return nullptr;  // No new messages
    Vector64f* slot = (Vector64f*)(slab + (pos & mask) * sizeof(Vector64f)); 
    tail++;                // Advance tail (no atomic needed)
    return slot;           // Direct pointer to data
}

// Note: In some designs, tail might need to be atomic if producer
// needs to check for buffer full condition, but this can often be
// avoided with careful buffer sizing or alternative flow control
```

**Why this is so fast:**
- **Zero atomic operations** in the critical path
- **Zero memory copies** - direct pointer arithmetic  
- **Zero allocations** - pre-allocated slab
- **Zero serialization** - raw memory layout
- **Optimal cache usage** - head/tail in separate cache lines

## 📜 Historical Context

This design emerged from the realization that traditional message queues:
1. **Over-serialize**: Convert objects to bytes unnecessarily  
2. **Over-copy**: Move data multiple times through the stack
3. **Over-allocate**: Create temporary objects for each message
4. **Under-perform**: Achieve only ~1-10% of theoretical bandwidth

Psyne eliminates ALL of these inefficiencies by treating messages as **typed views over pre-allocated ring buffer memory**.

## 🛡️ Design Preservation

**These principles must be protected during:**
- Adding new features (dynamic sizes, compression, etc.)
- Refactoring for new platforms (Windows, embedded, etc.)  
- Performance optimizations (SIMD, GPU, etc.)
- API improvements (async, templating, etc.)

**Any change that violates zero-copy semantics should be rejected.**

## 🏗️ Custom Message Types: User-Defined Schemas

**Users can define their own message types by inheriting from Message<T>:**

### ⚡ Fixed-Size (Maximum Performance)
```cpp
// Custom structured message - fixed size, maximum performance
class TaskRecord : public Message<TaskRecord> {
public:
    static constexpr size_t NAME_SIZE = 64;
    static constexpr size_t calculate_size() { 
        return sizeof(uint64_t) + sizeof(uint32_t) + NAME_SIZE; // 76 bytes
    }
    
    // Direct typed access to ring buffer memory
    uint64_t& task_id() { 
        return *reinterpret_cast<uint64_t*>(data()); 
    }
    
    uint32_t& hash() { 
        return *reinterpret_cast<uint32_t*>(data() + sizeof(uint64_t)); 
    }
    
    char* name() { 
        return reinterpret_cast<char*>(data() + sizeof(uint64_t) + sizeof(uint32_t)); 
    }
    
    void set_name(const char* new_name) {
        strncpy(name(), new_name, NAME_SIZE - 1);
        name()[NAME_SIZE - 1] = '\0';  // Ensure null termination
    }
};

// Usage - maximum performance
Channel<TaskRecord, SPSC, 256MB> channel("task_queue");
TaskRecord msg(channel);     // Points directly to 76-byte slot in ring buffer
msg.task_id() = 12345;       // Direct write to ring buffer
msg.hash() = 0xABCD;         // Direct write to ring buffer  
msg.set_name("process_data"); // Direct write to ring buffer
msg.send();                  // Just advances pointer
```

### 📊 Dynamic-Size (Flexible with Runtime Allocation)
```cpp
// Truly dynamic message - allocates exact size needed
class DynamicXMLMessage : public DynamicMessage<DynamicXMLMessage> {
    struct Header {
        uint32_t xml_length;     // Actual XML size
        uint32_t padding;        // 8-byte alignment
    };
    
public:
    // Constructor takes actual size needed
    DynamicXMLMessage(Channel& channel, const char* xml_data) 
        : DynamicMessage(channel, sizeof(Header) + strlen(xml_data) + 1) {
        
        size_t len = strlen(xml_data);
        header().xml_length = len;
        memcpy(xml_data_ptr(), xml_data, len);
        xml_data_ptr()[len] = '\0';
        
        // Message is now IMMUTABLE - enforce at send()
        finalize();  // Locks message from further modification
    }
    
    const char* xml_data() const { return xml_data_ptr(); }
    size_t xml_length() const { return header().xml_length; }
    
private:
    Header& header() { return *reinterpret_cast<Header*>(data()); }
    const Header& header() const { return *reinterpret_cast<const Header*>(data()); }
    const char* xml_data_ptr() const { 
        return reinterpret_cast<const char*>(data() + sizeof(Header)); 
    }
};

// Built-in JSON support using nlohmann::json
class JSONMessage : public DynamicMessage<JSONMessage> {
    struct Header {
        uint32_t json_length;
        uint32_t padding;
    };
    
public:
    JSONMessage(Channel& channel, const nlohmann::json& json_obj) 
        : DynamicMessage(channel, sizeof(Header) + json_obj.dump().size() + 1) {
        
        std::string json_str = json_obj.dump();
        header().json_length = json_str.size();
        memcpy(json_data_ptr(), json_str.c_str(), json_str.size());
        json_data_ptr()[json_str.size()] = '\0';
        
        finalize();  // IMMUTABLE after construction - no json modifications allowed
    }
    
    nlohmann::json to_json() const {
        return nlohmann::json::parse(json_data_ptr());  // Parse on read
    }
    
private:
    Header& header() { return *reinterpret_cast<Header*>(data()); }
    const Header& header() const { return *reinterpret_cast<const Header*>(data()); }
    const char* json_data_ptr() const { 
        return reinterpret_cast<const char*>(data() + sizeof(Header)); 
    }
};

// Usage - allocates exact size, enforces immutability
Channel<DynamicXMLMessage, SPSC, 512MB> channel("config_updates"); 
DynamicXMLMessage msg(channel, "<config><param>value</param></config>");
// msg is now IMMUTABLE - cannot modify XML content
msg.send();  // Exact size allocated, no waste
```

### 🎯 Built-in Types (Optimized Common Cases)
```cpp
// Psyne provides optimized built-ins for common use cases
using Float32Tensor = Tensor<float, 3, 224, 224>;  // 3D tensor: 3x224x224
using JSONMessage = JSON<4096>;                     // Fixed-size JSON buffer
using GPUFloat32Vector = GPUVector<float, 1024>;   // GPU-accessible vector
using BinaryBlob = Blob<1024>;                     // Raw binary data

// All follow same Message<T> interface for consistency
```

---

*"The fastest code is the code that never runs. The fastest memory copy is the copy that never happens."* - Psyne Philosophy