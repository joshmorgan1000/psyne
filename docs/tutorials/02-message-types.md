# Tutorial 2: Message Types and Patterns

This tutorial covers Psyne's message type system and common messaging patterns.

## Built-in Message Types

Psyne provides several built-in message types optimized for different use cases:

### FloatVector

Dynamic array of floats, perfect for ML tensors and time series data:

```cpp
FloatVector msg(*channel);
msg.resize(100);

// Fill with data
for (size_t i = 0; i < 100; ++i) {
    msg[i] = std::sin(i * 0.1f);
}

// STL-like interface
std::cout << "Size: " << msg.size() << std::endl;
std::cout << "First: " << msg[0] << std::endl;
std::cout << "Last: " << msg[msg.size()-1] << std::endl;

// Range-based for loop
float sum = 0;
for (float value : msg) {
    sum += value;
}
```

### ByteVector

Raw byte array for binary data:

```cpp
ByteVector msg(*channel);
msg.resize(1024);

// Copy binary data
std::memcpy(msg.data(), binary_data, data_size);

// Access as different types
auto* as_ints = msg.as<int32_t>();
auto* as_doubles = msg.as<double>();
```

### Enhanced Matrix Types

Fixed-size matrices for computer vision and robotics:

```cpp
#include <psyne/psyne.hpp>
using namespace psyne::types;

// 4x4 transformation matrix
Matrix4x4f transform(*channel);
transform.identity();
transform(0, 3) = 10.0f;  // X translation
transform(1, 3) = 20.0f;  // Y translation
transform(2, 3) = 30.0f;  // Z translation

std::cout << "Determinant: " << transform.determinant() << std::endl;

// 3D vectors
Vector3f position(*channel);
position.x() = 1.0f;
position.y() = 2.0f;
position.z() = 3.0f;
std::cout << "Length: " << position.length() << std::endl;
```

### Quantized Types

For efficient neural network inference:

```cpp
// 8-bit quantized values
Int8Vector quantized(*channel);
quantized.resize(1000);
quantized.set_scale(0.1f);
quantized.set_zero_point(128);

// Quantize float values
for (size_t i = 0; i < 1000; ++i) {
    float value = model_output[i];
    quantized[i] = quantized.quantize(value);
}

// Dequantize back
float original = quantized.dequantize(quantized[0]);
```

### Complex Numbers

For signal processing:

```cpp
ComplexVectorF signal(*channel);
signal.resize(1024);

// Fill with complex exponential
for (size_t i = 0; i < signal.size(); ++i) {
    float phase = 2.0f * M_PI * i / signal.size();
    signal[i] = std::complex<float>(std::cos(phase), std::sin(phase));
}

// Compute power
float power = signal.power();
```

### ML Tensors

Multi-dimensional tensors with layout support:

```cpp
MLTensorF tensor(*channel);
tensor.reshape({32, 3, 224, 224});  // Batch, Channels, Height, Width
tensor.set_layout(MLTensorF::Layout::NCHW);

// Access elements
size_t batch = 0, channel = 1, y = 100, x = 100;
size_t index = tensor.index({batch, channel, y, x});
tensor.data()[index] = 1.0f;

// Get strides for iteration
auto strides = tensor.strides();
```

## Creating Custom Message Types

You can create your own message types for domain-specific data:

```cpp
class SensorReading : public Message<SensorReading> {
public:
    static constexpr uint32_t message_type = 1000;  // Unique ID
    
    using Message<SensorReading>::Message;
    
    // Define the size of your message
    static size_t calculate_size() {
        return sizeof(Data);
    }
    
    // Data structure
    struct Data {
        float temperature;
        float pressure;
        float humidity;
        uint64_t timestamp;
        uint32_t sensor_id;
        uint32_t flags;
    };
    
    // Accessors
    Data& data() { return *reinterpret_cast<Data*>(Message::data()); }
    const Data& data() const { return *reinterpret_cast<const Data*>(Message::data()); }
    
    // Convenience methods
    void set_reading(float temp, float press, float humid) {
        data().temperature = temp;
        data().pressure = press;
        data().humidity = humid;
        data().timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    }
    
    // Initialize the message
    void initialize() {
        std::memset(Message::data(), 0, sizeof(Data));
    }
};

// Usage
SensorReading reading(*channel);
reading.set_reading(25.5f, 1013.25f, 60.0f);
reading.data().sensor_id = 42;
channel->send(reading);
```

## Message Patterns

### Request-Response Pattern

Using two channels for bidirectional communication:

```cpp
// Create request and response channels
auto request_channel = create_channel("memory://requests", 1024*1024);
auto response_channel = create_channel("memory://responses", 1024*1024);

// Client side
std::thread client([&]() {
    // Send request
    FloatVector request(*request_channel);
    request.resize(2);
    request[0] = 3.0f;  // a
    request[1] = 4.0f;  // b
    request_channel->send(request);
    
    // Wait for response
    auto response = response_channel->receive<FloatVector>();
    if (response) {
        std::cout << "Result: " << (*response)[0] << std::endl;
    }
});

// Server side
std::thread server([&]() {
    auto request = request_channel->receive<FloatVector>();
    if (request) {
        float a = (*request)[0];
        float b = (*request)[1];
        
        // Compute response
        FloatVector response(*response_channel);
        response.resize(1);
        response[0] = std::sqrt(a*a + b*b);  // Pythagorean theorem
        response_channel->send(response);
    }
});
```

### Streaming Pattern

Continuous data streaming with flow control:

```cpp
class StreamProcessor {
    Channel& input_;
    Channel& output_;
    std::atomic<bool> running_{true};
    
public:
    StreamProcessor(Channel& input, Channel& output) 
        : input_(input), output_(output) {}
    
    void process() {
        while (running_) {
            // Receive batch
            auto batch = input_.receive<FloatVector>();
            if (!batch) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            // Process data
            FloatVector result(output_);
            result.resize(batch->size());
            
            // Apply transformation
            for (size_t i = 0; i < batch->size(); ++i) {
                result[i] = std::tanh((*batch)[i]);  // Example: activation function
            }
            
            // Send result
            output_.send(result);
        }
    }
    
    void stop() { running_ = false; }
};
```

### Scatter-Gather Pattern

Distribute work and collect results:

```cpp
// Create worker channels
std::vector<ChannelPtr> worker_channels;
for (int i = 0; i < num_workers; ++i) {
    worker_channels.push_back(
        create_channel("memory://worker" + std::to_string(i), 1024*1024));
}

// Result collection channel
auto result_channel = create_channel("memory://results", 1024*1024, ChannelMode::MPSC);

// Scatter work
std::vector<std::thread> workers;
for (int i = 0; i < num_workers; ++i) {
    workers.emplace_back([i, &worker_channels, &result_channel]() {
        auto& my_channel = worker_channels[i];
        
        while (true) {
            auto work = my_channel->receive<FloatVector>();
            if (!work) break;
            
            // Process work
            FloatVector result(*result_channel);
            result.resize(1);
            result[0] = process_data(*work);
            
            // Send result
            result_channel->send(result);
        }
    });
}

// Distribute work
for (int i = 0; i < num_tasks; ++i) {
    int worker_id = i % num_workers;
    FloatVector task(*worker_channels[worker_id]);
    // ... fill task data ...
    worker_channels[worker_id]->send(task);
}

// Gather results
for (int i = 0; i < num_tasks; ++i) {
    auto result = result_channel->receive<FloatVector>();
    // ... process result ...
}
```

### Pipeline Pattern

Process data through multiple stages:

```cpp
class Pipeline {
    std::vector<ChannelPtr> channels_;
    std::vector<std::thread> stages_;
    
public:
    Pipeline(int num_stages, size_t buffer_size) {
        // Create channels between stages
        for (int i = 0; i <= num_stages; ++i) {
            channels_.push_back(
                create_channel("memory://stage" + std::to_string(i), buffer_size));
        }
    }
    
    template<typename Func>
    void add_stage(Func process_func) {
        int stage_id = stages_.size();
        stages_.emplace_back([this, stage_id, process_func]() {
            auto& input = channels_[stage_id];
            auto& output = channels_[stage_id + 1];
            
            while (true) {
                auto msg = input->receive<FloatVector>();
                if (!msg) break;
                
                FloatVector result(*output);
                process_func(*msg, result);
                output->send(result);
            }
        });
    }
    
    Channel& input() { return *channels_.front(); }
    Channel& output() { return *channels_.back(); }
};

// Usage
Pipeline pipeline(3, 1024*1024);

// Stage 1: Normalize
pipeline.add_stage([](const FloatVector& in, FloatVector& out) {
    out.resize(in.size());
    float sum = 0, sum_sq = 0;
    for (float v : in) {
        sum += v;
        sum_sq += v * v;
    }
    float mean = sum / in.size();
    float std = std::sqrt(sum_sq / in.size() - mean * mean);
    
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = (in[i] - mean) / std;
    }
});

// Stage 2: Apply activation
pipeline.add_stage([](const FloatVector& in, FloatVector& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = std::tanh(in[i]);
    }
});

// Stage 3: Scale
pipeline.add_stage([](const FloatVector& in, FloatVector& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = in[i] * 2.0f;
    }
});
```

## Best Practices

### 1. Choose the Right Message Type

- **FloatVector**: Variable-size numerical data
- **ByteVector**: Binary blobs, serialized data
- **Fixed-size types**: Known dimensions, better performance
- **Custom types**: Domain-specific, type-safe interfaces

### 2. Message Sizing

```cpp
// Check capacity before resizing
FloatVector msg(*channel);
size_t needed = calculate_needed_size();
if (needed > msg.capacity()) {
    // Handle error - can't resize beyond capacity
    std::cerr << "Message too large: need " << needed 
              << ", capacity " << msg.capacity() << std::endl;
    return;
}
msg.resize(needed);
```

### 3. Batching for Performance

```cpp
// Instead of sending one item at a time
for (const auto& item : items) {
    FloatVector msg(*channel);
    msg.resize(1);
    msg[0] = item;
    channel->send(msg);  // Overhead for each send
}

// Batch multiple items
FloatVector batch(*channel);
size_t batch_size = std::min(items.size(), batch.capacity());
batch.resize(batch_size);
for (size_t i = 0; i < batch_size; ++i) {
    batch[i] = items[i];
}
channel->send(batch);  // One send for entire batch
```

### 4. Error Handling

```cpp
// Always check message validity
FloatVector msg(*channel);
if (!msg.is_valid()) {
    // Channel might be full
    return false;
}

// Always check receive result
auto received = channel->receive<FloatVector>();
if (!received) {
    // No message available
    return false;
}

// Safe to use
process(*received);
```

## Next Steps

- Tutorial 3: Inter-Process Communication
- Tutorial 4: Network Channels
- Tutorial 5: Performance Optimization

## Exercises

1. Create a custom message type for RGB images with width, height, and pixel data
2. Implement a fan-out pattern where one producer sends to multiple consumers
3. Build a simple calculator using request-response pattern
4. Create a pipeline that filters, transforms, and aggregates sensor data