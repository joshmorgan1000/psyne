/**
 * @file arrow_demo.cpp
 * @brief Basic demonstration of Apache Arrow functionality
 *
 * This example shows basic Arrow operations that could be integrated
 * with Psyne for high-performance data transport. Future versions
 * could implement Arrow Flight protocol using Psyne's zero-copy messaging.
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <vector>

using namespace psyne;

// Generate sample data using Arrow
std::shared_ptr<arrow::RecordBatch> GenerateSampleData(int64_t num_rows) {
    // Create schema
    auto schema = arrow::schema({
        arrow::field("id", arrow::int64()),
        arrow::field("temperature", arrow::float32()),
        arrow::field("humidity", arrow::float32()),
        arrow::field("sensor_name", arrow::utf8())
    });

    // Create builders
    arrow::Int64Builder id_builder;
    arrow::FloatBuilder temp_builder;
    arrow::FloatBuilder humidity_builder;
    arrow::StringBuilder name_builder;

    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> temp_dist(15.0f, 35.0f);
    std::uniform_real_distribution<float> humidity_dist(20.0f, 80.0f);
    std::uniform_int_distribution<int> sensor_dist(0, 4);

    std::vector<std::string> sensor_names = {"sensor_a", "sensor_b", "sensor_c", 
                                             "sensor_d", "sensor_e"};

    // Generate data (ignore return values for this demo)
    for (int64_t i = 0; i < num_rows; ++i) {
        (void)id_builder.Append(i);
        (void)temp_builder.Append(temp_dist(gen));
        (void)humidity_builder.Append(humidity_dist(gen));
        (void)name_builder.Append(sensor_names[sensor_dist(gen)]);
    }

    // Build arrays
    std::shared_ptr<arrow::Array> id_array;
    std::shared_ptr<arrow::Array> temp_array;
    std::shared_ptr<arrow::Array> humidity_array;
    std::shared_ptr<arrow::Array> name_array;

    (void)id_builder.Finish(&id_array);
    (void)temp_builder.Finish(&temp_array);
    (void)humidity_builder.Finish(&humidity_array);
    (void)name_builder.Finish(&name_array);

    // Create record batch
    std::vector<std::shared_ptr<arrow::Array>> columns = {
        id_array, temp_array, humidity_array, name_array
    };
    
    return arrow::RecordBatch::Make(schema, num_rows, columns);
}

// Demonstrate basic Arrow operations
void DemoBasicArrow() {
    std::cout << "=== Basic Arrow Demo ===" << std::endl;
    
    // Generate sample data
    auto batch = GenerateSampleData(1000);
    std::cout << "Generated batch with " << batch->num_rows() << " rows and " 
              << batch->num_columns() << " columns" << std::endl;
    
    // Print schema
    std::cout << "Schema: " << batch->schema()->ToString() << std::endl;
    
    // Show first few rows
    std::cout << "\nFirst 5 rows:" << std::endl;
    auto sliced = batch->Slice(0, 5);
    std::cout << sliced->ToString() << std::endl;
}

// Demonstrate Arrow compute operations
void DemoArrowCompute() {
    std::cout << "\n=== Arrow Compute Demo ===" << std::endl;
    
    auto batch = GenerateSampleData(1000);
    
    // Get temperature column (index 1)
    auto temp_array = batch->column(1);
    auto humidity_array = batch->column(2);
    
    // Calculate statistics
    auto mean_result = arrow::compute::Mean(temp_array);
    if (mean_result.ok()) {
        auto mean_scalar = std::static_pointer_cast<arrow::DoubleScalar>(
            mean_result.ValueOrDie().scalar());
        std::cout << "Mean temperature: " << mean_scalar->value << "°C" << std::endl;
    }
    
    mean_result = arrow::compute::Mean(humidity_array);
    if (mean_result.ok()) {
        auto mean_scalar = std::static_pointer_cast<arrow::DoubleScalar>(
            mean_result.ValueOrDie().scalar());
        std::cout << "Mean humidity: " << mean_scalar->value << "%" << std::endl;
    }
    
    // Calculate min/max
    auto minmax_result = arrow::compute::MinMax(temp_array);
    if (minmax_result.ok()) {
        auto minmax_scalar = std::static_pointer_cast<arrow::StructScalar>(
            minmax_result.ValueOrDie().scalar());
        auto min_val = std::static_pointer_cast<arrow::FloatScalar>(minmax_scalar->value[0]);
        auto max_val = std::static_pointer_cast<arrow::FloatScalar>(minmax_scalar->value[1]);
        std::cout << "Temperature range: " << min_val->value << "°C to " 
                  << max_val->value << "°C" << std::endl;
    }
}

// Demonstrate zero-copy potential with Psyne
void DemoZeroCopyPotential() {
    std::cout << "\n=== Zero-Copy Integration Potential ===" << std::endl;
    
    // Create a simple Psyne ByteVector
    auto channel = create_channel("memory://arrow_demo", 1024 * 1024);
    ByteVector vec(*channel);
    vec.resize(1000);
    
    // Fill with sample data
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<uint8_t>(i % 256);
    }
    
    std::cout << "Created Psyne ByteVector with " << vec.size() << " bytes" << std::endl;
    std::cout << "Data pointer: " << static_cast<void*>(vec.data()) << std::endl;
    std::cout << "First 10 values: ";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(vec[i]) << " ";
    }
    std::cout << std::endl;
    
    // In a future implementation, we could create an Arrow buffer
    // that directly uses the Psyne memory without copying:
    // auto buffer = std::make_shared<arrow::Buffer>(vec.data(), vec.size());
    
    std::cout << "\nFuture integration possibilities:" << std::endl;
    std::cout << "1. Zero-copy Arrow buffers using Psyne memory" << std::endl;
    std::cout << "2. Arrow Flight protocol implementation over Psyne channels" << std::endl;
    std::cout << "3. Tensor transport using Arrow's tensor format" << std::endl;
}

// Demonstrate Arrow serialization (IPC format)
void DemoArrowSerialization() {
    std::cout << "\n=== Arrow Serialization Demo ===" << std::endl;
    
    auto batch = GenerateSampleData(100);
    
    // Create output stream
    std::shared_ptr<arrow::io::BufferOutputStream> stream;
    auto stream_result = arrow::io::BufferOutputStream::Create();
    if (!stream_result.ok()) {
        std::cerr << "Failed to create output stream" << std::endl;
        return;
    }
    stream = stream_result.ValueOrDie();
    
    // Create IPC writer
    std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
    auto writer_result = arrow::ipc::MakeStreamWriter(stream, batch->schema());
    if (!writer_result.ok()) {
        std::cerr << "Failed to create IPC writer" << std::endl;
        return;
    }
    writer = writer_result.ValueOrDie();
    
    // Write batch
    auto write_status = writer->WriteRecordBatch(*batch);
    if (!write_status.ok()) {
        std::cerr << "Failed to write record batch" << std::endl;
        return;
    }
    
    // Close writer
    auto close_status = writer->Close();
    if (!close_status.ok()) {
        std::cerr << "Failed to close writer" << std::endl;
        return;
    }
    
    // Get serialized data
    auto buffer_result = stream->Finish();
    if (!buffer_result.ok()) {
        std::cerr << "Failed to finish stream" << std::endl;
        return;
    }
    auto buffer = buffer_result.ValueOrDie();
    
    std::cout << "Serialized " << batch->num_rows() << " rows to " 
              << buffer->size() << " bytes" << std::endl;
    std::cout << "Compression ratio: " 
              << (batch->num_rows() * batch->num_columns() * 8) / static_cast<double>(buffer->size())
              << ":1 (estimated)" << std::endl;
    
    std::cout << "\nThis serialized data could be sent over Psyne channels" << std::endl;
    std::cout << "for efficient Arrow data transport!" << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        std::string mode = argv[1];
        
        if (mode == "basic") {
            DemoBasicArrow();
        } else if (mode == "compute") {
            DemoArrowCompute();
        } else if (mode == "zerocopy") {
            DemoZeroCopyPotential();
        } else if (mode == "serialize") {
            DemoArrowSerialization();
        } else {
            std::cerr << "Usage: " << argv[0] 
                      << " [basic|compute|zerocopy|serialize]" << std::endl;
            return 1;
        }
    } else {
        // Run all demos
        DemoBasicArrow();
        DemoArrowCompute();
        DemoZeroCopyPotential();
        DemoArrowSerialization();
    }
    
    return 0;
}