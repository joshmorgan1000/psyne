#include <psyne/psyne.hpp>
#include "../src/arrow/arrow_integration.hpp"
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <iostream>
#include <thread>
#include <random>

using namespace psyne;
using namespace psyne::arrow_integration;

// Generate sample data
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
    
    std::vector<std::string> sensor_names = {
        "sensor_a", "sensor_b", "sensor_c", "sensor_d", "sensor_e"
    };
    
    // Generate data
    for (int64_t i = 0; i < num_rows; ++i) {
        id_builder.Append(i);
        temp_builder.Append(temp_dist(gen));
        humidity_builder.Append(humidity_dist(gen));
        name_builder.Append(sensor_names[sensor_dist(gen)]);
    }
    
    // Build arrays
    std::shared_ptr<arrow::Array> id_array;
    std::shared_ptr<arrow::Array> temp_array;
    std::shared_ptr<arrow::Array> humidity_array;
    std::shared_ptr<arrow::Array> name_array;
    
    id_builder.Finish(&id_array);
    temp_builder.Finish(&temp_array);
    humidity_builder.Finish(&humidity_array);
    name_builder.Finish(&name_array);
    
    // Create record batch
    return arrow::RecordBatch::Make(schema, num_rows,
                                   {id_array, temp_array, humidity_array, name_array});
}

// Data producer using Arrow
void RunProducer() {
    std::cout << "Starting Arrow data producer..." << std::endl;
    
    // Create Arrow channel
    ArrowChannel channel("memory://sensor_data", 32 * 1024 * 1024);
    
    // Generate and send data
    for (int i = 0; i < 10; ++i) {
        auto batch = GenerateSampleData(1000);
        
        if (channel.SendBatch(batch)) {
            std::cout << "Sent batch " << i + 1 << " with " 
                      << batch->num_rows() << " rows" << std::endl;
        } else {
            std::cout << "Failed to send batch " << i + 1 << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    std::cout << "Producer finished" << std::endl;
}

// Data consumer with Arrow compute
void RunConsumer() {
    std::cout << "Starting Arrow data consumer..." << std::endl;
    
    // Create Arrow channel
    ArrowChannel channel("memory://sensor_data", 32 * 1024 * 1024);
    
    int64_t total_rows = 0;
    arrow::compute::ScalarAggregateOptions options;
    
    // Process batches
    for (int i = 0; i < 10; ++i) {
        auto batch = channel.ReceiveBatch(5000);  // 5 second timeout
        if (!batch) {
            std::cout << "No batch received" << std::endl;
            continue;
        }
        
        total_rows += batch->num_rows();
        
        // Compute statistics using Arrow compute
        auto temp_array = batch->column(1);  // temperature column
        auto humidity_array = batch->column(2);  // humidity column
        
        // Calculate mean temperature
        auto mean_result = arrow::compute::Mean(temp_array, options);
        if (mean_result.ok()) {
            auto mean_scalar = mean_result.ValueOrDie();
            auto mean_value = std::static_pointer_cast<arrow::FloatScalar>(mean_scalar)->value;
            std::cout << "Batch " << i + 1 << " - Mean temperature: " << mean_value << "°C";
        }
        
        // Calculate mean humidity
        mean_result = arrow::compute::Mean(humidity_array, options);
        if (mean_result.ok()) {
            auto mean_scalar = mean_result.ValueOrDie();
            auto mean_value = std::static_pointer_cast<arrow::FloatScalar>(mean_scalar)->value;
            std::cout << ", Mean humidity: " << mean_value << "%" << std::endl;
        }
    }
    
    std::cout << "Consumer processed " << total_rows << " total rows" << std::endl;
}

// Pipeline example with transformation
void RunPipeline() {
    std::cout << "Starting Arrow data pipeline..." << std::endl;
    
    // Create pipeline that converts Celsius to Fahrenheit and filters
    auto transform = [](const std::shared_ptr<arrow::RecordBatch>& input) {
        // Get temperature array
        auto temp_celsius = input->column(1);
        
        // Convert Celsius to Fahrenheit: F = C * 9/5 + 32
        auto multiply_result = arrow::compute::Multiply(
            temp_celsius, 
            arrow::MakeScalar(9.0f / 5.0f));
        
        if (!multiply_result.ok()) return input;
        
        auto add_result = arrow::compute::Add(
            multiply_result.ValueOrDie(),
            arrow::MakeScalar(32.0f));
        
        if (!add_result.ok()) return input;
        
        auto temp_fahrenheit = add_result.ValueOrDie();
        
        // Create new batch with converted temperatures
        return arrow::RecordBatch::Make(
            input->schema(),
            input->num_rows(),
            {input->column(0), temp_fahrenheit, input->column(2), input->column(3)});
    };
    
    ArrowPipeline pipeline("memory://sensor_data", 
                          "memory://sensor_data_fahrenheit",
                          transform);
    
    // Run pipeline in separate thread
    std::thread pipeline_thread([&pipeline]() {
        pipeline.Run();
    });
    
    // Give it time to process
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // Get stats
    auto stats = pipeline.GetStats();
    std::cout << "Pipeline processed " << stats.batches_processed << " batches, "
              << stats.rows_processed << " rows, "
              << stats.bytes_processed / 1024 << " KB, "
              << "avg latency: " << stats.avg_latency_ms << " ms" << std::endl;
    
    pipeline.Stop();
    pipeline_thread.join();
}

// Demonstrate zero-copy conversion
void DemoZeroCopy() {
    std::cout << "\nDemonstrating zero-copy Arrow integration..." << std::endl;
    
    // Create Psyne channel and FloatVector
    auto channel = create_channel("memory://demo", 1024 * 1024);
    FloatVector vec(*channel);
    vec.resize(1000);
    
    // Fill with sample data
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = std::sin(i * 0.01f) * 100.0f;
    }
    
    // Convert to Arrow (copy required)
    auto arrow_array = ArrowConverter::FloatVectorToArrow(vec);
    std::cout << "Created Arrow array with " << arrow_array->length() 
              << " elements" << std::endl;
    
    // Create zero-copy view (when possible)
    auto view = ArrowConverter::CreateArrowView<arrow::FloatArray>(
        vec.data(), vec.size());
    std::cout << "Created zero-copy Arrow view" << std::endl;
    
    // Verify data matches
    auto float_array = std::static_pointer_cast<arrow::FloatArray>(arrow_array);
    bool matches = true;
    for (int64_t i = 0; i < float_array->length(); ++i) {
        if (std::abs(float_array->Value(i) - view->Value(i)) > 0.0001f) {
            matches = false;
            break;
        }
    }
    std::cout << "Data integrity check: " << (matches ? "PASSED" : "FAILED") << std::endl;
}

// Table streaming example
void RunTableStreaming() {
    std::cout << "\nDemonstrating Arrow table streaming..." << std::endl;
    
    // Create channels
    ArrowChannel sender("memory://tables", 64 * 1024 * 1024);
    ArrowChannel receiver("memory://tables", 64 * 1024 * 1024);
    
    // Producer thread - send a large table in batches
    std::thread producer([&sender]() {
        // Create a large table
        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        for (int i = 0; i < 5; ++i) {
            batches.push_back(GenerateSampleData(2000));
        }
        
        auto schema = batches[0]->schema();
        auto table = arrow::Table::FromRecordBatches(schema, batches).ValueOrDie();
        
        std::cout << "Sending table with " << table->num_rows() << " rows" << std::endl;
        size_t batches_sent = sender.SendTable(table, 1000);  // 1000 rows per batch
        std::cout << "Sent " << batches_sent << " batches" << std::endl;
    });
    
    // Consumer thread - receive the complete table
    std::thread consumer([&receiver]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Let producer start
        
        std::cout << "Receiving table..." << std::endl;
        auto table = receiver.ReceiveTable(10);  // Expect 10 batches
        
        if (table) {
            std::cout << "Received table with " << table->num_rows() 
                      << " rows and " << table->num_columns() << " columns" << std::endl;
            
            // Print schema
            std::cout << "Schema: " << table->schema()->ToString() << std::endl;
        } else {
            std::cout << "Failed to receive table" << std::endl;
        }
    });
    
    producer.join();
    consumer.join();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] 
                  << " [producer|consumer|pipeline|zerocopy|streaming]" << std::endl;
        return 1;
    }
    
    std::string mode = argv[1];
    
    try {
        if (mode == "producer") {
            RunProducer();
        } else if (mode == "consumer") {
            RunConsumer();
        } else if (mode == "pipeline") {
            // Run producer in background
            std::thread producer_thread(RunProducer);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            RunPipeline();
            producer_thread.join();
        } else if (mode == "zerocopy") {
            DemoZeroCopy();
        } else if (mode == "streaming") {
            RunTableStreaming();
        } else {
            std::cerr << "Invalid mode. Use producer, consumer, pipeline, zerocopy, or streaming" 
                      << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}