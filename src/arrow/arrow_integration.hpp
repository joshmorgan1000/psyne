#pragma once

#include <psyne/psyne.hpp>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <memory>
#include <vector>

namespace psyne {
namespace arrow_integration {

/**
 * @brief Zero-copy Arrow array message type
 * 
 * Enables sending Apache Arrow arrays through Psyne channels with
 * minimal overhead. The Arrow data is serialized using Arrow's IPC
 * format for efficient transfer.
 */
class ArrowArrayMessage : public Message<ArrowArrayMessage> {
public:
    static constexpr uint32_t message_type = 2000;
    
    using Message<ArrowArrayMessage>::Message;
    
    static size_t calculate_size() {
        // Default size - will be resized as needed
        return 64 * 1024; // 64KB default
    }
    
    /**
     * @brief Serialize an Arrow array into this message
     * @param array The Arrow array to send
     * @return true if successful, false if buffer too small
     */
    bool SetArray(const std::shared_ptr<arrow::Array>& array);
    
    /**
     * @brief Get the Arrow array from this message
     * @return The deserialized Arrow array, or nullptr on error
     */
    std::shared_ptr<arrow::Array> GetArray() const;
    
    /**
     * @brief Get the Arrow data type of the contained array
     * @return Data type, or nullptr if not valid
     */
    std::shared_ptr<arrow::DataType> GetDataType() const;
    
    void initialize() {
        // No special initialization needed
    }
    
private:
    mutable std::shared_ptr<arrow::Array> cached_array_;
};

/**
 * @brief Zero-copy Arrow record batch message type
 * 
 * Enables sending Apache Arrow record batches (tables) through Psyne
 * channels. Record batches are collections of equal-length arrays.
 */
class ArrowRecordBatchMessage : public Message<ArrowRecordBatchMessage> {
public:
    static constexpr uint32_t message_type = 2001;
    
    using Message<ArrowRecordBatchMessage>::Message;
    
    static size_t calculate_size() {
        // Default size - will be resized as needed
        return 256 * 1024; // 256KB default
    }
    
    /**
     * @brief Serialize a record batch into this message
     * @param batch The record batch to send
     * @return true if successful, false if buffer too small
     */
    bool SetRecordBatch(const std::shared_ptr<arrow::RecordBatch>& batch);
    
    /**
     * @brief Get the record batch from this message
     * @return The deserialized record batch, or nullptr on error
     */
    std::shared_ptr<arrow::RecordBatch> GetRecordBatch() const;
    
    /**
     * @brief Get the schema of the record batch
     * @return Schema, or nullptr if not valid
     */
    std::shared_ptr<arrow::Schema> GetSchema() const;
    
    void initialize() {
        // No special initialization needed
    }
    
private:
    mutable std::shared_ptr<arrow::RecordBatch> cached_batch_;
};

/**
 * @brief Convert between Psyne FloatVector and Arrow arrays
 */
class ArrowConverter {
public:
    /**
     * @brief Convert FloatVector to Arrow Float32Array
     * @param vec The FloatVector to convert
     * @return Arrow array, or nullptr on error
     */
    static std::shared_ptr<arrow::Array> FloatVectorToArrow(const FloatVector& vec);
    
    /**
     * @brief Convert Arrow Float32Array to FloatVector
     * @param array The Arrow array
     * @param channel Channel to allocate the FloatVector in
     * @return FloatVector containing the data
     */
    static FloatVector ArrowToFloatVector(const std::shared_ptr<arrow::Array>& array,
                                         Channel& channel);
    
    /**
     * @brief Convert ByteVector to Arrow BinaryArray
     * @param vec The ByteVector to convert
     * @return Arrow array, or nullptr on error
     */
    static std::shared_ptr<arrow::Array> ByteVectorToArrow(const ByteVector& vec);
    
    /**
     * @brief Create an Arrow array view over Psyne message data (zero-copy)
     * @tparam T The Arrow array type (e.g., arrow::FloatArray)
     * @param data Pointer to the data
     * @param length Number of elements
     * @return Arrow array view
     */
    template<typename T>
    static std::shared_ptr<T> CreateArrowView(void* data, int64_t length);
};

/**
 * @brief High-level Arrow channel for streaming record batches
 * 
 * This provides a higher-level interface specifically designed for
 * streaming Arrow data between processes or over the network.
 */
class ArrowChannel {
public:
    /**
     * @brief Create an Arrow-specific channel
     * @param uri Psyne channel URI
     * @param buffer_size Buffer size for the underlying channel
     */
    ArrowChannel(const std::string& uri, size_t buffer_size = 16 * 1024 * 1024);
    
    /**
     * @brief Send a record batch
     * @param batch The record batch to send
     * @return true if successful
     */
    bool SendBatch(const std::shared_ptr<arrow::RecordBatch>& batch);
    
    /**
     * @brief Receive a record batch
     * @param timeout_ms Timeout in milliseconds (0 for non-blocking)
     * @return The received batch, or nullptr if none available
     */
    std::shared_ptr<arrow::RecordBatch> ReceiveBatch(int timeout_ms = 0);
    
    /**
     * @brief Send an Arrow table as a stream of batches
     * @param table The table to send
     * @param batch_size Maximum rows per batch
     * @return Number of batches sent
     */
    size_t SendTable(const std::shared_ptr<arrow::Table>& table, 
                     int64_t batch_size = 10000);
    
    /**
     * @brief Receive a complete table (blocks until all batches received)
     * @param expected_batches Number of batches to receive (-1 for unknown)
     * @return The complete table, or nullptr on error
     */
    std::shared_ptr<arrow::Table> ReceiveTable(int expected_batches = -1);
    
    /**
     * @brief Get the underlying Psyne channel
     */
    std::shared_ptr<Channel> GetChannel() const { return channel_; }
    
private:
    std::shared_ptr<Channel> channel_;
    std::shared_ptr<arrow::Schema> schema_;
};

/**
 * @brief Arrow-based data pipeline for ETL operations
 * 
 * Combines Psyne's zero-copy messaging with Arrow's columnar processing
 * for high-performance data pipelines.
 */
class ArrowPipeline {
public:
    using TransformFunc = std::function<std::shared_ptr<arrow::RecordBatch>(
        const std::shared_ptr<arrow::RecordBatch>&)>;
    
    /**
     * @brief Create a pipeline stage
     * @param input_uri Input channel URI
     * @param output_uri Output channel URI
     * @param transform Transformation function
     */
    ArrowPipeline(const std::string& input_uri,
                  const std::string& output_uri,
                  TransformFunc transform);
    
    /**
     * @brief Run the pipeline (blocks until stopped)
     */
    void Run();
    
    /**
     * @brief Stop the pipeline
     */
    void Stop();
    
    /**
     * @brief Get pipeline statistics
     */
    struct Stats {
        uint64_t batches_processed = 0;
        uint64_t rows_processed = 0;
        uint64_t bytes_processed = 0;
        double avg_latency_ms = 0;
    };
    Stats GetStats() const;
    
private:
    ArrowChannel input_;
    ArrowChannel output_;
    TransformFunc transform_;
    std::atomic<bool> running_{false};
    mutable Stats stats_;
};

/**
 * @brief Integration with Arrow Flight for RPC
 * 
 * Allows Psyne channels to act as Arrow Flight endpoints for
 * compatibility with the Arrow ecosystem.
 */
class ArrowFlightAdapter {
public:
    /**
     * @brief Create a Flight server backed by Psyne channels
     * @param location Flight server location (e.g., "grpc://0.0.0.0:8815")
     */
    explicit ArrowFlightAdapter(const std::string& location);
    
    /**
     * @brief Register a Psyne channel as a Flight stream
     * @param descriptor Flight descriptor for the stream
     * @param channel The Psyne channel to use
     */
    void RegisterStream(const arrow::flight::FlightDescriptor& descriptor,
                       std::shared_ptr<ArrowChannel> channel);
    
    /**
     * @brief Start the Flight server
     */
    void Start();
    
    /**
     * @brief Stop the Flight server
     */
    void Stop();
    
private:
    class FlightServerImpl;
    std::unique_ptr<FlightServerImpl> impl_;
};

// Template implementation
template<typename T>
std::shared_ptr<T> ArrowConverter::CreateArrowView(void* data, int64_t length) {
    using ValueType = typename T::value_type;
    
    // Create buffer that doesn't own the memory
    auto buffer = arrow::Buffer::Wrap(static_cast<ValueType*>(data), 
                                      length * sizeof(ValueType));
    
    // Create array
    auto data_array = arrow::ArrayData::Make(
        arrow::TypeTraits<ValueType>::type_singleton(),
        length, {nullptr, buffer});
    
    return std::static_pointer_cast<T>(arrow::MakeArray(data_array));
}

} // namespace arrow_integration
} // namespace psyne