#include "arrow_integration.hpp"
#include <arrow/buffer.h>
#include <arrow/builder.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/flight/api.h>
#include <iostream>

namespace psyne {
namespace arrow_integration {

// ArrowArrayMessage implementation
bool ArrowArrayMessage::SetArray(const std::shared_ptr<arrow::Array>& array) {
    if (!array) return false;
    
    // Create memory buffer to write to
    auto buffer_output_stream = arrow::io::BufferOutputStream::Create().ValueOrDie();
    
    // Create schema with single field
    auto field = arrow::field("array", array->type());
    auto schema = arrow::schema({field});
    
    // Create record batch with single column
    auto batch = arrow::RecordBatch::Make(schema, array->length(), {array});
    
    // Write using IPC format
    auto writer_result = arrow::ipc::MakeStreamWriter(
        buffer_output_stream, schema);
    if (!writer_result.ok()) return false;
    
    auto writer = writer_result.ValueOrDie();
    auto status = writer->WriteRecordBatch(*batch);
    if (!status.ok()) return false;
    
    status = writer->Close();
    if (!status.ok()) return false;
    
    // Get the serialized data
    auto buffer_result = buffer_output_stream->Finish();
    if (!buffer_result.ok()) return false;
    
    auto buffer = buffer_result.ValueOrDie();
    
    // Check if it fits in our message
    if (buffer->size() > capacity()) {
        return false;
    }
    
    // Copy to message buffer
    resize(buffer->size());
    std::memcpy(data(), buffer->data(), buffer->size());
    
    // Cache the array
    cached_array_ = array;
    
    return true;
}

std::shared_ptr<arrow::Array> ArrowArrayMessage::GetArray() const {
    if (cached_array_) {
        return cached_array_;
    }
    
    if (size() == 0) return nullptr;
    
    // Create buffer reader
    auto buffer = arrow::Buffer::Wrap(data(), size());
    auto buffer_reader = std::make_shared<arrow::io::BufferReader>(buffer);
    
    // Read IPC stream
    auto reader_result = arrow::ipc::RecordBatchStreamReader::Open(buffer_reader);
    if (!reader_result.ok()) return nullptr;
    
    auto reader = reader_result.ValueOrDie();
    
    // Read the batch
    auto batch_result = reader->Next();
    if (!batch_result.ok()) return nullptr;
    
    auto batch = batch_result.ValueOrDie();
    if (!batch || batch->num_columns() == 0) return nullptr;
    
    // Extract the array (first column)
    cached_array_ = batch->column(0);
    return cached_array_;
}

std::shared_ptr<arrow::DataType> ArrowArrayMessage::GetDataType() const {
    auto array = GetArray();
    return array ? array->type() : nullptr;
}

// ArrowRecordBatchMessage implementation
bool ArrowRecordBatchMessage::SetRecordBatch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    if (!batch) return false;
    
    // Create memory buffer to write to
    auto buffer_output_stream = arrow::io::BufferOutputStream::Create().ValueOrDie();
    
    // Write using IPC format
    auto writer_result = arrow::ipc::MakeStreamWriter(
        buffer_output_stream, batch->schema());
    if (!writer_result.ok()) return false;
    
    auto writer = writer_result.ValueOrDie();
    auto status = writer->WriteRecordBatch(*batch);
    if (!status.ok()) return false;
    
    status = writer->Close();
    if (!status.ok()) return false;
    
    // Get the serialized data
    auto buffer_result = buffer_output_stream->Finish();
    if (!buffer_result.ok()) return false;
    
    auto buffer = buffer_result.ValueOrDie();
    
    // Check if it fits in our message
    if (buffer->size() > capacity()) {
        return false;
    }
    
    // Copy to message buffer
    resize(buffer->size());
    std::memcpy(data(), buffer->data(), buffer->size());
    
    // Cache the batch
    cached_batch_ = batch;
    
    return true;
}

std::shared_ptr<arrow::RecordBatch> ArrowRecordBatchMessage::GetRecordBatch() const {
    if (cached_batch_) {
        return cached_batch_;
    }
    
    if (size() == 0) return nullptr;
    
    // Create buffer reader
    auto buffer = arrow::Buffer::Wrap(data(), size());
    auto buffer_reader = std::make_shared<arrow::io::BufferReader>(buffer);
    
    // Read IPC stream
    auto reader_result = arrow::ipc::RecordBatchStreamReader::Open(buffer_reader);
    if (!reader_result.ok()) return nullptr;
    
    auto reader = reader_result.ValueOrDie();
    
    // Read the batch
    auto batch_result = reader->Next();
    if (!batch_result.ok()) return nullptr;
    
    cached_batch_ = batch_result.ValueOrDie();
    return cached_batch_;
}

std::shared_ptr<arrow::Schema> ArrowRecordBatchMessage::GetSchema() const {
    auto batch = GetRecordBatch();
    return batch ? batch->schema() : nullptr;
}

// ArrowConverter implementation
std::shared_ptr<arrow::Array> ArrowConverter::FloatVectorToArrow(const FloatVector& vec) {
    // Create builder
    arrow::FloatBuilder builder;
    auto status = builder.Reserve(vec.size());
    if (!status.ok()) return nullptr;
    
    // Append all values
    status = builder.AppendValues(vec.begin(), vec.size());
    if (!status.ok()) return nullptr;
    
    // Build array
    auto result = builder.Finish();
    if (!result.ok()) return nullptr;
    
    return result.ValueOrDie();
}

FloatVector ArrowConverter::ArrowToFloatVector(const std::shared_ptr<arrow::Array>& array,
                                             Channel& channel) {
    FloatVector vec(channel);
    
    if (!array || array->type_id() != arrow::Type::FLOAT) {
        return vec;
    }
    
    auto float_array = std::static_pointer_cast<arrow::FloatArray>(array);
    vec.resize(float_array->length());
    
    // Copy data
    for (int64_t i = 0; i < float_array->length(); ++i) {
        vec[i] = float_array->Value(i);
    }
    
    return vec;
}

std::shared_ptr<arrow::Array> ArrowConverter::ByteVectorToArrow(const ByteVector& vec) {
    // Create binary array
    arrow::BinaryBuilder builder;
    auto status = builder.Append(vec.data(), vec.size());
    if (!status.ok()) return nullptr;
    
    auto result = builder.Finish();
    if (!result.ok()) return nullptr;
    
    return result.ValueOrDie();
}

// ArrowChannel implementation
ArrowChannel::ArrowChannel(const std::string& uri, size_t buffer_size)
    : channel_(create_channel(uri, buffer_size)) {}

bool ArrowChannel::SendBatch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    if (!batch) return false;
    
    ArrowRecordBatchMessage msg(*channel_);
    if (!msg.SetRecordBatch(batch)) {
        return false;
    }
    
    channel_->send(msg);
    return true;
}

std::shared_ptr<arrow::RecordBatch> ArrowChannel::ReceiveBatch(int timeout_ms) {
    auto msg = channel_->receive<ArrowRecordBatchMessage>(
        std::chrono::milliseconds(timeout_ms));
    
    if (!msg) return nullptr;
    
    return msg->GetRecordBatch();
}

size_t ArrowChannel::SendTable(const std::shared_ptr<arrow::Table>& table, 
                              int64_t batch_size) {
    if (!table) return 0;
    
    // Create table reader
    arrow::TableBatchReader reader(*table);
    reader.set_chunksize(batch_size);
    
    size_t batches_sent = 0;
    std::shared_ptr<arrow::RecordBatch> batch;
    
    while (true) {
        auto result = reader.Next();
        if (!result.ok()) break;
        
        batch = result.ValueOrDie();
        if (!batch) break;
        
        if (SendBatch(batch)) {
            batches_sent++;
        }
    }
    
    return batches_sent;
}

std::shared_ptr<arrow::Table> ArrowChannel::ReceiveTable(int expected_batches) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::shared_ptr<arrow::Schema> schema;
    
    while (expected_batches < 0 || batches.size() < static_cast<size_t>(expected_batches)) {
        auto batch = ReceiveBatch(1000);  // 1 second timeout
        if (!batch) break;
        
        if (!schema) {
            schema = batch->schema();
        }
        
        batches.push_back(batch);
    }
    
    if (batches.empty()) return nullptr;
    
    auto result = arrow::Table::FromRecordBatches(schema, batches);
    if (!result.ok()) return nullptr;
    
    return result.ValueOrDie();
}

// ArrowPipeline implementation
ArrowPipeline::ArrowPipeline(const std::string& input_uri,
                           const std::string& output_uri,
                           TransformFunc transform)
    : input_(input_uri), output_(output_uri), transform_(transform) {}

void ArrowPipeline::Run() {
    running_ = true;
    
    while (running_) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Receive batch
        auto input_batch = input_.ReceiveBatch(100);
        if (!input_batch) continue;
        
        // Transform
        auto output_batch = transform_(input_batch);
        if (!output_batch) continue;
        
        // Send
        if (output_.SendBatch(output_batch)) {
            stats_.batches_processed++;
            stats_.rows_processed += output_batch->num_rows();
            
            // Simple size estimation
            size_t batch_size = 0;
            for (int i = 0; i < output_batch->num_columns(); ++i) {
                batch_size += output_batch->column(i)->data()->buffers[1]->size();
            }
            stats_.bytes_processed += batch_size;
            
            // Update latency
            auto end = std::chrono::high_resolution_clock::now();
            auto latency = std::chrono::duration<double, std::milli>(end - start).count();
            stats_.avg_latency_ms = (stats_.avg_latency_ms * (stats_.batches_processed - 1) + 
                                    latency) / stats_.batches_processed;
        }
    }
}

void ArrowPipeline::Stop() {
    running_ = false;
}

ArrowPipeline::Stats ArrowPipeline::GetStats() const {
    return stats_;
}

// ArrowFlightAdapter implementation
class ArrowFlightAdapter::FlightServerImpl : public arrow::flight::FlightServerBase {
public:
    arrow::Status ListFlights(const arrow::flight::ServerCallContext& context,
                             const arrow::flight::Criteria* criteria,
                             std::unique_ptr<arrow::flight::FlightListing>* listings) override {
        // Implementation would list available streams
        return arrow::Status::OK();
    }
    
    arrow::Status DoGet(const arrow::flight::ServerCallContext& context,
                       const arrow::flight::Ticket& request,
                       std::unique_ptr<arrow::flight::FlightDataStream>* stream) override {
        // Implementation would create stream from Psyne channel
        return arrow::Status::OK();
    }
    
    arrow::Status DoPut(const arrow::flight::ServerCallContext& context,
                       std::unique_ptr<arrow::flight::FlightMessageReader> reader,
                       std::unique_ptr<arrow::flight::FlightMetadataWriter> writer) override {
        // Implementation would write to Psyne channel
        return arrow::Status::OK();
    }
    
    std::unordered_map<std::string, std::shared_ptr<ArrowChannel>> channels_;
};

ArrowFlightAdapter::ArrowFlightAdapter(const std::string& location)
    : impl_(std::make_unique<FlightServerImpl>()) {
    // Parse location and initialize
}

void ArrowFlightAdapter::RegisterStream(const arrow::flight::FlightDescriptor& descriptor,
                                      std::shared_ptr<ArrowChannel> channel) {
    // Store channel mapping
    impl_->channels_[descriptor.ToString()] = channel;
}

void ArrowFlightAdapter::Start() {
    // Start Flight server
}

void ArrowFlightAdapter::Stop() {
    // Stop Flight server
}

} // namespace arrow_integration
} // namespace psyne