#pragma once

#include <memory>
#include <opentelemetry/context/propagation/text_map_propagator.h>
#include <opentelemetry/exporters/jaeger/jaeger_exporter.h>
#include <opentelemetry/exporters/zipkin/zipkin_exporter.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/trace/span.h>
#include <opentelemetry/trace/tracer.h>
#include <psyne/psyne.hpp>
#include <string>
#include <unordered_map>

namespace psyne {
namespace tracing {

namespace otel = opentelemetry;

/**
 * @brief Configuration for distributed tracing
 */
struct TracingConfig {
    enum class Backend { None, Jaeger, Zipkin, OpenTelemetryCollector };

    Backend backend = Backend::Jaeger;
    std::string service_name = "psyne-service";
    std::string endpoint =
        "http://localhost:14268/api/traces"; // Jaeger default
    double sample_rate = 1.0; // 1.0 = trace everything, 0.0 = trace nothing
    bool propagate_context = true; // Propagate trace context across channels

    // Additional attributes to add to all spans
    std::unordered_map<std::string, std::string> resource_attributes;
};

/**
 * @brief Traced channel wrapper that adds distributed tracing to any channel
 *
 * Automatically creates spans for send/receive operations and propagates
 * trace context across process boundaries.
 */
class TracedChannel : public Channel {
public:
    /**
     * @brief Create a traced channel
     * @param underlying The channel to wrap
     * @param tracer OpenTelemetry tracer instance
     * @param config Tracing configuration
     */
    TracedChannel(std::shared_ptr<Channel> underlying,
                  std::shared_ptr<otel::trace::Tracer> tracer,
                  const TracingConfig &config);
    ~TracedChannel() override;

    // Channel interface
    void stop() override;
    bool is_stopped() const override;
    const std::string &uri() const override;
    ChannelType type() const override;
    ChannelMode mode() const override;
    void *receive_raw_message(size_t &size, uint32_t &type) override;
    void release_raw_message(void *handle) override;
    bool has_metrics() const override;
    debug::ChannelMetrics get_metrics() const override;
    void reset_metrics() override;

    /**
     * @brief Send a message with tracing
     * @param msg Message to send
     * @param span_name Optional custom span name
     */
    template <typename MessageType>
    void send_traced(MessageType &msg, const std::string &span_name = "");

    /**
     * @brief Receive a message with tracing
     * @param timeout Receive timeout
     * @param span_name Optional custom span name
     * @return Received message
     */
    template <typename MessageType>
    std::optional<MessageType> receive_traced(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero(),
        const std::string &span_name = "");

    /**
     * @brief Get tracing statistics
     */
    struct TracingStats {
        uint64_t spans_created = 0;
        uint64_t spans_exported = 0;
        uint64_t context_propagations = 0;
        uint64_t errors = 0;
    };
    TracingStats GetTracingStats() const;

protected:
    detail::ChannelImpl *impl() override;
    const detail::ChannelImpl *impl() const override;

private:
    struct TraceContext {
        std::string trace_id;
        std::string span_id;
        std::string trace_flags;
        std::unordered_map<std::string, std::string> baggage;
    };

    std::shared_ptr<Channel> underlying_;
    std::shared_ptr<otel::trace::Tracer> tracer_;
    TracingConfig config_;

    // Context propagation
    std::unique_ptr<otel::context::propagation::TextMapPropagator> propagator_;

    // Statistics
    mutable TracingStats stats_;
    mutable std::mutex stats_mutex_;

    // Helper methods
    TraceContext ExtractContext(const void *message_header);
    void InjectContext(void *message_header, const TraceContext &context);
};

/**
 * @brief Global tracing provider for Psyne
 *
 * Manages the OpenTelemetry tracer provider and provides factory methods
 * for creating traced channels.
 */
class TracingProvider {
public:
    /**
     * @brief Initialize the global tracing provider
     * @param config Tracing configuration
     * @return true if successful
     */
    static bool Initialize(const TracingConfig &config);

    /**
     * @brief Shutdown the tracing provider
     */
    static void Shutdown();

    /**
     * @brief Create a traced channel
     * @param uri Channel URI
     * @param buffer_size Buffer size
     * @param mode Channel mode
     * @param type Channel type
     * @return Traced channel
     */
    static std::shared_ptr<Channel>
    CreateTracedChannel(const std::string &uri,
                        size_t buffer_size = 1024 * 1024,
                        ChannelMode mode = ChannelMode::SPSC,
                        ChannelType type = ChannelType::MultiType);

    /**
     * @brief Wrap an existing channel with tracing
     * @param channel Channel to wrap
     * @return Traced channel
     */
    static std::shared_ptr<Channel>
    WrapWithTracing(std::shared_ptr<Channel> channel);

    /**
     * @brief Get a tracer instance
     * @param name Tracer name (defaults to service name)
     * @return Tracer instance
     */
    static std::shared_ptr<otel::trace::Tracer>
    GetTracer(const std::string &name = "");

    /**
     * @brief Create a custom span
     * @param name Span name
     * @param attributes Span attributes
     * @return Span handle
     */
    static std::shared_ptr<otel::trace::Span> CreateSpan(
        const std::string &name,
        const std::unordered_map<std::string, std::string> &attributes = {});

private:
    static std::shared_ptr<otel::sdk::trace::TracerProvider> provider_;
    static TracingConfig config_;
    static std::mutex mutex_;
};

/**
 * @brief Trace context propagation for cross-process tracing
 *
 * Handles serialization and deserialization of trace context for
 * propagation across Psyne channels.
 */
class TraceContextPropagator {
public:
    /**
     * @brief Serialize trace context to bytes
     * @param span Current span
     * @return Serialized context
     */
    static std::vector<uint8_t> Serialize(const otel::trace::Span &span);

    /**
     * @brief Deserialize trace context from bytes
     * @param data Serialized context data
     * @param size Data size
     * @return Trace context
     */
    static otel::trace::SpanContext Deserialize(const void *data, size_t size);

    /**
     * @brief Inject trace context into message header
     * @param message Psyne message
     * @param span Current span
     */
    template <typename MessageType>
    static void InjectContext(MessageType &message,
                              const otel::trace::Span &span);

    /**
     * @brief Extract trace context from message header
     * @param message Psyne message
     * @return Extracted span context
     */
    template <typename MessageType>
    static otel::trace::SpanContext ExtractContext(const MessageType &message);
};

/**
 * @brief Instrumentation helpers for common patterns
 */
namespace instrumentation {

/**
 * @brief Trace a request-response pattern
 * @param request_channel Channel for sending requests
 * @param response_channel Channel for receiving responses
 * @param operation_name Name of the operation
 * @param request Request data
 * @return Response data
 */
template <typename RequestType, typename ResponseType>
std::optional<ResponseType>
TraceRequestResponse(Channel &request_channel, Channel &response_channel,
                     const std::string &operation_name,
                     const RequestType &request);

/**
 * @brief Trace a pipeline stage
 * @param input_channel Input channel
 * @param output_channel Output channel
 * @param stage_name Pipeline stage name
 * @param processor Processing function
 */
template <typename InputType, typename OutputType>
void TracePipelineStage(Channel &input_channel, Channel &output_channel,
                        const std::string &stage_name,
                        std::function<OutputType(const InputType &)> processor);

/**
 * @brief Create a traced fan-out pattern
 * @param input_channel Input channel
 * @param output_channels Output channels
 * @param operation_name Operation name
 */
template <typename MessageType>
void TraceFanOut(Channel &input_channel,
                 const std::vector<std::shared_ptr<Channel>> &output_channels,
                 const std::string &operation_name);

/**
 * @brief Trace metrics collector
 *
 * Collects and exports channel metrics as OpenTelemetry metrics.
 */
class MetricsCollector {
public:
    MetricsCollector(std::chrono::seconds collection_interval);
    ~MetricsCollector();

    /**
     * @brief Add a channel to monitor
     * @param name Metric name for this channel
     * @param channel Channel to monitor
     */
    void AddChannel(const std::string &name, std::shared_ptr<Channel> channel);

    /**
     * @brief Start collecting metrics
     */
    void Start();

    /**
     * @brief Stop collecting metrics
     */
    void Stop();

private:
    std::chrono::seconds interval_;
    std::unordered_map<std::string, std::shared_ptr<Channel>> channels_;
    std::thread collector_thread_;
    std::atomic<bool> running_{false};

    void CollectionLoop();
};

} // namespace instrumentation

/**
 * @brief Distributed tracing macros for convenience
 */
#define PSYNE_TRACE_SEND(channel, message, ...)                                \
    do {                                                                       \
        auto span = TracingProvider::CreateSpan(                               \
            "send:" + std::string(#message), ##__VA_ARGS__);                   \
        span->SetAttribute("channel.uri", (channel).uri());                    \
        span->SetAttribute("message.type",                                     \
                           std::to_string((message).message_type));            \
        span->SetAttribute("message.size", std::to_string((message).size()));  \
        (channel).send(message);                                               \
        span->End();                                                           \
    } while (0)

#define PSYNE_TRACE_RECEIVE(channel, message_type, timeout, ...)               \
    [&]() {                                                                    \
        auto span = TracingProvider::CreateSpan(                               \
            "receive:" + std::string(#message_type), ##__VA_ARGS__);           \
        span->SetAttribute("channel.uri", (channel).uri());                    \
        span->SetAttribute("timeout.ms", std::to_string((timeout).count()));   \
        auto msg = (channel).receive<message_type>(timeout);                   \
        if (msg) {                                                             \
            span->SetAttribute("message.size", std::to_string(msg->size()));   \
            span->SetStatus(otel::trace::StatusCode::kOk);                     \
        } else {                                                               \
            span->SetStatus(otel::trace::StatusCode::kError,                   \
                            "No message received");                            \
        }                                                                      \
        span->End();                                                           \
        return msg;                                                            \
    }()

#define PSYNE_TRACE_SCOPE(name, ...)                                           \
    auto _trace_span = TracingProvider::CreateSpan(name, ##__VA_ARGS__);       \
    auto _trace_scope = otel::trace::Scope(_trace_span)

} // namespace tracing
} // namespace psyne