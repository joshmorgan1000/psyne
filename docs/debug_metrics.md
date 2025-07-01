# Psyne Debug Metrics Guide

The Psyne debug metrics system provides comprehensive real-time monitoring and analysis capabilities for debugging and performance tuning.

## Overview

The metrics system tracks:
- Message throughput (messages/second)
- Bandwidth utilization (MB/s)
- Latency distribution (percentiles)
- Buffer usage and pressure
- Memory allocation patterns
- Connection state (for network channels)
- Error conditions and failures

## Quick Start

### 1. Using Debug Channels

The easiest way to add metrics to your code:

```cpp
#include "psyne/debug/debug_channel.hpp"

// Start metrics collection
psyne::debug::start_metrics();

// Create debug-enabled channel
auto channel = psyne::debug::DebugChannel<MyMessage>::create(config);

// Use exactly like normal channel
auto msg = channel->allocate();
msg->data = 42;
msg.send();

// Stop metrics when done
psyne::debug::stop_metrics();
```

### 2. Manual Metrics Integration

For existing code with minimal changes:

```cpp
#include "psyne/debug/metrics_collector.hpp"

// Configure metrics
psyne::debug::MetricsConfig config;
config.enabled = true;
config.console_output = true;
config.file_output = true;
config.output_file = "my_metrics.csv";

psyne::debug::MetricsCollector::instance().configure(config);
psyne::debug::MetricsCollector::instance().start();

// Register your channels
MetricsCollector::instance().register_channel("my_channel", 
                                             ChannelTransport::IPC, 
                                             16 * 1024 * 1024);

// Track operations
MetricsCollector::instance().record_send("my_channel", message_size, sequence);
MetricsCollector::instance().record_receive("my_channel", message_size, 
                                           sequence, latency_ns);
```

### 3. Real-time Monitoring

Use the `psyne_monitor` tool for live visualization:

```bash
# Basic monitoring
./psyne_monitor

# With CSV export
./psyne_monitor --csv metrics.csv

# Without graphs (for SSH sessions)
./psyne_monitor --no-graphs
```

## Metrics Configuration

```cpp
struct MetricsConfig {
    bool enabled = true;              // Enable/disable all metrics
    bool console_output = true;       // Print to console
    bool file_output = false;         // Export to CSV
    std::string output_file = "psyne_metrics.log";
    uint32_t sampling_interval_ms = 1000;  // Sample rate
    bool detailed_histograms = true;  // Latency histograms
    bool memory_tracking = true;      // Track allocations
    bool event_tracing = false;       // Detailed event log
    size_t event_buffer_size = 10000; // Event buffer size
    bool live_dashboard = false;      // Built-in dashboard
};
```

## Compilation Flags

Enable debug metrics at compile time:

```bash
# Enable metrics collection
g++ -DPSYNE_DEBUG_METRICS=1 ...

# Debug build with all features
g++ -g -O0 -DPSYNE_DEBUG_METRICS=1 -DPSYNE_DEBUG_BUILD=1 ...

# Release build with metrics (for profiling)
g++ -O3 -DPSYNE_DEBUG_METRICS=1 ...
```

## Performance Impact

The metrics system is designed for minimal overhead:

- **Basic metrics**: ~1-2% overhead
- **Histograms**: ~3-5% overhead  
- **Memory tracking**: ~5-10% overhead
- **Event tracing**: ~10-20% overhead

For production profiling, disable event tracing and memory tracking.

## Output Formats

### Console Output

Real-time statistics printed every sampling interval:

```
[channel_1] Rate: 1250000/1248000 msg/s, 4.77/4.76 MB/s, Lat(µs): 45/120 (p50/p99)
[channel_2] Rate: 850000/849500 msg/s, 3.24/3.24 MB/s, Lat(µs): 52/135 (p50/p99)
```

### CSV Export

Detailed metrics in CSV format:

```csv
timestamp_ms,channel,transport,msg_sent,msg_recv,bytes_sent,bytes_recv,msg_rate_send,msg_rate_recv,bandwidth_send_mbps,bandwidth_recv_mbps,latency_p50_us,latency_p99_us,bytes_used,bytes_available
1635360000000,test_channel,1,1000000,999950,4096000000,4095795200,1250000.00,1249937.50,4.77,4.77,45.2,120.5,8388608,8388608
```

### Monitor Dashboard

Live terminal dashboard with:
- Channel summary table
- Performance graphs
- Alert notifications
- Memory usage tracking

## Debugging Scenarios

### 1. Identifying Bottlenecks

```cpp
// Enable detailed histograms
config.detailed_histograms = true;

// Look for:
// - High p99 latency: Buffer contention
// - Rising buffer usage: Consumer too slow
// - Allocation failures: Buffer too small
```

### 2. Memory Leak Detection

```cpp
// Enable memory tracking
config.memory_tracking = true;

// Monitor:
// - current_allocations should return to 0
// - peak_allocations shows maximum usage
// - total_allocation_size tracks cumulative
```

### 3. Network Issues

```cpp
// For TCP channels, monitor:
// - Connection state changes
// - Send/receive rate imbalance
// - Latency spikes during reconnection
```

### 4. Performance Regression

```cpp
// Export baseline metrics
./psyne_monitor --csv baseline.csv

// After changes, compare:
./psyne_monitor --csv new.csv

# Use diff tools or Excel to compare CSV files
```

## Advanced Features

### Custom Alerts

```cpp
// Set up alert thresholds
if (latency_p99 > 1000) { // 1ms
    std::cerr << "ALERT: High latency detected!" << std::endl;
}

if (buffer_usage > 90) {
    std::cerr << "ALERT: Buffer nearly full!" << std::endl;
}
```

### Event Tracing

```cpp
// Enable event tracing for detailed analysis
config.event_tracing = true;
config.event_buffer_size = 100000;

// Analyze event sequence
auto events = MetricsCollector::instance().get_events();
for (const auto& event : events) {
    if (event.type == EventType::BUFFER_FULL) {
        std::cout << "Buffer full at " << event.timestamp_ns << std::endl;
    }
}
```

### Integration with External Tools

Export metrics for analysis in:
- **Grafana**: Use CSV export with Telegraf
- **Prometheus**: Implement HTTP endpoint
- **Python**: Load CSV for matplotlib/pandas analysis

Example Python analysis:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('psyne_metrics.csv')

# Plot latency over time
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp_ms'], df['latency_p50_us'], label='p50')
plt.plot(df['timestamp_ms'], df['latency_p99_us'], label='p99')
plt.xlabel('Time (ms)')
plt.ylabel('Latency (µs)')
plt.legend()
plt.show()
```

## Best Practices

1. **Development**: Enable all metrics features
2. **Testing**: Focus on latency and error tracking
3. **Benchmarking**: Disable event tracing, keep histograms
4. **Production**: Minimal metrics or compile-time disabled
5. **Debugging**: Enable event tracing temporarily

## Troubleshooting

### High Memory Usage

If metrics collection uses too much memory:
- Reduce `event_buffer_size`
- Increase `sampling_interval_ms`
- Disable `memory_tracking`

### Missing Metrics

If channels don't appear in monitoring:
- Ensure channel is registered
- Check that metrics are enabled
- Verify channel name matches

### Performance Impact

If metrics cause performance issues:
- Disable `event_tracing`
- Increase sampling interval
- Use compile-time flag to disable completely