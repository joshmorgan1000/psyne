#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <psyne/psyne.hpp>
#include <thread>
#include <vector>

using namespace psyne;
using namespace std::chrono;

// Fixed-size message for consistent latency measurements
class TimestampMessage : public FixedMessage<TimestampMessage, 16> {
public:
    static constexpr uint32_t message_type = 1000;

    template <typename Channel>
    explicit TimestampMessage(Channel &channel) : FixedMessage(channel) {}

    explicit TimestampMessage(const void *data, size_t size)
        : FixedMessage(data, size) {}

    void set_timestamp(uint64_t ts) {
        if (data_) {
            *reinterpret_cast<uint64_t *>(data_) = ts;
        }
    }

    uint64_t get_timestamp() const {
        return data_ ? *reinterpret_cast<const uint64_t *>(data_) : 0;
    }

    void set_sequence(uint64_t seq) {
        if (data_) {
            *reinterpret_cast<uint64_t *>(data_ + 8) = seq;
        }
    }

    uint64_t get_sequence() const {
        return data_ ? *reinterpret_cast<const uint64_t *>(data_ + 8) : 0;
    }
};

struct LatencyStats {
    double min;
    double max;
    double mean;
    double median;
    double p50;
    double p90;
    double p95;
    double p99;
    double p999;
    double stddev;
};

LatencyStats calculate_stats(std::vector<double> &latencies) {
    LatencyStats stats;

    if (latencies.empty()) {
        return stats;
    }

    std::sort(latencies.begin(), latencies.end());

    stats.min = latencies.front();
    stats.max = latencies.back();

    // Mean
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    stats.mean = sum / latencies.size();

    // Median
    size_t mid = latencies.size() / 2;
    stats.median = latencies.size() % 2 == 0
                       ? (latencies[mid - 1] + latencies[mid]) / 2.0
                       : latencies[mid];

    // Percentiles
    auto percentile = [&](double p) {
        size_t idx = static_cast<size_t>(latencies.size() * p);
        return latencies[std::min(idx, latencies.size() - 1)];
    };

    stats.p50 = percentile(0.50);
    stats.p90 = percentile(0.90);
    stats.p95 = percentile(0.95);
    stats.p99 = percentile(0.99);
    stats.p999 = percentile(0.999);

    // Standard deviation
    double sq_sum = 0;
    for (double l : latencies) {
        sq_sum += (l - stats.mean) * (l - stats.mean);
    }
    stats.stddev = std::sqrt(sq_sum / latencies.size());

    return stats;
}

template <typename ChannelType>
LatencyStats benchmark_ping_pong(size_t iterations) {
    // Create two channels for bidirectional communication
    ChannelType request_channel("memory://request", 1024 * 1024,
                                ChannelType::SingleType);
    ChannelType response_channel("memory://response", 1024 * 1024,
                                 ChannelType::SingleType);

    std::vector<double> latencies;
    latencies.reserve(iterations);

    // Responder thread
    std::thread responder([&]() {
        for (size_t i = 0; i < iterations; ++i) {
            auto msg =
                request_channel.template receive_single<TimestampMessage>();
            if (msg) {
                // Echo back immediately
                TimestampMessage response(response_channel);
                response.set_timestamp(msg->get_timestamp());
                response.set_sequence(msg->get_sequence());
                response_channel.send(response);
            }
        }
    });

    // Measure round-trip latency
    for (size_t i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();

        // Send request
        TimestampMessage request(request_channel);
        request.set_timestamp(start.time_since_epoch().count());
        request.set_sequence(i);
        request_channel.send(request);

        // Wait for response
        auto response =
            response_channel.template receive_single<TimestampMessage>();

        auto end = high_resolution_clock::now();

        if (response && response->get_sequence() == i) {
            auto latency_ns = duration_cast<nanoseconds>(end - start).count();
            latencies.push_back(latency_ns / 1000.0); // Convert to microseconds
        }
    }

    responder.join();

    return calculate_stats(latencies);
}

void print_stats(const std::string &name, const LatencyStats &stats) {
    std::cout << "\n" << name << " Latency Statistics (μs):" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Min:    " << std::setw(10) << stats.min << std::endl;
    std::cout << "Mean:   " << std::setw(10) << stats.mean << std::endl;
    std::cout << "Median: " << std::setw(10) << stats.median << std::endl;
    std::cout << "Max:    " << std::setw(10) << stats.max << std::endl;
    std::cout << "StdDev: " << std::setw(10) << stats.stddev << std::endl;
    std::cout << "\nPercentiles:" << std::endl;
    std::cout << "  50%:  " << std::setw(10) << stats.p50 << std::endl;
    std::cout << "  90%:  " << std::setw(10) << stats.p90 << std::endl;
    std::cout << "  95%:  " << std::setw(10) << stats.p95 << std::endl;
    std::cout << "  99%:  " << std::setw(10) << stats.p99 << std::endl;
    std::cout << "  99.9%:" << std::setw(10) << stats.p999 << std::endl;
}

int main() {
    std::cout << "Psyne Latency Benchmark" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << "Measuring round-trip ping-pong latency" << std::endl;

    const size_t warmup_iterations = 1000;
    const size_t test_iterations = 10000;

    // Warmup
    std::cout << "\nWarming up..." << std::endl;
    benchmark_ping_pong<SPSCChannel>(warmup_iterations);

    // Run benchmarks
    std::cout << "\nRunning benchmarks (" << test_iterations
              << " iterations each)..." << std::endl;

    // SPSC - should have the lowest latency
    auto spsc_stats = benchmark_ping_pong<SPSCChannel>(test_iterations);
    print_stats("SPSC", spsc_stats);

    // SPMC
    auto spmc_stats = benchmark_ping_pong<SPMCChannel>(test_iterations);
    print_stats("SPMC", spmc_stats);

    // MPSC
    auto mpsc_stats = benchmark_ping_pong<MPSCChannel>(test_iterations);
    print_stats("MPSC", mpsc_stats);

    // MPMC
    auto mpmc_stats = benchmark_ping_pong<MPMCChannel>(test_iterations);
    print_stats("MPMC", mpmc_stats);

    // Summary comparison
    std::cout << "\n\nSummary Comparison:" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::setw(10) << "Channel" << std::setw(12) << "Mean (μs)"
              << std::setw(12) << "P99 (μs)" << std::setw(12) << "P99.9 (μs)"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    std::cout << std::setw(10) << "SPSC" << std::setw(12) << spsc_stats.mean
              << std::setw(12) << spsc_stats.p99 << std::setw(12)
              << spsc_stats.p999 << std::endl;

    std::cout << std::setw(10) << "SPMC" << std::setw(12) << spmc_stats.mean
              << std::setw(12) << spmc_stats.p99 << std::setw(12)
              << spmc_stats.p999 << std::endl;

    std::cout << std::setw(10) << "MPSC" << std::setw(12) << mpsc_stats.mean
              << std::setw(12) << mpsc_stats.p99 << std::setw(12)
              << mpsc_stats.p999 << std::endl;

    std::cout << std::setw(10) << "MPMC" << std::setw(12) << mpmc_stats.mean
              << std::setw(12) << mpmc_stats.p99 << std::setw(12)
              << mpmc_stats.p999 << std::endl;

    return 0;
}