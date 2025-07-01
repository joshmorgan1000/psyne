/**
 * @file beast_mode_test.cpp
 * @brief MAXIMUM THREAD COUNT STRESS TEST - LET'S GOOOOO!
 */

#include "../include/psyne/core/behaviors.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// Test message
struct TestMessage {
    uint64_t id;
    uint64_t timestamp;
    uint32_t producer_id;
    char data[32];
};

// ===== SUBSTRATES (reusing from before) =====
class QuickInProcess : public psyne::behaviors::SubstrateBehavior {
public:
    void *allocate_memory_slab(size_t size_bytes) override {
        allocated_memory_ = std::aligned_alloc(64, size_bytes);
        return allocated_memory_;
    }
    void deallocate_memory_slab(void *memory) override {
        if (memory == allocated_memory_)
            std::free(memory);
    }
    void transport_send(void *data, size_t size) override {}
    void transport_receive(void *buffer, size_t buffer_size) override {}
    const char *substrate_name() const override {
        return "InProcess";
    }
    bool is_zero_copy() const override {
        return true;
    }
    bool is_cross_process() const override {
        return false;
    }

private:
    void *allocated_memory_ = nullptr;
};

// ===== PATTERNS (with bigger buffers!) =====
class BeastSPSC : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        if (!slab_memory_) {
            slab_memory_ = slab_memory;
            message_size_ = message_size;
        }
        size_t slot = write_pos_.fetch_add(1) % buffer_size_;
        return static_cast<char *>(slab_memory) + (slot * message_size);
    }
    void *coordinate_receive() override {
        size_t current_read = read_pos_.load();
        size_t current_write = write_pos_.load();
        if (current_read >= current_write)
            return nullptr;
        size_t slot = current_read % buffer_size_;
        read_pos_.fetch_add(1);
        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }
    void producer_sync() override {}
    void consumer_sync() override {}
    const char *pattern_name() const override {
        return "SPSC";
    }
    bool needs_locks() const override {
        return false;
    }
    size_t max_producers() const override {
        return 1;
    }
    size_t max_consumers() const override {
        return 1;
    }

private:
    static constexpr size_t buffer_size_ = 1024 * 1024; // 1M slots!
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
};

class BeastMPSC : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        if (!slab_memory_) {
            std::lock_guard<std::mutex> lock(init_mutex_);
            if (!slab_memory_) {
                slab_memory_ = slab_memory;
                message_size_ = message_size;
            }
        }
        size_t slot = write_pos_.fetch_add(1) % buffer_size_;
        return static_cast<char *>(slab_memory) + (slot * message_size);
    }
    void *coordinate_receive() override {
        size_t current_read = read_pos_.load();
        size_t current_write = write_pos_.load();
        if (current_read >= current_write)
            return nullptr;
        size_t slot = current_read % buffer_size_;
        read_pos_.fetch_add(1);
        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }
    void producer_sync() override {}
    void consumer_sync() override {}
    const char *pattern_name() const override {
        return "MPSC";
    }
    bool needs_locks() const override {
        return true;
    }
    size_t max_producers() const override {
        return SIZE_MAX;
    }
    size_t max_consumers() const override {
        return 1;
    }

private:
    static constexpr size_t buffer_size_ = 1024 * 1024;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
    std::mutex init_mutex_;
};

class BeastSPMC : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        if (!slab_memory_) {
            slab_memory_ = slab_memory;
            message_size_ = message_size;
        }
        size_t slot = write_pos_.fetch_add(1) % buffer_size_;
        return static_cast<char *>(slab_memory) + (slot * message_size);
    }
    void *coordinate_receive() override {
        size_t current_read = read_pos_.fetch_add(1);
        size_t current_write = write_pos_.load();
        if (current_read >= current_write) {
            read_pos_.fetch_sub(1);
            return nullptr;
        }
        size_t slot = current_read % buffer_size_;
        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }
    void producer_sync() override {}
    void consumer_sync() override {}
    const char *pattern_name() const override {
        return "SPMC";
    }
    bool needs_locks() const override {
        return true;
    }
    size_t max_producers() const override {
        return 1;
    }
    size_t max_consumers() const override {
        return SIZE_MAX;
    }

private:
    static constexpr size_t buffer_size_ = 1024 * 1024;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
};

class BeastMPMC : public psyne::behaviors::PatternBehavior {
public:
    void *coordinate_allocation(void *slab_memory,
                                size_t message_size) override {
        if (!slab_memory_) {
            std::lock_guard<std::mutex> lock(init_mutex_);
            if (!slab_memory_) {
                slab_memory_ = slab_memory;
                message_size_ = message_size;
            }
        }
        size_t current_write = write_pos_.load();
        size_t current_read = read_pos_.load();
        if (current_write - current_read >= buffer_size_)
            return nullptr;
        size_t slot = write_pos_.fetch_add(1) % buffer_size_;
        return static_cast<char *>(slab_memory) + (slot * message_size);
    }
    void *coordinate_receive() override {
        size_t current_read = read_pos_.fetch_add(1);
        size_t current_write = write_pos_.load();
        if (current_read >= current_write) {
            read_pos_.fetch_sub(1);
            return nullptr;
        }
        size_t slot = current_read % buffer_size_;
        return static_cast<char *>(slab_memory_) + (slot * message_size_);
    }
    void producer_sync() override {}
    void consumer_sync() override {}
    const char *pattern_name() const override {
        return "MPMC";
    }
    bool needs_locks() const override {
        return true;
    }
    size_t max_producers() const override {
        return SIZE_MAX;
    }
    size_t max_consumers() const override {
        return SIZE_MAX;
    }

private:
    static constexpr size_t buffer_size_ = 1024 * 1024;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    void *slab_memory_ = nullptr;
    size_t message_size_ = 0;
    std::mutex init_mutex_;
};

// BEAST MODE TEST RUNNER
template <typename PatternType>
void beast_test(const std::string &pattern_name, size_t num_producers,
                size_t num_consumers, int duration_sec = 10) {
    std::cout << "\n🔥 " << pattern_name << " with " << num_producers << "P × "
              << num_consumers << "C";
    std::cout << " (" << duration_sec << "s test)\n";
    std::cout << std::string(60, '=') << "\n";

    try {
        using ChannelType =
            psyne::behaviors::ChannelBridge<TestMessage, QuickInProcess,
                                            PatternType>;
        ChannelType channel(256 * 1024 * 1024); // 256MB slab for BEAST MODE!

        std::atomic<size_t> sent{0}, received{0};
        std::atomic<bool> running{true};
        std::atomic<size_t> producer_errors{0}, consumer_errors{0};

        auto start_time = std::chrono::steady_clock::now();

        // UNLEASH THE PRODUCERS!
        std::vector<std::thread> producers;
        for (size_t i = 0; i < num_producers; ++i) {
            producers.emplace_back([&, id = i]() {
                size_t local_sent = 0;
                while (running.load()) {
                    try {
                        auto msg = channel.create_message();
                        msg->id = local_sent++;
                        msg->producer_id = id;
                        msg->timestamp =
                            std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count();
                        channel.send_message(msg);
                        sent.fetch_add(1);
                    } catch (...) {
                        producer_errors.fetch_add(1);
                        std::this_thread::yield();
                    }
                }
            });
        }

        // UNLEASH THE CONSUMERS!
        std::vector<std::thread> consumers;
        for (size_t i = 0; i < num_consumers; ++i) {
            consumers.emplace_back([&, id = i]() {
                size_t local_received = 0;
                while (running.load() || sent.load() > received.load()) {
                    auto msg_opt = channel.try_receive();
                    if (msg_opt) {
                        received.fetch_add(1);
                        local_received++;
                    } else {
                        std::this_thread::yield();
                    }
                }
            });
        }

        // Let it RIP!
        std::this_thread::sleep_for(std::chrono::seconds(duration_sec));
        running.store(false);

        // Wait for everyone
        for (auto &p : producers)
            p.join();
        for (auto &c : consumers)
            c.join();

        auto end_time = std::chrono::steady_clock::now();
        double actual_duration =
            std::chrono::duration<double>(end_time - start_time).count();

        // RESULTS!
        double throughput = received.load() / actual_duration;
        double throughput_millions = throughput / 1'000'000.0;

        std::cout << "Messages sent:     " << std::setw(12) << sent.load()
                  << "\n";
        std::cout << "Messages received: " << std::setw(12) << received.load()
                  << "\n";
        std::cout << "Producer errors:   " << std::setw(12)
                  << producer_errors.load() << "\n";
        std::cout << "Duration:          " << std::fixed << std::setprecision(2)
                  << actual_duration << "s\n";
        std::cout << "Throughput:        " << std::fixed << std::setprecision(2)
                  << throughput_millions << "M msgs/s";
        std::cout << " (" << std::fixed << std::setprecision(0) << throughput
                  << " msgs/s)\n";

        // Per-thread stats
        if (num_producers > 1) {
            std::cout << "Per producer:      " << std::fixed
                      << std::setprecision(2)
                      << (sent.load() / (double)num_producers /
                          actual_duration / 1'000'000.0)
                      << "M msgs/s\n";
        }
        if (num_consumers > 1) {
            std::cout << "Per consumer:      " << std::fixed
                      << std::setprecision(2)
                      << (received.load() / (double)num_consumers /
                          actual_duration / 1'000'000.0)
                      << "M msgs/s\n";
        }

        std::cout << "Status:            ";
        if (received.load() >= sent.load() * 0.99) {
            std::cout << "✅ BEAST MODE ACTIVATED!\n";
        } else {
            std::cout << "⚠️  Some messages lost\n";
        }

    } catch (const std::exception &e) {
        std::cout << "❌ FAILED: " << e.what() << "\n";
    }
}

int main() {
    auto cores = std::thread::hardware_concurrency();
    std::cout << "🚀 PSYNE v2.0 BEAST MODE BENCHMARK\n";
    std::cout << "==================================\n";
    std::cout << "System cores: " << cores << "\n";
    std::cout << "Let's see what this baby can do!\n";

    // WARM UP with standard configs
    std::cout << "\n📊 BASELINE TESTS (10s each)\n";
    beast_test<BeastSPSC>("SPSC", 1, 1);
    beast_test<BeastMPSC>("MPSC", 4, 1);
    beast_test<BeastSPMC>("SPMC", 1, 4);
    beast_test<BeastMPMC>("MPMC", 4, 4);

    // NOW LET'S GET CRAZY!
    std::cout << "\n⚡ HIGH THREAD COUNT TESTS\n";

    // MPSC scaling
    beast_test<BeastMPSC>("MPSC", 8, 1);
    beast_test<BeastMPSC>("MPSC", 16, 1);
    beast_test<BeastMPSC>("MPSC", 32, 1);
    if (cores >= 64)
        beast_test<BeastMPSC>("MPSC", 64, 1);

    // SPMC scaling - THIS IS THE MONEY MAKER FOR MANIFOLDB!
    beast_test<BeastSPMC>("SPMC", 1, 8);
    beast_test<BeastSPMC>("SPMC", 1, 16);
    beast_test<BeastSPMC>("SPMC", 1, 32);
    if (cores >= 64)
        beast_test<BeastSPMC>("SPMC", 1, 64);

    // MPMC MADNESS
    beast_test<BeastMPMC>("MPMC", 8, 8);
    beast_test<BeastMPMC>("MPMC", 16, 16);
    if (cores >= 32)
        beast_test<BeastMPMC>("MPMC", 32, 32);

    // EXTREME ASYMMETRIC TESTS
    std::cout << "\n💀 EXTREME ASYMMETRIC TESTS\n";
    beast_test<BeastMPSC>("MPSC", cores * 2, 1); // 2x oversubscription
    beast_test<BeastSPMC>("SPMC", 1, cores * 2); // 2x oversubscription

    // THE ULTIMATE TEST
    std::cout << "\n🔥🔥🔥 THE ULTIMATE BEAST MODE TEST 🔥🔥🔥\n";
    beast_test<BeastMPMC>("MPMC", cores, cores, 30); // 30 second ultimate test!

    std::cout << "\n🎉 BEAST MODE COMPLETE!\n";
    std::cout << "📊 Results ready for Psynetics!\n";
    std::cout << "🚀 SHIP IT!\n";

    return 0;
}