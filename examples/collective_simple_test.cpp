/**
 * @file collective_simple_test.cpp
 * @brief Simple test for collective operations with in-memory channels
 *
 * This test runs multiple "ranks" in threads to simulate distributed execution
 */

#include <atomic>
#include <barrier>
#include <iostream>
#include <psyne/collective.hpp>
#include <thread>
#include <vector>

using namespace psyne;
using namespace psyne::collective;

// Thread barrier for synchronization
std::barrier<> thread_barrier(3);
std::atomic<int> errors{0};

// Simple in-memory collective group for testing
class InMemoryCollectiveGroup : public CollectiveGroup {
public:
    InMemoryCollectiveGroup(RankId rank, size_t world_size)
        : rank_(rank), world_size_(world_size) {
        // Create memory channels for communication
        for (size_t i = 0; i < world_size; ++i) {
            if (i != rank_) {
                std::string uri = "memory://collective_" +
                                  std::to_string(rank_) + "_to_" +
                                  std::to_string(i);
                channels_[i] =
                    create_channel(uri, 1024 * 1024, ChannelMode::SPSC);
            }
        }
    }

    RankId rank() const override {
        return rank_;
    }
    size_t size() const override {
        return world_size_;
    }

    std::shared_ptr<Channel> get_channel(RankId target) override {
        auto it = channels_.find(target);
        if (it != channels_.end()) {
            return it->second;
        }

        // Create reverse channel for receiving
        std::string uri = "memory://collective_" + std::to_string(target) +
                          "_to_" + std::to_string(rank_);
        return create_channel(uri, 1024 * 1024, ChannelMode::SPSC);
    }

    void barrier() override {
        thread_barrier.arrive_and_wait();
    }

private:
    RankId rank_;
    size_t world_size_;
    std::map<RankId, std::shared_ptr<Channel>> channels_;
};

void test_broadcast(RankId rank) {
    auto group = std::make_shared<InMemoryCollectiveGroup>(rank, 3);

    // Test data
    std::vector<int> data(5);

    if (rank == 0) {
        // Root initializes data
        data = {10, 20, 30, 40, 50};
        std::cout << "[Rank " << rank << "] Broadcasting: ";
        for (int v : data)
            std::cout << v << " ";
        std::cout << std::endl;
    } else {
        // Others start with zeros
        std::fill(data.begin(), data.end(), 0);
    }

    // Execute broadcast
    Broadcast<int> bcast(group);
    bcast.execute(std::span<int>(data), 0);

    // Verify all ranks have the same data
    bool correct = (data == std::vector<int>{10, 20, 30, 40, 50});
    if (!correct) {
        std::cerr << "[Rank " << rank << "] Broadcast failed!" << std::endl;
        errors++;
    } else {
        std::cout << "[Rank " << rank << "] Broadcast successful!" << std::endl;
    }
}

void test_allreduce(RankId rank) {
    auto group = std::make_shared<InMemoryCollectiveGroup>(rank, 3);

    // Each rank contributes rank+1
    std::vector<float> data = {static_cast<float>(rank + 1),
                               static_cast<float>(rank + 1) * 2};

    std::cout << "[Rank " << rank << "] All-reduce input: ";
    for (float v : data)
        std::cout << v << " ";
    std::cout << std::endl;

    // Execute all-reduce (sum)
    AllReduce<float> allreduce(group);
    allreduce.execute(std::span<float>(data), ReduceOp::Sum);

    // Expected: sum of all ranks (1+2+3=6, 2+4+6=12)
    bool correct = (std::abs(data[0] - 6.0f) < 0.001f &&
                    std::abs(data[1] - 12.0f) < 0.001f);

    if (!correct) {
        std::cerr << "[Rank " << rank << "] All-reduce failed! Got: " << data[0]
                  << ", " << data[1] << std::endl;
        errors++;
    } else {
        std::cout << "[Rank " << rank
                  << "] All-reduce successful! Result: " << data[0] << ", "
                  << data[1] << std::endl;
    }
}

void test_scatter_gather(RankId rank) {
    auto group = std::make_shared<InMemoryCollectiveGroup>(rank, 3);

    // Scatter test
    std::vector<int> scatter_send;
    std::vector<int> scatter_recv(2); // Each rank gets 2 elements

    if (rank == 0) {
        scatter_send = {10, 11, 20, 21, 30, 31}; // 2 elements per rank
    }

    Scatter<int> scatter(group);
    scatter.execute(std::span<const int>(scatter_send),
                    std::span<int>(scatter_recv), 0);

    // Verify scatter
    bool scatter_correct = (scatter_recv[0] == (rank + 1) * 10 &&
                            scatter_recv[1] == (rank + 1) * 10 + 1);

    if (!scatter_correct) {
        std::cerr << "[Rank " << rank << "] Scatter failed!" << std::endl;
        errors++;
    } else {
        std::cout << "[Rank " << rank
                  << "] Scatter successful! Received: " << scatter_recv[0]
                  << ", " << scatter_recv[1] << std::endl;
    }

    // Gather test
    std::vector<int> gather_send = {rank * 100, rank * 100 + 1};
    std::vector<int> gather_recv;

    if (rank == 0) {
        gather_recv.resize(6); // 2 elements from each of 3 ranks
    }

    Gather<int> gather(group);
    gather.execute(std::span<const int>(gather_send),
                   std::span<int>(gather_recv), 0);

    if (rank == 0) {
        bool gather_correct = true;
        for (size_t i = 0; i < 3; ++i) {
            if (gather_recv[i * 2] != i * 100 ||
                gather_recv[i * 2 + 1] != i * 100 + 1) {
                gather_correct = false;
                break;
            }
        }

        if (!gather_correct) {
            std::cerr << "[Rank " << rank << "] Gather failed!" << std::endl;
            errors++;
        } else {
            std::cout << "[Rank " << rank << "] Gather successful! Received: ";
            for (int v : gather_recv)
                std::cout << v << " ";
            std::cout << std::endl;
        }
    }
}

void run_rank_tests(RankId rank) {
    std::cout << "\n=== Starting tests for Rank " << rank
              << " ===" << std::endl;

    // Wait for all threads to be ready
    thread_barrier.arrive_and_wait();

    // Test broadcast
    std::cout << "\n--- Broadcast Test ---" << std::endl;
    test_broadcast(rank);
    thread_barrier.arrive_and_wait();

    // Test all-reduce
    std::cout << "\n--- All-Reduce Test ---" << std::endl;
    test_allreduce(rank);
    thread_barrier.arrive_and_wait();

    // Test scatter/gather
    std::cout << "\n--- Scatter/Gather Test ---" << std::endl;
    test_scatter_gather(rank);
    thread_barrier.arrive_and_wait();
}

int main() {
    std::cout << "=== Collective Operations Test ===" << std::endl;
    std::cout << "Running 3 ranks in separate threads..." << std::endl;

    // Create threads for each rank
    std::vector<std::thread> threads;
    for (CollectiveGroup::RankId rank = 0; rank < 3; ++rank) {
        threads.emplace_back(run_rank_tests, rank);
    }

    // Wait for all threads to complete
    for (auto &t : threads) {
        t.join();
    }

    // Report results
    std::cout << "\n=== Test Summary ===" << std::endl;
    if (errors == 0) {
        std::cout << "✅ All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ " << errors << " tests failed!" << std::endl;
        return 1;
    }
}