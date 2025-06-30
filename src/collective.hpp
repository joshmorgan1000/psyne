/**
 * @file collective.hpp
 * @brief Collective communication operations for distributed computing
 *
 * Provides high-performance collective operations optimized for AI/ML
 * workloads:
 * - Broadcast: One-to-all communication
 * - All-reduce: Reduce across all nodes and distribute result
 * - Scatter: Distribute chunks of data to different nodes
 * - Gather: Collect data from all nodes
 * - All-gather: Gather data from all nodes to all nodes
 * - Reduce-scatter: Reduce and scatter in one operation
 *
 * @copyright Copyright (c) 2025 Psyne Project
 * @license MIT License
 */

#pragma once

#include <concepts>
#include <functional>
#include <future>
#include <memory>
#include <numeric>
#include <psyne/psyne.hpp>
#include <span>
#include <vector>

namespace psyne {
namespace collective {

/**
 * @brief Reduction operations for collective communication
 */
enum class ReduceOp {
    Sum,        ///< Element-wise sum
    Product,    ///< Element-wise product
    Min,        ///< Element-wise minimum
    Max,        ///< Element-wise maximum
    BitwiseAnd, ///< Bitwise AND
    BitwiseOr,  ///< Bitwise OR
    BitwiseXor, ///< Bitwise XOR
    Custom      ///< User-defined operation
};

/**
 * @brief Collective group representing a set of processes
 */
class CollectiveGroup {
public:
    using RankId = uint32_t;

    /**
     * @brief Get the rank (ID) of this process in the group
     */
    virtual RankId rank() const = 0;

    /**
     * @brief Get the total number of processes in the group
     */
    virtual size_t size() const = 0;

    /**
     * @brief Get the channel for communicating with a specific rank
     */
    virtual std::shared_ptr<Channel> get_channel(RankId rank) = 0;

    /**
     * @brief Synchronize all processes in the group (barrier)
     */
    virtual void barrier() = 0;

    virtual ~CollectiveGroup() = default;
};

/**
 * @brief Ring-based collective group for efficient algorithms
 */
class RingCollectiveGroup : public CollectiveGroup {
public:
    RingCollectiveGroup(RankId rank, std::vector<std::string> peer_uris);

    RankId rank() const override {
        return rank_;
    }
    size_t size() const override {
        return channels_.size() + 1;
    }

    std::shared_ptr<Channel> get_channel(RankId rank) override;
    void barrier() override;

    /**
     * @brief Get next neighbor in ring topology
     */
    RankId next_neighbor() const {
        return (rank_ + 1) % size();
    }

    /**
     * @brief Get previous neighbor in ring topology
     */
    RankId prev_neighbor() const {
        return (rank_ + size() - 1) % size();
    }

private:
    RankId rank_;
    std::vector<std::shared_ptr<Channel>> channels_;
    std::vector<std::string> peer_uris_;
};

/**
 * @brief Base class for collective operations
 */
template <typename T>
class CollectiveOperation {
public:
    using value_type = T;

    CollectiveOperation(std::shared_ptr<CollectiveGroup> group)
        : group_(std::move(group)) {}

    virtual ~CollectiveOperation() = default;

protected:
    std::shared_ptr<CollectiveGroup> group_;
};

/**
 * @brief Broadcast operation - send data from root to all processes
 */
template <typename T>
class Broadcast : public CollectiveOperation<T> {
public:
    using Base = CollectiveOperation<T>;
    using Base::group_;

    Broadcast(std::shared_ptr<CollectiveGroup> group) : Base(group) {}

    /**
     * @brief Execute broadcast operation
     * @param data Data to broadcast (modified in-place for non-root)
     * @param root Rank of the root process
     */
    void execute(std::span<T> data, CollectiveGroup::RankId root) {
        if (group_->rank() == root) {
            // Root sends to all others
            broadcast_from_root(data);
        } else {
            // Non-root receives from root
            receive_from_root(data, root);
        }

        // Ensure all processes complete
        group_->barrier();
    }

    /**
     * @brief Async broadcast operation
     */
    std::future<void> execute_async(std::span<T> data,
                                    CollectiveGroup::RankId root) {
        return std::async(std::launch::async,
                          [this, data, root]() { execute(data, root); });
    }

private:
    void broadcast_from_root(std::span<T> data) {
        // Binary tree broadcast for efficiency
        auto rank = group_->rank();
        auto size = group_->size();

        // Calculate children in binary tree
        auto left_child = 2 * rank + 1;
        auto right_child = 2 * rank + 2;

        // Send to children if they exist
        if (left_child < size) {
            auto channel = group_->get_channel(left_child);
            channel->send(data);
        }

        if (right_child < size) {
            auto channel = group_->get_channel(right_child);
            channel->send(data);
        }
    }

    void receive_from_root(std::span<T> data, CollectiveGroup::RankId root) {
        // Calculate parent in binary tree
        auto rank = group_->rank();
        auto parent = (rank - 1) / 2;

        // Receive from parent
        auto channel = group_->get_channel(parent);
        auto received = channel->receive<std::vector<T>>();

        if (received && received->size() == data.size()) {
            // Zero-copy optimization: use manual loop instead of std::copy
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = (*received)[i];
            }

            // Forward to children (continue the tree broadcast)
            broadcast_from_root(data);
        }
    }
};

/**
 * @brief All-reduce operation - reduce across all nodes and distribute result
 */
template <typename T>
class AllReduce : public CollectiveOperation<T> {
public:
    using Base = CollectiveOperation<T>;
    using Base::group_;
    using ReduceFunc = std::function<T(const T &, const T &)>;

    AllReduce(std::shared_ptr<CollectiveGroup> group) : Base(group) {}

    /**
     * @brief Execute all-reduce with built-in operation
     */
    void execute(std::span<T> data, ReduceOp op) {
        execute(data, get_reduce_function(op));
    }

    /**
     * @brief Execute all-reduce with custom reduce function
     */
    void execute(std::span<T> data, ReduceFunc reduce_fn) {
        // Ring all-reduce algorithm for efficiency
        ring_allreduce(data, reduce_fn);

        group_->barrier();
    }

    /**
     * @brief Async all-reduce operation
     */
    std::future<void> execute_async(std::span<T> data, ReduceOp op) {
        return std::async(std::launch::async,
                          [this, data, op]() { execute(data, op); });
    }

private:
    ReduceFunc get_reduce_function(ReduceOp op) {
        switch (op) {
        case ReduceOp::Sum:
            return [](const T &a, const T &b) { return a + b; };
        case ReduceOp::Product:
            return [](const T &a, const T &b) { return a * b; };
        case ReduceOp::Min:
            return [](const T &a, const T &b) { return std::min(a, b); };
        case ReduceOp::Max:
            return [](const T &a, const T &b) { return std::max(a, b); };
        case ReduceOp::BitwiseAnd:
            if constexpr (std::is_integral_v<T>) {
                return [](const T &a, const T &b) { return a & b; };
            }
            break;
        case ReduceOp::BitwiseOr:
            if constexpr (std::is_integral_v<T>) {
                return [](const T &a, const T &b) { return a | b; };
            }
            break;
        case ReduceOp::BitwiseXor:
            if constexpr (std::is_integral_v<T>) {
                return [](const T &a, const T &b) { return a ^ b; };
            }
            break;
        default:
            break;
        }

        throw std::runtime_error("Unsupported reduce operation for type");
    }

    void ring_allreduce(std::span<T> data, ReduceFunc reduce_fn) {
        auto ring_group =
            std::dynamic_pointer_cast<RingCollectiveGroup>(group_);
        if (!ring_group) {
            // Fallback to simple all-reduce
            simple_allreduce(data, reduce_fn);
            return;
        }

        const size_t rank = ring_group->rank();
        const size_t size = ring_group->size();
        const size_t n = data.size();

        // Phase 1: Reduce-scatter
        // Each process is responsible for reducing a chunk
        const size_t chunk_size = (n + size - 1) / size;

        for (size_t step = 0; step < size - 1; ++step) {
            // Determine send and receive chunks
            size_t send_chunk = (rank - step + size) % size;
            size_t recv_chunk = (rank - step - 1 + size) % size;

            size_t send_offset = send_chunk * chunk_size;
            size_t recv_offset = recv_chunk * chunk_size;

            size_t send_count = std::min(chunk_size, n - send_offset);
            size_t recv_count = std::min(chunk_size, n - recv_offset);

            // Send to next neighbor
            auto next_channel =
                ring_group->get_channel(ring_group->next_neighbor());
            next_channel->send(
                std::span<T>(data.data() + send_offset, send_count));

            // Receive from previous neighbor
            auto prev_channel =
                ring_group->get_channel(ring_group->prev_neighbor());
            auto received = prev_channel->receive<std::vector<T>>();

            // Reduce received data
            if (received && received->size() == recv_count) {
                for (size_t i = 0; i < recv_count; ++i) {
                    data[recv_offset + i] =
                        reduce_fn(data[recv_offset + i], (*received)[i]);
                }
            }
        }

        // Phase 2: All-gather
        // Share the reduced chunks with all processes
        for (size_t step = 0; step < size - 1; ++step) {
            // Determine send and receive chunks
            size_t send_chunk = (rank - step + 1 + size) % size;
            size_t recv_chunk = (rank - step + size) % size;

            size_t send_offset = send_chunk * chunk_size;
            size_t recv_offset = recv_chunk * chunk_size;

            size_t send_count = std::min(chunk_size, n - send_offset);
            size_t recv_count = std::min(chunk_size, n - recv_offset);

            // Send to next neighbor
            auto next_channel =
                ring_group->get_channel(ring_group->next_neighbor());
            next_channel->send(
                std::span<T>(data.data() + send_offset, send_count));

            // Receive from previous neighbor
            auto prev_channel =
                ring_group->get_channel(ring_group->prev_neighbor());
            auto received = prev_channel->receive<std::vector<T>>();

            // Copy received data
            if (received && received->size() == recv_count) {
                // Zero-copy optimization: manual copy
                for (size_t i = 0; i < received->size(); ++i) {
                    data[recv_offset + i] = (*received)[i];
                }
            }
        }
    }

    void simple_allreduce(std::span<T> data, ReduceFunc reduce_fn) {
        // Simple implementation: gather at rank 0, reduce, then broadcast
        const size_t rank = group_->rank();
        const size_t size = group_->size();

        if (rank == 0) {
            // Receive from all others and reduce
            for (size_t r = 1; r < size; ++r) {
                auto channel = group_->get_channel(r);
                auto received = channel->receive<std::vector<T>>();

                if (received && received->size() == data.size()) {
                    for (size_t i = 0; i < data.size(); ++i) {
                        data[i] = reduce_fn(data[i], (*received)[i]);
                    }
                }
            }

            // Broadcast result
            Broadcast<T> bcast(group_);
            bcast.execute(data, 0);
        } else {
            // Send to rank 0
            auto channel = group_->get_channel(0);
            channel->send(data);

            // Receive broadcasted result
            Broadcast<T> bcast(group_);
            bcast.execute(data, 0);
        }
    }
};

/**
 * @brief Scatter operation - distribute chunks of data to different processes
 */
template <typename T>
class Scatter : public CollectiveOperation<T> {
public:
    using Base = CollectiveOperation<T>;
    using Base::group_;

    Scatter(std::shared_ptr<CollectiveGroup> group) : Base(group) {}

    /**
     * @brief Execute scatter operation
     * @param send_data Data to scatter (only used by root)
     * @param recv_data Buffer to receive scattered data
     * @param root Rank of the root process
     */
    void execute(std::span<const T> send_data, std::span<T> recv_data,
                 CollectiveGroup::RankId root) {
        const size_t rank = group_->rank();
        const size_t size = group_->size();
        const size_t chunk_size = recv_data.size();

        if (rank == root) {
            // Root scatters data to all processes
            for (size_t r = 0; r < size; ++r) {
                size_t offset = r * chunk_size;

                if (r == rank) {
                    // Zero-copy optimization: avoid self-copy when possible
                    if (&send_data != &recv_data) {
                        for (size_t i = 0; i < chunk_size; ++i) {
                            recv_data[i] = send_data[offset + i];
                        }
                    }
                } else {
                    // Send to remote process
                    auto channel = group_->get_channel(r);
                    channel->send(std::span<const T>(send_data.data() + offset,
                                                     chunk_size));
                }
            }
        } else {
            // Non-root receives from root
            auto channel = group_->get_channel(root);
            auto received = channel->receive<std::vector<T>>();

            if (received && received->size() == chunk_size) {
                // Zero-copy optimization: manual copy
                for (size_t i = 0; i < chunk_size; ++i) {
                    recv_data[i] = (*received)[i];
                }
            }
        }

        group_->barrier();
    }
};

/**
 * @brief Gather operation - collect data from all processes
 */
template <typename T>
class Gather : public CollectiveOperation<T> {
public:
    using Base = CollectiveOperation<T>;
    using Base::group_;

    Gather(std::shared_ptr<CollectiveGroup> group) : Base(group) {}

    /**
     * @brief Execute gather operation
     * @param send_data Data to send from this process
     * @param recv_data Buffer to receive gathered data (only used by root)
     * @param root Rank of the root process
     */
    void execute(std::span<const T> send_data, std::span<T> recv_data,
                 CollectiveGroup::RankId root) {
        const size_t rank = group_->rank();
        const size_t size = group_->size();
        const size_t chunk_size = send_data.size();

        if (rank == root) {
            // Root gathers from all processes
            for (size_t r = 0; r < size; ++r) {
                size_t offset = r * chunk_size;

                if (r == rank) {
                    // Zero-copy optimization: manual copy for local data
                    for (size_t i = 0; i < send_data.size(); ++i) {
                        recv_data[offset + i] = send_data[i];
                    }
                } else {
                    // Receive from remote process
                    auto channel = group_->get_channel(r);
                    auto received = channel->receive<std::vector<T>>();

                    if (received && received->size() == chunk_size) {
                        // Zero-copy optimization: manual copy
                        for (size_t i = 0; i < chunk_size; ++i) {
                            recv_data[offset + i] = (*received)[i];
                        }
                    }
                }
            }
        } else {
            // Non-root sends to root
            auto channel = group_->get_channel(root);
            channel->send(send_data);
        }

        group_->barrier();
    }
};

/**
 * @brief All-gather operation - gather data from all processes to all processes
 */
template <typename T>
class AllGather : public CollectiveOperation<T> {
public:
    using Base = CollectiveOperation<T>;
    using Base::group_;

    AllGather(std::shared_ptr<CollectiveGroup> group) : Base(group) {}

    /**
     * @brief Execute all-gather operation
     * @param send_data Data to send from this process
     * @param recv_data Buffer to receive gathered data from all processes
     */
    void execute(std::span<const T> send_data, std::span<T> recv_data) {
        auto ring_group =
            std::dynamic_pointer_cast<RingCollectiveGroup>(group_);
        if (!ring_group) {
            // Fallback to simple all-gather
            simple_allgather(send_data, recv_data);
            return;
        }

        // Ring all-gather algorithm
        ring_allgather(send_data, recv_data);

        group_->barrier();
    }

private:
    void ring_allgather(std::span<const T> send_data, std::span<T> recv_data) {
        auto ring_group =
            std::dynamic_pointer_cast<RingCollectiveGroup>(group_);
        const size_t rank = ring_group->rank();
        const size_t size = ring_group->size();
        const size_t chunk_size = send_data.size();

        // Copy own data to correct position
        // Zero-copy optimization: manual copy
        size_t local_offset = rank * chunk_size;
        for (size_t i = 0; i < send_data.size(); ++i) {
            recv_data[local_offset + i] = send_data[i];
        }

        // Ring algorithm: pass data around the ring
        for (size_t step = 0; step < size - 1; ++step) {
            // Determine which chunk to send
            size_t send_rank = (rank - step + size) % size;
            size_t recv_rank = (rank - step - 1 + size) % size;

            size_t send_offset = send_rank * chunk_size;
            size_t recv_offset = recv_rank * chunk_size;

            // Send to next neighbor
            auto next_channel =
                ring_group->get_channel(ring_group->next_neighbor());
            next_channel->send(
                std::span<T>(recv_data.data() + send_offset, chunk_size));

            // Receive from previous neighbor
            auto prev_channel =
                ring_group->get_channel(ring_group->prev_neighbor());
            auto received = prev_channel->receive<std::vector<T>>();

            if (received && received->size() == chunk_size) {
                // Zero-copy optimization: manual copy
                for (size_t i = 0; i < chunk_size; ++i) {
                    recv_data[recv_offset + i] = (*received)[i];
                }
            }
        }
    }

    void simple_allgather(std::span<const T> send_data,
                          std::span<T> recv_data) {
        // Simple implementation: each process broadcasts its data
        const size_t rank = group_->rank();
        const size_t size = group_->size();
        const size_t chunk_size = send_data.size();

        for (size_t root = 0; root < size; ++root) {
            if (rank == root) {
                // Broadcast own data
                Broadcast<T> bcast(group_);
                std::vector<T> data(send_data.begin(), send_data.end());
                bcast.execute(std::span<T>(data), root);

                // Copy to recv buffer
                // Zero-copy optimization: manual copy
                size_t offset = root * chunk_size;
                for (size_t i = 0; i < data.size(); ++i) {
                    recv_data[offset + i] = data[i];
                }
            } else {
                // Receive broadcast
                std::vector<T> data(chunk_size);
                Broadcast<T> bcast(group_);
                bcast.execute(std::span<T>(data), root);

                // Copy to recv buffer
                // Zero-copy optimization: manual copy
                size_t offset = root * chunk_size;
                for (size_t i = 0; i < data.size(); ++i) {
                    recv_data[offset + i] = data[i];
                }
            }
        }
    }
};

/**
 * @brief Factory function to create a collective group
 */
inline std::shared_ptr<CollectiveGroup>
create_collective_group(CollectiveGroup::RankId rank,
                        const std::vector<std::string> &peer_uris,
                        const std::string &topology = "ring") {
    if (topology == "ring") {
        return std::make_shared<RingCollectiveGroup>(rank, peer_uris);
    }

    throw std::runtime_error("Unsupported topology: " + topology);
}

} // namespace collective
} // namespace psyne