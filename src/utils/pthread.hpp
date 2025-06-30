#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>
#ifdef __linux__
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#endif
#include "logger.hpp"

namespace psyne {

static size_t cpu_count = std::thread::hardware_concurrency() > 0
                              ? std::thread::hardware_concurrency()
                              : 1;

/**
 * @brief Enhanced thread pool with work stealing, priority queues, and affinity
 * support
 */
class PsynePool {
public:
    using TaskHandle = uint64_t;
    using Priority = int;

    static constexpr Priority DEFAULT_PRIORITY = 0;
    static constexpr Priority HIGH_PRIORITY = 10;
    static constexpr Priority LOW_PRIORITY = -10;

    struct ThreadAffinity {
        bool use_affinity = false;
        std::vector<int> cpu_ids; // Which CPUs this pool can use
        int numa_node = -1;       // NUMA node preference (-1 = any)
    };

    /**
     * @brief Construct a ThreadPool with advanced features
     */
    explicit PsynePool(size_t min_thread_count = 1,
                       size_t max_thread_count = psyne::cpu_count,
                       int thread_evict_wait_ms = -1,
                       bool enable_work_stealing = false,
                       const ThreadAffinity &affinity = {})
        : min_thread_count_(min_thread_count),
          max_thread_count_(max_thread_count),
          thread_evict_wait_ms_(thread_evict_wait_ms),
          enable_work_stealing_(enable_work_stealing), affinity_(affinity),
          stopSignal_(false), next_task_handle_(1) {
        if (min_thread_count == 0)
            min_thread_count = 1;
        if (max_thread_count == 0)
            max_thread_count = 1;
        if (min_thread_count > max_thread_count)
            throw std::invalid_argument(
                "min_thread_count cannot be greater than max_thread_count");

        // Initialize per-thread queues if work stealing is enabled
        if (enable_work_stealing_) {
            thread_queues_.resize(max_thread_count);
        }

        ensureCoreThreads();
    }

    /**
     * @brief Enqueue a callable with arbitrary args, returning a future to its
     * result
     */
    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        return enqueuePriority(DEFAULT_PRIORITY, std::forward<F>(f),
                               std::forward<Args>(args)...);
    }

    /**
     * @brief Enqueue with priority
     */
    template <class F, class... Args>
    auto enqueuePriority(Priority priority, F &&f, Args &&...args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;

        auto handle = next_task_handle_.fetch_add(1, std::memory_order_relaxed);
        auto taskPtr = std::make_shared<std::packaged_task<R()>>(
            [fn = std::forward<F>(f),
             tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
                return std::apply(std::move(fn), std::move(tup));
            });
        auto fut = taskPtr->get_future();

        if (stopSignal_.load(std::memory_order_relaxed)) {
            log_error("PsynePool::enqueue called after stop() was signaled. It "
                      "did not execute.");
            std::promise<R> promise;
            promise.set_exception(std::make_exception_ptr(
                std::runtime_error("ThreadPool has been stopped")));
            return promise.get_future();
        }

        auto task = [taskPtr, handle, this] {
            // Check if cancelled
            {
                std::lock_guard<std::mutex> lock(cancelMutex_);
                if (cancelled_tasks_.count(handle)) {
                    cancelled_tasks_.erase(handle);
                    return;
                }
            }
            (*taskPtr)();
        };

        if (enable_work_stealing_) {
            // Add to a random thread's queue for better distribution
            size_t thread_idx = thread_dist_(thread_rng_) % max_thread_count_;
            {
                std::lock_guard<std::mutex> lock(
                    thread_queues_[thread_idx].mutex);
                thread_queues_[thread_idx].priority_tasks.emplace(
                    priority, handle, std::move(task));
            }
            thread_queues_[thread_idx].cv.notify_one();
        } else {
            // Use global queue
            std::lock_guard<std::mutex> lock(tasksMutex_);
            priority_tasks_.emplace(priority, handle, std::move(task));
            tasksCondVar_.notify_one();
        }

        maybeSpawnExtra();
        return fut;
    }

    /**
     * @brief Enqueue with handle for cancellation
     */
    template <class F, class... Args>
    std::pair<TaskHandle, std::future<std::invoke_result_t<F, Args...>>>
    enqueueWithHandle(Priority priority, F &&f, Args &&...args) {
        auto future = enqueuePriority(priority, std::forward<F>(f),
                                      std::forward<Args>(args)...);
        auto handle = next_task_handle_.load(std::memory_order_relaxed) - 1;
        return {handle, std::move(future)};
    }

    /**
     * @brief Cancel a task by handle (if it hasn't started yet)
     */
    bool cancel(TaskHandle handle) {
        std::lock_guard<std::mutex> lock(cancelMutex_);
        cancelled_tasks_.insert(handle);
        return true;
    }

    /**
     * @brief Batch enqueue operations for better performance
     */
    template <typename Iterator>
    void enqueueBatch(Iterator begin, Iterator end,
                      Priority priority = DEFAULT_PRIORITY) {
        if (enable_work_stealing_) {
            // Distribute across thread queues
            size_t thread_idx = 0;
            for (auto it = begin; it != end; ++it) {
                auto handle =
                    next_task_handle_.fetch_add(1, std::memory_order_relaxed);
                {
                    std::lock_guard<std::mutex> lock(
                        thread_queues_[thread_idx].mutex);
                    thread_queues_[thread_idx].priority_tasks.emplace(
                        priority, handle, *it);
                }
                thread_idx = (thread_idx + 1) % max_thread_count_;
            }
            // Wake all threads
            for (auto &queue : thread_queues_) {
                queue.cv.notify_one();
            }
        } else {
            // Batch insert into global queue
            std::lock_guard<std::mutex> lock(tasksMutex_);
            for (auto it = begin; it != end; ++it) {
                auto handle =
                    next_task_handle_.fetch_add(1, std::memory_order_relaxed);
                priority_tasks_.emplace(priority, handle, *it);
            }
            tasksCondVar_.notify_all();
        }
        maybeSpawnExtra();
    }

    /**
     * @brief GPU task support - marks task as GPU-bound for special handling
     */
    template <class F, class... Args>
    auto enqueueGPU(F &&f, Args &&...args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        // GPU tasks get high priority and special handling
        return enqueuePriority(
            HIGH_PRIORITY, [fn = std::forward<F>(f),
                            args_tup = std::make_tuple(
                                std::forward<Args>(args)...)]() mutable {
                // Could add GPU context switching here
                return std::apply(std::move(fn), std::move(args_tup));
            });
    }

    ~PsynePool() {
        stop();
    }

    void stop() {
        stopSignal_.store(true, std::memory_order_relaxed);

        if (enable_work_stealing_) {
            for (auto &queue : thread_queues_) {
                queue.cv.notify_all();
            }
        } else {
            tasksCondVar_.notify_all();
        }

        {
            std::lock_guard<std::mutex> lock(tasksMutex_);
            // Clear all tasks
            while (!priority_tasks_.empty())
                priority_tasks_.pop();
            idleCondVar_.notify_all();
        }

        {
            std::lock_guard<std::mutex> lock(workersMutex_);
            max_thread_count_ = 0;
        }

        for (auto &t : workers_) {
            if (t.joinable())
                t.join();
        }
        workers_.clear();
    }

    void panic() {
        stopSignal_.store(true);

        if (enable_work_stealing_) {
            for (auto &queue : thread_queues_) {
                queue.cv.notify_all();
            }
        } else {
            tasksCondVar_.notify_all();
        }

        {
            std::lock_guard lock(workersMutex_);
            for (auto &t : workers_)
                if (t.joinable())
                    t.join();
            workers_.clear();
        }
    }

    void set_min_threads(size_t min_threads) {
        if (min_threads > max_thread_count_) {
            log_warn(
                "Attempted to set min_thread_count_ to ", min_threads,
                " which is greater than max_thread_count_ ",
                max_thread_count_.load(std::memory_order_relaxed),
                ". Setting min_thread_count_ to max_thread_count_ instead.");
            min_threads = max_thread_count_;
        }
        min_thread_count_.store(min_threads, std::memory_order_relaxed);
    }

    void set_max_threads(size_t max_threads) {
        if (max_threads < min_thread_count_.load(std::memory_order_relaxed)) {
            log_warn(
                "Attempted to set max_thread_count_ to ", max_threads,
                " which is less than min_thread_count_ ",
                min_thread_count_.load(std::memory_order_relaxed),
                ". Setting max_thread_count_ to min_thread_count_ instead.");
            max_threads = min_thread_count_.load(std::memory_order_relaxed);
        }
        max_thread_count_.store(max_threads, std::memory_order_relaxed);
        ensureCoreThreads();
    }

    template <class TaskFunc, class ThenFunc, class... Args>
    void enqueueThen(TaskFunc &&mainTask, ThenFunc &&thenFunc, Args &&...args) {
        using Ret = std::invoke_result_t<TaskFunc, Args...>;
        enqueue([=]() {
            if constexpr (std::is_void_v<Ret>) {
                mainTask(args...);
                thenFunc();
            } else {
                Ret r = mainTask(args...);
                thenFunc(r);
            }
        });
    }

    template <class F, class... Args>
    void fireAndForget(F &&f, Args &&...args) {
        enqueue([=]() { f(args...); });
    }

    void drain() {
        std::unique_lock<std::mutex> lock(idleMutex_);
        idleCondVar_.wait(lock, [&] {
            if (enable_work_stealing_) {
                for (const auto &queue : thread_queues_) {
                    std::lock_guard<std::mutex> qlock(queue.mutex);
                    if (!queue.priority_tasks.empty())
                        return false;
                }
            } else {
                std::lock_guard<std::mutex> tlock(tasksMutex_);
                if (!priority_tasks_.empty())
                    return false;
            }
            return activeCount_.load(std::memory_order_relaxed) == 0;
        });
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(workersMutex_);
        return workers_.size();
    }

    size_t pending() const {
        size_t total = 0;
        if (enable_work_stealing_) {
            for (const auto &queue : thread_queues_) {
                std::lock_guard<std::mutex> lock(queue.mutex);
                total += queue.priority_tasks.size();
            }
        } else {
            std::lock_guard<std::mutex> lock(tasksMutex_);
            total = priority_tasks_.size();
        }
        return total;
    }

    static inline void setThreadLocalData(void *ptr) {
        threadLocalData_ = ptr;
    }
    static inline void *getThreadLocalData() {
        return threadLocalData_;
    }

private:
    // Priority task wrapper
    struct PriorityTask {
        Priority priority;
        TaskHandle handle;
        std::function<void()> task;

        PriorityTask(Priority p, TaskHandle h, std::function<void()> t)
            : priority(p), handle(h), task(std::move(t)) {}

        bool operator<(const PriorityTask &other) const {
            return priority < other.priority; // Higher priority = higher value
        }
    };

    // Per-thread queue for work stealing
    struct ThreadQueue {
        mutable std::mutex mutex;
        std::condition_variable cv;
        std::priority_queue<PriorityTask> priority_tasks;
    };

    // Global queue (when work stealing is disabled)
    std::priority_queue<PriorityTask> priority_tasks_;
    mutable std::mutex tasksMutex_;
    std::condition_variable tasksCondVar_;

    // Per-thread queues (when work stealing is enabled)
    std::vector<ThreadQueue> thread_queues_;
    bool enable_work_stealing_;

    std::vector<std::thread> workers_;
    mutable std::mutex workersMutex_;

    std::atomic<size_t> min_thread_count_;
    std::atomic<size_t> max_thread_count_;

    int thread_evict_wait_ms_;
    std::atomic<bool> stopSignal_;

    std::atomic<size_t> activeCount_{0};
    std::condition_variable idleCondVar_;
    std::mutex idleMutex_;

    // Task cancellation
    std::unordered_set<TaskHandle> cancelled_tasks_;
    mutable std::mutex cancelMutex_;
    std::atomic<TaskHandle> next_task_handle_;

    // Thread affinity
    ThreadAffinity affinity_;

    // Work stealing RNG
    thread_local static inline std::mt19937 thread_rng_{std::random_device{}()};
    thread_local static inline std::uniform_int_distribution<size_t>
        thread_dist_;

    static inline thread_local void *threadLocalData_ = nullptr;

    /**
     * @brief Try to steal work from other threads
     */
    bool tryStealWork(size_t thread_idx, PriorityTask &stolen_task) {
        if (!enable_work_stealing_)
            return false;

        size_t attempts = thread_queues_.size() - 1;
        size_t victim_idx = (thread_idx + 1) % thread_queues_.size();

        while (attempts-- > 0) {
            auto &victim_queue = thread_queues_[victim_idx];

            if (std::unique_lock<std::mutex> lock(victim_queue.mutex,
                                                  std::try_to_lock);
                lock.owns_lock()) {
                if (!victim_queue.priority_tasks.empty()) {
                    stolen_task = std::move(const_cast<PriorityTask &>(
                        victim_queue.priority_tasks.top()));
                    victim_queue.priority_tasks.pop();
                    return true;
                }
            }

            victim_idx = (victim_idx + 1) % thread_queues_.size();
        }

        return false;
    }

    /**
     * @brief Worker loop with work stealing support
     */
    void workerLoop(size_t thread_idx) {
        // Set thread affinity if requested
        setThreadAffinity(thread_idx);

        using namespace std::chrono;
        auto timeout = thread_evict_wait_ms_ > 0
                           ? milliseconds(thread_evict_wait_ms_)
                           : milliseconds::max();

        while (!stopSignal_.load(std::memory_order_relaxed)) {
            PriorityTask task(0, 0, nullptr);
            bool found_task = false;

            if (enable_work_stealing_ && thread_idx < thread_queues_.size()) {
                // Try local queue first
                auto &my_queue = thread_queues_[thread_idx];
                {
                    std::unique_lock<std::mutex> lock(my_queue.mutex);
                    my_queue.cv.wait_for(lock, timeout, [&] {
                        return stopSignal_.load(std::memory_order_relaxed) ||
                               !my_queue.priority_tasks.empty();
                    });

                    if (stopSignal_.load(std::memory_order_relaxed))
                        break;

                    if (!my_queue.priority_tasks.empty()) {
                        task = std::move(const_cast<PriorityTask &>(
                            my_queue.priority_tasks.top()));
                        my_queue.priority_tasks.pop();
                        found_task = true;
                    }
                }

                // Try work stealing if no local work
                if (!found_task) {
                    found_task = tryStealWork(thread_idx, task);
                }
            } else {
                // Use global queue
                std::unique_lock<std::mutex> lock(tasksMutex_);
                tasksCondVar_.wait_for(lock, timeout, [&] {
                    return stopSignal_.load(std::memory_order_relaxed) ||
                           !priority_tasks_.empty();
                });

                if (stopSignal_.load(std::memory_order_relaxed))
                    break;

                if (!priority_tasks_.empty()) {
                    task = std::move(
                        const_cast<PriorityTask &>(priority_tasks_.top()));
                    priority_tasks_.pop();
                    found_task = true;
                }
            }

            if (!found_task) {
                // Check if we should evict this thread
                if (size() > min_thread_count_.load(std::memory_order_relaxed))
                    return;
                continue;
            }

            activeCount_.fetch_add(1, std::memory_order_relaxed);
            try {
                task.task();
            } catch (const std::exception &e) {
                log_error("Exception in thread pool worker: ", e.what());
            } catch (...) {
                log_error("Unknown exception in thread pool worker");
            }
            activeCount_.fetch_sub(1, std::memory_order_relaxed);
            idleCondVar_.notify_all();
        }
    }

    /**
     * @brief Set thread affinity based on configuration
     */
    void setThreadAffinity(size_t thread_idx) {
#ifdef __linux__
        if (!affinity_.use_affinity)
            return;

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        if (!affinity_.cpu_ids.empty()) {
            // Use specific CPU IDs
            for (int cpu_id : affinity_.cpu_ids) {
                CPU_SET(cpu_id, &cpuset);
            }
        } else {
            // Default: spread across all CPUs
            int cpu_id = thread_idx % cpu_count;
            CPU_SET(cpu_id, &cpuset);
        }

        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        // Set NUMA node if specified
        if (affinity_.numa_node >= 0 && numa_available() >= 0) {
            numa_run_on_node(affinity_.numa_node);
        }
#endif
    }

    void ensureCoreThreads() {
        std::lock_guard<std::mutex> lock(workersMutex_);
        while (workers_.size() <
               min_thread_count_.load(std::memory_order_relaxed)) {
            size_t thread_idx = workers_.size();
            workers_.emplace_back(
                [this, thread_idx] { workerLoop(thread_idx); });
        }
    }

    void maybeSpawnExtra() {
        std::lock_guard<std::mutex> lock(workersMutex_);
        if (workers_.size() >=
            max_thread_count_.load(std::memory_order_relaxed))
            return;

        size_t pending_tasks = pending();
        if (pending_tasks > workers_.size()) {
            size_t thread_idx = workers_.size();
            workers_.emplace_back(
                [this, thread_idx] { workerLoop(thread_idx); });
        }
    }
};

} // namespace psyne