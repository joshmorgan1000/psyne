#pragma once

#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include "utils.hpp"
#include <mutex>
#include <deque>
#include <condition_variable>

namespace psyne {

static inline size_t cpu_count = std::thread::hardware_concurrency() > 0
    ? std::thread::hardware_concurrency() : 1;

/**
 * @brief A simple non-templated thread pool with:
 *   - enqueue(...) returning a future
 *   - enqueueAndThen(...) for fire-and-forget plus a callback
 *   - An optional "worker function" you can set if you always want the same
 * logic
 *   - The ability to resize the pool at runtime
 */
class VecPool {
public:

    /**
     * @brief Construct a ThreadPool with `threadCount` threads.
     * 
     * @param threadCount The number of threads to create. If 0, uses the
     *                    maximum available concurrency.
     */
    explicit VecPool(
        size_t min_thread_count = 1,
        size_t max_thread_count = cpu_count,
        uint64_t thread_evict_wait_ms = 3000
    ) : min_thread_count_(min_thread_count)
      , max_thread_count_(max_thread_count)
      , thread_evict_wait_ms_(thread_evict_wait_ms)
      , stopSignal_(false)
    {
        if (min_thread_count == 0) min_thread_count = 1;
        if (max_thread_count == 0) max_thread_count = 1;
        if (min_thread_count > max_thread_count)
            throw std::invalid_argument("min_thread_count cannot be greater than max_thread_count");
        ensureCoreThreads();
    }

    /**
     * @brief Enqueue a callable with arbitrary args, returning a future to its
     * result.
     */
    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<typename std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;
        auto taskPtr = std::make_shared<std::packaged_task<R()>>(
            [fn = std::forward<F>(f), tup = std::make_tuple(std::forward<Args>(args)...) ]() mutable {
                return std::apply(std::move(fn), std::move(tup));
            }
        );
        auto fut = taskPtr->get_future();
        if (stopSignal_.load(std::memory_order_acquire)) {
            log_error("VecPool::enqueue called after stop() was signaled. It did not execute.");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(tasksMutex_);
            tasks_.emplace_back([taskPtr]{ (*taskPtr)(); });
        }
        tasksCondVar_.notify_one();
        maybeSpawnExtra();
        return fut;
    }

    /**
     * @brief Destroy the pool. Automatically calls stop() if not already stopped.
     *
     */

    ~VecPool() { stop(); }

    /**
     * @brief Stop the pool. Joins all threads and disallows further enqueues. //
     * inline After calling stop(), you cannot use this pool again.
     */
    void stop() {
        stopSignal_.store(true, std::memory_order_release);
        tasksCondVar_.notify_all();
        {
            std::lock_guard<std::mutex> lock(tasksMutex_);
            tasks_.clear();
            idleCondVar_.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(workersMutex_);
            max_thread_count_ = 0;
        }
        for (auto &t : workers_) { if (t.joinable()) t.join(); }
        workers_.clear();
    }

    /**
     * @brief Stops the pool immediately, without waiting for tasks to complete.
     */
    void panic() {
        stopSignal_.store(true);
        tasksCondVar_.notify_all();
        {
            std::lock_guard lock(workersMutex_);
            for (auto &t : workers_) if (t.joinable()) t.join();
            workers_.clear();               
        }
    }

    /**
     * @brief Set the maximum number of threads in the pool.
     *        If the new max is less than the current size, it will not shrink
     *        the pool immediately, but will prevent new threads from being created.
     */
    void set_min_threads(size_t min_threads) {
        if (min_threads > max_thread_count_) {
            log_warn("Attempted to set min_thread_count_ to ", min_threads,
                     " which is greater than max_thread_count_ ",
                     max_thread_count_.load(std::memory_order_acquire),
                     ". Setting min_thread_count_ to max_thread_count_ instead.");
            min_threads = max_thread_count_;
        }
        min_thread_count_.store(min_threads, std::memory_order_release);
    }

    /**
     * @brief Set the maximum number of threads in the pool.
     *        If the new max is less than the current size, it will not shrink
     *        the pool immediately, but will prevent new threads from being created.
     */
    void set_max_threads(size_t max_threads) {
        if (max_threads < min_thread_count_.load(std::memory_order_acquire)) {
            log_warn("Attempted to set max_thread_count_ to ", max_threads,
                     " which is less than min_thread_count_ ",
                     min_thread_count_.load(std::memory_order_acquire),
                     ". Setting max_thread_count_ to min_thread_count_ instead.");
            max_threads = min_thread_count_.load(std::memory_order_acquire);
        }
        max_thread_count_.store(max_threads, std::memory_order_release);
        ensureCoreThreads();
    }

    /**
     * @brief enqueueAndThen: "fire-and-forget" for the main task, but once it's
     * done, run a callback function on the same worker thread. //
     * inline
     *
     * If the task returns a value, the callback receives that value. If the task
     * is void, the callback receives no parameters.
     */
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

    /**
     * @brief Fire-and-forget: enqueue a task that runs and returns nothing. //
     * inline This is a non-blocking call, the task will run in the background. //
     * inline
     *
     * Example usage:
     *    fireAndForget([]{ std::cout << "Hello from thread!" << std::endl; }); //
     * inline
     */
    template <class F, class... Args>
    void fireAndForget(F &&f, Args &&...args) {
        enqueue([=]() { f(args...); });
    }

    /**
     * @brief Wait for all tasks to complete and all workers to be idle.
     *        This is a blocking call that waits until all tasks are done and
     *        no worker is active.
     */
    void drain() {
        std::unique_lock<std::mutex> lock(idleMutex_);
        idleCondVar_.wait(lock, [&] {
            std::lock_guard<std::mutex> tlock(tasksMutex_);
            return tasks_.empty() && activeCount_.load(std::memory_order_acquire) == 0;
        });
    }

    /**
     * @brief Return current worker count
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(workersMutex_);
        return workers_.size();
    }

    /**
     * @brief Return the number of tasks in the queue
     */
    size_t pending() const {
        std::lock_guard<std::mutex> lock(tasksMutex_);
        return tasks_.size();
    }

    /**
     * @brief Set thread-local data for the current thread.
     *        This is a static function that sets a thread_local variable. //
     * inline You can use this to store per-thread data.
     */
    static inline void setThreadLocalData(void *ptr) { threadLocalData_ = ptr; }

    /**
     * @brief Get thread-local data for the current thread.
     *        This is a static function that gets a thread_local variable. //
     * inline You can use this to retrieve per-thread data.
     */
    static inline void *getThreadLocalData() { return threadLocalData_; }

    /**
     * @brief Forcefully stop the thread pool and all workers.
     *        This is a non-graceful shutdown that will stop all threads immediately.
     */
    void panic() {
        stopSignal_.store(true, std::memory_order_release);
        tasksCondVar_.notify_all();
        {
            std::lock_guard<std::mutex> lock(tasksMutex_);
            tasks_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(workersMutex_);
            for (auto &t : workers_) {
                if (t.joinable()) t.detach();
            }
            workers_.clear();
        }
    }

private:
    std::deque<std::function<void()>> tasks_;
    mutable std::mutex tasksMutex_;
    std::condition_variable tasksCondVar_;

    std::vector<std::thread> workers_;
    mutable std::mutex workersMutex_;

    std::atomic<size_t> min_thread_count_;
    std::atomic<size_t> max_thread_count_;

    uint64_t thread_evict_wait_ms_;
    std::atomic<bool> stopSignal_;

    std::atomic<size_t> activeCount_{0};
    std::condition_variable idleCondVar_;
    std::mutex idleMutex_;

    static inline thread_local void *threadLocalData_ = nullptr;

    /**
     * @brief The worker loop for each thread.
     *        Repeatedly pop tasks and run them until stop.
     */
    void workerLoop() {
        using namespace std::chrono;
        auto timeout = milliseconds(thread_evict_wait_ms_);
        while (!stopSignal_.load(std::memory_order_acquire)) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(tasksMutex_);
                tasksCondVar_.wait_for(lock, timeout, [&] {
                    return stopSignal_.load(std::memory_order_acquire) || !tasks_.empty();
                });
                if (stopSignal_.load(std::memory_order_acquire)) break;
                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                } else {
                    if (size() > min_thread_count_.load(std::memory_order_acquire)) return;
                    continue;
                }
            }
            activeCount_.fetch_add(1, std::memory_order_acquire);
            try {
                task();
            } catch (const std::exception &e) {
                log_error("Exception in thread pool worker: ", e.what());
            } catch (...) {
                log_error("Unknown exception in thread pool worker");
            }
            activeCount_.fetch_sub(1, std::memory_order_release);
            idleCondVar_.notify_all();
        }
    }

    /**
     * @brief Ensure the pool has at least `min_thread_count_` threads running.
     *        If not, create new threads until the count is met.
     */
    void ensureCoreThreads() {
        std::lock_guard<std::mutex> lock(workersMutex_);
        while (workers_.size() < min_thread_count_.load(std::memory_order_acquire)) {
            workers_.emplace_back([this]{ workerLoop(); });
        }
    }

    /**
     * @brief Maybe spawn an extra thread if the current count is below `max_thread_count_`
     *        and there are tasks waiting.
     */
    void maybeSpawnExtra() {
        std::lock_guard<std::mutex> lock(workersMutex_);
        if (workers_.size() >= max_thread_count_.load(std::memory_order_acquire)) return;
        if (tasks_.size() > workers_.size()) {
            workers_.emplace_back([this]{ workerLoop(); });
        }
    }
};

} // namespace manifoldb