#include "../../include/psyne/psyne.hpp"
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>

namespace psyne {

// Forward declarations for reliability components
class AcknowledgmentManager {
public:
    AcknowledgmentManager(Channel& channel) : channel_(channel) {}
    void start() { running_ = true; }
    void stop() { running_ = false; }
    bool is_running() const { return running_; }
    
private:
    Channel& channel_;
    std::atomic<bool> running_{false};
};

class RetryManager {
public:
    RetryManager(Channel& channel, const ReliabilityConfig& config) 
        : channel_(channel), config_(config) {}
    void start() { running_ = true; }
    void stop() { running_ = false; }
    bool is_running() const { return running_; }
    
private:
    Channel& channel_;
    ReliabilityConfig config_;
    std::atomic<bool> running_{false};
};

class HeartbeatManager {
public:
    HeartbeatManager(Channel& channel, const ReliabilityConfig& config)
        : channel_(channel), config_(config) {}
    void start() { running_ = true; }
    void stop() { running_ = false; }
    bool is_running() const { return running_; }
    
private:
    Channel& channel_;
    ReliabilityConfig config_;
    std::atomic<bool> running_{false};
};

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity) : capacity_(capacity) {}
    void start() { running_ = true; }
    void stop() { running_ = false; }
    bool is_running() const { return running_; }
    
private:
    size_t capacity_;
    std::atomic<bool> running_{false};
};

// ReliabilityManager::Impl
class ReliabilityManager::Impl {
public:
    Impl(Channel& channel, const ReliabilityConfig& config)
        : channel_(channel)
        , config_(config)
        , ack_manager_(std::make_unique<AcknowledgmentManager>(channel))
        , retry_manager_(std::make_unique<RetryManager>(channel, config))
        , heartbeat_manager_(std::make_unique<HeartbeatManager>(channel, config))
        , replay_buffer_(std::make_unique<ReplayBuffer>(config.replay_buffer_size))
        , running_(false) {}
    
    ~Impl() {
        stop();
    }
    
    void start() {
        if (!running_.exchange(true)) {
            ack_manager_->start();
            retry_manager_->start();
            heartbeat_manager_->start();
            replay_buffer_->start();
        }
    }
    
    void stop() {
        if (running_.exchange(false)) {
            ack_manager_->stop();
            retry_manager_->stop();
            heartbeat_manager_->stop();
            replay_buffer_->stop();
        }
    }
    
    bool is_running() const {
        return running_;
    }
    
    Channel& channel_;
    ReliabilityConfig config_;
    std::unique_ptr<AcknowledgmentManager> ack_manager_;
    std::unique_ptr<RetryManager> retry_manager_;
    std::unique_ptr<HeartbeatManager> heartbeat_manager_;
    std::unique_ptr<ReplayBuffer> replay_buffer_;
    std::atomic<bool> running_;
};

// ReliabilityManager implementation
ReliabilityManager::ReliabilityManager(Channel& channel, const ReliabilityConfig& config)
    : impl_(std::make_unique<Impl>(channel, config)) {
}

ReliabilityManager::~ReliabilityManager() = default;

void ReliabilityManager::start() {
    impl_->start();
}

void ReliabilityManager::stop() {
    impl_->stop();
}

bool ReliabilityManager::is_running() const {
    return impl_->is_running();
}

AcknowledgmentManager* ReliabilityManager::acknowledgment_manager() {
    return impl_->ack_manager_.get();
}

RetryManager* ReliabilityManager::retry_manager() {
    return impl_->retry_manager_.get();
}

HeartbeatManager* ReliabilityManager::heartbeat_manager() {
    return impl_->heartbeat_manager_.get();
}

ReplayBuffer* ReliabilityManager::replay_buffer() {
    return impl_->replay_buffer_.get();
}

} // namespace psyne