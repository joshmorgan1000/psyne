#include "channel_multiplexer.hpp"
#include <cstring>
#include <iostream>

namespace psyne {
namespace multiplexing {

// Multiplexed message format
struct ChannelMultiplexer::MultiplexedMessage {
    uint8_t channel_id;
    uint32_t type;
    std::vector<uint8_t> data;
};

// Logical channel implementation
class ChannelMultiplexer::LogicalChannel : public Channel {
public:
    LogicalChannel(uint8_t id, ChannelMultiplexer *mux)
        : channel_id_(id), multiplexer_(mux) {}

    void stop() override {
        stopped_ = true;
    }

    bool is_stopped() const override {
        return stopped_ || !multiplexer_;
    }

    const std::string &uri() const override {
        static std::string uri = "mux://logical_" + std::to_string(channel_id_);
        return uri;
    }

    ChannelType type() const override {
        return ChannelType::MultiType;
    }

    ChannelMode mode() const override {
        return ChannelMode::MPMC; // Multiplexed channels are always MPMC
    }

    void *receive_raw_message(size_t &size, uint32_t &type) override {
        // Would need to implement receive queue per logical channel
        return nullptr;
    }

    void release_raw_message(void *handle) override {
        // Release message memory
    }

    bool has_metrics() const override {
        return false;
    }
    debug::ChannelMetrics get_metrics() const override {
        return {};
    }
    void reset_metrics() override {}

    // Send through multiplexer
    void send_message(const void *data, size_t size, uint32_t type) {
        if (multiplexer_) {
            MultiplexedMessage msg;
            msg.channel_id = channel_id_;
            msg.type = type;
            msg.data.assign(static_cast<const uint8_t *>(data),
                            static_cast<const uint8_t *>(data) + size);

            {
                std::lock_guard<std::mutex> lock(multiplexer_->send_mutex_);
                multiplexer_->send_queue_.push(std::move(msg));
            }
            multiplexer_->send_cv_.notify_one();
        }
    }

protected:
    detail::ChannelImpl *impl() override {
        return nullptr;
    }
    const detail::ChannelImpl *impl() const override {
        return nullptr;
    }

private:
    uint8_t channel_id_;
    ChannelMultiplexer *multiplexer_;
    std::atomic<bool> stopped_{false};
};

// ChannelMultiplexer implementation
ChannelMultiplexer::ChannelMultiplexer(
    std::shared_ptr<Channel> physical_channel, size_t max_logical_channels)
    : physical_channel_(physical_channel),
      max_logical_channels_(max_logical_channels) {}

ChannelMultiplexer::~ChannelMultiplexer() {
    Stop();
}

std::shared_ptr<Channel>
ChannelMultiplexer::CreateLogicalChannel(uint8_t channel_id) {
    std::lock_guard<std::mutex> lock(channels_mutex_);

    if (logical_channels_.find(channel_id) != logical_channels_.end()) {
        throw std::runtime_error("Logical channel already exists: " +
                                 std::to_string(channel_id));
    }

    auto logical = std::make_shared<LogicalChannel>(channel_id, this);
    logical_channels_[channel_id] = logical;
    return logical;
}

void ChannelMultiplexer::Start() {
    if (running_.exchange(true)) {
        return; // Already running
    }

    send_thread_ = std::thread(&ChannelMultiplexer::SendLoop, this);
    receive_thread_ = std::thread(&ChannelMultiplexer::ReceiveLoop, this);
}

void ChannelMultiplexer::Stop() {
    running_ = false;
    send_cv_.notify_all();

    if (send_thread_.joinable()) {
        send_thread_.join();
    }
    if (receive_thread_.joinable()) {
        receive_thread_.join();
    }
}

void ChannelMultiplexer::SendLoop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(send_mutex_);
        send_cv_.wait(lock,
                      [this] { return !send_queue_.empty() || !running_; });

        if (!running_)
            break;

        while (!send_queue_.empty()) {
            auto msg = std::move(send_queue_.front());
            send_queue_.pop();
            lock.unlock();

            // Create multiplexed message format:
            // [1 byte channel_id][4 bytes type][4 bytes size][data...]
            ByteVector mux_msg(*physical_channel_);
            size_t total_size = 1 + 4 + 4 + msg.data.size();
            mux_msg.resize(total_size);

            uint8_t *ptr = mux_msg.data();
            *ptr++ = msg.channel_id;
            *reinterpret_cast<uint32_t *>(ptr) = msg.type;
            ptr += 4;
            *reinterpret_cast<uint32_t *>(ptr) =
                static_cast<uint32_t>(msg.data.size());
            ptr += 4;
            std::memcpy(ptr, msg.data.data(), msg.data.size());

            physical_channel_->send(mux_msg);

            // Update stats
            {
                std::lock_guard<std::mutex> stats_lock(stats_mutex_);
                stats_.messages_sent++;
                stats_.bytes_sent += msg.data.size();
                stats_.channel_message_counts[msg.channel_id]++;
            }

            lock.lock();
        }
    }
}

void ChannelMultiplexer::ReceiveLoop() {
    while (running_) {
        auto msg = physical_channel_->receive<ByteVector>(
            std::chrono::milliseconds(100));

        if (!msg || msg->size() < 9)
            continue; // Minimum header size

        // Parse multiplexed message
        const uint8_t *ptr = msg->data();
        uint8_t channel_id = *ptr++;
        uint32_t type = *reinterpret_cast<const uint32_t *>(ptr);
        ptr += 4;
        uint32_t size = *reinterpret_cast<const uint32_t *>(ptr);
        ptr += 4;

        if (msg->size() < 9 + size)
            continue; // Invalid message

        // Update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.messages_received++;
            stats_.bytes_received += size;
        }

        // Route to logical channel
        // (Would need to implement receive queues per logical channel)
    }
}

ChannelMultiplexer::Stats ChannelMultiplexer::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// ChannelDemultiplexer implementation
class ChannelDemultiplexer::LogicalReceiveChannel : public Channel {
public:
    LogicalReceiveChannel(uint8_t id) : channel_id_(id) {}

    void stop() override {
        stopped_ = true;
    }
    bool is_stopped() const override {
        return stopped_;
    }

    const std::string &uri() const override {
        static std::string uri =
            "demux://logical_" + std::to_string(channel_id_);
        return uri;
    }

    ChannelType type() const override {
        return ChannelType::MultiType;
    }
    ChannelMode mode() const override {
        return ChannelMode::MPMC;
    }

    void *receive_raw_message(size_t &size, uint32_t &type) override {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (receive_queue_.empty()) {
            return nullptr;
        }

        auto &msg = receive_queue_.front();
        size = msg.data.size();
        type = msg.type;

        // Allocate and copy data
        void *data = new uint8_t[size];
        std::memcpy(data, msg.data.data(), size);

        receive_queue_.pop();
        return data;
    }

    void release_raw_message(void *handle) override {
        delete[] static_cast<uint8_t *>(handle);
    }

    bool has_metrics() const override {
        return false;
    }
    debug::ChannelMetrics get_metrics() const override {
        return {};
    }
    void reset_metrics() override {}

    void deliver_message(const void *data, size_t size, uint32_t type) {
        Message msg;
        msg.type = type;
        msg.data.assign(static_cast<const uint8_t *>(data),
                        static_cast<const uint8_t *>(data) + size);

        std::lock_guard<std::mutex> lock(queue_mutex_);
        receive_queue_.push(std::move(msg));
    }

protected:
    detail::ChannelImpl *impl() override {
        return nullptr;
    }
    const detail::ChannelImpl *impl() const override {
        return nullptr;
    }

private:
    struct Message {
        uint32_t type;
        std::vector<uint8_t> data;
    };

    uint8_t channel_id_;
    std::atomic<bool> stopped_{false};
    std::queue<Message> receive_queue_;
    mutable std::mutex queue_mutex_;
};

ChannelDemultiplexer::ChannelDemultiplexer(
    std::shared_ptr<Channel> physical_channel)
    : physical_channel_(physical_channel) {}

ChannelDemultiplexer::~ChannelDemultiplexer() {
    Stop();
}

void ChannelDemultiplexer::RegisterHandler(uint8_t channel_id,
                                           MessageHandler handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    handlers_[channel_id] = handler;
}

std::shared_ptr<Channel>
ChannelDemultiplexer::GetLogicalChannel(uint8_t channel_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = logical_channels_.find(channel_id);
    if (it != logical_channels_.end()) {
        return it->second;
    }

    auto channel = std::make_shared<LogicalReceiveChannel>(channel_id);
    logical_channels_[channel_id] = channel;
    return channel;
}

void ChannelDemultiplexer::Start() {
    if (running_.exchange(true)) {
        return;
    }

    receive_thread_ = std::thread(&ChannelDemultiplexer::ReceiveLoop, this);
}

void ChannelDemultiplexer::Stop() {
    running_ = false;
    if (receive_thread_.joinable()) {
        receive_thread_.join();
    }
}

void ChannelDemultiplexer::ReceiveLoop() {
    while (running_) {
        auto msg = physical_channel_->receive<ByteVector>(
            std::chrono::milliseconds(100));

        if (!msg || msg->size() < 9)
            continue;

        // Parse multiplexed message
        const uint8_t *ptr = msg->data();
        uint8_t channel_id = *ptr++;
        uint32_t type = *reinterpret_cast<const uint32_t *>(ptr);
        ptr += 4;
        uint32_t size = *reinterpret_cast<const uint32_t *>(ptr);
        ptr += 4;

        if (msg->size() < 9 + size)
            continue;

        // Call handler if registered
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto handler_it = handlers_.find(channel_id);
            if (handler_it != handlers_.end()) {
                handler_it->second(ptr, size, type);
            }

            // Deliver to logical channel if exists
            auto channel_it = logical_channels_.find(channel_id);
            if (channel_it != logical_channels_.end()) {
                channel_it->second->deliver_message(ptr, size, type);
            }
        }
    }
}

// ChannelPool implementation
ChannelPool::ChannelPool(const Config &config) : config_(config) {
    // Create initial channels
    for (size_t i = 0; i < config_.initial_channels; ++i) {
        auto channel =
            create_channel(config_.channel_uri_prefix + std::to_string(i),
                           config_.channel_buffer_size);
        channels_.push_back(channel);
        available_.push(channel);
    }
    stats_.total_channels = config_.initial_channels;
    stats_.available_channels = config_.initial_channels;
}

ChannelPool::~ChannelPool() {
    // Stop all multiplexers
    for (auto &mux : multiplexers_) {
        mux->Stop();
    }
}

std::shared_ptr<Channel> ChannelPool::Acquire(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (timeout_ms > 0) {
        cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                     [this] { return !available_.empty(); });
    }

    if (available_.empty()) {
        // Try to create new channel if under limit
        if (channels_.size() < config_.max_channels) {
            auto channel = create_channel(config_.channel_uri_prefix +
                                              std::to_string(channels_.size()),
                                          config_.channel_buffer_size);
            channels_.push_back(channel);
            stats_.total_channels++;
            stats_.acquisitions++;
            return channel;
        }

        // Start multiplexing if threshold reached
        if (channels_.size() >= config_.multiplex_threshold &&
            multiplexers_.empty()) {
            // Create multiplexer on first channel
            auto mux = std::make_shared<ChannelMultiplexer>(channels_[0], 256);
            mux->Start();
            multiplexers_.push_back(mux);

            // Create logical channels
            for (uint8_t i = 0; i < 10; ++i) {
                auto logical = mux->CreateLogicalChannel(i);
                available_.push(logical);
                stats_.multiplexed_channels++;
            }
        }

        if (available_.empty()) {
            return nullptr;
        }
    }

    auto channel = available_.front();
    available_.pop();
    stats_.available_channels--;
    stats_.acquisitions++;

    return channel;
}

void ChannelPool::Release(std::shared_ptr<Channel> channel) {
    if (!channel)
        return;

    std::lock_guard<std::mutex> lock(mutex_);
    available_.push(channel);
    stats_.available_channels++;
    stats_.releases++;
    cv_.notify_one();
}

ChannelPool::PoolStats ChannelPool::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

// BondedChannel implementation
BondedChannel::BondedChannel(std::vector<std::shared_ptr<Channel>> channels,
                             BondingMode mode)
    : channels_(std::move(channels)), mode_(mode),
      start_time_(std::chrono::steady_clock::now()) {
    stats_.channel_sent.resize(channels_.size(), 0);
    stats_.channel_received.resize(channels_.size(), 0);
}

BondedChannel::~BondedChannel() = default;

bool BondedChannel::Send(const void *data, size_t size, uint32_t type) {
    size_t channel_idx = 0;

    switch (mode_) {
    case BondingMode::RoundRobin:
        channel_idx = next_channel_++ % channels_.size();
        break;

    case BondingMode::LeastLoaded:
        // Find channel with least bytes sent
        channel_idx =
            std::distance(stats_.channel_sent.begin(),
                          std::min_element(stats_.channel_sent.begin(),
                                           stats_.channel_sent.end()));
        break;

    case BondingMode::Broadcast:
        // Send to all channels
        for (size_t i = 0; i < channels_.size(); ++i) {
            ByteVector msg(*channels_[i]);
            msg.resize(size);
            std::memcpy(msg.data(), data, size);
            channels_[i]->send(msg);

            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.channel_sent[i] += size;
            stats_.total_sent += size;
        }
        return true;

    case BondingMode::Striped:
        // Stripe large messages across channels
        if (size > 64 * 1024 && channels_.size() > 1) {
            size_t stripe_size = size / channels_.size();
            const uint8_t *ptr = static_cast<const uint8_t *>(data);

            for (size_t i = 0; i < channels_.size(); ++i) {
                size_t this_stripe = (i == channels_.size() - 1)
                                         ? size - (stripe_size * i)
                                         : stripe_size;

                ByteVector msg(*channels_[i]);
                msg.resize(this_stripe + 8); // Add header

                // Header: [stripe_id][total_stripes]
                *reinterpret_cast<uint32_t *>(msg.data()) = i;
                *reinterpret_cast<uint32_t *>(msg.data() + 4) =
                    channels_.size();
                std::memcpy(msg.data() + 8, ptr, this_stripe);

                channels_[i]->send(msg);
                ptr += this_stripe;

                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.channel_sent[i] += this_stripe;
            }

            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.total_sent += size;
            return true;
        }
        // Fall through to round-robin for small messages
        channel_idx = next_channel_++ % channels_.size();
        break;
    }

    // Send to selected channel
    ByteVector msg(*channels_[channel_idx]);
    msg.resize(size);
    std::memcpy(msg.data(), data, size);
    channels_[channel_idx]->send(msg);

    // Update stats
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.channel_sent[channel_idx] += size;
        stats_.total_sent += size;
    }

    return true;
}

std::unique_ptr<std::vector<uint8_t>> BondedChannel::Receive(uint32_t &type,
                                                             int timeout_ms) {
    // Simple implementation - check each channel
    for (size_t i = 0; i < channels_.size(); ++i) {
        auto msg = channels_[i]->receive<ByteVector>(
            std::chrono::milliseconds(timeout_ms / channels_.size()));

        if (msg) {
            auto result = std::make_unique<std::vector<uint8_t>>(msg->begin(),
                                                                 msg->end());

            // Update stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.channel_received[i] += msg->size();
                stats_.total_received += msg->size();
            }

            type = 0; // Would need proper type handling
            return result;
        }
    }

    return nullptr;
}

BondedChannel::BondStats BondedChannel::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Calculate bandwidth
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(now - start_time_).count();
    if (duration > 0) {
        stats_.aggregate_bandwidth_mbps =
            (stats_.total_sent + stats_.total_received) /
            (duration * 1024 * 1024);
    }

    return stats_;
}

} // namespace multiplexing
} // namespace psyne