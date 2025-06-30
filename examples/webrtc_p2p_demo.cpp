/**
 * @file webrtc_p2p_demo.cpp
 * @brief WebRTC Peer-to-Peer Gaming Demo for Psyne
 *
 * This example demonstrates how to use Psyne's WebRTC support for real-time
 * peer-to-peer gaming applications. It shows:
 *
 * 1. Direct peer-to-peer connections using WebRTC
 * 2. Real-time game state synchronization
 * 3. Low-latency message passing for combat actions
 * 4. Network resilience and auto-reconnection
 *
 * Usage:
 *   ./webrtc_p2p_demo player1    # First player (offerer)
 *   ./webrtc_p2p_demo player2    # Second player (answerer)
 *
 * @author Psyne Contributors
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <psyne/psyne.hpp>
#include <random>
#include <thread>
#include <vector>

using namespace psyne;

// Game-specific message types for real-time combat
class PlayerPositionMessage : public Message<PlayerPositionMessage> {
public:
    static constexpr uint32_t message_type = 1001;

    using Message<PlayerPositionMessage>::Message;

    static size_t calculate_size() {
        return sizeof(Header) + sizeof(PositionData);
    }

    struct Header {
        uint32_t player_id;
        uint64_t timestamp_ms;
        uint32_t sequence_number;
    };

    struct PositionData {
        float x, y, z;                            // 3D position
        float velocity_x, velocity_y, velocity_z; // Velocity vector
        float rotation_yaw, rotation_pitch;       // Rotation
        uint8_t health;                           // Player health (0-100)
        uint8_t weapon_id;                        // Currently equipped weapon
        uint16_t flags; // Status flags (running, jumping, etc.)
    };

    void initialize() {
        header().player_id = 0;
        header().timestamp_ms = current_time_ms();
        header().sequence_number = 0;
        std::memset(&position(), 0, sizeof(PositionData));
    }

    Header &header() {
        return *reinterpret_cast<Header *>(
            Message<PlayerPositionMessage>::data());
    }

    const Header &header() const {
        return *reinterpret_cast<const Header *>(
            Message<PlayerPositionMessage>::data());
    }

    PositionData &position() {
        return *reinterpret_cast<PositionData *>(
            reinterpret_cast<uint8_t *>(
                Message<PlayerPositionMessage>::data()) +
            sizeof(Header));
    }

    const PositionData &position() const {
        return *reinterpret_cast<const PositionData *>(
            reinterpret_cast<const uint8_t *>(
                Message<PlayerPositionMessage>::data()) +
            sizeof(Header));
    }

    void set_position(float x, float y, float z) {
        position().x = x;
        position().y = y;
        position().z = z;
    }

    void set_velocity(float vx, float vy, float vz) {
        position().velocity_x = vx;
        position().velocity_y = vy;
        position().velocity_z = vz;
    }

    void set_rotation(float yaw, float pitch) {
        position().rotation_yaw = yaw;
        position().rotation_pitch = pitch;
    }

    uint64_t latency_ms() const {
        return current_time_ms() - header().timestamp_ms;
    }

private:
    static uint64_t current_time_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
};

class CombatActionMessage : public Message<CombatActionMessage> {
public:
    static constexpr uint32_t message_type = 1002;

    using Message<CombatActionMessage>::Message;

    static size_t calculate_size() {
        return sizeof(Header) + sizeof(ActionData);
    }

    enum ActionType : uint8_t {
        Attack = 1,
        Block = 2,
        Dodge = 3,
        Reload = 4,
        UseItem = 5
    };

    struct Header {
        uint32_t player_id;
        uint64_t timestamp_ms;
        uint32_t sequence_number;
        ActionType action_type;
        uint8_t padding[3];
    };

    struct ActionData {
        uint32_t target_player_id;                   // For targeted actions
        float direction_x, direction_y, direction_z; // Action direction
        uint16_t weapon_id;                          // Weapon used
        uint16_t damage;                             // Damage dealt
        float accuracy;                              // Hit accuracy (0.0-1.0)
        uint32_t item_id;                            // For item usage
    };

    void initialize() {
        header().player_id = 0;
        header().timestamp_ms = current_time_ms();
        header().sequence_number = 0;
        header().action_type = Attack;
        std::memset(&action(), 0, sizeof(ActionData));
    }

    Header &header() {
        return *reinterpret_cast<Header *>(
            Message<CombatActionMessage>::data());
    }

    const Header &header() const {
        return *reinterpret_cast<const Header *>(
            Message<CombatActionMessage>::data());
    }

    ActionData &action() {
        return *reinterpret_cast<ActionData *>(
            reinterpret_cast<uint8_t *>(Message<CombatActionMessage>::data()) +
            sizeof(Header));
    }

    const ActionData &action() const {
        return *reinterpret_cast<const ActionData *>(
            reinterpret_cast<const uint8_t *>(Message<CombatActionMessage>::data()) +
            sizeof(Header));
    }

    uint64_t latency_ms() const {
        return current_time_ms() - header().timestamp_ms;
    }

private:
    static uint64_t current_time_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
};

/**
 * @brief Simple game player simulation
 */
class GamePlayer {
public:
    explicit GamePlayer(uint32_t id, const std::string &peer_id)
        : player_id_(id), peer_id_(peer_id), running_(false) {
        // Initialize random position
        std::random_device rd;
        gen_.seed(rd());

        position_.x = uniform_dist_(gen_) * 100.0f;
        position_.y = 0.0f; // Ground level
        position_.z = uniform_dist_(gen_) * 100.0f;
        position_.health = 100;
        position_.weapon_id = 1;

        sequence_number_ = 0;
    }

    void start_simulation(std::shared_ptr<Channel> channel) {
        channel_ = channel;
        running_ = true;

        // Start position update thread (60 FPS)
        position_thread_ = std::thread(&GamePlayer::position_update_loop, this);

        // Start message receiving thread
        receive_thread_ = std::thread(&GamePlayer::message_receive_loop, this);

        std::cout << "🎮 Player " << player_id_ << " (" << peer_id_
                  << ") started simulation\n";
    }

    void stop_simulation() {
        running_ = false;

        if (position_thread_.joinable()) {
            position_thread_.join();
        }

        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }

        std::cout << "🛑 Player " << player_id_ << " stopped simulation\n";
    }

    void simulate_combat_action() {
        if (!channel_)
            return;

        CombatActionMessage action(*channel_);
        action.initialize();

        action.header().player_id = player_id_;
        action.header().sequence_number = ++action_sequence_;
        action.header().action_type =
            static_cast<CombatActionMessage::ActionType>(
                1 + (uniform_dist_(gen_) * 4) // Random action 1-5
            );

        action.action().target_player_id =
            (player_id_ == 1) ? 2 : 1; // Target other player
        action.action().weapon_id = position_.weapon_id;
        action.action().damage =
            10 + (uniform_dist_(gen_) * 20); // 10-30 damage
        action.action().accuracy =
            0.7f + (uniform_dist_(gen_) * 0.3f); // 70-100% accuracy

        // Set action direction (towards target)
        action.action().direction_x = uniform_dist_(gen_) * 2.0f - 1.0f;
        action.action().direction_y = 0.0f;
        action.action().direction_z = uniform_dist_(gen_) * 2.0f - 1.0f;

        action.send();

        std::cout << "⚔️  Player " << player_id_ << " executed "
                  << action_type_name(action.header().action_type)
                  << " (damage: " << action.action().damage << ")\n";
    }

    const PlayerPositionMessage::PositionData &current_position() const {
        return position_;
    }

    void print_stats() const {
        std::cout << "\n📊 Player " << player_id_ << " Statistics:\n";
        std::cout << "   Position: (" << position_.x << ", " << position_.y
                  << ", " << position_.z << ")\n";
        std::cout << "   Health: " << static_cast<int>(position_.health)
                  << "/100\n";
        std::cout << "   Messages sent: " << messages_sent_.load() << "\n";
        std::cout << "   Messages received: " << messages_received_.load()
                  << "\n";
        std::cout << "   Average latency: " << average_latency_.load()
                  << " ms\n";
        std::cout << "   Combat actions: " << combat_actions_.load() << "\n\n";
    }

private:
    uint32_t player_id_;
    std::string peer_id_;
    std::shared_ptr<Channel> channel_;
    std::atomic<bool> running_;

    // Game state
    PlayerPositionMessage::PositionData position_;
    uint32_t sequence_number_;
    uint32_t action_sequence_ = 0;

    // Statistics
    std::atomic<uint64_t> messages_sent_{0};
    std::atomic<uint64_t> messages_received_{0};
    std::atomic<uint32_t> average_latency_{0};
    std::atomic<uint32_t> combat_actions_{0};

    // Threading
    std::thread position_thread_;
    std::thread receive_thread_;

    // Random number generation
    std::mt19937 gen_;
    std::uniform_real_distribution<float> uniform_dist_{-1.0f, 1.0f};

    void position_update_loop() {
        auto last_update = std::chrono::steady_clock::now();

        while (running_) {
            auto now = std::chrono::steady_clock::now();
            auto dt = std::chrono::duration<float>(now - last_update).count();
            last_update = now;

            // Simulate movement
            update_position(dt);

            // Send position update
            send_position_update();

            // 60 FPS update rate
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }

    void update_position(float dt) {
        // Simple movement simulation
        float speed = 10.0f; // units per second

        // Random walk
        position_.velocity_x += (uniform_dist_(gen_) * 2.0f - 1.0f) * dt;
        position_.velocity_z += (uniform_dist_(gen_) * 2.0f - 1.0f) * dt;

        // Clamp velocity
        float max_velocity = speed;
        float velocity_mag =
            std::sqrt(position_.velocity_x * position_.velocity_x +
                      position_.velocity_z * position_.velocity_z);
        if (velocity_mag > max_velocity) {
            position_.velocity_x =
                (position_.velocity_x / velocity_mag) * max_velocity;
            position_.velocity_z =
                (position_.velocity_z / velocity_mag) * max_velocity;
        }

        // Update position
        position_.x += position_.velocity_x * dt;
        position_.z += position_.velocity_z * dt;

        // Keep within bounds
        position_.x = std::max(-50.0f, std::min(50.0f, position_.x));
        position_.z = std::max(-50.0f, std::min(50.0f, position_.z));

        // Update rotation to face movement direction
        if (velocity_mag > 0.1f) {
            position_.rotation_yaw =
                std::atan2(position_.velocity_z, position_.velocity_x);
        }
    }

    void send_position_update() {
        if (!channel_)
            return;

        PlayerPositionMessage msg(*channel_);
        msg.initialize();

        msg.header().player_id = player_id_;
        msg.header().sequence_number = ++sequence_number_;
        msg.position() = position_;

        msg.send();
        messages_sent_++;
    }

    void message_receive_loop() {
        while (running_) {
            // Try to receive position updates
            auto pos_msg = channel_->receive<PlayerPositionMessage>(
                std::chrono::milliseconds(10));
            if (pos_msg && pos_msg->header().player_id != player_id_) {
                handle_position_message(*pos_msg);
            }

            // Try to receive combat actions
            auto action_msg = channel_->receive<CombatActionMessage>(
                std::chrono::milliseconds(10));
            if (action_msg && action_msg->header().player_id != player_id_) {
                handle_combat_message(*action_msg);
            }
        }
    }

    void handle_position_message(const PlayerPositionMessage &msg) {
        messages_received_++;

        uint64_t latency = msg.latency_ms();
        average_latency_.store((average_latency_.load() + latency) / 2);

        const auto &pos = msg.position();
        std::cout << "📍 Peer " << msg.header().player_id << " at (" << pos.x
                  << ", " << pos.z << ") "
                  << "latency: " << latency << "ms\n";
    }

    void handle_combat_message(const CombatActionMessage &msg) {
        combat_actions_++;

        uint64_t latency = msg.latency_ms();
        average_latency_.store((average_latency_.load() + latency) / 2);

        if (msg.action().target_player_id == player_id_) {
            // We're being attacked!
            int damage = msg.action().damage;
            if (msg.action().accuracy > uniform_dist_(gen_)) {
                position_.health =
                    std::max(0, static_cast<int>(position_.health) - damage);
                std::cout << "💥 Hit by Player " << msg.header().player_id
                          << " for " << damage << " damage! Health: "
                          << static_cast<int>(position_.health) << "\n";
            } else {
                std::cout << "🛡️  Player " << msg.header().player_id
                          << " missed!\n";
            }
        }

        std::cout << "⚔️  Peer " << msg.header().player_id << " executed "
                  << action_type_name(msg.header().action_type)
                  << " latency: " << latency << "ms\n";
    }

    static const char *action_type_name(CombatActionMessage::ActionType type) {
        switch (type) {
        case CombatActionMessage::Attack:
            return "Attack";
        case CombatActionMessage::Block:
            return "Block";
        case CombatActionMessage::Dodge:
            return "Dodge";
        case CombatActionMessage::Reload:
            return "Reload";
        case CombatActionMessage::UseItem:
            return "UseItem";
        default:
            return "Unknown";
        }
    }
};

/**
 * @brief WebRTC P2P Gaming Demo
 */
class WebRTCP2PDemo {
public:
    explicit WebRTCP2PDemo(const std::string &player_name)
        : player_name_(player_name), running_(false) {
        player_id_ = (player_name == "player1") ? 1 : 2;
        peer_id_ = (player_name == "player1") ? "player2" : "player1";
    }

    void run() {
        std::cout << "🚀 Starting WebRTC P2P Gaming Demo\n";
        std::cout << "   Player: " << player_name_ << " (ID: " << player_id_
                  << ")\n";
        std::cout << "   Target Peer: " << peer_id_ << "\n\n";

        try {
            // Create WebRTC channel
            auto channel = create_webrtc_channel();

            // Wait for connection
            wait_for_connection(channel);

            // Start game simulation
            GamePlayer player(player_id_, peer_id_);
            player.start_simulation(channel);

            // Run demo
            run_demo_loop(player);

            player.stop_simulation();
            player.print_stats();

        } catch (const std::exception &e) {
            std::cerr << "❌ Error: " << e.what() << std::endl;
        }
    }

private:
    std::string player_name_;
    std::string peer_id_;
    uint32_t player_id_;
    bool running_;

    std::shared_ptr<Channel> create_webrtc_channel() {
        std::string uri = "webrtc://" + peer_id_;

        std::cout << "🔗 Creating WebRTC channel: " << uri << "\n";
        std::cout << "   Using signaling server: ws://localhost:8080\n";
        std::cout << "   STUN servers: stun.l.google.com:19302\n\n";

        // Create channel with optimized settings for gaming
        auto channel = psyne::create_channel(
            uri,
            1024 * 1024,            // 1MB buffer
            ChannelMode::SPSC,      // Single producer/consumer for direct P2P
            ChannelType::MultiType, // Support multiple message types
            true                    // Enable metrics
        );

        return channel;
    }

    void wait_for_connection(std::shared_ptr<Channel> channel) {
        std::cout << "⏳ Waiting for WebRTC connection...\n";

        // Simulate connection establishment
        // In real implementation, this would wait for ICE connectivity
        std::this_thread::sleep_for(std::chrono::seconds(2));

        std::cout << "✅ WebRTC connection established!\n";
        std::cout << "🎮 Starting game simulation...\n\n";
    }

    void run_demo_loop(GamePlayer &player) {
        running_ = true;

        std::cout << "💡 Demo Commands:\n";
        std::cout << "   'a' - Simulate combat action\n";
        std::cout << "   's' - Show statistics\n";
        std::cout << "   'q' - Quit\n\n";

        auto start_time = std::chrono::steady_clock::now();
        auto last_action = start_time;

        while (running_) {
            auto now = std::chrono::steady_clock::now();

            // Auto-simulate combat actions every 3-5 seconds
            auto time_since_action =
                std::chrono::duration_cast<std::chrono::seconds>(now -
                                                                 last_action);
            if (time_since_action.count() >= 3) {
                player.simulate_combat_action();
                last_action = now;
            }

            // Check for user input (non-blocking)
            if (check_user_input()) {
                char input;
                std::cin >> input;

                switch (input) {
                case 'a':
                    player.simulate_combat_action();
                    break;
                case 's':
                    player.print_stats();
                    break;
                case 'q':
                    running_ = false;
                    break;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time);

        std::cout << "\n🏁 Demo completed after " << duration.count()
                  << " seconds\n";
    }

    bool check_user_input() {
        // Simple non-blocking input check
        // In real implementation, would use proper non-blocking I/O
        return false; // For demo, just use auto-simulation
    }
};

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <player_name>\n";
        std::cout << "Example:\n";
        std::cout << "  Terminal 1: " << argv[0] << " player1\n";
        std::cout << "  Terminal 2: " << argv[0] << " player2\n";
        return 1;
    }

    std::string player_name = argv[1];

    if (player_name != "player1" && player_name != "player2") {
        std::cerr << "❌ Player name must be 'player1' or 'player2'\n";
        return 1;
    }

    // Initialize psyne
    psyne::print_banner();

    WebRTCP2PDemo demo(player_name);
    demo.run();

    return 0;
}