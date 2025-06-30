#pragma once

#include "../webrtc_channel.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace psyne {
namespace detail {
namespace webrtc {

/**
 * @brief STUN message types (RFC 5389)
 */
enum class STUNMessageType : uint16_t {
    BindingRequest = 0x0001,
    BindingResponse = 0x0101,
    BindingErrorResponse = 0x0111,
    BindingIndication = 0x0011
};

/**
 * @brief STUN attribute types
 */
enum class STUNAttributeType : uint16_t {
    MappedAddress = 0x0001,
    Username = 0x0006,
    MessageIntegrity = 0x0008,
    ErrorCode = 0x0009,
    UnknownAttributes = 0x000A,
    Realm = 0x0014,
    Nonce = 0x0015,
    XorMappedAddress = 0x0020,
    Software = 0x8022,
    AlternateServer = 0x8023,
    Fingerprint = 0x8028
};

/**
 * @brief STUN message header
 */
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
struct STUNHeader {
    uint16_t message_type;
    uint16_t message_length;
    uint32_t magic_cookie;
    uint8_t transaction_id[12];
}
#ifdef _MSC_VER
#pragma pack(pop)
#else
__attribute__((packed))
#endif
;

/**
 * @brief STUN attribute header
 */
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
struct STUNAttributeHeader {
    uint16_t type;
    uint16_t length;
}
#ifdef _MSC_VER
#pragma pack(pop)
#else
__attribute__((packed))
#endif
;

/**
 * @brief Network address representation
 */
struct NetworkAddress {
    enum Family { IPv4 = 1, IPv6 = 2 } family;
    std::string address;
    uint16_t port;

    std::string to_string() const {
        return address + ":" + std::to_string(port);
    }

    bool operator==(const NetworkAddress &other) const {
        return family == other.family && address == other.address &&
               port == other.port;
    }
};

/**
 * @brief ICE candidate pair for connectivity checks
 */
struct CandidatePair {
    RTCIceCandidate local;
    RTCIceCandidate remote;
    uint64_t priority;
    enum State {
        Waiting,
        InProgress,
        Succeeded,
        Failed,
        Frozen
    } state = Waiting;

    std::chrono::system_clock::time_point last_check;
    uint32_t rtt_ms = 0;
};

/**
 * @brief STUN client for NAT discovery and binding requests
 */
class STUNClient {
public:
    explicit STUNClient(const STUNServerConfig &config);
    ~STUNClient();

    // Asynchronous binding request
    void send_binding_request(
        std::function<void(bool success, const NetworkAddress &mapped_address)>
            callback);

    // Synchronous binding request with timeout
    std::optional<NetworkAddress> get_mapped_address(
        std::chrono::milliseconds timeout = std::chrono::seconds(5));

    // Keep-alive functionality
    void start_keep_alive(
        std::chrono::milliseconds interval = std::chrono::seconds(30));
    void stop_keep_alive();

private:
    STUNServerConfig config_;
    int socket_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    std::thread keep_alive_thread_;

    // Transaction management
    std::mutex transactions_mutex_;
    std::unordered_map<std::string,
                       std::function<void(bool, const NetworkAddress &)>>
        pending_transactions_;

    void initialize_socket();
    void run_worker();
    void run_keep_alive(std::chrono::milliseconds interval);
    std::vector<uint8_t>
    create_binding_request(const std::string &transaction_id);
    bool parse_binding_response(const std::vector<uint8_t> &data,
                                NetworkAddress &mapped_address);
    std::string generate_transaction_id();
    uint32_t calculate_crc32(const std::vector<uint8_t> &data);
    void add_fingerprint_attribute(std::vector<uint8_t> &message);
};

/**
 * @brief ICE agent for managing ICE candidates and connectivity checks
 */
class ICEAgent {
public:
    explicit ICEAgent(const WebRTCConfig &config);
    ~ICEAgent();

    // ICE gathering
    void start_gathering();
    void stop_gathering();
    bool is_gathering() const {
        return gathering_.load();
    }

    // Candidate management
    void add_remote_candidate(const RTCIceCandidate &candidate);
    std::vector<RTCIceCandidate> get_local_candidates() const;

    // Connectivity checks
    void start_connectivity_checks();
    void stop_connectivity_checks();
    bool is_checking() const {
        return checking_.load();
    }

    // Best candidate pair
    std::optional<CandidatePair> get_selected_pair() const;
    bool has_connection() const;

    // Event callbacks
    std::function<void(const RTCIceCandidate &)> on_candidate;
    std::function<void(RTCIceConnectionState)> on_connection_state_change;
    std::function<void(const CandidatePair &)> on_selected_pair_change;

private:
    WebRTCConfig config_;
    std::atomic<bool> gathering_{false};
    std::atomic<bool> checking_{false};
    RTCIceConnectionState connection_state_ = RTCIceConnectionState::New;

    // Candidates
    mutable std::mutex candidates_mutex_;
    std::vector<RTCIceCandidate> local_candidates_;
    std::vector<RTCIceCandidate> remote_candidates_;

    // Candidate pairs and connectivity checks
    mutable std::mutex pairs_mutex_;
    std::vector<CandidatePair> candidate_pairs_;
    std::optional<CandidatePair> selected_pair_;

    // Threading
    std::thread gathering_thread_;
    std::thread checking_thread_;

    // STUN clients for each server
    std::vector<std::unique_ptr<STUNClient>> stun_clients_;

    // Private methods
    void run_gathering();
    void run_connectivity_checks();
    void gather_host_candidates();
    void gather_server_reflexive_candidates();
    void gather_relay_candidates();
    void create_candidate_pairs();
    void perform_connectivity_check(CandidatePair &pair);
    uint64_t calculate_pair_priority(const RTCIceCandidate &local,
                                     const RTCIceCandidate &remote);
    uint32_t calculate_candidate_priority(RTCIceCandidateType type,
                                          uint16_t local_pref,
                                          uint8_t component_id);
    void update_connection_state();
    void notify_connection_state_change(RTCIceConnectionState new_state);
    std::vector<NetworkAddress> get_local_interfaces();
    RTCIceCandidate create_host_candidate(const NetworkAddress &address,
                                          uint8_t component);
    bool send_stun_binding_request(const NetworkAddress &local,
                                   const NetworkAddress &remote);
    std::string generate_transaction_id();
    std::vector<uint8_t>
    create_stun_check_request(const std::string &transaction_id);
    bool validate_stun_response(const std::vector<uint8_t> &data,
                                const std::string &transaction_id);
};

/**
 * @brief TURN client for relayed connectivity
 */
class TURNClient {
public:
    explicit TURNClient(const TURNServerConfig &config);
    ~TURNClient();

    // Allocation management
    void allocate_relay(
        std::function<void(bool success, const NetworkAddress &relay_address)>
            callback);
    void refresh_allocation();
    void deallocate();

    // Data forwarding
    void send_data(const NetworkAddress &peer, const void *data, size_t size);

    // Event callbacks
    std::function<void(const NetworkAddress &peer, const void *data,
                       size_t size)>
        on_data;

private:
    TURNServerConfig config_;
    NetworkAddress relay_address_;
    std::string allocation_id_;
    std::atomic<bool> allocated_{false};

    // Implementation details would include TURN protocol handling
};

} // namespace webrtc
} // namespace detail
} // namespace psyne