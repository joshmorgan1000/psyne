#include "ice_agent.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <random>

#ifdef _WIN32
#include <iphlpapi.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#define close closesocket
#define ssize_t int
#define socklen_t int
// Windows byte order functions
#define htobe64(x) _byteswap_uint64(x)
#define be64toh(x) _byteswap_uint64(x)
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <net/if.h>
#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#include <machine/endian.h>
#define htobe64(x) OSSwapHostToBigInt64(x)
#define be64toh(x) OSSwapBigToHostInt64(x)
#define htobe32(x) OSSwapHostToBigInt32(x)
#define be32toh(x) OSSwapBigToHostInt32(x)
#define htobe16(x) OSSwapHostToBigInt16(x)
#define be16toh(x) OSSwapBigToHostInt16(x)
#elif defined(__linux__)
#include <endian.h>
// Linux should have these in endian.h, but provide fallbacks if needed
#ifndef htobe64
#include <byteswap.h>
#define htobe64(x) __bswap_64(x)
#define be64toh(x) __bswap_64(x)
#define htobe32(x) __bswap_32(x)
#define be32toh(x) __bswap_32(x)
#define htobe16(x) __bswap_16(x)
#define be16toh(x) __bswap_16(x)
#endif
#else
#include <endian.h>
#endif
#endif

namespace psyne {
namespace detail {
namespace webrtc {

// STUN magic cookie (RFC 5389)
static constexpr uint32_t STUN_MAGIC_COOKIE = 0x2112A442;

STUNClient::STUNClient(const STUNServerConfig &config) : config_(config) {
    initialize_socket();
}

STUNClient::~STUNClient() {
    stop_keep_alive();
    running_.store(false);
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    if (socket_fd_ >= 0) {
        close(socket_fd_);
    }
#ifdef _WIN32
    WSACleanup();
#endif
}

void STUNClient::initialize_socket() {
#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        throw std::runtime_error("Failed to initialize Winsock");
    }
#endif

    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        std::string error = "Failed to create UDP socket for STUN client: ";
#ifdef __linux__
        error += std::string(strerror(errno));
#endif
        throw std::runtime_error(error);
    }

    // Set socket options for better Linux compatibility
#ifndef _WIN32
    // Allow address reuse (important for Linux)
    int reuse = 1;
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        std::cerr << "Warning: Failed to set SO_REUSEADDR on STUN socket" << std::endl;
    }
    
    // On Linux, also set SO_REUSEPORT if available
#ifdef SO_REUSEPORT
    if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse)) < 0) {
        // This is not critical, just log
        std::cerr << "Note: SO_REUSEPORT not available on this system" << std::endl;
    }
#endif
#endif

    // Set non-blocking
#ifdef _WIN32
    u_long mode = 1;
    if (ioctlsocket(socket_fd_, FIONBIO, &mode) != 0) {
        throw std::runtime_error("Failed to set non-blocking mode on Windows");
    }
#else
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    if (flags == -1 || fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) == -1) {
        std::string error = "Failed to set non-blocking mode: ";
        error += std::string(strerror(errno));
        throw std::runtime_error(error);
    }
#endif

    // Start worker thread
    running_.store(true);
    worker_thread_ = std::thread(&STUNClient::run_worker, this);
}

void STUNClient::send_binding_request(
    std::function<void(bool, const NetworkAddress &)> callback) {
    std::string transaction_id = generate_transaction_id();
    auto request = create_binding_request(transaction_id);

    // Store callback for this transaction
    {
        std::lock_guard<std::mutex> lock(transactions_mutex_);
        pending_transactions_[transaction_id] = callback;
    }

    // Send to STUN server
    struct sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(config_.port);
    inet_pton(AF_INET, config_.host.c_str(), &server_addr.sin_addr);

    ssize_t sent = sendto(socket_fd_, request.data(), request.size(), 0,
                          (struct sockaddr *)&server_addr, sizeof(server_addr));

    if (sent < 0) {
        std::lock_guard<std::mutex> lock(transactions_mutex_);
        pending_transactions_.erase(transaction_id);
#ifdef __linux__
        std::cerr << "STUN sendto failed: " << strerror(errno) << " (errno=" << errno << ")" << std::endl;
        // Common Linux networking issues
        if (errno == ENETUNREACH) {
            std::cerr << "Network unreachable - check routing table" << std::endl;
        } else if (errno == EACCES) {
            std::cerr << "Permission denied - check firewall settings" << std::endl;
        } else if (errno == ENOBUFS) {
            std::cerr << "No buffer space available - system overloaded?" << std::endl;
        }
#endif
        callback(false, {});
    }
}

std::optional<NetworkAddress>
STUNClient::get_mapped_address(std::chrono::milliseconds timeout) {
    std::mutex result_mutex;
    std::condition_variable result_cv;
    std::optional<NetworkAddress> result;
    bool completed = false;

    send_binding_request([&](bool success, const NetworkAddress &addr) {
        std::lock_guard<std::mutex> lock(result_mutex);
        if (success) {
            result = addr;
        }
        completed = true;
        result_cv.notify_one();
    });

    std::unique_lock<std::mutex> lock(result_mutex);
    result_cv.wait_for(lock, timeout, [&] { return completed; });

    return result;
}

void STUNClient::run_worker() {
    std::vector<uint8_t> buffer(1500); // Standard MTU size

    while (running_.load()) {
        struct sockaddr_in from_addr{};
        socklen_t from_len = sizeof(from_addr);

        ssize_t received = recvfrom(socket_fd_, buffer.data(), buffer.size(), 0,
                                    (struct sockaddr *)&from_addr, &from_len);

        if (received > 0) {
            buffer.resize(received);

            // Parse STUN response
            NetworkAddress mapped_address;
            if (parse_binding_response(buffer, mapped_address)) {
                // Extract transaction ID to find callback
                if (received >= sizeof(STUNHeader)) {
                    STUNHeader *header =
                        reinterpret_cast<STUNHeader *>(buffer.data());
                    std::string transaction_id(
                        reinterpret_cast<char *>(header->transaction_id), 12);

                    std::lock_guard<std::mutex> lock(transactions_mutex_);
                    auto it = pending_transactions_.find(transaction_id);
                    if (it != pending_transactions_.end()) {
                        it->second(true, mapped_address);
                        pending_transactions_.erase(it);
                    }
                }
            }
        } else if (received == -1) {
            // Handle different error conditions
#ifdef __linux__
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No data available, this is normal for non-blocking socket
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else if (errno == EINTR) {
                // Interrupted by signal, continue
                continue;
            } else if (errno == ECONNREFUSED) {
                std::cerr << "STUN server connection refused" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } else {
                std::cerr << "STUN recvfrom error: " << strerror(errno) << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
#else
            // Small delay to avoid busy waiting on other platforms
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
#endif
        } else {
            // received == 0, should not happen with UDP
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

std::vector<uint8_t>
STUNClient::create_binding_request(const std::string &transaction_id) {
    std::vector<uint8_t> message;

    // STUN header
    STUNHeader header{};
    header.message_type =
        htons(static_cast<uint16_t>(STUNMessageType::BindingRequest));
    header.message_length = 0; // Will be updated later
    header.magic_cookie = htonl(STUN_MAGIC_COOKIE);
    std::memcpy(header.transaction_id, transaction_id.data(), 12);

    message.insert(message.end(), reinterpret_cast<uint8_t *>(&header),
                   reinterpret_cast<uint8_t *>(&header) + sizeof(header));

    // Add SOFTWARE attribute
    std::string software = "psyne-webrtc-1.0";
    STUNAttributeHeader software_attr{};
    software_attr.type =
        htons(static_cast<uint16_t>(STUNAttributeType::Software));
    software_attr.length = htons(software.length());

    message.insert(message.end(), reinterpret_cast<uint8_t *>(&software_attr),
                   reinterpret_cast<uint8_t *>(&software_attr) +
                       sizeof(software_attr));
    message.insert(message.end(), software.begin(), software.end());

    // Pad to 4-byte boundary
    while (message.size() % 4 != 0) {
        message.push_back(0);
    }

    // Update message length
    uint16_t length = message.size() - sizeof(STUNHeader);
    reinterpret_cast<STUNHeader *>(message.data())->message_length =
        htons(length);

    // Add FINGERPRINT attribute
    add_fingerprint_attribute(message);

    return message;
}

bool STUNClient::parse_binding_response(const std::vector<uint8_t> &data,
                                        NetworkAddress &mapped_address) {
    if (data.size() < sizeof(STUNHeader)) {
        return false;
    }

    const STUNHeader *header =
        reinterpret_cast<const STUNHeader *>(data.data());
    if (ntohl(header->magic_cookie) != STUN_MAGIC_COOKIE) {
        return false;
    }

    if (ntohs(header->message_type) !=
        static_cast<uint16_t>(STUNMessageType::BindingResponse)) {
        return false;
    }

    // Parse attributes
    size_t offset = sizeof(STUNHeader);
    uint16_t message_length = ntohs(header->message_length);

    while (offset + sizeof(STUNAttributeHeader) <=
           sizeof(STUNHeader) + message_length) {
        const STUNAttributeHeader *attr =
            reinterpret_cast<const STUNAttributeHeader *>(data.data() + offset);
        uint16_t attr_type = ntohs(attr->type);
        uint16_t attr_length = ntohs(attr->length);

        if (attr_type ==
                static_cast<uint16_t>(STUNAttributeType::XorMappedAddress) ||
            attr_type ==
                static_cast<uint16_t>(STUNAttributeType::MappedAddress)) {
            if (attr_length >= 8) { // Minimum size for IPv4 address
                const uint8_t *addr_data =
                    data.data() + offset + sizeof(STUNAttributeHeader);
                uint8_t family = addr_data[1];
                uint16_t port =
                    ntohs(*reinterpret_cast<const uint16_t *>(addr_data + 2));
                uint32_t address =
                    ntohl(*reinterpret_cast<const uint32_t *>(addr_data + 4));

                if (attr_type == static_cast<uint16_t>(
                                     STUNAttributeType::XorMappedAddress)) {
                    // XOR with magic cookie
                    port ^= (STUN_MAGIC_COOKIE >> 16);
                    address ^= STUN_MAGIC_COOKIE;
                }

                if (family == 1) { // IPv4
                    struct in_addr in_addr;
                    in_addr.s_addr = htonl(address);
                    char addr_str[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &in_addr, addr_str, sizeof(addr_str));

                    mapped_address.family = NetworkAddress::IPv4;
                    mapped_address.address = addr_str;
                    mapped_address.port = port;
                    return true;
                }
            }
        }

        // Move to next attribute (padded to 4-byte boundary)
        offset += sizeof(STUNAttributeHeader) + ((attr_length + 3) & ~3);
    }

    return false;
}

std::string STUNClient::generate_transaction_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint8_t> dis(0, 255);

    std::string id(12, 0);
    for (size_t i = 0; i < 12; ++i) {
        id[i] = dis(gen);
    }
    return id;
}

void STUNClient::add_fingerprint_attribute(std::vector<uint8_t> &message) {
    // Calculate CRC32 over the message
    uint32_t crc = calculate_crc32(message) ^ 0x5354554e; // XOR with "STUN"

    STUNAttributeHeader fingerprint_attr{};
    fingerprint_attr.type =
        htons(static_cast<uint16_t>(STUNAttributeType::Fingerprint));
    fingerprint_attr.length = htons(4);

    message.insert(message.end(),
                   reinterpret_cast<uint8_t *>(&fingerprint_attr),
                   reinterpret_cast<uint8_t *>(&fingerprint_attr) +
                       sizeof(fingerprint_attr));

    uint32_t crc_be = htonl(crc);
    message.insert(message.end(), reinterpret_cast<uint8_t *>(&crc_be),
                   reinterpret_cast<uint8_t *>(&crc_be) + sizeof(crc_be));

    // Update message length
    uint16_t length = message.size() - sizeof(STUNHeader);
    reinterpret_cast<STUNHeader *>(message.data())->message_length =
        htons(length);
}

uint32_t STUNClient::calculate_crc32(const std::vector<uint8_t> &data) {
    // Simple CRC32 implementation
    static const uint32_t crc_table[256] = {
        // Standard CRC32 table (truncated for brevity)
        0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
        0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
        // ... (full table would be here)
    };

    uint32_t crc = 0xffffffff;
    for (uint8_t byte : data) {
        crc = crc_table[(crc ^ byte) & 0xff] ^ (crc >> 8);
    }
    return crc ^ 0xffffffff;
}

void STUNClient::start_keep_alive(std::chrono::milliseconds interval) {
    stop_keep_alive();
    keep_alive_thread_ =
        std::thread(&STUNClient::run_keep_alive, this, interval);
}

void STUNClient::stop_keep_alive() {
    if (keep_alive_thread_.joinable()) {
        keep_alive_thread_.join();
    }
}

void STUNClient::run_keep_alive(std::chrono::milliseconds interval) {
    while (running_.load()) {
        send_binding_request([](bool success, const NetworkAddress &addr) {
            // Keep-alive binding request, ignore result
        });

        std::this_thread::sleep_for(interval);
    }
}

// ICEAgent implementation

ICEAgent::ICEAgent(const WebRTCConfig &config) : config_(config) {
    // Initialize STUN clients
    for (const auto &stun_config : config_.stun_servers) {
        stun_clients_.push_back(std::make_unique<STUNClient>(stun_config));
    }
}

ICEAgent::~ICEAgent() {
    stop_gathering();
    stop_connectivity_checks();
}

void ICEAgent::start_connectivity_checks() {
    if (checking_.exchange(true)) {
        return; // Already checking
    }

    checking_thread_ = std::thread(&ICEAgent::run_connectivity_checks, this);
}

void ICEAgent::stop_connectivity_checks() {
    if (!checking_.exchange(false)) {
        return; // Not checking
    }

    if (checking_thread_.joinable()) {
        checking_thread_.join();
    }
}

void ICEAgent::run_connectivity_checks() {
    while (checking_.load()) {
        {
            std::lock_guard<std::mutex> lock(pairs_mutex_);

            // Process candidate pairs in priority order
            for (auto &pair : candidate_pairs_) {
                if (pair.state == CandidatePair::Waiting) {
                    perform_connectivity_check(pair);

                    // Rate limit checks
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            }

            // Update connection state based on check results
            update_connection_state();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void ICEAgent::start_gathering() {
    if (gathering_.exchange(true)) {
        return; // Already gathering
    }

    notify_connection_state_change(RTCIceConnectionState::Gathering);
    gathering_thread_ = std::thread(&ICEAgent::run_gathering, this);
}

void ICEAgent::stop_gathering() {
    if (!gathering_.exchange(false)) {
        return; // Not gathering
    }

    if (gathering_thread_.joinable()) {
        gathering_thread_.join();
    }
}

void ICEAgent::run_gathering() {
    gather_host_candidates();
    gather_server_reflexive_candidates();

    if (!config_.turn_servers.empty()) {
        gather_relay_candidates();
    }

    notify_connection_state_change(RTCIceConnectionState::Checking);
}

void ICEAgent::gather_host_candidates() {
    auto interfaces = get_local_interfaces();

    for (const auto &addr : interfaces) {
        auto candidate = create_host_candidate(addr, 1); // Component 1 for RTP

        {
            std::lock_guard<std::mutex> lock(candidates_mutex_);
            local_candidates_.push_back(candidate);
        }

        if (on_candidate) {
            on_candidate(candidate);
        }
    }
}

void ICEAgent::gather_server_reflexive_candidates() {
    for (auto &stun_client : stun_clients_) {
        auto mapped_addr = stun_client->get_mapped_address();
        if (mapped_addr) {
            RTCIceCandidate candidate;
            candidate.type = RTCIceCandidateType::ServerReflexive;
            candidate.foundation = "srflx";
            candidate.component = 1;
            candidate.transport = "udp";
            candidate.priority = calculate_candidate_priority(
                RTCIceCandidateType::ServerReflexive, 65535, 1);
            candidate.address = mapped_addr->address;
            candidate.port = mapped_addr->port;
            candidate.candidate = "candidate:" + candidate.foundation + " " +
                                  std::to_string(candidate.component) + " " +
                                  candidate.transport + " " +
                                  std::to_string(candidate.priority) + " " +
                                  candidate.address + " " +
                                  std::to_string(candidate.port) + " typ srflx";

            {
                std::lock_guard<std::mutex> lock(candidates_mutex_);
                local_candidates_.push_back(candidate);
            }

            if (on_candidate) {
                on_candidate(candidate);
            }
        }
    }
}

void ICEAgent::gather_relay_candidates() {
    // TURN implementation would go here
    // For now, just placeholder
}

std::vector<NetworkAddress> ICEAgent::get_local_interfaces() {
    std::vector<NetworkAddress> interfaces;

#ifdef _WIN32
    // Windows implementation using GetAdaptersAddresses
    ULONG bufferSize = 0;
    GetAdaptersAddresses(AF_INET,
                         GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                             GAA_FLAG_SKIP_DNS_SERVER,
                         nullptr, nullptr, &bufferSize);

    std::vector<char> buffer(bufferSize);
    PIP_ADAPTER_ADDRESSES adapters =
        reinterpret_cast<PIP_ADAPTER_ADDRESSES>(buffer.data());

    if (GetAdaptersAddresses(AF_INET,
                             GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                                 GAA_FLAG_SKIP_DNS_SERVER,
                             nullptr, adapters, &bufferSize) == NO_ERROR) {
        for (PIP_ADAPTER_ADDRESSES adapter = adapters; adapter != nullptr;
             adapter = adapter->Next) {
            if (adapter->OperStatus == IfOperStatusUp) {
                for (PIP_ADAPTER_UNICAST_ADDRESS addr =
                         adapter->FirstUnicastAddress;
                     addr != nullptr; addr = addr->Next) {
                    if (addr->Address.lpSockaddr->sa_family == AF_INET) {
                        struct sockaddr_in *addr_in =
                            reinterpret_cast<struct sockaddr_in *>(
                                addr->Address.lpSockaddr);
                        char addr_str[INET_ADDRSTRLEN];
                        inet_ntop(AF_INET, &addr_in->sin_addr, addr_str,
                                  sizeof(addr_str));

                        std::string addr_string(addr_str);

                        // Skip loopback and other special addresses
                        if (addr_string != "127.0.0.1" &&
                            addr_string.substr(0, 3) != "169") {
                            NetworkAddress address;
                            address.family = NetworkAddress::IPv4;
                            address.address = addr_string;
                            address.port = 0; // Will be assigned later
                            interfaces.push_back(address);
                        }
                    }
                }
            }
        }
    }
#else
    // Unix implementation using getifaddrs
    struct ifaddrs *ifaddr;
    if (getifaddrs(&ifaddr) == -1) {
        return interfaces;
    }

    for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr)
            continue;

        if (ifa->ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in *addr_in =
                reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
            char addr_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &addr_in->sin_addr, addr_str, sizeof(addr_str));

            std::string addr_string(addr_str);

            // Skip loopback and invalid addresses, but include private IPs for WebRTC
            // More comprehensive filtering for Linux compatibility
            if (addr_string != "127.0.0.1" &&                     // Loopback
                addr_string.substr(0, 3) != "169" &&               // Link-local (169.254.x.x)
                addr_string != "0.0.0.0" &&                        // Invalid
                addr_string != "255.255.255.255" &&                // Broadcast
                ifa->ifa_flags & IFF_UP &&                         // Interface must be up
                ifa->ifa_flags & IFF_RUNNING &&                    // Interface must be running
                !(ifa->ifa_flags & IFF_LOOPBACK)) {                // Not loopback interface
                NetworkAddress addr;
                addr.family = NetworkAddress::IPv4;
                addr.address = addr_string;
                addr.port = 0; // Will be assigned later
                interfaces.push_back(addr);
            }
        }
    }

    freeifaddrs(ifaddr);
#endif
    return interfaces;
}

RTCIceCandidate ICEAgent::create_host_candidate(const NetworkAddress &address,
                                                uint8_t component) {
    RTCIceCandidate candidate;
    candidate.type = RTCIceCandidateType::Host;
    candidate.foundation = "host";
    candidate.component = component;
    candidate.transport = "udp";
    candidate.priority = calculate_candidate_priority(RTCIceCandidateType::Host,
                                                      65535, component);
    candidate.address = address.address;
    candidate.port = address.port;
    candidate.candidate =
        "candidate:" + candidate.foundation + " " +
        std::to_string(candidate.component) + " " + candidate.transport + " " +
        std::to_string(candidate.priority) + " " + candidate.address + " " +
        std::to_string(candidate.port) + " typ host";

    return candidate;
}

uint32_t ICEAgent::calculate_candidate_priority(RTCIceCandidateType type,
                                                uint16_t local_pref,
                                                uint8_t component_id) {
    uint8_t type_preference;
    switch (type) {
    case RTCIceCandidateType::Host:
        type_preference = 126;
        break;
    case RTCIceCandidateType::PeerReflexive:
        type_preference = 110;
        break;
    case RTCIceCandidateType::ServerReflexive:
        type_preference = 100;
        break;
    case RTCIceCandidateType::Relay:
        type_preference = 0;
        break;
    }

    return (type_preference << 24) | (local_pref << 8) | (256 - component_id);
}

void ICEAgent::add_remote_candidate(const RTCIceCandidate &candidate) {
    {
        std::lock_guard<std::mutex> lock(candidates_mutex_);
        remote_candidates_.push_back(candidate);
    }

    // Create new candidate pairs
    create_candidate_pairs();
}

void ICEAgent::create_candidate_pairs() {
    std::lock_guard<std::mutex> candidates_lock(candidates_mutex_);
    std::lock_guard<std::mutex> pairs_lock(pairs_mutex_);

    for (const auto &local : local_candidates_) {
        for (const auto &remote : remote_candidates_) {
            CandidatePair pair;
            pair.local = local;
            pair.remote = remote;
            pair.priority = calculate_pair_priority(local, remote);
            pair.state = CandidatePair::Waiting;

            candidate_pairs_.push_back(pair);
        }
    }

    // Sort by priority
    std::sort(candidate_pairs_.begin(), candidate_pairs_.end(),
              [](const CandidatePair &a, const CandidatePair &b) {
                  return a.priority > b.priority;
              });
}

uint64_t ICEAgent::calculate_pair_priority(const RTCIceCandidate &local,
                                           const RTCIceCandidate &remote) {
    uint32_t controlling_priority = std::max(local.priority, remote.priority);
    uint32_t controlled_priority = std::min(local.priority, remote.priority);

    return (static_cast<uint64_t>(controlling_priority) << 32) |
           controlled_priority;
}

void ICEAgent::notify_connection_state_change(RTCIceConnectionState new_state) {
    if (connection_state_ != new_state) {
        connection_state_ = new_state;
        if (on_connection_state_change) {
            on_connection_state_change(new_state);
        }
    }
}

std::vector<RTCIceCandidate> ICEAgent::get_local_candidates() const {
    std::lock_guard<std::mutex> lock(candidates_mutex_);
    return local_candidates_;
}

std::optional<CandidatePair> ICEAgent::get_selected_pair() const {
    std::lock_guard<std::mutex> lock(pairs_mutex_);
    return selected_pair_;
}

bool ICEAgent::has_connection() const {
    return connection_state_ == RTCIceConnectionState::Connected ||
           connection_state_ == RTCIceConnectionState::Completed;
}

void ICEAgent::perform_connectivity_check(CandidatePair &pair) {
    pair.state = CandidatePair::InProgress;
    pair.last_check = std::chrono::system_clock::now();

    NetworkAddress local_addr{NetworkAddress::IPv4, pair.local.address,
                              pair.local.port};
    NetworkAddress remote_addr{NetworkAddress::IPv4, pair.remote.address,
                               pair.remote.port};

    auto start_time = std::chrono::steady_clock::now();

    // Perform STUN connectivity check
    bool success = send_stun_binding_request(local_addr, remote_addr);

    if (success) {
        auto end_time = std::chrono::steady_clock::now();
        pair.rtt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_time - start_time)
                          .count();
        pair.state = CandidatePair::Succeeded;

        // Update selected pair if this is better
        if (!selected_pair_ || pair.priority > selected_pair_->priority) {
            selected_pair_ = pair;

            if (on_selected_pair_change) {
                on_selected_pair_change(pair);
            }
        }
    } else {
        pair.state = CandidatePair::Failed;
    }
}

bool ICEAgent::send_stun_binding_request(const NetworkAddress &local,
                                         const NetworkAddress &remote) {
    // Create a socket for the connectivity check
    int check_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (check_socket < 0) {
        return false;
    }

    // Bind to local address
    struct sockaddr_in local_addr{};
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(local.port);
    inet_pton(AF_INET, local.address.c_str(), &local_addr.sin_addr);

    if (bind(check_socket, reinterpret_cast<struct sockaddr *>(&local_addr),
             sizeof(local_addr)) < 0) {
        close(check_socket);
        return false;
    }

    // Create STUN binding request
    std::string transaction_id = generate_transaction_id();
    auto request = create_stun_check_request(transaction_id);

    // Send to remote address
    struct sockaddr_in remote_addr{};
    remote_addr.sin_family = AF_INET;
    remote_addr.sin_port = htons(remote.port);
    inet_pton(AF_INET, remote.address.c_str(), &remote_addr.sin_addr);

    ssize_t sent = sendto(check_socket, request.data(), request.size(), 0,
                          reinterpret_cast<struct sockaddr *>(&remote_addr),
                          sizeof(remote_addr));

    bool success = (sent > 0);

    // Set timeout for response
#ifdef _WIN32
    DWORD timeout = 1000; // 1 second in milliseconds
    setsockopt(check_socket, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout,
               sizeof(timeout));
#else
    struct timeval timeout{};
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(check_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout,
               sizeof(timeout));
#endif

    if (success) {
        // Wait for response
        std::vector<uint8_t> response(1500);
        struct sockaddr_in from_addr{};
        socklen_t from_len = sizeof(from_addr);

        ssize_t received = recvfrom(
            check_socket, response.data(), response.size(), 0,
            reinterpret_cast<struct sockaddr *>(&from_addr), &from_len);

        if (received > 0) {
            response.resize(received);
            success = validate_stun_response(response, transaction_id);
        } else {
            success = false;
        }
    }

    close(check_socket);
    return success;
}

std::string ICEAgent::generate_transaction_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint8_t> dis(0, 255);

    std::string id(12, 0);
    for (size_t i = 0; i < 12; ++i) {
        id[i] = dis(gen);
    }
    return id;
}

std::vector<uint8_t>
ICEAgent::create_stun_check_request(const std::string &transaction_id) {
    std::vector<uint8_t> message;

    // STUN header for binding request
    STUNHeader header{};
    header.message_type =
        htons(static_cast<uint16_t>(STUNMessageType::BindingRequest));
    header.message_length = 0; // Will be updated
    header.magic_cookie = htonl(STUN_MAGIC_COOKIE);
    std::memcpy(header.transaction_id, transaction_id.data(), 12);

    message.insert(message.end(), reinterpret_cast<uint8_t *>(&header),
                   reinterpret_cast<uint8_t *>(&header) + sizeof(header));

    // Add ICE-CONTROLLING or ICE-CONTROLLED attribute
    STUNAttributeHeader ice_attr{};
    ice_attr.type = htons(0x802A); // ICE-CONTROLLING
    ice_attr.length = htons(8);

    message.insert(message.end(), reinterpret_cast<uint8_t *>(&ice_attr),
                   reinterpret_cast<uint8_t *>(&ice_attr) + sizeof(ice_attr));

    // Add tie-breaker value (randomly generated for security)
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;
    uint64_t tie_breaker = dis(gen);
    tie_breaker = htobe64(tie_breaker);
    message.insert(message.end(), reinterpret_cast<uint8_t *>(&tie_breaker),
                   reinterpret_cast<uint8_t *>(&tie_breaker) +
                       sizeof(tie_breaker));

    // Update message length
    uint16_t length = message.size() - sizeof(STUNHeader);
    reinterpret_cast<STUNHeader *>(message.data())->message_length =
        htons(length);

    return message;
}

bool ICEAgent::validate_stun_response(const std::vector<uint8_t> &data,
                                      const std::string &transaction_id) {
    if (data.size() < sizeof(STUNHeader)) {
        return false;
    }

    const STUNHeader *header =
        reinterpret_cast<const STUNHeader *>(data.data());

    // Validate magic cookie
    if (ntohl(header->magic_cookie) != STUN_MAGIC_COOKIE) {
        return false;
    }

    // Check transaction ID
    if (std::memcmp(header->transaction_id, transaction_id.data(), 12) != 0) {
        return false;
    }

    // Check message type (should be binding response)
    uint16_t msg_type = ntohs(header->message_type);
    return (msg_type ==
            static_cast<uint16_t>(STUNMessageType::BindingResponse));
}

void ICEAgent::update_connection_state() {
    RTCIceConnectionState new_state = connection_state_;

    bool has_succeeded = false;
    bool has_in_progress = false;
    bool has_waiting = false;

    for (const auto &pair : candidate_pairs_) {
        switch (pair.state) {
        case CandidatePair::Succeeded:
            has_succeeded = true;
            break;
        case CandidatePair::InProgress:
            has_in_progress = true;
            break;
        case CandidatePair::Waiting:
            has_waiting = true;
            break;
        default:
            break;
        }
    }

    if (has_succeeded) {
        if (selected_pair_) {
            new_state = RTCIceConnectionState::Connected;
        }
    } else if (has_in_progress || has_waiting) {
        new_state = RTCIceConnectionState::Checking;
    } else {
        new_state = RTCIceConnectionState::Failed;
    }

    if (new_state != connection_state_) {
        notify_connection_state_change(new_state);
    }
}

} // namespace webrtc
} // namespace detail
} // namespace psyne
