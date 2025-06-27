#pragma once

#include <cstdint>
#include <cstring>
#include <boost/asio.hpp>

namespace psyne {

// TCP message frame header
struct TCPFrameHeader {
    uint32_t length;      // Total bytes after this header (network byte order)
    uint32_t checksum;    // xxhash32 of the payload
    
    // Convert to/from network byte order
    void to_network() {
        length = boost::asio::detail::socket_ops::host_to_network_long(length);
        checksum = boost::asio::detail::socket_ops::host_to_network_long(checksum);
    }
    
    void to_host() {
        length = boost::asio::detail::socket_ops::network_to_host_long(length);
        checksum = boost::asio::detail::socket_ops::network_to_host_long(checksum);
    }
};

// Calculate xxhash32 checksum
uint32_t calculate_xxhash32(const void* data, size_t len);

// Message framing for TCP
class TCPFramer {
public:
    // Create a frame header for a message
    static TCPFrameHeader create_header(const void* data, size_t len) {
        TCPFrameHeader header;
        header.length = static_cast<uint32_t>(len);
        header.checksum = calculate_xxhash32(data, len);
        header.to_network();
        return header;
    }
    
    // Verify a received frame
    static bool verify_frame(const TCPFrameHeader& header, const void* data, size_t len) {
        TCPFrameHeader local_header = header;
        local_header.to_host();
        
        if (local_header.length != len) {
            return false;
        }
        
        uint32_t calculated_checksum = calculate_xxhash32(data, len);
        return calculated_checksum == local_header.checksum;
    }
    
    // Maximum frame size (64MB)
    static constexpr size_t MAX_FRAME_SIZE = 64 * 1024 * 1024;
};

}  // namespace psyne