/**
 * @file compile_test.cpp
 * @brief Simple compile test for WebRTC components
 */

#include "../webrtc_channel.hpp"
#include "ice_agent.hpp"
#include <iostream>

using namespace psyne::detail;

int main() {
    std::cout << "WebRTC compile test starting...\n";

    // Test basic type creation
    WebRTCConfig config;
    config.stun_servers.push_back({"stun.l.google.com", 19302, "", ""});

    std::cout << "WebRTCConfig created with " << config.stun_servers.size()
              << " STUN servers\n";

    // Test ICE candidate creation
    RTCIceCandidate candidate;
    candidate.type = RTCIceCandidateType::Host;
    candidate.address = "192.168.1.100";
    candidate.port = 54321;

    std::cout << "ICE candidate: " << candidate.address << ":" << candidate.port
              << "\n";

    // Test session description
    RTCSessionDescription desc;
    desc.type = RTCSessionDescription::Offer;
    desc.sdp = "v=0\r\no=- 123 2 IN IP4 127.0.0.1\r\n";

    std::cout << "SDP created: " << desc.sdp.substr(0, 20) << "...\n";

    std::cout << "✅ WebRTC compile test passed!\n";
    return 0;
}