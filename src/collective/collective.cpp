/**
 * @file collective.cpp
 * @brief Implementation of collective communication operations
 * 
 * @copyright Copyright (c) 2024 Psyne Project
 * @license MIT License
 */

#include <psyne/collective.hpp>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>

namespace psyne {
namespace collective {

RingCollectiveGroup::RingCollectiveGroup(RankId rank, std::vector<std::string> peer_uris)
    : rank_(rank), peer_uris_(std::move(peer_uris)) {
    
    // Create channels to all peers
    channels_.reserve(peer_uris_.size());
    
    for (size_t i = 0; i < peer_uris_.size(); ++i) {
        if (i != rank_) {
            try {
                // Create bidirectional channel to peer
                auto channel = create_channel(peer_uris_[i], 
                                            1024 * 1024, // 1MB buffer
                                            ChannelMode::SPSC);
                channels_.push_back(channel);
            } catch (const std::exception& e) {
                std::cerr << "Failed to create channel to " << peer_uris_[i] 
                         << ": " << e.what() << std::endl;
                channels_.push_back(nullptr);
            }
        } else {
            // Placeholder for self
            channels_.push_back(nullptr);
        }
    }
}

std::shared_ptr<Channel> RingCollectiveGroup::get_channel(RankId rank) {
    if (rank == rank_) {
        throw std::runtime_error("Cannot get channel to self");
    }
    
    if (rank >= channels_.size()) {
        throw std::out_of_range("Invalid rank: " + std::to_string(rank));
    }
    
    auto channel = channels_[rank];
    if (!channel) {
        // Lazy creation if needed
        try {
            channel = create_channel(peer_uris_[rank],
                                   1024 * 1024,
                                   ChannelMode::SPSC);
            channels_[rank] = channel;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create channel to rank " + 
                                   std::to_string(rank) + ": " + e.what());
        }
    }
    
    return channel;
}

void RingCollectiveGroup::barrier() {
    // Simple barrier implementation using all-to-all communication
    const size_t size = this->size();
    const uint32_t barrier_token = 0xDEADBEEF;
    
    // Send token to all other ranks
    for (size_t r = 0; r < size; ++r) {
        if (r != rank_) {
            try {
                auto channel = get_channel(r);
                channel->send(barrier_token);
            } catch (const std::exception& e) {
                std::cerr << "Barrier send failed to rank " << r << ": " 
                         << e.what() << std::endl;
            }
        }
    }
    
    // Receive token from all other ranks
    for (size_t r = 0; r < size; ++r) {
        if (r != rank_) {
            try {
                auto channel = get_channel(r);
                auto token = channel->receive<uint32_t>();
                
                if (!token || *token != barrier_token) {
                    std::cerr << "Barrier receive failed from rank " << r << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Barrier receive failed from rank " << r << ": " 
                         << e.what() << std::endl;
            }
        }
    }
}

} // namespace collective
} // namespace psyne