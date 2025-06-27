#pragma once

#include "arena/vectorslab.hpp"
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

namespace psyne {

class Broker {
public:
  using HashFn = std::function<uint32_t(uint64_t)>;

  Broker(std::shared_ptr<VectorSlab> ingress,
         std::vector<std::shared_ptr<VectorSlab>> shards,
         HashFn hash = {})
      : in_{std::move(ingress)}, shards_{std::move(shards)}, hash_{std::move(hash)} {
    if (!hash_)
      hash_ = [this](uint64_t k) { return k % static_cast<uint32_t>(shards_.size()); };
  }

  void pump(std::size_t maxBatch = 32) {
    for (std::size_t i = 0; i < maxBatch; ++i) {
      auto msg = in_->tryPop();
      if (msg.empty())
        break;
      if (msg.size() < sizeof(uint64_t))
        continue;
      uint64_t id;
      std::memcpy(&id, msg.data(), sizeof(id));
      auto idx = hash_(id) % static_cast<uint32_t>(shards_.size());
      while (!shards_[idx]->trySend(msg))
        ;
    }
  }

private:
  std::shared_ptr<VectorSlab> in_;
  std::vector<std::shared_ptr<VectorSlab>> shards_;
  HashFn hash_;
};

} // namespace psyne
