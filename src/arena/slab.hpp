#pragma once

#include "VulkanContext.hpp"
#include "ShaderBuffer.hpp"
#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <queue>
#include <vector>
#include <utility>
#include <unordered_map>
#include <shared_mutex>
#include <string>
#include <typeindex>
#include <memory>
#include <span>

namespace psyne {

struct Block {
    std::byte*  ptr;
    std::size_t bytes;
    uint32_t    id;
};

struct IBuffer {
    virtual ~IBuffer() = default;

    virtual VkBuffer   vkBuffer() const = 0;       // GPU handle
    virtual void*      mapPtr()   const = 0;       // CPU mapping
    virtual std::size_t size()    const = 0;       // bytes
    virtual const std::string& socket() const = 0; // key for registry
    virtual bool trySend(std::span<const std::byte> payload) = 0; // fast-path
    virtual std::span<const std::byte> tryPop() = 0; // fast-path

    virtual boost::asio::awaitable<std::span<const std::byte>>
    receive(boost::asio::io_context& io, int timeoutMs = 10) = 0; // async receive
};


class BufferRegistry {
public:
    // ───────── 1. Singleton accessor ─────────
    static BufferRegistry& instance() {
        static BufferRegistry reg;          // Meyers-singleton
        return reg;
    }

    // ──────────────────────────────────────────────────────────
    // Create-or-fetch typed slab
    // ──────────────────────────────────────────────────────────
    template<class T, class... Args>
    std::shared_ptr<T> getOrCreate(const std::string& socket, Args&&... a)
    {
        std::unique_lock lk(mu_);

        auto it = slabs_.find(socket);
        if (it == slabs_.end()) {
            auto ptr = std::make_shared<T>(socket, std::forward<Args>(a)...);
            slabs_.emplace(socket, std::static_pointer_cast<IBuffer>(ptr));
            kinds_.emplace(socket, std::type_index(typeid(T)));
            return ptr;
        }

        if (kinds_.at(socket) != std::type_index(typeid(T)))
            throw std::runtime_error("Slab type mismatch on " + socket);

        return std::static_pointer_cast<T>(it->second);
    }

    // ──────────────────────────────────────────────────────────
    // Lookup only
    // ──────────────────────────────────────────────────────────
    template<class T>
    std::shared_ptr<T> at(const std::string& socket) const
    {
        std::shared_lock lk(mu_);
        auto it = slabs_.find(socket);
        if (it == slabs_.end())
            throw std::runtime_error("No slab for " + socket);
        if (kinds_.at(socket) != std::type_index(typeid(T)))
            throw std::runtime_error("Slab type mismatch on " + socket);
        return std::static_pointer_cast<T>(it->second);
    }

    void erase(const std::string& socket)
    {
        std::unique_lock lk(mu_);
        slabs_.erase(socket);
        kinds_.erase(socket);
    }

    BufferRegistry(const BufferRegistry&)            = delete;
    BufferRegistry& operator=(const BufferRegistry&) = delete;

private:
    BufferRegistry()  = default;
    ~BufferRegistry() = default;

    mutable std::shared_mutex                                 mu_;
    std::unordered_map<std::string, std::shared_ptr<IBuffer>> slabs_;
    std::unordered_map<std::string, std::type_index>          kinds_;
};

} // namespace psyne