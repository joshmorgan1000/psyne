#pragma once

#include "VulkanContext.hpp"
#include "ShaderBuffer.hpp"
#include "utils/utils.hpp"
#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include "slab.hpp"

namespace psyne {

// ──────────────────────────────────────────────────────────────────────────────
// ByteSlab: a slab for raw byte buffers with in-process fast path and optional socket
// ingress.
// ──────────────────────────────────────────────────────────────────────────────
class ByteSlab : public IBuffer {
public:
    ByteSlab(std::string socketAddr_,
             std::size_t slabBytes = 1ull << 30,     // default 1 GiB
             uint32_t    deviceIdx = 0)
    : socket_addr(std::move(socketAddr_))
    , slabSize(slabBytes)
    , head(0)
    , tail(0)
    {
        // Vulkan buffer
        auto& ctx          = VulkanContext::getInstance(deviceIdx);
        ShaderBuffer::CreateInfo ci{};
        ci.name            = "psyne_bytes_slab";
        ci.device          = ctx.getDevice();
        ci.physicalDevice  = ctx.getPhysicalDevice();
        ci.size            = slabSize;
        ci.memoryProperties= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ci.access          = ShaderBuffer::Access::InputOutput;
        buf_   = std::make_unique<ShaderBuffer>(ci);
        base_  = static_cast<std::byte*>(buf_->map());
        end_   = base_ + slabSize;
    }

    // ──────────────────────────  IBuffer interface
    VkBuffer   vkBuffer() const final { return buf_->handle(); }
    void*      mapPtr()   const final { return base_; }
    std::size_t size()    const final { return slabSize; }
    const std::string& socket() const final { return socket_addr; }

    // ──────────────────────────  producer (in-proc fast path)
    // message = <4-byte len><payload[len]>
    bool trySend(std::span<const std::byte> payload)
    {
        std::size_t bytes = 4 + payload.size();
        std::byte*  w     = alloc(bytes);
        if (!w) return false;

        uint32_t len = static_cast<uint32_t>(payload.size());
        std::memcpy(w, &len, 4);
        std::memcpy(w + 4, payload.data(), payload.size());
        publish(w, bytes);
        return true;
    }

    // ──────────────────────────  single consumer coroutine
    boost::asio::awaitable<std::span<const std::byte>>
    receive(boost::asio::io_context& io, int timeoutMs = 10)
    {
        using namespace boost::asio;
        steady_timer t(io);
        for (;;) {
            if (auto s = tryPop(); !s.empty()) co_return s;

            t.expires_after(std::chrono::milliseconds(timeoutMs));
            co_await t.async_wait(use_awaitable);
        }
    }

    // ──────────────────────────  optional socket ingress
    void startSocket(boost::asio::io_context& io)
    {
        using tcp = boost::asio::ip::tcp;
        acc_ = std::make_unique<tcp::acceptor>(
            io, tcp::endpoint(tcp::v4(), parsePort(socket_addr)));

        boost::asio::co_spawn(io,
            [this]() -> boost::asio::awaitable<void> {
                while (true) {
                    tcp::socket s = co_await acc_->async_accept(
                                         boost::asio::use_awaitable);
                    co_spawn(acc_->get_executor(),
                             reader(std::move(s)),
                             boost::asio::detached);
                }
            },
            boost::asio::detached);
    }

private:
    // ▶ internal header
    struct Hdr { uint32_t bytes; };

    // ▶ ring helpers
    std::byte* alloc(std::size_t need)
    {
        uint32_t h = head.load(std::memory_order_acquire);
        uint32_t t = tail.load(std::memory_order_relaxed);

        if (t >= h) {
            if (offset(t) + need <= slabSize) return base_ + offset(t);
            if (need < h) return base_;           // wrap front
        } else {
            if (offset(t) + need < h) return base_ + offset(t);
        }
        return nullptr;                           // full
    }
    void publish(std::byte* w, std::size_t bytes)
    {
        tail.store(advance(static_cast<uint32_t>(w - base_), bytes), std::memory_order_release);
    }
    std::span<const std::byte> tryPop()
    {
        uint32_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return {};

        auto* hdr   = reinterpret_cast<Hdr*>(base_ + h);
        auto* data  = reinterpret_cast<const std::byte*>(hdr + 1);
        auto  len   = hdr->bytes - 4;          // bytes of payload

        head.store(advance(h, hdr->bytes), std::memory_order_release);
        return { data, len };
    }
    uint32_t advance(uint32_t i, std::size_t n) const {
        return (i + n >= slabSize) ? 0 : i + static_cast<uint32_t>(n);
    }
    uint32_t offset(uint32_t i) const { return i; }

    // ▶ socket parser: <4-byte len><payload>
    boost::asio::awaitable<void> reader(boost::asio::ip::tcp::socket sock)
    {
        using namespace boost::asio;
        for (;;) {
            uint32_t len;
            co_await async_read(sock, buffer(&len,4), use_awaitable);
            std::vector<std::byte> tmp(len);
            co_await async_read(sock, buffer(tmp), use_awaitable);

            while (!trySend(tmp)) {
                steady_timer timer(co_await this_coro::executor);
                timer.expires_after(std::chrono::milliseconds(1));
                co_await timer.async_wait(use_awaitable);
            }
        }
    }
    static uint16_t parsePort(const std::string& a) {
        auto p = a.rfind(':');
        return static_cast<uint16_t>(std::stoi(p == std::string::npos ? a : a.substr(p+1)));
    }

    // ▶ data members
public:
    std::string  socket_addr;
    std::size_t  slabSize;

private:
    std::unique_ptr<ShaderBuffer> buf_;
    std::byte*  base_{};
    std::byte*  end_{};

    std::atomic<uint32_t> head;   // consumer pointer
    std::atomic<uint32_t> tail;   // producer pointer

    std::unique_ptr<boost::asio::ip::tcp::acceptor> acc_;
};

} // namespace psyne