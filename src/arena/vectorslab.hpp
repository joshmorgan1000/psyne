#pragma once
#include "VulkanContext.hpp"
#include "ShaderBuffer.hpp"
#include "slab.hpp"
#include <boost/asio.hpp>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>

namespace psyne {

class VectorSlab : public IBuffer, public std::enable_shared_from_this<VectorSlab> {
public:
    VectorSlab(std::string socketAddr,
               uint16_t    vecDim,
               std::size_t slabBytes = 1ull << 30,
               uint32_t    deviceIdx = 0)
    : socket_addr(std::move(socketAddr))
    , vector_size(vecDim)
    , slabSize(slabBytes)
    {
        auto& ctx = VulkanContext::getInstance(deviceIdx);
        ShaderBuffer::CreateInfo ci{};
        ci.name             = "psyne_vector_slab";
        ci.device           = ctx.getDevice();
        ci.physicalDevice   = ctx.getPhysicalDevice();
        ci.size             = slabSize;
        ci.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ci.access           = ShaderBuffer::Access::InputOutput;

        buf_  = std::make_unique<ShaderBuffer>(ci);
        base_ = static_cast<std::byte*>(buf_->map());   // persistent map
    }

    // ───────── IBuffer interface (GPU helpers can rely on it) ────────
    VkBuffer vkBuffer() const override          { return buf_->getBuffer(); }
    void*    mapPtr()  const override           { return base_; }
    std::size_t size() const override           { return slabSize; }
    const std::string& socket() const override  { return socket_addr; }

    // ────────────────────────────────────────────────────────────────
    // Producer fast-path
    // ────────────────────────────────────────────────────────────────
    bool trySend(uint64_t id, std::span<const float> vec)
    {
        const std::size_t payload = 8 + 4 + vec.size_bytes();
        const std::size_t need    = sizeof(Header) + payload;

        std::byte* w = alloc(need);
        if (!w) return false;

        auto* h  = reinterpret_cast<Header*>(w);
        h->bytes = static_cast<uint32_t>(payload);

        std::byte* p = w + sizeof(Header);
        std::memcpy(p, &id , 8);                     p += 8;
        uint32_t dim = static_cast<uint32_t>(vec.size());
        std::memcpy(p, &dim, 4);                     p += 4;
        std::memcpy(p,  vec.data(), vec.size_bytes());

        publish(w, need);
        return true;
    }

    // Generic byte‑level producer required by IBuffer
    bool trySend(std::span<const std::byte> payload) override
    {
        const std::size_t need = sizeof(Header) + payload.size();
        std::byte* w = alloc(need);
        if (!w) return false;

        auto* h = reinterpret_cast<Header*>(w);
        h->bytes = static_cast<uint32_t>(payload.size());
        std::memcpy(w + sizeof(Header), payload.data(), payload.size());

        publish(w, need);
        return true;
    }

    // Single‑consumer coroutine returning raw bytes
    boost::asio::awaitable<std::span<const std::byte>>
    receive(boost::asio::io_context& io, int timeoutMs = 10) override
    {
        boost::asio::steady_timer t(io);
        for (;;)
        {
            auto s = tryPop();
            if (!s.empty()) co_return s;

            t.expires_after(std::chrono::milliseconds(timeoutMs));
            co_await t.async_wait(boost::asio::use_awaitable);
        }
    }

    // ───────── optional TCP ingestion ─────────
    void startSocket(const std::shared_ptr<boost::asio::io_context>& io)
    {
        io_ = io;                                           // keep context alive
        using tcp = boost::asio::ip::tcp;
        uint16_t port = parsePort(socket_addr);
        acc_ = std::make_unique<tcp::acceptor>(*io_, tcp::endpoint(tcp::v4(), port));

        boost::asio::co_spawn(*io_,
            [self = shared_from_this()]() -> boost::asio::awaitable<void> {
                for (;;) {
                    tcp::socket s = co_await self->acc_->async_accept(boost::asio::use_awaitable);
                    boost::asio::co_spawn(self->acc_->get_executor(),
                        self->reader(std::move(s)),
                        boost::asio::detached);
                }
            },
            boost::asio::detached);
    }

private:
    // ───────── ring-buffer internals ─────────
    struct Header { uint32_t bytes; };

    std::byte* alloc(std::size_t need)
    {
        uint32_t h = head.load(std::memory_order_acquire);
        uint32_t t = tail.load(std::memory_order_relaxed);

        if (t >= h) {
            if (t + need <= slabSize) return base_ + t;
            if (need < h)            return base_;
        } else {
            if (t + need < h)        return base_ + t;
        }
        return nullptr;
    }

    void publish(std::byte* w, std::size_t need)
    {
        uint32_t newTail = static_cast<uint32_t>(w - base_) +
                           static_cast<uint32_t>(need);
        if (newTail >= slabSize) newTail = 0;
        tail.store(newTail, std::memory_order_release);
    }

    std::span<const std::byte> tryPop() override
    {
        uint32_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire))
            return {};                              // empty

        auto*    hdr = reinterpret_cast<Header*>(base_ + h);
        std::byte* p = reinterpret_cast<std::byte*>(hdr + 1);

        uint32_t next = h + sizeof(Header) + hdr->bytes;
        if (next >= slabSize) next = 0;
        head.store(next, std::memory_order_release);

        // Return the entire payload as bytes (id + dim + vector data)
        return { p, static_cast<std::size_t>(hdr->bytes) };
    }

    inline boost::asio::awaitable<void> coro_yield()
    {
        // post an empty noop to the same executor and co_await it
        auto ex = co_await boost::asio::this_coro::executor;
        co_await boost::asio::post(ex, boost::asio::use_awaitable);
    }

    // ───────── socket reader → slab writer ─────────
    boost::asio::awaitable<void> reader(boost::asio::ip::tcp::socket sock)
    {
        using namespace boost::asio;
        std::vector<float> buf(vector_size);

        for (;;) {
            uint64_t id; uint32_t dim;
            co_await async_read(sock, buffer(&id ,8), use_awaitable);
            co_await async_read(sock, buffer(&dim,4), use_awaitable);

            buf.resize(dim);
            co_await async_read(sock, buffer(buf),    use_awaitable);

            std::vector<std::byte> bytes(sizeof(id) + sizeof(dim) + buf.size());
            std::memcpy(bytes.data(), &id, sizeof(id));
            std::memcpy(bytes.data() + sizeof(id), &dim, sizeof(dim));
            std::memcpy(bytes.data() + sizeof(id) + sizeof(dim),
                        reinterpret_cast<const std::byte*>(buf.data()),
                        buf.size());
            std::span<const std::byte> sp(bytes.data(), bytes.size());

            // write into slab as raw bytes (id+dim+vec)
            while (!trySend(sp))
                co_await coro_yield();
        }
    }

    static uint16_t parsePort(const std::string& addr) {
        auto pos = addr.rfind(':');
        return static_cast<uint16_t>(
            std::stoi(pos == std::string::npos ? addr : addr.substr(pos+1)));
    }

    // ───────── data members ─────────
    std::string  socket_addr;
    uint16_t     vector_size;
    std::size_t  slabSize;

    std::unique_ptr<ShaderBuffer> buf_;
    std::byte*  base_{};
    std::atomic<uint32_t> head{0}, tail{0};

    std::unique_ptr<boost::asio::ip::tcp::acceptor> acc_;

    std::shared_ptr<boost::asio::io_context> io_;
};

} // namespace psyne