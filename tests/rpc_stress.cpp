#include <boost/asio.hpp>
#include <boost/asio/steady_timer.hpp>
#include "blake3.h"
#include <random>
#include <iostream>
#include <thread>
#include "vectorslab.hpp"
#include "slab.hpp"

using boost::asio::ip::tcp;
using namespace boost::asio;
using namespace std::chrono_literals;

constexpr std::size_t kVecDim   = 1024;
constexpr std::size_t kMsgsPerProducer = 50'000;
constexpr int         kShardCount      = 4;
constexpr int         kProducerCount   = 8;

// simple key -> shard hash
inline std::uint32_t ring_hash(std::uint64_t key) { return key % kShardCount; }

// random float helper
std::vector<float> random_vec(std::size_t n) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> v(n);
    for (auto& f: v) f = dist(rng);
    return v;
}

// ------------ shard service ------------
struct Shard {
    std::shared_ptr<io_context> io = std::make_shared<io_context>();
    std::shared_ptr<psyne::VectorSlab> slab;
    std::thread                        thread;

    explicit Shard(std::string addr) {
        slab = psyne::BufferRegistry::instance()
                 .getOrCreate<psyne::VectorSlab>(addr, kVecDim);
        slab->startSocket(io);
        thread = std::thread([ctx = io] { ctx->run(); });
    }
};

// ------------ broker (dumb router) ------------
struct Broker {
    std::shared_ptr<io_context> io = std::make_shared<io_context>();
    tcp::acceptor               acc{*io, {tcp::v4(), 5010}};
    std::vector<std::shared_ptr<Shard>> shards;

    Broker() {
        // spawn shards on successive ports 6000+
        for (int i=0;i<kShardCount;++i) {
            shards.push_back(std::make_shared<Shard>("127.0.0.1:" + std::to_string(6000+i)));
        }

        co_spawn(*io, [this]() -> awaitable<void> {
            for (;;) {
                tcp::socket cli = co_await acc.async_accept(use_awaitable);
                co_spawn(acc.get_executor(),
                         handle_client(std::move(cli)),
                         detached);
            }
        }, detached);

        std::thread([ctx=io]{ ctx->run(); }).detach();
    }

    awaitable<void> handle_client(tcp::socket cli) {
        std::vector<float> buf(kVecDim);
        for (;;) {
            uint64_t id; uint32_t dim;
            if (co_await async_read(cli, buffer(&id,8), transfer_exactly(8), use_awaitable); false) {}
            if (co_await async_read(cli, buffer(&dim,4), transfer_exactly(4), use_awaitable); false) {}
            buf.resize(dim);
            co_await async_read(cli, buffer(buf), use_awaitable);

            // route to shard
            auto shardIdx = ring_hash(id);
            tcp::socket s(co_await this_coro::executor); // outgoing
            co_await s.async_connect(
                {tcp::v4(), static_cast<uint16_t>(6000+shardIdx)}, use_awaitable);

            co_await async_write(s, buffer(&id,8),  use_awaitable);
            co_await async_write(s, buffer(&dim,4), use_awaitable);
            co_await async_write(s, buffer(buf),    use_awaitable);

            // echo back minimal ack
            co_await async_write(cli, buffer(&id,8), use_awaitable);
        }
    }
};

// ------------ producer ------------
awaitable<void> producer_task(int idx) {
    tcp::socket s(co_await this_coro::executor);
    co_await s.async_connect({tcp::v4(), 5010}, use_awaitable);

    for (std::size_t i=0;i<kMsgsPerProducer;++i) {
        auto vec = random_vec(kVecDim);
        std::uint64_t id = (static_cast<uint64_t>(idx)<<32) | i;

        uint32_t dim = kVecDim;
        co_await async_write(s, buffer(&id,8),  use_awaitable);
        co_await async_write(s, buffer(&dim,4), use_awaitable);
        co_await async_write(s, buffer(vec),    use_awaitable);

        // await ack
        uint64_t echo;
        co_await async_read(s, buffer(&echo,8), use_awaitable);
        if (echo != id) std::cerr<<"bad echo\n";
    }
    std::cout << "[Producer "<<idx<<"] done\n";
    co_return;
}

// ------------ main test harness ------------
int main() {
    std::cout << "=== Psyne RPC stress ===\n";
    Broker broker;  // spawns shards internally

    io_context ctx;

    // spin up producers
    for (int p=0;p<kProducerCount;++p)
        co_spawn(ctx, producer_task(p), detached);

    auto start = std::chrono::steady_clock::now();
    ctx.run();
    auto end   = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(end-start).count();
    std::cout << "Sent " << kProducerCount * kMsgsPerProducer
              << " msgs in " << secs << " s → "
              << (kProducerCount*kMsgsPerProducer)/secs/1e6 << " M msg/s\n";
}