#include "vectorslab.hpp"
#include "slab.hpp"
#include <boost/asio.hpp>
#include <thread>
#include <random>
#include <iostream>
#include <numeric>

constexpr std::string_view kAddr = "127.0.0.1:5010";
constexpr uint16_t         kDim  = 4000;
constexpr int              kMsgs = 2;

template<class SyncWriteStream, class Span>
void writeAll(SyncWriteStream& s, Span bytes)
{
    namespace asio = boost::asio;              // avoid long names
    auto b = asio::buffer(bytes.data(), bytes.size()); // ← explicit
    asio::write(s, b);                         // no ambiguity
}

int main() {
    using boost::asio::ip::tcp;
    using boost::asio::co_spawn;
    using boost::asio::detached;
    using boost::asio::use_awaitable;
    auto ctx = std::make_shared<boost::asio::io_context>();
    std::cout << "  _____  ______ __    _ ____   _  ______  \n";
    std::cout << " |     ||   ___|\\ \\  //|    \\ | ||   ___| \n";
    std::cout << " |    _| `-.`-.  \\ \\// |     \\| ||   ___| \n";
    std::cout << " |___|  |______| /__/  |__/\\____||______| \n";
    std::cout << " Psyne - a high-performance C++ library for AI\n";

    //------------------------------------------------------------------
    // 1.  Create VectorSlab and start its socket listener
    //------------------------------------------------------------------
    auto slab = psyne::BufferRegistry::instance()
                  .getOrCreate<psyne::VectorSlab>(std::string(kAddr), kDim);
    slab->startSocket(ctx);                         // runs on ctx thread

    //------------------------------------------------------------------
    // 2.  Consumer coroutine (reads from slab, prints checksum)
    //------------------------------------------------------------------
    co_spawn(ctx->get_executor(), [&]() -> boost::asio::awaitable<void> {
        for (int i=0;i<kMsgs;++i) {
            auto raw = co_await slab->receive(*ctx/*for timer*/);
            // reinterpret the raw bytes as floats
            auto* fptr = reinterpret_cast<const float*>(raw.data());
            std::span<const float> fspan{fptr, raw.size() / sizeof(float)};
            float sum = std::accumulate(fspan.begin(), fspan.end(), 0.f);
            std::cout << "[Consumer] got #" << i << "  Σ=" << sum << '\n';
        }
        ctx->stop();
        co_return;
    }, detached);

    //------------------------------------------------------------------
    // 3.  Producer thread (connects, sends two vectors)
    //------------------------------------------------------------------
    std::thread prod([]{
        boost::asio::io_context cliIO;
        tcp::socket sock(cliIO);
        sock.connect({boost::asio::ip::make_address("127.0.0.1"), 5010});

        std::mt19937 rng(123);
        std::uniform_real_distribution<float> d(-1,1);
        std::vector<float> v(kDim);

        for (int i=0;i<kMsgs;++i) {
            std::generate(v.begin(), v.end(), [&]{ return d(rng); });

            uint64_t id  = 42+i;                   // 8 B
            uint32_t dim = kDim;                   // 4 B

            writeAll(sock, std::span{reinterpret_cast<std::byte*>(&id), 8});
            writeAll(sock, std::span{reinterpret_cast<std::byte*>(&dim),4});
            writeAll(sock, std::span{reinterpret_cast<std::byte*>(v.data()),
                                      v.size()*sizeof(float)});
            std::cout << "[Producer] sent #" << i << '\n';
        }
    });

    //------------------------------------------------------------------
    // 4.  Run until consumer finishes, join producer
    //------------------------------------------------------------------
    ctx->run();
    prod.join();
    return 0;
}