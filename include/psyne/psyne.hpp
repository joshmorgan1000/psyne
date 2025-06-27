#pragma once
/**
 * ────────────────────────────────────────────────────────────────────────
 *  Psyne – Single‑header public interface
 *
 *  End‑users only need to:
 *
 *      #include <psyne/psyne.hpp>
 *
 *  and they immediately get:
 *    • The global Psyne I/O context
 *    • Simple helpers to `listen()` on a port or `connect()` to a peer
 *    • Access to Variant “lens” types for zero‑copy views
 *
 *  Internally we still rely on the heavier headers but the application
 *  sees just a clean, minimal surface.
 * ────────────────────────────────────────────────────────────────────────
 */

#include "arena/byteslab.hpp"
#include "arena/slab.hpp"
#include "arena/vectorslab.hpp"
#include "opcodes.hpp"
#include "utils/utils.hpp"
#include "variant.hpp"
#include <boost/asio.hpp>
#include <memory>
#include <string>

namespace psyne {

using matrix_message = psyne::VariantView<float, 2>;
using vector_message = psyne::VariantView<float, 1>;
using scalar_message = psyne::VariantView<float, 0>;
template <typename T, std::size_t R> using message = psyne::VariantView<T, R>;
using psyne::Float32ArrayView;
using psyne::Float32ScalarView;
using psyne::Int64ArrayView;
using psyne::Int64ScalarView;
using psyne::JsonView;
template <std::size_t R> using Float32Tensor = psyne::Float32Tensor<R>;

class Internal {
public:
  class Context {
  public:
    static Context &instance() {
      static Context inst;
      return inst;
    }
    boost::asio::io_context &io() { return *io_; }
    void run() { io_->run(); }
    void stop() { io_->stop(); }

    boost::asio::ip::tcp::socket connect(const std::string &addr) {
      using boost::asio::ip::tcp;
      auto pos = addr.rfind(':');
      auto host = addr.substr(0, pos);
      uint16_t port = static_cast<uint16_t>(std::stoi(addr.substr(pos + 1)));

      tcp::resolver resolver(*io_);
      tcp::socket sock(*io_);
      auto endp = resolver.resolve(host, std::to_string(port));
      boost::asio::connect(sock, endp);
      return sock;
    }

    template <typename SlabT, typename... Args>
    std::shared_ptr<SlabT> makeSlab(const std::string &addr, Args &&...args) {
      auto slab = BufferRegistry::instance().getOrCreate<SlabT>(
          addr, std::forward<Args>(args)...);
      slab->startSocket(io_);
      owned_.push_back(slab);
      return slab;
    }

  private:
    Context() : io_(std::make_shared<boost::asio::io_context>()) {}
    std::shared_ptr<boost::asio::io_context> io_;
    std::vector<std::shared_ptr<IBuffer>> owned_;
  };

  static Context &context() { return Context::instance(); }
};

class Socket {
public:
  explicit Socket(std::string addr, uint16_t vecDim = 1024,
                  std::size_t slabBytes = 1024 * 1024 * 64) // 64 MiB
      : socket_addr_(std::move(addr)) {
    // Ensure slab exists / listening
    Internal::context().makeSlab<VectorSlab>(socket_addr_, vecDim, slabBytes);
  }

  // convenience factories
  static Socket listen(std::string addr, uint16_t vecDim = 1024,
                       std::size_t slabBytes = 1ull << 30) {
    return Socket(std::move(addr), vecDim, slabBytes);
  }

  static Socket connect(std::string addr) { return Socket(std::move(addr)); }

  // --- Send helpers -------------------------------------------------
  bool send_scalar(float v) {
    VariantHdr hdr{static_cast<uint8_t>(Opcode::NONE), 0,
                   static_cast<uint8_t>(sizeof(float)), 0,
                   static_cast<uint32_t>(sizeof(float))};
    std::array<std::byte, sizeof(hdr) + sizeof(v)> pkt{};
    std::memcpy(pkt.data(), &hdr, sizeof(hdr));
    std::memcpy(pkt.data() + sizeof(hdr), &v, sizeof(v));
    return getSlab().trySend(
        std::span<const std::byte>(pkt.data(), pkt.size()));
  }

  bool send_vector(std::span<const float> vec) {
    VariantHdr hdr{static_cast<uint8_t>(Opcode::NONE), 1,
                   static_cast<uint8_t>(sizeof(float)), 0,
                   static_cast<uint32_t>(vec.size_bytes())};
    std::vector<std::byte> pkt(sizeof(hdr) + vec.size_bytes());
    std::memcpy(pkt.data(), &hdr, sizeof(hdr));
    std::memcpy(pkt.data() + sizeof(hdr), vec.data(), vec.size_bytes());
    return getSlab().trySend(
        std::span<const std::byte>(pkt.data(), pkt.size()));
  }

  bool send_bytes(std::span<const std::byte> payload) {
    VariantHdr hdr{static_cast<uint8_t>(Opcode::NONE), 0, 1, 0,
                   static_cast<uint32_t>(payload.size())};
    std::vector<std::byte> pkt(sizeof(hdr) + payload.size());
    std::memcpy(pkt.data(), &hdr, sizeof(hdr));
    std::memcpy(pkt.data() + sizeof(hdr), payload.data(), payload.size());
    return getSlab().trySend(
        std::span<const std::byte>(pkt.data(), pkt.size()));
  }

  bool send_put(std::span<const float> vec) {
    VariantHdr hdr{static_cast<uint8_t>(Opcode::PUT), 1,
                   static_cast<uint8_t>(sizeof(float)), 0,
                   static_cast<uint32_t>(vec.size_bytes())};
    std::vector<std::byte> pkt(sizeof(hdr) + vec.size_bytes());
    std::memcpy(pkt.data(), &hdr, sizeof(hdr));
    std::memcpy(pkt.data() + sizeof(hdr), vec.data(), vec.size_bytes());
    return getSlab().trySend(
        std::span<const std::byte>(pkt.data(), pkt.size()));
  }

  bool send_get(uint64_t key) {
    VariantHdr hdr{static_cast<uint8_t>(Opcode::GET), 0,
                   static_cast<uint8_t>(sizeof(uint64_t)), 0,
                   static_cast<uint32_t>(sizeof(uint64_t))};
    std::array<std::byte, sizeof(hdr) + sizeof(key)> pkt{};
    std::memcpy(pkt.data(), &hdr, sizeof(hdr));
    std::memcpy(pkt.data() + sizeof(hdr), &key, sizeof(key));
    return getSlab().trySend(
        std::span<const std::byte>(pkt.data(), pkt.size()));
  }

  // --- Awaitable receive (vector) ----------------------------------
  boost::asio::awaitable<std::span<const std::byte>> receive_bytes() {
    co_return co_await getSlab().receive(Internal::context().io());
  }

  boost::asio::awaitable<std::span<const float>> receive_vector() {
    auto bytes = co_await getVecSlab().receive(Internal::context().io());
    if (bytes.size() < sizeof(VariantHdr))
      co_return std::span<const float>{};
    auto hdr = reinterpret_cast<const VariantHdr *>(bytes.data());
    const float *data =
        reinterpret_cast<const float *>(bytes.data() + sizeof(VariantHdr));
    co_return std::span<const float>(data, hdr->byteLen / sizeof(float));
  }

private:
  VectorSlab &getVecSlab() {
    auto ptr = BufferRegistry::instance().at<VectorSlab>(socket_addr_);
    if (!ptr)
      throw std::runtime_error("Not a VectorSlab");
    return *ptr;
  }
  IBuffer &getSlab() {
    return *BufferRegistry::instance().at<IBuffer>(socket_addr_);
  }

  std::string socket_addr_;
};

// ───────────────────────────────────────────────────────────────
//  Free helpers that forward to the global Context
// ───────────────────────────────────────────────────────────────
inline boost::asio::io_context &io() { return psyne::Internal::context().io(); }
inline void run() { psyne::Internal::context().run(); }
inline void stop() { psyne::Internal::context().stop(); }

template <typename SlabT = psyne::VectorSlab, typename... Args>
std::shared_ptr<SlabT> listen(const std::string &addr, Args &&...args) {
  return psyne::Internal::context().makeSlab<SlabT>(
      addr, std::forward<Args>(args)...);
}

inline boost::asio::ip::tcp::socket connect(const std::string &addr) {
  return psyne::Internal::context().connect(addr);
}

inline void psyne_banner() {
  std::cout << "  _____  ______ __    _ ____   _  ______  \n";
  std::cout << " |  .  ||   ___|\\ \\  //|    \\ | ||   ___| \n";
  std::cout << " |    _| `-.`-.  \\ \\// |     \\| ||   ___| \n";
  std::cout << " |___|  |______| /__/  |__/\\____||______| \n";
  std::cout << " High-Performance Vector-Based RPC for AI\n";
}

} // namespace psyne