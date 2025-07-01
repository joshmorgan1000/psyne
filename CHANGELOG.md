# Changelog

All notable changes to Psyne will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2025-07-01

### Added
- Comprehensive backpressure system with multiple policies:
  - Drop: Discard messages when channel is full
  - Block: Block producer with configurable timeout
  - Retry: Exponential backoff with jitter
  - Callback: Application-defined handling
  - Adaptive: Automatic strategy switching based on load
- Coroutine support documentation and examples using boost::asio
- Windows CI support with vcpkg and boost
- Platform compatibility layer (`platform.hpp`)
- Boost-free test suite for basic CI validation (`test_simple_channel.cpp`)
- Windows-specific test (`test_windows.cpp`)

### Fixed
- Race condition in SPSC pattern with proper memory ordering
- MPMC pattern race condition using compare-and-exchange
- Buffer overflow protection in TCP substrate
- Missing bounds checks in all patterns
- CI builds on all platforms (Linux, macOS, Windows)

### Changed
- Enhanced TCP substrate with input validation against malicious payloads
- Improved error handling consistency across patterns
- Updated CI to install boost dependencies on all platforms
- Moved logger.hpp and threadpool.hpp to src/global/

### Security
- Added size validation in TCP substrate to prevent buffer overflows
- Added bounds checking in message allocation
- Protected against integer overflow in ring buffer calculations

## [2.0.0] - 2025-06-30

### Added
- Revolutionary v2.0 architecture with complete separation of concerns
- Physical substrates (InProcess, IPC, TCP) for memory and transport
- Abstract message types with zero-copy lens pattern  
- Pattern behaviors (SPSC, MPSC, SPMC, MPMC) for coordination
- Channel bridge orchestrating all components
- TDT compression protocol support
- Comprehensive benchmarking suite
- Full cross-platform support

### Performance
- Up to 22.5M msgs/s on AMD Ryzen (MPSC pattern)
- Up to 20.7M msgs/s on Apple M4 Pro Max
- 10 nanosecond cross-thread latency
- True zero-copy architecture

[2.0.1]: https://github.com/joshmorgan1000/psyne/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/joshmorgan1000/psyne/releases/tag/v2.0.0