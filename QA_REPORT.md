# QA Report

## Summary
This report reviews the latest changes in commit `c8435124c9811600264f6225a83fa822dca5e301` and highlights potential security, performance and reliability concerns.

## Issues

1. **Memory leak in `WebSocketChannel`**
   - `src/channel/websocket_channel.cpp` allocates message buffers with `new std::vector<uint8_t>` in `receive_message` but never frees them in `release_message`. The code explicitly leaks the memory (lines 301‑333) with a TODO to use a proper allocator.
   - Repeated messages can exhaust memory, impacting both reliability and security (denial of service).
   - **Suggested fix:** manage receive buffers using a memory pool or delete the allocated vector in `release_message`.

2. **Orphan declaration**
   - `process_messages()` is declared in `src/channel/websocket_channel.hpp` (line 59) but has no definition in the source tree.
   - This suggests dead or incomplete code which can confuse maintainers.

3. **Stubbed `stop()` in RDMA wrappers**
   - Wrapper classes inside `src/channel/rdma_channel.cpp` override `stop()` but leave the body empty (`/* impl_->stop(); */`). See lines around 344‑352 and 414‑415.
   - Calling `stop()` on these wrappers has no effect, which may prevent clean shutdowns.
   - **Suggested fix:** implement the stop logic or remove the override if not needed.

4. **AddressSanitizer flags always enabled**
   - `CMakeLists.txt` sets `-fsanitize=address` for all builds (line 6). This reduces performance and is typically unsuitable for production releases.
   - **Suggested fix:** enable sanitizers only for debug builds or behind an option.

5. **Missing newline at end of files**
   - Several newly added files (`README.md`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`) lack a trailing newline. While minor, it can cause tooling warnings.

## Conclusion
Addressing the memory management issues in `WebSocketChannel`, implementing proper stop logic for RDMA wrappers and adjusting build flags will improve the reliability and performance of the project.
