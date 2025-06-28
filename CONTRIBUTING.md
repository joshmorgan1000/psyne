# Contributing to Psyne

<div align="center">
  <img src="docs/assets/psyne_logo.png" alt="Psyne Logo" width="150"/>
</div>

Thank you for your interest in contributing to Psyne! This document provides guidelines and information for contributors.

## 🌟 How to Contribute

### 🐛 Reporting Bugs
- **Search existing issues** first to avoid duplicates
- **Use the bug report template** when creating new issues
- **Include system information**: OS, compiler version, dependencies
- **Provide minimal reproduction** steps and expected vs actual behavior
- **Add relevant logs** and stack traces

### 💡 Suggesting Features
- **Check the roadmap** to see if it's already planned
- **Use the feature request template** for new suggestions
- **Explain the use case** and why it benefits users
- **Consider backwards compatibility** and API impact

### 🔧 Code Contributions

#### Prerequisites
- **C++20** compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- **CMake 3.16+** for building
- **Git** for version control
- **Knowledge of** zero-copy messaging, high-performance computing, or related areas

#### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/joshmorgan1000/psyne.git
cd psyne

# Create a development branch
git checkout -b feature/your-feature-name

# Build with development options
mkdir build && cd build
cmake .. -DPSYNE_BUILD_TESTS=ON -DPSYNE_BUILD_EXAMPLES=ON
make -j$(nproc)

# Run tests to ensure everything works
ctest --output-on-failure
```

#### Code Style & Standards

##### **C++ Guidelines**
- **Follow C++20 standards** and best practices
- **Use RAII** for resource management
- **Zero-copy principles** - avoid unnecessary data copying
- **Exception safety** - provide strong exception guarantees
- **Const correctness** - use const wherever possible
- **Modern C++** features - prefer `auto`, range-based loops, smart pointers

##### **Naming Conventions**
```cpp
// Classes: PascalCase
class MessageRouter { };

// Functions and variables: snake_case
void send_message();
int buffer_size = 1024;

// Constants: UPPER_SNAKE_CASE
static constexpr size_t MAX_BUFFER_SIZE = 1024 * 1024;

// Template parameters: PascalCase
template<typename MessageType>
class Channel { };
```

##### **Code Formatting**
- **Use clang-format** with the provided `.clang-format` file
- **Format before committing**: `./scripts/format.sh`
- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)

##### **Documentation**
- **Doxygen comments** for all public APIs
- **File headers** with brief description and copyright
- **Examples** in documentation comments where helpful
```cpp
/**
 * @brief Creates a new high-performance channel
 * @param uri Channel URI specifying transport (e.g., "memory://buffer")
 * @param buffer_size Size of the ring buffer in bytes
 * @return Unique pointer to the created channel
 * @throws std::invalid_argument if URI format is invalid
 * 
 * @example
 * @code
 * auto channel = create_channel("tcp://localhost:8080", 1024*1024);
 * @endcode
 */
std::unique_ptr<Channel> create_channel(const std::string& uri, size_t buffer_size);
```

#### Testing Requirements
- **Unit tests** for all new functionality
- **Integration tests** for complex features
- **Performance tests** for optimization claims
- **Tests must pass** before submission

```cpp
// Example test structure
#include <psyne/psyne.hpp>
#include <cassert>

int main() {
    // Test setup
    auto channel = psyne::create_channel("memory://test", 1024*1024);
    
    // Test execution
    assert(channel != nullptr);
    
    // Test cleanup happens automatically with RAII
    return 0;
}
```

#### Performance Considerations
- **Profile before optimizing** - measure actual bottlenecks
- **Benchmark critical paths** - include performance tests
- **Memory efficiency** - minimize allocations and fragmentation
- **Cache awareness** - consider data layout and access patterns
- **Lock-free algorithms** - prefer when possible for high-performance paths

### 📚 Documentation Contributions
- **Fix typos and errors** in existing documentation
- **Add examples** to make concepts clearer
- **Improve navigation** and cross-references
- **Update outdated information** as the library evolves

#### Documentation Style
- **Clear and concise** language
- **Progressive complexity** - start simple, add details
- **Working examples** for all concepts
- **Cross-references** to related topics

### 🌍 Language Binding Contributions
We maintain bindings for multiple languages. When contributing:

- **Follow language idioms** - make it feel native
- **Comprehensive coverage** - don't leave gaps in the API
- **Good error handling** - convert C++ exceptions appropriately
- **Documentation** - language-specific docs and examples
- **Package management** - proper integration with language package managers

### 📝 Pull Request Process

#### Before Submitting
1. **Create an issue** to discuss major changes
2. **Fork the repository** and create a feature branch
3. **Write tests** for your changes
4. **Update documentation** as needed
5. **Run all tests** and ensure they pass
6. **Format your code** with clang-format
7. **Write clear commit messages** (see guidelines below)

#### Commit Message Guidelines
```
type(scope): brief description

Longer explanation if needed.

- List any breaking changes
- Reference related issues: Fixes #123
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
**Scopes**: `core`, `channel`, `message`, `bindings`, `docs`, `tests`

**Examples**:
```
feat(channel): add WebSocket transport support

Implements WebSocket channels for web-compatible real-time
communication with automatic reconnection and proper framing.

- Add WebSocketChannel class
- Integrate with channel factory
- Add comprehensive tests
- Update documentation

Fixes #456
```

#### Review Process
1. **Automated checks** must pass (CI, formatting, tests)
2. **Code review** by maintainers
3. **Performance review** for performance-critical changes
4. **Documentation review** for user-facing changes
5. **Final approval** and merge

### 🎯 Areas Where We Need Help

#### High Priority
- **Performance optimizations** - SIMD, cache optimization, memory layout
- **Platform support** - Windows, ARM, embedded systems
- **Documentation** - tutorials, examples, API reference improvements
- **Testing** - edge cases, stress testing, cross-platform validation

#### Medium Priority
- **Language bindings** - improvements to existing bindings
- **Transport protocols** - new transport types (QUIC, HTTP/3, etc.)
- **Monitoring & Observability** - metrics, tracing, debugging tools
- **Security** - audit, fuzzing, security hardening

#### Low Priority
- **Developer tools** - IDE integration, debugging extensions
- **Examples** - real-world use cases and patterns
- **Build system** - packaging, distribution improvements

### 🏆 Recognition

We value all contributions and recognize contributors:
- **Contributors** listed in README and CHANGELOG
- **Significant contributions** highlighted in release notes
- **Regular contributors** invited to join the maintainer team

### 📞 Getting Help

- **💬 Discussions**: [GitHub Discussions](https://github.com/yourusername/psyne/discussions) - general questions
- **🐛 Issues**: [GitHub Issues](https://github.com/yourusername/psyne/issues) - bugs and feature requests
- **📧 Email**: maintainers@psyne.io - sensitive issues or private communication
- **🗣️ Chat**: Join our Discord/Slack (link in README) - real-time discussion

### 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Be respectful, inclusive, and constructive in all interactions.

### ⚖️ License

By contributing to Psyne, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping make Psyne better!** 🚀