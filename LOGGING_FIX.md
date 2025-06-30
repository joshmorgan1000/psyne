# Logging Violations Fix

## Problem
The codebase had 143 instances where `std::cout` and `std::cerr` were used directly instead of the required `log_info`, `log_warn`, `log_debug`, `log_trace`, and `log_error` functions.

## Root Cause
The logging utilities in `src/utils/logger.hpp` were not included in the main public API header (`include/psyne/psyne.hpp`), making them unavailable to examples and other code.

## Solution Implemented
1. Created a new simplified logging header at `include/psyne/logging.hpp` that:
   - Provides all required logging functions
   - Has no external dependencies
   - Uses simple C++ standard library features
   - Maintains color-coded output for different log levels
   - Provides thread context support

2. Added the logging header to the main public API (`psyne.hpp`)

## How to Use

### In Examples and Application Code
```cpp
#include <psyne/psyne.hpp>

int main() {
    psyne::log_info("Application started");
    psyne::log_warn("This is a warning");
    psyne::log_error("This is an error");
    psyne::log_debug("Debug info");  // Only shown if compiled with -DDEBUG
    psyne::log_trace("Trace info");  // Only shown if compiled with -DTRACE
    
    // Set thread context for logging
    psyne::thread_context = "worker-1";
    psyne::log_info("Message from worker thread");
}
```

### Migrating Existing Code
Replace all instances of:
- `std::cout << "message" << std::endl;` → `psyne::log_info("message");`
- `std::cerr << "error" << std::endl;` → `psyne::log_error("error");`
- Debug prints → `psyne::log_debug("debug info");`

## Next Steps
1. Update all examples to use the logging functions
2. Update any internal code using `std::cout`/`std::cerr`
3. Consider adding log level configuration at runtime

## Benefits
- Consistent logging format across the codebase
- Timestamped messages
- Thread context identification
- Color-coded output for better visibility
- Conditional compilation for debug/trace levels
- Zero overhead when debug/trace disabled