#pragma once

#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace psyne {

// Thread-safe random number generation using thread-local generators
class ThreadSafeRandom {
private:
    // Thread-local random engine - each thread gets its own generator
    static thread_local std::mt19937 rng_;
    static thread_local bool initialized_;

    static void ensure_initialized() {
        if (!initialized_) {
            // Seed with a combination of thread ID and high-resolution time
            auto seed =
                std::hash<std::thread::id>{}(std::this_thread::get_id()) ^
                static_cast<uint64_t>(std::chrono::high_resolution_clock::now()
                                          .time_since_epoch()
                                          .count());
            rng_.seed(static_cast<std::mt19937::result_type>(seed));
            initialized_ = true;
        }
    }

public:
    // Thread-safe random integer generation
    static uint32_t random_uint32() {
        ensure_initialized();
        return rng_();
    }

    // Thread-safe random integer in range [0, max)
    static uint32_t random_uint32(uint32_t max) {
        if (max == 0)
            return 0;
        ensure_initialized();
        std::uniform_int_distribution<uint32_t> dist(0, max - 1);
        return dist(rng_);
    }

    // Thread-safe random byte generation
    static uint8_t random_byte() {
        ensure_initialized();
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        return dist(rng_);
    }
};

// Thread-local static member definitions
inline thread_local std::mt19937 ThreadSafeRandom::rng_;
inline thread_local bool ThreadSafeRandom::initialized_{false};

/**
 * Thread-safe replacement for the original random_string function
 * Ensures first character is a letter (a-z) for compatibility
 */
inline std::string random_string(size_t length) {
    static constexpr char alphanum[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    static constexpr size_t alphanum_size =
        sizeof(alphanum) - 1;                  // -1 for null terminator
    static constexpr size_t letter_start = 10; // Start of letters in alphanum

    if (length == 0)
        return "";

    std::string result;
    result.reserve(length);

    // Ensure first char is a letter (a-z) for compatibility with original
    // behavior
    result +=
        alphanum[ThreadSafeRandom::random_uint32(alphanum_size - letter_start) +
                 letter_start];

    // Fill remaining characters with any alphanumeric character
    for (size_t i = 1; i < length; ++i) {
        result += alphanum[ThreadSafeRandom::random_uint32(alphanum_size)];
    }

    return result;
}

/**
 * Generate a random string of specified length with alphanumeric characters
 * Used for ID generation in progress bars and other utilities
 */
inline std::string generate_random_id(size_t length = 32) {
    static constexpr char chars[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    static constexpr size_t chars_size = sizeof(chars) - 1;

    std::string result;
    result.reserve(length);

    for (size_t i = 0; i < length; ++i) {
        result += chars[ThreadSafeRandom::random_uint32(chars_size)];
    }

    return result;
}

} // namespace psyne