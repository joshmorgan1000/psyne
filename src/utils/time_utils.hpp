#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace psyne {

/**
 * @brief Returns the epoch timestamp in milliseconds.
 *
 * Static so it can be called by other classes that might need an epoch
 * timestamp for various reasons.
 *
 * @return uint64_t The epoch timestamp in milliseconds
 */
static uint64_t getCurrentTimestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

/**
 * @brief Returns the current time in milliseconds (alias for
 * getCurrentTimestamp)
 * @return uint64_t The current timestamp in milliseconds
 */
inline static uint64_t get_now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

/**
 * @brief Get formatted timestamp string for logging
 * @return std::string Formatted timestamp
 */
inline static std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

/**
 * @brief Format duration in human-readable format
 * @param duration_ms Duration in milliseconds
 * @return std::string Formatted duration string
 */
inline static std::string FormatDuration(uint64_t duration_ms) {
    if (duration_ms < 1000) {
        return std::to_string(duration_ms) + "ms";
    } else if (duration_ms < 60000) {
        return std::to_string(duration_ms / 1000.0) + "s";
    } else if (duration_ms < 3600000) {
        uint64_t minutes = duration_ms / 60000;
        uint64_t seconds = (duration_ms % 60000) / 1000;
        return std::to_string(minutes) + "m " + std::to_string(seconds) + "s";
    } else {
        uint64_t hours = duration_ms / 3600000;
        uint64_t minutes = (duration_ms % 3600000) / 60000;
        return std::to_string(hours) + "h " + std::to_string(minutes) + "m";
    }
}

/**
 * @brief High precision timestamp for performance measurements
 * @return uint64_t Nanoseconds since epoch
 */
inline static uint64_t getHighPrecisionTimestamp() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

/**
 * @brief Get microsecond precision timestamp
 * @return uint64_t Microseconds since epoch
 */
inline static uint64_t getMicrosecondTimestamp() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace psyne