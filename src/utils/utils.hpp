#pragma once

#include <ctime>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <iomanip>
#include <shared_mutex>
#include <system_error>
#include <utility>
#include <functional>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <sys/ioctl.h>
#include <unistd.h>
#include <future>

namespace psyne {

struct GlobalContext;
inline GlobalContext& getGlobalContext();

/**
 * @brief Converts a byte vector to a hex string.
 */
inline std::string toHex(const std::vector<uint8_t>& data) {
    std::string hex;
    hex.reserve(data.size() * 2);
    for (uint8_t byte : data) {
        hex += "0123456789abcdef"[byte >> 4];
        hex += "0123456789abcdef"[byte & 0x0F];
    }
    return hex;
}

/**
 * @brief Converts a hex string to a byte buffer.
 */
inline std::vector<uint8_t> fromHex(const std::string& hex) {
    if (hex.size() % 2 != 0)
        throw std::invalid_argument("Hex string must have an even length");
    std::vector<uint8_t> data;
    data.reserve(hex.size() / 2);
    auto parse = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9')
            return static_cast<uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f')
            return static_cast<uint8_t>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F')
            return static_cast<uint8_t>(c - 'A' + 10);
        throw std::invalid_argument("Invalid hex character");
    };
    for (size_t i = 0; i < hex.size(); i += 2) {
        uint8_t hi = parse(hex[i]);
        uint8_t lo = parse(hex[i + 1]);
        data.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return data;
}

/**
 * @brief Returns the epoch timestamp.
 *
 * Static so it can be called by other classes that might need an epoch
 * timestamp for various reasons.
 *
 * @return uint64_t The epoch timestamp
 */
static uint64_t getCurrentTimestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static constexpr char b64_enc_table[64] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
    'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'};

static constexpr uint8_t b64_dec_table[256] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, // 0..15
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, // 16..31
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 62, 0xFF, 0xFF, 0xFF,
    63, // 32..47 ('+'=43 => 62, '/'=47 =>63)
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF,                                                   // '0'..'9' => 52..61
    0xFF, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, // 'A'..'O' => 0..14
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, // 'P'..'Z' => 15..25
    0xFF, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, // 'a'..'o' => 26..40
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, // 'p'..'z' => 41..51
    // 128..255 => invalid
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

/**
 * @brief This seemed like a decent place to stick this utility method. This
 * code runs faster than what is offered by the OpenSSL libraries.
 *
 * NOTE: NOT ACTUALLY ENCRYPTION! This is just here to view/log binary data if
 * necessary for debugging.
 *
 * @param data The data to encode
 * @return std::string The base64 encoded data
 */
inline static std::string base64Encode(const std::vector<uint8_t>& data_bytes) {
    const uint8_t* data = data_bytes.data();
    size_t length = data_bytes.size();
    if (length == 0) return "";
    size_t output_size = 4 * ((length + 2) / 3);
    std::string result;
    result.resize(output_size);
    size_t i = 0, j = 0;
    while (i + 2 < length) {
        uint32_t val = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        i += 3;
        result[j + 0] = b64_enc_table[(val >> 18) & 0x3F];
        result[j + 1] = b64_enc_table[(val >> 12) & 0x3F];
        result[j + 2] = b64_enc_table[(val >> 6) & 0x3F];
        result[j + 3] = b64_enc_table[val & 0x3F];
        j += 4;
    }
    if (i < length) {
        uint32_t val = data[i] << 16;
        if ((i + 1) < length) val |= (data[i + 1] << 8);
        result[j + 0] = b64_enc_table[(val >> 18) & 0x3F];
        result[j + 1] = b64_enc_table[(val >> 12) & 0x3F];
        if ((i + 1) < length) {
            result[j + 2] = b64_enc_table[(val >> 6) & 0x3F];
        } else {
            result[j + 2] = '=';
        }
        result[j + 3] = '=';
        j += 4;
    }
    return result;
}

/**
 * @brief Again, not actually encryption, this does not make your data safe. So
 * far only used for debugging in order to log/combare byte vector values.
 *
 * @param in The base64 encoded data
 * @return std::vector<uint8_t> The decoded data
 */
inline static std::vector<uint8_t> base64Decode(const std::string& in) {
    size_t length = in.size();
    while (length > 0 && in[length - 1] == '=') {
        length--;
    }
    std::vector<uint8_t> out;
    out.reserve((length * 3) / 4);
    size_t i = 0;
    while (i < in.size()) {
        uint8_t c0 = (i < in.size()) ? b64_dec_table[(unsigned char)in[i++]] : 0xFF;
        uint8_t c1 = (i < in.size()) ? b64_dec_table[(unsigned char)in[i++]] : 0xFF;
        uint8_t c2 = (i < in.size()) ? b64_dec_table[(unsigned char)in[i++]] : 0xFF;
        uint8_t c3 = (i < in.size()) ? b64_dec_table[(unsigned char)in[i++]] : 0xFF;
        if (c0 == 0xFF || c1 == 0xFF) break;
        uint32_t val = (c0 << 18) | (c1 << 12);
        out.push_back((val >> 16) & 0xFF);
        if (c2 != 0xFF) {
            val |= (c2 << 6);
            out.push_back((val >> 8) & 0xFF);
        }
        if (c3 != 0xFF) {
            val |= c3;
            out.push_back(val & 0xFF);
        }
    }
    return out;
}
enum LogLevel { TRACE = 0, DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4 };
static inline std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm* ptm = std::localtime(&time);
    std::ostringstream oss;
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", ptm);
    oss << buffer;
    return oss.str();
}
static inline std::string FormatDuration(uint64_t duration) {
    int minutes = duration / 60000;
    int seconds = (duration % 60000) / 1000;
    std::ostringstream oss;
    if (minutes > 0) {
        oss << std::setw(2) << minutes << "m ";
    }
    oss << std::setw(2) << std::setfill('0') << seconds << "s";
    return oss.str();
}
struct ProgressBar {
    std::string id;
    std::string thread_name;
    std::string header;
    std::string start_time_str;
    float progress;
    int width;
    unsigned long start_line;
    uint64_t start_time;
};
struct GlobalContext {
    std::atomic<uint64_t> stdout_current_line{0};
    std::unique_lock<std::mutex> stdout_thread_lock;
    static thread_local std::string thread_context;
    std::mutex stdout_mutex;
    std::condition_variable stdout_cv;
    std::atomic<uint64_t> next_ticket{0};
    std::atomic<uint64_t> currently_serving{0};
    std::atomic<uint64_t> threadCounter{0};
    std::atomic<bool> stopFlag{false};
    std::atomic<bool> standalone{true};
    std::shared_mutex progress_mutex;
    std::atomic<bool> banner_animation_done{true};
    std::shared_ptr<std::thread> crypto_hash_init_ptr;
    size_t num_cpu_cores = std::thread::hardware_concurrency();
    static constexpr uint8_t initialObfuscation[4] = {0x13, 0x6E, 0x68, 0x70};
    LogLevel global_log_level = LogLevel::DEBUG;
    std::unordered_map<std::string, ProgressBar> progress_bars;
};
inline GlobalContext& getGlobalContext() {
    static GlobalContext instance;
    return instance;
}
inline thread_local std::string GlobalContext::thread_context{};
static inline std::vector<uint8_t> string_to_vec(const std::string& str) {
    return std::vector<uint8_t>(str.begin(), str.end());
}
static inline std::string vec_to_string(const std::vector<uint8_t>& vec) {
    return std::string(vec.begin(), vec.end());
}
static inline void stdout_lock() {
    unsigned long ticket = getGlobalContext().next_ticket.fetch_add(1);
    getGlobalContext().stdout_thread_lock =
        std::unique_lock<std::mutex>(getGlobalContext().stdout_mutex);
    getGlobalContext().stdout_cv.wait(getGlobalContext().stdout_thread_lock, [&]() {
        return ticket == getGlobalContext().currently_serving.load();
    });
}
static inline void stdout_unlock() {
    getGlobalContext().currently_serving.fetch_add(1);
    getGlobalContext().stdout_cv.notify_all();
    getGlobalContext().stdout_thread_lock.unlock();
}
constexpr char base32_chars[32] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a',
                                   'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                                   'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v'};
constexpr uint8_t base32_decode_table[256] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  0,  0,  0,  0,  0,  0,  10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
inline static uint64_t get_now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
inline static std::string random_string(size_t length) {
    static const char alphanum[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::string result;
    result.reserve(length);
    result += alphanum[(rand() % (sizeof(alphanum) - 11)) + 10]; // Ensure first char is a letter
    for (size_t i = 1; i < length; ++i) {
        result += alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    return result;
}

template <typename Func, typename... Args>
inline std::shared_ptr<std::thread> execute_shared(const std::string& thread_name, Func&& func,
                                                   Args&&... args) {
    return std::make_shared<std::thread>([thread_name, func = std::forward<Func>(func),
                                          ... args = std::forward<Args>(args)]() mutable {
        std::invoke(func, std::forward<Args>(args)...);
    });
}

template <typename Real>
concept FloatingPoint = std::is_same_v<Real, float> || std::is_same_v<Real, double> ||
                        std::is_same_v<Real, long double>;

static inline void UpdateProgressBar(const ProgressBar& bar) {
    float progress = bar.progress;
    int filled_width = static_cast<int>(bar.width * progress);
    stdout_lock();
    int lines_down = getGlobalContext().stdout_current_line.load() - bar.start_line;
    std::ostringstream oss;
    for (int i = 0; i < lines_down; i++) {
        oss << "\033[1A";
    }
    oss << "\033[2K" << "\r";
    if (progress >= 1.0f) {
        uint64_t duration = getCurrentTimestamp() - bar.start_time;
        oss << bar.start_time_str << " [INFO ] [" << bar.thread_name << "] " << bar.header
            << " Completed in " << FormatDuration(duration);
        getGlobalContext().progress_bars.erase(bar.id);
    } else {
        uint64_t elapsed_time = getCurrentTimestamp() - bar.start_time;
        uint64_t estimated_time_remaining = (elapsed_time / progress) - elapsed_time;
        std::string time_remaining = FormatDuration(estimated_time_remaining);
        oss << bar.start_time_str << " [INFO ] [" << bar.thread_name << "] " << bar.header << " ["
            << std::string(filled_width, '=') << std::string(bar.width - filled_width, ' ') << "] "
            << std::fixed << std::setprecision(1) << (progress * 100) << "% est:" << time_remaining;
    }
    oss << "\r";
    for (int i = 0; i < lines_down; i++) {
        oss << "\033[1B";
    }
    std::cout << oss.str() << "\r";
    std::cout.flush();
    stdout_unlock();
}
static inline void RedrawAllProgressBars() {
    std::unique_lock lock(getGlobalContext().progress_mutex);
    for (const auto& [id, bar] : getGlobalContext().progress_bars) {
        UpdateProgressBar(bar);
    }
}
static inline std::function<void(float)>
log_progress(const std::string& header, const std::string& thread_name = getGlobalContext().thread_context) {
    std::string random_string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::string random_id;
    for (int i = 0; i < 32; i++) {
        random_id += random_string[rand() % random_string.length()];
    }
    std::thread log_progress_bar_thread([header, random_id, thread_name]() {
        getGlobalContext().thread_context = thread_name;
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        int term_width = w.ws_col > 0 ? w.ws_col : 80;
        int bar_width = term_width - 50 - header.length() - 38;
        std::string timestamp = GetTimestamp();
        std::ostringstream oss;
        oss << timestamp << " [INFO ] [" << getGlobalContext().thread_context << "] " << header << " ["
            << std::string(bar_width, ' ') << "] 0%\n";
        stdout_lock();
        std::cout << oss.str() << "\r";
        std::cout.flush();
        {
            std::unique_lock lock(getGlobalContext().progress_mutex);
            ProgressBar progress_bar_instance = {random_id,
                                                 getGlobalContext().thread_context,
                                                 header,
                                                 timestamp,
                                                 0.0f,
                                                 bar_width,
                                                 getGlobalContext().stdout_current_line.load(),
                                                 getCurrentTimestamp()};
            getGlobalContext().progress_bars[random_id] = progress_bar_instance;
        }
        getGlobalContext().stdout_current_line += 1;
        stdout_unlock();
    });
    log_progress_bar_thread.detach();
    return [random_id](float progress) mutable {
        std::unique_lock lock(getGlobalContext().progress_mutex);
        if (getGlobalContext().progress_bars.find(random_id) ==
            getGlobalContext().progress_bars.end()) {
            return;
        }
        getGlobalContext().progress_bars[random_id].progress = progress;
        UpdateProgressBar(getGlobalContext().progress_bars[random_id]);
    };
}
template <typename T>
concept Streamable = requires(std::ostream& os, T const& t) {
    { os << t } -> std::convertible_to<std::ostream&>;
};
template <typename T>
concept HasToJson = requires(T t) {
    { t.toJson() } -> std::convertible_to<std::string>;
};
template <typename T>
concept HasToString = requires(T t) {
    { t.toString() } -> std::convertible_to<std::string>;
};
template <typename T>
concept EssentiallyStreamable = Streamable<T> || HasToJson<T> || HasToString<T>;

template <typename T>
    requires Streamable<T>
inline static void concat_multi_parameter_inputs(std::stringstream& currentstream, T first) {
    if constexpr (Streamable<T>)
        currentstream << first;
    else if constexpr (HasToJson<T>)
        currentstream << first.toJson().dump();
    else if constexpr (HasToString<T>)
        currentstream << first.toString();
}

template <typename T, typename... Args>
    requires Streamable<T>
static void concat_multi_parameter_inputs(std::stringstream& currentstream, T first, Args... args) {
    if constexpr (Streamable<T>)
        currentstream << first;
    else if constexpr (HasToJson<T>)
        currentstream << first.toJson().dump();
    else if constexpr (HasToString<T>)
        currentstream << first.toString();
    if constexpr (sizeof...(args) > 0) {
        concat_multi_parameter_inputs(currentstream, args...);
    }
}

template <typename T, typename... Args>
    requires EssentiallyStreamable<T>
static void log_info(T first, Args... args) {
    std::stringstream oss;
    oss << GetTimestamp() << " [INFO ] [" << getGlobalContext().thread_context << "] ";
    concat_multi_parameter_inputs(oss, first, args...);
    oss << "\n";
    int lines = 0;
    int char_count = 0;
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int term_width = w.ws_col > 0 ? w.ws_col : 80;
    for (char c : oss.str()) {
        char_count++;
        if (char_count % term_width == 0)
            lines++;
        if (c == '\n') {
            lines++;
            char_count = 0;
        }
    }
    std::cout << oss.str();
    std::cout.flush();
    getGlobalContext().stdout_current_line += lines;
    RedrawAllProgressBars();
}
template <typename T, typename... Args>
    requires EssentiallyStreamable<T>
static void log_error(T first, Args... args) {
    std::stringstream oss;
    oss << "\033[1;31m";
    oss << GetTimestamp() << " [ERROR] [" << getGlobalContext().thread_context << "] ";
    concat_multi_parameter_inputs(oss, first, args...);
    oss << "\n";
    int lines = 0;
    int char_count = 0;
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int term_width = w.ws_col > 0 ? w.ws_col : 80;
    for (char c : oss.str()) {
        char_count++;
        if (char_count % term_width == 0)
            lines++;
        if (c == '\n') {
            lines++;
            char_count = 0;
        }
    }
    oss << "\033[0m";
    stdout_lock();
    std::cout << oss.str();
    std::cout.flush();
    getGlobalContext().stdout_current_line += lines;
    stdout_unlock();
    RedrawAllProgressBars();
}

template <typename T, typename... Args>
    requires EssentiallyStreamable<T>
static void log_warn(T first, Args... args) {
    if (getGlobalContext().global_log_level > LogLevel::WARN) {
        return;
    }
    std::stringstream oss;
    oss << "\033[1;33m";
    oss << GetTimestamp() << " [WARN ] [" << getGlobalContext().thread_context << "] ";
    concat_multi_parameter_inputs(oss, first, args...);
    oss << "\n";
    int lines = 0;
    int char_count = 0;
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int term_width = w.ws_col > 0 ? w.ws_col : 80;
    for (char c : oss.str()) {
        char_count++;
        if (char_count % term_width == 0)
            lines++;
        if (c == '\n') {
            lines++;
            char_count = 0;
        }
    }
    oss << "\033[0m";
    // stdout_lock();
    std::cout << oss.str();
    std::cout.flush();
    getGlobalContext().stdout_current_line += lines;
    // stdout_unlock();
    RedrawAllProgressBars();
}
template <typename T, typename... Args>
    requires EssentiallyStreamable<T>
static void log_debug(T first, Args... args) {
    if (getGlobalContext().global_log_level > LogLevel::DEBUG) {
        return;
    }
    std::stringstream oss;
    oss << "\033[1;34m";
    oss << GetTimestamp() << " [DEBUG] [" << getGlobalContext().thread_context << "] ";
    concat_multi_parameter_inputs(oss, first, args...);
    oss << "\n";
    int lines = 0;
    int char_count = 0;
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int term_width = w.ws_col > 0 ? w.ws_col : 80;
    for (char c : oss.str()) {
        char_count++;
        if (char_count % term_width == 0)
            lines++;
        if (c == '\n') {
            lines++;
            char_count = 0;
        }
    }
    oss << "\033[0m";
    stdout_lock();
    std::cout << oss.str();
    std::cout.flush();
    getGlobalContext().stdout_current_line += lines;
    stdout_unlock();
    RedrawAllProgressBars();
}

} // namespace psyne