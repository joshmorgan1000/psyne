#pragma once

/**
 * @file platform.hpp
 * @brief Platform-specific includes and compatibility layer
 */

// Detect platform
#if defined(_WIN32) || defined(_WIN64)
    #define PSYNE_PLATFORM_WINDOWS
#elif defined(__APPLE__) && defined(__MACH__)
    #define PSYNE_PLATFORM_MACOS
#elif defined(__linux__)
    #define PSYNE_PLATFORM_LINUX
#else
    #define PSYNE_PLATFORM_UNKNOWN
#endif

// Platform-specific includes
#ifdef PSYNE_PLATFORM_WINDOWS
    #include <windows.h>
    #include <process.h>
    
    // Windows doesn't have unistd.h
    #define sleep(x) Sleep((x) * 1000)
    
    // Aligned allocation for older MSVC
    #if _MSC_VER < 1914
        inline void* aligned_alloc(size_t alignment, size_t size) {
            return _aligned_malloc(size, alignment);
        }
        inline void aligned_free(void* ptr) {
            _aligned_free(ptr);
        }
        #define std::free(ptr) aligned_free(ptr)
    #endif
#else
    #include <unistd.h>
#endif

// Thread naming
inline void set_thread_name(const char* name) {
#ifdef PSYNE_PLATFORM_WINDOWS
    // Windows 10+ supports thread naming
    typedef HRESULT (WINAPI *SetThreadDescriptionFunc)(HANDLE, PCWSTR);
    static auto SetThreadDescription = (SetThreadDescriptionFunc)GetProcAddress(
        GetModuleHandle("kernel32.dll"), "SetThreadDescription");
    
    if (SetThreadDescription) {
        wchar_t wname[256];
        MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, 256);
        SetThreadDescription(GetCurrentThread(), wname);
    }
#elif defined(PSYNE_PLATFORM_MACOS)
    pthread_setname_np(name);
#elif defined(PSYNE_PLATFORM_LINUX)
    pthread_setname_np(pthread_self(), name);
#endif
}