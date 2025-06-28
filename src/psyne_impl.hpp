#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

namespace psyne {

inline void psyne_banner() {
    std::cout << "  _____  ______ __    _ ____   _  ______  \n";
    std::cout << " |  .  ||   ___|\\ \\  //|    \\ | ||   ___| \n";
    std::cout << " |    _| `-.`-.  \\ \\// |     \\| ||   ___| \n";
    std::cout << " |___|  |______| /__/  |__/\\____||______| \n";
    std::cout << " Zero-copy RPC library optimized for AI/ML\n";
    std::cout << " Version " << psyne::version() << "\n";
}

// Version information function
inline const char* get_version() {
    return psyne::version();
}

// Print banner
inline void print_banner() {
    psyne_banner();
}

} // namespace psyne