#include <iostream>
#include <psyne/psyne.hpp>
#include <string>
#include <vector>
#include <sstream>

namespace psyne {

static void psyne_banner() {
    std::cout << "  _____  ______ __    _ ____   _  ______  \n";
    std::cout << " |  .  ||   ___|\\ \\  //|    \\ | ||   ___| \n";
    std::cout << " |    _| `-.`-.  \\ \\// |     \\| ||   ___| \n";
    std::cout << " |___|  |______| /__/  |__/\\____||______| \n";
    std::cout << " Zero-copy RPC library optimized for AI/ML\n";
    std::cout << " Version " << version() << "\n";
}

// Version information function
const char* get_version() {
    return version();
}

// Print banner
void print_banner() {
    psyne_banner();
}

} // namespace psyne