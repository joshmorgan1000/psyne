#include <iostream>
#include <psyne/psyne.hpp>

using namespace psyne;

int main() {
    auto channel = create_channel("memory://test", 64 * 1024);

    std::cout << "Testing ByteVector..." << std::endl;

    ByteVector msg(*channel);
    std::cout << "Initial size: " << msg.size() << std::endl;
    std::cout << "Capacity: " << msg.capacity() << std::endl;

    try {
        msg.resize(50);
        std::cout << "Resized to 50, new size: " << msg.size() << std::endl;
    } catch (const std::exception &e) {
        std::cout << "Error resizing to 50: " << e.what() << std::endl;
    }

    try {
        msg.resize(100);
        std::cout << "Resized to 100, new size: " << msg.size() << std::endl;
    } catch (const std::exception &e) {
        std::cout << "Error resizing to 100: " << e.what() << std::endl;
    }

    try {
        msg.resize(200);
        std::cout << "Resized to 200, new size: " << msg.size() << std::endl;
    } catch (const std::exception &e) {
        std::cout << "Error resizing to 200: " << e.what() << std::endl;
    }

    return 0;
}