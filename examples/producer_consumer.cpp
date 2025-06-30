#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <psyne/psyne.hpp>
#include <thread>

using namespace psyne;

// Simulate sensor data production
void sensor_producer(Channel &channel, std::atomic<bool> &running) {
    std::cout << "[Producer] Starting sensor data generation...\n";

    const size_t num_sensors = 8;
    float time = 0.0f;

    while (running.load()) {
        // Create message directly in channel buffer
        FloatVector sensor_data(channel);

        if (!sensor_data.is_valid()) {
            std::cerr << "[Producer] Buffer full, dropping sample\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Resize to hold timestamp + sensor values
        sensor_data.resize(num_sensors + 1);

        // Write timestamp
        sensor_data[0] = time;

        // Simulate sensor readings (sine waves at different frequencies)
        for (size_t i = 0; i < num_sensors; ++i) {
            float frequency = 0.1f * (i + 1);
            sensor_data[i + 1] = std::sin(2 * M_PI * frequency * time);
        }

        // Send the data (zero-copy notification)
        sensor_data.send();

        // Advance time
        time += 0.01f;

        // Simulate 100Hz sampling rate
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "[Producer] Stopped\n";
}

/**
 * @brief Consumer thread function that processes sensor data
 * @param channel Reference to the channel for receiving data
 * @param running Atomic flag to control thread execution
 * 
 * Receives sensor data, computes running statistics (min/max/mean),
 * and displays results periodically.
 */
void sensor_consumer(Channel &channel, std::atomic<bool> &running) {
    std::cout << "[Consumer] Starting sensor data processing...\n";

    size_t samples_processed = 0;
    auto start_time = std::chrono::steady_clock::now();

    // Event-driven processing
    auto listener = channel.listen<FloatVector>([&](FloatVector &&data) {
        // Data is a zero-copy view into the channel buffer
        if (data.size() < 2)
            return;

        float timestamp = data[0];

        // Calculate RMS of sensor values
        float sum_squares = 0.0f;
        for (size_t i = 1; i < data.size(); ++i) {
            sum_squares += data[i] * data[i];
        }
        float rms = std::sqrt(sum_squares / (data.size() - 1));

        samples_processed++;

        // Print status every 100 samples
        if (samples_processed % 100 == 0) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(elapsed)
                    .count();
            float rate = samples_processed * 1000.0f / elapsed_ms;

            std::cout << "[Consumer] Processed " << samples_processed
                      << " samples | Time: " << std::fixed
                      << std::setprecision(2) << timestamp << "s | RMS: " << rms
                      << " | Rate: " << rate << " Hz\n";
        }
    });

    // Wait for stop signal
    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop listening and wait for thread
    channel.stop();
    listener->join();

    auto elapsed = std::chrono::steady_clock::now() - start_time;
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    std::cout << "[Consumer] Stopped. Processed " << samples_processed
              << " samples in " << elapsed_ms << "ms ("
              << (samples_processed * 1000.0f / elapsed_ms) << " Hz average)\n";
}

int main() {
    std::cout << "Psyne Producer/Consumer Example\n";
    std::cout << "===============================\n\n";

    // Create a high-performance channel
    auto channel = create_channel("memory://sensors", 10 * 1024 * 1024,
                                  ChannelMode::SPSC, ChannelType::SingleType);

    std::cout << "Channel created with 10MB buffer\n";
    std::cout << "Simulating 100Hz sensor data with 8 channels\n\n";

    std::atomic<bool> running{true};

    // Start producer and consumer threads
    std::thread producer(sensor_producer, std::ref(*channel),
                         std::ref(running));
    std::thread consumer(sensor_consumer, std::ref(*channel),
                         std::ref(running));

    // Run for 5 seconds
    std::cout << "Running for 5 seconds...\n\n";
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Stop threads
    std::cout << "\nStopping...\n";
    running = false;

    producer.join();
    consumer.join();

    std::cout << "\nExample completed successfully!\n";
    return 0;
}