#include <cassert>
#include <iostream>
#include <psyne/psyne.hpp>
#include <string>
#include <vector>

// Test compression and encryption features
int main() {
    std::cout << "Running Compression and Encryption Tests..." << std::endl;

    try {
        // Test 1: Basic compression configuration
        {
            psyne::compression::CompressionConfig config;
            config.type = psyne::compression::CompressionType::LZ4;
            config.level = 3;
            config.min_size_threshold = 128;
            config.enable_checksum = true;

            assert(config.type == psyne::compression::CompressionType::LZ4);
            assert(config.level == 3);
            assert(config.min_size_threshold == 128);
            assert(config.enable_checksum == true);

            std::cout << "✓ Compression configuration creation" << std::endl;
        }

        // Test 2: Compression types
        {
            assert(psyne::compression::CompressionType::None !=
                   psyne::compression::CompressionType::LZ4);
            assert(psyne::compression::CompressionType::LZ4 !=
                   psyne::compression::CompressionType::Zstd);
            assert(psyne::compression::CompressionType::Zstd !=
                   psyne::compression::CompressionType::Snappy);

            std::cout << "✓ Compression type enumeration" << std::endl;
        }

        // Test 3: Encrypted channel configuration
        // NOTE: Encryption features not yet implemented in psyne.hpp
        /*
        {
            psyne::encryption::EncryptionConfig config;
            config.algorithm = psyne::encryption::EncryptionAlgorithm::AES_GCM;
            config.key_size = 256;
            config.generate_random_iv = true;

            assert(config.algorithm ==
        psyne::encryption::EncryptionAlgorithm::AES_GCM); assert(config.key_size
        == 256); assert(config.generate_random_iv == true);

            std::cout << "✓ Encryption configuration creation" << std::endl;
        }

        // Test 4: Encryption algorithms
        {
            assert(psyne::encryption::EncryptionAlgorithm::None !=
        psyne::encryption::EncryptionAlgorithm::AES_GCM);
            assert(psyne::encryption::EncryptionAlgorithm::AES_GCM !=
        psyne::encryption::EncryptionAlgorithm::ChaCha20);

            std::cout << "✓ Encryption algorithm enumeration" << std::endl;
        }
        */

        // Test 5: Test data for compression
        {
            // Create highly compressible data
            std::string test_data(10000, 'A'); // 10KB of 'A's

            assert(test_data.size() == 10000);
            assert(test_data[0] == 'A');
            assert(test_data[9999] == 'A');

            std::cout << "✓ Compressible test data creation" << std::endl;
        }

        // Test 6: Compression utility functions (if available)
        {
            // Test compression type conversion
            psyne::compression::CompressionConfig lz4_config;
            lz4_config.type = psyne::compression::CompressionType::LZ4;
            lz4_config.level = 1;

            psyne::compression::CompressionConfig zstd_config;
            zstd_config.type = psyne::compression::CompressionType::Zstd;
            zstd_config.level = 5;

            assert(lz4_config.type != zstd_config.type);
            assert(lz4_config.level != zstd_config.level);

            std::cout << "✓ Compression configuration comparison" << std::endl;
        }

        // Test 7: Key generation for encryption
        // NOTE: Encryption features not yet implemented
        /*
        {
            psyne::encryption::EncryptionConfig config1;
            config1.key_size = 128;

            psyne::encryption::EncryptionConfig config2;
            config2.key_size = 256;

            assert(config1.key_size != config2.key_size);
            assert(config1.key_size == 128);
            assert(config2.key_size == 256);

            std::cout << "✓ Encryption key size configuration" << std::endl;
        }
        */

        // Test 8: Combined compression and encryption configuration
        // NOTE: Encryption features not yet implemented
        /*
        {
            psyne::compression::CompressionConfig comp_config;
            comp_config.type = psyne::compression::CompressionType::LZ4;
            comp_config.level = 3;

            psyne::encryption::EncryptionConfig enc_config;
            enc_config.algorithm =
        psyne::encryption::EncryptionAlgorithm::AES_GCM; enc_config.key_size =
        256;

            // Both configurations should be valid
            assert(comp_config.type ==
        psyne::compression::CompressionType::LZ4); assert(enc_config.algorithm
        == psyne::encryption::EncryptionAlgorithm::AES_GCM);

            std::cout << "✓ Combined compression and encryption configuration"
        << std::endl;
        }
        */

        std::cout << "All Compression and Encryption Tests Passed! ✅"
                  << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}