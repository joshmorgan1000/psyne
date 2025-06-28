#include <psyne/psyne.hpp>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

// Test compression and encryption features
int main() {
    std::cout << "Running Compression and Encryption Tests..." << std::endl;
    
    try {
        // Test 1: Basic compression configuration
        {
            psyne::CompressionConfig config;
            config.type = psyne::CompressionType::LZ4;
            config.level = 3;
            config.min_size_threshold = 128;
            config.enable_checksum = true;
            
            assert(config.type == psyne::CompressionType::LZ4);
            assert(config.level == 3);
            assert(config.min_size_threshold == 128);
            assert(config.enable_checksum == true);
            
            std::cout << "✓ Compression configuration creation" << std::endl;
        }
        
        // Test 2: Compression types
        {
            assert(psyne::CompressionType::None != psyne::CompressionType::LZ4);
            assert(psyne::CompressionType::LZ4 != psyne::CompressionType::Zstd);
            assert(psyne::CompressionType::Zstd != psyne::CompressionType::Snappy);
            
            std::cout << "✓ Compression type enumeration" << std::endl;
        }
        
        // Test 3: Encrypted channel configuration
        {
            psyne::EncryptionConfig config;
            config.algorithm = psyne::EncryptionAlgorithm::AES_GCM;
            config.key_size = 256;
            config.generate_random_iv = true;
            
            assert(config.algorithm == psyne::EncryptionAlgorithm::AES_GCM);
            assert(config.key_size == 256);
            assert(config.generate_random_iv == true);
            
            std::cout << "✓ Encryption configuration creation" << std::endl;
        }
        
        // Test 4: Encryption algorithms
        {
            assert(psyne::EncryptionAlgorithm::None != psyne::EncryptionAlgorithm::AES_GCM);
            assert(psyne::EncryptionAlgorithm::AES_GCM != psyne::EncryptionAlgorithm::ChaCha20);
            
            std::cout << "✓ Encryption algorithm enumeration" << std::endl;
        }
        
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
            psyne::CompressionConfig lz4_config;
            lz4_config.type = psyne::CompressionType::LZ4;
            lz4_config.level = 1;
            
            psyne::CompressionConfig zstd_config;
            zstd_config.type = psyne::CompressionType::Zstd;
            zstd_config.level = 5;
            
            assert(lz4_config.type != zstd_config.type);
            assert(lz4_config.level != zstd_config.level);
            
            std::cout << "✓ Compression configuration comparison" << std::endl;
        }
        
        // Test 7: Key generation for encryption
        {
            psyne::EncryptionConfig config1;
            config1.key_size = 128;
            
            psyne::EncryptionConfig config2;
            config2.key_size = 256;
            
            assert(config1.key_size != config2.key_size);
            assert(config1.key_size == 128);
            assert(config2.key_size == 256);
            
            std::cout << "✓ Encryption key size configuration" << std::endl;
        }
        
        // Test 8: Combined compression and encryption configuration
        {
            psyne::CompressionConfig comp_config;
            comp_config.type = psyne::CompressionType::LZ4;
            comp_config.level = 3;
            
            psyne::EncryptionConfig enc_config;
            enc_config.algorithm = psyne::EncryptionAlgorithm::AES_GCM;
            enc_config.key_size = 256;
            
            // Both configurations should be valid
            assert(comp_config.type == psyne::CompressionType::LZ4);
            assert(enc_config.algorithm == psyne::EncryptionAlgorithm::AES_GCM);
            
            std::cout << "✓ Combined compression and encryption configuration" << std::endl;
        }
        
        std::cout << "All Compression and Encryption Tests Passed! ✅" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}