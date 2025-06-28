#pragma once

#include <psyne/psyne.hpp>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/aes.h>
#include <memory>
#include <vector>
#include <string>

namespace psyne {
namespace encryption {

/**
 * @brief Encryption configuration for secure channels
 */
struct EncryptionConfig {
    enum class Algorithm {
        None = 0,
        AES_128_GCM,
        AES_256_GCM,
        ChaCha20_Poly1305
    };
    
    Algorithm algorithm = Algorithm::AES_256_GCM;
    std::vector<uint8_t> key;      // Symmetric key
    std::vector<uint8_t> iv;       // Initialization vector
    bool generate_random_iv = true; // Generate new IV for each message
    
    // For key exchange
    std::string public_key_pem;    // RSA public key for key exchange
    std::string private_key_pem;   // RSA private key
};

/**
 * @brief Encrypted channel wrapper providing transparent encryption/decryption
 * 
 * Wraps any Psyne channel to add encryption support. Messages are encrypted
 * before sending and decrypted after receiving.
 */
class EncryptedChannel : public Channel {
public:
    /**
     * @brief Create an encrypted channel
     * @param underlying The underlying channel to wrap
     * @param config Encryption configuration
     */
    EncryptedChannel(std::shared_ptr<Channel> underlying,
                    const EncryptionConfig& config);
    ~EncryptedChannel() override;

    // Channel interface
    void stop() override;
    bool is_stopped() const override;
    const std::string& uri() const override;
    ChannelType type() const override;
    ChannelMode mode() const override;
    void* receive_raw_message(size_t& size, uint32_t& type) override;
    void release_raw_message(void* handle) override;
    bool has_metrics() const override;
    debug::ChannelMetrics get_metrics() const override;
    void reset_metrics() override;

    /**
     * @brief Perform key exchange with remote peer
     * @param timeout_ms Timeout in milliseconds
     * @return true if successful
     */
    bool PerformKeyExchange(int timeout_ms = 5000);

    /**
     * @brief Get encryption statistics
     */
    struct EncryptionStats {
        uint64_t messages_encrypted = 0;
        uint64_t messages_decrypted = 0;
        uint64_t bytes_encrypted = 0;
        uint64_t bytes_decrypted = 0;
        uint64_t encryption_errors = 0;
        uint64_t decryption_errors = 0;
        double avg_encryption_time_us = 0;
        double avg_decryption_time_us = 0;
    };
    EncryptionStats GetEncryptionStats() const;

protected:
    detail::ChannelImpl* impl() override;
    const detail::ChannelImpl* impl() const override;

private:
    class EncryptedMessage;
    
    std::shared_ptr<Channel> underlying_;
    EncryptionConfig config_;
    
    // OpenSSL contexts
    EVP_CIPHER_CTX* encrypt_ctx_;
    EVP_CIPHER_CTX* decrypt_ctx_;
    
    // Key material
    std::vector<uint8_t> session_key_;
    std::vector<uint8_t> current_iv_;
    
    // Statistics
    mutable EncryptionStats stats_;
    mutable std::mutex stats_mutex_;
    
    // Methods
    bool InitializeCrypto();
    void CleanupCrypto();
    std::vector<uint8_t> Encrypt(const void* data, size_t size);
    std::vector<uint8_t> Decrypt(const void* data, size_t size);
};

/**
 * @brief Secure key exchange protocol
 * 
 * Implements Diffie-Hellman key exchange with RSA signatures for
 * establishing shared session keys.
 */
class KeyExchange {
public:
    /**
     * @brief Initialize key exchange
     * @param channel Channel for communication
     * @param private_key_pem RSA private key in PEM format
     * @param public_key_pem RSA public key in PEM format
     */
    KeyExchange(std::shared_ptr<Channel> channel,
               const std::string& private_key_pem,
               const std::string& public_key_pem);
    ~KeyExchange();

    /**
     * @brief Initiate key exchange (client side)
     * @param peer_public_key_pem Peer's public key
     * @return Shared session key, or empty on failure
     */
    std::vector<uint8_t> InitiateExchange(const std::string& peer_public_key_pem);

    /**
     * @brief Respond to key exchange (server side)
     * @return Shared session key, or empty on failure
     */
    std::vector<uint8_t> RespondToExchange();

private:
    std::shared_ptr<Channel> channel_;
    EVP_PKEY* private_key_;
    EVP_PKEY* public_key_;
    
    // DH parameters
    DH* dh_params_;
    BIGNUM* dh_private_;
    BIGNUM* dh_public_;
};

/**
 * @brief Certificate-based authentication for channels
 * 
 * Provides mutual TLS-style authentication for Psyne channels.
 */
class ChannelAuthenticator {
public:
    struct AuthConfig {
        std::string ca_cert_path;        // CA certificate for verification
        std::string client_cert_path;    // Client certificate
        std::string client_key_path;     // Client private key
        bool verify_peer = true;         // Verify peer certificate
        bool require_client_cert = false; // Require client certificate (server)
    };

    /**
     * @brief Create an authenticator
     * @param config Authentication configuration
     */
    explicit ChannelAuthenticator(const AuthConfig& config);
    ~ChannelAuthenticator();

    /**
     * @brief Perform mutual authentication
     * @param channel Channel to authenticate over
     * @param is_server true for server, false for client
     * @param timeout_ms Timeout in milliseconds
     * @return true if authentication successful
     */
    bool Authenticate(std::shared_ptr<Channel> channel, 
                     bool is_server,
                     int timeout_ms = 5000);

    /**
     * @brief Get peer certificate info after authentication
     */
    struct PeerInfo {
        std::string subject;
        std::string issuer;
        std::string serial_number;
        std::string not_before;
        std::string not_after;
        std::vector<std::string> san_list; // Subject Alternative Names
    };
    PeerInfo GetPeerInfo() const;

private:
    AuthConfig config_;
    X509* ca_cert_;
    X509* client_cert_;
    EVP_PKEY* client_key_;
    X509* peer_cert_;
    
    bool LoadCertificates();
    bool VerifyPeerCertificate(X509* cert);
};

/**
 * @brief End-to-end encrypted channel with perfect forward secrecy
 * 
 * Combines encryption and authentication for secure communication.
 */
class SecureChannel {
public:
    struct SecureConfig {
        EncryptionConfig::Algorithm encryption = EncryptionConfig::Algorithm::AES_256_GCM;
        ChannelAuthenticator::AuthConfig auth;
        bool enable_compression = true;    // Compress before encryption
        bool enable_replay_protection = true; // Detect replay attacks
        int key_rotation_interval_s = 3600;   // Rotate keys every hour
    };

    /**
     * @brief Create a secure channel
     * @param uri Channel URI
     * @param config Security configuration
     * @param is_server true for server, false for client
     */
    SecureChannel(const std::string& uri,
                 const SecureConfig& config,
                 bool is_server);
    ~SecureChannel();

    /**
     * @brief Connect and perform handshake
     * @param timeout_ms Timeout for handshake
     * @return true if successful
     */
    bool Connect(int timeout_ms = 5000);

    /**
     * @brief Get the underlying secure channel
     */
    std::shared_ptr<Channel> GetChannel() const { return secure_channel_; }

    /**
     * @brief Get security status
     */
    struct SecurityStatus {
        bool authenticated = false;
        bool encrypted = false;
        std::string peer_identity;
        std::string cipher_suite;
        uint64_t messages_sent = 0;
        uint64_t messages_received = 0;
        std::chrono::system_clock::time_point key_established;
        std::chrono::system_clock::time_point next_key_rotation;
    };
    SecurityStatus GetStatus() const;

private:
    SecureConfig config_;
    bool is_server_;
    std::shared_ptr<Channel> raw_channel_;
    std::shared_ptr<EncryptedChannel> secure_channel_;
    std::unique_ptr<ChannelAuthenticator> authenticator_;
    std::unique_ptr<KeyExchange> key_exchange_;
    
    // Security state
    SecurityStatus status_;
    std::thread key_rotation_thread_;
    std::atomic<bool> running_{false};
    
    void KeyRotationLoop();
};

/**
 * @brief Utility functions for encryption
 */
namespace crypto_utils {

/**
 * @brief Generate cryptographically secure random bytes
 * @param size Number of bytes to generate
 * @return Random bytes
 */
std::vector<uint8_t> GenerateRandomBytes(size_t size);

/**
 * @brief Generate an RSA key pair
 * @param key_size Key size in bits (2048, 4096)
 * @param[out] public_key_pem Public key in PEM format
 * @param[out] private_key_pem Private key in PEM format
 * @return true if successful
 */
bool GenerateRSAKeyPair(int key_size,
                       std::string& public_key_pem,
                       std::string& private_key_pem);

/**
 * @brief Derive key from password using PBKDF2
 * @param password Password string
 * @param salt Salt bytes
 * @param key_size Desired key size in bytes
 * @param iterations PBKDF2 iterations (default: 100000)
 * @return Derived key
 */
std::vector<uint8_t> DeriveKeyFromPassword(const std::string& password,
                                          const std::vector<uint8_t>& salt,
                                          size_t key_size,
                                          int iterations = 100000);

/**
 * @brief Calculate message authentication code (HMAC-SHA256)
 * @param key HMAC key
 * @param data Data to authenticate
 * @param size Data size
 * @return HMAC bytes (32 bytes)
 */
std::vector<uint8_t> CalculateHMAC(const std::vector<uint8_t>& key,
                                  const void* data,
                                  size_t size);

/**
 * @brief Constant-time comparison for cryptographic values
 * @param a First value
 * @param b Second value
 * @param size Size to compare
 * @return true if equal
 */
bool ConstantTimeCompare(const void* a, const void* b, size_t size);

} // namespace crypto_utils

} // namespace encryption
} // namespace psyne