#include "encrypted_channel.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>
#include <openssl/rand.h>

namespace psyne {
namespace encryption {

// Helper to get OpenSSL error string
static std::string GetOpenSSLError() {
    char buf[256];
    ERR_error_string_n(ERR_get_error(), buf, sizeof(buf));
    return std::string(buf);
}

// EncryptedChannel implementation
EncryptedChannel::EncryptedChannel(std::shared_ptr<Channel> underlying,
                                   const EncryptionConfig &config)
    : underlying_(underlying), config_(config), encrypt_ctx_(nullptr),
      decrypt_ctx_(nullptr) {
    if (!InitializeCrypto()) {
        throw std::runtime_error("Failed to initialize encryption");
    }
}

EncryptedChannel::~EncryptedChannel() {
    CleanupCrypto();
}

bool EncryptedChannel::InitializeCrypto() {
    // Create contexts
    encrypt_ctx_ = EVP_CIPHER_CTX_new();
    decrypt_ctx_ = EVP_CIPHER_CTX_new();

    if (!encrypt_ctx_ || !decrypt_ctx_) {
        return false;
    }

    // Select cipher
    const EVP_CIPHER *cipher = nullptr;
    switch (config_.algorithm) {
    case EncryptionConfig::Algorithm::AES_128_GCM:
        cipher = EVP_aes_128_gcm();
        break;
    case EncryptionConfig::Algorithm::AES_256_GCM:
        cipher = EVP_aes_256_gcm();
        break;
    case EncryptionConfig::Algorithm::ChaCha20_Poly1305:
        cipher = EVP_chacha20_poly1305();
        break;
    default:
        return true; // No encryption
    }

    // Use provided key or generate one
    if (config_.key.empty()) {
        try {
            int key_len = EVP_CIPHER_key_length(cipher);
            config_.key = crypto_utils::GenerateRandomBytes(key_len);
        } catch (const std::exception &e) {
            return false; // Return false on crypto initialization failure
        }
    }
    session_key_ = config_.key;

    // Use provided IV or generate one
    if (config_.iv.empty()) {
        try {
            int iv_len = EVP_CIPHER_iv_length(cipher);
            config_.iv = crypto_utils::GenerateRandomBytes(iv_len);
        } catch (const std::exception &e) {
            return false; // Return false on crypto initialization failure
        }
    }
    current_iv_ = config_.iv;

    return true;
}

void EncryptedChannel::CleanupCrypto() {
    if (encrypt_ctx_) {
        EVP_CIPHER_CTX_free(encrypt_ctx_);
        encrypt_ctx_ = nullptr;
    }
    if (decrypt_ctx_) {
        EVP_CIPHER_CTX_free(decrypt_ctx_);
        decrypt_ctx_ = nullptr;
    }
}

std::vector<uint8_t> EncryptedChannel::Encrypt(const void *data, size_t size) {
    if (config_.algorithm == EncryptionConfig::Algorithm::None) {
        return std::vector<uint8_t>(static_cast<const uint8_t *>(data),
                                    static_cast<const uint8_t *>(data) + size);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Select cipher
    const EVP_CIPHER *cipher = nullptr;
    switch (config_.algorithm) {
    case EncryptionConfig::Algorithm::AES_128_GCM:
        cipher = EVP_aes_128_gcm();
        break;
    case EncryptionConfig::Algorithm::AES_256_GCM:
        cipher = EVP_aes_256_gcm();
        break;
    case EncryptionConfig::Algorithm::ChaCha20_Poly1305:
        cipher = EVP_chacha20_poly1305();
        break;
    default:
        return {};
    }

    // Generate new IV if requested
    if (config_.generate_random_iv) {
        try {
            int iv_len = EVP_CIPHER_iv_length(cipher);
            current_iv_ = crypto_utils::GenerateRandomBytes(iv_len);
        } catch (const std::exception &e) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.encryption_errors++;
            return {}; // Return empty vector on IV generation failure
        }
    }

    // Initialize encryption
    if (!EVP_EncryptInit_ex(encrypt_ctx_, cipher, nullptr, session_key_.data(),
                            current_iv_.data())) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.encryption_errors++;
        return {};
    }

    // Allocate output buffer
    // Format: [IV_SIZE][IV][TAG_SIZE][CIPHERTEXT]
    int iv_len = EVP_CIPHER_iv_length(cipher);
    int tag_len = 16; // GCM/Poly1305 tag size
    std::vector<uint8_t> output(1 + iv_len + 1 + size +
                                EVP_CIPHER_block_size(cipher) + tag_len);

    // Write IV length and generate IV directly in output buffer
    output[0] = static_cast<uint8_t>(iv_len);
    // Zero-copy: generate IV directly in the output buffer
    if (!RAND_bytes(&output[1], iv_len)) {
        throw std::runtime_error("Failed to generate IV");
    }
    // Update current_iv_ from the generated IV
    for (int i = 0; i < iv_len; ++i) {
        current_iv_[i] = output[1 + i];
    }

    // Encrypt data
    int len;
    int ciphertext_len;
    uint8_t *ciphertext_start = &output[1 + iv_len + 1];

    if (!EVP_EncryptUpdate(encrypt_ctx_, ciphertext_start, &len,
                           static_cast<const uint8_t *>(data), size)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.encryption_errors++;
        return {};
    }
    ciphertext_len = len;

    if (!EVP_EncryptFinal_ex(encrypt_ctx_, ciphertext_start + len, &len)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.encryption_errors++;
        return {};
    }
    ciphertext_len += len;

    // Get tag for AEAD ciphers
    if (!EVP_CIPHER_CTX_ctrl(encrypt_ctx_, EVP_CTRL_AEAD_GET_TAG, tag_len,
                             ciphertext_start + ciphertext_len)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.encryption_errors++;
        return {};
    }

    // Write tag position
    output[1 + iv_len] = static_cast<uint8_t>(ciphertext_len);

    // Resize to actual size
    output.resize(1 + iv_len + 1 + ciphertext_len + tag_len);

    // Update stats
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration<double, std::micro>(end - start).count();

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.messages_encrypted++;
        stats_.bytes_encrypted += size;
        stats_.avg_encryption_time_us =
            (stats_.avg_encryption_time_us * (stats_.messages_encrypted - 1) +
             duration) /
            stats_.messages_encrypted;
    }

    return output;
}

std::vector<uint8_t> EncryptedChannel::Decrypt(const void *data, size_t size) {
    if (config_.algorithm == EncryptionConfig::Algorithm::None) {
        return std::vector<uint8_t>(static_cast<const uint8_t *>(data),
                                    static_cast<const uint8_t *>(data) + size);
    }

    auto start = std::chrono::high_resolution_clock::now();

    const uint8_t *ptr = static_cast<const uint8_t *>(data);

    // Read IV length and IV
    if (size < 1) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return {};
    }

    int iv_len = ptr[0];
    if (size < static_cast<size_t>(1 + iv_len + 1)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return {};
    }

    std::vector<uint8_t> iv(ptr + 1, ptr + 1 + iv_len);
    int ciphertext_end = ptr[1 + iv_len];

    // Select cipher
    const EVP_CIPHER *cipher = nullptr;
    switch (config_.algorithm) {
    case EncryptionConfig::Algorithm::AES_128_GCM:
        cipher = EVP_aes_128_gcm();
        break;
    case EncryptionConfig::Algorithm::AES_256_GCM:
        cipher = EVP_aes_256_gcm();
        break;
    case EncryptionConfig::Algorithm::ChaCha20_Poly1305:
        cipher = EVP_chacha20_poly1305();
        break;
    default:
        return {};
    }

    // Initialize decryption
    if (!EVP_DecryptInit_ex(decrypt_ctx_, cipher, nullptr, session_key_.data(),
                            iv.data())) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return {};
    }

    // Set tag for AEAD ciphers
    int tag_len = 16;
    const uint8_t *ciphertext_start = ptr + 1 + iv_len + 1;
    const uint8_t *tag_start = ciphertext_start + ciphertext_end;

    if (!EVP_CIPHER_CTX_ctrl(decrypt_ctx_, EVP_CTRL_AEAD_SET_TAG, tag_len,
                             const_cast<uint8_t *>(tag_start))) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return {};
    }

    // Decrypt
    std::vector<uint8_t> plaintext(ciphertext_end +
                                   EVP_CIPHER_block_size(cipher));
    int len;
    int plaintext_len;

    if (!EVP_DecryptUpdate(decrypt_ctx_, plaintext.data(), &len,
                           ciphertext_start, ciphertext_end)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return {};
    }
    plaintext_len = len;

    if (!EVP_DecryptFinal_ex(decrypt_ctx_, plaintext.data() + len, &len)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return {};
    }
    plaintext_len += len;

    plaintext.resize(plaintext_len);

    // Update stats
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration<double, std::micro>(end - start).count();

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.messages_decrypted++;
        stats_.bytes_decrypted += plaintext_len;
        stats_.avg_decryption_time_us =
            (stats_.avg_decryption_time_us * (stats_.messages_decrypted - 1) +
             duration) /
            stats_.messages_decrypted;
    }

    return plaintext;
}

size_t EncryptedChannel::DecryptInPlace(const void *encrypted_data,
                                        size_t encrypted_size,
                                        void *output_buffer,
                                        size_t output_capacity) {
    if (config_.algorithm == EncryptionConfig::Algorithm::None) {
        // No encryption - just copy data
        if (output_capacity < encrypted_size) {
            return 0;
        }
        // Zero-copy: direct pointer assignment would be better, but we need to
        // copy here since we're working with different buffers
        const uint8_t *src = static_cast<const uint8_t *>(encrypted_data);
        uint8_t *dst = static_cast<uint8_t *>(output_buffer);
        for (size_t i = 0; i < encrypted_size; ++i) {
            dst[i] = src[i];
        }
        return encrypted_size;
    }

    auto start = std::chrono::high_resolution_clock::now();
    const uint8_t *ptr = static_cast<const uint8_t *>(encrypted_data);

    // Read IV length and IV
    if (encrypted_size < 1) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return 0;
    }

    int iv_len = ptr[0];
    if (encrypted_size < static_cast<size_t>(1 + iv_len + 1)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return 0;
    }

    // Extract IV directly from input buffer (no copy)
    const uint8_t *iv = ptr + 1;
    int ciphertext_end = ptr[1 + iv_len];

    // Select cipher
    const EVP_CIPHER *cipher = nullptr;
    switch (config_.algorithm) {
    case EncryptionConfig::Algorithm::AES_128_GCM:
        cipher = EVP_aes_128_gcm();
        break;
    case EncryptionConfig::Algorithm::AES_256_GCM:
        cipher = EVP_aes_256_gcm();
        break;
    case EncryptionConfig::Algorithm::ChaCha20_Poly1305:
        cipher = EVP_chacha20_poly1305();
        break;
    default:
        return 0;
    }

    // Initialize decryption
    if (!EVP_DecryptInit_ex(decrypt_ctx_, cipher, nullptr, session_key_.data(),
                            iv)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return 0;
    }

    // Set tag for AEAD ciphers
    int tag_len = 16;
    const uint8_t *ciphertext_start = ptr + 1 + iv_len + 1;
    const uint8_t *tag_start = ciphertext_start + ciphertext_end;

    if (!EVP_CIPHER_CTX_ctrl(decrypt_ctx_, EVP_CTRL_AEAD_SET_TAG, tag_len,
                             const_cast<uint8_t *>(tag_start))) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return 0;
    }

    // Decrypt directly into output buffer
    uint8_t *out_ptr = static_cast<uint8_t *>(output_buffer);
    int len;
    int plaintext_len;

    if (!EVP_DecryptUpdate(decrypt_ctx_, out_ptr, &len, ciphertext_start,
                           ciphertext_end)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return 0;
    }
    plaintext_len = len;

    if (!EVP_DecryptFinal_ex(decrypt_ctx_, out_ptr + len, &len)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.decryption_errors++;
        return 0;
    }
    plaintext_len += len;

    // Update stats
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.messages_decrypted++;
        stats_.bytes_decrypted += plaintext_len;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        stats_.avg_decryption_time_us =
            (stats_.avg_decryption_time_us * (stats_.messages_decrypted - 1) +
             duration.count()) /
            stats_.messages_decrypted;
    }

    return plaintext_len;
}

void EncryptedChannel::stop() {
    underlying_->stop();
}

bool EncryptedChannel::is_stopped() const {
    return underlying_->is_stopped();
}

const std::string &EncryptedChannel::uri() const {
    static std::string uri = "encrypted:" + underlying_->uri();
    return uri;
}

ChannelType EncryptedChannel::type() const {
    return underlying_->type();
}

ChannelMode EncryptedChannel::mode() const {
    return underlying_->mode();
}

void *EncryptedChannel::receive_raw_message(size_t &size, uint32_t &type) {
    // Receive encrypted message
    size_t encrypted_size;
    uint32_t encrypted_type;
    void *encrypted_data =
        underlying_->receive_raw_message(encrypted_size, encrypted_type);

    if (!encrypted_data) {
        return nullptr;
    }

    // Calculate decrypted size and allocate buffer
    size_t decrypted_size = encrypted_size; // Upper bound
    void *result = new uint8_t[decrypted_size];

    // Decrypt directly into the allocated buffer
    size_t actual_size =
        DecryptInPlace(encrypted_data, encrypted_size, result, decrypted_size);
    underlying_->release_raw_message(encrypted_data);

    if (actual_size == 0) {
        delete[] static_cast<uint8_t *>(result);
        return nullptr;
    }

    size = actual_size;
    type = encrypted_type; // Preserve original type

    return result;
}

void EncryptedChannel::release_raw_message(void *handle) {
    delete[] static_cast<uint8_t *>(handle);
}

bool EncryptedChannel::has_metrics() const {
    return underlying_->has_metrics();
}

debug::ChannelMetrics EncryptedChannel::get_metrics() const {
    return underlying_->get_metrics();
}

void EncryptedChannel::reset_metrics() {
    underlying_->reset_metrics();
}

detail::ChannelImpl *EncryptedChannel::impl() {
    return underlying_->get_impl();
}

const detail::ChannelImpl *EncryptedChannel::impl() const {
    return underlying_->get_impl();
}

EncryptedChannel::EncryptionStats EncryptedChannel::GetEncryptionStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// Crypto utility functions
namespace crypto_utils {

std::vector<uint8_t> GenerateRandomBytes(size_t size) {
    std::vector<uint8_t> bytes(size);
    if (RAND_bytes(bytes.data(), size) != 1) {
        throw std::runtime_error("Failed to generate random bytes: " +
                                 GetOpenSSLError());
    }
    return bytes;
}

bool GenerateRSAKeyPair(int key_size, std::string &public_key_pem,
                        std::string &private_key_pem) {
    // Generate RSA key
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr);
    if (!ctx)
        return false;

    if (EVP_PKEY_keygen_init(ctx) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return false;
    }

    if (EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, key_size) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return false;
    }

    EVP_PKEY *pkey = nullptr;
    if (EVP_PKEY_keygen(ctx, &pkey) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return false;
    }

    EVP_PKEY_CTX_free(ctx);

    // Export public key
    BIO *pub_bio = BIO_new(BIO_s_mem());
    if (!PEM_write_bio_PUBKEY(pub_bio, pkey)) {
        BIO_free(pub_bio);
        EVP_PKEY_free(pkey);
        return false;
    }

    char *pub_key_data;
    long pub_key_len = BIO_get_mem_data(pub_bio, &pub_key_data);
    public_key_pem.assign(pub_key_data, pub_key_len);
    BIO_free(pub_bio);

    // Export private key
    BIO *priv_bio = BIO_new(BIO_s_mem());
    if (!PEM_write_bio_PrivateKey(priv_bio, pkey, nullptr, nullptr, 0, nullptr,
                                  nullptr)) {
        BIO_free(priv_bio);
        EVP_PKEY_free(pkey);
        return false;
    }

    char *priv_key_data;
    long priv_key_len = BIO_get_mem_data(priv_bio, &priv_key_data);
    private_key_pem.assign(priv_key_data, priv_key_len);
    BIO_free(priv_bio);

    EVP_PKEY_free(pkey);
    return true;
}

std::vector<uint8_t> DeriveKeyFromPassword(const std::string &password,
                                           const std::vector<uint8_t> &salt,
                                           size_t key_size, int iterations) {
    std::vector<uint8_t> key(key_size);

    if (PKCS5_PBKDF2_HMAC(password.c_str(), password.length(), salt.data(),
                          salt.size(), iterations, EVP_sha256(), key_size,
                          key.data()) != 1) {
        throw std::runtime_error("Failed to derive key: " + GetOpenSSLError());
    }

    return key;
}

std::vector<uint8_t> CalculateHMAC(const std::vector<uint8_t> &key,
                                   const void *data, size_t size) {
    std::vector<uint8_t> hmac(32); // SHA256 size
    unsigned int hmac_len = 32;

    if (!HMAC(EVP_sha256(), key.data(), key.size(),
              static_cast<const uint8_t *>(data), size, hmac.data(),
              &hmac_len)) {
        throw std::runtime_error("Failed to calculate HMAC: " +
                                 GetOpenSSLError());
    }

    hmac.resize(hmac_len);
    return hmac;
}

bool ConstantTimeCompare(const void *a, const void *b, size_t size) {
    const uint8_t *pa = static_cast<const uint8_t *>(a);
    const uint8_t *pb = static_cast<const uint8_t *>(b);
    uint8_t result = 0;

    for (size_t i = 0; i < size; ++i) {
        result |= pa[i] ^ pb[i];
    }

    return result == 0;
}

} // namespace crypto_utils

} // namespace encryption
} // namespace psyne