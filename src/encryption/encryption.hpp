#pragma once

#include "utils.hpp"
#include "BLAKE3/c/blake3.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <openssl/aes.h>
#include <openssl/buffer.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <utility>
#include "obfuscate.hpp"
#include "openssl_raii.hpp"

namespace manifoldb {

/**
 * @brief Custom deleter that cleanses the allocated memory before freeing it.
 *
 * OpenSSL provides @c OPENSSL_cleanse() which attempts to overwrite the
 * buffer so that sensitive material does not linger in RAM after deallocation.
 */
struct cleanse_delete_key {
    void operator()(uint8_t* p) const {
        if (p) {
            OPENSSL_cleanse(p, ENCRYPTED_KEY_SIZE);
            delete[] p;
        }
    }
};

/**
 * @brief RAII deleter for EVP_PKEY objects.
 *
 * Used together with @c std::unique_ptr to manage OpenSSL key structures in a
 * safe manner without leaking resources.
 */
struct cleanse_delete_evp {
    void operator()(EVP_PKEY* p) const {
        if (p) {
            EVP_PKEY_free(p);
        }
    }
};
using key_ptr = std::unique_ptr<uint8_t[], cleanse_delete_key>;
using evp_ptr = std::unique_ptr<EVP_PKEY, cleanse_delete_evp>;

/**
 * @class SecureKey
 * @brief Lightweight wrapper around raw key material.
 *
 * The class stores a private key in OpenSSL's secure memory arena and provides
 * helpers for deriving public keys and exporting the underlying bytes in a
 * safe way.  It intentionally avoids copying secret data unless explicitly
 * requested through @c get_key() or @c export_keypair().
 */
class SecureKey {
public:
    enum class KeyType { AES, ED25519, X25519 };

private:
    uint8_t* private_key;
    std::string public_key;
    KeyType type;
    void secure_alloc();
    void secure_free();
    void derive_public_key(KeyType keyType);
    int to_evp_type(KeyType ktype) const;

public:
    /** Default constructor generates a random AES key. */
    SecureKey();

    /** Construct an asymmetric key of the given type. */
    SecureKey(KeyType ktype);

    /**
     * @brief Import an existing AES key from a string.
     *
     * The provided string is hashed with SHA‑256 to obtain a fixed length
     * key suitable for use with OpenSSL primitives.
     */
    SecureKey(KeyType ktype, std::string key);

    /**
     * @brief Construct a key from raw public/private byte strings.
     */
    SecureKey(KeyType ktype, std::string pub, std::string priv);

    /** Convenience constructor for AES keys from an arbitrary string. */
    SecureKey(std::string key);

    /** Cleanses and frees the underlying key material. */
    ~SecureKey();

    /** Return the key type represented by this object. */
    KeyType key_type() const;

    /**
     * @brief Copy the private key into a newly allocated buffer.
     * @return Unique pointer owning the copied bytes.
     */
    key_ptr get_key() const;

    /**
     * @brief Expose the raw key bytes without copying.
     * The returned pointer is valid until the object is destroyed.
     */
    const uint8_t* raw_bytes() const;

    /** Obtain an EVP_PKEY wrapper for asymmetric keys. */
    evp_ptr get_pkey() const;

    /** Retrieve the associated public key as a string. */
    std::string get_pubkey() const;

    /** Export both public and private portions as strings. */
    std::pair<std::string, std::string> export_keypair() const;

    /** True if the key represents an AES secret. */
    bool is_aes() const;
};

/**
 * @class Encryption
 * @brief High level cryptographic helper.
 */
class Encryption {
private:
    std::unique_ptr<SecureKey> aes_key;
    std::unique_ptr<SecureKey> mac_key; ///< Key used for MAC/signature
    std::unique_ptr<SecureKey> sign_keypair;
    std::unique_ptr<SecureKey> enc_keypair;
    static std::function<SecureKey(void)> obfuscatedSecondSecretFunction;
    static SecureKey obfuscatedSecondSecret;

public:
    /** Construct a new instance with freshly generated keys. */
    Encryption();

    /** Copy constructor performs deep copy of key material. */
    Encryption(const Encryption& other);

    /** Move constructor transfers ownership of key material. */
    Encryption(Encryption&& other) noexcept;

    /** Copy assignment. */
    Encryption& operator=(const Encryption& other);

    /** Move assignment. */
    Encryption& operator=(Encryption&& other) noexcept;

    /** Optionally generate Ed25519/X25519 key pairs on construction. */
    Encryption(bool generate_keypair);

    /**
     * @brief Initialise from an AES key string and optionally generate
     *        asymmetric pairs.
     */
    Encryption(const std::string& key, bool generate_keypair = false);

    /**
     * @brief Construct from explicit AES and MAC keys.
     */
    Encryption(const std::string& aes_key, const std::string& mac_key, bool generate_keypair);

    /** Import an encoded key pair obtained externally. */
    void import_key(const std::string& key, bool is_private_key);

    /** Encrypt arbitrary data with AES‑GCM. */
    std::string encrypt(std::string data) const;

    /** Vector overload of encrypt(). */
    std::vector<uint8_t> encrypt(std::vector<uint8_t> data) const;

    /** Get the size of the encrypted data for a given input size. */
    size_t getEncryptedSize(size_t in_size) const;

    /** Encrypt data into a preallocated buffer. */
    void encryptInto(uint8_t* out, size_t out_size, const uint8_t* in, size_t in_size) const;

    /** Encrypt data for a given recipient using sealed boxes. */
    static std::string pairEncrypt(const std::string& data, const std::string& publicKey);

    /** Buffer-based variant of pairEncrypt(). */
    static void pairEncrypt(uint8_t*& targetbuf, size_t& targetsize, uint8_t* src,
                            const size_t& src_size, const std::string& publicKey);

    /** Encrypt data using the object's own public key. */
    std::string pairEncrypt(const std::string& data);

    /** Buffer-based variant of the above convenience method. */
    void pairEncrypt(uint8_t*& targetbuf, size_t& targetsize, uint8_t* src, const size_t& src_size);

    /** Decrypt AES‑GCM encrypted string data. */
    std::string decrypt(std::string layeredDataInput) const;

    /** Decrypt a vector of bytes. */
    std::vector<uint8_t> decrypt(std::vector<uint8_t> data) const;

    /** Decrypt sealed box data produced by pairEncrypt(). */
    std::string pairDecrypt(const std::string& data);

    /** Sign a message using Ed25519 and append the signature. */
    std::string signMessage(const std::string& message);

    /** Verify an attached Ed25519 signature. */
    static bool verifySignature(const std::string& message, std::string public_key);

    /** Concatenate the Ed25519 and X25519 public keys. */
    std::string getPublicKey();

    /** Concatenate the Ed25519 and X25519 private keys. */
    std::string getPrivateKey();

    /**
     * @brief Construct from 64 character hex encoded AES and MAC keys.
     */
    static Encryption fromHexStrings(const std::string& aes_hex, const std::string& mac_hex = "");

    EncryptedKey generate_hmac_key(const std::string& input) const;
};

/**
 * Allocate secure memory for the private key.  OpenSSL's secure heap is used
 * so that the data can be explicitly cleansed and is less likely to be swapped
 * to disk.
 */
void SecureKey::secure_alloc() {
    private_key = (uint8_t*)OPENSSL_secure_malloc(32);
    if (!private_key)
        throw std::runtime_error("Secure malloc failed");
    std::memset(private_key, 0, 32);
}
/**
 * Release the secure memory region holding the key and wipe its contents.
 */
void SecureKey::secure_free() {
    if (private_key) {
        OPENSSL_cleanse(private_key, 32);
        OPENSSL_secure_free(private_key);
        private_key = nullptr;
    }
}
/**
 * Derive the public key portion for the stored private key.
 *
 * Only applicable to asymmetric key types.  The result is cached in
 * @c public_key for later retrieval.
 */
void SecureKey::derive_public_key(KeyType keyType) {
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(to_evp_type(keyType), nullptr, private_key, 32);
    if (!pkey)
        throw std::runtime_error("EVP_PKEY_new_raw_private_key failed");
    public_key.resize(32);
    size_t len = 32;
    EVP_PKEY_get_raw_public_key(pkey, reinterpret_cast<unsigned char*>(public_key.data()), &len);
    EVP_PKEY_free(pkey);
}
/** Map our enum to OpenSSL constant values. */
int SecureKey::to_evp_type(KeyType ktype) const {
    switch (ktype) {
    case KeyType::ED25519:
        return EVP_PKEY_ED25519;
    case KeyType::X25519:
        return EVP_PKEY_X25519;
    default:
        throw std::runtime_error("Invalid asymmetric key type");
    }
}
/**
 * Default constructor initialises a random 256‑bit AES key.
 */
SecureKey::SecureKey() : type(KeyType::AES) {
    secure_alloc();
    RAND_bytes(private_key, 32);
}
/**
 * Generate a new asymmetric key of the requested type.
 */
SecureKey::SecureKey(KeyType keyType) : type(keyType) {
    if (keyType == KeyType::AES)
        throw std::invalid_argument("Use default constructor for AES");
    auto ctx = make_pkey_ctx_id(to_evp_type(keyType));
    EVP_PKEY* pkey = nullptr;
    if (!EVP_PKEY_keygen_init(ctx.get()) || !EVP_PKEY_keygen(ctx.get(), &pkey)) {
        throw std::runtime_error("Key generation failed");
    }
    secure_alloc();
    size_t len = 32;
    EVP_PKEY_get_raw_private_key(pkey, private_key, &len);
    public_key = std::string(32, '\0');
    EVP_PKEY_get_raw_public_key(pkey, reinterpret_cast<unsigned char*>(public_key.data()), &len);
    EVP_PKEY_free(pkey);
}
/**
 * Construct an AES key from arbitrary input by hashing it with SHA‑256.
 */
SecureKey::SecureKey(std::string key) : type(KeyType::AES) {
    secure_alloc();
    auto mdctx = make_md_ctx();
    if (EVP_DigestInit_ex(mdctx.get(), EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(mdctx.get(), key.data(), key.length()) != 1 ||
        EVP_DigestFinal_ex(mdctx.get(), private_key, nullptr) != 1) {
        throw std::runtime_error("SHA256 computation failed");
    }
}
/**
 * Construct a SecureKey from explicit public and private byte strings.
 * The caller is responsible for providing correctly sized material.
 */
SecureKey::SecureKey(KeyType keyType, std::string pub, std::string priv) : type(keyType) {
    if (pub.length() != 32)
        throw std::invalid_argument("Key sizes must be 32 std::string");
    if (priv.length() == 0) {
        public_key = pub;
        return;
    }
    if (priv.length() != 32)
        throw std::invalid_argument("Key sizes must be 32 std::string");
    secure_alloc();
    std::memcpy(private_key, priv.data(), 32);
    public_key = pub;
}
/** Ensure private material is wiped on destruction. */
/** Ensure private material is wiped on destruction. */
SecureKey::~SecureKey() { secure_free(); }

/** True if this SecureKey represents an AES secret. */
bool SecureKey::is_aes() const { return type == KeyType::AES; }

/** Retrieve the type of key stored here. */
SecureKey::KeyType SecureKey::key_type() const { return type; }

/**
 * Return a copy of the private key bytes.  Callers take ownership of the
 * returned buffer and must zero it out when done.
 */
key_ptr SecureKey::get_key() const {
    if (!is_aes())
        return key_ptr(nullptr);
    uint8_t* copy = new uint8_t[32];
    std::memcpy(copy, private_key, 32);
    return key_ptr(copy);
}
/** Convert to an EVP_PKEY for use with OpenSSL APIs. */
evp_ptr SecureKey::get_pkey() const {
    if (is_aes())
        return {nullptr};
    EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(to_evp_type(type), nullptr, private_key, 32);
    return evp_ptr(pkey);
}
/** Return the public key as a string. */
std::string SecureKey::get_pubkey() const { return public_key; }
/** Export both public and private portions as strings. */
std::pair<std::string, std::string> SecureKey::export_keypair() const {
    std::string priv(32, '\0');
    std::memcpy(priv.data(), private_key, 32);
    return {public_key, priv};
}
/**
 * The Encryption class bundles a symmetric key with a pair of asymmetric keys
 * and exposes a small toolkit for common cryptographic tasks.  The idea is to
 * centralise all OpenSSL interaction so that the rest of the code base can
 * operate on simple containers and strings.  Instances can be copied or moved
 * around safely; copying will duplicate the underlying key material while moves
 * transfer ownership without reallocations.
 */
/**
 * Construct a new Encryption helper with freshly generated keys.  The AES key
 * is randomised and asymmetric pairs are created as well.
 */
Encryption::Encryption() {
    aes_key = std::make_unique<SecureKey>();
    mac_key = std::make_unique<SecureKey>();
    sign_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::ED25519);
    enc_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::X25519);
}
/** Deep copy constructor. */
Encryption::Encryption(const Encryption& other) {
    aes_key = std::make_unique<SecureKey>(*other.aes_key);
    mac_key = std::make_unique<SecureKey>(*other.mac_key);
    sign_keypair = std::make_unique<SecureKey>(*other.sign_keypair);
    enc_keypair = std::make_unique<SecureKey>(*other.enc_keypair);
}
/** Move constructor transfers ownership of all key objects. */
Encryption::Encryption(Encryption&& other) noexcept {
    aes_key = std::move(other.aes_key);
    mac_key = std::move(other.mac_key);
    sign_keypair = std::move(other.sign_keypair);
    enc_keypair = std::move(other.enc_keypair);
}
/** Deep copy assignment. */
Encryption& Encryption::operator=(const Encryption& other) {
    if (this != &other) {
        aes_key = std::make_unique<SecureKey>(*other.aes_key);
        mac_key = std::make_unique<SecureKey>(*other.mac_key);
        sign_keypair = std::make_unique<SecureKey>(*other.sign_keypair);
        enc_keypair = std::make_unique<SecureKey>(*other.enc_keypair);
    }
    return *this;
}
/** Move assignment operator. */
Encryption& Encryption::operator=(Encryption&& other) noexcept {
    if (this != &other) {
        aes_key = std::move(other.aes_key);
        mac_key = std::move(other.mac_key);
        sign_keypair = std::move(other.sign_keypair);
        enc_keypair = std::move(other.enc_keypair);
    }
    return *this;
}
/**
 * Construct with a random AES key and optionally generate asymmetric pairs.
 */
Encryption::Encryption(bool generate_keypair) {
    aes_key = std::make_unique<SecureKey>();
    mac_key = std::make_unique<SecureKey>();
    if (generate_keypair) {
        sign_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::ED25519);
        enc_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::X25519);
    }
}
/**
 * Construct from an explicit AES key represented as a string. The key is
 * hashed to 256 bits and used for all symmetric encryption operations.
 */
Encryption::Encryption(const std::string& key, bool generate_keypair) {
    aes_key = std::make_unique<SecureKey>(key);
    mac_key = std::make_unique<SecureKey>(key);
    if (generate_keypair) {
        sign_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::ED25519);
        enc_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::X25519);
    }
}

Encryption::Encryption(const std::string& aes, const std::string& mac, bool generate_keypair) {
    aes_key = std::make_unique<SecureKey>(aes);
    mac_key = mac.empty() ? std::make_unique<SecureKey>(aes) : std::make_unique<SecureKey>(mac);
    if (generate_keypair) {
        sign_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::ED25519);
        enc_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::X25519);
    }
}
/**
 * Import an external key pair.  When @p is_private_key is true the string
 * contains concatenated private keys which are converted to the appropriate
 * OpenSSL representations.  Otherwise the string is interpreted as two public
 * keys.
 */
void Encryption::import_key(const std::string& key_pair, bool is_private_key) {
    if (is_private_key) {
        if (key_pair.length() != 64)
            throw std::invalid_argument("Key size must be 64 std::string");
        std::vector<uint8_t> sign_key(32);
        std::vector<uint8_t> enc_key(32);
        std::memcpy(sign_key.data(), key_pair.data(), 32);
        std::memcpy(enc_key.data(), key_pair.data() + 32, 32);
        std::vector<uint8_t> sign_pubkey(32);
        std::vector<uint8_t> enc_pubkey(32);
        EVP_PKEY* pkey = EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr, sign_key.data(),
                                                      sign_key.size());
        if (!pkey)
            throw std::runtime_error("Failed to create private key");
        size_t pubkey_len = 32;
        if (EVP_PKEY_get_raw_public_key(pkey, sign_pubkey.data(), &pubkey_len) != 1) {
            EVP_PKEY_free(pkey);
            throw std::runtime_error("Failed to get public key");
        }
        EVP_PKEY_free(pkey);
        sign_keypair = std::make_unique<SecureKey>(
            SecureKey::KeyType::ED25519, std::string(sign_pubkey.begin(), sign_pubkey.end()),
            std::string(sign_key.begin(), sign_key.end()));
        EVP_PKEY* pkey2 =
            EVP_PKEY_new_raw_private_key(EVP_PKEY_X25519, nullptr, enc_key.data(), enc_key.size());
        if (!pkey2)
            throw std::runtime_error("Failed to create private key");
        size_t pubkey_len2 = 32;
        if (EVP_PKEY_get_raw_public_key(pkey2, enc_pubkey.data(), &pubkey_len2) != 1) {
            EVP_PKEY_free(pkey2);
            throw std::runtime_error("Failed to get public key");
        }
        EVP_PKEY_free(pkey2);
        enc_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::X25519,
                                                  std::string(enc_pubkey.begin(), enc_pubkey.end()),
                                                  std::string(enc_key.begin(), enc_key.end()));
    } else {
        if (key_pair.length() != 64)
            throw std::invalid_argument("Key size must be 64 std::string");
        std::vector<uint8_t> sign_key(32);
        std::vector<uint8_t> enc_key(32);
        std::memcpy(sign_key.data(), key_pair.data(), 32);
        std::memcpy(enc_key.data(), key_pair.data() + 32, 32);
        // These are just public keys, no need to derive private keys
        sign_keypair = std::make_unique<SecureKey>(SecureKey::KeyType::ED25519,
                                                   std::string(sign_key.begin(), sign_key.end()),
                                                   std::string());
        enc_keypair = std::make_unique<SecureKey>(
            SecureKey::KeyType::X25519, std::string(enc_key.begin(), enc_key.end()), std::string());
    }
}
/**
 * Encrypt a string using AES‑256‑GCM.  A derived key is generated via HMAC
 * using an obfuscation secret so that the raw AES key is never used directly.
 */
std::string Encryption::encrypt(std::string data) const {
    if (!aes_key)
        throw std::runtime_error("AES key not initialized");
    auto aes = aes_key->get_key();
    if (!aes)
        throw std::runtime_error("AES key not initialized");
    auto obf = obfuscatedSecondSecret.get_key();
    if (!obf)
        throw std::runtime_error("Obfuscation key not initialized");
    uint8_t derived_key[32];
    unsigned int len = 0;
    // Derive a working key from the two secrets. This step intentionally keeps
    // the AES key itself out of subsequent API calls.
    if (!HMAC(EVP_sha256(), obf.get(), 32, aes.get(), 32, derived_key, &len) || len != 32)
        throw std::runtime_error("Key derivation failed");
    constexpr size_t IV_LEN = 12;  // 96‑bit nonce for GCM
    constexpr size_t TAG_LEN = 16; // 128‑bit auth‑tag
    // Generate a fresh IV for every encryption call.
    std::array<uint8_t, IV_LEN> iv;
    if (RAND_bytes(iv.data(), iv.size()) != 1)
        throw std::runtime_error("IV generation failed");
    auto ctx = make_cipher_ctx();
    EVP_EncryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, IV_LEN, nullptr);
    // Provide the derived key and IV to initialise the context for encryption.
    EVP_EncryptInit_ex(ctx.get(), nullptr, nullptr, derived_key, iv.data());
    std::string out;
    out.resize(IV_LEN + data.length() + TAG_LEN);
    std::memcpy(out.data(), iv.data(), IV_LEN);
    int outl = 0;
    // Encrypt the plaintext in a single call. For large inputs it may be
    // desirable to loop and process chunks instead.
    if (EVP_EncryptUpdate(ctx.get(), reinterpret_cast<unsigned char*>(out.data()) + IV_LEN, &outl,
                          reinterpret_cast<const unsigned char*>(data.data()),
                          static_cast<int>(data.length())) != 1) {

        throw std::runtime_error("EncryptUpdate failed");
    }
    if (EVP_EncryptFinal_ex(ctx.get(), nullptr, &outl) != 1) { // GCM: no extra bytes
        throw std::runtime_error("EncryptFinal failed");
    }
    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_GET_TAG, TAG_LEN,
                            out.data() + IV_LEN + data.length()) != 1) {
        throw std::runtime_error("GetTag failed");
    }
    return out;
}
/**
 * Encrypt a vector of bytes using AES‑256‑GCM.  This overload avoids
 * reallocations when working with binary data directly.
 */
std::vector<uint8_t> Encryption::encrypt(std::vector<uint8_t> data) const {
    if (!aes_key)
        throw std::runtime_error("AES key not initialized");
    auto aes = aes_key->get_key();
    if (!aes)
        throw std::runtime_error("AES key not initialized");
    auto obf = obfuscatedSecondSecret.get_key();
    if (!obf)
        throw std::runtime_error("Obfuscation key not initialized");
    uint8_t derived_key[32];
    unsigned int len = 0;
    // Recreate the same derived key used during encryption.
    if (!HMAC(EVP_sha256(), obf.get(), 32, aes.get(), 32, derived_key, &len) || len != 32)
        throw std::runtime_error("Key derivation failed");
    constexpr size_t IV_LEN = 12;  // 96‑bit nonce for GCM
    constexpr size_t TAG_LEN = 16; // 128‑bit auth‑tag
    std::array<uint8_t, IV_LEN> iv;
    if (RAND_bytes(iv.data(), iv.size()) != 1)
        throw std::runtime_error("IV generation failed");
    auto ctx = make_cipher_ctx();
    EVP_EncryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, IV_LEN, nullptr);
    EVP_EncryptInit_ex(ctx.get(), nullptr, nullptr, derived_key, iv.data());
    std::vector<uint8_t> out(IV_LEN + data.size() + TAG_LEN);
    std::memcpy(out.data(), iv.data(), IV_LEN);
    int outl = 0;
    if (EVP_EncryptUpdate(ctx.get(), out.data() + IV_LEN, &outl, data.data(),
                          static_cast<int>(data.size())) != 1) {
        throw std::runtime_error("EncryptUpdate failed");
    }
    if (EVP_EncryptFinal_ex(ctx.get(), nullptr, &outl) != 1) { // GCM: no extra bytes
        throw std::runtime_error("EncryptFinal failed");
    }
    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_GET_TAG, TAG_LEN,
                            out.data() + IV_LEN + data.size()) != 1) {
        throw std::runtime_error("GetTag failed");
    }
    return out;
}
/**
 * Get the size of the encrypted data for a given input size.
 * This includes the IV, ciphertext and authentication tag.
 */
size_t Encryption::getEncryptedSize(size_t in_size) const {
    constexpr size_t IV_LEN = 12;
    constexpr size_t TAG_LEN = 16;
    if (in_size == 0) return IV_LEN + TAG_LEN;
    return IV_LEN + in_size + TAG_LEN;
}
/**
 * Encrypt data into a preallocated buffer.
 */
void Encryption::encryptInto(uint8_t* out, size_t out_size,
                             const uint8_t* in, size_t in_size) const {
    if (!aes_key) throw std::runtime_error("AES key not initialized");
    auto aes = aes_key->get_key();
    if (!aes) throw std::runtime_error("AES key not initialized");
    auto obf = obfuscatedSecondSecret.get_key();
    if (!obf) throw std::runtime_error("Obfuscation key not initialized");
    uint8_t derived_key[32];
    unsigned int len = 0;
    // Derive the AES key from the obfuscation secret and the AES key.
    if (!HMAC(EVP_sha256(), obf.get(), 32, aes.get(), 32, derived_key, &len) || len != 32)
        throw std::runtime_error("Key derivation failed");
    constexpr size_t IV_LEN = 12;  // 96‑bit nonce for GCM
    constexpr size_t TAG_LEN = 16; // 128‑bit auth‑tag
    if (out_size < IV_LEN + in_size + TAG_LEN)
        throw std::runtime_error("Output buffer too small");
    std::array<uint8_t, IV_LEN> iv;
    if (RAND_bytes(iv.data(), iv.size()) != 1)
        throw std::runtime_error("IV generation failed");
    auto ctx = make_cipher_ctx();
    EVP_EncryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, IV_LEN, nullptr);
    EVP_EncryptInit_ex(ctx.get(), nullptr, nullptr, derived_key, iv.data());
    int outl = 0;
    std::memcpy(out, iv.data(), IV_LEN);
    if (EVP_EncryptUpdate(ctx.get(), out + IV_LEN, &outl, in, static_cast<int>(in_size)) != 1) {
        throw std::runtime_error("EncryptUpdate failed");
    }
    if (EVP_EncryptFinal_ex(ctx.get(), nullptr, &outl) != 1) {
        throw std::runtime_error("EncryptFinal failed");
    }
    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_GET_TAG, TAG_LEN,
                            out + IV_LEN + in_size) != 1) {
        throw std::runtime_error("GetTag failed");
    }
}

/**
 * Encrypt @p data for a recipient using ephemeral X25519 key exchange. The
 * returned blob contains the ephemeral public key, IV, authentication tag and
 * ciphertext concatenated together.
 */
std::string Encryption::pairEncrypt(const std::string& data, const std::string& recipient_pub) {
    std::string recipient_pubkey = recipient_pub;
    if (recipient_pubkey.length() != 32 && recipient_pubkey.length() != 64)
        throw std::invalid_argument("Invalid public key size");
    if (recipient_pub.length() == 64)
        recipient_pubkey = recipient_pubkey.substr(32, 32);
    auto eph_ctx = make_pkey_ctx_id(EVP_PKEY_X25519);
    EVP_PKEY* eph_key = nullptr;
    EVP_PKEY_keygen_init(eph_ctx.get());
    EVP_PKEY_keygen(eph_ctx.get(), &eph_key);
    EVP_PKEY* peer_pub = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_X25519, nullptr, reinterpret_cast<const unsigned char*>(recipient_pubkey.data()),
        32);
    auto dh_ctx = make_pkey_ctx(eph_key);
    EVP_PKEY_derive_init(dh_ctx.get());
    EVP_PKEY_derive_set_peer(dh_ctx.get(), peer_pub);
    std::string shared_secret(32, '\0');
    size_t len = 32;
    EVP_PKEY_derive(dh_ctx.get(), reinterpret_cast<unsigned char*>(shared_secret.data()), &len);
    EVP_PKEY_free(peer_pub);
    uint8_t aes_key[32];
    HMAC(EVP_sha256(), "pairEncrypt", 11,
         reinterpret_cast<const unsigned char*>(shared_secret.data()), shared_secret.length(),
         aes_key, nullptr);
    std::vector<unsigned char> iv(12);
    RAND_bytes(iv.data(), 12);
    std::string ciphertext(data.length(), '\0');
    std::string tag(16, '\0');
    auto ctx = make_cipher_ctx();
    int outlen;
    EVP_EncryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, iv.size(), nullptr);
    EVP_EncryptInit_ex(ctx.get(), nullptr, nullptr, aes_key, iv.data());
    EVP_EncryptUpdate(ctx.get(), reinterpret_cast<unsigned char*>(ciphertext.data()), &outlen,
                      reinterpret_cast<const unsigned char*>(data.data()), data.length());
    int tmplen;
    EVP_EncryptFinal_ex(ctx.get(), reinterpret_cast<unsigned char*>(ciphertext.data()) + outlen,
                        &tmplen);
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_GET_TAG, 16, tag.data());
    std::string eph_pub(32, '\0');
    size_t pub_len = 32;
    EVP_PKEY_get_raw_public_key(eph_key, reinterpret_cast<unsigned char*>(eph_pub.data()),
                                &pub_len);
    EVP_PKEY_free(eph_key);
    std::string output;
    output.insert(output.end(), eph_pub.begin(), eph_pub.end());
    output.insert(output.end(), iv.begin(), iv.end());
    output.insert(output.end(), tag.begin(), tag.end());
    output.insert(output.end(), ciphertext.begin(), ciphertext.end());
    return output;
}
/**
 * Buffer based sealed box encryption.  This variant avoids extra allocations
 * by writing the resulting blob into @p targetbuf.  The buffer is allocated by
 * the function and must be freed by the caller using @c std::free.
 */
void Encryption::pairEncrypt(uint8_t*& targetbuf, size_t& targetsize, uint8_t* src,
                             const size_t& src_size, const std::string& publicKey) {
    // Free target buffer if it was previously allocated
    if (targetbuf) {
        std::free(targetbuf);
        targetbuf = nullptr;
    }
    std::string recipient_pubkey = publicKey;
    if (recipient_pubkey.length() != 32 && recipient_pubkey.length() != 64)
        throw std::invalid_argument("Invalid public key size");
    if (recipient_pubkey.length() == 64)
        recipient_pubkey = recipient_pubkey.substr(32, 32);
    auto eph_ctx = make_pkey_ctx_id(EVP_PKEY_X25519);
    EVP_PKEY* eph_key = nullptr;
    EVP_PKEY_keygen_init(eph_ctx.get());
    EVP_PKEY_keygen(eph_ctx.get(), &eph_key);
    EVP_PKEY* peer_pub = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_X25519, nullptr, reinterpret_cast<const unsigned char*>(recipient_pubkey.data()),
        32);
    auto dh_ctx = make_pkey_ctx(eph_key);
    EVP_PKEY_derive_init(dh_ctx.get());
    EVP_PKEY_derive_set_peer(dh_ctx.get(), peer_pub);
    std::string shared_secret(32, '\0');
    size_t len = 32;
    EVP_PKEY_derive(dh_ctx.get(), reinterpret_cast<unsigned char*>(shared_secret.data()), &len);
    EVP_PKEY_free(peer_pub);
    uint8_t aes_key[32];
    HMAC(EVP_sha256(), "pairEncrypt", 11,
         reinterpret_cast<const unsigned char*>(shared_secret.data()), shared_secret.length(),
         aes_key, nullptr);
    std::vector<unsigned char> iv(12);
    RAND_bytes(iv.data(), 12);
    std::string ciphertext(src_size, '\0');
    std::string tag(16, '\0');
    auto ctx = make_cipher_ctx();
    int outlen;
    EVP_EncryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, iv.size(), nullptr);
    EVP_EncryptInit_ex(ctx.get(), nullptr, nullptr, aes_key, iv.data());
    EVP_EncryptUpdate(ctx.get(), reinterpret_cast<unsigned char*>(ciphertext.data()), &outlen, src,
                      src_size);
    int tmplen;
    EVP_EncryptFinal_ex(ctx.get(), reinterpret_cast<unsigned char*>(ciphertext.data()) + outlen,
                        &tmplen);
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_GET_TAG, 16, tag.data());
    std::string eph_pub(32, '\0');
    size_t pub_len = 32;
    EVP_PKEY_get_raw_public_key(eph_key, reinterpret_cast<unsigned char*>(eph_pub.data()),
                                &pub_len);
    EVP_PKEY_free(eph_key);
    std::string output;
    output.insert(output.end(), eph_pub.begin(), eph_pub.end());
    output.insert(output.end(), iv.begin(), iv.end());
    output.insert(output.end(), tag.begin(), tag.end());
    output.insert(output.end(), ciphertext.begin(), ciphertext.end());
    targetsize = output.length();
    targetbuf = static_cast<uint8_t*>(std::malloc(targetsize));
    if (!targetbuf)
        throw std::bad_alloc();
    std::memcpy(targetbuf, output.data(), targetsize);
}
/** Encrypt using this object's public key. */
std::string Encryption::pairEncrypt(const std::string& data) {
    if (!enc_keypair)
        throw std::runtime_error("Encryption keypair not initialized");
    auto enc = enc_keypair->get_pubkey();
    if (enc.length() != 64 && enc.length() != 32)
        throw std::runtime_error("Invalid keypair size");
    std::string enc32(32, '\0');
    if (enc.length() == 64)
        std::memcpy(enc32.data(), enc.data() + 32, 32);
    else
        std::memcpy(enc32.data(), enc.data(), 32);
    return pairEncrypt(data, enc32);
}
/** Buffer variant using the object's public key for convenience. */
void Encryption::pairEncrypt(uint8_t*& targetbuf, size_t& targetsize, uint8_t* src,
                             const size_t& src_size) {
    // Free target buffer if it was previously allocated
    if (targetbuf) {
        std::free(targetbuf);
        targetbuf = nullptr;
    }
    if (!enc_keypair)
        throw std::runtime_error("Encryption keypair not initialized");
    auto enc = enc_keypair->get_pubkey();
    if (enc.length() != 64 && enc.length() != 32)
        throw std::runtime_error("Invalid keypair size");
    std::string enc32(32, '\0');
    if (enc.length() == 64)
        std::memcpy(enc32.data(), enc.data() + 32, 32);
    else
        std::memcpy(enc32.data(), enc.data(), 32);
    pairEncrypt(targetbuf, targetsize, src, src_size, enc32);
}
/**
 * Decrypt a string produced by @ref encrypt().  The function validates the GCM
 * authentication tag before returning the plaintext.
 */
std::string Encryption::decrypt(std::string layeredDataInput) const {
    if (!aes_key)
        throw std::runtime_error("AES key not initialized");
    auto aes = aes_key->get_key();
    if (!aes)
        throw std::runtime_error("AES key not initialized");
    auto obf = obfuscatedSecondSecret.get_key();
    if (!obf)
        throw std::runtime_error("Obfuscation key not initialized");
    uint8_t derived_key[32];
    unsigned int len = 0;
    // Recreate the derived key as in the encryption path.
    if (!HMAC(EVP_sha256(), obf.get(), 32, aes.get(), 32, derived_key, &len) || len != 32)
        throw std::runtime_error("Key derivation failed");
    constexpr size_t IV_LEN = 12;
    constexpr size_t TAG_LEN = 16;
    // Basic sanity check on the ciphertext length.
    if (layeredDataInput.length() < IV_LEN + TAG_LEN)
        throw std::runtime_error("Ciphertext too short");
    const uint8_t* iv = reinterpret_cast<const uint8_t*>(layeredDataInput.data());
    const uint8_t* tag = iv + (layeredDataInput.length() - TAG_LEN);
    const uint8_t* cipher = iv + IV_LEN;
    size_t cipher_len = layeredDataInput.length() - IV_LEN - TAG_LEN;
    auto ctx = make_cipher_ctx();
    EVP_DecryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, IV_LEN, nullptr);
    EVP_DecryptInit_ex(ctx.get(), nullptr, nullptr, derived_key, iv);
    std::string plain(cipher_len, '\0');
    int outl = 0;
    if (EVP_DecryptUpdate(ctx.get(), reinterpret_cast<unsigned char*>(plain.data()), &outl, cipher,
                          static_cast<int>(cipher_len)) != 1) {
        throw std::runtime_error("DecryptUpdate failed");
    }
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_TAG, TAG_LEN, const_cast<uint8_t*>(tag));
    if (EVP_DecryptFinal_ex(ctx.get(), nullptr, &outl) != 1) {
        throw std::runtime_error("Authentication tag mismatch");
    }
    return plain;
}
/** Vector overload of decrypt(). */
std::vector<uint8_t> Encryption::decrypt(std::vector<uint8_t> layeredDataInput) const {
    if (!aes_key)
        throw std::runtime_error("AES key not initialized");
    auto aes = aes_key->get_key();
    if (!aes)
        throw std::runtime_error("AES key not initialized");
    auto obf = obfuscatedSecondSecret.get_key();
    if (!obf)
        throw std::runtime_error("Obfuscation key not initialized");
    uint8_t derived_key[32];
    unsigned int len = 0;
    if (!HMAC(EVP_sha256(), obf.get(), 32, aes.get(), 32, derived_key, &len) || len != 32)
        throw std::runtime_error("Key derivation failed");
    constexpr size_t IV_LEN = 12;
    constexpr size_t TAG_LEN = 16;
    // Validate the buffer length before reading fixed offsets.
    if (layeredDataInput.size() < IV_LEN + TAG_LEN)
        throw std::runtime_error("Ciphertext too short");
    const uint8_t* iv = layeredDataInput.data();
    const uint8_t* tag = iv + (layeredDataInput.size() - TAG_LEN);
    const uint8_t* cipher = iv + IV_LEN;
    size_t cipher_len = layeredDataInput.size() - IV_LEN - TAG_LEN;
    auto ctx = make_cipher_ctx();
    EVP_DecryptInit_ex(ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, IV_LEN, nullptr);
    EVP_DecryptInit_ex(ctx.get(), nullptr, nullptr, derived_key, iv);
    std::vector<uint8_t> plain(cipher_len);
    int outl = 0;
    if (EVP_DecryptUpdate(ctx.get(), plain.data(), &outl, cipher, static_cast<int>(cipher_len)) !=
        1) {
        throw std::runtime_error("DecryptUpdate failed");
    }
    EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_TAG, TAG_LEN, const_cast<uint8_t*>(tag));
    if (EVP_DecryptFinal_ex(ctx.get(), nullptr, &outl) != 1) {
        throw std::runtime_error("Authentication tag mismatch");
    }
    plain.resize(outl);
    return plain;
}

EncryptedKey Encryption::generate_hmac_key(const std::string& input_data) const {
    if (!aes_key) throw std::runtime_error("AES key not initialized");
    auto aes = aes_key->get_key();
    if (!aes) throw std::runtime_error("AES key not initialized");
    auto obf = obfuscatedSecondSecret.get_key();
    if (!obf) throw std::runtime_error("Obfuscation key not initialized");
    uint8_t derived_key[32];
    unsigned int len = 0;
    // Derive a key from the obfuscation secret and the AES key.
    if (!HMAC(EVP_sha256(), obf.get(), 32, aes.get(), 32, derived_key, &len) || len != 32)
        throw std::runtime_error("Key derivation failed");
    // Use BLAKE3 to generate an HMAC key that is ENCRYPTED_KEY_SIZE bytes long.
    std::string key_str(reinterpret_cast<const char*>(derived_key), 32);
    return hmac_blake3(key_str, input_data);
}

/**
 * Decrypt data produced by pairEncrypt().  Performs ephemeral key agreement and
 * verifies the embedded authentication tag before returning the plaintext.
 */
std::string Encryption::pairDecrypt(const std::string& encrypted) {
    if (!enc_keypair)
        throw std::runtime_error("Encryption keypair not initialized");
    if (encrypted.length() < (32 + 12 + 16))
        throw std::runtime_error("Invalid encrypted message length");
    const size_t offset_pub = 0;
    const size_t offset_iv = 32;
    const size_t offset_tag = offset_iv + 12;
    const size_t offset_ciphertext = offset_tag + 16;
    std::string ephemeral_pub(encrypted.begin(), encrypted.begin() + 32);
    std::string iv(encrypted.begin() + offset_iv, encrypted.begin() + offset_iv + 12);
    std::string tag(encrypted.begin() + offset_tag, encrypted.begin() + offset_tag + 16);
    std::string ciphertext(encrypted.begin() + offset_ciphertext, encrypted.end());
    EVP_PKEY* peer_pub = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_X25519, nullptr, reinterpret_cast<const unsigned char*>(ephemeral_pub.data()), 32);
    if (!peer_pub)
        throw std::runtime_error("Invalid ephemeral public key");
    auto my_pkey = enc_keypair->get_pkey();
    auto ctx = make_pkey_ctx(my_pkey.get());
    EVP_PKEY_derive_init(ctx.get());
    EVP_PKEY_derive_set_peer(ctx.get(), peer_pub);
    std::string shared_secret(32, '\0');
    size_t len = shared_secret.length();
    EVP_PKEY_derive(ctx.get(), reinterpret_cast<unsigned char*>(shared_secret.data()), &len);
    EVP_PKEY_free(peer_pub);
    uint8_t aes_key[32];
    HMAC(EVP_sha256(), "pairEncrypt", 11,
         reinterpret_cast<const unsigned char*>(shared_secret.data()), shared_secret.length(),
         aes_key, nullptr);
    std::string plaintext(ciphertext.length(), '\0');
    auto decrypt_ctx = make_cipher_ctx();
    int outlen;
    EVP_DecryptInit_ex(decrypt_ctx.get(), EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(decrypt_ctx.get(), EVP_CTRL_GCM_SET_IVLEN, iv.size(), nullptr);
    EVP_DecryptInit_ex(decrypt_ctx.get(), nullptr, nullptr, aes_key,
                       reinterpret_cast<const unsigned char*>(iv.data()));
    EVP_DecryptUpdate(decrypt_ctx.get(), reinterpret_cast<unsigned char*>(plaintext.data()),
                      &outlen, reinterpret_cast<const unsigned char*>(ciphertext.data()),
                      ciphertext.length());
    EVP_CIPHER_CTX_ctrl(decrypt_ctx.get(), EVP_CTRL_GCM_SET_TAG, tag.length(),
                        reinterpret_cast<void*>(tag.data()));
    int final = EVP_DecryptFinal_ex(
        decrypt_ctx.get(), reinterpret_cast<unsigned char*>(plaintext.data()) + outlen, &outlen);
    if (final <= 0)
        throw std::runtime_error("Decryption failed: authentication tag mismatch");
    return plaintext;
}
/**
 * Sign @p message using Ed25519 and append the signature to the returned string.
 */
std::string Encryption::signMessage(const std::string& message) {
    if (!sign_keypair || sign_keypair->key_type() != SecureKey::KeyType::ED25519) {
        throw std::runtime_error("SecureKey must be Ed25519 and non-null");
    }
    auto pkey = sign_keypair->get_pkey();
    auto ctx = make_md_ctx();
    size_t siglen = 64;
    std::string signature;
    signature.resize(siglen);
    if (!EVP_DigestSignInit(ctx.get(), nullptr, nullptr, nullptr, pkey.get())) {
        throw std::runtime_error("DigestSignInit failed");
    }
    if (!EVP_DigestSign(ctx.get(), reinterpret_cast<unsigned char*>(signature.data()), &siglen,
                        reinterpret_cast<const unsigned char*>(message.data()), message.length())) {
        throw std::runtime_error("DigestSign failed");
    }
    std::string signed_message;
    signed_message.reserve(message.length() + siglen);
    signed_message.insert(signed_message.end(), message.begin(), message.end());
    signed_message.insert(signed_message.end(), signature.begin(), signature.end());
    return signed_message;
}
/** Verify a message signed by signMessage(). */
/**
 * Validate a message signed with signMessage().  Returns true only if the
 * signature is correct for the provided public key.
 */
bool Encryption::verifySignature(const std::string& message, std::string public_key) {
    if (public_key.length() != 64 && public_key.length() != 32)
        throw std::invalid_argument(
            "Expected 64-byte or 32-byte public_key (first 32 std::string = Ed25519)");
    size_t sig_offset = message.length() - 64;
    const uint8_t* msg_ptr = reinterpret_cast<const uint8_t*>(message.data());
    size_t msg_len = sig_offset;
    const uint8_t* sig_ptr = reinterpret_cast<const uint8_t*>(message.data()) + sig_offset;
    const uint8_t* pubkey_ptr = reinterpret_cast<const uint8_t*>(public_key.data());
    EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr, pubkey_ptr, 32);
    if (!pkey)
        return false;
    auto ctx = make_md_ctx();
    // Perform the verification in one shot.  Any failure is treated as an invalid signature.
    bool verified = false;
    if (EVP_DigestVerifyInit(ctx.get(), nullptr, nullptr, nullptr, pkey) == 1) {
        if (EVP_DigestVerify(ctx.get(), sig_ptr, 64, msg_ptr, msg_len) == 1)
            verified = true;
    }
    EVP_PKEY_free(pkey);
    return verified;
}
/** Concatenate and return the public signing and encryption keys. */
std::string Encryption::getPublicKey() {
    if (!sign_keypair)
        throw std::runtime_error("Sign keypair not initialized");
    if (!enc_keypair)
        throw std::runtime_error("Encryption keypair not initialized");
    std::string total_pub_key(64, '\0');
    auto sign_pub_key = sign_keypair->get_pubkey();
    if (sign_pub_key.length() != 32)
        throw std::runtime_error("Public key not initialized");
    auto enc_pub_key = enc_keypair->get_pubkey();
    if (enc_pub_key.length() != 32)
        throw std::runtime_error("Public key not initialized");
    std::memcpy(total_pub_key.data(), sign_pub_key.data(), 32);
    std::memcpy(total_pub_key.data() + 32, enc_pub_key.data(), 32);
    return total_pub_key;
}
/** Concatenate and return the private signing and encryption keys. */
std::string Encryption::getPrivateKey() {
    if (!sign_keypair)
        throw std::runtime_error("Sign keypair not initialized");
    if (!enc_keypair)
        throw std::runtime_error("Encryption keypair not initialized");
    auto priv_key = sign_keypair->get_key();
    if (!priv_key)
        throw std::runtime_error("Private key not initialized");
    auto enc_priv_key = enc_keypair->get_key();
    if (!enc_priv_key)
        throw std::runtime_error("Private key not initialized");
    std::string total_priv_key(64, '\0');
    std::memcpy(total_priv_key.data(), priv_key.get(), 32);
    std::memcpy(total_priv_key.data() + 32, enc_priv_key.get(), 32);
    return total_priv_key;
}

Encryption Encryption::fromHexStrings(const std::string& aes_hex, const std::string& mac_hex) {
    if (aes_hex.size() != 64)
        throw std::invalid_argument("Hex key must be 64 characters");
    std::string aes_bytes(32, '\0');
    for (size_t i = 0; i < 32; ++i) {
        unsigned int byte = 0;
        if (std::sscanf(aes_hex.c_str() + i * 2, "%2x", &byte) != 1)
            throw std::invalid_argument("Invalid hex character");
        aes_bytes[i] = static_cast<char>(byte);
    }
    std::string mac_bytes;
    if (!mac_hex.empty()) {
        if (mac_hex.size() != 64)
            throw std::invalid_argument("Hex key must be 64 characters");
        mac_bytes.resize(32);
        for (size_t i = 0; i < 32; ++i) {
            unsigned int byte = 0;
            if (std::sscanf(mac_hex.c_str() + i * 2, "%2x", &byte) != 1)
                throw std::invalid_argument("Invalid hex character");
            mac_bytes[i] = static_cast<char>(byte);
        }
    }
    return Encryption(aes_bytes, mac_bytes, false);
}

/**
 * Lazy initialiser for a secondary secret used when deriving encryption keys.
 * The bytes are stored obfuscated in the binary and decoded on first use.
 *
 * The obfuscation scheme itself is intentionally simple: a static lookup table
 * performs a bytewise substitution which is then reversed when the lambda is
 * invoked.  This is not meant as a strong protection mechanism, merely a way to
 * avoid shipping the raw key in the binary.  The returned SecureKey behaves the
 * same as any other user provided key and participates in the HMAC based key
 * derivation during encryption and decryption.
 */
std::function<SecureKey(void)> Encryption::obfuscatedSecondSecretFunction = []() -> SecureKey {
    std::string obfuscatedBytes(64, '\0'); // Initialize with 64 null characters
    for (size_t i = 0; i < obfuscatedBytes.length(); ++i) {
        obfuscatedBytes[i] = finalObfuscation[i];
    }
    for (size_t i = 0; i < obfuscatedBytes.length(); ++i) {
        obfuscatedBytes[i] = obfuscationSBox[static_cast<unsigned char>(obfuscatedBytes[i])];
    }
    return SecureKey(obfuscatedBytes);
};
SecureKey Encryption::obfuscatedSecondSecret = Encryption::obfuscatedSecondSecretFunction();

} // namespace manifoldb