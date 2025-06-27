#pragma once

#include <memory>
#include <openssl/evp.h>
#include <stdexcept>

namespace manifoldb {

/**
 * @brief Unique pointer aliases and helper factories for OpenSSL structures.
 *
 * These helpers enforce RAII semantics for EVP context objects so that
 * resources are always released, preventing leaks when exceptions occur.
 */
using cipher_ctx_ptr = std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)>;
using md_ctx_ptr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
using pkey_ctx_ptr = std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)>;

/** Create a new EVP_CIPHER_CTX wrapped in a unique_ptr. */
inline cipher_ctx_ptr make_cipher_ctx() {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx)
        throw std::runtime_error("EVP_CIPHER_CTX_new failed");
    return cipher_ctx_ptr(ctx, &EVP_CIPHER_CTX_free);
}

/** Create a new EVP_MD_CTX wrapped in a unique_ptr. */
inline md_ctx_ptr make_md_ctx() {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx)
        throw std::runtime_error("EVP_MD_CTX_new failed");
    return md_ctx_ptr(ctx, &EVP_MD_CTX_free);
}

/** Create a new EVP_PKEY_CTX for the given key wrapped in a unique_ptr. */
inline pkey_ctx_ptr make_pkey_ctx(EVP_PKEY* pkey) {
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new(pkey, nullptr);
    if (!ctx)
        throw std::runtime_error("EVP_PKEY_CTX_new failed");
    return pkey_ctx_ptr(ctx, &EVP_PKEY_CTX_free);
}

/** Create a new EVP_PKEY_CTX for a key type wrapped in a unique_ptr. */
inline pkey_ctx_ptr make_pkey_ctx_id(int id) {
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(id, nullptr);
    if (!ctx)
        throw std::runtime_error("EVP_PKEY_CTX_new_id failed");
    return pkey_ctx_ptr(ctx, &EVP_PKEY_CTX_free);
}

} // namespace manifoldb
