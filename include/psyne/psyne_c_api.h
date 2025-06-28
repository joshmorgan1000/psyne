/**
 * @file psyne_c_api.h
 * @brief C API for Psyne zero-copy messaging library
 * 
 * This header provides a C-compatible API for using Psyne from C and
 * other languages via FFI (Foreign Function Interface).
 */

#ifndef PSYNE_C_API_H
#define PSYNE_C_API_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version information */
#define PSYNE_VERSION_MAJOR 0
#define PSYNE_VERSION_MINOR 1
#define PSYNE_VERSION_PATCH 1

/* Error codes */
typedef enum {
    PSYNE_OK = 0,
    PSYNE_ERROR_INVALID_ARGUMENT = -1,
    PSYNE_ERROR_OUT_OF_MEMORY = -2,
    PSYNE_ERROR_CHANNEL_FULL = -3,
    PSYNE_ERROR_NO_MESSAGE = -4,
    PSYNE_ERROR_CHANNEL_STOPPED = -5,
    PSYNE_ERROR_UNSUPPORTED = -6,
    PSYNE_ERROR_IO = -7,
    PSYNE_ERROR_TIMEOUT = -8,
    PSYNE_ERROR_UNKNOWN = -99
} psyne_error_t;

/* Channel modes */
typedef enum {
    PSYNE_MODE_SPSC = 0,  /* Single Producer, Single Consumer */
    PSYNE_MODE_SPMC = 1,  /* Single Producer, Multiple Consumer */
    PSYNE_MODE_MPSC = 2,  /* Multiple Producer, Single Consumer */
    PSYNE_MODE_MPMC = 3   /* Multiple Producer, Multiple Consumer */
} psyne_channel_mode_t;

/* Channel types */
typedef enum {
    PSYNE_TYPE_SINGLE = 0,  /* Single message type */
    PSYNE_TYPE_MULTI = 1    /* Multiple message types */
} psyne_channel_type_t;

/* Compression types */
typedef enum {
    PSYNE_COMPRESSION_NONE = 0,
    PSYNE_COMPRESSION_LZ4 = 1,
    PSYNE_COMPRESSION_ZSTD = 2,
    PSYNE_COMPRESSION_SNAPPY = 3
} psyne_compression_type_t;

/* Opaque types */
typedef struct psyne_channel psyne_channel_t;
typedef struct psyne_message psyne_message_t;

/* Channel metrics */
typedef struct {
    uint64_t messages_sent;
    uint64_t bytes_sent;
    uint64_t messages_received;
    uint64_t bytes_received;
    uint64_t send_blocks;
    uint64_t receive_blocks;
} psyne_metrics_t;

/* Compression configuration */
typedef struct {
    psyne_compression_type_t type;
    int level;
    size_t min_size_threshold;
    bool enable_checksum;
} psyne_compression_config_t;

/* ============================================================================
 * Library Management
 * ============================================================================ */

/**
 * Initialize the Psyne library
 * @return Error code
 */
psyne_error_t psyne_init(void);

/**
 * Cleanup the Psyne library
 */
void psyne_cleanup(void);

/**
 * Get version string
 * @return Version string (do not free)
 */
const char* psyne_version(void);

/**
 * Get error description
 * @param error Error code
 * @return Error description (do not free)
 */
const char* psyne_error_string(psyne_error_t error);

/* ============================================================================
 * Channel Management
 * ============================================================================ */

/**
 * Create a channel
 * @param uri Channel URI (e.g., "memory://buffer1", "tcp://localhost:8080")
 * @param buffer_size Size of the channel buffer in bytes
 * @param mode Channel synchronization mode
 * @param type Channel type (single or multi-type)
 * @param channel Output channel handle
 * @return Error code
 */
psyne_error_t psyne_channel_create(
    const char* uri,
    size_t buffer_size,
    psyne_channel_mode_t mode,
    psyne_channel_type_t type,
    psyne_channel_t** channel);

/**
 * Create a channel with compression
 * @param uri Channel URI
 * @param buffer_size Buffer size
 * @param mode Channel mode
 * @param type Channel type
 * @param compression Compression configuration (can be NULL)
 * @param channel Output channel handle
 * @return Error code
 */
psyne_error_t psyne_channel_create_compressed(
    const char* uri,
    size_t buffer_size,
    psyne_channel_mode_t mode,
    psyne_channel_type_t type,
    const psyne_compression_config_t* compression,
    psyne_channel_t** channel);

/**
 * Destroy a channel
 * @param channel Channel to destroy
 */
void psyne_channel_destroy(psyne_channel_t* channel);

/**
 * Stop a channel
 * @param channel Channel to stop
 * @return Error code
 */
psyne_error_t psyne_channel_stop(psyne_channel_t* channel);

/**
 * Check if channel is stopped
 * @param channel Channel to check
 * @param stopped Output: true if stopped
 * @return Error code
 */
psyne_error_t psyne_channel_is_stopped(psyne_channel_t* channel, bool* stopped);

/**
 * Get channel URI
 * @param channel Channel
 * @param uri Output buffer for URI
 * @param uri_size Size of output buffer
 * @return Error code
 */
psyne_error_t psyne_channel_get_uri(psyne_channel_t* channel, 
                                    char* uri, size_t uri_size);

/**
 * Get channel metrics
 * @param channel Channel
 * @param metrics Output metrics
 * @return Error code
 */
psyne_error_t psyne_channel_get_metrics(psyne_channel_t* channel,
                                        psyne_metrics_t* metrics);

/* ============================================================================
 * Message Operations
 * ============================================================================ */

/**
 * Reserve space for a message
 * @param channel Channel to send on
 * @param size Size of message data
 * @param message Output message handle
 * @return Error code
 */
psyne_error_t psyne_message_reserve(psyne_channel_t* channel,
                                    size_t size,
                                    psyne_message_t** message);

/**
 * Get message data pointer
 * @param message Message handle
 * @param data Output data pointer
 * @param size Output data size
 * @return Error code
 */
psyne_error_t psyne_message_get_data(psyne_message_t* message,
                                     void** data,
                                     size_t* size);

/**
 * Send a message
 * @param message Message to send
 * @param type Message type ID
 * @return Error code
 */
psyne_error_t psyne_message_send(psyne_message_t* message, uint32_t type);

/**
 * Cancel a reserved message without sending
 * @param message Message to cancel
 */
void psyne_message_cancel(psyne_message_t* message);

/**
 * Receive a message (non-blocking)
 * @param channel Channel to receive from
 * @param message Output message handle
 * @param type Output message type
 * @return Error code (PSYNE_ERROR_NO_MESSAGE if none available)
 */
psyne_error_t psyne_message_receive(psyne_channel_t* channel,
                                    psyne_message_t** message,
                                    uint32_t* type);

/**
 * Receive a message with timeout
 * @param channel Channel to receive from
 * @param timeout_ms Timeout in milliseconds (0 = non-blocking)
 * @param message Output message handle
 * @param type Output message type
 * @return Error code
 */
psyne_error_t psyne_message_receive_timeout(psyne_channel_t* channel,
                                            uint32_t timeout_ms,
                                            psyne_message_t** message,
                                            uint32_t* type);

/**
 * Release a received message
 * @param message Message to release
 */
void psyne_message_release(psyne_message_t* message);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Send raw data (convenience function)
 * @param channel Channel to send on
 * @param data Data to send
 * @param size Size of data
 * @param type Message type
 * @return Error code
 */
psyne_error_t psyne_send_data(psyne_channel_t* channel,
                              const void* data,
                              size_t size,
                              uint32_t type);

/**
 * Receive raw data (convenience function)
 * @param channel Channel to receive from
 * @param buffer Output buffer
 * @param buffer_size Size of output buffer
 * @param received_size Actual size received
 * @param type Output message type
 * @param timeout_ms Timeout in milliseconds
 * @return Error code
 */
psyne_error_t psyne_receive_data(psyne_channel_t* channel,
                                 void* buffer,
                                 size_t buffer_size,
                                 size_t* received_size,
                                 uint32_t* type,
                                 uint32_t timeout_ms);

/* ============================================================================
 * Advanced Features (Optional)
 * ============================================================================ */

/**
 * Enable channel metrics
 * @param channel Channel
 * @param enable True to enable metrics
 * @return Error code
 */
psyne_error_t psyne_channel_enable_metrics(psyne_channel_t* channel,
                                           bool enable);

/**
 * Reset channel metrics
 * @param channel Channel
 * @return Error code
 */
psyne_error_t psyne_channel_reset_metrics(psyne_channel_t* channel);

/**
 * Get channel buffer size
 * @param channel Channel
 * @param size Output buffer size
 * @return Error code
 */
psyne_error_t psyne_channel_get_buffer_size(psyne_channel_t* channel,
                                            size_t* size);

/**
 * Set channel receive callback (for event-driven operation)
 * @param channel Channel
 * @param callback Callback function (NULL to disable)
 * @param user_data User data passed to callback
 * @return Error code
 */
typedef void (*psyne_receive_callback_t)(psyne_message_t* message,
                                         uint32_t type,
                                         void* user_data);

psyne_error_t psyne_channel_set_receive_callback(
    psyne_channel_t* channel,
    psyne_receive_callback_t callback,
    void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* PSYNE_C_API_H */