#include <psyne/psyne_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

// Producer thread function
void* producer_thread(void* arg) {
    psyne_channel_t* channel = (psyne_channel_t*)arg;
    psyne_error_t err;
    
    printf("Producer started\n");
    
    for (int i = 0; i < 10; i++) {
        // Create test data
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "Message %d from C API", i);
        size_t msg_size = strlen(buffer) + 1;
        
        // Send using convenience function
        err = psyne_send_data(channel, buffer, msg_size, 1);
        if (err != PSYNE_OK) {
            fprintf(stderr, "Failed to send message: %s\n", 
                    psyne_error_string(err));
        } else {
            printf("Producer sent: %s\n", buffer);
        }
        
        usleep(100000);  // 100ms
    }
    
    printf("Producer finished\n");
    return NULL;
}

// Consumer thread function
void* consumer_thread(void* arg) {
    psyne_channel_t* channel = (psyne_channel_t*)arg;
    psyne_error_t err;
    char buffer[1024];
    
    printf("Consumer started\n");
    
    int received = 0;
    while (received < 10) {
        size_t received_size;
        uint32_t type;
        
        // Receive with timeout
        err = psyne_receive_data(channel, buffer, sizeof(buffer),
                                &received_size, &type, 1000);
        
        if (err == PSYNE_OK) {
            buffer[received_size - 1] = '\0';  // Ensure null termination
            printf("Consumer received: %s (type=%u, size=%zu)\n", 
                   buffer, type, received_size);
            received++;
        } else if (err == PSYNE_ERROR_TIMEOUT) {
            printf("Consumer timeout\n");
        } else if (err != PSYNE_ERROR_NO_MESSAGE) {
            fprintf(stderr, "Receive error: %s\n", psyne_error_string(err));
        }
    }
    
    printf("Consumer finished\n");
    return NULL;
}

// Callback for event-driven receiving
void message_callback(psyne_message_t* message, uint32_t type, void* user_data) {
    int* counter = (int*)user_data;
    
    void* data;
    size_t size;
    psyne_message_get_data(message, &data, &size);
    
    printf("Callback received message %d: %.*s\n", 
           (*counter)++, (int)size, (char*)data);
}

// Demonstrate manual message operations
void demo_manual_messages(psyne_channel_t* channel) {
    printf("\n=== Manual Message Demo ===\n");
    
    // Send a message
    psyne_message_t* send_msg;
    psyne_error_t err = psyne_message_reserve(channel, 64, &send_msg);
    if (err == PSYNE_OK) {
        void* data;
        size_t size;
        psyne_message_get_data(send_msg, &data, &size);
        
        // Fill message data
        snprintf(data, size, "Manual message from C");
        
        // Send it
        err = psyne_message_send(send_msg, 42);
        if (err == PSYNE_OK) {
            printf("Sent manual message\n");
        }
    }
    
    // Receive a message
    psyne_message_t* recv_msg;
    uint32_t type;
    err = psyne_message_receive_timeout(channel, 1000, &recv_msg, &type);
    if (err == PSYNE_OK) {
        void* data;
        size_t size;
        psyne_message_get_data(recv_msg, &data, &size);
        
        printf("Received manual message: %.*s (type=%u)\n", 
               (int)size, (char*)data, type);
        
        psyne_message_release(recv_msg);
    }
}

// Demonstrate metrics
void demo_metrics(psyne_channel_t* channel) {
    printf("\n=== Metrics Demo ===\n");
    
    psyne_metrics_t metrics;
    psyne_error_t err = psyne_channel_get_metrics(channel, &metrics);
    
    if (err == PSYNE_OK) {
        printf("Channel Metrics:\n");
        printf("  Messages sent: %llu\n", (unsigned long long)metrics.messages_sent);
        printf("  Bytes sent: %llu\n", (unsigned long long)metrics.bytes_sent);
        printf("  Messages received: %llu\n", (unsigned long long)metrics.messages_received);
        printf("  Bytes received: %llu\n", (unsigned long long)metrics.bytes_received);
        printf("  Send blocks: %llu\n", (unsigned long long)metrics.send_blocks);
        printf("  Receive blocks: %llu\n", (unsigned long long)metrics.receive_blocks);
    } else {
        printf("Metrics not available: %s\n", psyne_error_string(err));
    }
}

// Main demonstration
int main(int argc, char* argv[]) {
    printf("Psyne C API Demo\n");
    printf("Version: %s\n", psyne_version());
    
    // Initialize library
    psyne_error_t err = psyne_init();
    if (err != PSYNE_OK) {
        fprintf(stderr, "Failed to initialize: %s\n", psyne_error_string(err));
        return 1;
    }
    
    // Create channel
    psyne_channel_t* channel;
    err = psyne_channel_create("memory://c_demo", 
                              1024 * 1024,  // 1MB buffer
                              PSYNE_MODE_SPSC,
                              PSYNE_TYPE_MULTI,
                              &channel);
    
    if (err != PSYNE_OK) {
        fprintf(stderr, "Failed to create channel: %s\n", 
                psyne_error_string(err));
        psyne_cleanup();
        return 1;
    }
    
    printf("Created channel\n");
    
    // Test 1: Basic producer/consumer
    printf("\n=== Basic Producer/Consumer Test ===\n");
    pthread_t producer, consumer;
    pthread_create(&producer, NULL, producer_thread, channel);
    pthread_create(&consumer, NULL, consumer_thread, channel);
    
    pthread_join(producer, NULL);
    pthread_join(consumer, NULL);
    
    // Test 2: Manual message operations
    demo_manual_messages(channel);
    
    // Test 3: Event-driven with callback
    printf("\n=== Callback Test ===\n");
    int callback_counter = 0;
    psyne_channel_set_receive_callback(channel, message_callback, &callback_counter);
    
    // Send some messages
    for (int i = 0; i < 5; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Callback test %d", i);
        psyne_send_data(channel, msg, strlen(msg) + 1, 0);
        usleep(200000);  // 200ms
    }
    
    // Disable callback
    psyne_channel_set_receive_callback(channel, NULL, NULL);
    
    // Test 4: Metrics
    demo_metrics(channel);
    
    // Test 5: Compression
    printf("\n=== Compression Test ===\n");
    psyne_compression_config_t compression = {
        .type = PSYNE_COMPRESSION_LZ4,
        .level = 1,
        .min_size_threshold = 128,
        .enable_checksum = true
    };
    
    psyne_channel_t* compressed_channel;
    err = psyne_channel_create_compressed("memory://compressed",
                                         1024 * 1024,
                                         PSYNE_MODE_SPSC,
                                         PSYNE_TYPE_MULTI,
                                         &compression,
                                         &compressed_channel);
    
    if (err == PSYNE_OK) {
        printf("Created compressed channel\n");
        
        // Send large message that will benefit from compression
        char* large_msg = malloc(10000);
        memset(large_msg, 'A', 10000);  // Highly compressible
        
        err = psyne_send_data(compressed_channel, large_msg, 10000, 0);
        if (err == PSYNE_OK) {
            printf("Sent 10KB compressible message\n");
        }
        
        free(large_msg);
        psyne_channel_destroy(compressed_channel);
    }
    
    // Cleanup
    printf("\n=== Cleanup ===\n");
    psyne_channel_stop(channel);
    
    bool stopped;
    psyne_channel_is_stopped(channel, &stopped);
    printf("Channel stopped: %s\n", stopped ? "yes" : "no");
    
    psyne_channel_destroy(channel);
    psyne_cleanup();
    
    printf("Demo completed successfully\n");
    return 0;
}