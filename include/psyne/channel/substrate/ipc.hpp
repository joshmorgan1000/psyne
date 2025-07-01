/**
 * @file ipc.hpp
 * @brief Inter-Process Communication substrate using shared memory
 *
 * IPC substrate enables zero-copy communication between processes on the same
 * machine using POSIX shared memory and semaphores for synchronization.
 */

#pragma once

#include "../../core/behaviors.hpp"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <semaphore.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace psyne::substrate {

/**
 * @brief IPC substrate using POSIX shared memory
 *
 * Creates shared memory regions that can be accessed by multiple processes.
 * Uses named shared memory objects and semaphores for cross-process
 * coordination.
 */
class IPC : public psyne::behaviors::SubstrateBehavior {
public:
    /**
     * @brief Create IPC substrate with shared memory name
     *
     * @param shm_name Name for shared memory object (e.g., "/psyne_channel_0")
     * @param create_new If true, creates new shared memory. If false, attaches
     * to existing.
     */
    explicit IPC(const std::string &shm_name = "/psyne_ipc_default",
                 bool create_new = true)
        : shm_name_(shm_name), create_new_(create_new), shm_fd_(-1),
          shm_ptr_(nullptr), shm_size_(0) {
        // Create semaphore names
        sem_producer_name_ = shm_name + "_prod_sem";
        sem_consumer_name_ = shm_name + "_cons_sem";

        if (create_new_) {
            // Clean up any existing objects
            cleanup_existing_objects();
        }
    }

    ~IPC() {
        cleanup();
    }

    /**
     * @brief Allocate shared memory slab
     */
    void *allocate_memory_slab(size_t size_bytes) override {
        if (shm_ptr_) {
            throw std::runtime_error("IPC: Shared memory already allocated");
        }

        shm_size_ = size_bytes;

        if (create_new_) {
            // Create new shared memory object
            shm_fd_ =
                shm_open(shm_name_.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
            if (shm_fd_ == -1) {
                throw std::runtime_error(
                    "IPC: Failed to create shared memory object: " +
                    std::string(strerror(errno)));
            }

            // Set the size
            if (ftruncate(shm_fd_, size_bytes) == -1) {
                close(shm_fd_);
                shm_unlink(shm_name_.c_str());
                throw std::runtime_error(
                    "IPC: Failed to set shared memory size: " +
                    std::string(strerror(errno)));
            }

            // Create producer semaphore (starts at max capacity)
            sem_producer_ = sem_open(sem_producer_name_.c_str(),
                                     O_CREAT | O_EXCL, 0666, size_bytes / 64);
            if (sem_producer_ == SEM_FAILED) {
                close(shm_fd_);
                shm_unlink(shm_name_.c_str());
                throw std::runtime_error(
                    "IPC: Failed to create producer semaphore: " +
                    std::string(strerror(errno)));
            }

            // Create consumer semaphore (starts at 0)
            sem_consumer_ =
                sem_open(sem_consumer_name_.c_str(), O_CREAT | O_EXCL, 0666, 0);
            if (sem_consumer_ == SEM_FAILED) {
                close(shm_fd_);
                shm_unlink(shm_name_.c_str());
                sem_close(sem_producer_);
                sem_unlink(sem_producer_name_.c_str());
                throw std::runtime_error(
                    "IPC: Failed to create consumer semaphore: " +
                    std::string(strerror(errno)));
            }

        } else {
            // Attach to existing shared memory
            shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
            if (shm_fd_ == -1) {
                throw std::runtime_error(
                    "IPC: Failed to open existing shared memory: " +
                    std::string(strerror(errno)));
            }

            // Open existing semaphores
            sem_producer_ = sem_open(sem_producer_name_.c_str(), 0);
            if (sem_producer_ == SEM_FAILED) {
                close(shm_fd_);
                throw std::runtime_error(
                    "IPC: Failed to open producer semaphore: " +
                    std::string(strerror(errno)));
            }

            sem_consumer_ = sem_open(sem_consumer_name_.c_str(), 0);
            if (sem_consumer_ == SEM_FAILED) {
                close(shm_fd_);
                sem_close(sem_producer_);
                throw std::runtime_error(
                    "IPC: Failed to open consumer semaphore: " +
                    std::string(strerror(errno)));
            }
        }

        // Map the shared memory
        shm_ptr_ = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                        shm_fd_, 0);
        if (shm_ptr_ == MAP_FAILED) {
            cleanup();
            throw std::runtime_error("IPC: Failed to map shared memory: " +
                                     std::string(strerror(errno)));
        }

        // Initialize to zero if we created it
        if (create_new_) {
            std::memset(shm_ptr_, 0, size_bytes);
        }

        return shm_ptr_;
    }

    /**
     * @brief Deallocate shared memory
     */
    void deallocate_memory_slab(void *memory) override {
        if (memory == shm_ptr_) {
            cleanup();
        }
    }

    /**
     * @brief Transport send (notify consumer via semaphore)
     */
    void transport_send(void *data, size_t size) override {
        // In shared memory IPC, sending means notifying consumer
        if (sem_post(sem_consumer_) == -1) {
            throw std::runtime_error(
                "IPC: Failed to signal consumer semaphore: " +
                std::string(strerror(errno)));
        }
    }

    /**
     * @brief Transport receive (wait for producer via semaphore)
     */
    void transport_receive(void *buffer, size_t buffer_size) override {
        // In shared memory IPC, receiving means waiting for producer
        if (sem_wait(sem_consumer_) == -1) {
            throw std::runtime_error(
                "IPC: Failed to wait on consumer semaphore: " +
                std::string(strerror(errno)));
        }

        // Signal producer that we consumed a message
        if (sem_post(sem_producer_) == -1) {
            throw std::runtime_error(
                "IPC: Failed to signal producer semaphore: " +
                std::string(strerror(errno)));
        }
    }

    /**
     * @brief Try to receive without blocking
     */
    bool try_transport_receive() {
        if (sem_trywait(sem_consumer_) == 0) {
            // Got a message, signal producer
            if (sem_post(sem_producer_) == -1) {
                throw std::runtime_error(
                    "IPC: Failed to signal producer semaphore: " +
                    std::string(strerror(errno)));
            }
            return true;
        }
        return false; // No message available
    }

    /**
     * @brief Substrate identity
     */
    const char *substrate_name() const override {
        return "IPC";
    }
    bool is_zero_copy() const override {
        return true;
    } // True shared memory
    bool is_cross_process() const override {
        return true;
    } // Cross-process by design

    /**
     * @brief Get shared memory info
     */
    const std::string &get_shm_name() const {
        return shm_name_;
    }
    size_t get_shm_size() const {
        return shm_size_;
    }
    void *get_shm_ptr() const {
        return shm_ptr_;
    }

    /**
     * @brief Check if we're the creator of the shared memory
     */
    bool is_creator() const {
        return create_new_;
    }

private:
    void cleanup_existing_objects() {
        // Try to clean up any existing shared memory objects
        shm_unlink(shm_name_.c_str());
        sem_unlink(sem_producer_name_.c_str());
        sem_unlink(sem_consumer_name_.c_str());
        // Ignore errors - objects might not exist
    }

    void cleanup() {
        if (shm_ptr_ && shm_ptr_ != MAP_FAILED) {
            munmap(shm_ptr_, shm_size_);
            shm_ptr_ = nullptr;
        }

        if (shm_fd_ != -1) {
            close(shm_fd_);
            shm_fd_ = -1;
        }

        if (sem_producer_ && sem_producer_ != SEM_FAILED) {
            sem_close(sem_producer_);
            if (create_new_) {
                sem_unlink(sem_producer_name_.c_str());
            }
            sem_producer_ = nullptr;
        }

        if (sem_consumer_ && sem_consumer_ != SEM_FAILED) {
            sem_close(sem_consumer_);
            if (create_new_) {
                sem_unlink(sem_consumer_name_.c_str());
            }
            sem_consumer_ = nullptr;
        }

        if (create_new_) {
            shm_unlink(shm_name_.c_str());
        }
    }

private:
    std::string shm_name_;
    std::string sem_producer_name_;
    std::string sem_consumer_name_;
    bool create_new_;

    int shm_fd_;
    void *shm_ptr_;
    size_t shm_size_;

    sem_t *sem_producer_;
    sem_t *sem_consumer_;
};

} // namespace psyne::substrate