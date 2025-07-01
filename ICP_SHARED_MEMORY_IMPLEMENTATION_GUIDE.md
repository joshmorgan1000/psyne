# IPC Shared Memory Implementation Guide

## Overview

This guide provides a complete implementation reference for Inter-Process Communication (IPC) using shared memory. Shared memory is the fastest IPC method because processes directly access the same memory region without kernel involvement for data transfer.

## Key Concepts

1. **Shared Memory**: A memory segment accessible by multiple processes
2. **Synchronization**: Required to prevent race conditions (mutexes, semaphores)
3. **Memory Layout**: Must be carefully designed for different process architectures
4. **Lifetime Management**: The memory persists until explicitly removed

## Platform-Specific APIs

### POSIX (Linux/macOS)
- `shm_open()`: Create/open shared memory
- `mmap()`: Map memory into process space
- `sem_open()`: Create named semaphores for synchronization
- `shm_unlink()`: Remove shared memory

### Windows
- `CreateFileMapping()`: Create shared memory
- `MapViewOfFile()`: Map into process space
- `CreateMutex()`: For synchronization
- `CloseHandle()`: Cleanup

## Complete Implementation

### 1. Basic Shared Memory Wrapper

```cpp
#pragma once
#include <string>
#include <cstring>
#include <stdexcept>
#include <memory>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <semaphore.h>
#endif

class SharedMemory {
private:
    std::string name_;
    size_t size_;
    void* ptr_;
    bool is_creator_;
    
#ifdef _WIN32
    HANDLE handle_;
#else
    int fd_;
#endif

public:
    SharedMemory(const std::string& name, size_t size, bool create = false)
        : name_(name), size_(size), ptr_(nullptr), is_creator_(create) {
        
#ifdef _WIN32
        // Windows implementation
        std::string full_name = "Global\\" + name;
        
        if (create) {
            handle_ = CreateFileMapping(
                INVALID_HANDLE_VALUE,
                NULL,
                PAGE_READWRITE,
                0,
                static_cast<DWORD>(size),
                full_name.c_str()
            );
            
            if (!handle_) {
                throw std::runtime_error("Failed to create shared memory");
            }
        } else {
            handle_ = OpenFileMapping(
                FILE_MAP_ALL_ACCESS,
                FALSE,
                full_name.c_str()
            );
            
            if (!handle_) {
                throw std::runtime_error("Failed to open shared memory");
            }
        }
        
        ptr_ = MapViewOfFile(
            handle_,
            FILE_MAP_ALL_ACCESS,
            0, 0,
            size
        );
        
        if (!ptr_) {
            CloseHandle(handle_);
            throw std::runtime_error("Failed to map shared memory");
        }
#else
        // POSIX implementation
        int flags = O_RDWR;
        if (create) {
            flags |= O_CREAT | O_EXCL;
        }
        
        fd_ = shm_open(name.c_str(), flags, 0666);
        if (fd_ == -1) {
            throw std::runtime_error("Failed to open shared memory");
        }
        
        if (create) {
            if (ftruncate(fd_, size) == -1) {
                close(fd_);
                shm_unlink(name.c_str());
                throw std::runtime_error("Failed to resize shared memory");
            }
        }
        
        ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                    MAP_SHARED, fd_, 0);
        
        if (ptr_ == MAP_FAILED) {
            close(fd_);
            if (create) shm_unlink(name.c_str());
            throw std::runtime_error("Failed to map shared memory");
        }
#endif
        
        // Initialize memory if creator
        if (create) {
            std::memset(ptr_, 0, size);
        }
    }
    
    ~SharedMemory() {
#ifdef _WIN32
        if (ptr_) {
            UnmapViewOfFile(ptr_);
        }
        if (handle_) {
            CloseHandle(handle_);
        }
#else
        if (ptr_ && ptr_ != MAP_FAILED) {
            munmap(ptr_, size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }
        if (is_creator_) {
            shm_unlink(name_.c_str());
        }
#endif
    }
    
    // No copy
    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;
    
    // Move operations
    SharedMemory(SharedMemory&& other) noexcept
        : name_(std::move(other.name_))
        , size_(other.size_)
        , ptr_(other.ptr_)
        , is_creator_(other.is_creator_)
#ifdef _WIN32
        , handle_(other.handle_)
#else
        , fd_(other.fd_)
#endif
    {
        other.ptr_ = nullptr;
#ifdef _WIN32
        other.handle_ = nullptr;
#else
        other.fd_ = -1;
#endif
    }
    
    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t size() const { return size_; }
    
    template<typename T>
    T* as() { return static_cast<T*>(ptr_); }
    
    template<typename T>
    const T* as() const { return static_cast<const T*>(ptr_); }
};
```

### 2. Shared Memory with Synchronization

```cpp
class SyncedSharedMemory {
private:
    SharedMemory shm_;
    
#ifdef _WIN32
    HANDLE mutex_;
#else
    sem_t* semaphore_;
#endif
    
public:
    SyncedSharedMemory(const std::string& name, size_t size, bool create = false)
        : shm_(name, size, create) {
        
        std::string sync_name = name + "_mutex";
        
#ifdef _WIN32
        if (create) {
            mutex_ = CreateMutex(NULL, FALSE, sync_name.c_str());
        } else {
            mutex_ = OpenMutex(MUTEX_ALL_ACCESS, FALSE, sync_name.c_str());
        }
        
        if (!mutex_) {
            throw std::runtime_error("Failed to create/open mutex");
        }
#else
        if (create) {
            sem_unlink(sync_name.c_str()); // Clean up any existing
            semaphore_ = sem_open(sync_name.c_str(), O_CREAT | O_EXCL, 0666, 1);
        } else {
            semaphore_ = sem_open(sync_name.c_str(), 0);
        }
        
        if (semaphore_ == SEM_FAILED) {
            throw std::runtime_error("Failed to create/open semaphore");
        }
#endif
    }
    
    ~SyncedSharedMemory() {
#ifdef _WIN32
        if (mutex_) {
            CloseHandle(mutex_);
        }
#else
        if (semaphore_ && semaphore_ != SEM_FAILED) {
            sem_close(semaphore_);
            if (shm_.is_creator_) {
                std::string sync_name = shm_.name_ + "_mutex";
                sem_unlink(sync_name.c_str());
            }
        }
#endif
    }
    
    class Lock {
    private:
        SyncedSharedMemory& parent_;
        
    public:
        explicit Lock(SyncedSharedMemory& parent) : parent_(parent) {
#ifdef _WIN32
            WaitForSingleObject(parent_.mutex_, INFINITE);
#else
            sem_wait(parent_.semaphore_);
#endif
        }
        
        ~Lock() {
#ifdef _WIN32
            ReleaseMutex(parent_.mutex_);
#else
            sem_post(parent_.semaphore_);
#endif
        }
    };
    
    Lock lock() { return Lock(*this); }
    
    void* data() { return shm_.data(); }
    const void* data() const { return shm_.data(); }
    size_t size() const { return shm_.size(); }
    
    template<typename T>
    T* as() { return shm_.as<T>(); }
};
```

### 3. Ring Buffer in Shared Memory

```cpp
template<typename T>
class SharedRingBuffer {
private:
    struct Header {
        std::atomic<size_t> write_pos;
        std::atomic<size_t> read_pos;
        size_t capacity;
        char padding[64 - 3 * sizeof(size_t)]; // Cache line alignment
    };
    
    SyncedSharedMemory shm_;
    Header* header_;
    T* buffer_;
    
public:
    SharedRingBuffer(const std::string& name, size_t capacity, bool create = false)
        : shm_(name, sizeof(Header) + capacity * sizeof(T), create) {
        
        header_ = shm_.as<Header>();
        buffer_ = reinterpret_cast<T*>(
            reinterpret_cast<char*>(shm_.data()) + sizeof(Header)
        );
        
        if (create) {
            header_->write_pos = 0;
            header_->read_pos = 0;
            header_->capacity = capacity;
        }
    }
    
    bool push(const T& item) {
        auto lock = shm_.lock();
        
        size_t write_pos = header_->write_pos.load();
        size_t read_pos = header_->read_pos.load();
        
        size_t next_write = (write_pos + 1) % header_->capacity;
        
        if (next_write == read_pos) {
            return false; // Buffer full
        }
        
        buffer_[write_pos] = item;
        header_->write_pos.store(next_write);
        return true;
    }
    
    bool pop(T& item) {
        auto lock = shm_.lock();
        
        size_t write_pos = header_->write_pos.load();
        size_t read_pos = header_->read_pos.load();
        
        if (read_pos == write_pos) {
            return false; // Buffer empty
        }
        
        item = buffer_[read_pos];
        header_->read_pos.store((read_pos + 1) % header_->capacity);
        return true;
    }
    
    size_t size() const {
        auto lock = const_cast<SharedRingBuffer*>(this)->shm_.lock();
        
        size_t write_pos = header_->write_pos.load();
        size_t read_pos = header_->read_pos.load();
        
        if (write_pos >= read_pos) {
            return write_pos - read_pos;
        } else {
            return header_->capacity - read_pos + write_pos;
        }
    }
    
    bool empty() const { return size() == 0; }
    bool full() const { return size() == header_->capacity - 1; }
};
```

### 4. Usage Examples

```cpp
// Example 1: Simple shared memory
void producer_process() {
    SharedMemory shm("my_data", 1024, true); // Create
    
    int* counter = shm.as<int>();
    for (int i = 0; i < 100; ++i) {
        *counter = i;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void consumer_process() {
    SharedMemory shm("my_data", 1024, false); // Open existing
    
    int* counter = shm.as<int>();
    int last_value = -1;
    
    while (true) {
        int current = *counter;
        if (current != last_value) {
            std::cout << "Value: " << current << std::endl;
            last_value = current;
        }
        
        if (current >= 99) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// Example 2: Ring buffer for messages
struct Message {
    int type;
    char data[256];
};

void message_producer() {
    SharedRingBuffer<Message> buffer("msg_queue", 100, true);
    
    for (int i = 0; i < 1000; ++i) {
        Message msg;
        msg.type = i % 3;
        snprintf(msg.data, sizeof(msg.data), "Message %d", i);
        
        while (!buffer.push(msg)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void message_consumer() {
    SharedRingBuffer<Message> buffer("msg_queue", 100, false);
    
    Message msg;
    while (true) {
        if (buffer.pop(msg)) {
            std::cout << "Type: " << msg.type 
                      << ", Data: " << msg.data << std::endl;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}
```

## Important Considerations

### 1. Memory Layout
- Use POD (Plain Old Data) types only
- No pointers or references (they won't be valid in other processes)
- Be careful with alignment and padding
- Consider endianness for cross-platform use

### 2. Synchronization
- Always use proper synchronization (mutex/semaphore)
- Consider lock-free algorithms for high performance
- Be aware of priority inversion and deadlocks

### 3. Error Handling
- Shared memory might not exist when reader starts
- Handle cleanup if processes crash
- Set appropriate permissions

### 4. Performance Tips
- Align data to cache lines (64 bytes)
- Minimize lock contention
- Use lock-free atomics where possible
- Batch operations to reduce synchronization overhead

### 5. Security
- Set appropriate permissions on shared memory
- Validate all data from shared memory (treat as untrusted)
- Consider encryption for sensitive data

## Platform-Specific Notes

### Linux
- Shared memory files appear in `/dev/shm/`
- Clean up with `rm /dev/shm/name` if process crashes
- Check limits with `ipcs -lm`

### macOS
- Similar to Linux but some differences in limits
- May need to increase shared memory limits in system settings

### Windows
- Use "Global\" prefix for system-wide shared memory
- Requires appropriate privileges for global objects
- Different security model than POSIX

## Debugging Tips

1. Check if shared memory exists:
   - Linux/macOS: `ls /dev/shm/`
   - Windows: Use Process Explorer

2. Monitor with tools:
   - Linux: `ipcs`, `pmap`
   - macOS: `vmmap`
   - Windows: VMMap from Sysinternals

3. Common issues:
   - Permission denied: Check user/group permissions
   - Already exists: Previous process didn't clean up
   - Size mismatch: Ensure all processes use same size