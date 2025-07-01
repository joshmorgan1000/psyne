/**
 * @file ssd_substrate_demo.cpp
 * @brief SSD as a substrate - the physical storage device IS the message layer!
 * 
 * This demonstrates how an SSD can be a substrate that owns:
 * - Memory (flash storage sectors) 
 * - Transport (NVMe/SATA protocol)
 * 
 * Messages become direct data structures written to storage sectors.
 * Patterns coordinate I/O operations (sequential, random, queued, etc.)
 */

#include "../include/psyne/concepts/substrate_concepts.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <cstring>

// Message that represents a database record stored on SSD
struct DatabaseRecord {
    uint64_t record_id;
    uint32_t user_id;
    uint64_t timestamp;
    char data[64];
    uint32_t checksum;
    
    DatabaseRecord() : record_id(0), user_id(0), timestamp(0), checksum(0) {
        std::memset(data, 0, sizeof(data));
    }
    
    DatabaseRecord(uint64_t rid, uint32_t uid, const char* record_data) 
        : record_id(rid), user_id(uid), timestamp(0), checksum(0) {
        std::strncpy(data, record_data, sizeof(data) - 1);
        // Simple checksum
        checksum = record_id + user_id + std::strlen(data);
    }
};

// SSD Substrate - The physical storage device!
struct SSDSubstrate {
    explicit SSDSubstrate(const char* device_path = "virtual_ssd.dat") 
        : device_path_(device_path), sector_size_(4096), total_sectors_(1024) {
        
        // Initialize virtual SSD file
        ssd_file_.open(device_path_, std::ios::in | std::ios::out | std::ios::binary);
        if (!ssd_file_) {
            // Create new SSD file
            ssd_file_.open(device_path_, std::ios::out | std::ios::binary);
            if (ssd_file_) {
                // Initialize with zeros (format the drive)
                std::vector<char> zero_sector(sector_size_, 0);
                for (size_t i = 0; i < total_sectors_; ++i) {
                    ssd_file_.write(zero_sector.data(), sector_size_);
                }
                ssd_file_.close();
                
                // Reopen for read/write
                ssd_file_.open(device_path_, std::ios::in | std::ios::out | std::ios::binary);
            }
        }
        
        std::cout << "SSD: Initialized virtual SSD device at " << device_path_ << "\n";
        std::cout << "     Sectors: " << total_sectors_ << ", Sector size: " << sector_size_ << " bytes\n";
    }
    
    ~SSDSubstrate() {
        if (ssd_file_.is_open()) {
            ssd_file_.close();
        }
    }
    
    // MEMORY OWNERSHIP = Flash storage sectors
    void* allocate_memory_slab(size_t size_bytes) {
        // Calculate how many sectors we need
        size_t sectors_needed = (size_bytes + sector_size_ - 1) / sector_size_;
        
        if (next_sector_ + sectors_needed > total_sectors_) {
            std::cout << "SSD: Out of storage space!\n";
            return nullptr;
        }
        
        // Allocate RAM buffer that represents our view of the SSD sectors
        void* buffer = std::aligned_alloc(sector_size_, sectors_needed * sector_size_);
        if (buffer) {
            allocated_sector_ = next_sector_;
            allocated_sectors_ = sectors_needed;
            next_sector_ += sectors_needed;
            
            std::cout << "SSD: Allocated " << sectors_needed << " sectors starting at sector " 
                      << allocated_sector_ << " (" << size_bytes << " bytes)\n";
        }
        
        return buffer;
    }
    
    void deallocate_memory_slab(void* ptr) {
        std::cout << "SSD: Deallocating sector buffer\n";
        std::free(ptr);
        // In real implementation, could mark sectors as free in allocation table
    }
    
    // TRANSPORT = NVMe/SATA protocol operations
    void transport_send(void* data, size_t size) {
        auto* record = static_cast<DatabaseRecord*>(data);
        std::cout << "SSD: Writing record " << record->record_id << " to storage\n";
        
        if (!ssd_file_.is_open()) {
            std::cout << "SSD: ERROR - Device not ready!\n";
            return;
        }
        
        // Calculate which sector to write to based on record ID
        uint64_t target_sector = allocated_sector_ + (record->record_id % allocated_sectors_);
        uint64_t byte_offset = target_sector * sector_size_;
        
        // Seek to sector and write
        ssd_file_.seekp(byte_offset);
        ssd_file_.write(static_cast<const char*>(data), size);
        ssd_file_.flush(); // Force write to "disk"
        
        write_operations_++;
        std::cout << "     -> Wrote to sector " << target_sector << " (offset " << byte_offset << ")\n";
        std::cout << "     -> Total write ops: " << write_operations_ << "\n";
    }
    
    void transport_receive(void* buffer, size_t size) {
        std::cout << "SSD: Reading from storage sectors\n";
        
        if (!ssd_file_.is_open()) {
            std::cout << "SSD: ERROR - Device not ready!\n";
            return;
        }
        
        // For demo, read from first allocated sector
        uint64_t byte_offset = allocated_sector_ * sector_size_;
        ssd_file_.seekg(byte_offset);
        ssd_file_.read(static_cast<char*>(buffer), size);
        
        read_operations_++;
        std::cout << "     -> Read from sector " << allocated_sector_ << "\n";
        std::cout << "     -> Total read ops: " << read_operations_ << "\n";
    }
    
    // SUBSTRATE IDENTITY
    const char* substrate_name() const { return "SSD"; }
    bool is_zero_copy() const { return false; } // SSD requires I/O operations
    bool is_cross_process() const { return true; } // Storage persists across processes
    
    // SSD-specific capabilities
    void print_stats() const {
        std::cout << "SSD Statistics:\n";
        std::cout << "  Device: " << device_path_ << "\n";
        std::cout << "  Total sectors: " << total_sectors_ << "\n";
        std::cout << "  Used sectors: " << next_sector_ << "\n";
        std::cout << "  Write operations: " << write_operations_ << "\n";
        std::cout << "  Read operations: " << read_operations_ << "\n";
        std::cout << "  Utilization: " << (100.0 * next_sector_ / total_sectors_) << "%\n";
    }
    
    void defragment() {
        std::cout << "SSD: Running defragmentation...\n";
        // Could implement actual defrag logic here
    }
    
    void trim_unused_sectors() {
        std::cout << "SSD: TRIM operation on unused sectors\n";
        // Could implement TRIM/UNMAP commands here
    }

private:
    std::string device_path_;
    std::fstream ssd_file_;
    size_t sector_size_;
    size_t total_sectors_;
    size_t next_sector_ = 0;
    size_t allocated_sector_ = 0;
    size_t allocated_sectors_ = 0;
    size_t write_operations_ = 0;
    size_t read_operations_ = 0;
};

// Sequential I/O pattern for SSD
struct SequentialIOPattern {
    void* coordinate_allocation(void* slab, size_t max_messages, size_t message_size) {
        // Sequential allocation in the slab
        size_t offset = allocation_count_ * message_size;
        allocation_count_++;
        
        std::cout << "SequentialIO: Allocating message #" << allocation_count_ 
                  << " for sequential write\n";
        
        return static_cast<char*>(slab) + offset;
    }
    
    void* coordinate_receive() {
        std::cout << "SequentialIO: Coordinating sequential read\n";
        return nullptr; // Simplified for demo
    }
    
    void producer_sync() {
        std::cout << "SequentialIO: Producer sync - batching writes for efficiency\n";
    }
    
    void consumer_sync() {
        std::cout << "SequentialIO: Consumer sync - read-ahead optimization\n";
    }
    
    const char* pattern_name() const { return "SequentialIO"; }
    bool needs_locks() const { return false; } // Sequential is naturally ordered
    size_t max_producers() const { return 1; }
    size_t max_consumers() const { return 1; }
    
private:
    size_t allocation_count_ = 0;
};

// Random I/O pattern for SSD
struct RandomIOPattern {
    void* coordinate_allocation(void* slab, size_t max_messages, size_t message_size) {
        // Random allocation pattern (hash-based)
        size_t random_offset = (allocation_count_ * 97) % max_messages; // Simple hash
        allocation_count_++;
        
        std::cout << "RandomIO: Allocating message #" << allocation_count_ 
                  << " for random access at offset " << random_offset << "\n";
        
        return static_cast<char*>(slab) + (random_offset * message_size);
    }
    
    void* coordinate_receive() {
        std::cout << "RandomIO: Coordinating random access read\n";
        return nullptr;
    }
    
    void producer_sync() {
        std::cout << "RandomIO: Producer sync - optimizing for random writes\n";
    }
    
    void consumer_sync() {
        std::cout << "RandomIO: Consumer sync - cache-aware random reads\n";
    }
    
    const char* pattern_name() const { return "RandomIO"; }
    bool needs_locks() const { return true; } // Random access needs coordination
    size_t max_producers() const { return 4; }
    size_t max_consumers() const { return 4; }
    
private:
    size_t allocation_count_ = 0;
};

// Database record lens (same as before)
template<typename T>
struct RecordLens {
    explicit RecordLens(void* sector_memory) 
        : memory_view_(static_cast<T*>(sector_memory)) {
        new (memory_view_) T{};
    }
    
    template<typename... Args>
    RecordLens(void* sector_memory, Args&&... args)
        : memory_view_(static_cast<T*>(sector_memory)) {
        new (memory_view_) T(std::forward<Args>(args)...);
    }
    
    T* operator->() { return memory_view_; }
    const T* operator->() const { return memory_view_; }
    T& operator*() { return *memory_view_; }
    const T& operator*() const { return *memory_view_; }
    
    void* raw_memory() const { return memory_view_; }
    size_t size() const { return sizeof(T); }
    
private:
    T* memory_view_;
};

// SSD-based channel
template<typename RecordType, typename SSDSubstrateType, typename IOPatternType>
    requires psyne::concepts::ChannelConfiguration<RecordType, SSDSubstrateType, IOPatternType>
struct SSDChannel {
    explicit SSDChannel(size_t storage_size = 32 * 4096) { // 32 sectors
        storage_slab_ = substrate_.allocate_memory_slab(storage_size);
        max_records_ = storage_size / sizeof(RecordType);
        
        std::cout << "SSDChannel: Created with " << substrate_.substrate_name() 
                  << " storage and " << pattern_.pattern_name() << " I/O pattern\n";
        std::cout << "            Capacity: " << max_records_ << " records\n";
    }
    
    ~SSDChannel() {
        if (storage_slab_) {
            substrate_.deallocate_memory_slab(storage_slab_);
        }
    }
    
    RecordLens<RecordType> create_record() {
        void* memory = pattern_.coordinate_allocation(storage_slab_, max_records_, sizeof(RecordType));
        return RecordLens<RecordType>(memory);
    }
    
    template<typename... Args>
    RecordLens<RecordType> create_record(Args&&... args) {
        void* memory = pattern_.coordinate_allocation(storage_slab_, max_records_, sizeof(RecordType));
        return RecordLens<RecordType>(memory, std::forward<Args>(args)...);
    }
    
    void persist_record(const RecordLens<RecordType>& record) {
        substrate_.transport_send(record.raw_memory(), record.size());
    }
    
    void print_storage_stats() {
        substrate_.print_stats();
    }
    
    void optimize_storage() {
        substrate_.defragment();
        substrate_.trim_unused_sectors();
    }
    
private:
    SSDSubstrateType substrate_;
    IOPatternType pattern_;
    void* storage_slab_ = nullptr;
    size_t max_records_ = 0;
};

void demonstrate_ssd_as_substrate() {
    std::cout << "\n=== SSD as Substrate Demo ===\n";
    std::cout << "The SSD IS the physical layer of messaging!\n\n";
    
    // Sequential I/O database
    {
        std::cout << "1. Sequential I/O Database on SSD:\n";
        SSDChannel<DatabaseRecord, SSDSubstrate, SequentialIOPattern> sequential_db;
        
        // Create and persist records sequentially
        auto record1 = sequential_db.create_record(1001, 500, "User login event");
        auto record2 = sequential_db.create_record(1002, 501, "User purchase transaction");
        auto record3 = sequential_db.create_record(1003, 502, "User logout event");
        
        sequential_db.persist_record(record1);
        sequential_db.persist_record(record2);
        sequential_db.persist_record(record3);
        
        sequential_db.print_storage_stats();
        std::cout << "\n";
    }
    
    // Random I/O database  
    {
        std::cout << "2. Random I/O Database on SSD:\n";
        SSDChannel<DatabaseRecord, SSDSubstrate, RandomIOPattern> random_db;
        
        // Create and persist records with random access pattern
        auto record1 = random_db.create_record(2001, 600, "Random access record A");
        auto record2 = random_db.create_record(2002, 601, "Random access record B");
        
        random_db.persist_record(record1);
        random_db.persist_record(record2);
        
        random_db.print_storage_stats();
        random_db.optimize_storage();
        std::cout << "\n";
    }
    
    std::cout << "=== SSD Substrate Success! ===\n";
    std::cout << "✅ SSD owns the storage sectors (memory)\n";
    std::cout << "✅ SSD handles I/O protocol (transport)\n";
    std::cout << "✅ Records are lenses into storage sectors\n";
    std::cout << "✅ I/O patterns coordinate access (sequential vs random)\n";
    std::cout << "✅ The PHYSICAL DEVICE is the substrate!\n";
    std::cout << "🗄️  DATABASE ON SUBSTRATE = PURE PERFORMANCE!\n";
}

int main() {
    std::cout << "SSD as Substrate Demonstration\n";
    std::cout << "==============================\n";
    std::cout << "Physical storage devices as message substrates!\n";
    
    try {
        demonstrate_ssd_as_substrate();
        
        std::cout << "\nWhat this means:\n";
        std::cout << "===============\n";
        std::cout << "• Your SSD becomes the message layer\n";
        std::cout << "• Database records are messages stored directly on device\n";
        std::cout << "• I/O patterns optimize for sequential vs random access\n";
        std::cout << "• Zero-copy between memory and storage when possible\n";
        std::cout << "• Hardware-aware optimization built into the message system\n\n";
        
        std::cout << "Database vendors could build:\n";
        std::cout << "• MySQL substrate (InnoDB page management)\n";
        std::cout << "• PostgreSQL substrate (WAL integration)\n";
        std::cout << "• NVMe substrate (direct device access)\n";
        std::cout << "• Object storage substrate (S3/MinIO)\n\n";
        
        std::cout << "The substrate IS the physical reality! 🤯\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}