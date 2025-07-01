/**
 * @file physical_vs_abstract_demo.cpp
 * @brief Demonstrates the distinction between physical substrate and abstract message
 * 
 * SUBSTRATE = Physical reality (electrons, magnetic fields, photons)
 * MESSAGE = Abstract interpretation lens that gives meaning to physical states
 * 
 * The same physical bits can be interpreted as completely different messages!
 */

#include <iostream>
#include <cstring>
#include <memory>
#include <cstdint>

// The PHYSICAL substrate - just raw storage and transport
struct PhysicalStorage {
    explicit PhysicalStorage(size_t size_bytes) : size_(size_bytes) {
        // Allocate raw physical memory (just bytes)
        physical_memory_ = std::aligned_alloc(64, size_bytes);
        std::memset(physical_memory_, 0, size_bytes);
        
        std::cout << "PhysicalStorage: Allocated " << size_bytes << " bytes of raw physical memory\n";
        std::cout << "                Physical reality: electrons in DRAM cells\n";
    }
    
    ~PhysicalStorage() {
        std::cout << "PhysicalStorage: Deallocating physical memory\n";
        std::free(physical_memory_);
    }
    
    // Raw physical operations - no interpretation
    void* get_raw_memory() { return physical_memory_; }
    size_t size() const { return size_; }
    
    void write_raw_bytes(size_t offset, const void* data, size_t len) {
        std::memcpy(static_cast<char*>(physical_memory_) + offset, data, len);
        std::cout << "PhysicalStorage: Wrote " << len << " raw bytes at offset " << offset << "\n";
        std::cout << "                Physical: Changed electron states in memory cells\n";
    }
    
    void read_raw_bytes(size_t offset, void* buffer, size_t len) {
        std::memcpy(buffer, static_cast<char*>(physical_memory_) + offset, len);
        std::cout << "PhysicalStorage: Read " << len << " raw bytes from offset " << offset << "\n";
        std::cout << "                Physical: Sensed electron states from memory cells\n";
    }
    
    // Show the raw physical bytes (no interpretation)
    void dump_raw_bytes(size_t offset, size_t len) {
        std::cout << "Raw physical bytes at offset " << offset << ": ";
        unsigned char* bytes = static_cast<unsigned char*>(physical_memory_) + offset;
        for (size_t i = 0; i < len; ++i) {
            printf("%02X ", bytes[i]);
        }
        std::cout << "\n";
    }

private:
    void* physical_memory_;
    size_t size_;
};

// ABSTRACT MESSAGE 1: Interpret physical bytes as a database record
struct DatabaseRecordLens {
    struct Layout {
        uint32_t id;
        float value;
        char name[16];
    };
    
    explicit DatabaseRecordLens(void* physical_memory) 
        : physical_ptr_(static_cast<Layout*>(physical_memory)) {
        std::cout << "DatabaseRecordLens: Interpreting physical memory as database record\n";
    }
    
    // Abstract operations on the physical memory
    void set_record(uint32_t id, float value, const char* name) {
        physical_ptr_->id = id;
        physical_ptr_->value = value;
        std::strncpy(physical_ptr_->name, name, sizeof(physical_ptr_->name) - 1);
        std::cout << "DatabaseRecord: Set ID=" << id << " Value=" << value << " Name='" << name << "'\n";
        std::cout << "               Abstract: Logical database record\n";
    }
    
    void print_record() {
        std::cout << "DatabaseRecord: ID=" << physical_ptr_->id 
                  << " Value=" << physical_ptr_->value 
                  << " Name='" << physical_ptr_->name << "'\n";
    }
    
    void* raw_physical() { return physical_ptr_; }
    size_t size() const { return sizeof(Layout); }

private:
    Layout* physical_ptr_;
};

// ABSTRACT MESSAGE 2: Interpret THE SAME physical bytes as a network packet
struct NetworkPacketLens {
    struct Layout {
        uint32_t dest_ip;      // Same 4 bytes as database ID
        float bandwidth;       // Same 4 bytes as database value  
        char payload[16];      // Same 16 bytes as database name
    };
    
    explicit NetworkPacketLens(void* physical_memory) 
        : physical_ptr_(static_cast<Layout*>(physical_memory)) {
        std::cout << "NetworkPacketLens: Interpreting physical memory as network packet\n";
    }
    
    void set_packet(uint32_t dest_ip, float bandwidth, const char* payload) {
        physical_ptr_->dest_ip = dest_ip;
        physical_ptr_->bandwidth = bandwidth;
        std::strncpy(physical_ptr_->payload, payload, sizeof(physical_ptr_->payload) - 1);
        std::cout << "NetworkPacket: DestIP=" << std::hex << dest_ip << std::dec 
                  << " Bandwidth=" << bandwidth << " Payload='" << payload << "'\n";
        std::cout << "              Abstract: Logical network packet\n";
    }
    
    void print_packet() {
        std::cout << "NetworkPacket: DestIP=" << std::hex << physical_ptr_->dest_ip << std::dec
                  << " Bandwidth=" << physical_ptr_->bandwidth 
                  << " Payload='" << physical_ptr_->payload << "'\n";
    }
    
    void* raw_physical() { return physical_ptr_; }
    size_t size() const { return sizeof(Layout); }

private:
    Layout* physical_ptr_;
};

// ABSTRACT MESSAGE 3: Interpret THE SAME physical bytes as image pixels
struct ImagePixelLens {
    struct RGBAPixel {
        uint8_t r, g, b, a;
    };
    
    explicit ImagePixelLens(void* physical_memory) 
        : physical_ptr_(static_cast<RGBAPixel*>(physical_memory)) {
        std::cout << "ImagePixelLens: Interpreting physical memory as image pixels\n";
    }
    
    void set_pixels_from_bytes(const void* data, size_t num_pixels) {
        std::memcpy(physical_ptr_, data, num_pixels * sizeof(RGBAPixel));
        std::cout << "ImagePixels: Set " << num_pixels << " pixels from raw bytes\n";
        std::cout << "            Abstract: Visual image data\n";
    }
    
    void print_pixels(size_t num_pixels) {
        std::cout << "ImagePixels: ";
        for (size_t i = 0; i < num_pixels; ++i) {
            std::cout << "(" << (int)physical_ptr_[i].r << "," << (int)physical_ptr_[i].g 
                      << "," << (int)physical_ptr_[i].b << "," << (int)physical_ptr_[i].a << ") ";
        }
        std::cout << "\n";
    }
    
    void* raw_physical() { return physical_ptr_; }

private:
    RGBAPixel* physical_ptr_;
};

void demonstrate_physical_vs_abstract() {
    std::cout << "\n=== Physical vs Abstract Demo ===\n";
    std::cout << "Same physical bytes, completely different abstract interpretations!\n\n";
    
    // Create physical substrate
    PhysicalStorage storage(1024);
    void* physical_memory = storage.get_raw_memory();
    
    std::cout << "\n1. Writing data through DatabaseRecord lens:\n";
    {
        DatabaseRecordLens db_lens(physical_memory);
        db_lens.set_record(0x12345678, 3.14159f, "test_record");
        
        std::cout << "\nPhysical reality after database write:\n";
        storage.dump_raw_bytes(0, 24);
        
        std::cout << "\nReading back through database lens:\n";
        db_lens.print_record();
    }
    
    std::cout << "\n2. Interpreting THE SAME physical bytes as NetworkPacket:\n";
    {
        NetworkPacketLens net_lens(physical_memory);
        
        std::cout << "Reading the same physical memory through network lens:\n";
        net_lens.print_packet();
        
        std::cout << "\nNotice: Same bytes, completely different meaning!\n";
        std::cout << "- Database ID 0x12345678 becomes Network DestIP 0x12345678\n";
        std::cout << "- Database Value 3.14159 becomes Network Bandwidth 3.14159\n";
        std::cout << "- Database Name becomes Network Payload\n";
    }
    
    std::cout << "\n3. Interpreting THE SAME physical bytes as Image Pixels:\n";
    {
        ImagePixelLens img_lens(physical_memory);
        
        std::cout << "Reading the same physical memory through image lens:\n";
        img_lens.print_pixels(6); // 24 bytes = 6 RGBA pixels
        
        std::cout << "\nNotice: Same bytes, now interpreted as color values!\n";
        std::cout << "- The 0x78 byte becomes red=120\n";
        std::cout << "- The 0x56 byte becomes green=86\n";
        std::cout << "- Same physical electrons, different abstract meaning!\n";
    }
    
    std::cout << "\n4. Modifying through one lens affects all others:\n";
    {
        std::cout << "Modifying through NetworkPacket lens:\n";
        NetworkPacketLens net_lens(physical_memory);
        net_lens.set_packet(0xDEADBEEF, 9.99f, "new_payload");
        
        std::cout << "\nPhysical reality after network write:\n";
        storage.dump_raw_bytes(0, 24);
        
        std::cout << "\nReading through database lens (same physical memory):\n";
        DatabaseRecordLens db_lens(physical_memory);
        db_lens.print_record();
        
        std::cout << "\nReading through image lens (same physical memory):\n";
        ImagePixelLens img_lens(physical_memory);
        img_lens.print_pixels(6);
        
        std::cout << "\n🤯 SAME PHYSICAL BYTES, THREE DIFFERENT REALITIES!\n";
    }
    
    std::cout << "\n=== The Truth ===\n";
    std::cout << "• SUBSTRATE = Physical reality (electrons, photons, magnetic fields)\n";
    std::cout << "• MESSAGE = Abstract lens that interprets physical states\n";
    std::cout << "• Same substrate can support infinite abstract interpretations\n";
    std::cout << "• The physical doesn't care about our abstractions\n";
    std::cout << "• Our abstractions give meaning to meaningless physics\n";
    std::cout << "• This is the FOUNDATION of all computing! ⚡\n";
}

int main() {
    std::cout << "Physical Substrate vs Abstract Message Demonstration\n";
    std::cout << "===================================================\n";
    std::cout << "Understanding the reality beneath the abstractions\n";
    
    try {
        demonstrate_physical_vs_abstract();
        
        std::cout << "\nPsyne Architecture Reality:\n";
        std::cout << "==========================\n";
        std::cout << "SUBSTRATE owns the physical:\n";
        std::cout << "• Memory allocation = Controlling electrons in DRAM\n";
        std::cout << "• Transport send = Moving photons through fiber optic\n";
        std::cout << "• Storage write = Aligning magnetic domains on disk\n\n";
        
        std::cout << "MESSAGE provides the abstraction:\n";
        std::cout << "• DatabaseRecord lens = Interprets bytes as structured data\n";
        std::cout << "• NetworkPacket lens = Interprets bytes as protocol data\n";
        std::cout << "• ImagePixel lens = Interprets bytes as visual data\n\n";
        
        std::cout << "The substrate doesn't know or care what the message thinks it is.\n";
        std::cout << "The message gives semantic meaning to meaningless physical states.\n";
        std::cout << "This separation enables infinite flexibility! 🌟\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}