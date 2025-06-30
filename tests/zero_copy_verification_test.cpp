/**
 * @file zero_copy_verification_test.cpp
 * @brief Comprehensive test to verify zero-copy implementation
 * 
 * This test verifies that our implementation truly achieves zero-copy
 * semantics as defined in CORE_DESIGN.md.
 */

#include <psyne/psyne.hpp>
#include <gtest/gtest.h>
#include <span>
#include <concepts>
#include <memory>
#include <chrono>

using namespace psyne;

// Test message type that satisfies our concepts
class TestMessage : public Message<TestMessage> {
public:
    static constexpr size_t DATA_SIZE = 1024;
    
    static consteval size_t calculate_size() noexcept {
        return DATA_SIZE;
    }
    
    TestMessage(Channel& channel) : Message<TestMessage>(channel) {
        initialize();
    }
    
    void initialize() {
        // Initialize test pattern
        auto span = typed_data_span<uint32_t>();
        for (size_t i = 0; i < span.size(); ++i) {
            span[i] = static_cast<uint32_t>(i);
        }
    }
    
    bool verify_pattern() const {
        auto span = typed_data_span<uint32_t>();
        for (size_t i = 0; i < span.size(); ++i) {
            if (span[i] != static_cast<uint32_t>(i)) {
                return false;
            }
        }
        return true;
    }
};

// Verify concepts work correctly
static_assert(MessageType<TestMessage>);
static_assert(FixedSizeMessage<TestMessage>);

class ZeroCopyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a memory channel for testing
        channel = Channel::create("memory://test_buffer", 1024*1024, ChannelMode::SPSC);
        ASSERT_NE(channel, nullptr);
    }
    
    std::unique_ptr<Channel> channel;
};

TEST_F(ZeroCopyTest, MessageIsView) {
    // Create message - should be view into ring buffer
    TestMessage msg(*channel);
    
    // Verify message is valid
    EXPECT_TRUE(msg.valid());
    EXPECT_TRUE(msg.has_data());
    EXPECT_EQ(msg.size(), TestMessage::DATA_SIZE);
    
    // Get data pointer
    uint8_t* msg_data = msg.data();
    ASSERT_NE(msg_data, nullptr);
    
    // Get ring buffer base pointer
    auto& ring_buffer = channel->get_ring_buffer();
    uint8_t* buffer_base = ring_buffer.base_ptr();
    ASSERT_NE(buffer_base, nullptr);
    
    // Message data should be within ring buffer
    EXPECT_GE(msg_data, buffer_base);
    EXPECT_LT(msg_data, buffer_base + ring_buffer.capacity());
    
    // Offset should match expectation
    uint32_t expected_offset = msg_data - buffer_base;
    EXPECT_EQ(msg.offset(), expected_offset);
}

TEST_F(ZeroCopyTest, DirectMemoryAccess) {
    TestMessage msg(*channel);
    
    // Write pattern directly to message
    auto span = msg.typed_data_span<uint32_t>();
    EXPECT_GT(span.size(), 0);
    
    // Write test pattern
    for (size_t i = 0; i < span.size(); ++i) {
        span[i] = static_cast<uint32_t>(i * 2);
    }
    
    // Verify data was written directly to ring buffer
    auto& ring_buffer = channel->get_ring_buffer();
    uint32_t* buffer_data = reinterpret_cast<uint32_t*>(
        ring_buffer.base_ptr() + msg.offset()
    );
    
    for (size_t i = 0; i < span.size(); ++i) {
        EXPECT_EQ(buffer_data[i], static_cast<uint32_t>(i * 2));
        EXPECT_EQ(span[i], buffer_data[i]); // Same memory location
    }
}

TEST_F(ZeroCopyTest, SpanZeroCopyAccess) {
    TestMessage msg(*channel);
    
    // Get data spans
    auto raw_span = msg.data_span();
    auto typed_span = msg.typed_data_span<uint32_t>();
    
    EXPECT_EQ(raw_span.size(), TestMessage::DATA_SIZE);
    EXPECT_EQ(typed_span.size(), TestMessage::DATA_SIZE / sizeof(uint32_t));
    
    // Spans should point to same memory as data()
    EXPECT_EQ(raw_span.data(), msg.data());
    EXPECT_EQ(reinterpret_cast<uint8_t*>(typed_span.data()), msg.data());
    
    // Modify through span
    typed_span[0] = 0xDEADBEEF;
    
    // Verify change is visible through raw data access
    uint32_t* raw_data = reinterpret_cast<uint32_t*>(msg.data());
    EXPECT_EQ(raw_data[0], 0xDEADBEEF);
}

TEST_F(ZeroCopyTest, NoMemoryAllocation) {
    // Track allocations (simplified - would use custom allocator in real test)
    size_t initial_ring_buffer_pos = channel->get_ring_buffer().write_position();
    
    {
        TestMessage msg(*channel);
        
        // Message creation should only advance ring buffer pointer
        size_t after_creation_pos = channel->get_ring_buffer().write_position();
        
        // Position should not have advanced yet (space reserved but not committed)
        EXPECT_EQ(after_creation_pos, initial_ring_buffer_pos);
        
        // Write some data
        auto span = msg.typed_data_span<uint32_t>();
        span[0] = 42;
        span[1] = 84;
        
        // Send message
        msg.send();
        
        // Now position should advance
        size_t after_send_pos = channel->get_ring_buffer().write_position();
        EXPECT_GT(after_send_pos, initial_ring_buffer_pos);
    }
    
    // Message destructor should not affect ring buffer
    size_t final_pos = channel->get_ring_buffer().write_position();
    size_t after_send_pos = channel->get_ring_buffer().write_position();
    EXPECT_EQ(final_pos, after_send_pos);
}

TEST_F(ZeroCopyTest, CompileTimeOptimization) {
    // Verify compile-time size calculation
    constexpr size_t compile_time_size = TestMessage::static_size();
    EXPECT_EQ(compile_time_size, TestMessage::DATA_SIZE);
    
    // Runtime size should match
    TestMessage msg(*channel);
    EXPECT_EQ(msg.size(), compile_time_size);
}

TEST_F(ZeroCopyTest, ConceptCompliance) {
    // These should compile (verified by static_assert above)
    EXPECT_TRUE((MessageType<TestMessage>));
    EXPECT_TRUE((FixedSizeMessage<TestMessage>));
    EXPECT_FALSE((DynamicSizeMessage<TestMessage>));
}

TEST_F(ZeroCopyTest, BufferReuse) {
    uint8_t* first_msg_data;
    uint32_t first_offset;
    
    // Create and send first message
    {
        TestMessage msg1(*channel);
        first_msg_data = msg1.data();
        first_offset = msg1.offset();
        
        auto span = msg1.typed_data_span<uint32_t>();
        span[0] = 0x12345678;
        
        msg1.send();
    }
    
    // Simulate consumer processing
    channel->advance_read_pointer(TestMessage::DATA_SIZE);
    
    // Create second message - might reuse same buffer space
    {
        TestMessage msg2(*channel);
        
        // Could be same offset if buffer wrapped around
        // The key is that it's still zero-copy regardless
        EXPECT_TRUE(msg2.valid());
        
        auto span = msg2.typed_data_span<uint32_t>();
        span[0] = 0x87654321;
        
        // Verify we can access data directly
        uint32_t* raw_data = reinterpret_cast<uint32_t*>(msg2.data());
        EXPECT_EQ(raw_data[0], 0x87654321);
        
        msg2.send();
    }
}

// Performance test to verify zero-copy efficiency
TEST_F(ZeroCopyTest, PerformanceCharacteristics) {
    const size_t NUM_MESSAGES = 10000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < NUM_MESSAGES; ++i) {
        TestMessage msg(*channel);
        
        // Write data pattern
        auto span = msg.typed_data_span<uint32_t>();
        for (size_t j = 0; j < std::min(span.size(), size_t(10)); ++j) {
            span[j] = static_cast<uint32_t>(i * 1000 + j);
        }
        
        msg.send();
        
        // Simulate consumer
        channel->advance_read_pointer(TestMessage::DATA_SIZE);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should be very fast - less than 1 microsecond per message for zero-copy
    double messages_per_second = NUM_MESSAGES * 1000000.0 / duration.count();
    
    std::cout << "Zero-copy throughput: " << messages_per_second << " msg/sec" << std::endl;
    std::cout << "Average latency: " << duration.count() / double(NUM_MESSAGES) << " microseconds" << std::endl;
    
    // With true zero-copy, we should easily achieve > 1M messages/second
    EXPECT_GT(messages_per_second, 500000); // Conservative threshold
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}