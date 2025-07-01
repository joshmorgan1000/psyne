#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace psyne::message {

/**
 * @brief Substrate-aware message that allocates additional resources
 *
 * This message type demonstrates how messages can take a substrate
 * and use it for sophisticated resource management.
 */
template <typename Substrate>
class DynamicVectorMessage {
public:
    /**
     * @brief Constructor that takes substrate - required by MessageType concept
     */
    explicit DynamicVectorMessage(Substrate &substrate)
        : substrate_(substrate), size_(0) {
        // Message could register itself with substrate, allocate additional
        // resources, etc.
    }

    /**
     * @brief Constructor with substrate and size
     */
    DynamicVectorMessage(Substrate &substrate, size_t size)
        : substrate_(substrate), size_(size) {
        if (size_ > 0) {
            // Use substrate to allocate additional memory if needed
            if constexpr (requires {
                              substrate.allocate_additional(size_t{});
                          }) {
                data_ = static_cast<float *>(
                    substrate_.allocate_additional(size_ * sizeof(float)));
            } else {
                // Fallback to regular allocation
                data_ = std::make_unique<float[]>(size_);
            }
        }
    }

    /**
     * @brief Constructor with substrate and initial data
     */
    DynamicVectorMessage(Substrate &substrate, std::vector<float> initial_data)
        : substrate_(substrate), size_(initial_data.size()) {
        if (size_ > 0) {
            if constexpr (requires {
                              substrate.allocate_additional(size_t{});
                          }) {
                data_ = static_cast<float *>(
                    substrate_.allocate_additional(size_ * sizeof(float)));
                std::copy(initial_data.begin(), initial_data.end(),
                          data_.get());
            } else {
                data_ = std::make_unique<float[]>(size_);
                std::copy(initial_data.begin(), initial_data.end(),
                          data_.get());
            }
        }
    }

    // Move-only (but not trivially copyable!)
    DynamicVectorMessage(const DynamicVectorMessage &) = delete;
    DynamicVectorMessage &operator=(const DynamicVectorMessage &) = delete;

    DynamicVectorMessage(DynamicVectorMessage &&other) noexcept
        : substrate_(other.substrate_), size_(other.size_),
          data_(std::move(other.data_)) {
        other.size_ = 0;
    }

    DynamicVectorMessage &operator=(DynamicVectorMessage &&other) noexcept {
        if (this != &other) {
            substrate_ = other.substrate_;
            size_ = other.size_;
            data_ = std::move(other.data_);
            other.size_ = 0;
        }
        return *this;
    }

    ~DynamicVectorMessage() {
        // Could notify substrate of destruction, clean up resources, etc.
        if constexpr (requires { substrate_.get().on_message_destroyed(); }) {
            substrate_.get().on_message_destroyed();
        }
    }

    /**
     * @brief Access data
     */
    float *data() {
        return data_.get();
    }
    const float *data() const {
        return data_.get();
    }

    size_t size() const {
        return size_;
    }

    float &operator[](size_t index) {
        return data_[index];
    }
    const float &operator[](size_t index) const {
        return data_[index];
    }

    /**
     * @brief Get substrate reference (messages are substrate-aware!)
     */
    Substrate &substrate() {
        return substrate_;
    }
    const Substrate &substrate() const {
        return substrate_;
    }

private:
    std::reference_wrapper<Substrate> substrate_;
    size_t size_;
    std::unique_ptr<float[]> data_;
};

/**
 * @brief String message that can use substrate for encoding/compression
 */
template <typename Substrate>
class StringMessage {
public:
    explicit StringMessage(Substrate &substrate) : substrate_(substrate) {}

    StringMessage(Substrate &substrate, const std::string &content)
        : substrate_(substrate), content_(content) {
        // Could use substrate for compression, encoding, etc.
        if constexpr (requires { substrate.compress_string(std::string{}); }) {
            compressed_content_ = substrate_.compress_string(content_);
        }
    }

    StringMessage(Substrate &substrate, std::string &&content)
        : substrate_(substrate), content_(std::move(content)) {
        if constexpr (requires { substrate.compress_string(std::string{}); }) {
            compressed_content_ = substrate_.compress_string(content_);
        }
    }

    // Move-only
    StringMessage(const StringMessage &) = delete;
    StringMessage &operator=(const StringMessage &) = delete;

    StringMessage(StringMessage &&other) noexcept
        : substrate_(other.substrate_), content_(std::move(other.content_)),
          compressed_content_(std::move(other.compressed_content_)) {}

    const std::string &content() const {
        return content_;
    }
    std::string &content() {
        return content_;
    }

    void set_content(const std::string &new_content) {
        content_ = new_content;
        if constexpr (requires {
                          substrate_.get().compress_string(std::string{});
                      }) {
            compressed_content_ = substrate_.get().compress_string(content_);
        }
    }

    Substrate &substrate() {
        return substrate_;
    }

private:
    std::reference_wrapper<Substrate> substrate_;
    std::string content_;
    std::vector<uint8_t> compressed_content_; // Optional compressed form
};

/**
 * @brief GPU-aware tensor message
 */
template <typename Substrate>
class GPUTensorMessage {
public:
    explicit GPUTensorMessage(Substrate &substrate) : substrate_(substrate) {}

    GPUTensorMessage(Substrate &substrate, std::vector<size_t> shape)
        : substrate_(substrate), shape_(std::move(shape)) {
        size_t total_elements = 1;
        for (size_t dim : shape_) {
            total_elements *= dim;
        }

        // Use substrate to allocate GPU memory if available
        if constexpr (requires { substrate.allocate_gpu_memory(size_t{}); }) {
            gpu_data_ =
                substrate_.allocate_gpu_memory(total_elements * sizeof(float));
        }
    }

    // Move-only
    GPUTensorMessage(const GPUTensorMessage &) = delete;
    GPUTensorMessage &operator=(const GPUTensorMessage &) = delete;

    GPUTensorMessage(GPUTensorMessage &&other) noexcept
        : substrate_(other.substrate_), shape_(std::move(other.shape_)),
          gpu_data_(other.gpu_data_) {
        other.gpu_data_ = nullptr;
    }

    ~GPUTensorMessage() {
        if (gpu_data_ &&
            requires { substrate_.get().deallocate_gpu_memory(gpu_data_); }) {
            substrate_.get().deallocate_gpu_memory(gpu_data_);
        }
    }

    const std::vector<size_t> &shape() const {
        return shape_;
    }
    void *gpu_data() const {
        return gpu_data_;
    }

    Substrate &substrate() {
        return substrate_;
    }

private:
    std::reference_wrapper<Substrate> substrate_;
    std::vector<size_t> shape_;
    void *gpu_data_ = nullptr;
};

/**
 * @brief Self-describing message with metadata
 */
template <typename Substrate>
class SelfDescribingMessage {
public:
    explicit SelfDescribingMessage(Substrate &substrate)
        : substrate_(substrate) {
        // Could register message type with substrate
        if constexpr (requires {
                          substrate.register_message_type("SelfDescribing");
                      }) {
            substrate_.register_message_type("SelfDescribing");
        }
    }

    template <typename T>
    SelfDescribingMessage(Substrate &substrate, T &&payload)
        : substrate_(substrate) {
        set_payload(std::forward<T>(payload));
    }

    template <typename T>
    void set_payload(T &&payload) {
        payload_ = std::forward<T>(payload);
        type_name_ = typeid(T).name();

        // Could use substrate for serialization
        if constexpr (requires { substrate_.get().serialize(payload); }) {
            serialized_data_ = substrate_.get().serialize(payload_);
        }
    }

    template <typename T>
    T &get_payload() {
        return std::any_cast<T &>(payload_);
    }

    const std::string &type_name() const {
        return type_name_;
    }
    Substrate &substrate() {
        return substrate_;
    }

private:
    std::reference_wrapper<Substrate> substrate_;
    std::any payload_;
    std::string type_name_;
    std::vector<uint8_t> serialized_data_;
};

} // namespace psyne::message