/**
 * @file psyne_bindings.cpp
 * @brief Python bindings for the Psyne library using pybind11
 * 
 * This file provides Python bindings for the core Psyne functionality,
 * allowing Python applications to use Psyne's high-performance messaging.
 * 
 * @author Psyne Contributors
 * @date 2025
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <psyne/psyne.hpp>
#include <memory>
#include <vector>

namespace py = pybind11;
using namespace psyne;

/**
 * @brief Python wrapper for FloatVector that works with NumPy arrays
 */
class PyFloatVector {
public:
    explicit PyFloatVector(Channel& channel) : channel_(channel) {
        // Reserve space for default size
        reserve(1024);
    }
    
    explicit PyFloatVector(Channel& channel, size_t size) : channel_(channel) {
        reserve(size);
    }
    
    ~PyFloatVector() {
        if (data_ptr_) {
            channel_.get_impl()->release_message(data_ptr_);
        }
    }
    
    void reserve(size_t size) {
        if (data_ptr_) {
            channel_.get_impl()->release_message(data_ptr_);
        }
        
        data_ptr_ = channel_.get_impl()->reserve_space(size * sizeof(float));
        if (data_ptr_) {
            size_ = size;
            capacity_ = size;
        }
    }
    
    void resize(size_t new_size) {
        if (new_size > capacity_) {
            reserve(new_size);
        }
        size_ = new_size;
    }
    
    void from_numpy(py::array_t<float> input) {
        py::buffer_info buf_info = input.request();
        
        if (buf_info.ndim != 1) {
            throw std::runtime_error("Input array must be 1-dimensional");
        }
        
        size_t size = buf_info.shape[0];
        resize(size);
        
        float* data = static_cast<float*>(data_ptr_);
        float* src = static_cast<float*>(buf_info.ptr);
        std::memcpy(data, src, size * sizeof(float));
    }
    
    py::array_t<float> to_numpy() const {
        if (!data_ptr_ || size_ == 0) {
            return py::array_t<float>();
        }
        
        // Create numpy array that shares data (read-only)
        return py::array_t<float>(
            size_,                           // size
            {sizeof(float)},                 // strides
            static_cast<float*>(data_ptr_),  // data pointer
            py::none()                       // parent (no ownership transfer)
        );
    }
    
    void send() {
        if (data_ptr_) {
            channel_.get_impl()->commit_message(data_ptr_);
            data_ptr_ = nullptr; // Message is now owned by channel
        }
    }
    
    size_t size() const { return size_; }
    
    float& operator[](size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return static_cast<float*>(data_ptr_)[index];
    }
    
    const float& operator[](size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Index out of range");
        }
        return static_cast<float*>(data_ptr_)[index];
    }

private:
    Channel& channel_;
    void* data_ptr_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
};

/**
 * @brief Python wrapper for Channel that provides simplified interface
 */
class PyChannel {
public:
    explicit PyChannel(std::unique_ptr<Channel> channel) 
        : channel_(std::move(channel)) {}
    
    PyFloatVector create_float_vector(size_t size = 1024) {
        return PyFloatVector(*channel_, size);
    }
    
    py::object receive_float_vector() {
        size_t size;
        uint32_t type;
        void* data = channel_->receive_raw_message(size, type);
        
        if (!data) {
            return py::none();
        }
        
        // Convert to numpy array
        size_t float_count = size / sizeof(float);
        auto result = py::array_t<float>(
            float_count,
            {sizeof(float)},
            static_cast<float*>(data),
            py::none()
        );
        
        // Release the message after copying data
        channel_->release_raw_message(data);
        
        return result;
    }
    
    std::string uri() const {
        return channel_->uri();
    }
    
    void stop() {
        channel_->stop();
    }
    
    bool is_stopped() const {
        return channel_->is_stopped();
    }
    
    debug::ChannelMetrics get_metrics() const {
        return channel_->get_metrics();
    }
    
    void reset_metrics() {
        channel_->reset_metrics();
    }
    
    // Zero-copy API methods (v1.3.0)
    uint32_t reserve_write_slot(size_t size) {
        return channel_->reserve_write_slot(size);
    }
    
    void notify_message_ready(uint32_t offset, size_t size) {
        channel_->notify_message_ready(offset, size);
    }
    
    void advance_read_pointer(size_t size) {
        channel_->advance_read_pointer(size);
    }
    
    py::object get_buffer_view() {
        auto span = channel_->buffer_span();
        if (span.empty()) {
            return py::none();
        }
        
        // Create read-only numpy array view of the ring buffer
        return py::array_t<uint8_t>(
            span.size(),
            {sizeof(uint8_t)},
            span.data(),
            py::none()
        );
    }

private:
    std::unique_ptr<Channel> channel_;
};

PYBIND11_MODULE(psyne, m) {
    m.doc() = "Psyne - High-performance zero-copy messaging library for Python";
    
    m.attr("__version__") = "1.3.0";
    
    // Version functions
    m.def("version", &version, "Get library version string");
    m.def("print_banner", &print_banner, "Print Psyne banner");
    
    // Enums
    py::enum_<ChannelMode>(m, "ChannelMode")
        .value("SPSC", ChannelMode::SPSC, "Single Producer, Single Consumer")
        .value("SPMC", ChannelMode::SPMC, "Single Producer, Multiple Consumer")
        .value("MPSC", ChannelMode::MPSC, "Multiple Producer, Single Consumer")
        .value("MPMC", ChannelMode::MPMC, "Multiple Producer, Multiple Consumer")
        .export_values();
    
    py::enum_<ChannelType>(m, "ChannelType")
        .value("SingleType", ChannelType::SingleType, "Single message type")
        .value("MultiType", ChannelType::MultiType, "Multiple message types")
        .export_values();
    
    // Compression enums and config
    py::enum_<compression::CompressionType>(m, "CompressionType")
        .value("None", compression::CompressionType::None, "No compression")
        .value("LZ4", compression::CompressionType::LZ4, "LZ4 compression")
        .value("Zstd", compression::CompressionType::Zstd, "Zstandard compression")
        .value("Snappy", compression::CompressionType::Snappy, "Snappy compression")
        .export_values();
    
    py::class_<compression::CompressionConfig>(m, "CompressionConfig")
        .def(py::init<>())
        .def_readwrite("type", &compression::CompressionConfig::type)
        .def_readwrite("level", &compression::CompressionConfig::level)
        .def_readwrite("min_size_threshold", &compression::CompressionConfig::min_size_threshold)
        .def_readwrite("enabled", &compression::CompressionConfig::enabled);
    
    // Metrics
    py::class_<debug::ChannelMetrics>(m, "ChannelMetrics")
        .def(py::init<>())
        .def_readonly("messages_sent", &debug::ChannelMetrics::messages_sent)
        .def_readonly("bytes_sent", &debug::ChannelMetrics::bytes_sent)
        .def_readonly("messages_received", &debug::ChannelMetrics::messages_received)
        .def_readonly("bytes_received", &debug::ChannelMetrics::bytes_received)
        .def_readonly("send_blocks", &debug::ChannelMetrics::send_blocks)
        .def_readonly("receive_blocks", &debug::ChannelMetrics::receive_blocks)
        .def("__repr__", [](const debug::ChannelMetrics& m) {
            return "<ChannelMetrics: sent=" + std::to_string(m.messages_sent) + 
                   " received=" + std::to_string(m.messages_received) + ">";
        });
    
    // FloatVector wrapper
    py::class_<PyFloatVector>(m, "FloatVector")
        .def("resize", &PyFloatVector::resize, "Resize the vector")
        .def("from_numpy", &PyFloatVector::from_numpy, "Copy data from NumPy array")
        .def("to_numpy", &PyFloatVector::to_numpy, "Convert to NumPy array")
        .def("send", &PyFloatVector::send, "Send the message")
        .def("size", &PyFloatVector::size, "Get size of vector")
        .def("__getitem__", py::overload_cast<size_t>(&PyFloatVector::operator[], py::const_))
        .def("__setitem__", [](PyFloatVector& v, size_t i, float val) { v[i] = val; })
        .def("__len__", &PyFloatVector::size)
        .def("__repr__", [](const PyFloatVector& v) {
            return "<FloatVector size=" + std::to_string(v.size()) + ">";
        });
    
    // Channel wrapper
    py::class_<PyChannel>(m, "Channel")
        .def("create_float_vector", &PyChannel::create_float_vector, 
             "Create a new FloatVector message", py::arg("size") = 1024)
        .def("receive_float_vector", &PyChannel::receive_float_vector,
             "Receive a FloatVector as NumPy array")
        .def("uri", &PyChannel::uri, "Get channel URI")
        .def("stop", &PyChannel::stop, "Stop the channel")
        .def("is_stopped", &PyChannel::is_stopped, "Check if channel is stopped")
        .def("get_metrics", &PyChannel::get_metrics, "Get channel metrics")
        .def("reset_metrics", &PyChannel::reset_metrics, "Reset channel metrics")
        // Zero-copy API methods (v1.3.0)
        .def("reserve_write_slot", &PyChannel::reserve_write_slot, 
             "Reserve space in ring buffer (zero-copy API)", py::arg("size"))
        .def("notify_message_ready", &PyChannel::notify_message_ready,
             "Notify that message is ready (zero-copy API)", py::arg("offset"), py::arg("size"))
        .def("advance_read_pointer", &PyChannel::advance_read_pointer,
             "Advance read pointer after consuming message", py::arg("size"))
        .def("get_buffer_view", &PyChannel::get_buffer_view,
             "Get NumPy view of ring buffer for zero-copy access")
        .def("__repr__", [](const PyChannel& c) {
            return "<Channel uri='" + c.uri() + "'>";
        });
    
    // Factory functions
    m.def("create_ipc_channel", [](const std::string& name, size_t buffer_size, ChannelMode mode) {
        auto channel = create_ipc_channel(name, buffer_size, mode);
        return std::make_unique<PyChannel>(std::move(channel));
    }, "Create IPC channel", py::arg("name"), py::arg("buffer_size") = 1024*1024, py::arg("mode") = ChannelMode::SPSC);
    
    m.def("create_tcp_server", [](uint16_t port, size_t buffer_size) {
        auto channel = tcp::create_server(port, buffer_size);
        return std::make_unique<PyChannel>(std::move(channel));
    }, "Create TCP server", py::arg("port"), py::arg("buffer_size") = 1024*1024);
    
    m.def("create_tcp_client", [](const std::string& host, uint16_t port, size_t buffer_size) {
        auto channel = tcp::create_client(host, port, buffer_size);
        return std::make_unique<PyChannel>(std::move(channel));
    }, "Create TCP client", py::arg("host"), py::arg("port"), py::arg("buffer_size") = 1024*1024);
    
    m.def("create_unix_channel", [](const std::string& path, unix_socket::Role role, size_t buffer_size) {
        auto channel = unix_socket::create_channel(path, role, buffer_size);
        return std::make_unique<PyChannel>(std::move(channel));
    }, "Create Unix domain socket channel", py::arg("path"), py::arg("role"), py::arg("buffer_size") = 1024*1024);
    
    // Unix socket role enum
    py::enum_<unix_socket::Role>(m, "UnixSocketRole")
        .value("Server", unix_socket::Role::Server, "Unix socket server")
        .value("Client", unix_socket::Role::Client, "Unix socket client")
        .export_values();
    
    // Multicast functions
    m.def("create_multicast_publisher", [](const std::string& address, uint16_t port, 
                                          size_t buffer_size, const compression::CompressionConfig& config) {
        auto channel = multicast::create_publisher(address, port, buffer_size, config);
        return std::make_unique<PyChannel>(std::move(channel));
    }, "Create UDP multicast publisher", 
       py::arg("address"), py::arg("port"), py::arg("buffer_size") = 1024*1024, 
       py::arg("compression_config") = compression::CompressionConfig{});
    
    m.def("create_multicast_subscriber", [](const std::string& address, uint16_t port, 
                                           size_t buffer_size, const std::string& interface) {
        auto channel = multicast::create_subscriber(address, port, buffer_size, interface);
        return std::make_unique<PyChannel>(std::move(channel));
    }, "Create UDP multicast subscriber", 
       py::arg("address"), py::arg("port"), py::arg("buffer_size") = 1024*1024, 
       py::arg("interface_address") = "");
    
    // WebRTC support
    m.def("create_webrtc_channel", [](const std::string& peer_id, size_t buffer_size, 
                                     const std::string& signaling_server_uri) {
        auto channel = webrtc::create_channel(peer_id, buffer_size, signaling_server_uri);
        return std::make_unique<PyChannel>(std::move(channel));
    }, "Create WebRTC channel for peer-to-peer communication", 
       py::arg("peer_id"), py::arg("buffer_size") = 1024*1024, 
       py::arg("signaling_server_uri") = "ws://localhost:8080");
    
    // Zero-copy API support
    m.def("create_channel", [](const std::string& uri, size_t buffer_size, ChannelMode mode, 
                              ChannelType type, bool enable_metrics, 
                              const compression::CompressionConfig& compression_config) {
        auto channel = create_channel(uri, buffer_size, mode, type, enable_metrics, compression_config);
        return std::make_unique<PyChannel>(std::move(channel));
    }, "Create channel with full v1.3.0 API support", 
       py::arg("uri"), py::arg("buffer_size") = 1024*1024, py::arg("mode") = ChannelMode::SPSC,
       py::arg("type") = ChannelType::MultiType, py::arg("enable_metrics") = false,
       py::arg("compression_config") = compression::CompressionConfig{});
    
    // Debugging utilities
    py::class_<debug::ChannelInspector>(m, "ChannelInspector")
        .def(py::init<>())
        .def("inspect_channel", [](debug::ChannelInspector& inspector, PyChannel& channel) {
            // Simple inspection - could be expanded
            return "Channel: " + channel.uri();
        }, "Inspect channel state");
}