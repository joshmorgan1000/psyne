// Package psyne provides Go bindings for the Psyne zero-copy messaging library.
//
// Psyne is a high-performance messaging library optimized for AI/ML applications.
// It provides zero-copy message passing between threads and processes.
//
// Example:
//
//	import "github.com/joshmorgan1000/psyne/go"
//	
//	// Initialize the library
//	if err := psyne.Init(); err != nil {
//	    log.Fatal(err)
//	}
//	defer psyne.Cleanup()
//	
//	// Create a channel
//	channel, err := psyne.NewChannel("memory://demo", 1024*1024, psyne.ModeSPSC, psyne.TypeMulti)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer channel.Close()
//	
//	// Send a message
//	if err := channel.SendData([]byte("Hello from Go!"), 1); err != nil {
//	    log.Fatal(err)
//	}
//	
//	// Receive a message
//	data, msgType, err := channel.ReceiveData(1024, time.Second)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Received: %s (type: %d)\n", string(data), msgType)
package psyne

// #cgo CFLAGS: -I../../include
// #cgo LDFLAGS: -L/usr/local/lib -lpsyne -lstdc++
// #include <psyne/psyne_c_api.h>
// #include <stdlib.h>
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

// Error types
var (
	ErrInvalidArgument = errors.New("invalid argument")
	ErrOutOfMemory     = errors.New("out of memory")
	ErrChannelFull     = errors.New("channel full")
	ErrNoMessage       = errors.New("no message available")
	ErrChannelStopped  = errors.New("channel stopped")
	ErrUnsupported     = errors.New("unsupported operation")
	ErrIO              = errors.New("I/O error")
	ErrTimeout         = errors.New("timeout")
	ErrUnknown         = errors.New("unknown error")
)

// ChannelMode defines the synchronization mode for a channel
type ChannelMode int

const (
	// ModeSPSC is Single Producer, Single Consumer mode
	ModeSPSC ChannelMode = C.PSYNE_MODE_SPSC
	// ModeSPMC is Single Producer, Multiple Consumer mode
	ModeSPMC ChannelMode = C.PSYNE_MODE_SPMC
	// ModeMPSC is Multiple Producer, Single Consumer mode
	ModeMPSC ChannelMode = C.PSYNE_MODE_MPSC
	// ModeMPMC is Multiple Producer, Multiple Consumer mode
	ModeMPMC ChannelMode = C.PSYNE_MODE_MPMC
)

// ChannelType defines whether a channel supports single or multiple message types
type ChannelType int

const (
	// TypeSingle supports a single message type
	TypeSingle ChannelType = C.PSYNE_TYPE_SINGLE
	// TypeMulti supports multiple message types
	TypeMulti ChannelType = C.PSYNE_TYPE_MULTI
)

// CompressionType defines the compression algorithm
type CompressionType int

const (
	// CompressionNone disables compression
	CompressionNone CompressionType = C.PSYNE_COMPRESSION_NONE
	// CompressionLZ4 uses LZ4 compression
	CompressionLZ4 CompressionType = C.PSYNE_COMPRESSION_LZ4
	// CompressionZstd uses Zstandard compression
	CompressionZstd CompressionType = C.PSYNE_COMPRESSION_ZSTD
	// CompressionSnappy uses Snappy compression
	CompressionSnappy CompressionType = C.PSYNE_COMPRESSION_SNAPPY
)

// CompressionConfig specifies compression settings
type CompressionConfig struct {
	Type             CompressionType
	Level            int
	MinSizeThreshold int
	EnableChecksum   bool
}

// Metrics contains channel performance metrics
type Metrics struct {
	MessagesSent     uint64
	BytesSent        uint64
	MessagesReceived uint64
	BytesReceived    uint64
	SendBlocks       uint64
	ReceiveBlocks    uint64
}

// Init initializes the Psyne library. Must be called before using any other functions.
func Init() error {
	return handleError(C.psyne_init())
}

// Cleanup cleans up the Psyne library. Should be called when done using the library.
func Cleanup() {
	C.psyne_cleanup()
}

// Version returns the Psyne library version string
func Version() string {
	return C.GoString(C.psyne_version())
}

// Channel represents a Psyne communication channel
type Channel struct {
	ptr *C.psyne_channel_t
	uri string
}

// NewChannel creates a new channel
func NewChannel(uri string, bufferSize int, mode ChannelMode, channelType ChannelType) (*Channel, error) {
	cURI := C.CString(uri)
	defer C.free(unsafe.Pointer(cURI))
	
	var channelPtr *C.psyne_channel_t
	err := handleError(C.psyne_channel_create(
		cURI,
		C.size_t(bufferSize),
		C.psyne_channel_mode_t(mode),
		C.psyne_channel_type_t(channelType),
		&channelPtr,
	))
	
	if err != nil {
		return nil, err
	}
	
	ch := &Channel{
		ptr: channelPtr,
		uri: uri,
	}
	
	// Set finalizer to ensure cleanup
	runtime.SetFinalizer(ch, (*Channel).Close)
	
	return ch, nil
}

// NewChannelWithCompression creates a new channel with compression enabled
func NewChannelWithCompression(uri string, bufferSize int, mode ChannelMode, 
	channelType ChannelType, compression CompressionConfig) (*Channel, error) {
	
	cURI := C.CString(uri)
	defer C.free(unsafe.Pointer(cURI))
	
	cCompression := C.psyne_compression_config_t{
		type_:              C.psyne_compression_type_t(compression.Type),
		level:              C.int(compression.Level),
		min_size_threshold: C.size_t(compression.MinSizeThreshold),
		enable_checksum:    C.bool(compression.EnableChecksum),
	}
	
	var channelPtr *C.psyne_channel_t
	err := handleError(C.psyne_channel_create_compressed(
		cURI,
		C.size_t(bufferSize),
		C.psyne_channel_mode_t(mode),
		C.psyne_channel_type_t(channelType),
		&cCompression,
		&channelPtr,
	))
	
	if err != nil {
		return nil, err
	}
	
	ch := &Channel{
		ptr: channelPtr,
		uri: uri,
	}
	
	runtime.SetFinalizer(ch, (*Channel).Close)
	
	return ch, nil
}

// Close destroys the channel
func (ch *Channel) Close() error {
	if ch.ptr != nil {
		C.psyne_channel_destroy(ch.ptr)
		ch.ptr = nil
		runtime.SetFinalizer(ch, nil)
	}
	return nil
}

// Stop stops the channel
func (ch *Channel) Stop() error {
	return handleError(C.psyne_channel_stop(ch.ptr))
}

// IsStopped checks if the channel is stopped
func (ch *Channel) IsStopped() (bool, error) {
	var stopped C.bool
	err := handleError(C.psyne_channel_is_stopped(ch.ptr, &stopped))
	return bool(stopped), err
}

// URI returns the channel URI
func (ch *Channel) URI() string {
	return ch.uri
}

// Metrics returns the channel metrics
func (ch *Channel) Metrics() (*Metrics, error) {
	var cMetrics C.psyne_metrics_t
	err := handleError(C.psyne_channel_get_metrics(ch.ptr, &cMetrics))
	if err != nil {
		return nil, err
	}
	
	return &Metrics{
		MessagesSent:     uint64(cMetrics.messages_sent),
		BytesSent:        uint64(cMetrics.bytes_sent),
		MessagesReceived: uint64(cMetrics.messages_received),
		BytesReceived:    uint64(cMetrics.bytes_received),
		SendBlocks:       uint64(cMetrics.send_blocks),
		ReceiveBlocks:    uint64(cMetrics.receive_blocks),
	}, nil
}

// SendData sends raw data through the channel
func (ch *Channel) SendData(data []byte, msgType uint32) error {
	if len(data) == 0 {
		return ErrInvalidArgument
	}
	
	return handleError(C.psyne_send_data(
		ch.ptr,
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)),
		C.uint32_t(msgType),
	))
}

// ReceiveData receives raw data from the channel
func (ch *Channel) ReceiveData(maxSize int, timeout time.Duration) ([]byte, uint32, error) {
	buffer := make([]byte, maxSize)
	var receivedSize C.size_t
	var msgType C.uint32_t
	timeoutMs := uint32(timeout.Milliseconds())
	
	err := handleError(C.psyne_receive_data(
		ch.ptr,
		unsafe.Pointer(&buffer[0]),
		C.size_t(maxSize),
		&receivedSize,
		&msgType,
		C.uint32_t(timeoutMs),
	))
	
	if err != nil {
		if err == ErrNoMessage || err == ErrTimeout {
			return nil, 0, err
		}
		return nil, 0, err
	}
	
	return buffer[:receivedSize], uint32(msgType), nil
}

// ResetMetrics resets the channel metrics
func (ch *Channel) ResetMetrics() error {
	return handleError(C.psyne_channel_reset_metrics(ch.ptr))
}

// Message represents a manual message for zero-copy operations
type Message struct {
	ptr     *C.psyne_message_t
	channel *Channel
}

// ReserveMessage reserves space for a message
func (ch *Channel) ReserveMessage(size int) (*Message, error) {
	var msgPtr *C.psyne_message_t
	err := handleError(C.psyne_message_reserve(ch.ptr, C.size_t(size), &msgPtr))
	if err != nil {
		return nil, err
	}
	
	msg := &Message{
		ptr:     msgPtr,
		channel: ch,
	}
	
	runtime.SetFinalizer(msg, (*Message).Cancel)
	return msg, nil
}

// Data returns the message data buffer
func (msg *Message) Data() ([]byte, error) {
	var dataPtr unsafe.Pointer
	var size C.size_t
	
	err := handleError(C.psyne_message_get_data(msg.ptr, &dataPtr, &size))
	if err != nil {
		return nil, err
	}
	
	// Create a slice that references the C memory
	// This is safe because the message owns the memory until Send or Cancel
	return (*[1 << 30]byte)(dataPtr)[:size:size], nil
}

// Send sends the message
func (msg *Message) Send(msgType uint32) error {
	err := handleError(C.psyne_message_send(msg.ptr, C.uint32_t(msgType)))
	// Clear finalizer since message is consumed
	runtime.SetFinalizer(msg, nil)
	msg.ptr = nil
	return err
}

// Cancel cancels a reserved message without sending
func (msg *Message) Cancel() {
	if msg.ptr != nil {
		C.psyne_message_cancel(msg.ptr)
		runtime.SetFinalizer(msg, nil)
		msg.ptr = nil
	}
}

// ReceiveMessage receives a message manually
func (ch *Channel) ReceiveMessage(timeout time.Duration) ([]byte, uint32, error) {
	var msgPtr *C.psyne_message_t
	var msgType C.uint32_t
	timeoutMs := uint32(timeout.Milliseconds())
	
	err := handleError(C.psyne_message_receive_timeout(
		ch.ptr,
		C.uint32_t(timeoutMs),
		&msgPtr,
		&msgType,
	))
	
	if err != nil {
		return nil, 0, err
	}
	
	// Get data from message
	var dataPtr unsafe.Pointer
	var size C.size_t
	
	err = handleError(C.psyne_message_get_data(msgPtr, &dataPtr, &size))
	if err != nil {
		C.psyne_message_release(msgPtr)
		return nil, 0, err
	}
	
	// Copy data since we need to release the message
	data := make([]byte, size)
	copy(data, (*[1 << 30]byte)(dataPtr)[:size:size])
	
	C.psyne_message_release(msgPtr)
	
	return data, uint32(msgType), nil
}

// handleError converts C error codes to Go errors
func handleError(err C.psyne_error_t) error {
	switch err {
	case C.PSYNE_OK:
		return nil
	case C.PSYNE_ERROR_INVALID_ARGUMENT:
		return ErrInvalidArgument
	case C.PSYNE_ERROR_OUT_OF_MEMORY:
		return ErrOutOfMemory
	case C.PSYNE_ERROR_CHANNEL_FULL:
		return ErrChannelFull
	case C.PSYNE_ERROR_NO_MESSAGE:
		return ErrNoMessage
	case C.PSYNE_ERROR_CHANNEL_STOPPED:
		return ErrChannelStopped
	case C.PSYNE_ERROR_UNSUPPORTED:
		return ErrUnsupported
	case C.PSYNE_ERROR_IO:
		return ErrIO
	case C.PSYNE_ERROR_TIMEOUT:
		return ErrTimeout
	default:
		return fmt.Errorf("%w: %s", ErrUnknown, C.GoString(C.psyne_error_string(err)))
	}
}