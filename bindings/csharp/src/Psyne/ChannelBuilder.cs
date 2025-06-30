using System;
using Psyne.Native;

namespace Psyne
{
    /// <summary>
    /// Builder class for creating channels with a fluent API.
    /// </summary>
    public sealed class ChannelBuilder
    {
        private string? _uri;
        private long _bufferSize = 1024 * 1024; // 1MB default
        private ChannelMode _mode = ChannelMode.SingleProducerSingleConsumer;
        private ChannelType _type = ChannelType.Single;
        private CompressionConfig? _compression;
        private bool _enableMetrics;

        /// <summary>
        /// Sets the URI for the channel.
        /// </summary>
        /// <param name="uri">The channel URI (e.g., "memory://buffer1", "tcp://localhost:8080").</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithUri(string uri)
        {
            _uri = uri ?? throw new ArgumentNullException(nameof(uri));
            return this;
        }

        /// <summary>
        /// Sets the buffer size for the channel.
        /// </summary>
        /// <param name="size">The buffer size in bytes.</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithBufferSize(long size)
        {
            if (size <= 0)
                throw new ArgumentOutOfRangeException(nameof(size), "Buffer size must be positive");
            
            _bufferSize = size;
            return this;
        }

        /// <summary>
        /// Sets the buffer size for the channel using convenient units.
        /// </summary>
        /// <param name="size">The buffer size.</param>
        /// <param name="unit">The size unit (KB, MB, GB).</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithBufferSize(long size, SizeUnit unit)
        {
            var multiplier = unit switch
            {
                SizeUnit.Bytes => 1L,
                SizeUnit.KB => 1024L,
                SizeUnit.MB => 1024L * 1024L,
                SizeUnit.GB => 1024L * 1024L * 1024L,
                _ => throw new ArgumentOutOfRangeException(nameof(unit))
            };

            return WithBufferSize(size * multiplier);
        }

        /// <summary>
        /// Sets the synchronization mode for the channel.
        /// </summary>
        /// <param name="mode">The channel mode.</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithMode(ChannelMode mode)
        {
            _mode = mode;
            return this;
        }

        /// <summary>
        /// Sets the channel to single producer, single consumer mode.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder SingleProducerSingleConsumer()
        {
            return WithMode(ChannelMode.SingleProducerSingleConsumer);
        }

        /// <summary>
        /// Sets the channel to single producer, multiple consumer mode.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder SingleProducerMultipleConsumer()
        {
            return WithMode(ChannelMode.SingleProducerMultipleConsumer);
        }

        /// <summary>
        /// Sets the channel to multiple producer, single consumer mode.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder MultipleProducerSingleConsumer()
        {
            return WithMode(ChannelMode.MultipleProducerSingleConsumer);
        }

        /// <summary>
        /// Sets the channel to multiple producer, multiple consumer mode.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder MultipleProducerMultipleConsumer()
        {
            return WithMode(ChannelMode.MultipleProducerMultipleConsumer);
        }

        /// <summary>
        /// Sets the channel type.
        /// </summary>
        /// <param name="type">The channel type.</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithType(ChannelType type)
        {
            _type = type;
            return this;
        }

        /// <summary>
        /// Sets the channel to support a single message type.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder SingleType()
        {
            return WithType(ChannelType.Single);
        }

        /// <summary>
        /// Sets the channel to support multiple message types.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder MultiType()
        {
            return WithType(ChannelType.Multi);
        }

        /// <summary>
        /// Enables compression with the specified configuration.
        /// </summary>
        /// <param name="compression">The compression configuration.</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithCompression(CompressionConfig compression)
        {
            _compression = compression ?? throw new ArgumentNullException(nameof(compression));
            return this;
        }

        /// <summary>
        /// Enables LZ4 compression with default settings.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithLz4Compression()
        {
            return WithCompression(CompressionConfig.Lz4());
        }

        /// <summary>
        /// Enables Zstd compression with default settings.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithZstdCompression()
        {
            return WithCompression(CompressionConfig.Zstd());
        }

        /// <summary>
        /// Enables Snappy compression with default settings.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithSnappyCompression()
        {
            return WithCompression(CompressionConfig.Snappy());
        }

        /// <summary>
        /// Enables metrics collection on the channel.
        /// </summary>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WithMetrics()
        {
            _enableMetrics = true;
            return this;
        }

        /// <summary>
        /// Creates a memory-based channel with the specified name.
        /// </summary>
        /// <param name="name">The memory channel name.</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder Memory(string name)
        {
            return WithUri($"memory://{name}");
        }

        /// <summary>
        /// Creates a TCP channel with the specified host and port.
        /// </summary>
        /// <param name="host">The host address.</param>
        /// <param name="port">The port number.</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder Tcp(string host, int port)
        {
            return WithUri($"tcp://{host}:{port}");
        }

        /// <summary>
        /// Creates a Unix socket channel with the specified path.
        /// </summary>
        /// <param name="path">The Unix socket path.</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder UnixSocket(string path)
        {
            return WithUri($"unix://{path}");
        }

        /// <summary>
        /// Creates a UDP multicast channel with the specified address and port.
        /// </summary>
        /// <param name="address">The multicast address.</param>
        /// <param name="port">The port number.</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder UdpMulticast(string address, int port)
        {
            return WithUri($"udp://{address}:{port}");
        }

        /// <summary>
        /// Creates a WebRTC channel for peer-to-peer communication.
        /// </summary>
        /// <param name="peerId">The target peer identifier.</param>
        /// <param name="signalingServerUri">The WebSocket signaling server URI (default: ws://localhost:8080).</param>
        /// <returns>This builder instance for method chaining.</returns>
        public ChannelBuilder WebRtc(string peerId, string signalingServerUri = "ws://localhost:8080")
        {
            return WithUri($"webrtc://{peerId}?signaling={signalingServerUri}");
        }

        /// <summary>
        /// Creates the channel with the configured settings.
        /// </summary>
        /// <returns>A new channel instance.</returns>
        public Channel Build()
        {
            if (string.IsNullOrEmpty(_uri))
                throw new InvalidOperationException("URI must be specified");

            PsyneNative.ErrorCode result;
            IntPtr channelHandle;

            if (_compression != null)
            {
                var nativeCompression = _compression.ToNative();
                result = PsyneNative.psyne_channel_create_compressed(
                    _uri,
                    (UIntPtr)_bufferSize,
                    (PsyneNative.ChannelMode)_mode,
                    (PsyneNative.ChannelType)_type,
                    ref nativeCompression,
                    out channelHandle);
            }
            else
            {
                result = PsyneNative.psyne_channel_create(
                    _uri,
                    (UIntPtr)_bufferSize,
                    (PsyneNative.ChannelMode)_mode,
                    (PsyneNative.ChannelType)_type,
                    out channelHandle);
            }

            PsyneException.ThrowIfError(result);

            var channel = new Channel(channelHandle);

            if (_enableMetrics)
            {
                channel.EnableMetrics();
            }

            return channel;
        }
    }

    /// <summary>
    /// Size units for buffer size specification.
    /// </summary>
    public enum SizeUnit
    {
        /// <summary>Bytes</summary>
        Bytes,
        /// <summary>Kilobytes (1024 bytes)</summary>
        KB,
        /// <summary>Megabytes (1024^2 bytes)</summary>
        MB,
        /// <summary>Gigabytes (1024^3 bytes)</summary>
        GB
    }
}