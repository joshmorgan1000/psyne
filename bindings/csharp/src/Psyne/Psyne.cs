using System;
using System.Runtime.InteropServices;
using Psyne.Native;

namespace Psyne
{
    /// <summary>
    /// Main entry point for the Psyne high-performance messaging library.
    /// </summary>
    public static class Psyne
    {
        private static readonly object _initLock = new();
        private static bool _initialized;

        /// <summary>
        /// Gets the version of the Psyne library.
        /// </summary>
        public static string Version
        {
            get
            {
                var ptr = PsyneNative.psyne_version();
                return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) ?? "Unknown" : "Unknown";
            }
        }

        /// <summary>
        /// Initializes the Psyne library. This must be called before using any other Psyne functionality.
        /// </summary>
        /// <exception cref="PsyneException">Thrown if initialization fails.</exception>
        public static void Initialize()
        {
            lock (_initLock)
            {
                if (_initialized)
                    return;

                var result = PsyneNative.psyne_init();
                PsyneException.ThrowIfError(result);
                _initialized = true;
            }
        }

        /// <summary>
        /// Cleans up the Psyne library. Call this when you're done using Psyne.
        /// </summary>
        public static void Cleanup()
        {
            lock (_initLock)
            {
                if (_initialized)
                {
                    PsyneNative.psyne_cleanup();
                    _initialized = false;
                }
            }
        }

        /// <summary>
        /// Creates a new channel builder for configuring and creating channels.
        /// </summary>
        /// <returns>A new ChannelBuilder instance.</returns>
        public static ChannelBuilder CreateChannel()
        {
            EnsureInitialized();
            return new ChannelBuilder();
        }

        /// <summary>
        /// Creates a simple memory channel with default settings.
        /// </summary>
        /// <param name="name">The memory channel name.</param>
        /// <param name="bufferSize">The buffer size in bytes (default: 1MB).</param>
        /// <returns>A new Channel instance.</returns>
        public static Channel CreateMemoryChannel(string name, long bufferSize = 1024 * 1024)
        {
            return CreateChannel()
                .Memory(name)
                .WithBufferSize(bufferSize)
                .Build();
        }

        /// <summary>
        /// Creates a simple TCP channel with default settings.
        /// </summary>
        /// <param name="host">The host address.</param>
        /// <param name="port">The port number.</param>
        /// <param name="bufferSize">The buffer size in bytes (default: 1MB).</param>
        /// <returns>A new Channel instance.</returns>
        public static Channel CreateTcpChannel(string host, int port, long bufferSize = 1024 * 1024)
        {
            return CreateChannel()
                .Tcp(host, port)
                .WithBufferSize(bufferSize)
                .Build();
        }

        /// <summary>
        /// Creates a simple Unix socket channel with default settings.
        /// </summary>
        /// <param name="path">The Unix socket path.</param>
        /// <param name="bufferSize">The buffer size in bytes (default: 1MB).</param>
        /// <returns>A new Channel instance.</returns>
        public static Channel CreateUnixSocketChannel(string path, long bufferSize = 1024 * 1024)
        {
            return CreateChannel()
                .UnixSocket(path)
                .WithBufferSize(bufferSize)
                .Build();
        }

        /// <summary>
        /// Gets the description for an error code.
        /// </summary>
        /// <param name="errorCode">The error code.</param>
        /// <returns>A human-readable description of the error.</returns>
        public static string GetErrorDescription(PsyneNative.ErrorCode errorCode)
        {
            var ptr = PsyneNative.psyne_error_string(errorCode);
            return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) ?? errorCode.ToString() : errorCode.ToString();
        }

        private static void EnsureInitialized()
        {
            if (!_initialized)
            {
                Initialize();
            }
        }
    }
}