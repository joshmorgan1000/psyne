using System;
using System.Runtime.InteropServices;

namespace Psyne.Native
{
    /// <summary>
    /// Native P/Invoke declarations for the Psyne C API.
    /// </summary>
    internal static class PsyneNative
    {
        private const string LibraryName = "psyne";

        #region Enums

        /// <summary>
        /// Error codes returned by Psyne functions.
        /// </summary>
        public enum ErrorCode : int
        {
            Ok = 0,
            InvalidArgument = -1,
            OutOfMemory = -2,
            ChannelFull = -3,
            NoMessage = -4,
            ChannelStopped = -5,
            Unsupported = -6,
            IO = -7,
            Timeout = -8,
            Unknown = -99
        }

        /// <summary>
        /// Channel synchronization modes.
        /// </summary>
        public enum ChannelMode : int
        {
            /// <summary>Single Producer, Single Consumer</summary>
            SPSC = 0,
            /// <summary>Single Producer, Multiple Consumer</summary>
            SPMC = 1,
            /// <summary>Multiple Producer, Single Consumer</summary>
            MPSC = 2,
            /// <summary>Multiple Producer, Multiple Consumer</summary>
            MPMC = 3
        }

        /// <summary>
        /// Channel type modes.
        /// </summary>
        public enum ChannelType : int
        {
            /// <summary>Single message type</summary>
            Single = 0,
            /// <summary>Multiple message types</summary>
            Multi = 1
        }

        /// <summary>
        /// Compression types supported by Psyne.
        /// </summary>
        public enum CompressionType : int
        {
            None = 0,
            LZ4 = 1,
            Zstd = 2,
            Snappy = 3
        }

        #endregion

        #region Structs

        /// <summary>
        /// Channel metrics information.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct Metrics
        {
            public ulong MessagesSent;
            public ulong BytesSent;
            public ulong MessagesReceived;
            public ulong BytesReceived;
            public ulong SendBlocks;
            public ulong ReceiveBlocks;
        }

        /// <summary>
        /// Compression configuration.
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct CompressionConfig
        {
            public CompressionType Type;
            public int Level;
            public UIntPtr MinSizeThreshold;
            [MarshalAs(UnmanagedType.I1)]
            public bool EnableChecksum;
        }

        #endregion

        #region Delegates

        /// <summary>
        /// Callback for message reception.
        /// </summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ReceiveCallback(IntPtr message, uint type, IntPtr userData);

        #endregion

        #region Library Management

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_init();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void psyne_cleanup();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr psyne_version();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr psyne_error_string(ErrorCode error);

        #endregion

        #region Channel Management

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_create(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string uri,
            UIntPtr bufferSize,
            ChannelMode mode,
            ChannelType type,
            out IntPtr channel);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_create_compressed(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string uri,
            UIntPtr bufferSize,
            ChannelMode mode,
            ChannelType type,
            ref CompressionConfig compression,
            out IntPtr channel);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_create_compressed(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string uri,
            UIntPtr bufferSize,
            ChannelMode mode,
            ChannelType type,
            IntPtr compression, // For null pointer
            out IntPtr channel);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void psyne_channel_destroy(IntPtr channel);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_stop(IntPtr channel);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_is_stopped(IntPtr channel, out bool stopped);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_get_uri(
            IntPtr channel,
            IntPtr uri,
            UIntPtr uriSize);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_get_metrics(
            IntPtr channel,
            out Metrics metrics);

        #endregion

        #region Message Operations

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_message_reserve(
            IntPtr channel,
            UIntPtr size,
            out IntPtr message);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_message_get_data(
            IntPtr message,
            out IntPtr data,
            out UIntPtr size);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_message_send(IntPtr message, uint type);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void psyne_message_cancel(IntPtr message);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_message_receive(
            IntPtr channel,
            out IntPtr message,
            out uint type);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_message_receive_timeout(
            IntPtr channel,
            uint timeoutMs,
            out IntPtr message,
            out uint type);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void psyne_message_release(IntPtr message);

        #endregion

        #region Utility Functions

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_send_data(
            IntPtr channel,
            IntPtr data,
            UIntPtr size,
            uint type);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_receive_data(
            IntPtr channel,
            IntPtr buffer,
            UIntPtr bufferSize,
            out UIntPtr receivedSize,
            out uint type,
            uint timeoutMs);

        #endregion

        #region Advanced Features

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_enable_metrics(
            IntPtr channel,
            [MarshalAs(UnmanagedType.I1)] bool enable);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_reset_metrics(IntPtr channel);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_get_buffer_size(
            IntPtr channel,
            out UIntPtr size);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ErrorCode psyne_channel_set_receive_callback(
            IntPtr channel,
            ReceiveCallback? callback,
            IntPtr userData);

        #endregion
    }
}