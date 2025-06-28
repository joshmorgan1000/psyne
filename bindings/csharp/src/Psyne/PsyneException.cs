using System;
using Psyne.Native;

namespace Psyne
{
    /// <summary>
    /// Exception thrown by Psyne operations.
    /// </summary>
    public class PsyneException : Exception
    {
        /// <summary>
        /// Gets the error code associated with this exception.
        /// </summary>
        public PsyneNative.ErrorCode ErrorCode { get; }

        /// <summary>
        /// Initializes a new instance of the PsyneException class.
        /// </summary>
        /// <param name="errorCode">The error code.</param>
        public PsyneException(PsyneNative.ErrorCode errorCode)
            : base(GetErrorMessage(errorCode))
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Initializes a new instance of the PsyneException class.
        /// </summary>
        /// <param name="errorCode">The error code.</param>
        /// <param name="message">The error message.</param>
        public PsyneException(PsyneNative.ErrorCode errorCode, string message)
            : base(message)
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Initializes a new instance of the PsyneException class.
        /// </summary>
        /// <param name="errorCode">The error code.</param>
        /// <param name="message">The error message.</param>
        /// <param name="innerException">The inner exception.</param>
        public PsyneException(PsyneNative.ErrorCode errorCode, string message, Exception innerException)
            : base(message, innerException)
        {
            ErrorCode = errorCode;
        }

        /// <summary>
        /// Throws an exception if the error code indicates a failure.
        /// </summary>
        /// <param name="errorCode">The error code to check.</param>
        internal static void ThrowIfError(PsyneNative.ErrorCode errorCode)
        {
            if (errorCode != PsyneNative.ErrorCode.Ok)
            {
                throw new PsyneException(errorCode);
            }
        }

        private static string GetErrorMessage(PsyneNative.ErrorCode errorCode)
        {
            var ptr = PsyneNative.psyne_error_string(errorCode);
            return ptr != IntPtr.Zero ? System.Runtime.InteropServices.Marshal.PtrToStringUTF8(ptr) ?? errorCode.ToString() : errorCode.ToString();
        }
    }
}