package com.psyne;

/**
 * Exception thrown by Psyne operations.
 * 
 * This exception wraps the native error codes returned by the Psyne C API
 * and provides meaningful error messages for Java applications.
 */
public class PsyneException extends Exception {
    
    /**
     * The native error code that caused this exception.
     */
    private final ErrorCode errorCode;
    
    /**
     * Constructs a new PsyneException with the specified error code.
     * 
     * @param errorCode The native error code
     */
    public PsyneException(ErrorCode errorCode) {
        super(errorCode.getDescription());
        this.errorCode = errorCode;
    }
    
    /**
     * Constructs a new PsyneException with the specified error code and detail message.
     * 
     * @param errorCode The native error code
     * @param message Additional detail message
     */
    public PsyneException(ErrorCode errorCode, String message) {
        super(errorCode.getDescription() + ": " + message);
        this.errorCode = errorCode;
    }
    
    /**
     * Constructs a new PsyneException with the specified error code and cause.
     * 
     * @param errorCode The native error code
     * @param cause The cause of this exception
     */
    public PsyneException(ErrorCode errorCode, Throwable cause) {
        super(errorCode.getDescription(), cause);
        this.errorCode = errorCode;
    }
    
    /**
     * Gets the error code associated with this exception.
     * 
     * @return The error code
     */
    public ErrorCode getErrorCode() {
        return errorCode;
    }
    
    /**
     * Enumeration of Psyne error codes.
     */
    public enum ErrorCode {
        OK(0, "Success"),
        INVALID_ARGUMENT(-1, "Invalid argument"),
        OUT_OF_MEMORY(-2, "Out of memory"),
        CHANNEL_FULL(-3, "Channel is full"),
        NO_MESSAGE(-4, "No message available"),
        CHANNEL_STOPPED(-5, "Channel has been stopped"),
        UNSUPPORTED(-6, "Operation not supported"),
        IO(-7, "I/O error"),
        TIMEOUT(-8, "Operation timed out"),
        UNKNOWN(-99, "Unknown error");
        
        private final int code;
        private final String description;
        
        ErrorCode(int code, String description) {
            this.code = code;
            this.description = description;
        }
        
        /**
         * Gets the numeric error code.
         * 
         * @return The error code
         */
        public int getCode() {
            return code;
        }
        
        /**
         * Gets the error description.
         * 
         * @return The error description
         */
        public String getDescription() {
            return description;
        }
        
        /**
         * Converts a native error code to an ErrorCode enum.
         * 
         * @param code The native error code
         * @return The corresponding ErrorCode
         */
        public static ErrorCode fromNativeCode(int code) {
            for (ErrorCode ec : values()) {
                if (ec.code == code) {
                    return ec;
                }
            }
            return UNKNOWN;
        }
    }
}