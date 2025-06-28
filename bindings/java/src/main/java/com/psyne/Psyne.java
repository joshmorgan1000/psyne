package com.psyne;

/**
 * Main entry point for the Psyne library.
 * 
 * This class provides static methods for initializing and managing the Psyne library.
 * The library must be initialized before creating any channels.
 * 
 * Example usage:
 * <pre>{@code
 * // Initialize the library
 * Psyne.init();
 * 
 * try {
 *     // Use Psyne channels...
 * } finally {
 *     // Cleanup when done
 *     Psyne.cleanup();
 * }
 * }</pre>
 */
public final class Psyne {
    
    private static boolean initialized = false;
    private static final Object initLock = new Object();
    
    // Private constructor to prevent instantiation
    private Psyne() {
        throw new AssertionError("Psyne class should not be instantiated");
    }
    
    /**
     * Initializes the Psyne library.
     * This method must be called before creating any channels.
     * It is safe to call this method multiple times.
     * 
     * @throws PsyneException if initialization fails
     */
    public static void init() throws PsyneException {
        synchronized (initLock) {
            if (initialized) {
                return;
            }
            
            // Load the native library
            System.loadLibrary("psyne_jni");
            
            // Initialize the native library
            int result = initNative();
            if (result != 0) {
                throw new PsyneException(PsyneException.ErrorCode.fromNativeCode(result),
                    "Failed to initialize Psyne library");
            }
            
            initialized = true;
            
            // Register shutdown hook for automatic cleanup
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    cleanup();
                } catch (Exception e) {
                    // Ignore exceptions during shutdown
                }
            }));
        }
    }
    
    /**
     * Cleans up the Psyne library.
     * This method should be called when the library is no longer needed.
     * It is safe to call this method multiple times.
     */
    public static void cleanup() {
        synchronized (initLock) {
            if (!initialized) {
                return;
            }
            
            cleanupNative();
            initialized = false;
        }
    }
    
    /**
     * Gets the Psyne library version string.
     * 
     * @return The version string
     * @throws IllegalStateException if the library is not initialized
     */
    public static String getVersion() {
        ensureInitialized();
        return getVersionNative();
    }
    
    /**
     * Gets a human-readable description of an error code.
     * 
     * @param errorCode The error code
     * @return The error description
     * @throws IllegalStateException if the library is not initialized
     */
    public static String getErrorString(PsyneException.ErrorCode errorCode) {
        ensureInitialized();
        return getErrorStringNative(errorCode.getCode());
    }
    
    /**
     * Checks if the library is initialized.
     * 
     * @return true if initialized, false otherwise
     */
    public static boolean isInitialized() {
        synchronized (initLock) {
            return initialized;
        }
    }
    
    private static void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("Psyne library not initialized. Call Psyne.init() first.");
        }
    }
    
    // Native methods
    private static native int initNative();
    private static native void cleanupNative();
    private static native String getVersionNative();
    private static native String getErrorStringNative(int errorCode);
}