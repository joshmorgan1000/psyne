using System;
using Psyne;

namespace Psyne.Examples
{
    /// <summary>
    /// Demonstrates basic Psyne usage with memory channels.
    /// </summary>
    public class BasicExample
    {
        public static void Run()
        {
            Console.WriteLine("=== Basic Psyne Example ===");

            // Initialize the Psyne library
            Psyne.Initialize();

            try
            {
                // Create a simple memory channel
                using var channel = Psyne.CreateMemoryChannel("basic-example", bufferSize: 1024 * 1024);

                Console.WriteLine($"Created channel with URI: {channel.Uri}");
                Console.WriteLine($"Buffer size: {channel.BufferSize} bytes");

                // Send a string message
                var message = "Hello, Psyne!";
                Console.WriteLine($"Sending: '{message}'");
                channel.Send(message);

                // Receive the message
                using var receivedMessage = channel.Receive(timeoutMs: 1000);
                if (receivedMessage != null)
                {
                    var receivedText = receivedMessage.GetString();
                    Console.WriteLine($"Received: '{receivedText}'");
                    Console.WriteLine($"Message size: {receivedMessage.Size} bytes");
                    Console.WriteLine($"Message type: {receivedMessage.Type}");
                }
                else
                {
                    Console.WriteLine("No message received within timeout");
                }

                // Send binary data
                byte[] binaryData = { 0x01, 0x02, 0x03, 0x04, 0x05 };
                Console.WriteLine("\nSending binary data...");
                channel.Send(binaryData, messageType: 42);

                // Receive binary data
                using var binaryMessage = channel.Receive(timeoutMs: 1000);
                if (binaryMessage != null)
                {
                    var receivedData = binaryMessage.GetData();
                    Console.WriteLine($"Received binary data: [{string.Join(", ", receivedData)}]");
                    Console.WriteLine($"Message type: {binaryMessage.Type}");
                }

                Console.WriteLine("\nBasic example completed successfully!");
            }
            catch (PsyneException ex)
            {
                Console.WriteLine($"Psyne error: {ex.ErrorCode} - {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Unexpected error: {ex.Message}");
            }
            finally
            {
                // Clean up the library
                Psyne.Cleanup();
            }
        }
    }
}