using System;
using Psyne;

namespace Psyne.Examples
{
    /// <summary>
    /// Demonstrates the fluent builder API for channel creation and configuration.
    /// </summary>
    public class BuilderPatternExample
    {
        public static void Run()
        {
            Console.WriteLine("=== Builder Pattern Example ===");

            Psyne.Initialize();

            try
            {
                // Example 1: Basic memory channel with custom buffer size
                Console.WriteLine("1. Creating memory channel with builder...");
                using var memoryChannel = Psyne.CreateChannel()
                    .Memory("builder-example")
                    .WithBufferSize(5, SizeUnit.MB)
                    .SingleProducerSingleConsumer()
                    .SingleType()
                    .WithMetrics()
                    .Build();

                Console.WriteLine($"Memory channel created: {memoryChannel.Uri}");
                Console.WriteLine($"Buffer size: {memoryChannel.BufferSize:N0} bytes");

                // Test the memory channel
                memoryChannel.Send("Memory channel test message");
                using var msg1 = memoryChannel.Receive();
                Console.WriteLine($"Received: {msg1?.GetString()}");

                // Example 2: TCP channel with compression
                Console.WriteLine("\n2. Creating TCP channel with compression...");
                using var tcpChannel = Psyne.CreateChannel()
                    .Tcp("localhost", 8080)
                    .WithBufferSize(2, SizeUnit.MB)
                    .MultipleProducerMultipleConsumer()
                    .MultiType()
                    .WithLz4Compression()
                    .WithMetrics()
                    .Build();

                Console.WriteLine($"TCP channel created: {tcpChannel.Uri}");

                // Example 3: Unix socket channel with custom compression
                Console.WriteLine("\n3. Creating Unix socket channel with custom compression...");
                var customCompression = new CompressionConfig
                {
                    Type = CompressionType.Zstd,
                    Level = 6,
                    MinSizeThreshold = 512,
                    EnableChecksum = true
                };

                using var unixChannel = Psyne.CreateChannel()
                    .UnixSocket("/tmp/psyne-builder-example.sock")
                    .WithBufferSize(1024, SizeUnit.KB)
                    .MultipleProducerSingleConsumer()
                    .WithCompression(customCompression)
                    .Build();

                Console.WriteLine($"Unix channel created: {unixChannel.Uri}");

                // Example 4: UDP multicast channel with Snappy compression
                Console.WriteLine("\n4. Creating UDP multicast channel...");
                using var udpChannel = Psyne.CreateChannel()
                    .UdpMulticast("239.1.1.1", 8080)
                    .WithBufferSize(512, SizeUnit.KB)
                    .SingleProducerMultipleConsumer()
                    .WithSnappyCompression()
                    .Build();

                Console.WriteLine($"UDP channel created: {udpChannel.Uri}");

                // Example 5: Demonstrate different compression methods
                Console.WriteLine("\n5. Testing different compression methods...");

                var compressionConfigs = new[]
                {
                    ("LZ4", CompressionConfig.Lz4()),
                    ("Zstd", CompressionConfig.Zstd()),
                    ("Snappy", CompressionConfig.Snappy())
                };

                foreach (var (name, config) in compressionConfigs)
                {
                    using var compressedChannel = Psyne.CreateChannel()
                        .Memory($"compression-test-{name.ToLower()}")
                        .WithBufferSize(1, SizeUnit.MB)
                        .WithCompression(config)
                        .WithMetrics()
                        .Build();

                    Console.WriteLine($"Created {name} compressed channel: {compressedChannel.Uri}");

                    // Send a large message to test compression
                    var largeMessage = new string('A', 10000); // 10KB of 'A's
                    compressedChannel.Send(largeMessage);

                    using var received = compressedChannel.Receive();
                    if (received != null && received.GetString() == largeMessage)
                    {
                        Console.WriteLine($"  {name} compression test passed");
                        
                        var metrics = compressedChannel.GetMetrics();
                        Console.WriteLine($"  Messages sent: {metrics.MessagesSent}");
                        Console.WriteLine($"  Bytes sent: {metrics.BytesSent:N0}");
                    }
                }

                // Example 6: Method chaining variations
                Console.WriteLine("\n6. Method chaining variations...");

                // Variation 1: Explicit methods
                using var explicit1 = Psyne.CreateChannel()
                    .Memory("explicit-1")
                    .SingleProducerSingleConsumer()
                    .SingleType()
                    .Build();

                // Variation 2: Shorthand methods
                using var explicit2 = Psyne.CreateChannel()
                    .Memory("explicit-2")
                    .WithMode(ChannelMode.SingleProducerSingleConsumer)
                    .WithType(ChannelType.Single)
                    .Build();

                Console.WriteLine("Both variations created successfully");

                Console.WriteLine("\nBuilder pattern example completed successfully!");
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
                Psyne.Cleanup();
            }
        }
    }
}