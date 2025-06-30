using System;
using System.Diagnostics;
using System.Text;
using Psyne;

namespace Psyne.Examples
{
    /// <summary>
    /// Demonstrates compression features and performance comparison.
    /// </summary>
    public class CompressionExample
    {
        public static void Run()
        {
            Console.WriteLine("=== Compression Example ===");

            Psyne.Initialize();

            try
            {
                // Test data - repetitive data that compresses well
                var testData = GenerateTestData();
                Console.WriteLine($"Original data size: {testData.Length:N0} bytes");

                // Test different compression algorithms
                TestCompression("No Compression", null, testData);
                TestCompression("LZ4", CompressionConfig.Lz4(), testData);
                TestCompression("Zstd", CompressionConfig.Zstd(), testData);
                TestCompression("Snappy", CompressionConfig.Snappy(), testData);

                // Test custom compression configuration
                Console.WriteLine("\n--- Custom Compression Configurations ---");
                
                var highCompressionZstd = new CompressionConfig
                {
                    Type = CompressionType.Zstd,
                    Level = 15, // High compression
                    MinSizeThreshold = 100,
                    EnableChecksum = true
                };

                var fastLz4 = new CompressionConfig
                {
                    Type = CompressionType.LZ4,
                    Level = 1, // Fast compression
                    MinSizeThreshold = 1000,
                    EnableChecksum = false
                };

                TestCompression("Zstd High Compression", highCompressionZstd, testData);
                TestCompression("LZ4 Fast", fastLz4, testData);

                // Test with different data types
                Console.WriteLine("\n--- Different Data Types ---");
                TestWithDifferentDataTypes();

                Console.WriteLine("\nCompression example completed successfully!");
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

        private static void TestCompression(string name, CompressionConfig? compression, byte[] testData)
        {
            Console.WriteLine($"\n--- {name} ---");

            var channelBuilder = Psyne.CreateChannel()
                .Memory($"compression-test-{name.ToLower().Replace(" ", "-")}")
                .WithBufferSize(10, SizeUnit.MB)
                .WithMetrics();

            if (compression != null)
            {
                channelBuilder = channelBuilder.WithCompression(compression);
            }

            using var channel = channelBuilder.Build();

            var stopwatch = Stopwatch.StartNew();

            // Send data
            channel.Send(testData);
            var sendTime = stopwatch.ElapsedMilliseconds;

            stopwatch.Restart();

            // Receive data
            using var message = channel.Receive(timeoutMs: 5000);
            var receiveTime = stopwatch.ElapsedMilliseconds;

            if (message != null)
            {
                var receivedData = message.GetData();
                var isValid = CompareArrays(testData, receivedData);

                var metrics = channel.GetMetrics();

                Console.WriteLine($"  Data integrity: {(isValid ? "✓ Valid" : "✗ Invalid")}");
                Console.WriteLine($"  Send time: {sendTime} ms");
                Console.WriteLine($"  Receive time: {receiveTime} ms");
                Console.WriteLine($"  Total time: {sendTime + receiveTime} ms");
                Console.WriteLine($"  Bytes sent: {metrics.BytesSent:N0}");
                Console.WriteLine($"  Original size: {testData.Length:N0} bytes");
                
                if (compression != null)
                {
                    var compressionRatio = (double)testData.Length / metrics.BytesSent;
                    Console.WriteLine($"  Compression ratio: {compressionRatio:F2}:1");
                    Console.WriteLine($"  Space saved: {(1 - (double)metrics.BytesSent / testData.Length) * 100:F1}%");
                }
            }
            else
            {
                Console.WriteLine("  ✗ Failed to receive data");
            }
        }

        private static void TestWithDifferentDataTypes()
        {
            var compressionConfig = CompressionConfig.Zstd();

            // Test 1: Text data (highly compressible)
            Console.WriteLine("\nText data (highly compressible):");
            var textData = Encoding.UTF8.GetBytes(new string("Hello World! ", 1000));
            TestSingleMessage(compressionConfig, textData, "text");

            // Test 2: JSON data (moderately compressible)
            Console.WriteLine("\nJSON data (moderately compressible):");
            var jsonData = Encoding.UTF8.GetBytes(GenerateJsonData());
            TestSingleMessage(compressionConfig, jsonData, "json");

            // Test 3: Random data (not compressible)
            Console.WriteLine("\nRandom data (not compressible):");
            var randomData = GenerateRandomData(10000);
            TestSingleMessage(compressionConfig, randomData, "random");

            // Test 4: Small data (below threshold)
            Console.WriteLine("\nSmall data (below compression threshold):");
            var smallData = Encoding.UTF8.GetBytes("Small message");
            TestSingleMessage(compressionConfig, smallData, "small");
        }

        private static void TestSingleMessage(CompressionConfig compression, byte[] data, string dataType)
        {
            using var channel = Psyne.CreateChannel()
                .Memory($"test-{dataType}")
                .WithCompression(compression)
                .WithMetrics()
                .Build();

            channel.Send(data);
            using var message = channel.Receive();

            if (message != null)
            {
                var metrics = channel.GetMetrics();
                var compressionRatio = (double)data.Length / metrics.BytesSent;
                
                Console.WriteLine($"  Original: {data.Length:N0} bytes");
                Console.WriteLine($"  Transmitted: {metrics.BytesSent:N0} bytes");
                Console.WriteLine($"  Ratio: {compressionRatio:F2}:1");
                Console.WriteLine($"  Valid: {CompareArrays(data, message.GetData())}");
            }
        }

        private static byte[] GenerateTestData()
        {
            // Generate repetitive data that compresses well
            var sb = new StringBuilder();
            
            // Add some structure that compresses well
            for (int i = 0; i < 1000; i++)
            {
                sb.AppendLine($"This is line number {i:D4} with some repetitive content that should compress well.");
                sb.AppendLine("AAAAAAAAAABBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEEEEEEEEEFFFFFFFFFF");
                
                if (i % 10 == 0)
                {
                    sb.AppendLine("--- Section Break ---");
                }
            }

            return Encoding.UTF8.GetBytes(sb.ToString());
        }

        private static string GenerateJsonData()
        {
            var sb = new StringBuilder();
            sb.AppendLine("{");
            sb.AppendLine("  \"users\": [");
            
            for (int i = 0; i < 100; i++)
            {
                sb.AppendLine("    {");
                sb.AppendLine($"      \"id\": {i},");
                sb.AppendLine($"      \"name\": \"User {i:D3}\",");
                sb.AppendLine($"      \"email\": \"user{i}@example.com\",");
                sb.AppendLine("      \"active\": true,");
                sb.AppendLine("      \"roles\": [\"user\", \"viewer\"],");
                sb.AppendLine("      \"metadata\": {");
                sb.AppendLine("        \"created\": \"2025-01-01T00:00:00Z\",");
                sb.AppendLine("        \"last_login\": \"2025-06-28T12:00:00Z\"");
                sb.AppendLine("      }");
                sb.Append("    }");
                if (i < 99) sb.AppendLine(",");
                else sb.AppendLine();
            }
            
            sb.AppendLine("  ]");
            sb.AppendLine("}");
            
            return sb.ToString();
        }

        private static byte[] GenerateRandomData(int size)
        {
            var random = new Random(42); // Fixed seed for reproducibility
            var data = new byte[size];
            random.NextBytes(data);
            return data;
        }

        private static bool CompareArrays(byte[] a, byte[] b)
        {
            if (a.Length != b.Length) return false;
            
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i]) return false;
            }
            
            return true;
        }
    }
}