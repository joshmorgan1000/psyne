using System;
using System.Threading;
using System.Threading.Tasks;
using Psyne;

namespace Psyne.Examples
{
    /// <summary>
    /// Demonstrates async/await support and concurrent operations.
    /// </summary>
    public class AsyncExample
    {
        public static async Task RunAsync()
        {
            Console.WriteLine("=== Async Example ===");

            Psyne.Initialize();

            try
            {
                using var channel = Psyne.CreateChannel()
                    .Memory("async-example")
                    .WithBufferSize(1, SizeUnit.MB)
                    .MultipleProducerMultipleConsumer()
                    .WithMetrics()
                    .Build();

                Console.WriteLine($"Created channel: {channel.Uri}");

                // Example 1: Basic async receive
                await BasicAsyncReceiveExample(channel);

                // Example 2: Producer-Consumer pattern
                await ProducerConsumerExample(channel);

                // Example 3: Multiple concurrent operations
                await ConcurrentOperationsExample(channel);

                // Example 4: Cancellation support
                await CancellationExample(channel);

                // Example 5: Event-driven operations
                await EventDrivenExample(channel);

                Console.WriteLine("\nAsync example completed successfully!");
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

        private static async Task BasicAsyncReceiveExample(Channel channel)
        {
            Console.WriteLine("\n--- Basic Async Receive ---");

            // Start a task to send a message after a delay
            _ = Task.Run(async () =>
            {
                await Task.Delay(1000);
                channel.Send("Async message after 1 second");
                Console.WriteLine("Message sent asynchronously");
            });

            // Receive the message asynchronously
            var message = await channel.ReceiveAsync();
            if (message != null)
            {
                Console.WriteLine($"Received async: {message.GetString()}");
                message.Dispose();
            }
        }

        private static async Task ProducerConsumerExample(Channel channel)
        {
            Console.WriteLine("\n--- Producer-Consumer Pattern ---");

            const int messageCount = 10;
            var cts = new CancellationTokenSource();

            // Producer task
            var producerTask = Task.Run(async () =>
            {
                for (int i = 0; i < messageCount; i++)
                {
                    await Task.Delay(100); // Simulate work
                    channel.Send($"Message {i + 1}");
                    Console.WriteLine($"Produced: Message {i + 1}");
                }
                
                // Send a stop signal
                channel.Send("STOP");
                Console.WriteLine("Producer finished");
            });

            // Consumer task
            var consumerTask = Task.Run(async () =>
            {
                int received = 0;
                while (!cts.Token.IsCancellationRequested)
                {
                    var message = await channel.ReceiveAsync(cts.Token);
                    if (message != null)
                    {
                        var content = message.GetString();
                        message.Dispose();

                        if (content == "STOP")
                        {
                            Console.WriteLine("Consumer received stop signal");
                            break;
                        }

                        Console.WriteLine($"Consumed: {content}");
                        received++;
                    }
                }
                Console.WriteLine($"Consumer finished, received {received} messages");
            });

            // Wait for both tasks to complete
            await Task.WhenAll(producerTask, consumerTask);
            cts.Cancel();
        }

        private static async Task ConcurrentOperationsExample(Channel channel)
        {
            Console.WriteLine("\n--- Concurrent Operations ---");

            const int concurrentTasks = 5;
            const int messagesPerTask = 3;

            // Create multiple producer tasks
            var producerTasks = new Task[concurrentTasks];
            for (int taskId = 0; taskId < concurrentTasks; taskId++)
            {
                int capturedTaskId = taskId;
                producerTasks[taskId] = Task.Run(async () =>
                {
                    for (int i = 0; i < messagesPerTask; i++)
                    {
                        await Task.Delay(Random.Shared.Next(50, 200));
                        var message = $"Task-{capturedTaskId}-Message-{i}";
                        channel.Send(message, messageType: (uint)capturedTaskId);
                        Console.WriteLine($"[Task {capturedTaskId}] Sent: {message}");
                    }
                });
            }

            // Create multiple consumer tasks
            var consumerTasks = new Task[2];
            var receivedCount = 0;
            
            for (int consumerId = 0; consumerId < 2; consumerId++)
            {
                int capturedConsumerId = consumerId;
                consumerTasks[consumerId] = Task.Run(async () =>
                {
                    while (Interlocked.Read(ref receivedCount) < concurrentTasks * messagesPerTask)
                    {
                        var message = await channel.ReceiveAsync();
                        if (message != null)
                        {
                            var content = message.GetString();
                            Console.WriteLine($"[Consumer {capturedConsumerId}] Received: {content} (Type: {message.Type})");
                            message.Dispose();
                            
                            Interlocked.Increment(ref receivedCount);
                        }
                        else
                        {
                            await Task.Delay(10);
                        }
                    }
                    Console.WriteLine($"Consumer {capturedConsumerId} finished");
                });
            }

            // Wait for all producers to finish
            await Task.WhenAll(producerTasks);
            Console.WriteLine("All producers finished");

            // Wait for all consumers to finish
            await Task.WhenAll(consumerTasks);
            Console.WriteLine("All consumers finished");

            // Display metrics
            var metrics = channel.GetMetrics();
            Console.WriteLine($"Total messages sent: {metrics.MessagesSent}");
            Console.WriteLine($"Total messages received: {metrics.MessagesReceived}");
        }

        private static async Task CancellationExample(Channel channel)
        {
            Console.WriteLine("\n--- Cancellation Example ---");

            // Test cancellation during receive
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));

            try
            {
                Console.WriteLine("Starting receive with 2-second cancellation timeout...");
                var message = await channel.ReceiveAsync(cts.Token);
                
                if (message != null)
                {
                    Console.WriteLine($"Unexpected message received: {message.GetString()}");
                    message.Dispose();
                }
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("Receive operation was cancelled as expected");
            }

            // Test successful receive before cancellation
            var quickCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            
            // Send a message quickly
            _ = Task.Run(async () =>
            {
                await Task.Delay(500);
                channel.Send("Quick message before cancellation");
            });

            try
            {
                var message = await channel.ReceiveAsync(quickCts.Token);
                if (message != null)
                {
                    Console.WriteLine($"Received before cancellation: {message.GetString()}");
                    message.Dispose();
                }
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("Unexpected cancellation");
            }
        }

        private static async Task EventDrivenExample(Channel channel)
        {
            Console.WriteLine("\n--- Event-Driven Example ---");

            var messageReceived = new TaskCompletionSource<bool>();
            var receivedMessages = 0;

            // Set up event-driven receive callback
            channel.SetReceiveCallback((message, messageType) =>
            {
                var content = message.GetString();
                Console.WriteLine($"Callback received: {content} (Type: {messageType})");
                message.Dispose();

                Interlocked.Increment(ref receivedMessages);
                if (receivedMessages >= 3)
                {
                    messageReceived.TrySetResult(true);
                }
            });

            Console.WriteLine("Set up receive callback, sending messages...");

            // Send messages that will trigger the callback
            for (int i = 0; i < 3; i++)
            {
                await Task.Delay(500);
                channel.Send($"Callback message {i + 1}", messageType: (uint)(i + 1));
                Console.WriteLine($"Sent callback message {i + 1}");
            }

            // Wait for all messages to be processed by the callback
            await messageReceived.Task;
            Console.WriteLine("All callback messages processed");

            // Clear the callback
            channel.SetReceiveCallback(null);
            Console.WriteLine("Callback cleared");
        }
    }
}