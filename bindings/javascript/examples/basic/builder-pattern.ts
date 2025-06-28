#!/usr/bin/env npx ts-node

/**
 * @file builder-pattern.ts
 * @brief TypeScript example demonstrating the fluent builder pattern
 * 
 * Shows how to use the ChannelBuilder for creating channels with
 * complex configurations using method chaining.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

import { 
  Channel, 
  ChannelMode, 
  ChannelType, 
  CompressionType,
  FloatVector,
  DoubleMatrix,
  ByteVector
} from 'psyne';

async function demonstrateBuilderPattern(): Promise<void> {
  console.log('=== Psyne Builder Pattern Example ===\n');

  try {
    // Example 1: Simple channel with method chaining
    console.log('Creating simple channel with builder...');
    const simpleChannel = Channel.builder()
      .uri('memory://builder-simple')
      .mode(ChannelMode.SPSC)
      .bufferSize(512 * 1024)
      .enableMetrics()
      .build();

    console.log(`✓ Simple channel created: ${simpleChannel.uri}\n`);

    // Example 2: Advanced channel with compression
    console.log('Creating advanced channel with compression...');
    const advancedChannel = Channel.builder()
      .uri('memory://builder-advanced')
      .mode(ChannelMode.MPSC)
      .type(ChannelType.MultiType)
      .bufferSize(2 * 1024 * 1024)
      .enableMetrics()
      .compression({
        type: CompressionType.LZ4,
        level: 3,
        minSizeThreshold: 256,
        enableChecksum: true
      })
      .build();

    console.log(`✓ Advanced channel created: ${advancedChannel.uri}`);
    console.log(`  Mode: ${ChannelMode[advancedChannel.mode]}`);
    console.log(`  Type: ${ChannelType[advancedChannel.type]}`);
    console.log(`  Metrics: ${advancedChannel.hasMetrics}\n`);

    // Example 3: Reliable channel with error recovery
    console.log('Creating reliable channel...');
    const reliableChannel = Channel.builder()
      .uri('memory://builder-reliable')
      .mode(ChannelMode.SPMC)
      .bufferSize(1024 * 1024)
      .enableMetrics()
      .buildReliable({
        enableAcknowledgments: true,
        enableRetries: true,
        maxRetries: 5,
        ackTimeout: 2000,
        retryDelay: 200
      });

    console.log(`✓ Reliable channel created: ${reliableChannel.uri}\n`);

    // Demonstrate sending typed messages
    console.log('Sending typed messages...');

    // Send FloatVector
    const floatData = new FloatVector([1.1, 2.2, 3.3, 4.4, 5.5]);
    await advancedChannel.send(floatData);
    console.log('✓ Sent FloatVector');

    // Send DoubleMatrix
    const matrixData = new DoubleMatrix(2, 3, [1, 2, 3, 4, 5, 6]);
    await advancedChannel.send(matrixData);
    console.log('✓ Sent DoubleMatrix');

    // Send ByteVector
    const byteData = new ByteVector(Buffer.from('TypeScript rocks!'));
    await advancedChannel.send(byteData);
    console.log('✓ Sent ByteVector\n');

    // Receive and process messages
    console.log('Receiving messages...');
    
    for (let i = 0; i < 3; i++) {
      const message = await advancedChannel.receive(1000); // 1 second timeout
      
      if (message) {
        console.log(`Message ${i + 1}:`, {
          type: message.type,
          typeId: message.typeId,
          size: message.size,
          timestamp: message.timestamp
        });

        // Type-specific processing
        switch (message.type) {
          case 'floatVector':
            console.log(`  Float data preview: [${message.data.slice(0, 3).join(', ')}...]`);
            break;
          case 'doubleMatrix':
            console.log(`  Matrix dimensions: ${message.data.rows}x${message.data.cols}`);
            break;
          case 'byteVector':
            console.log(`  Text content: "${message.data.toString()}"`);
            break;
        }
      } else {
        console.log(`No message received on attempt ${i + 1}`);
      }
    }

    // Performance monitoring example
    console.log('\nSetting up performance monitoring...');
    
    const { createPerformanceMonitor } = await import('psyne/utils');
    const monitor = createPerformanceMonitor(advancedChannel, 500);
    
    monitor.onUpdate((summary) => {
      console.log(`Performance update - Rate: ${summary.messageRate.toFixed(1)} msg/s, ` +
                  `Throughput: ${summary.throughputMbps.toFixed(2)} MB/s`);
    });
    
    monitor.start();

    // Send a burst of messages for performance demo
    console.log('Sending burst of messages for performance demo...');
    const burstSize = 50;
    const testData = new Float32Array(100);
    
    for (let i = 0; i < burstSize; i++) {
      testData.fill(i);
      await advancedChannel.send(testData);
    }
    
    // Wait a bit for metrics to update
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    monitor.stop();

    // Show final metrics
    console.log('\n=== Final Metrics ===');
    
    const channels = [
      { name: 'Simple', channel: simpleChannel },
      { name: 'Advanced', channel: advancedChannel },
      { name: 'Reliable', channel: reliableChannel }
    ];

    for (const { name, channel } of channels) {
      if (channel.hasMetrics) {
        const metrics = channel.getMetrics();
        console.log(`${name} Channel:`, {
          sent: metrics?.messagesSent || 0,
          received: metrics?.messagesReceived || 0,
          bytes: metrics?.bytesSent || 0
        });
      }
    }

    // Clean up
    simpleChannel.close();
    advancedChannel.close();
    reliableChannel.close();
    
    console.log('\n✓ All channels closed successfully');

  } catch (error) {
    console.error('Error in builder pattern demo:', error);
    throw error;
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  process.exit(0);
});

if (require.main === module) {
  demonstrateBuilderPattern().catch((error) => {
    console.error('Demo failed:', error);
    process.exit(1);
  });
}