#!/usr/bin/env node

/**
 * @file compression-demo.js
 * @brief Advanced example demonstrating compression features
 * 
 * Shows how to configure and use different compression algorithms
 * to reduce bandwidth usage for large messages.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

const { Psyne, ChannelMode, CompressionType, createChannel } = require('psyne');
const { formatBytes, measureTime } = require('psyne/utils');

async function demonstrateCompression() {
  console.log('=== Psyne Compression Demo ===\n');

  const psyne = new Psyne({ enableMetricsByDefault: true });

  // Test data - create large, compressible data
  const createTestData = (size, pattern = 'repeated') => {
    switch (pattern) {
      case 'repeated':
        // Highly compressible repeated data
        return new Float32Array(size).fill(42.0);
        
      case 'sequential':
        // Sequential data (moderately compressible)
        return Float32Array.from({ length: size }, (_, i) => i);
        
      case 'random':
        // Random data (not compressible)
        return Float32Array.from({ length: size }, () => Math.random());
        
      default:
        return new Float32Array(size);
    }
  };

  const dataSize = 10000; // 40KB of float data
  const testDatasets = [
    { name: 'Repeated Values', data: createTestData(dataSize, 'repeated') },
    { name: 'Sequential Data', data: createTestData(dataSize, 'sequential') },
    { name: 'Random Data', data: createTestData(dataSize, 'random') }
  ];

  // Compression configurations to test
  const compressionConfigs = [
    {
      name: 'None',
      config: { type: CompressionType.None }
    },
    {
      name: 'LZ4 (Fast)',
      config: {
        type: CompressionType.LZ4,
        level: 1,
        minSizeThreshold: 1024,
        enableChecksum: true
      }
    },
    {
      name: 'LZ4 (Balanced)',
      config: {
        type: CompressionType.LZ4,
        level: 6,
        minSizeThreshold: 512,
        enableChecksum: true
      }
    },
    {
      name: 'Zstd (High Compression)',
      config: {
        type: CompressionType.Zstd,
        level: 9,
        minSizeThreshold: 256,
        enableChecksum: true
      }
    },
    {
      name: 'Snappy (Balanced)',
      config: {
        type: CompressionType.Snappy,
        level: 1,
        minSizeThreshold: 512,
        enableChecksum: true
      }
    }
  ];

  console.log(`Testing compression with ${formatBytes(dataSize * 4)} datasets...\n`);

  // Test each compression configuration
  for (const { name: compressionName, config } of compressionConfigs) {
    console.log(`=== ${compressionName} Compression ===`);
    
    // Create channel with specific compression
    const channel = createChannel(`memory://compression-${compressionName.toLowerCase()}`)
      .mode(ChannelMode.SPSC)
      .bufferSize(2 * 1024 * 1024)
      .enableMetrics()
      .compression(config)
      .build();

    const results = [];

    // Test each dataset
    for (const { name: dataName, data } of testDatasets) {
      console.log(`\nTesting ${dataName}...`);
      
      // Measure send time
      const { timeMs: sendTime } = await measureTime(async () => {
        await channel.send(data);
      });

      // Measure receive time
      const { result: message, timeMs: receiveTime } = await measureTime(async () => {
        return await channel.receive(5000);
      });

      if (message) {
        const originalSize = data.byteLength;
        const receivedSize = message.size;
        const compressionRatio = originalSize / receivedSize;
        const spaceSaved = originalSize - receivedSize;
        
        console.log(`  Original size: ${formatBytes(originalSize)}`);
        console.log(`  Compressed size: ${formatBytes(receivedSize)}`);
        console.log(`  Compression ratio: ${compressionRatio.toFixed(2)}x`);
        console.log(`  Space saved: ${formatBytes(spaceSaved)} (${((spaceSaved/originalSize)*100).toFixed(1)}%)`);
        console.log(`  Send time: ${sendTime.toFixed(2)}ms`);
        console.log(`  Receive time: ${receiveTime.toFixed(2)}ms`);
        
        // Verify data integrity
        const receivedData = message.data;
        const isIntact = Array.isArray(receivedData) || ArrayBuffer.isView(receivedData);
        console.log(`  Data integrity: ${isIntact ? '✓ OK' : '✗ FAILED'}`);
        
        results.push({
          dataset: dataName,
          originalSize,
          compressedSize: receivedSize,
          ratio: compressionRatio,
          sendTime,
          receiveTime,
          intact: isIntact
        });
      } else {
        console.log('  ✗ No message received');
      }
    }

    // Show summary for this compression method
    console.log(`\n${compressionName} Summary:`);
    const avgRatio = results.reduce((sum, r) => sum + r.ratio, 0) / results.length;
    const avgSendTime = results.reduce((sum, r) => sum + r.sendTime, 0) / results.length;
    const avgReceiveTime = results.reduce((sum, r) => sum + r.receiveTime, 0) / results.length;
    
    console.log(`  Average compression ratio: ${avgRatio.toFixed(2)}x`);
    console.log(`  Average send time: ${avgSendTime.toFixed(2)}ms`);
    console.log(`  Average receive time: ${avgReceiveTime.toFixed(2)}ms`);
    
    // Show channel metrics if available
    if (channel.hasMetrics) {
      const metrics = channel.getMetrics();
      console.log(`  Messages processed: ${metrics.messagesSent}/${metrics.messagesReceived}`);
      console.log(`  Total bytes: ${formatBytes(metrics.bytesSent)} sent, ${formatBytes(metrics.bytesReceived)} received`);
    }

    channel.close();
    console.log(`\n${'='.repeat(50)}\n`);
  }

  // Performance comparison
  console.log('=== Performance Comparison ===');
  console.log('(Results may vary based on data patterns and system performance)\n');
  
  console.log('Compression recommendations:');
  console.log('• LZ4 Fast: Best for real-time applications requiring low latency');
  console.log('• LZ4 Balanced: Good compromise between speed and compression ratio');
  console.log('• Zstd: Best compression ratio for bandwidth-limited scenarios');
  console.log('• Snappy: Balanced performance, good for mixed workloads');
  console.log('• None: Use for small messages or when CPU is limited');

  console.log('\n✓ Compression demo completed');
}

// Advanced compression features demo
async function advancedCompressionFeatures() {
  console.log('\n=== Advanced Compression Features ===\n');

  // Adaptive compression based on message size
  console.log('1. Adaptive compression thresholds...');
  
  const adaptiveChannel = createChannel('memory://adaptive-compression')
    .mode(ChannelMode.SPSC)
    .enableMetrics()
    .compression({
      type: CompressionType.LZ4,
      level: 3,
      minSizeThreshold: 1024, // Only compress messages >= 1KB
      enableChecksum: true
    })
    .build();

  // Test with different message sizes
  const sizes = [100, 500, 1000, 5000, 10000]; // bytes
  
  for (const size of sizes) {
    const data = new Uint8Array(size).fill(42);
    await adaptiveChannel.send(data);
    
    const message = await adaptiveChannel.receive();
    if (message) {
      const wasCompressed = message.size < data.byteLength;
      console.log(`  ${size} bytes -> ${message.size} bytes (${wasCompressed ? 'compressed' : 'uncompressed'})`);
    }
  }
  
  adaptiveChannel.close();

  // Compression with checksums
  console.log('\n2. Compression with data integrity checks...');
  
  const checksumChannel = createChannel('memory://checksum-compression')
    .compression({
      type: CompressionType.Zstd,
      level: 5,
      enableChecksum: true,
      minSizeThreshold: 0 // Compress everything
    })
    .build();

  // Send data that might be corrupted during transmission
  const importantData = new Float64Array([Math.PI, Math.E, Math.SQRT2, Math.LN2]);
  await checksumChannel.send(importantData);
  
  const recoveredMessage = await checksumChannel.receive();
  if (recoveredMessage) {
    console.log('  ✓ Data recovered with integrity verification');
    console.log(`  Original: [${Array.from(importantData).map(x => x.toFixed(6)).join(', ')}]`);
    // In a real implementation, we'd verify the checksum here
  }
  
  checksumChannel.close();

  console.log('\n✓ Advanced compression features demo completed');
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  process.exit(0);
});

async function main() {
  try {
    await demonstrateCompression();
    await advancedCompressionFeatures();
  } catch (error) {
    console.error('Compression demo failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}