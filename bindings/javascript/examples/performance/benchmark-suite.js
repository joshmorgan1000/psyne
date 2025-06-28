#!/usr/bin/env node

/**
 * @file benchmark-suite.js
 * @brief Comprehensive performance benchmarking suite
 * 
 * Measures channel performance across different configurations,
 * message sizes, and usage patterns to help optimize applications.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

const { Psyne, ChannelMode, ChannelType, CompressionType } = require('psyne');
const { formatBytes, formatDuration, measureTime } = require('psyne/utils');
const { performance } = require('perf_hooks');

class BenchmarkSuite {
  constructor() {
    this.psyne = new Psyne({ enableMetricsByDefault: true });
    this.results = [];
  }

  /**
   * Run a single benchmark test
   */
  async runBenchmark(config) {
    const {
      name,
      channelConfig,
      messageSize,
      messageCount,
      pattern = 'sequential'
    } = config;

    console.log(`\nRunning benchmark: ${name}`);
    console.log(`  Message size: ${formatBytes(messageSize)}`);
    console.log(`  Message count: ${messageCount}`);
    console.log(`  Pattern: ${pattern}`);

    // Create channel
    const channel = this.psyne.createChannel(`memory://benchmark-${Date.now()}`, channelConfig);
    
    // Generate test data
    const testData = this.generateTestData(messageSize, pattern);
    
    // Warmup
    await this.warmup(channel, testData, Math.min(10, messageCount));
    
    // Measure send performance
    const sendResults = await this.measureSendPerformance(channel, testData, messageCount);
    
    // Measure receive performance
    const receiveResults = await this.measureReceivePerformance(channel, messageCount);
    
    // Calculate metrics
    const metrics = channel.getMetrics();
    const totalBytes = messageSize * messageCount;
    const totalTime = sendResults.totalTime + receiveResults.totalTime;
    
    const result = {
      name,
      config: channelConfig,
      messageSize,
      messageCount,
      pattern,
      sendTime: sendResults.totalTime,
      receiveTime: receiveResults.totalTime,
      totalTime,
      throughputMbps: (totalBytes / (1024 * 1024)) / (totalTime / 1000),
      messagesPerSecond: messageCount / (totalTime / 1000),
      avgSendLatency: sendResults.avgLatency,
      avgReceiveLatency: receiveResults.avgLatency,
      p99SendLatency: sendResults.p99Latency,
      p99ReceiveLatency: receiveResults.p99Latency,
      metrics
    };

    this.results.push(result);
    
    // Display results
    console.log(`  Results:`);
    console.log(`    Throughput: ${result.throughputMbps.toFixed(2)} MB/s`);
    console.log(`    Message rate: ${result.messagesPerSecond.toFixed(0)} msg/s`);
    console.log(`    Avg send latency: ${result.avgSendLatency.toFixed(2)}μs`);
    console.log(`    P99 send latency: ${result.p99SendLatency.toFixed(2)}μs`);
    console.log(`    Total time: ${formatDuration(totalTime)}`);

    channel.close();
    return result;
  }

  /**
   * Generate test data based on pattern
   */
  generateTestData(size, pattern) {
    const elementCount = Math.floor(size / 4); // Assuming Float32Array
    
    switch (pattern) {
      case 'sequential':
        return Float32Array.from({ length: elementCount }, (_, i) => i);
      case 'random':
        return Float32Array.from({ length: elementCount }, () => Math.random());
      case 'constant':
        return new Float32Array(elementCount).fill(42.0);
      case 'sine':
        return Float32Array.from({ length: elementCount }, (_, i) => Math.sin(i * 0.1));
      default:
        return new Float32Array(elementCount);
    }
  }

  /**
   * Warmup the channel
   */
  async warmup(channel, testData, count) {
    for (let i = 0; i < count; i++) {
      await channel.send(testData);
      await channel.receive();
    }
    channel.resetMetrics();
  }

  /**
   * Measure send performance with detailed timing
   */
  async measureSendPerformance(channel, testData, count) {
    const latencies = [];
    const startTime = performance.now();

    for (let i = 0; i < count; i++) {
      const { timeMs } = await measureTime(async () => {
        await channel.send(testData);
      });
      latencies.push(timeMs * 1000); // Convert to microseconds
    }

    const endTime = performance.now();
    const totalTime = endTime - startTime;

    return {
      totalTime,
      avgLatency: latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length,
      p99Latency: this.calculatePercentile(latencies, 0.99),
      latencies
    };
  }

  /**
   * Measure receive performance
   */
  async measureReceivePerformance(channel, count) {
    const latencies = [];
    const startTime = performance.now();

    for (let i = 0; i < count; i++) {
      const { timeMs } = await measureTime(async () => {
        await channel.receive();
      });
      latencies.push(timeMs * 1000); // Convert to microseconds
    }

    const endTime = performance.now();
    const totalTime = endTime - startTime;

    return {
      totalTime,
      avgLatency: latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length,
      p99Latency: this.calculatePercentile(latencies, 0.99),
      latencies
    };
  }

  /**
   * Calculate percentile from array of values
   */
  calculatePercentile(values, percentile) {
    const sorted = values.slice().sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * percentile) - 1;
    return sorted[index];
  }

  /**
   * Run comprehensive benchmark suite
   */
  async runFullSuite() {
    console.log('=== Psyne Performance Benchmark Suite ===\n');
    console.log(`Psyne version: ${Psyne.getVersion()}`);
    console.log(`Node.js version: ${process.version}`);
    console.log(`Platform: ${process.platform} ${process.arch}`);
    console.log(`Memory: ${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(1)} MB used\n`);

    const benchmarks = [
      // Basic performance tests
      {
        name: 'Small Messages (SPSC)',
        channelConfig: { mode: ChannelMode.SPSC, bufferSize: 1024 * 1024 },
        messageSize: 64,
        messageCount: 10000,
        pattern: 'sequential'
      },
      {
        name: 'Medium Messages (SPSC)',
        channelConfig: { mode: ChannelMode.SPSC, bufferSize: 4 * 1024 * 1024 },
        messageSize: 4096,
        messageCount: 5000,
        pattern: 'random'
      },
      {
        name: 'Large Messages (SPSC)',
        channelConfig: { mode: ChannelMode.SPSC, bufferSize: 16 * 1024 * 1024 },
        messageSize: 64 * 1024,
        messageCount: 1000,
        pattern: 'sine'
      },

      // Multi-threading mode comparisons
      {
        name: 'Medium Messages (MPSC)',
        channelConfig: { mode: ChannelMode.MPSC, bufferSize: 4 * 1024 * 1024 },
        messageSize: 4096,
        messageCount: 5000,
        pattern: 'random'
      },
      {
        name: 'Medium Messages (SPMC)',
        channelConfig: { mode: ChannelMode.SPMC, bufferSize: 4 * 1024 * 1024 },
        messageSize: 4096,
        messageCount: 5000,
        pattern: 'random'
      },
      {
        name: 'Medium Messages (MPMC)',
        channelConfig: { mode: ChannelMode.MPMC, bufferSize: 4 * 1024 * 1024 },
        messageSize: 4096,
        messageCount: 5000,
        pattern: 'random'
      },

      // Compression benchmarks
      {
        name: 'Compressed (LZ4)',
        channelConfig: {
          mode: ChannelMode.SPSC,
          bufferSize: 4 * 1024 * 1024,
          compression: { type: CompressionType.LZ4, level: 1 }
        },
        messageSize: 8192,
        messageCount: 2000,
        pattern: 'constant'
      },
      {
        name: 'Compressed (Zstd)',
        channelConfig: {
          mode: ChannelMode.SPSC,
          bufferSize: 4 * 1024 * 1024,
          compression: { type: CompressionType.Zstd, level: 3 }
        },
        messageSize: 8192,
        messageCount: 2000,
        pattern: 'constant'
      },

      // High-throughput test
      {
        name: 'High Throughput Test',
        channelConfig: { 
          mode: ChannelMode.SPSC, 
          bufferSize: 32 * 1024 * 1024,
          type: ChannelType.SingleType
        },
        messageSize: 1024,
        messageCount: 20000,
        pattern: 'sequential'
      }
    ];

    // Run all benchmarks
    for (const benchmark of benchmarks) {
      try {
        await this.runBenchmark(benchmark);
        
        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Force garbage collection if available
        if (global.gc) {
          global.gc();
        }
      } catch (error) {
        console.error(`Benchmark "${benchmark.name}" failed:`, error.message);
      }
    }

    // Generate summary report
    this.generateReport();
  }

  /**
   * Generate comprehensive performance report
   */
  generateReport() {
    console.log('\n' + '='.repeat(80));
    console.log('PERFORMANCE BENCHMARK REPORT');
    console.log('='.repeat(80));

    // Summary table
    console.log('\nSUMMARY TABLE:');
    console.log('-'.repeat(120));
    console.log('| Test Name                    | Throughput | Msg/sec  | Avg Latency | P99 Latency | Total Time |');
    console.log('-'.repeat(120));

    for (const result of this.results) {
      const name = result.name.padEnd(28);
      const throughput = `${result.throughputMbps.toFixed(1)} MB/s`.padEnd(10);
      const msgRate = `${result.messagesPerSecond.toFixed(0)}`.padEnd(8);
      const avgLat = `${result.avgSendLatency.toFixed(1)}μs`.padEnd(11);
      const p99Lat = `${result.p99SendLatency.toFixed(1)}μs`.padEnd(11);
      const totalTime = formatDuration(result.totalTime).padEnd(10);
      
      console.log(`| ${name} | ${throughput} | ${msgRate} | ${avgLat} | ${p99Lat} | ${totalTime} |`);
    }
    console.log('-'.repeat(120));

    // Analysis
    console.log('\nPERFORMANCE ANALYSIS:');
    
    // Find best performers
    const bestThroughput = this.results.reduce((best, current) => 
      current.throughputMbps > best.throughputMbps ? current : best
    );
    
    const bestLatency = this.results.reduce((best, current) => 
      current.avgSendLatency < best.avgSendLatency ? current : best
    );
    
    const bestMessageRate = this.results.reduce((best, current) => 
      current.messagesPerSecond > best.messagesPerSecond ? current : best
    );

    console.log(`• Highest throughput: ${bestThroughput.name} (${bestThroughput.throughputMbps.toFixed(2)} MB/s)`);
    console.log(`• Lowest latency: ${bestLatency.name} (${bestLatency.avgSendLatency.toFixed(2)}μs)`);
    console.log(`• Highest message rate: ${bestMessageRate.name} (${bestMessageRate.messagesPerSecond.toFixed(0)} msg/s)`);

    // Mode comparison
    console.log('\nCHANNEL MODE COMPARISON:');
    const modeResults = this.groupByMode();
    for (const [mode, results] of Object.entries(modeResults)) {
      const avgThroughput = results.reduce((sum, r) => sum + r.throughputMbps, 0) / results.length;
      const avgLatency = results.reduce((sum, r) => sum + r.avgSendLatency, 0) / results.length;
      console.log(`• ${mode}: ${avgThroughput.toFixed(2)} MB/s avg, ${avgLatency.toFixed(2)}μs avg latency`);
    }

    // Recommendations
    console.log('\nRECOMMENDations:');
    console.log('• Use SPSC mode for highest performance in single-threaded scenarios');
    console.log('• Consider compression for large, repetitive data (>4KB)');
    console.log('• Adjust buffer size based on message size and frequency');
    console.log('• Monitor P99 latency for latency-sensitive applications');
    
    console.log('\n✓ Benchmark suite completed');
  }

  /**
   * Group results by channel mode
   */
  groupByMode() {
    const grouped = {};
    for (const result of this.results) {
      const mode = ChannelMode[result.config.mode];
      if (!grouped[mode]) {
        grouped[mode] = [];
      }
      grouped[mode].push(result);
    }
    return grouped;
  }
}

// Memory stress test
async function memoryStressTest() {
  console.log('\n=== Memory Stress Test ===');
  
  const psyne = new Psyne({ enableMetricsByDefault: true });
  const channel = psyne.createChannel('memory://stress-test', {
    mode: ChannelMode.SPSC,
    bufferSize: 64 * 1024 * 1024 // 64MB
  });

  const initialMemory = process.memoryUsage();
  console.log('Initial memory usage:');
  console.log(`  Heap used: ${formatBytes(initialMemory.heapUsed)}`);
  console.log(`  External: ${formatBytes(initialMemory.external)}`);

  // Send large amounts of data
  const messageSize = 1024 * 1024; // 1MB
  const messageCount = 100;
  const testData = new Float32Array(messageSize / 4);

  console.log(`\nSending ${messageCount} messages of ${formatBytes(messageSize)} each...`);
  
  for (let i = 0; i < messageCount; i++) {
    testData.fill(i);
    await channel.send(testData);
    
    // Receive immediately to prevent buffer overflow
    await channel.receive();
    
    if (i % 10 === 0) {
      const currentMemory = process.memoryUsage();
      console.log(`  Progress: ${i}/${messageCount}, Heap: ${formatBytes(currentMemory.heapUsed)}`);
    }
  }

  const finalMemory = process.memoryUsage();
  console.log('\nFinal memory usage:');
  console.log(`  Heap used: ${formatBytes(finalMemory.heapUsed)}`);
  console.log(`  Memory increase: ${formatBytes(finalMemory.heapUsed - initialMemory.heapUsed)}`);

  const metrics = channel.getMetrics();
  console.log(`\nData processed: ${formatBytes(metrics.bytesSent)} sent, ${formatBytes(metrics.bytesReceived)} received`);
  
  channel.close();
  console.log('✓ Memory stress test completed');
}

// Main execution
async function main() {
  try {
    const suite = new BenchmarkSuite();
    await suite.runFullSuite();
    await memoryStressTest();
  } catch (error) {
    console.error('Benchmark suite failed:', error);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down benchmark...');
  process.exit(0);
});

if (require.main === module) {
  main();
}