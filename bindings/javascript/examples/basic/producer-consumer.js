#!/usr/bin/env node

/**
 * @file producer-consumer.js
 * @brief Producer-consumer pattern example using EventEmitter
 * 
 * Demonstrates asynchronous message handling using the EventEmitter
 * pattern for real-time message processing.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

const { Psyne, ChannelMode } = require('psyne');
const { performance } = require('perf_hooks');

async function runProducerConsumer() {
  console.log('=== Psyne Producer-Consumer Example ===\n');

  const psyne = new Psyne({
    enableMetricsByDefault: true
  });

  // Create channels for producer and consumer
  const producerChannel = psyne.createChannel('memory://producer-consumer', {
    mode: ChannelMode.SPSC,
    bufferSize: 2 * 1024 * 1024 // 2MB
  });

  const consumerChannel = psyne.createChannel('memory://producer-consumer', {
    mode: ChannelMode.SPSC
  });

  let messagesProduced = 0;
  let messagesConsumed = 0;
  const totalMessages = 100;
  const startTime = performance.now();

  // Set up consumer
  console.log('Setting up consumer...');
  consumerChannel.on('message', (message) => {
    messagesConsumed++;
    
    if (messagesConsumed <= 5 || messagesConsumed % 20 === 0) {
      console.log(`Consumer received message ${messagesConsumed}:`, {
        type: message.type,
        size: message.size,
        preview: Array.isArray(message.data) 
          ? `[${message.data.slice(0, 3).join(', ')}...]`
          : message.data.toString().substring(0, 30)
      });
    }

    // Check if we're done
    if (messagesConsumed >= totalMessages) {
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      console.log(`\n✓ All ${totalMessages} messages processed in ${duration.toFixed(2)}ms`);
      console.log(`Average rate: ${(totalMessages / duration * 1000).toFixed(2)} messages/second`);
      
      showFinalMetrics();
    }
  });

  consumerChannel.on('error', (error) => {
    console.error('Consumer error:', error.message);
  });

  // Start listening
  consumerChannel.startListening();
  console.log('Consumer started listening\n');

  // Producer function
  async function produceMessages() {
    console.log('Starting producer...');
    
    for (let i = 1; i <= totalMessages; i++) {
      // Generate different types of test data
      let data;
      const messageType = i % 4;
      
      switch (messageType) {
        case 0:
          // Float array
          data = Array.from({ length: 10 }, (_, idx) => i + idx * 0.1);
          break;
        case 1:
          // String message
          data = `Message ${i}: ${new Date().toISOString()}`;
          break;
        case 2:
          // Typed array
          data = new Float32Array([i, i * 2, i * 3, i * 4]);
          break;
        case 3:
          // JSON object
          data = {
            id: i,
            timestamp: Date.now(),
            value: Math.random() * 100,
            metadata: { type: 'test', batch: Math.floor(i / 10) }
          };
          break;
      }

      await producerChannel.send(data);
      messagesProduced++;

      if (i <= 5 || i % 20 === 0) {
        console.log(`Producer sent message ${i}`);
      }

      // Small delay to simulate real-world timing
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 1));
      }
    }
    
    console.log(`Producer finished sending ${messagesProduced} messages\n`);
  }

  function showFinalMetrics() {
    console.log('\n=== Final Metrics ===');
    
    if (producerChannel.hasMetrics) {
      const prodMetrics = producerChannel.getMetrics();
      console.log('Producer channel:');
      console.log(`  Messages sent: ${prodMetrics.messagesSent}`);
      console.log(`  Bytes sent: ${prodMetrics.bytesSent}`);
      console.log(`  Send blocks: ${prodMetrics.sendBlocks}`);
    }
    
    if (consumerChannel.hasMetrics) {
      const consMetrics = consumerChannel.getMetrics();
      console.log('Consumer channel:');
      console.log(`  Messages received: ${consMetrics.messagesReceived}`);
      console.log(`  Bytes received: ${consMetrics.bytesReceived}`);
      console.log(`  Receive blocks: ${consMetrics.receiveBlocks}`);
    }

    // Clean up
    setTimeout(() => {
      consumerChannel.close();
      producerChannel.close();
      console.log('\n✓ Channels closed');
      process.exit(0);
    }, 100);
  }

  // Start the producer
  await produceMessages();
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  process.exit(0);
});

if (require.main === module) {
  runProducerConsumer().catch(console.error);
}