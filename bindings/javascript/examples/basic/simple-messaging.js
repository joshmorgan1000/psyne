#!/usr/bin/env node

/**
 * @file simple-messaging.js
 * @brief Basic messaging example using Psyne JavaScript bindings
 * 
 * Demonstrates the simplest use case: creating a channel and
 * sending/receiving messages in the same process.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

const { Psyne, ChannelMode } = require('psyne');

async function main() {
  console.log('=== Psyne Simple Messaging Example ===\n');
  
  // Print library information
  console.log(`Psyne version: ${Psyne.getVersion()}`);
  Psyne.printBanner();
  console.log();

  try {
    // Create a Psyne instance
    const psyne = new Psyne({
      enableMetricsByDefault: true
    });

    // Create an in-memory channel
    console.log('Creating channel...');
    const channel = psyne.createChannel('memory://simple-example', {
      mode: ChannelMode.SPSC,
      bufferSize: 1024 * 1024 // 1MB
    });

    console.log(`Channel created: ${channel.uri}`);
    console.log(`Channel mode: ${channel.mode}`);
    console.log(`Metrics enabled: ${channel.hasMetrics}\n`);

    // Send some different types of data
    console.log('Sending messages...');
    
    // Send an array of numbers
    await channel.send([1.0, 2.0, 3.0, 4.0, 5.0]);
    console.log('✓ Sent number array');

    // Send a string (will be converted to bytes)
    await channel.send('Hello, Psyne!');
    console.log('✓ Sent string');

    // Send a typed array
    const typedData = new Float32Array([10.5, 20.3, 30.7]);
    await channel.send(typedData);
    console.log('✓ Sent Float32Array');

    // Send a buffer
    const buffer = Buffer.from('Binary data example');
    await channel.send(buffer);
    console.log('✓ Sent Buffer\n');

    // Receive messages
    console.log('Receiving messages...');
    
    for (let i = 0; i < 4; i++) {
      const message = await channel.receive();
      if (message) {
        console.log(`Message ${i + 1}:`, {
          type: message.type,
          size: message.size,
          dataPreview: Array.isArray(message.data) 
            ? message.data.slice(0, 5) 
            : message.data.toString().substring(0, 50)
        });
      } else {
        console.log(`No message received on attempt ${i + 1}`);
      }
    }

    // Show metrics
    if (channel.hasMetrics) {
      console.log('\nChannel metrics:');
      const metrics = channel.getMetrics();
      console.log(`  Messages sent: ${metrics.messagesSent}`);
      console.log(`  Bytes sent: ${metrics.bytesSent}`);
      console.log(`  Messages received: ${metrics.messagesReceived}`);
      console.log(`  Bytes received: ${metrics.bytesReceived}`);
    }

    // Clean up
    channel.close();
    console.log('\n✓ Channel closed successfully');

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  process.exit(0);
});

if (require.main === module) {
  main().catch(console.error);
}