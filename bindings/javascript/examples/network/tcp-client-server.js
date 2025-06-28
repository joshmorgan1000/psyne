#!/usr/bin/env node

/**
 * @file tcp-client-server.js
 * @brief TCP client-server communication example
 * 
 * Demonstrates how to create TCP channels for network communication
 * between separate processes, with proper error handling and reconnection.
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

const { Psyne, ChannelMode, createChannel } = require('psyne');
const { formatBytes, retryWithBackoff } = require('psyne/utils');

const DEFAULT_PORT = 8080;
const DEFAULT_HOST = 'localhost';

class TCPServer {
  constructor(port = DEFAULT_PORT) {
    this.port = port;
    this.psyne = new Psyne({ enableMetricsByDefault: true });
    this.server = null;
    this.clients = new Map();
    this.messageCount = 0;
  }

  async start() {
    console.log(`Starting TCP server on port ${this.port}...`);

    try {
      // Create server channel
      this.server = createChannel(`tcp://:${this.port}`)
        .mode(ChannelMode.MPSC) // Multiple clients, single server
        .bufferSize(2 * 1024 * 1024)
        .enableMetrics()
        .build();

      console.log(`✓ Server listening on port ${this.port}`);

      // Handle incoming messages
      this.server.on('message', (message) => {
        this.handleMessage(message);
      });

      this.server.on('connected', () => {
        console.log('New client connected');
      });

      this.server.on('disconnected', () => {
        console.log('Client disconnected');
      });

      this.server.on('error', (error) => {
        console.error('Server error:', error.message);
      });

      // Start listening
      this.server.startListening();

      // Start periodic status updates
      this.startStatusUpdates();

    } catch (error) {
      console.error('Failed to start server:', error.message);
      throw error;
    }
  }

  handleMessage(message) {
    this.messageCount++;
    
    console.log(`Received message ${this.messageCount}:`, {
      type: message.type,
      size: formatBytes(message.size),
      timestamp: message.timestamp?.toISOString()
    });

    // Echo message back with server timestamp
    const response = {
      original: message,
      serverTimestamp: new Date().toISOString(),
      messageId: this.messageCount,
      echo: true
    };

    // Send response back to client
    this.server.send(response).catch(error => {
      console.error('Failed to send response:', error.message);
    });
  }

  startStatusUpdates() {
    setInterval(() => {
      if (this.server && this.server.hasMetrics) {
        const metrics = this.server.getMetrics();
        console.log(`[Status] Messages: ${metrics.messagesReceived}/${metrics.messagesSent}, ` +
                   `Bytes: ${formatBytes(metrics.bytesReceived)}/${formatBytes(metrics.bytesSent)}`);
      }
    }, 10000); // Every 10 seconds
  }

  async stop() {
    console.log('Stopping server...');
    if (this.server) {
      this.server.close();
    }
    console.log('✓ Server stopped');
  }
}

class TCPClient {
  constructor(host = DEFAULT_HOST, port = DEFAULT_PORT, clientId = 'client1') {
    this.host = host;
    this.port = port;
    this.clientId = clientId;
    this.psyne = new Psyne({ enableMetricsByDefault: true });
    this.client = null;
    this.connected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  async connect() {
    console.log(`[${this.clientId}] Connecting to ${this.host}:${this.port}...`);

    try {
      // Create client channel with reliability features
      this.client = createChannel(`tcp://${this.host}:${this.port}`)
        .mode(ChannelMode.SPSC)
        .bufferSize(1024 * 1024)
        .enableMetrics()
        .buildReliable({
          enableAcknowledgments: true,
          enableRetries: true,
          maxRetries: 3,
          ackTimeout: 5000
        });

      // Set up event handlers
      this.client.on('connected', () => {
        console.log(`[${this.clientId}] ✓ Connected to server`);
        this.connected = true;
        this.reconnectAttempts = 0;
      });

      this.client.on('disconnected', () => {
        console.log(`[${this.clientId}] Disconnected from server`);
        this.connected = false;
        this.handleDisconnection();
      });

      this.client.on('message', (message) => {
        this.handleServerResponse(message);
      });

      this.client.on('error', (error) => {
        console.error(`[${this.clientId}] Error:`, error.message);
      });

      // Start listening for responses
      this.client.startListening();

    } catch (error) {
      console.error(`[${this.clientId}] Connection failed:`, error.message);
      throw error;
    }
  }

  async handleDisconnection() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`[${this.clientId}] Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      try {
        await retryWithBackoff(
          () => this.connect(),
          3, // max retries for each attempt
          1000, // 1 second base delay
          2 // exponential backoff
        );
      } catch (error) {
        console.error(`[${this.clientId}] Reconnection failed:`, error.message);
      }
    } else {
      console.error(`[${this.clientId}] Max reconnection attempts reached`);
    }
  }

  handleServerResponse(message) {
    console.log(`[${this.clientId}] Server response:`, {
      messageId: message.data?.messageId,
      serverTimestamp: message.data?.serverTimestamp,
      isEcho: message.data?.echo,
      roundTripTime: message.timestamp ? 
        Date.now() - new Date(message.data?.original?.timestamp || 0).getTime() : 'unknown'
    });
  }

  async sendMessage(data, type = 'data') {
    if (!this.connected) {
      throw new Error('Not connected to server');
    }

    const message = {
      clientId: this.clientId,
      type,
      data,
      timestamp: new Date().toISOString(),
      messageNumber: Date.now()
    };

    try {
      await this.client.send(message);
      console.log(`[${this.clientId}] Sent ${type} message`);
    } catch (error) {
      console.error(`[${this.clientId}] Failed to send message:`, error.message);
      throw error;
    }
  }

  async runDemo() {
    console.log(`[${this.clientId}] Starting demo...`);

    // Send different types of data
    const testData = [
      { type: 'greeting', data: `Hello from ${this.clientId}!` },
      { type: 'numbers', data: [1, 2, 3, 4, 5] },
      { type: 'floats', data: new Float32Array([3.14, 2.718, 1.414]) },
      { type: 'object', data: { name: this.clientId, value: Math.random(), timestamp: Date.now() } },
      { type: 'binary', data: Buffer.from('Binary data from client') }
    ];

    for (const { type, data } of testData) {
      try {
        await this.sendMessage(data, type);
        // Wait for response before sending next message
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        console.error(`[${this.clientId}] Demo message failed:`, error.message);
      }
    }

    // Show final metrics
    if (this.client && this.client.hasMetrics) {
      const metrics = this.client.getMetrics();
      console.log(`[${this.clientId}] Final metrics:`, {
        sent: metrics.messagesSent,
        received: metrics.messagesReceived,
        bytesSent: formatBytes(metrics.bytesSent),
        bytesReceived: formatBytes(metrics.bytesReceived)
      });
    }
  }

  async disconnect() {
    console.log(`[${this.clientId}] Disconnecting...`);
    if (this.client) {
      this.client.close();
    }
    this.connected = false;
    console.log(`[${this.clientId}] ✓ Disconnected`);
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  const mode = args[0] || 'demo';
  const port = parseInt(args[1]) || DEFAULT_PORT;
  const host = args[2] || DEFAULT_HOST;

  console.log('=== Psyne TCP Client-Server Example ===\n');

  switch (mode) {
    case 'server':
      await runServer(port);
      break;
    case 'client':
      await runClient(host, port);
      break;
    case 'demo':
      await runDemo(port);
      break;
    default:
      console.log('Usage: node tcp-client-server.js [server|client|demo] [port] [host]');
      console.log('  server: Run as server');
      console.log('  client: Run as client');
      console.log('  demo: Run integrated demo (default)');
      break;
  }
}

async function runServer(port) {
  const server = new TCPServer(port);
  
  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\nShutting down server...');
    await server.stop();
    process.exit(0);
  });

  await server.start();
  
  // Keep server running
  console.log('Server running. Press Ctrl+C to stop.');
  await new Promise(() => {}); // Run forever
}

async function runClient(host, port) {
  const client = new TCPClient(host, port, 'standalone-client');
  
  try {
    await client.connect();
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for connection
    
    if (client.connected) {
      await client.runDemo();
    } else {
      console.error('Failed to connect to server');
    }
  } finally {
    await client.disconnect();
  }
}

async function runDemo(port) {
  const server = new TCPServer(port);
  
  try {
    // Start server
    await server.start();
    await new Promise(resolve => setTimeout(resolve, 1000)); // Let server start

    // Create multiple clients
    const clients = [
      new TCPClient(DEFAULT_HOST, port, 'demo-client-1'),
      new TCPClient(DEFAULT_HOST, port, 'demo-client-2')
    ];

    // Connect all clients
    console.log('\nConnecting clients...');
    await Promise.all(clients.map(client => client.connect()));
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for connections

    // Run demos in parallel
    console.log('\nRunning client demos...');
    await Promise.all(clients.map(client => client.runDemo()));

    // Wait a bit for final messages
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Disconnect clients
    console.log('\nDisconnecting clients...');
    await Promise.all(clients.map(client => client.disconnect()));

  } catch (error) {
    console.error('Demo failed:', error.message);
  } finally {
    // Stop server
    await server.stop();
  }

  console.log('\n✓ Demo completed');
}

if (require.main === module) {
  main().catch(error => {
    console.error('Application failed:', error);
    process.exit(1);
  });
}