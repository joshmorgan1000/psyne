/**
 * @file channel.test.ts
 * @brief Unit tests for Channel functionality
 * 
 * @author Psyne Contributors
 * @version 0.1.0
 */

import { Channel, ChannelMode, ChannelType, CompressionType } from '../src';
import { PsyneError, ChannelClosedError } from '../src/errors';

// Mock the native module since we don't have it available during testing
jest.mock('../../build/Release/psyne_native.node', () => ({
  createChannel: jest.fn(() => ({
    send: jest.fn(),
    receive: jest.fn(),
    listen: jest.fn(),
    stopListening: jest.fn(),
    stop: jest.fn(),
    isStopped: jest.fn(() => false),
    getUri: jest.fn(() => 'memory://test'),
    getMode: jest.fn(() => ChannelMode.SPSC),
    getType: jest.fn(() => ChannelType.MultiType),
    hasMetrics: jest.fn(() => true),
    getMetrics: jest.fn(() => ({
      messagesSent: 0,
      bytesSent: 0,
      messagesReceived: 0,
      bytesReceived: 0,
      sendBlocks: 0,
      receiveBlocks: 0
    })),
    resetMetrics: jest.fn()
  })),
  getVersion: jest.fn(() => '0.1.0'),
  printBanner: jest.fn(),
  ChannelMode: {
    SPSC: 0,
    SPMC: 1,
    MPSC: 2,
    MPMC: 3
  },
  ChannelType: {
    SingleType: 0,
    MultiType: 1
  },
  CompressionType: {
    None: 0,
    LZ4: 1,
    Zstd: 2,
    Snappy: 3
  }
}));

describe('Channel', () => {
  let channel: Channel;

  beforeEach(() => {
    channel = new Channel('memory://test', {
      mode: ChannelMode.SPSC,
      enableMetrics: true
    });
  });

  afterEach(() => {
    if (!channel.closed) {
      channel.close();
    }
  });

  describe('constructor', () => {
    it('should create a channel with default options', () => {
      const ch = new Channel('memory://default');
      expect(ch.uri).toBe('memory://test');
      expect(ch.mode).toBe(ChannelMode.SPSC);
      expect(ch.type).toBe(ChannelType.MultiType);
      ch.close();
    });

    it('should create a channel with custom options', () => {
      const ch = new Channel('memory://custom', {
        mode: ChannelMode.MPSC,
        type: ChannelType.SingleType,
        bufferSize: 2 * 1024 * 1024,
        enableMetrics: true,
        compression: {
          type: CompressionType.LZ4,
          level: 3
        }
      });

      expect(ch.hasMetrics).toBe(true);
      ch.close();
    });

    it('should throw error for invalid URI', () => {
      expect(() => {
        new Channel('invalid-uri');
      }).toThrow(PsyneError);
    });
  });

  describe('send', () => {
    it('should send array data', async () => {
      await expect(channel.send([1, 2, 3, 4])).resolves.toBeUndefined();
    });

    it('should send string data', async () => {
      await expect(channel.send('hello world')).resolves.toBeUndefined();
    });

    it('should send typed array data', async () => {
      const data = new Float32Array([1.0, 2.0, 3.0]);
      await expect(channel.send(data)).resolves.toBeUndefined();
    });

    it('should send buffer data', async () => {
      const data = Buffer.from('binary data');
      await expect(channel.send(data)).resolves.toBeUndefined();
    });

    it('should throw error when channel is closed', async () => {
      channel.close();
      await expect(channel.send([1, 2, 3])).rejects.toThrow(ChannelClosedError);
    });
  });

  describe('receive', () => {
    it('should receive message with timeout', async () => {
      // Mock successful receive
      const mockMessage = {
        type: 'floatVector',
        typeId: 1,
        size: 16,
        data: [1, 2, 3, 4]
      };

      const mockNativeChannel = (channel as any).nativeChannel;
      mockNativeChannel.receive.mockResolvedValue(mockMessage);

      const message = await channel.receive(1000);
      expect(message).toBeDefined();
      expect(message?.type).toBe('floatVector');
      expect(message?.data).toEqual([1, 2, 3, 4]);
    });

    it('should return null when no message available', async () => {
      const mockNativeChannel = (channel as any).nativeChannel;
      mockNativeChannel.receive.mockResolvedValue(null);

      const message = await channel.receive(0);
      expect(message).toBeNull();
    });

    it('should throw error when channel is closed', async () => {
      channel.close();
      await expect(channel.receive()).rejects.toThrow(ChannelClosedError);
    });
  });

  describe('listening', () => {
    it('should start and stop listening', () => {
      expect(channel.isListening).toBe(false);
      
      channel.startListening();
      expect(channel.isListening).toBe(true);
      
      channel.stopListening();
      expect(channel.isListening).toBe(false);
    });

    it('should emit message events when listening', (done) => {
      const testMessage = {
        type: 'test',
        typeId: 999,
        size: 4,
        data: 'test'
      };

      channel.on('message', (message) => {
        expect(message.type).toBe('test');
        expect(message.data).toBe('test');
        done();
      });

      channel.startListening();
      
      // Simulate message callback
      const mockCallback = (channel as any).nativeChannel.listen.mock.calls[0][0];
      mockCallback(testMessage);
    });

    it('should handle listener errors', (done) => {
      const testError = new Error('Listener error');

      channel.on('error', (error) => {
        expect(error).toBeInstanceOf(PsyneError);
        done();
      });

      channel.startListening();
      
      // Simulate error callback
      const mockCallback = (channel as any).nativeChannel.listen.mock.calls[0][0];
      mockCallback(null, testError);
    });
  });

  describe('metrics', () => {
    it('should get metrics when enabled', () => {
      const metrics = channel.getMetrics();
      expect(metrics).toBeDefined();
      expect(typeof metrics?.messagesSent).toBe('number');
      expect(typeof metrics?.bytesSent).toBe('number');
    });

    it('should reset metrics', () => {
      expect(() => channel.resetMetrics()).not.toThrow();
    });

    it('should return null when metrics disabled', () => {
      const ch = new Channel('memory://no-metrics', { enableMetrics: false });
      const mockNativeChannel = (ch as any).nativeChannel;
      mockNativeChannel.hasMetrics.mockReturnValue(false);
      
      expect(ch.getMetrics()).toBeNull();
      ch.close();
    });
  });

  describe('builder pattern', () => {
    it('should create channel using builder', () => {
      const ch = Channel.builder()
        .uri('memory://builder-test')
        .mode(ChannelMode.MPSC)
        .bufferSize(1024 * 1024)
        .enableMetrics()
        .compression({
          type: CompressionType.LZ4,
          level: 1
        })
        .build();

      expect(ch).toBeInstanceOf(Channel);
      ch.close();
    });

    it('should create reliable channel using builder', () => {
      const ch = Channel.builder()
        .uri('memory://reliable-test')
        .buildReliable({
          enableAcknowledgments: true,
          maxRetries: 5
        });

      expect(ch).toBeInstanceOf(Channel);
      ch.close();
    });

    it('should throw error if URI not set in builder', () => {
      expect(() => {
        Channel.builder().build();
      }).toThrow(PsyneError);
    });
  });

  describe('properties', () => {
    it('should return correct properties', () => {
      expect(channel.uri).toBe('memory://test');
      expect(channel.mode).toBe(ChannelMode.SPSC);
      expect(channel.type).toBe(ChannelType.MultiType);
      expect(channel.hasMetrics).toBe(true);
      expect(channel.closed).toBe(false);
      expect(channel.isListening).toBe(false);
    });

    it('should report closed status after close', () => {
      expect(channel.closed).toBe(false);
      channel.close();
      expect(channel.closed).toBe(true);
    });
  });

  describe('events', () => {
    it('should emit sent event after sending', async () => {
      let sentEvent: any = null;
      
      channel.on('sent', (event) => {
        sentEvent = event;
      });

      await channel.send([1, 2, 3]);
      
      expect(sentEvent).toBeDefined();
      expect(sentEvent.data).toEqual([1, 2, 3]);
    });

    it('should emit closed event when closed', (done) => {
      channel.on('closed', () => {
        done();
      });

      channel.close();
    });
  });
});

describe('Channel static methods', () => {
  it('should create reliable channel', () => {
    const channel = Channel.createReliable('memory://reliable', {
      mode: ChannelMode.SPSC
    }, {
      enableAcknowledgments: true,
      maxRetries: 3
    });

    expect(channel).toBeInstanceOf(Channel);
    channel.close();
  });
});