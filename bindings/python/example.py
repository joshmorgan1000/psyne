#!/usr/bin/env python3
"""
Example usage of Psyne Python bindings

This script demonstrates basic usage of the Psyne Python bindings,
including creating channels, sending/receiving messages with NumPy arrays.
"""

import numpy as np
import time
import threading
import sys
import os

# Add the build directory to path to find the psyne module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

try:
    import psyne
except ImportError as e:
    print(f"Failed to import psyne: {e}")
    print("Make sure to build the Python bindings first:")
    print("  cd python && python setup.py build_ext --inplace")
    sys.exit(1)

def test_ipc_channel():
    """Test IPC channel with NumPy data"""
    print("=== Testing IPC Channel ===")
    
    # Create IPC channel
    channel_name = "test_python_ipc"
    channel = psyne.create_ipc_channel(channel_name, buffer_size=1024*1024)
    
    print(f"Created IPC channel: {channel}")
    
    # Create test data
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    print(f"Original data: {test_data}")
    
    # Send data
    msg = channel.create_float_vector(len(test_data))
    msg.from_numpy(test_data)
    msg.send()
    print("Sent message")
    
    # Receive data
    received = channel.receive_float_vector()
    if received is not None:
        print(f"Received data: {received}")
        print(f"Data matches: {np.array_equal(test_data, received)}")
    else:
        print("No data received")
    
    # Show metrics
    metrics = channel.get_metrics()
    print(f"Channel metrics: {metrics}")
    
    channel.stop()
    print("Channel stopped\n")

def test_tcp_channel():
    """Test TCP channel with server/client"""
    print("=== Testing TCP Channel ===")
    
    port = 8888
    received_data = None
    server_ready = threading.Event()
    
    def server_thread():
        try:
            server = psyne.create_tcp_server(port)
            print(f"TCP server created on port {port}")
            server_ready.set()
            
            # Wait for client connection and receive data
            time.sleep(0.1)  # Give client time to connect
            
            nonlocal received_data
            received_data = server.receive_float_vector()
            if received_data is not None:
                print(f"Server received: {received_data}")
            else:
                print("Server received no data")
                
            server.stop()
        except Exception as e:
            print(f"Server error: {e}")
            server_ready.set()
    
    def client_thread():
        try:
            server_ready.wait()  # Wait for server to start
            time.sleep(0.1)  # Additional delay for server setup
            
            client = psyne.create_tcp_client("localhost", port)
            print("TCP client connected")
            
            # Send test data
            test_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
            print(f"Client sending: {test_data}")
            
            msg = client.create_float_vector(len(test_data))
            msg.from_numpy(test_data)
            msg.send()
            print("Client sent message")
            
            time.sleep(0.1)  # Give server time to receive
            client.stop()
        except Exception as e:
            print(f"Client error: {e}")
    
    # Start server and client threads
    server = threading.Thread(target=server_thread)
    client = threading.Thread(target=client_thread)
    
    server.start()
    client.start()
    
    server.join()
    client.join()
    
    if received_data is not None:
        print("TCP test completed successfully")
    else:
        print("TCP test failed - no data received")
    
    print()

def test_compression():
    """Test compression features"""
    print("=== Testing Compression ===")
    
    try:
        # Create compression config
        comp_config = psyne.CompressionConfig()
        comp_config.type = psyne.CompressionType.LZ4
        comp_config.level = 1
        comp_config.min_size_threshold = 100
        comp_config.enabled = True
        
        print(f"Compression config: type={comp_config.type}, level={comp_config.level}")
        
        # Create multicast publisher with compression
        publisher = psyne.create_multicast_publisher("239.255.0.100", 9999, 
                                                    buffer_size=1024*1024, 
                                                    compression_config=comp_config)
        print("Created multicast publisher with compression")
        
        # Create large test data that should trigger compression
        large_data = np.ones(1000, dtype=np.float32) * 42.0  # Highly compressible
        print(f"Created test data: {len(large_data)} floats (all 42.0)")
        
        # Send compressed data
        msg = publisher.create_float_vector(len(large_data))
        msg.from_numpy(large_data)
        msg.send()
        print("Sent compressed message")
        
        publisher.stop()
        print("Compression test completed\n")
        
    except Exception as e:
        print(f"Compression test error: {e}\n")

def test_metrics():
    """Test metrics functionality"""
    print("=== Testing Metrics ===")
    
    channel = psyne.create_ipc_channel("metrics_test")
    
    # Send some messages
    for i in range(5):
        data = np.array([float(i)], dtype=np.float32)
        msg = channel.create_float_vector(1)
        msg.from_numpy(data)
        msg.send()
    
    # Get metrics
    metrics = channel.get_metrics()
    print(f"Messages sent: {metrics.messages_sent}")
    print(f"Bytes sent: {metrics.bytes_sent}")
    print(f"Messages received: {metrics.messages_received}")
    print(f"Bytes received: {metrics.bytes_received}")
    
    # Reset metrics
    channel.reset_metrics()
    new_metrics = channel.get_metrics()
    print(f"After reset - Messages sent: {new_metrics.messages_sent}")
    
    channel.stop()
    print("Metrics test completed\n")

def main():
    """Run all tests"""
    print("Psyne Python Bindings Example")
    print("=" * 35)
    
    # Print version info
    print(f"Psyne version: {psyne.version()}")
    psyne.print_banner()
    print()
    
    try:
        test_ipc_channel()
        test_tcp_channel()
        test_compression()
        test_metrics()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())