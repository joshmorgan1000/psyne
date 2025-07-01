# Psyne Benchmarks

This directory contains comprehensive benchmarks for the Psyne messaging library, designed to test various messaging patterns and saturate hardware capabilities.

## Benchmarks

### 1. Messaging Patterns Benchmark (`messaging_patterns_benchmark`)

Tests various messaging patterns to understand performance characteristics:

- **Throughput Test**: Maximum messages per second with small messages
- **Bandwidth Test**: Maximum data transfer rate with large messages  
- **Burst Test**: Performance under bursty traffic patterns
- **Stress Test**: Multi-threaded producer/consumer scenarios

Usage:
```bash
./messaging_patterns_benchmark --transport <ipc|tcp|spsc> --pattern <throughput|bandwidth|burst|stress> [options]

Options:
  --size <bytes>        Message size (default: 64)
  --count <num>         Number of messages (default: 1000000)
  --producers <num>     Number of producers (default: 1)
  --consumers <num>     Number of consumers (default: 1)
  --burst <size>        Burst size (default: 1000)
  --affinity           Enable CPU affinity
  --host <hostname>     TCP server host (default: localhost)
  --port <port>        TCP server port (default: 9999)
```

### 2. Network Saturation Test (`network_saturation_test`)

Specialized benchmark for testing TCP performance between machines:

- Multiple parallel connections
- Bidirectional traffic testing
- TCP tuning options
- Detailed latency percentiles

Usage:
```bash
# On server machine:
./network_saturation_test server --port 10000 --connections 8

# On client machine:
./network_saturation_test client --host <server-ip> --port 10000 --connections 8 --size 4096

Options:
  --connections <num>   Number of parallel connections
  --size <bytes>        Message payload size
  --count <num>         Messages per connection
  --bidirectional      Enable echo mode for round-trip testing
  --nodelay            Disable Nagle's algorithm
  --sendbuf <KB>       TCP send buffer size
  --recvbuf <KB>       TCP receive buffer size
```

### 3. Automated Benchmark Suite (`run_benchmarks.sh`)

Shell script that runs a comprehensive suite of benchmarks:

```bash
./run_benchmarks.sh
```

This script will:
- Detect system configuration
- Set CPU governor to performance mode (Linux)
- Run throughput, bandwidth, burst, and stress tests
- Test with various message sizes
- Generate a summary report

## Hardware Saturation Guidelines

### CPU Saturation

To saturate CPU cores:

1. **Single Core**: Use SPSC with CPU affinity
   ```bash
   ./messaging_patterns_benchmark --transport spsc --pattern throughput --affinity
   ```

2. **All Cores**: Use multi-threaded stress test
   ```bash
   ./messaging_patterns_benchmark --transport ipc --pattern stress --producers 4 --consumers 4 --affinity
   ```

### Memory Bandwidth Saturation

Test with large messages:
```bash
./messaging_patterns_benchmark --transport ipc --pattern bandwidth
```

This will test with message sizes from 1KB to 1MB.

### Network Saturation

For 10GbE network saturation between two machines:

1. **Server** (receiving machine):
   ```bash
   ./network_saturation_test server --connections 16 --port 10000
   ```

2. **Client** (sending machine):
   ```bash
   ./network_saturation_test client --host <server-ip> --connections 16 --size 65536 --count 100000
   ```

### Recommended Test Configurations

#### Low Latency Test
```bash
./messaging_patterns_benchmark --transport ipc --pattern throughput --size 64 --affinity
```

#### High Throughput Test
```bash
./messaging_patterns_benchmark --transport ipc --pattern throughput --size 4096 --count 10000000
```

#### Network Bandwidth Test
```bash
# Jumbo frames (if supported)
./network_saturation_test client --host <server> --connections 8 --size 8192 --nodelay

# Standard frames
./network_saturation_test client --host <server> --connections 16 --size 1400
```

## Performance Tuning

### Linux

1. **Disable CPU frequency scaling**:
   ```bash
   sudo cpupower frequency-set -g performance
   ```

2. **Increase network buffers**:
   ```bash
   sudo sysctl -w net.core.rmem_max=134217728
   sudo sysctl -w net.core.wmem_max=134217728
   sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
   sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
   ```

3. **Enable huge pages**:
   ```bash
   sudo sysctl -w vm.nr_hugepages=1024
   ```

### macOS

1. **Increase maxfiles limit**:
   ```bash
   sudo launchctl limit maxfiles 65536 200000
   ```

2. **Disable App Nap**:
   ```bash
   defaults write NSGlobalDomain NSAppSleepDisabled -bool YES
   ```

## Interpreting Results

### Key Metrics

- **Messages/second**: Raw message throughput
- **MB/s**: Data throughput (includes headers)
- **Latency percentiles**: 50th, 90th, 99th, 99.9th
- **Min/Max latency**: Best and worst case

### Expected Performance

On modern hardware (2020+):

- **SPSC (in-process)**: 50-100M msg/s for small messages
- **IPC (shared memory)**: 10-20M msg/s
- **TCP (localhost)**: 1-5M msg/s
- **TCP (10GbE)**: 500K-2M msg/s depending on message size

### Bottleneck Identification

1. **CPU bound**: High CPU usage, performance scales with frequency
2. **Memory bound**: Performance doesn't scale with CPU frequency
3. **Network bound**: TCP performance significantly lower than IPC
4. **Latency bound**: Small messages perform poorly

## Visualization

To visualize results over time:

```bash
# Collect samples
for i in {1..10}; do
    ./messaging_patterns_benchmark --transport ipc --pattern throughput >> results.log
    sleep 1
done

# Plot with gnuplot
gnuplot -e "set terminal png; set output 'throughput.png'; plot 'results.log' using 1:2 with lines"
```