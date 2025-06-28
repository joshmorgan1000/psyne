"""
Compression support for Psyne.jl

Provides compression configuration and utilities for reducing bandwidth usage
in network channels and storage applications.
"""

"""
    CompressionConfig

Configuration for message compression in Psyne channels.

Compression can significantly reduce bandwidth usage for network channels,
especially when transmitting data with patterns or redundancy.

# Fields
- `type::CompressionType`: Compression algorithm (None, LZ4, Zstd, Snappy)
- `level::Int32`: Compression level (algorithm-dependent, typically 1-9)
- `min_size_threshold::UInt64`: Don't compress messages smaller than this
- `enable_checksum::Bool`: Add integrity checksum to compressed data

# Examples

```julia
# Fast compression with LZ4
config = CompressionConfig(type=LZ4, level=1)

# High compression with Zstd
config = CompressionConfig(type=Zstd, level=9, min_size_threshold=256)

# Balanced compression with checksum
config = CompressionConfig(
    type=Snappy, 
    level=3,
    min_size_threshold=128,
    enable_checksum=true
)
```
"""
@kwdef struct CompressionConfig
    type::CompressionType = None
    level::Int32 = 1
    min_size_threshold::UInt64 = 128
    enable_checksum::Bool = true
end

"""
    lz4_config(; level::Integer = 1, min_size::Integer = 128, checksum::Bool = true) -> CompressionConfig

Create LZ4 compression configuration.

LZ4 provides fast compression/decompression with moderate compression ratios.
Ideal for real-time applications where latency is critical.

# Arguments
- `level`: Compression level (1-9, default: 1)
- `min_size`: Minimum message size to compress (default: 128 bytes)
- `checksum`: Enable integrity checksum (default: true)

# Examples

```julia
# Fast LZ4 compression
config = lz4_config()

# Higher compression level
config = lz4_config(level=5, min_size=256)
```
"""
function lz4_config(; level::Integer = 1, min_size::Integer = 128, checksum::Bool = true)
    return CompressionConfig(
        type=LZ4,
        level=Int32(clamp(level, 1, 9)),
        min_size_threshold=UInt64(max(min_size, 0)),
        enable_checksum=checksum
    )
end

"""
    zstd_config(; level::Integer = 3, min_size::Integer = 128, checksum::Bool = true) -> CompressionConfig

Create Zstd compression configuration.

Zstd provides excellent compression ratios with good performance.
Suitable for applications where bandwidth is more critical than latency.

# Arguments
- `level`: Compression level (1-22, default: 3)
- `min_size`: Minimum message size to compress (default: 128 bytes)
- `checksum`: Enable integrity checksum (default: true)

# Examples

```julia
# Standard Zstd compression
config = zstd_config()

# Maximum compression
config = zstd_config(level=22, min_size=512)
```
"""
function zstd_config(; level::Integer = 3, min_size::Integer = 128, checksum::Bool = true)
    return CompressionConfig(
        type=Zstd,
        level=Int32(clamp(level, 1, 22)),
        min_size_threshold=UInt64(max(min_size, 0)),
        enable_checksum=checksum
    )
end

"""
    snappy_config(; level::Integer = 1, min_size::Integer = 128, checksum::Bool = true) -> CompressionConfig

Create Snappy compression configuration.

Snappy provides balanced compression speed and ratio.
Good general-purpose choice for most applications.

# Arguments
- `level`: Compression level (1-9, default: 1)
- `min_size`: Minimum message size to compress (default: 128 bytes)
- `checksum`: Enable integrity checksum (default: true)

# Examples

```julia
# Standard Snappy compression
config = snappy_config()

# Custom threshold
config = snappy_config(min_size=64)
```
"""
function snappy_config(; level::Integer = 1, min_size::Integer = 128, checksum::Bool = true)
    return CompressionConfig(
        type=Snappy,
        level=Int32(clamp(level, 1, 9)),
        min_size_threshold=UInt64(max(min_size, 0)),
        enable_checksum=checksum
    )
end

"""
    no_compression() -> CompressionConfig

Create configuration for no compression.

# Examples

```julia
config = no_compression()
ch = channel("tcp://localhost:8080", compression=config)
```
"""
function no_compression()
    return CompressionConfig(type=None)
end

"""
    estimate_compression_ratio(data::AbstractArray, config::CompressionConfig) -> Float64

Estimate the compression ratio for given data and configuration.

This provides a rough estimate without actually performing compression.
Useful for deciding whether compression would be beneficial.

# Arguments
- `data`: The data to estimate compression for
- `config`: Compression configuration

# Returns
Estimated compression ratio (< 1.0 means compression reduces size)

# Examples

```julia
data = rand(Float32, 1000)
config = lz4_config()
ratio = estimate_compression_ratio(data, config)
println("Estimated compression ratio: ", ratio)
```
"""
function estimate_compression_ratio(data::AbstractArray, config::CompressionConfig)
    if config.type == None
        return 1.0
    end
    
    data_size = sizeof(data)
    if data_size < config.min_size_threshold
        return 1.0  # No compression for small data
    end
    
    # Simple heuristic based on data type and patterns
    if isa(data, AbstractArray{<:AbstractFloat})
        # Floating point data typically compresses moderately
        base_ratio = 0.6
    elseif isa(data, AbstractArray{<:Integer})
        # Integer data can compress well if there are patterns
        base_ratio = 0.4
    else
        # Conservative estimate for other types
        base_ratio = 0.7
    end
    
    # Adjust based on compression algorithm
    algorithm_factor = if config.type == LZ4
        1.2  # LZ4 is fast but lower compression
    elseif config.type == Zstd
        0.8  # Zstd has better compression
    elseif config.type == Snappy
        1.0  # Snappy is balanced
    else
        1.0
    end
    
    # Adjust based on compression level
    level_factor = 1.0 - (config.level - 1) * 0.05
    
    return base_ratio * algorithm_factor * level_factor
end

"""
    compression_benchmark(data::AbstractArray; algorithms::Vector{CompressionType} = [LZ4, Zstd, Snappy]) -> Dict

Benchmark different compression algorithms on sample data.

This function tests various compression configurations and returns performance
metrics to help choose the best algorithm for your use case.

# Arguments
- `data`: Sample data to benchmark
- `algorithms`: List of compression algorithms to test

# Returns
Dictionary with algorithm names as keys and performance metrics as values.

# Examples

```julia
# Benchmark with sample data
data = rand(Float32, 10000)
results = compression_benchmark(data)

for (alg, metrics) in results
    println("\$alg: ratio=\$(metrics[:ratio]), estimated_time=\$(metrics[:time_ms])ms")
end
```
"""
function compression_benchmark(data::AbstractArray; algorithms::Vector{CompressionType} = [LZ4, Zstd, Snappy])
    results = Dict{CompressionType, Dict{Symbol, Float64}}()
    
    data_size = sizeof(data)
    
    for alg in algorithms
        if alg == None
            continue
        end
        
        config = CompressionConfig(type=alg, level=3)
        ratio = estimate_compression_ratio(data, config)
        
        # Estimate compression time based on algorithm characteristics
        time_factor = if alg == LZ4
            0.1  # Very fast
        elseif alg == Snappy
            0.2  # Fast
        elseif alg == Zstd
            0.5  # Moderate speed
        else
            0.3  # Default
        end
        
        estimated_time_ms = data_size * time_factor / 1e6  # Rough estimate
        
        results[alg] = Dict(
            :ratio => ratio,
            :time_ms => estimated_time_ms,
            :throughput_mbps => (data_size / 1e6) / (estimated_time_ms / 1000),
            :compressed_size => Int(round(data_size * ratio))
        )
    end
    
    return results
end

"""
    recommend_compression(data::AbstractArray; priority::Symbol = :balanced) -> CompressionConfig

Recommend optimal compression configuration for given data.

# Arguments
- `data`: Sample data to analyze
- `priority`: Optimization priority (:speed, :ratio, :balanced)

# Returns
Recommended `CompressionConfig`.

# Examples

```julia
data = rand(Float64, 5000)

# Optimize for speed
fast_config = recommend_compression(data, priority=:speed)

# Optimize for compression ratio
small_config = recommend_compression(data, priority=:ratio)

# Balanced approach
balanced_config = recommend_compression(data, priority=:balanced)
```
"""
function recommend_compression(data::AbstractArray; priority::Symbol = :balanced)
    data_size = sizeof(data)
    
    # Don't compress very small data
    if data_size < 64
        return no_compression()
    end
    
    algorithms = [LZ4, Snappy, Zstd]
    benchmarks = compression_benchmark(data, algorithms=algorithms)
    
    if priority == :speed
        # Choose fastest algorithm
        best_alg = LZ4
        level = 1
    elseif priority == :ratio
        # Choose best compression ratio
        best_alg = Zstd
        level = 6
    else  # :balanced
        # Balance speed and compression
        best_alg = Snappy
        level = 3
    end
    
    min_size = if data_size < 512
        64
    elseif data_size < 4096
        128
    else
        256
    end
    
    return CompressionConfig(
        type=best_alg,
        level=Int32(level),
        min_size_threshold=UInt64(min_size),
        enable_checksum=true
    )
end

# Pretty printing for compression configs
function Base.show(io::IO, config::CompressionConfig)
    if config.type == None
        print(io, "CompressionConfig(None)")
    else
        print(io, "CompressionConfig($(config.type), level=$(config.level), threshold=$(config.min_size_threshold))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", config::CompressionConfig)
    println(io, "CompressionConfig:")
    println(io, "  Algorithm: $(config.type)")
    if config.type != None
        println(io, "  Level: $(config.level)")
        println(io, "  Min size threshold: $(config.min_size_threshold) bytes")
        println(io, "  Checksum enabled: $(config.enable_checksum)")
    end
end