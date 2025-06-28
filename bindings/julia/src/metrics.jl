"""
Metrics and monitoring for Psyne.jl

Provides performance monitoring and channel diagnostics for optimizing
message passing performance and debugging issues.
"""

"""
    Metrics

Performance metrics for a Psyne channel.

These metrics provide insight into channel usage patterns and can help
identify performance bottlenecks or configuration issues.

# Fields
- `messages_sent::UInt64`: Total number of messages sent
- `bytes_sent::UInt64`: Total bytes sent through the channel
- `messages_received::UInt64`: Total number of messages received
- `bytes_received::UInt64`: Total bytes received from the channel
- `send_blocks::UInt64`: Number of times send operations blocked
- `receive_blocks::UInt64`: Number of times receive operations blocked

# Examples

```julia
ch = channel("memory://perf", enable_metrics=true)

# ... send and receive messages ...

metrics = get_metrics(ch)
println("Throughput: \$(metrics.bytes_sent / 1e6) MB sent")
println("Messages: \$(metrics.messages_sent) sent, \$(metrics.messages_received) received")
```
"""
struct Metrics
    messages_sent::UInt64
    bytes_sent::UInt64
    messages_received::UInt64
    bytes_received::UInt64
    send_blocks::UInt64
    receive_blocks::UInt64
end

"""
    get_metrics(channel::PsyneChannel) -> Metrics

Get current performance metrics for a channel.

# Arguments
- `channel`: The channel to get metrics for

# Returns
Current `Metrics` snapshot.

# Throws
- `PsyneError`: If the channel doesn't support metrics

# Examples

```julia
ch = channel("memory://test")
metrics = get_metrics(ch)
println("Messages sent: \$(metrics.messages_sent)")
```
"""
function get_metrics(channel::PsyneChannel)
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    # Allocate C struct for metrics
    metrics_ref = Ref{NTuple{6, UInt64}}()
    
    code = ccall((:psyne_channel_get_metrics, libpsyne), Int32,
                (Ptr{Cvoid}, Ptr{NTuple{6, UInt64}}),
                channel.handle, metrics_ref)
    check_error(code)
    
    metrics_tuple = metrics_ref[]
    return Metrics(
        metrics_tuple[1],  # messages_sent
        metrics_tuple[2],  # bytes_sent
        metrics_tuple[3],  # messages_received
        metrics_tuple[4],  # bytes_received
        metrics_tuple[5],  # send_blocks
        metrics_tuple[6]   # receive_blocks
    )
end

"""
    reset_metrics!(channel::PsyneChannel)

Reset all metrics counters for a channel to zero.

# Arguments
- `channel`: The channel to reset metrics for

# Examples

```julia
ch = channel("memory://test")
# ... use channel ...
reset_metrics!(ch)  # Start fresh counters
```
"""
function reset_metrics!(channel::PsyneChannel)
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    code = ccall((:psyne_channel_reset_metrics, libpsyne), Int32,
                (Ptr{Cvoid},), channel.handle)
    check_error(code)
end

"""
    enable_metrics!(channel::PsyneChannel, enable::Bool = true)

Enable or disable metrics collection for a channel.

# Arguments
- `channel`: The channel to configure
- `enable`: True to enable metrics, false to disable

# Examples

```julia
ch = channel("memory://test")
enable_metrics!(ch, true)  # Enable metrics
# ... use channel ...
metrics = get_metrics(ch)
```
"""
function enable_metrics!(channel::PsyneChannel, enable::Bool = true)
    channel.handle == C_NULL && throw(PsyneError("Channel is closed", -1))
    
    code = ccall((:psyne_channel_enable_metrics, libpsyne), Int32,
                (Ptr{Cvoid}, Bool), channel.handle, enable)
    check_error(code)
end

"""
    throughput(metrics::Metrics, duration_seconds::Real) -> NamedTuple

Calculate throughput metrics from channel metrics.

# Arguments
- `metrics`: Channel metrics snapshot
- `duration_seconds`: Time period for the metrics

# Returns
NamedTuple with throughput calculations:
- `messages_per_second`: Message rate
- `bytes_per_second`: Byte rate
- `mbps`: Megabits per second

# Examples

```julia
metrics_start = get_metrics(ch)
# ... run workload for 10 seconds ...
metrics_end = get_metrics(ch)

delta_metrics = metrics_end - metrics_start
throughput_stats = throughput(delta_metrics, 10.0)
println("Throughput: \$(throughput_stats.mbps) Mbps")
```
"""
function throughput(metrics::Metrics, duration_seconds::Real)
    duration_seconds > 0 || throw(ArgumentError("Duration must be positive"))
    
    messages_per_second = (metrics.messages_sent + metrics.messages_received) / duration_seconds
    bytes_per_second = (metrics.bytes_sent + metrics.bytes_received) / duration_seconds
    mbps = bytes_per_second * 8 / 1e6  # Convert to megabits per second
    
    return (
        messages_per_second = messages_per_second,
        bytes_per_second = bytes_per_second,
        mbps = mbps
    )
end

"""
    efficiency(metrics::Metrics) -> NamedTuple

Calculate efficiency metrics from channel metrics.

# Arguments
- `metrics`: Channel metrics snapshot

# Returns
NamedTuple with efficiency calculations:
- `send_block_rate`: Fraction of sends that blocked
- `receive_block_rate`: Fraction of receives that blocked
- `utilization`: Overall channel utilization estimate

# Examples

```julia
metrics = get_metrics(ch)
efficiency_stats = efficiency(metrics)
if efficiency_stats.send_block_rate > 0.1
    println("Warning: High send blocking rate!")
end
```
"""
function efficiency(metrics::Metrics)
    total_sends = metrics.messages_sent + metrics.send_blocks
    total_receives = metrics.messages_received + metrics.receive_blocks
    
    send_block_rate = total_sends > 0 ? metrics.send_blocks / total_sends : 0.0
    receive_block_rate = total_receives > 0 ? metrics.receive_blocks / total_receives : 0.0
    
    # Estimate utilization based on balance of sends/receives and blocking
    balance = if metrics.messages_sent > 0 && metrics.messages_received > 0
        min(metrics.messages_sent, metrics.messages_received) / 
        max(metrics.messages_sent, metrics.messages_received)
    else
        0.0
    end
    
    utilization = balance * (1.0 - max(send_block_rate, receive_block_rate))
    
    return (
        send_block_rate = send_block_rate,
        receive_block_rate = receive_block_rate,
        utilization = utilization
    )
end

"""
    health_check(channel::PsyneChannel) -> NamedTuple

Perform a health check on a channel and return diagnostic information.

# Arguments
- `channel`: The channel to check

# Returns
NamedTuple with health information:
- `status`: :healthy, :warning, or :error
- `issues`: Array of issue descriptions
- `recommendations`: Array of recommended actions

# Examples

```julia
ch = channel("memory://test")
health = health_check(ch)
if health.status != :healthy
    println("Channel issues: ", health.issues)
    println("Recommendations: ", health.recommendations)
end
```
"""
function health_check(channel::PsyneChannel)
    issues = String[]
    recommendations = String[]
    
    # Check if channel is valid
    if channel.handle == C_NULL
        push!(issues, "Channel handle is null (closed)")
        push!(recommendations, "Create a new channel")
        return (status = :error, issues = issues, recommendations = recommendations)
    end
    
    # Check if channel is stopped
    if is_stopped(channel)
        push!(issues, "Channel is stopped")
        push!(recommendations, "Create a new channel or check stop conditions")
    end
    
    # Get metrics if available
    try
        metrics = get_metrics(channel)
        efficiency_stats = efficiency(metrics)
        
        # Check for high blocking rates
        if efficiency_stats.send_block_rate > 0.2
            push!(issues, "High send blocking rate ($(round(efficiency_stats.send_block_rate * 100, digits=1))%)")
            push!(recommendations, "Increase buffer size or reduce send rate")
        end
        
        if efficiency_stats.receive_block_rate > 0.2
            push!(issues, "High receive blocking rate ($(round(efficiency_stats.receive_block_rate * 100, digits=1))%)")
            push!(recommendations, "Increase receive polling frequency")
        end
        
        # Check for imbalanced send/receive
        if metrics.messages_sent > 0 && metrics.messages_received == 0
            push!(issues, "Messages sent but none received")
            push!(recommendations, "Check receiver process/thread")
        elseif metrics.messages_received > 0 && metrics.messages_sent == 0
            push!(issues, "Messages received but none sent")
            push!(recommendations, "Check sender process/thread")
        end
        
        # Check utilization
        if efficiency_stats.utilization < 0.5 && metrics.messages_sent > 100
            push!(issues, "Low channel utilization ($(round(efficiency_stats.utilization * 100, digits=1))%)")
            push!(recommendations, "Consider using a simpler channel mode or smaller buffer")
        end
        
    catch e
        if isa(e, PsyneError) && e.message == "Channel doesn't support metrics"
            # Metrics not enabled, can't perform detailed health check
            push!(issues, "Metrics not available for detailed health check")
            push!(recommendations, "Enable metrics for better monitoring")
        else
            push!(issues, "Error getting metrics: $(e)")
        end
    end
    
    # Determine overall status
    status = if any(contains.(issues, "error")) || any(contains.(issues, "stopped"))
        :error
    elseif !isempty(issues)
        :warning
    else
        :healthy
    end
    
    return (status = status, issues = issues, recommendations = recommendations)
end

"""
    monitor(channel::PsyneChannel, duration_seconds::Real; interval_seconds::Real = 1.0) -> Vector{Metrics}

Monitor a channel over time and collect metrics snapshots.

# Arguments
- `channel`: The channel to monitor
- `duration_seconds`: Total monitoring duration
- `interval_seconds`: Interval between metrics snapshots

# Returns
Vector of `Metrics` snapshots collected over time.

# Examples

```julia
ch = channel("memory://perf")

# Monitor for 30 seconds with 1-second intervals
snapshots = monitor(ch, 30.0, interval_seconds=1.0)

# Analyze throughput over time
for (i, snapshot) in enumerate(snapshots)
    if i > 1
        delta = snapshot - snapshots[i-1]
        tput = throughput(delta, 1.0)
        println("Second \$i: \$(tput.mbps) Mbps")
    end
end
```
"""
function monitor(channel::PsyneChannel, duration_seconds::Real; interval_seconds::Real = 1.0)
    duration_seconds > 0 || throw(ArgumentError("Duration must be positive"))
    interval_seconds > 0 || throw(ArgumentError("Interval must be positive"))
    
    snapshots = Metrics[]
    start_time = time()
    next_snapshot = start_time + interval_seconds
    
    # Initial snapshot
    try
        push!(snapshots, get_metrics(channel))
    catch e
        if isa(e, PsyneError)
            @warn "Could not get initial metrics: $(e.message)"
            return snapshots
        else
            rethrow(e)
        end
    end
    
    while time() - start_time < duration_seconds
        if time() >= next_snapshot
            try
                push!(snapshots, get_metrics(channel))
                next_snapshot += interval_seconds
            catch e
                if isa(e, PsyneError)
                    @warn "Could not get metrics: $(e.message)"
                    break
                else
                    rethrow(e)
                end
            end
        end
        
        # Small sleep to avoid busy waiting
        sleep(min(0.1, interval_seconds / 10))
    end
    
    return snapshots
end

# Arithmetic operations on Metrics
function Base.:-(m1::Metrics, m2::Metrics)
    return Metrics(
        m1.messages_sent - m2.messages_sent,
        m1.bytes_sent - m2.bytes_sent,
        m1.messages_received - m2.messages_received,
        m1.bytes_received - m2.bytes_received,
        m1.send_blocks - m2.send_blocks,
        m1.receive_blocks - m2.receive_blocks
    )
end

function Base.:+(m1::Metrics, m2::Metrics)
    return Metrics(
        m1.messages_sent + m2.messages_sent,
        m1.bytes_sent + m2.bytes_sent,
        m1.messages_received + m2.messages_received,
        m1.bytes_received + m2.bytes_received,
        m1.send_blocks + m2.send_blocks,
        m1.receive_blocks + m2.receive_blocks
    )
end

# Pretty printing for metrics
function Base.show(io::IO, metrics::Metrics)
    print(io, "Metrics(sent: $(metrics.messages_sent) msgs/$(metrics.bytes_sent) bytes, ",
              "recv: $(metrics.messages_received) msgs/$(metrics.bytes_received) bytes)")
end

function Base.show(io::IO, ::MIME"text/plain", metrics::Metrics)
    println(io, "Psyne Channel Metrics:")
    println(io, "  Messages sent: $(metrics.messages_sent)")
    println(io, "  Bytes sent: $(metrics.bytes_sent)")
    println(io, "  Messages received: $(metrics.messages_received)")
    println(io, "  Bytes received: $(metrics.bytes_received)")
    println(io, "  Send blocks: $(metrics.send_blocks)")
    println(io, "  Receive blocks: $(metrics.receive_blocks)")
    
    # Calculate derived statistics
    total_messages = metrics.messages_sent + metrics.messages_received
    total_bytes = metrics.bytes_sent + metrics.bytes_received
    total_blocks = metrics.send_blocks + metrics.receive_blocks
    
    if total_messages > 0
        avg_message_size = total_bytes / total_messages
        println(io, "  Average message size: $(round(avg_message_size, digits=1)) bytes")
    end
    
    if total_messages + total_blocks > 0
        block_rate = total_blocks / (total_messages + total_blocks) * 100
        println(io, "  Blocking rate: $(round(block_rate, digits=1))%")
    end
end