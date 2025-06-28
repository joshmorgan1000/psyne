package com.psyne;

/**
 * Channel performance metrics.
 * 
 * This class contains various performance metrics collected by a Psyne channel,
 * providing insights into message throughput and blocking behavior.
 */
public class Metrics {
    
    private final long messagesSent;
    private final long bytesSent;
    private final long messagesReceived;
    private final long bytesReceived;
    private final long sendBlocks;
    private final long receiveBlocks;
    
    /**
     * Constructs a new Metrics instance.
     * 
     * @param messagesSent Number of messages sent
     * @param bytesSent Number of bytes sent
     * @param messagesReceived Number of messages received
     * @param bytesReceived Number of bytes received
     * @param sendBlocks Number of times send operations blocked
     * @param receiveBlocks Number of times receive operations blocked
     */
    public Metrics(long messagesSent, long bytesSent, long messagesReceived,
                   long bytesReceived, long sendBlocks, long receiveBlocks) {
        this.messagesSent = messagesSent;
        this.bytesSent = bytesSent;
        this.messagesReceived = messagesReceived;
        this.bytesReceived = bytesReceived;
        this.sendBlocks = sendBlocks;
        this.receiveBlocks = receiveBlocks;
    }
    
    /**
     * Gets the number of messages sent through the channel.
     * 
     * @return The number of messages sent
     */
    public long getMessagesSent() {
        return messagesSent;
    }
    
    /**
     * Gets the number of bytes sent through the channel.
     * 
     * @return The number of bytes sent
     */
    public long getBytesSent() {
        return bytesSent;
    }
    
    /**
     * Gets the number of messages received from the channel.
     * 
     * @return The number of messages received
     */
    public long getMessagesReceived() {
        return messagesReceived;
    }
    
    /**
     * Gets the number of bytes received from the channel.
     * 
     * @return The number of bytes received
     */
    public long getBytesReceived() {
        return bytesReceived;
    }
    
    /**
     * Gets the number of times send operations blocked.
     * A high value indicates channel congestion.
     * 
     * @return The number of send blocks
     */
    public long getSendBlocks() {
        return sendBlocks;
    }
    
    /**
     * Gets the number of times receive operations blocked.
     * A high value indicates the receiver is waiting for messages.
     * 
     * @return The number of receive blocks
     */
    public long getReceiveBlocks() {
        return receiveBlocks;
    }
    
    /**
     * Calculates the average message size for sent messages.
     * 
     * @return The average message size in bytes, or 0 if no messages sent
     */
    public double getAverageSentMessageSize() {
        return messagesSent > 0 ? (double) bytesSent / messagesSent : 0;
    }
    
    /**
     * Calculates the average message size for received messages.
     * 
     * @return The average message size in bytes, or 0 if no messages received
     */
    public double getAverageReceivedMessageSize() {
        return messagesReceived > 0 ? (double) bytesReceived / messagesReceived : 0;
    }
    
    @Override
    public String toString() {
        return String.format(
            "Metrics{messagesSent=%d, bytesSent=%d, messagesReceived=%d, " +
            "bytesReceived=%d, sendBlocks=%d, receiveBlocks=%d}",
            messagesSent, bytesSent, messagesReceived, bytesReceived,
            sendBlocks, receiveBlocks
        );
    }
}