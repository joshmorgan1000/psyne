namespace Psyne
{
    /// <summary>
    /// Contains performance metrics for a channel.
    /// </summary>
    public class Metrics
    {
        /// <summary>
        /// Gets the total number of messages sent through the channel.
        /// </summary>
        public ulong MessagesSent { get; internal set; }

        /// <summary>
        /// Gets the total number of bytes sent through the channel.
        /// </summary>
        public ulong BytesSent { get; internal set; }

        /// <summary>
        /// Gets the total number of messages received from the channel.
        /// </summary>
        public ulong MessagesReceived { get; internal set; }

        /// <summary>
        /// Gets the total number of bytes received from the channel.
        /// </summary>
        public ulong BytesReceived { get; internal set; }

        /// <summary>
        /// Gets the number of times a send operation was blocked.
        /// </summary>
        public ulong SendBlocks { get; internal set; }

        /// <summary>
        /// Gets the number of times a receive operation was blocked.
        /// </summary>
        public ulong ReceiveBlocks { get; internal set; }

        /// <summary>
        /// Gets the average message size for sent messages.
        /// </summary>
        public double AverageSentMessageSize => MessagesSent > 0 ? (double)BytesSent / MessagesSent : 0;

        /// <summary>
        /// Gets the average message size for received messages.
        /// </summary>
        public double AverageReceivedMessageSize => MessagesReceived > 0 ? (double)BytesReceived / MessagesReceived : 0;

        /// <summary>
        /// Gets the total number of messages processed (sent + received).
        /// </summary>
        public ulong TotalMessages => MessagesSent + MessagesReceived;

        /// <summary>
        /// Gets the total number of bytes processed (sent + received).
        /// </summary>
        public ulong TotalBytes => BytesSent + BytesReceived;

        internal static Metrics FromNative(Native.PsyneNative.Metrics nativeMetrics)
        {
            return new Metrics
            {
                MessagesSent = nativeMetrics.MessagesSent,
                BytesSent = nativeMetrics.BytesSent,
                MessagesReceived = nativeMetrics.MessagesReceived,
                BytesReceived = nativeMetrics.BytesReceived,
                SendBlocks = nativeMetrics.SendBlocks,
                ReceiveBlocks = nativeMetrics.ReceiveBlocks
            };
        }

        /// <summary>
        /// Returns a string representation of the metrics.
        /// </summary>
        public override string ToString()
        {
            return $"Messages: {MessagesSent} sent, {MessagesReceived} received | " +
                   $"Bytes: {BytesSent} sent, {BytesReceived} received | " +
                   $"Blocks: {SendBlocks} send, {ReceiveBlocks} receive";
        }
    }
}