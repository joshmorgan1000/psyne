namespace Psyne
{
    /// <summary>
    /// Specifies the synchronization mode for a channel.
    /// </summary>
    public enum ChannelMode
    {
        /// <summary>
        /// Single Producer, Single Consumer mode.
        /// Most efficient mode when there's only one producer and one consumer.
        /// </summary>
        SingleProducerSingleConsumer = 0,

        /// <summary>
        /// Single Producer, Multiple Consumer mode.
        /// Allows multiple consumers to read from a single producer.
        /// </summary>
        SingleProducerMultipleConsumer = 1,

        /// <summary>
        /// Multiple Producer, Single Consumer mode.
        /// Allows multiple producers to write to a single consumer.
        /// </summary>
        MultipleProducerSingleConsumer = 2,

        /// <summary>
        /// Multiple Producer, Multiple Consumer mode.
        /// Most flexible but potentially less efficient mode.
        /// </summary>
        MultipleProducerMultipleConsumer = 3
    }
}