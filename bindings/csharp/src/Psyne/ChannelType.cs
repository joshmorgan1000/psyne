namespace Psyne
{
    /// <summary>
    /// Specifies the message type support for a channel.
    /// </summary>
    public enum ChannelType
    {
        /// <summary>
        /// Channel supports a single message type.
        /// More efficient when all messages have the same structure.
        /// </summary>
        Single = 0,

        /// <summary>
        /// Channel supports multiple message types.
        /// Allows sending different message types through the same channel.
        /// </summary>
        Multi = 1
    }
}