# WebRTC Server-to-Browser Streaming Example

This example demonstrates how to stream video from a Node.js server to web browsers using WebRTC.

## Features

- **Server-side WebRTC** using node-webrtc
- **Animated test video** generated on the server
- **Real-time streaming** with low latency
- **Stream statistics** (bitrate, latency, packet loss)
- **Clean UI** with connection status

## Installation

```bash
cd streaming-example
npm install
```

## Usage

1. Start the server:
```bash
npm start
```

2. Open browser to http://localhost:3001

3. Click "Start Stream" to begin receiving video from the server

## How It Works

1. **Server generates video** - Creates animated canvas frames
2. **WebRTC connection** - Browser requests stream via WebSocket
3. **Media track added** - Server adds video track to peer connection
4. **Low latency stream** - Direct P2P connection for minimal delay

## Use Cases

- **Live streaming** from server to multiple clients
- **Screen sharing** from server applications
- **Video processing** - Apply filters server-side
- **Game streaming** - Stream gameplay to browsers
- **Surveillance** - Stream camera feeds
- **Broadcasting** - One-to-many streaming

## Production Considerations

### Video Sources
Instead of the test pattern, you can stream:
- **FFmpeg input** - Stream files or live sources
- **Screen capture** - Using libraries like `screenshot-desktop`
- **Camera feeds** - From USB cameras or IP cameras
- **Game engines** - Stream rendered game frames

### Scaling
For production streaming to many clients:
- Use **SFU** (Selective Forwarding Unit) like Janus or mediasoup
- Implement **TURN servers** for better connectivity
- Add **authentication** and room management
- Consider **CDN integration** for global distribution

### Example: Stream Video File

```javascript
// Replace TestVideoSource with FFmpeg streaming
const ffmpeg = require('fluent-ffmpeg');
const { PassThrough } = require('stream');

function streamVideoFile(filename, peerConnection) {
    const stream = new PassThrough();
    
    ffmpeg(filename)
        .format('rawvideo')
        .size('640x480')
        .fps(30)
        .videoCodec('rawvideo')
        .outputOptions([
            '-pix_fmt yuv420p'
        ])
        .pipe(stream);
        
    // Convert stream to WebRTC track
    const videoSource = new RTCVideoSource();
    const track = videoSource.createTrack();
    peerConnection.addTrack(track);
    
    // Feed frames to WebRTC
    stream.on('data', (frame) => {
        videoSource.onFrame({
            width: 640,
            height: 480,
            data: frame
        });
    });
}
```

### Example: Screen Capture

```javascript
const screenshot = require('screenshot-desktop');

async function streamScreen(peerConnection) {
    const videoSource = new RTCVideoSource();
    const track = videoSource.createTrack();
    peerConnection.addTrack(track);
    
    // Capture screen at 30 FPS
    setInterval(async () => {
        const img = await screenshot({ format: 'png' });
        // Convert to I420 and send to videoSource.onFrame()
    }, 33); // ~30 FPS
}
```

## WebRTC Advantages for Streaming

- **Ultra-low latency** - Sub-second delay possible
- **P2P when possible** - Direct connections reduce server load
- **Adaptive bitrate** - Automatically adjusts to network conditions
- **NAT traversal** - Works through firewalls
- **Secure** - Mandatory encryption (DTLS/SRTP)

## Limitations

- **Browser compatibility** - Not all browsers support all codecs
- **Firewall issues** - Some corporate networks block WebRTC
- **Complexity** - More complex than simple HTTP streaming
- **CPU intensive** - Encoding/decoding requires processing power

## Next Steps

1. **Add audio streaming** - Include audio tracks
2. **Multiple streams** - Support multiple video sources
3. **Recording** - Save streams server-side
4. **Transcoding** - Convert between formats
5. **Analytics** - Track viewer statistics