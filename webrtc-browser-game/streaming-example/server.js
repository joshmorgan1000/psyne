/**
 * @file server.js
 * @brief WebRTC streaming server example
 * 
 * Demonstrates:
 * - Server-side WebRTC with node-webrtc
 * - Video streaming from server to browser
 * - Screen capture or video file streaming
 * - Low-latency real-time streaming
 */

const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const { RTCPeerConnection, RTCVideoSource, RTCSessionDescription } = require('wrtc');
const { createCanvas } = require('canvas');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Serve static files
app.use(express.static('public'));

// Track active streams
const activeStreams = new Map();

// WebRTC configuration
const rtcConfig = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' }
    ]
};

// Create a test video source (animated canvas)
class TestVideoSource {
    constructor(width = 640, height = 480, fps = 30) {
        this.width = width;
        this.height = height;
        this.fps = fps;
        this.canvas = createCanvas(width, height);
        this.ctx = this.canvas.getContext('2d');
        this.frameCount = 0;
        this.videoSource = new RTCVideoSource();
        this.track = this.videoSource.createTrack();
        this.intervalId = null;
    }

    start() {
        const frameInterval = 1000 / this.fps;
        
        this.intervalId = setInterval(() => {
            this.renderFrame();
            this.frameCount++;
        }, frameInterval);
        
        console.log('📹 Test video source started');
    }

    stop() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.track.stop();
        console.log('📹 Test video source stopped');
    }

    renderFrame() {
        const { ctx, width, height } = this;
        
        // Clear canvas
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, width, height);
        
        // Draw animated content
        const time = Date.now() / 1000;
        
        // Moving gradient background
        const gradient = ctx.createLinearGradient(0, 0, width, height);
        gradient.addColorStop(0, `hsl(${(time * 30) % 360}, 70%, 20%)`);
        gradient.addColorStop(1, `hsl(${(time * 30 + 180) % 360}, 70%, 30%)`);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        // Server info
        ctx.fillStyle = '#00ff41';
        ctx.font = 'bold 24px monospace';
        ctx.fillText('WebRTC Server Stream', 20, 40);
        
        // Time and frame counter
        ctx.font = '18px monospace';
        ctx.fillText(`Time: ${new Date().toLocaleTimeString()}`, 20, 80);
        ctx.fillText(`Frame: ${this.frameCount}`, 20, 110);
        ctx.fillText(`FPS: ${this.fps}`, 20, 140);
        
        // Moving elements
        const centerX = width / 2;
        const centerY = height / 2;
        
        // Rotating circles
        for (let i = 0; i < 6; i++) {
            const angle = (time + i * Math.PI / 3) % (Math.PI * 2);
            const x = centerX + Math.cos(angle) * 100;
            const y = centerY + Math.sin(angle) * 100;
            
            ctx.beginPath();
            ctx.arc(x, y, 20, 0, Math.PI * 2);
            ctx.fillStyle = `hsl(${i * 60}, 100%, 50%)`;
            ctx.fill();
        }
        
        // Waveform visualization
        ctx.strokeStyle = '#00ff41';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let x = 0; x < width; x += 5) {
            const y = centerY + Math.sin(x * 0.02 + time * 2) * 50;
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Send frame to video track
        const rgbaFrame = ctx.getImageData(0, 0, width, height);
        const i420Frame = this.convertToI420(rgbaFrame);
        
        this.videoSource.onFrame({
            width: this.width,
            height: this.height,
            data: i420Frame
        });
    }

    convertToI420(rgbaFrame) {
        const { width, height } = this;
        const rgbaData = rgbaFrame.data;
        const i420Data = new Uint8ClampedArray(width * height * 1.5);
        
        // Convert RGBA to I420 (YUV420)
        let yIndex = 0;
        let uIndex = width * height;
        let vIndex = width * height * 1.25;
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const rgbaIndex = (y * width + x) * 4;
                const r = rgbaData[rgbaIndex];
                const g = rgbaData[rgbaIndex + 1];
                const b = rgbaData[rgbaIndex + 2];
                
                // Y component
                i420Data[yIndex++] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
                
                // U and V components (subsample 2x2)
                if (x % 2 === 0 && y % 2 === 0) {
                    i420Data[uIndex++] = Math.round(-0.169 * r - 0.331 * g + 0.5 * b + 128);
                    i420Data[vIndex++] = Math.round(0.5 * r - 0.419 * g - 0.081 * b + 128);
                }
            }
        }
        
        return i420Data;
    }
}

// Handle WebSocket connections
wss.on('connection', (ws) => {
    console.log('🔌 New WebSocket connection');
    
    let peerConnection = null;
    let videoSource = null;
    
    ws.on('message', async (message) => {
        const data = JSON.parse(message);
        
        switch (data.type) {
            case 'start-stream':
                try {
                    // Create peer connection
                    peerConnection = new RTCPeerConnection(rtcConfig);
                    
                    // Create video source
                    videoSource = new TestVideoSource();
                    videoSource.start();
                    
                    // Add video track to peer connection
                    peerConnection.addTrack(videoSource.track);
                    
                    // Handle ICE candidates
                    peerConnection.onicecandidate = (event) => {
                        if (event.candidate) {
                            ws.send(JSON.stringify({
                                type: 'ice-candidate',
                                candidate: event.candidate
                            }));
                        }
                    };
                    
                    // Create offer
                    const offer = await peerConnection.createOffer();
                    await peerConnection.setLocalDescription(offer);
                    
                    ws.send(JSON.stringify({
                        type: 'offer',
                        offer: offer
                    }));
                    
                    console.log('📡 Stream started for client');
                } catch (error) {
                    console.error('Error starting stream:', error);
                    ws.send(JSON.stringify({
                        type: 'error',
                        message: error.message
                    }));
                }
                break;
                
            case 'answer':
                try {
                    await peerConnection.setRemoteDescription(
                        new RTCSessionDescription(data.answer)
                    );
                    console.log('✅ WebRTC connection established');
                } catch (error) {
                    console.error('Error setting answer:', error);
                }
                break;
                
            case 'ice-candidate':
                try {
                    await peerConnection.addIceCandidate(data.candidate);
                } catch (error) {
                    console.error('Error adding ICE candidate:', error);
                }
                break;
                
            case 'stop-stream':
                if (videoSource) {
                    videoSource.stop();
                    videoSource = null;
                }
                if (peerConnection) {
                    peerConnection.close();
                    peerConnection = null;
                }
                console.log('🛑 Stream stopped');
                break;
        }
    });
    
    ws.on('close', () => {
        console.log('👋 WebSocket disconnected');
        if (videoSource) {
            videoSource.stop();
        }
        if (peerConnection) {
            peerConnection.close();
        }
    });
});

// Start server
const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
    console.log(`🚀 WebRTC streaming server running on http://localhost:${PORT}`);
    console.log('📹 Server will stream test video to connected clients');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down server...');
    wss.clients.forEach(client => client.close());
    server.close(() => {
        console.log('👋 Server closed');
        process.exit(0);
    });
});