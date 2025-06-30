# WebRTC with Outgoing-Only Connections

## The Challenge

Setting up WebRTC between two clients when your web server can only make **outgoing** connections (common in restrictive environments like corporate networks, Lambda functions, or certain hosting providers).

## Understanding the Problem

Traditional WebRTC setup requires:
1. **Signaling server** - Exchanges offer/answer/ICE candidates
2. **STUN server** - Discovers public IP addresses
3. **TURN server** - Relays traffic when direct connection fails

The challenge: How do you handle signaling when your server can't accept incoming WebSocket/HTTP connections?

## Solution Approaches

### 1. External Signaling Service (Recommended)

Use a third-party service that both your server and clients can connect to:

```javascript
// Server-side (outgoing connections only)
const io = require('socket.io-client');
const socket = io('https://signaling.example.com', {
    transports: ['websocket']
});

// Your server connects OUT to the signaling service
socket.on('connect', () => {
    console.log('Connected to external signaling server');
    
    // Register as a relay
    socket.emit('register-relay', {
        serverId: 'my-server-123',
        capabilities: ['webrtc-relay']
    });
});

// Relay signaling between clients
socket.on('relay-signal', (data) => {
    const { from, to, signal } = data;
    
    // Forward to the target client
    socket.emit('forward-signal', {
        from: from,
        to: to,
        signal: signal
    });
});
```

**Popular Services:**
- **PeerJS Cloud** - Free tier available
- **Socket.io Cloud** - Managed WebSocket service
- **Pusher** - Real-time messaging service
- **Ably** - Real-time data delivery network

### 2. Long Polling Approach

If WebSockets are blocked but HTTP requests work:

```javascript
// Client polls server for signaling messages
class LongPollingSignaler {
    constructor(serverUrl) {
        this.serverUrl = serverUrl;
        this.clientId = generateUniqueId();
        this.polling = false;
    }
    
    async start() {
        this.polling = true;
        while (this.polling) {
            try {
                // Server holds request open until message available
                const response = await fetch(`${this.serverUrl}/poll`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        clientId: this.clientId,
                        timeout: 30000 // 30 second timeout
                    })
                });
                
                if (response.ok) {
                    const messages = await response.json();
                    messages.forEach(msg => this.handleSignal(msg));
                }
            } catch (error) {
                console.error('Polling error:', error);
                await this.wait(5000); // Backoff on error
            }
        }
    }
    
    async sendSignal(targetId, signal) {
        // Server queues message for target client
        await fetch(`${this.serverUrl}/signal`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                from: this.clientId,
                to: targetId,
                signal: signal
            })
        });
    }
}
```

### 3. Shared Database Signaling

Use a database both server and clients can access:

```javascript
// Using Firebase Realtime Database as signaling channel
class FirebaseSignaler {
    constructor() {
        this.db = firebase.database();
        this.userId = generateUniqueId();
        this.signalsRef = this.db.ref('signals');
    }
    
    async sendOffer(targetId, offer) {
        // Write offer to database
        await this.signalsRef.child(targetId).push({
            from: this.userId,
            type: 'offer',
            offer: offer,
            timestamp: firebase.database.ServerValue.TIMESTAMP
        });
    }
    
    listenForSignals(callback) {
        // Listen for signals addressed to us
        this.signalsRef.child(this.userId).on('child_added', (snapshot) => {
            const signal = snapshot.val();
            callback(signal);
            
            // Clean up processed signal
            snapshot.ref.remove();
        });
    }
}
```

### 4. QR Code / Manual Exchange

For ultimate simplicity in restricted environments:

```javascript
// Generate connection info as QR code
class QRSignaling {
    async createConnectionOffer() {
        const pc = new RTCPeerConnection(rtcConfig);
        const dc = pc.createDataChannel('data');
        
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        
        // Wait for ICE gathering
        await new Promise(resolve => {
            pc.onicecandidate = (e) => {
                if (!e.candidate) resolve();
            };
        });
        
        // Encode complete offer + ICE candidates
        const connectionData = {
            offer: pc.localDescription,
            ice: pc.localDescription.sdp // ICE candidates included
        };
        
        // Generate QR code
        const qrCode = await QRCode.toDataURL(
            JSON.stringify(connectionData)
        );
        
        return { pc, qrCode };
    }
    
    async connectWithQR(qrData) {
        const { offer, ice } = JSON.parse(qrData);
        
        const pc = new RTCPeerConnection(rtcConfig);
        await pc.setRemoteDescription(offer);
        
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        
        // Return answer as QR for other peer
        return QRCode.toDataURL(
            JSON.stringify({ answer: pc.localDescription })
        );
    }
}
```

## Complete Example: Outgoing-Only WebRTC

Here's a full implementation using multiple fallback approaches:

```javascript
class OutgoingOnlyWebRTC {
    constructor(config) {
        this.config = config;
        this.signalers = [];
        this.peers = new Map();
        
        // Initialize multiple signaling methods
        this.initializeSignalers();
    }
    
    initializeSignalers() {
        // Try external service first
        if (this.config.externalSignalingUrl) {
            this.signalers.push(
                new ExternalServiceSignaler(this.config.externalSignalingUrl)
            );
        }
        
        // Fallback to database signaling
        if (this.config.firebaseConfig) {
            this.signalers.push(
                new FirebaseSignaler(this.config.firebaseConfig)
            );
        }
        
        // Last resort: manual exchange
        this.signalers.push(new ManualSignaler());
    }
    
    async connect(targetPeerId) {
        // Try each signaling method
        for (const signaler of this.signalers) {
            try {
                console.log(`Trying ${signaler.name} signaling...`);
                const pc = await this.establishConnection(targetPeerId, signaler);
                
                if (pc && pc.connectionState === 'connected') {
                    this.peers.set(targetPeerId, pc);
                    return pc;
                }
            } catch (error) {
                console.warn(`${signaler.name} failed:`, error);
            }
        }
        
        throw new Error('All signaling methods failed');
    }
    
    async establishConnection(targetPeerId, signaler) {
        const pc = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                // Include TURN servers for restrictive networks
                {
                    urls: 'turn:turn.example.com',
                    username: 'user',
                    credential: 'pass'
                }
            ]
        });
        
        // Create data channel
        const dc = pc.createDataChannel('chat');
        
        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        
        // Send offer via signaler
        await signaler.sendSignal(targetPeerId, {
            type: 'offer',
            offer: offer
        });
        
        // Wait for answer
        const answer = await signaler.waitForSignal('answer', 30000);
        await pc.setRemoteDescription(answer);
        
        // Exchange ICE candidates
        pc.onicecandidate = (event) => {
            if (event.candidate) {
                signaler.sendSignal(targetPeerId, {
                    type: 'ice-candidate',
                    candidate: event.candidate
                });
            }
        };
        
        // Handle incoming ICE candidates
        signaler.onSignal('ice-candidate', async (data) => {
            await pc.addIceCandidate(data.candidate);
        });
        
        // Wait for connection
        await this.waitForConnection(pc);
        
        return pc;
    }
    
    waitForConnection(pc) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Connection timeout'));
            }, 30000);
            
            pc.onconnectionstatechange = () => {
                if (pc.connectionState === 'connected') {
                    clearTimeout(timeout);
                    resolve();
                } else if (pc.connectionState === 'failed') {
                    clearTimeout(timeout);
                    reject(new Error('Connection failed'));
                }
            };
        });
    }
}
```

## Best Practices for Restrictive Environments

### 1. Use Multiple STUN Servers
```javascript
const rtcConfig = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' },
        { urls: 'stun:stun3.l.google.com:19302' },
        { urls: 'stun:stun4.l.google.com:19302' },
        { urls: 'stun:stun.services.mozilla.com' },
        { urls: 'stun:stun.stunprotocol.org:3478' }
    ]
};
```

### 2. Always Include TURN Servers
```javascript
// Free TURN servers (for testing only)
const rtcConfig = {
    iceServers: [
        {
            urls: 'turn:openrelay.metered.ca:80',
            username: 'openrelayproject',
            credential: 'openrelayproject'
        },
        {
            urls: 'turn:openrelay.metered.ca:443',
            username: 'openrelayproject',
            credential: 'openrelayproject'
        }
    ]
};
```

### 3. Implement Connection Retry Logic
```javascript
async function connectWithRetry(targetId, maxAttempts = 3) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            console.log(`Connection attempt ${attempt}/${maxAttempts}`);
            const pc = await connect(targetId);
            return pc;
        } catch (error) {
            console.error(`Attempt ${attempt} failed:`, error);
            
            if (attempt < maxAttempts) {
                // Exponential backoff
                const delay = Math.pow(2, attempt) * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }
    
    throw new Error('All connection attempts failed');
}
```

### 4. Handle Network Changes
```javascript
// Monitor network changes and reconnect
class ResilientWebRTC {
    constructor() {
        this.setupNetworkMonitoring();
    }
    
    setupNetworkMonitoring() {
        // Monitor online/offline
        window.addEventListener('online', () => {
            console.log('Network restored, reconnecting...');
            this.reconnectAll();
        });
        
        // Monitor connection changes
        if ('connection' in navigator) {
            navigator.connection.addEventListener('change', () => {
                console.log('Network changed:', navigator.connection.effectiveType);
                this.optimizeConnections();
            });
        }
    }
    
    async reconnectAll() {
        for (const [peerId, pc] of this.peers) {
            if (pc.connectionState !== 'connected') {
                await this.reconnect(peerId);
            }
        }
    }
}
```

## Deployment Scenarios

### AWS Lambda / Serverless
```javascript
// Lambda function that only makes outgoing connections
exports.handler = async (event) => {
    const signaler = new HTTPSignaler('https://signaling-service.com');
    
    // Lambda connects OUT to signaling service
    await signaler.relaySignal(event.from, event.to, event.signal);
    
    return {
        statusCode: 200,
        body: JSON.stringify({ success: true })
    };
};
```

### Docker Container with Egress Only
```dockerfile
# Dockerfile for outgoing-only WebRTC service
FROM node:18-alpine

# Configure firewall to block incoming connections
RUN apk add --no-cache iptables
RUN iptables -A INPUT -j DROP
RUN iptables -A OUTPUT -j ACCEPT

WORKDIR /app
COPY . .
RUN npm install

# Service connects out to external signaling
CMD ["node", "outgoing-only-server.js"]
```

### Corporate Proxy Environment
```javascript
// Configure WebRTC to work through corporate proxy
const { HttpsProxyAgent } = require('https-proxy-agent');

const agent = new HttpsProxyAgent('http://corporate-proxy:8080');

// Use proxy for signaling connections
const socket = io('https://external-signaling.com', {
    agent: agent,
    transports: ['polling'] // WebSocket might be blocked
});
```

## Summary

When restricted to outgoing connections only:

1. **Use external signaling services** (PeerJS, Socket.io, Firebase)
2. **Implement multiple fallback methods**
3. **Always include TURN servers** for relay
4. **Handle connection failures gracefully**
5. **Consider manual exchange** for ultimate compatibility

The key is to be creative with signaling while maintaining the peer-to-peer benefits of WebRTC for the actual data transfer.