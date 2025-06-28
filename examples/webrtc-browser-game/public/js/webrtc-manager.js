/**
 * @file webrtc-manager.js
 * @brief WebRTC P2P connection management for real-time gaming
 * 
 * Handles:
 * - ICE/STUN NAT traversal
 * - Data channel setup (ordered/unordered)
 * - Automatic reconnection
 * - Latency monitoring
 */

class WebRTCManager {
    constructor() {
        this.peers = new Map();
        this.localPlayerId = null;
        this.signalingSocket = null;
        this.currentRoom = null;
        
        // WebRTC configuration with STUN servers
        this.rtcConfig = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' },
                { urls: 'stun:stun.stunprotocol.org:3478' }
            ],
            iceCandidatePoolSize: 10
        };
        
        // Event callbacks
        this.onPeerConnected = null;
        this.onPeerDisconnected = null;
        this.onGameMessage = null;
        this.onChatMessage = null;
        this.onRoomUpdate = null;
        
        // Stats tracking
        this.stats = {
            packetsReceived: 0,
            packetsSent: 0,
            lastLatencyCheck: 0,
            averageLatency: 0
        };
        
        this.connectToSignalingServer();
    }
    
    connectToSignalingServer() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}`;
        
        console.log('🔌 Connecting to signaling server:', wsUrl);
        
        this.signalingSocket = new WebSocket(wsUrl);
        
        this.signalingSocket.onopen = () => {
            console.log('✅ Connected to signaling server');
            this.updateConnectionStatus('connected');
        };
        
        this.signalingSocket.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleSignalingMessage(message);
        };
        
        this.signalingSocket.onclose = () => {
            console.log('❌ Signaling server disconnected');
            this.updateConnectionStatus('disconnected');
            
            // Auto-reconnect after 3 seconds
            setTimeout(() => {
                if (this.signalingSocket.readyState === WebSocket.CLOSED) {
                    this.connectToSignalingServer();
                }
            }, 3000);
        };
        
        this.signalingSocket.onerror = (error) => {
            console.error('🚨 Signaling error:', error);
        };
    }
    
    handleSignalingMessage(message) {
        switch (message.type) {
            case 'connected':
                this.localPlayerId = message.playerId;
                console.log('🎮 Player ID assigned:', this.localPlayerId);
                break;
                
            case 'roomJoined':
                this.currentRoom = message.roomId;
                console.log('🏟️ Joined room:', message.roomId);
                
                // Connect to existing players
                for (const player of message.playerList) {
                    if (player.id !== this.localPlayerId) {
                        this.connectToPeer(player.id, true); // We initiate
                    }
                }
                
                if (this.onRoomUpdate) {
                    this.onRoomUpdate(message);
                }
                break;
                
            case 'playerJoined':
                if (message.playerId !== this.localPlayerId) {
                    console.log('👋 New player joined:', message.playerId);
                    // Don't initiate - let them connect to us
                }
                break;
                
            case 'playerLeft':
                this.disconnectFromPeer(message.playerId);
                break;
                
            case 'webrtcSignaling':
                this.handleWebRTCSignaling(message);
                break;
                
            case 'roomList':
                if (this.onRoomUpdate) {
                    this.onRoomUpdate(message);
                }
                break;
                
            case 'pong':
                this.updateLatency(Date.now() - message.timestamp);
                break;
        }
    }
    
    async connectToPeer(peerId, isInitiator = false) {
        console.log(`🤝 ${isInitiator ? 'Initiating' : 'Receiving'} connection to/from:`, peerId);
        
        const peerConnection = new RTCPeerConnection(this.rtcConfig);
        
        const peerInfo = {
            id: peerId,
            connection: peerConnection,
            channels: {},
            isInitiator,
            state: 'connecting',
            lastSeen: Date.now(),
            latency: 0
        };
        
        this.peers.set(peerId, peerInfo);
        
        // Set up ICE candidate handling
        peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                this.sendSignalingMessage({
                    type: 'webrtcSignaling',
                    targetPlayerId: peerId,
                    signalingData: {
                        type: 'ice-candidate',
                        candidate: event.candidate
                    }
                });
            }
        };
        
        // Handle connection state changes
        peerConnection.onconnectionstatechange = () => {
            console.log(`🔗 Connection state with ${peerId}:`, peerConnection.connectionState);
            peerInfo.state = peerConnection.connectionState;
            
            if (peerConnection.connectionState === 'connected') {
                console.log(`✅ P2P connection established with ${peerId}`);
                if (this.onPeerConnected) {
                    this.onPeerConnected(peerId);
                }
            } else if (peerConnection.connectionState === 'disconnected' || 
                      peerConnection.connectionState === 'failed') {
                this.disconnectFromPeer(peerId);
            }
        };
        
        if (isInitiator) {
            // Create data channels (initiator only)
            await this.createDataChannels(peerInfo);
            
            // Create and send offer
            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);
            
            this.sendSignalingMessage({
                type: 'webrtcSignaling',
                targetPlayerId: peerId,
                signalingData: {
                    type: 'offer',
                    sdp: offer
                }
            });
        } else {
            // Handle incoming data channels
            peerConnection.ondatachannel = (event) => {
                this.setupDataChannel(peerInfo, event.channel);
            };
        }
    }
    
    async createDataChannels(peerInfo) {
        const pc = peerInfo.connection;
        
        // Game state channel (unordered, no retries for position updates)
        const gameChannel = pc.createDataChannel('game', {
            ordered: false,
            maxRetransmits: 0
        });
        
        // Chat channel (ordered, reliable)
        const chatChannel = pc.createDataChannel('chat', {
            ordered: true,
            maxRetransmits: 3
        });
        
        // Combat channel (ordered, reliable for important events)
        const combatChannel = pc.createDataChannel('combat', {
            ordered: true,
            maxRetransmits: 5
        });
        
        this.setupDataChannel(peerInfo, gameChannel);
        this.setupDataChannel(peerInfo, chatChannel);
        this.setupDataChannel(peerInfo, combatChannel);
    }
    
    setupDataChannel(peerInfo, channel) {
        peerInfo.channels[channel.label] = channel;
        
        channel.onopen = () => {
            console.log(`📡 Data channel '${channel.label}' opened with ${peerInfo.id}`);
        };
        
        channel.onmessage = (event) => {
            this.handleDataChannelMessage(peerInfo.id, channel.label, event.data);
            this.stats.packetsReceived++;
        };
        
        channel.onclose = () => {
            console.log(`📡 Data channel '${channel.label}' closed with ${peerInfo.id}`);
        };
        
        channel.onerror = (error) => {
            console.error(`🚨 Data channel error with ${peerInfo.id}:`, error);
        };
    }
    
    handleDataChannelMessage(peerId, channelType, data) {
        try {
            const message = JSON.parse(data);
            message.fromPeer = peerId;
            message.receivedAt = performance.now();
            
            // Update peer last seen
            const peer = this.peers.get(peerId);
            if (peer) {
                peer.lastSeen = Date.now();
            }
            
            switch (channelType) {
                case 'game':
                    if (this.onGameMessage) {
                        this.onGameMessage(message);
                    }
                    break;
                    
                case 'chat':
                    if (this.onChatMessage) {
                        this.onChatMessage(message);
                    }
                    break;
                    
                case 'combat':
                    if (this.onGameMessage) {
                        this.onGameMessage(message);
                    }
                    break;
            }
            
            // Handle latency measurement
            if (message.type === 'ping') {
                this.sendToPeer(peerId, 'game', {
                    type: 'pong',
                    originalTimestamp: message.timestamp,
                    timestamp: performance.now()
                });
            } else if (message.type === 'pong') {
                const latency = performance.now() - message.originalTimestamp;
                if (peer) {
                    peer.latency = latency;
                }
            }
            
        } catch (error) {
            console.error('🚨 Failed to parse data channel message:', error);
        }
    }
    
    async handleWebRTCSignaling(message) {
        const { fromPlayerId, signalingData } = message;
        
        let peerInfo = this.peers.get(fromPlayerId);
        if (!peerInfo) {
            // Create peer connection for incoming connection
            await this.connectToPeer(fromPlayerId, false);
            peerInfo = this.peers.get(fromPlayerId);
        }
        
        const pc = peerInfo.connection;
        
        try {
            switch (signalingData.type) {
                case 'offer':
                    await pc.setRemoteDescription(signalingData.sdp);
                    const answer = await pc.createAnswer();
                    await pc.setLocalDescription(answer);
                    
                    this.sendSignalingMessage({
                        type: 'webrtcSignaling',
                        targetPlayerId: fromPlayerId,
                        signalingData: {
                            type: 'answer',
                            sdp: answer
                        }
                    });
                    break;
                    
                case 'answer':
                    await pc.setRemoteDescription(signalingData.sdp);
                    break;
                    
                case 'ice-candidate':
                    await pc.addIceCandidate(signalingData.candidate);
                    break;
            }
        } catch (error) {
            console.error('🚨 WebRTC signaling error:', error);
        }
    }
    
    sendToPeer(peerId, channelType, message) {
        const peer = this.peers.get(peerId);
        if (!peer || !peer.channels[channelType]) {
            return false;
        }
        
        const channel = peer.channels[channelType];
        if (channel.readyState === 'open') {
            try {
                message.timestamp = performance.now();
                message.fromPlayer = this.localPlayerId;
                channel.send(JSON.stringify(message));
                this.stats.packetsSent++;
                return true;
            } catch (error) {
                console.error(`🚨 Failed to send to ${peerId}:`, error);
                return false;
            }
        }
        return false;
    }
    
    broadcastToAllPeers(channelType, message) {
        let successCount = 0;
        for (const peerId of this.peers.keys()) {
            if (this.sendToPeer(peerId, channelType, message)) {
                successCount++;
            }
        }
        return successCount;
    }
    
    sendSignalingMessage(message) {
        if (this.signalingSocket && this.signalingSocket.readyState === WebSocket.OPEN) {
            this.signalingSocket.send(JSON.stringify(message));
        }
    }
    
    disconnectFromPeer(peerId) {
        const peer = this.peers.get(peerId);
        if (peer) {
            console.log('👋 Disconnecting from peer:', peerId);
            
            // Close all data channels
            for (const channel of Object.values(peer.channels)) {
                if (channel.readyState === 'open') {
                    channel.close();
                }
            }
            
            // Close peer connection
            if (peer.connection) {
                peer.connection.close();
            }
            
            this.peers.delete(peerId);
            
            if (this.onPeerDisconnected) {
                this.onPeerDisconnected(peerId);
            }
        }
    }
    
    joinRoom(roomId, playerName) {
        this.sendSignalingMessage({
            type: 'joinRoom',
            roomId,
            playerName
        });
    }
    
    createRoom(maxPlayers = 16) {
        this.sendSignalingMessage({
            type: 'createRoom',
            maxPlayers
        });
    }
    
    listRooms() {
        this.sendSignalingMessage({
            type: 'listRooms'
        });
    }
    
    updateConnectionStatus(status) {
        const statusEl = document.getElementById('connectionStatus');
        if (statusEl) {
            statusEl.className = `status-${status}`;
            statusEl.textContent = status.toUpperCase();
        }
    }
    
    updateLatency(latency) {
        this.stats.averageLatency = (this.stats.averageLatency + latency) / 2;
    }
    
    measureLatency() {
        // Ping all connected peers
        for (const peerId of this.peers.keys()) {
            this.sendToPeer(peerId, 'game', {
                type: 'ping',
                timestamp: performance.now()
            });
        }
        
        // Ping signaling server
        this.sendSignalingMessage({
            type: 'ping',
            timestamp: Date.now()
        });
    }
    
    getConnectedPeers() {
        return Array.from(this.peers.keys()).filter(peerId => {
            const peer = this.peers.get(peerId);
            return peer && peer.connection.connectionState === 'connected';
        });
    }
    
    getPeerLatency(peerId) {
        const peer = this.peers.get(peerId);
        return peer ? peer.latency : 0;
    }
    
    getStats() {
        return {
            ...this.stats,
            connectedPeers: this.getConnectedPeers().length,
            totalPeers: this.peers.size
        };
    }
    
    disconnect() {
        // Close all peer connections
        for (const peerId of this.peers.keys()) {
            this.disconnectFromPeer(peerId);
        }
        
        // Close signaling socket
        if (this.signalingSocket) {
            this.signalingSocket.close();
        }
    }
}