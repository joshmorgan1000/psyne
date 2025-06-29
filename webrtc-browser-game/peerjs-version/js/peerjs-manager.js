/**
 * @file peerjs-manager.js
 * @brief PeerJS-based WebRTC manager for P2P gaming
 * 
 * Uses PeerJS for simplified WebRTC connections with:
 * - Hosted broker service (no server required)
 * - Automatic NAT traversal
 * - Simple peer discovery
 * - Room-based connections via shared room codes
 */

class PeerJSManager {
    constructor() {
        this.peer = null;
        this.connections = new Map();
        this.localPlayerId = null;
        this.currentRoom = null;
        this.isHost = false;
        
        // Event callbacks
        this.onPeerConnected = null;
        this.onPeerDisconnected = null;
        this.onGameMessage = null;
        this.onChatMessage = null;
        this.onReady = null;
        
        // Stats tracking
        this.stats = {
            packetsReceived: 0,
            packetsSent: 0,
            averageLatency: 0,
            connectionCount: 0
        };
        
        // Room management for discovery
        this.knownPeers = new Set();
        this.roomPeers = new Map(); // roomCode -> Set of peer IDs
        
        this.initializePeer();
    }
    
    initializePeer() {
        console.log('🔌 Initializing PeerJS connection...');
        
        // Create peer with random ID or use existing one
        this.peer = new Peer({
            config: {
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' },
                    { urls: 'stun:stun1.l.google.com:19302' }
                ]
            },
            debug: 2 // Enable debug logging
        });
        
        this.peer.on('open', (id) => {
            this.localPlayerId = id;
            console.log('✅ PeerJS connected with ID:', id);
            this.updateConnectionStatus('connected');
            
            if (this.onReady) {
                this.onReady(id);
            }
        });
        
        this.peer.on('connection', (conn) => {
            console.log('📞 Incoming connection from:', conn.peer);
            this.setupConnection(conn);
        });
        
        this.peer.on('error', (error) => {
            console.error('🚨 PeerJS error:', error);
            this.updateConnectionStatus('error');
            
            // Auto-reconnect on some errors
            if (error.type === 'network' || error.type === 'server-error') {
                setTimeout(() => {
                    console.log('🔄 Attempting to reconnect...');
                    this.initializePeer();
                }, 3000);
            }
        });
        
        this.peer.on('disconnected', () => {
            console.log('❌ PeerJS disconnected');
            this.updateConnectionStatus('disconnected');
            
            // Try to reconnect
            if (!this.peer.destroyed) {
                this.peer.reconnect();
            }
        });
    }
    
    // Connect to a specific peer by ID
    connectToPeer(peerId) {
        if (peerId === this.localPlayerId) {
            console.warn('⚠️ Cannot connect to self');
            return false;
        }
        
        if (this.connections.has(peerId)) {
            console.warn('⚠️ Already connected to peer:', peerId);
            return false;
        }
        
        console.log('🤝 Connecting to peer:', peerId);
        
        const conn = this.peer.connect(peerId, {
            label: 'game-data',
            serialization: 'json',
            reliable: false // Fast delivery for game data
        });
        
        this.setupConnection(conn);
        return true;
    }
    
    // Join a room by connecting to all peers with the same room code
    async joinRoom(roomCode, playerName) {
        this.currentRoom = roomCode;
        
        // Store our room association
        const roomKey = `room_${roomCode}`;
        
        // In a real implementation, you'd use a discovery service
        // For now, we'll implement a simple broadcast discovery
        console.log('🏟️ Joining room:', roomCode);
        
        // Try to connect to known room peers
        this.discoverRoomPeers(roomCode);
        
        return true;
    }
    
    // Simple room discovery using localStorage (works for same-origin tabs)
    discoverRoomPeers(roomCode) {
        const roomKey = `peerjs_room_${roomCode}`;
        
        // Get existing room members
        let roomData = {};
        try {
            roomData = JSON.parse(localStorage.getItem(roomKey) || '{}');
        } catch (e) {
            roomData = {};
        }
        
        // Add ourselves to the room
        roomData[this.localPlayerId] = {
            id: this.localPlayerId,
            timestamp: Date.now(),
            playerName: window.gameEngine?.localPlayer?.name || 'Unknown'
        };
        
        // Clean up old entries (older than 5 minutes)
        const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
        Object.keys(roomData).forEach(peerId => {
            if (roomData[peerId].timestamp < fiveMinutesAgo) {
                delete roomData[peerId];
            }
        });
        
        // Save updated room data
        localStorage.setItem(roomKey, JSON.stringify(roomData));
        
        // Connect to other room members
        Object.keys(roomData).forEach(peerId => {
            if (peerId !== this.localPlayerId && !this.connections.has(peerId)) {
                console.log('🔗 Attempting connection to room peer:', peerId);
                this.connectToPeer(peerId);
            }
        });
        
        // Set up periodic room discovery
        if (!this.roomDiscoveryInterval) {
            this.roomDiscoveryInterval = setInterval(() => {
                if (this.currentRoom) {
                    this.discoverRoomPeers(this.currentRoom);
                }
            }, 10000); // Check every 10 seconds
        }
    }
    
    setupConnection(conn) {
        this.connections.set(conn.peer, {
            connection: conn,
            lastSeen: Date.now(),
            latency: 0,
            state: 'connecting'
        });
        
        conn.on('open', () => {
            console.log('✅ Connection opened with:', conn.peer);
            const connInfo = this.connections.get(conn.peer);
            if (connInfo) {
                connInfo.state = 'connected';
            }
            
            if (this.onPeerConnected) {
                this.onPeerConnected(conn.peer);
            }
            
            this.stats.connectionCount++;
            this.updatePeerConnectionsDisplay();
        });
        
        conn.on('data', (data) => {
            this.handleMessage(conn.peer, data);
        });
        
        conn.on('close', () => {
            console.log('👋 Connection closed with:', conn.peer);
            this.connections.delete(conn.peer);
            
            if (this.onPeerDisconnected) {
                this.onPeerDisconnected(conn.peer);
            }
            
            this.updatePeerConnectionsDisplay();
        });
        
        conn.on('error', (error) => {
            console.error('🚨 Connection error with', conn.peer, ':', error);
            this.connections.delete(conn.peer);
            
            if (this.onPeerDisconnected) {
                this.onPeerDisconnected(conn.peer);
            }
        });
    }
    
    handleMessage(peerId, data) {
        this.stats.packetsReceived++;
        
        // Update last seen
        const connInfo = this.connections.get(peerId);
        if (connInfo) {
            connInfo.lastSeen = Date.now();
        }
        
        // Add sender info
        data.fromPeer = peerId;
        data.receivedAt = performance.now();
        
        // Handle latency measurement
        if (data.type === 'ping') {
            this.sendToPeer(peerId, {
                type: 'pong',
                originalTimestamp: data.timestamp,
                timestamp: performance.now()
            });
            return;
        } else if (data.type === 'pong') {
            const latency = performance.now() - data.originalTimestamp;
            if (connInfo) {
                connInfo.latency = latency;
                this.stats.averageLatency = (this.stats.averageLatency + latency) / 2;
            }
            return;
        }
        
        // Route messages based on type
        if (data.type === 'chat' && this.onChatMessage) {
            this.onChatMessage(data);
        } else if (this.onGameMessage) {
            this.onGameMessage(data);
        }
    }
    
    sendToPeer(peerId, message) {
        const connInfo = this.connections.get(peerId);
        if (!connInfo || connInfo.state !== 'connected') {
            return false;
        }
        
        const conn = connInfo.connection;
        if (conn.open) {
            try {
                message.timestamp = performance.now();
                message.fromPlayer = this.localPlayerId;
                conn.send(message);
                this.stats.packetsSent++;
                return true;
            } catch (error) {
                console.error(`🚨 Failed to send to ${peerId}:`, error);
                return false;
            }
        }
        return false;
    }
    
    broadcastToAllPeers(message) {
        let successCount = 0;
        for (const peerId of this.connections.keys()) {
            if (this.sendToPeer(peerId, message)) {
                successCount++;
            }
        }
        return successCount;
    }
    
    measureLatency() {
        for (const peerId of this.connections.keys()) {
            this.sendToPeer(peerId, {
                type: 'ping',
                timestamp: performance.now()
            });
        }
    }
    
    getConnectedPeers() {
        return Array.from(this.connections.keys()).filter(peerId => {
            const connInfo = this.connections.get(peerId);
            return connInfo && connInfo.state === 'connected';
        });
    }
    
    getPeerLatency(peerId) {
        const connInfo = this.connections.get(peerId);
        return connInfo ? connInfo.latency : 0;
    }
    
    updateConnectionStatus(status) {
        const statusEl = document.getElementById('connectionStatus');
        if (statusEl) {
            statusEl.className = `status-${status}`;
            statusEl.textContent = status.toUpperCase();
        }
        
        // Update peer ID displays
        const myPeerIdEl = document.getElementById('myPeerId');
        const yourPeerIdEl = document.getElementById('yourPeerId');
        
        if (this.localPlayerId) {
            const displayId = this.localPlayerId;
            if (myPeerIdEl) myPeerIdEl.textContent = displayId;
            if (yourPeerIdEl) yourPeerIdEl.textContent = displayId;
        }
    }
    
    updatePeerConnectionsDisplay() {
        const container = document.getElementById('peerConnections');
        if (!container) return;
        
        const connectedPeers = this.getConnectedPeers();
        
        container.innerHTML = '';
        container.innerHTML = `<div style="color: #00ff41; font-weight: bold;">P2P Connections:</div>`;
        
        if (connectedPeers.length === 0) {
            container.innerHTML += '<div style="color: #666;">No direct connections</div>';
            return;
        }
        
        connectedPeers.forEach(peerId => {
            const latency = this.getPeerLatency(peerId);
            const peerDiv = document.createElement('div');
            peerDiv.className = 'peer-connection peer-connected';
            peerDiv.innerHTML = `📡 ${peerId.slice(0, 8)}... (${Math.round(latency)}ms)`;
            container.appendChild(peerDiv);
        });
    }
    
    getStats() {
        return {
            ...this.stats,
            connectedPeers: this.getConnectedPeers().length,
            totalConnections: this.connections.size,
            myPeerId: this.localPlayerId,
            currentRoom: this.currentRoom
        };
    }
    
    disconnect() {
        // Close all connections
        for (const [peerId, connInfo] of this.connections) {
            if (connInfo.connection.open) {
                connInfo.connection.close();
            }
        }
        this.connections.clear();
        
        // Clear room discovery
        if (this.roomDiscoveryInterval) {
            clearInterval(this.roomDiscoveryInterval);
            this.roomDiscoveryInterval = null;
        }
        
        // Destroy peer
        if (this.peer && !this.peer.destroyed) {
            this.peer.destroy();
        }
    }
    
    // Room management helpers
    listAvailableRooms() {
        const rooms = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith('peerjs_room_')) {
                const roomCode = key.replace('peerjs_room_', '');
                try {
                    const roomData = JSON.parse(localStorage.getItem(key) || '{}');
                    const activePeers = Object.values(roomData).filter(peer => 
                        Date.now() - peer.timestamp < 5 * 60 * 1000 // Active in last 5 minutes
                    );
                    
                    if (activePeers.length > 0) {
                        rooms.push({
                            code: roomCode,
                            playerCount: activePeers.length,
                            players: activePeers
                        });
                    }
                } catch (e) {
                    // Invalid room data, skip
                }
            }
        }
        return rooms;
    }
    
    // Generate a random room code
    generateRoomCode() {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
        let result = '';
        for (let i = 0; i < 6; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }
}