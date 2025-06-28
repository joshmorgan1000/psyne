/**
 * @file gossip-protocol.js
 * @brief Gossip protocol implementation for decentralized peer discovery
 * 
 * Features:
 * - Epidemic information spreading
 * - Automatic peer discovery
 * - Network partition recovery
 * - Load balancing across connections
 */

class GossipProtocol {
    constructor(webrtcManager) {
        this.webrtcManager = webrtcManager;
        
        // Gossip configuration
        this.config = {
            gossipInterval: 5000,        // Send gossip every 5 seconds
            maxPeersPerGossip: 3,        // Forward to max 3 peers
            heartbeatInterval: 10000,     // Heartbeat every 10 seconds
            peerTimeout: 30000,          // Consider peer dead after 30s
            maxPeerHistory: 100,         // Remember up to 100 peers
            discoveryRadius: 3           // How many hops to discover peers
        };
        
        // Peer knowledge base
        this.knownPeers = new Map();
        this.peerHistory = new Map();
        this.lastGossipTime = 0;
        this.lastHeartbeat = 0;
        
        // Gossip message types
        this.messageTypes = {
            PEER_DISCOVERY: 'peer_discovery',
            PEER_ANNOUNCEMENT: 'peer_announcement',
            HEARTBEAT: 'heartbeat',
            GAME_STATE_SYNC: 'game_state_sync',
            ROOM_INFO: 'room_info',
            NETWORK_TOPOLOGY: 'network_topology'
        };
        
        // Current network view
        this.networkTopology = {
            nodes: new Map(),
            edges: new Map(),
            lastUpdate: 0
        };
        
        this.startGossipLoop();
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        // Handle WebRTC manager events
        this.webrtcManager.onPeerConnected = (peerId) => {
            this.onPeerConnected(peerId);
        };
        
        this.webrtcManager.onPeerDisconnected = (peerId) => {
            this.onPeerDisconnected(peerId);
        };
        
        this.webrtcManager.onGameMessage = (message) => {
            if (message.type && message.type.startsWith('gossip_')) {
                this.handleGossipMessage(message);
            }
        };
    }
    
    startGossipLoop() {
        setInterval(() => {
            this.performGossipRound();
        }, this.config.gossipInterval);
        
        setInterval(() => {
            this.sendHeartbeats();
        }, this.config.heartbeatInterval);
        
        setInterval(() => {
            this.cleanupStaleePeers();
        }, this.config.peerTimeout);
    }
    
    onPeerConnected(peerId) {
        console.log('🤝 Gossip: New peer connected:', peerId);
        
        // Add to known peers
        this.knownPeers.set(peerId, {
            id: peerId,
            connectedAt: Date.now(),
            lastSeen: Date.now(),
            hops: 0, // Direct connection
            via: null, // Directly connected
            status: 'connected',
            capabilities: [],
            metadata: {}
        });
        
        // Update network topology
        this.updateNetworkTopology();
        
        // Send our peer list to the new peer
        this.sendPeerDiscovery(peerId);
        
        // Announce ourselves to the network
        this.announceToNetwork();
    }
    
    onPeerDisconnected(peerId) {
        console.log('👋 Gossip: Peer disconnected:', peerId);
        
        const peer = this.knownPeers.get(peerId);
        if (peer) {
            peer.status = 'disconnected';
            peer.disconnectedAt = Date.now();
            
            // Move to history
            this.peerHistory.set(peerId, peer);
            this.knownPeers.delete(peerId);
            
            // Update topology
            this.updateNetworkTopology();
        }
        
        // Try to discover alternative connections
        this.requestPeerDiscovery();
    }
    
    performGossipRound() {
        const now = Date.now();
        if (now - this.lastGossipTime < this.config.gossipInterval) {
            return;
        }
        
        this.lastGossipTime = now;
        
        // Select random peers to gossip with
        const connectedPeers = this.webrtcManager.getConnectedPeers();
        const gossipTargets = this.selectGossipTargets(connectedPeers);
        
        for (const peerId of gossipTargets) {
            this.sendGossipMessage(peerId, {
                type: 'gossip_' + this.messageTypes.NETWORK_TOPOLOGY,
                topology: this.serializeTopology(),
                peerList: this.serializeKnownPeers(),
                timestamp: now,
                ttl: this.config.discoveryRadius
            });
        }
        
        console.log(`💬 Gossip round: sent to ${gossipTargets.length} peers`);
    }
    
    selectGossipTargets(availablePeers) {
        const maxTargets = Math.min(this.config.maxPeersPerGossip, availablePeers.length);
        const shuffled = [...availablePeers].sort(() => Math.random() - 0.5);
        return shuffled.slice(0, maxTargets);
    }
    
    sendGossipMessage(peerId, message) {
        this.webrtcManager.sendToPeer(peerId, 'game', message);
    }
    
    sendPeerDiscovery(targetPeerId) {
        const knownPeersList = Array.from(this.knownPeers.values())
            .filter(peer => peer.id !== targetPeerId && peer.status === 'connected')
            .map(peer => ({
                id: peer.id,
                hops: peer.hops + 1,
                via: this.webrtcManager.localPlayerId,
                capabilities: peer.capabilities,
                lastSeen: peer.lastSeen
            }));
        
        this.sendGossipMessage(targetPeerId, {
            type: 'gossip_' + this.messageTypes.PEER_DISCOVERY,
            peers: knownPeersList,
            sender: this.webrtcManager.localPlayerId,
            timestamp: Date.now()
        });
    }
    
    announceToNetwork() {
        const announcement = {
            type: 'gossip_' + this.messageTypes.PEER_ANNOUNCEMENT,
            peerId: this.webrtcManager.localPlayerId,
            capabilities: ['combat', 'chat', 'game_state'],
            metadata: {
                gameVersion: '1.0.0',
                playerName: window.gameEngine?.localPlayer?.name || 'Unknown',
                roomId: this.webrtcManager.currentRoom
            },
            timestamp: Date.now(),
            ttl: this.config.discoveryRadius
        };
        
        this.webrtcManager.broadcastToAllPeers('game', announcement);
    }
    
    requestPeerDiscovery() {
        const request = {
            type: 'gossip_' + this.messageTypes.PEER_DISCOVERY,
            request: true,
            sender: this.webrtcManager.localPlayerId,
            timestamp: Date.now()
        };
        
        this.webrtcManager.broadcastToAllPeers('game', request);
    }
    
    sendHeartbeats() {
        const now = Date.now();
        if (now - this.lastHeartbeat < this.config.heartbeatInterval) {
            return;
        }
        
        this.lastHeartbeat = now;
        
        const heartbeat = {
            type: 'gossip_' + this.messageTypes.HEARTBEAT,
            sender: this.webrtcManager.localPlayerId,
            timestamp: now,
            peerCount: this.knownPeers.size,
            uptime: now - (window.gameStartTime || now)
        };
        
        this.webrtcManager.broadcastToAllPeers('game', heartbeat);
    }
    
    handleGossipMessage(message) {
        const messageType = message.type.replace('gossip_', '');
        
        switch (messageType) {
            case this.messageTypes.PEER_DISCOVERY:
                this.handlePeerDiscovery(message);
                break;
                
            case this.messageTypes.PEER_ANNOUNCEMENT:
                this.handlePeerAnnouncement(message);
                break;
                
            case this.messageTypes.HEARTBEAT:
                this.handleHeartbeat(message);
                break;
                
            case this.messageTypes.NETWORK_TOPOLOGY:
                this.handleNetworkTopology(message);
                break;
                
            case this.messageTypes.GAME_STATE_SYNC:
                this.handleGameStateSync(message);
                break;
                
            default:
                console.log('🤷 Unknown gossip message type:', messageType);
        }
    }
    
    handlePeerDiscovery(message) {
        console.log('🔍 Received peer discovery from:', message.sender);
        
        if (message.request) {
            // Someone is requesting peer discovery
            this.sendPeerDiscovery(message.fromPeer);
            return;
        }
        
        if (message.peers) {
            // Process discovered peers
            for (const peerInfo of message.peers) {
                if (peerInfo.id === this.webrtcManager.localPlayerId) {
                    continue; // Skip ourselves
                }
                
                const existingPeer = this.knownPeers.get(peerInfo.id);
                
                if (!existingPeer || peerInfo.hops < existingPeer.hops) {
                    // New peer or better route found
                    this.knownPeers.set(peerInfo.id, {
                        id: peerInfo.id,
                        hops: peerInfo.hops,
                        via: peerInfo.via,
                        lastSeen: peerInfo.lastSeen,
                        discoveredAt: Date.now(),
                        status: 'discovered',
                        capabilities: peerInfo.capabilities || []
                    });
                    
                    console.log(`📍 Discovered peer ${peerInfo.id} via ${peerInfo.via} (${peerInfo.hops} hops)`);
                    
                    // Try to connect if we have room and they're close
                    if (peerInfo.hops <= 2 && this.shouldConnectToPeer(peerInfo)) {
                        this.initiateConnectionToPeer(peerInfo.id);
                    }
                }
            }
        }
    }
    
    handlePeerAnnouncement(message) {
        console.log('📢 Peer announcement from:', message.peerId);
        
        // Update peer info
        let peer = this.knownPeers.get(message.peerId);
        if (!peer) {
            peer = {
                id: message.peerId,
                hops: 1, // Announced, so at least 1 hop
                via: message.fromPeer,
                status: 'announced'
            };
            this.knownPeers.set(message.peerId, peer);
        }
        
        peer.capabilities = message.capabilities || [];
        peer.metadata = message.metadata || {};
        peer.lastSeen = message.timestamp;
        
        // Forward announcement to other peers (with TTL)
        if (message.ttl > 1) {
            const forwardMessage = {
                ...message,
                ttl: message.ttl - 1,
                forwardedBy: this.webrtcManager.localPlayerId
            };
            
            const connectedPeers = this.webrtcManager.getConnectedPeers()
                .filter(id => id !== message.fromPeer); // Don't send back to sender
            
            const forwardTargets = this.selectGossipTargets(connectedPeers);
            for (const peerId of forwardTargets) {
                this.sendGossipMessage(peerId, forwardMessage);
            }
        }
    }
    
    handleHeartbeat(message) {
        const peer = this.knownPeers.get(message.sender);
        if (peer) {
            peer.lastSeen = message.timestamp;
            peer.peerCount = message.peerCount;
            peer.uptime = message.uptime;
        }
    }
    
    handleNetworkTopology(message) {
        if (message.topology) {
            this.mergeTopology(message.topology);
        }
        
        if (message.peerList) {
            this.mergePeerList(message.peerList);
        }
    }
    
    handleGameStateSync(message) {
        // Handle game state synchronization
        if (window.gameEngine && message.gameState) {
            // Sync critical game state information
            console.log('🔄 Syncing game state via gossip');
        }
    }
    
    shouldConnectToPeer(peerInfo) {
        const connectedCount = this.webrtcManager.getConnectedPeers().length;
        const maxConnections = 8; // Reasonable limit for P2P mesh
        
        // Don't connect if we're at capacity
        if (connectedCount >= maxConnections) {
            return false;
        }
        
        // Prefer peers with compatible capabilities
        const hasCompatibleCapabilities = peerInfo.capabilities.some(cap => 
            ['combat', 'game_state'].includes(cap)
        );
        
        return hasCompatibleCapabilities;
    }
    
    initiateConnectionToPeer(peerId) {
        // Check if we're already connected
        if (this.webrtcManager.peers.has(peerId)) {
            return;
        }
        
        console.log('🔗 Initiating connection to discovered peer:', peerId);
        
        // This would need to be implemented with proper signaling
        // For now, we just log the intention
        // In a real implementation, you'd need to:
        // 1. Find a route to the peer through the gossip network
        // 2. Send a connection request through intermediary peers
        // 3. Establish WebRTC connection once both sides agree
    }
    
    updateNetworkTopology() {
        const now = Date.now();
        
        // Update nodes
        this.networkTopology.nodes.clear();
        
        // Add ourselves
        this.networkTopology.nodes.set(this.webrtcManager.localPlayerId, {
            id: this.webrtcManager.localPlayerId,
            type: 'self',
            connections: this.webrtcManager.getConnectedPeers().length,
            lastUpdate: now
        });
        
        // Add known peers
        for (const [peerId, peer] of this.knownPeers) {
            this.networkTopology.nodes.set(peerId, {
                id: peerId,
                type: peer.status,
                hops: peer.hops,
                via: peer.via,
                lastSeen: peer.lastSeen
            });
        }
        
        // Update edges (connections)
        this.networkTopology.edges.clear();
        for (const peerId of this.webrtcManager.getConnectedPeers()) {
            this.networkTopology.edges.set(
                `${this.webrtcManager.localPlayerId}-${peerId}`,
                {
                    from: this.webrtcManager.localPlayerId,
                    to: peerId,
                    type: 'direct',
                    established: now
                }
            );
        }
        
        this.networkTopology.lastUpdate = now;
    }
    
    serializeTopology() {
        return {
            nodes: Array.from(this.networkTopology.nodes.values()),
            edges: Array.from(this.networkTopology.edges.values()),
            lastUpdate: this.networkTopology.lastUpdate
        };
    }
    
    serializeKnownPeers() {
        return Array.from(this.knownPeers.values()).map(peer => ({
            id: peer.id,
            hops: peer.hops,
            via: peer.via,
            status: peer.status,
            lastSeen: peer.lastSeen,
            capabilities: peer.capabilities
        }));
    }
    
    mergeTopology(remoteTopology) {
        // Merge remote network topology with our view
        if (remoteTopology.lastUpdate > this.networkTopology.lastUpdate) {
            // Remote topology is newer, consider merging
            for (const node of remoteTopology.nodes) {
                if (!this.networkTopology.nodes.has(node.id)) {
                    this.networkTopology.nodes.set(node.id, node);
                }
            }
            
            for (const edge of remoteTopology.edges) {
                const edgeId = `${edge.from}-${edge.to}`;
                if (!this.networkTopology.edges.has(edgeId)) {
                    this.networkTopology.edges.set(edgeId, edge);
                }
            }
        }
    }
    
    mergePeerList(remotePeerList) {
        for (const remotePeer of remotePeerList) {
            const existingPeer = this.knownPeers.get(remotePeer.id);
            
            if (!existingPeer || remotePeer.lastSeen > existingPeer.lastSeen) {
                this.knownPeers.set(remotePeer.id, {
                    ...remotePeer,
                    discoveredAt: Date.now()
                });
            }
        }
    }
    
    cleanupStaleePeers() {
        const now = Date.now();
        const staleThreshold = this.config.peerTimeout;
        
        for (const [peerId, peer] of this.knownPeers) {
            if (now - peer.lastSeen > staleThreshold) {
                console.log('🧹 Removing stale peer:', peerId);
                this.knownPeers.delete(peerId);
                
                // Move to history
                this.peerHistory.set(peerId, {
                    ...peer,
                    removedAt: now,
                    reason: 'timeout'
                });
            }
        }
        
        // Limit history size
        if (this.peerHistory.size > this.config.maxPeerHistory) {
            const oldestEntries = Array.from(this.peerHistory.entries())
                .sort((a, b) => a[1].removedAt - b[1].removedAt);
            
            const toRemove = oldestEntries.slice(0, this.peerHistory.size - this.config.maxPeerHistory);
            for (const [peerId] of toRemove) {
                this.peerHistory.delete(peerId);
            }
        }
    }
    
    getNetworkStats() {
        return {
            knownPeers: this.knownPeers.size,
            connectedPeers: this.webrtcManager.getConnectedPeers().length,
            peerHistory: this.peerHistory.size,
            networkNodes: this.networkTopology.nodes.size,
            networkEdges: this.networkTopology.edges.size,
            lastTopologyUpdate: this.networkTopology.lastUpdate
        };
    }
    
    getConnectedPeersList() {
        return Array.from(this.knownPeers.values())
            .filter(peer => peer.status === 'connected')
            .map(peer => ({
                id: peer.id,
                hops: peer.hops,
                latency: this.webrtcManager.getPeerLatency(peer.id),
                capabilities: peer.capabilities,
                metadata: peer.metadata
            }));
    }
    
    // Network visualization data
    getNetworkVisualization() {
        return {
            nodes: Array.from(this.networkTopology.nodes.values()),
            edges: Array.from(this.networkTopology.edges.values()),
            center: this.webrtcManager.localPlayerId
        };
    }
}