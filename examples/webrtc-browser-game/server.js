/**
 * @file server.js
 * @brief Minimal WebRTC signaling server for P2P gaming
 * 
 * This lightweight server only handles:
 * 1. Initial peer discovery and room management
 * 2. WebRTC signaling (SDP exchange, ICE candidates)
 * 3. Gossip protocol bootstrap
 * 
 * All game traffic flows directly P2P between browsers!
 */

const express = require('express');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8080;

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Game rooms and players
const gameRooms = new Map();
const playerSessions = new Map();

class GameRoom {
    constructor(roomId, maxPlayers = 16) {
        this.id = roomId;
        this.maxPlayers = maxPlayers;
        this.players = new Map();
        this.createdAt = Date.now();
        this.gameState = 'waiting'; // waiting, active, ended
    }
    
    addPlayer(playerId, playerData) {
        if (this.players.size >= this.maxPlayers) {
            return false;
        }
        
        this.players.set(playerId, {
            ...playerData,
            joinedAt: Date.now(),
            lastSeen: Date.now()
        });
        
        console.log(`🎮 Player ${playerId} joined room ${this.id} (${this.players.size}/${this.maxPlayers})`);
        return true;
    }
    
    removePlayer(playerId) {
        const removed = this.players.delete(playerId);
        if (removed) {
            console.log(`👋 Player ${playerId} left room ${this.id} (${this.players.size}/${this.maxPlayers})`);
        }
        return removed;
    }
    
    getPlayerList() {
        return Array.from(this.players.entries()).map(([id, data]) => ({
            id,
            name: data.name,
            position: data.position
        }));
    }
    
    isEmpty() {
        return this.players.size === 0;
    }
}

// Start HTTP server
const server = app.listen(PORT, () => {
    console.log(`🚀 WebRTC P2P Combat Game Server`);
    console.log(`   HTTP: http://localhost:${PORT}`);
    console.log(`   WebSocket: ws://localhost:${PORT}`);
    console.log(`   Ready for P2P gaming! 🎮`);
});

// WebSocket server for signaling
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws, req) => {
    const playerId = uuidv4();
    let currentRoom = null;
    
    console.log(`🔗 New connection: ${playerId}`);
    
    // Store player session
    playerSessions.set(playerId, {
        ws,
        playerId,
        connectedAt: Date.now()
    });
    
    // Send welcome message
    ws.send(JSON.stringify({
        type: 'connected',
        playerId,
        serverTime: Date.now()
    }));
    
    ws.on('message', (data) => {
        try {
            const message = JSON.parse(data);
            handleMessage(ws, playerId, message);
        } catch (error) {
            console.error(`❌ Invalid message from ${playerId}:`, error);
        }
    });
    
    ws.on('close', () => {
        console.log(`📡 Disconnected: ${playerId}`);
        
        // Remove from room
        if (currentRoom) {
            const room = gameRooms.get(currentRoom);
            if (room) {
                room.removePlayer(playerId);
                
                // Notify other players
                broadcastToRoom(currentRoom, {
                    type: 'playerLeft',
                    playerId,
                    timestamp: Date.now()
                }, playerId);
                
                // Clean up empty rooms
                if (room.isEmpty()) {
                    gameRooms.delete(currentRoom);
                    console.log(`🗑️  Removed empty room: ${currentRoom}`);
                }
            }
        }
        
        playerSessions.delete(playerId);
    });
    
    function handleMessage(ws, playerId, message) {
        switch (message.type) {
            case 'joinRoom':
                handleJoinRoom(ws, playerId, message);
                break;
                
            case 'createRoom':
                handleCreateRoom(ws, playerId, message);
                break;
                
            case 'listRooms':
                handleListRooms(ws);
                break;
                
            case 'webrtcSignaling':
                handleWebRTCSignaling(playerId, message);
                break;
                
            case 'gossipDiscovery':
                handleGossipDiscovery(playerId, message);
                break;
                
            case 'ping':
                ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
                break;
                
            default:
                console.log(`🤷 Unknown message type: ${message.type}`);
        }
    }
    
    function handleJoinRoom(ws, playerId, message) {
        const { roomId, playerName } = message;
        
        let room = gameRooms.get(roomId);
        if (!room) {
            ws.send(JSON.stringify({
                type: 'error',
                message: `Room ${roomId} not found`
            }));
            return;
        }
        
        const playerData = {
            name: playerName || `Player${playerId.slice(0, 8)}`,
            ws,
            position: { x: 0, y: 0, z: 0 }
        };
        
        if (room.addPlayer(playerId, playerData)) {
            currentRoom = roomId;
            
            // Send success response
            ws.send(JSON.stringify({
                type: 'roomJoined',
                roomId,
                playerId,
                playerList: room.getPlayerList()
            }));
            
            // Notify other players
            broadcastToRoom(roomId, {
                type: 'playerJoined',
                playerId,
                playerName: playerData.name,
                timestamp: Date.now()
            }, playerId);
            
        } else {
            ws.send(JSON.stringify({
                type: 'error',
                message: 'Room is full'
            }));
        }
    }
    
    function handleCreateRoom(ws, playerId, message) {
        const roomId = message.roomId || `room-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const maxPlayers = message.maxPlayers || 16;
        
        if (gameRooms.has(roomId)) {
            ws.send(JSON.stringify({
                type: 'error',
                message: 'Room already exists'
            }));
            return;
        }
        
        const room = new GameRoom(roomId, maxPlayers);
        gameRooms.set(roomId, room);
        
        console.log(`🏗️  Created room: ${roomId} (max ${maxPlayers} players)`);
        
        ws.send(JSON.stringify({
            type: 'roomCreated',
            roomId,
            maxPlayers
        }));
    }
    
    function handleListRooms(ws) {
        const roomList = Array.from(gameRooms.values()).map(room => ({
            id: room.id,
            playerCount: room.players.size,
            maxPlayers: room.maxPlayers,
            gameState: room.gameState,
            createdAt: room.createdAt
        }));
        
        ws.send(JSON.stringify({
            type: 'roomList',
            rooms: roomList
        }));
    }
    
    function handleWebRTCSignaling(fromPlayerId, message) {
        const { targetPlayerId, signalingData } = message;
        
        const targetSession = playerSessions.get(targetPlayerId);
        if (targetSession) {
            targetSession.ws.send(JSON.stringify({
                type: 'webrtcSignaling',
                fromPlayerId,
                signalingData,
                timestamp: Date.now()
            }));
        }
    }
    
    function handleGossipDiscovery(playerId, message) {
        const { roomId, discoveryType } = message;
        
        if (discoveryType === 'peerList') {
            const room = gameRooms.get(roomId);
            if (room) {
                const peers = room.getPlayerList().filter(p => p.id !== playerId);
                
                ws.send(JSON.stringify({
                    type: 'gossipResponse',
                    discoveryType: 'peerList',
                    peers,
                    timestamp: Date.now()
                }));
            }
        }
    }
});

function broadcastToRoom(roomId, message, excludePlayerId = null) {
    const room = gameRooms.get(roomId);
    if (!room) return;
    
    for (const [playerId, playerData] of room.players) {
        if (playerId !== excludePlayerId && playerData.ws) {
            try {
                playerData.ws.send(JSON.stringify(message));
            } catch (error) {
                console.error(`Failed to send to ${playerId}:`, error);
            }
        }
    }
}

// API endpoints
app.get('/api/stats', (req, res) => {
    res.json({
        rooms: gameRooms.size,
        totalPlayers: Array.from(gameRooms.values()).reduce((sum, room) => sum + room.players.size, 0),
        activeSessions: playerSessions.size,
        uptime: process.uptime()
    });
});

app.get('/api/rooms', (req, res) => {
    const roomList = Array.from(gameRooms.values()).map(room => ({
        id: room.id,
        playerCount: room.players.size,
        maxPlayers: room.maxPlayers,
        gameState: room.gameState
    }));
    
    res.json({ rooms: roomList });
});

// Cleanup old rooms periodically
setInterval(() => {
    const now = Date.now();
    const oldRoomThreshold = 24 * 60 * 60 * 1000; // 24 hours
    
    for (const [roomId, room] of gameRooms) {
        if (room.isEmpty() && (now - room.createdAt) > oldRoomThreshold) {
            gameRooms.delete(roomId);
            console.log(`🧹 Cleaned up old room: ${roomId}`);
        }
    }
}, 60 * 60 * 1000); // Check every hour

console.log(`\n🎯 Game Server Ready!`);
console.log(`   Open http://localhost:${PORT} to start playing`);
console.log(`   Press Ctrl+C to stop\n`);