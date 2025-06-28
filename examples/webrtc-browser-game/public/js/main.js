/**
 * @file main.js
 * @brief Main game initialization and coordination
 * 
 * Coordinates:
 * - WebRTC P2P connections
 * - Game engine rendering
 * - Combat system
 * - Gossip protocol for peer discovery
 * - UI management
 */

// Global game instances
let webrtcManager = null;
let gameEngine = null;
let combatSystem = null;
let gossipProtocol = null;
let isGameRunning = false;

// Game state
window.gameStartTime = null;
let animationId = null;

// Initialize the game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎮 Initializing WebRTC P2P Combat Game...');
    initializeGame();
});

function initializeGame() {
    // Get canvas
    const canvas = document.getElementById('gameCanvas');
    if (!canvas) {
        console.error('❌ Game canvas not found!');
        return;
    }

    // Resize canvas to fit screen
    resizeCanvas(canvas);
    window.addEventListener('resize', () => resizeCanvas(canvas));

    // Initialize WebRTC manager
    webrtcManager = new WebRTCManager();
    window.webrtcManager = webrtcManager; // Make globally accessible

    // Initialize game engine
    gameEngine = new GameEngine(canvas);
    window.gameEngine = gameEngine;

    // Initialize combat system
    combatSystem = new CombatSystem(gameEngine);
    window.combatSystem = combatSystem;

    // Initialize gossip protocol
    gossipProtocol = new GossipProtocol(webrtcManager);
    window.gossipProtocol = gossipProtocol;

    // Set up WebRTC event handlers
    setupWebRTCEvents();

    // Set up game message routing
    setupMessageRouting();

    // Set up UI handlers
    setupUIHandlers();

    console.log('✅ Game initialized successfully');
}

function resizeCanvas(canvas) {
    // Make canvas responsive while maintaining aspect ratio
    const container = document.getElementById('gameContainer');
    const aspectRatio = 1200 / 800;
    
    let width = container.clientWidth;
    let height = container.clientHeight;

    if (width / height > aspectRatio) {
        width = height * aspectRatio;
    } else {
        height = width / aspectRatio;
    }

    canvas.width = Math.min(1200, width);
    canvas.height = Math.min(800, height);
    canvas.style.position = 'absolute';
    canvas.style.left = '50%';
    canvas.style.top = '50%';
    canvas.style.transform = 'translate(-50%, -50%)';
}

function setupWebRTCEvents() {
    // Peer connection events
    webrtcManager.onPeerConnected = (peerId) => {
        console.log('🤝 Peer connected:', peerId);
        updatePeerConnectionsDisplay();
        
        // Send our player info to new peer
        if (gameEngine.localPlayer) {
            webrtcManager.sendToPeer(peerId, 'game', {
                type: 'playerInfo',
                player: {
                    id: webrtcManager.localPlayerId,
                    x: gameEngine.localPlayer.x,
                    y: gameEngine.localPlayer.y,
                    health: gameEngine.localPlayer.health,
                    name: gameEngine.localPlayer.name
                }
            });
        }
    };

    webrtcManager.onPeerDisconnected = (peerId) => {
        console.log('👋 Peer disconnected:', peerId);
        updatePeerConnectionsDisplay();
        
        // Remove player from game
        if (gameEngine) {
            gameEngine.removePlayer(peerId);
        }
    };

    // Room events
    webrtcManager.onRoomUpdate = (message) => {
        switch (message.type) {
            case 'roomJoined':
                hideAllMenus();
                startGame();
                break;
            case 'roomList':
                updateRoomList(message.rooms);
                break;
        }
    };
}

function setupMessageRouting() {
    // Game messages
    webrtcManager.onGameMessage = (message) => {
        switch (message.type) {
            case 'position':
                if (gameEngine) {
                    gameEngine.handleRemotePlayerUpdate(message);
                }
                break;

            case 'playerInfo':
                if (gameEngine) {
                    gameEngine.handleRemotePlayerUpdate(message);
                }
                break;

            case 'projectile':
                if (gameEngine) {
                    gameEngine.handleRemoteProjectile(message);
                }
                break;

            // Combat messages
            case 'shot':
                if (combatSystem) {
                    combatSystem.handleRemoteShot(message);
                }
                break;

            case 'shotgunBlast':
                if (combatSystem) {
                    combatSystem.handleRemoteShotgunBlast(message);
                }
                break;

            case 'hitConfirmed':
                if (combatSystem) {
                    combatSystem.handleHitConfirmation(message);
                }
                break;

            case 'damage':
                if (combatSystem) {
                    combatSystem.handleDamageReceived(message);
                }
                break;

            case 'killConfirmed':
                if (combatSystem) {
                    combatSystem.combatStats.kills++;
                    console.log('💀 Kill confirmed!');
                }
                break;

            // Gossip protocol messages are handled automatically
            default:
                if (message.type && message.type.startsWith('gossip_')) {
                    // Gossip messages are handled by the gossip protocol
                    break;
                }
                console.log('🤷 Unknown game message:', message.type);
        }
    };

    // Chat messages
    webrtcManager.onChatMessage = (message) => {
        if (gameEngine && message.type === 'chat') {
            gameEngine.handleRemoteChat(message);
        }
    };
}

function setupUIHandlers() {
    // Menu navigation
    window.showMainMenu = () => {
        document.getElementById('mainMenu').style.display = 'flex';
        document.getElementById('roomMenu').style.display = 'none';
    };

    window.showRoomMenu = () => {
        const playerName = document.getElementById('playerName').value.trim();
        if (!playerName) {
            alert('Please enter your warrior name!');
            return;
        }

        document.getElementById('mainMenu').style.display = 'none';
        document.getElementById('roomMenu').style.display = 'flex';
        
        // Set player name in game engine
        if (gameEngine) {
            gameEngine.setLocalPlayerName(playerName);
        }

        // Request room list
        webrtcManager.listRooms();
    };

    window.createRoom = () => {
        const playerName = document.getElementById('playerName').value.trim();
        if (!playerName) {
            alert('Please enter your warrior name!');
            return;
        }

        if (gameEngine) {
            gameEngine.setLocalPlayerName(playerName);
        }

        webrtcManager.createRoom(16); // Max 16 players
    };

    window.refreshRooms = () => {
        webrtcManager.listRooms();
    };

    window.joinRoom = (roomId) => {
        const playerName = document.getElementById('playerName').value.trim();
        if (!playerName) {
            alert('Please enter your warrior name!');
            return;
        }

        if (gameEngine) {
            gameEngine.setLocalPlayerName(playerName);
        }

        webrtcManager.joinRoom(roomId, playerName);
    };

    // Weapon switching indicators
    window.addEventListener('keydown', (e) => {
        switch (e.code) {
            case 'Digit1':
            case 'Digit2':
            case 'Digit3':
                updateWeaponDisplay();
                break;
        }
    });

    // Start latency measurements
    setInterval(() => {
        if (webrtcManager && isGameRunning) {
            webrtcManager.measureLatency();
        }
    }, 5000);

    // Update UI stats
    setInterval(updateUIStats, 1000);
}

function hideAllMenus() {
    document.getElementById('mainMenu').style.display = 'none';
    document.getElementById('roomMenu').style.display = 'none';
}

function startGame() {
    if (isGameRunning) return;

    console.log('🚀 Starting game...');
    isGameRunning = true;
    window.gameStartTime = Date.now();

    // Start game loop
    gameLoop();

    // Show game UI
    document.getElementById('ui').style.display = 'block';
    document.getElementById('stats').style.display = 'block';
    document.getElementById('chatPanel').style.display = 'flex';
}

function gameLoop() {
    if (!isGameRunning) return;

    const currentTime = performance.now();

    // Update game engine
    if (gameEngine) {
        gameEngine.update(currentTime);
        gameEngine.render();
    }

    // Continue loop
    animationId = requestAnimationFrame(gameLoop);
}

function updateRoomList(rooms) {
    const roomList = document.getElementById('roomList');
    if (!roomList) return;

    roomList.innerHTML = '';

    if (!rooms || rooms.length === 0) {
        roomList.innerHTML = '<div class="room-item">No active arenas found. Create one!</div>';
        return;
    }

    rooms.forEach(room => {
        const roomItem = document.createElement('div');
        roomItem.className = 'room-item';
        roomItem.innerHTML = `
            <strong>🏟️ ${room.id}</strong><br>
            👥 ${room.playerCount}/${room.maxPlayers} warriors<br>
            🕐 ${new Date(room.createdAt).toLocaleTimeString()}
        `;
        roomItem.onclick = () => joinRoom(room.id);
        roomList.appendChild(roomItem);
    });
}

function updatePeerConnectionsDisplay() {
    const container = document.getElementById('peerConnections');
    if (!container || !webrtcManager) return;

    const connectedPeers = webrtcManager.getConnectedPeers();
    
    container.innerHTML = '';
    container.innerHTML = `<div style="color: #00ff41; font-weight: bold;">P2P Connections:</div>`;

    if (connectedPeers.length === 0) {
        container.innerHTML += '<div style="color: #666;">No direct connections</div>';
        return;
    }

    connectedPeers.forEach(peerId => {
        const latency = webrtcManager.getPeerLatency(peerId);
        const peerDiv = document.createElement('div');
        peerDiv.className = 'peer-connection peer-connected';
        peerDiv.innerHTML = `📡 ${peerId.slice(0, 8)}... (${Math.round(latency)}ms)`;
        container.appendChild(peerDiv);
    });
}

function updateWeaponDisplay() {
    if (!combatSystem) return;

    const weapon = combatSystem.getCurrentWeapon();
    const indicator = document.createElement('div');
    indicator.textContent = `🔫 ${weapon.type.toUpperCase()}`;
    indicator.style.cssText = `
        position: absolute;
        top: 80px;
        left: 20px;
        color: ${weapon.color};
        font-weight: bold;
        font-size: 14px;
        pointer-events: none;
        z-index: 1000;
        animation: fadeOut 2s ease-out forwards;
    `;
    
    document.body.appendChild(indicator);
    setTimeout(() => indicator.remove(), 2000);
}

function updateUIStats() {
    if (!webrtcManager || !isGameRunning) return;

    // Update latency display
    const latencyEl = document.getElementById('latencyDisplay');
    if (latencyEl) {
        const stats = webrtcManager.getStats();
        latencyEl.textContent = `${Math.round(stats.averageLatency)}ms`;
    }

    // Update packet rate
    const packetRateEl = document.getElementById('packetRate');
    if (packetRateEl && webrtcManager.stats) {
        const packetsPerSecond = webrtcManager.stats.packetsSent + webrtcManager.stats.packetsReceived;
        packetRateEl.textContent = packetsPerSecond;
        
        // Reset counters
        webrtcManager.stats.packetsSent = 0;
        webrtcManager.stats.packetsReceived = 0;
    }

    // Update gossip network stats
    if (gossipProtocol) {
        const networkStats = gossipProtocol.getNetworkStats();
        console.log('📊 Network stats:', networkStats);
    }

    // Update combat stats
    if (combatSystem) {
        const combatStats = combatSystem.getStats();
        // Could display these in a combat stats panel
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (webrtcManager) {
        webrtcManager.disconnect();
    }
    
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
});

// Handle visibility change (tab switching)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Reduce update frequency when tab is hidden
        console.log('⏸️ Game paused (tab hidden)');
    } else {
        // Resume normal operation
        console.log('▶️ Game resumed (tab visible)');
    }
});

// Development helpers
if (window.location.hostname === 'localhost') {
    // Debug shortcuts for development
    window.addEventListener('keydown', (e) => {
        if (e.code === 'F1') {
            e.preventDefault();
            console.log('🐛 Debug info:');
            console.log('WebRTC Stats:', webrtcManager?.getStats());
            console.log('Gossip Stats:', gossipProtocol?.getNetworkStats());
            console.log('Combat Stats:', combatSystem?.getStats());
            console.log('Connected Peers:', webrtcManager?.getConnectedPeers());
        }
        
        if (e.code === 'F2') {
            e.preventDefault();
            console.log('🕸️ Network topology:', gossipProtocol?.getNetworkVisualization());
        }
    });
    
    console.log('🐛 Development mode: Press F1 for debug info, F2 for network topology');
}

console.log('🎮 Main game controller loaded successfully');