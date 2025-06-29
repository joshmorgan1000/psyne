/**
 * @file main.js
 * @brief Main game controller for PeerJS version
 * 
 * Simplified P2P game using PeerJS hosted service:
 * - No custom server required
 * - Room-based discovery via localStorage
 * - Direct peer connections
 * - Automatic reconnection
 */

// Global game instances
let peerManager = null;
let gameEngine = null;
let combatSystem = null;
let isGameRunning = false;

// Game state
window.gameStartTime = null;
let animationId = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎮 Initializing PeerJS P2P Combat Game...');
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

    // Initialize PeerJS manager
    peerManager = new PeerJSManager();
    window.peerManager = peerManager; // Make globally accessible

    // Initialize game engine
    gameEngine = new GameEngine(canvas);
    window.gameEngine = gameEngine;

    // Initialize combat system
    combatSystem = new CombatSystem(gameEngine);
    window.combatSystem = combatSystem;

    // Set up PeerJS event handlers
    setupPeerJSEvents();

    // Set up message routing
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

function setupPeerJSEvents() {
    // Peer ready
    peerManager.onReady = (peerId) => {
        console.log('🎮 PeerJS ready with ID:', peerId);
        document.getElementById('myPeerId').textContent = peerId;
        document.getElementById('yourPeerId').textContent = peerId;
    };

    // Peer connection events
    peerManager.onPeerConnected = (peerId) => {
        console.log('🤝 Peer connected:', peerId);
        peerManager.updatePeerConnectionsDisplay();
        
        // Send our player info to new peer
        if (gameEngine.localPlayer) {
            peerManager.sendToPeer(peerId, {
                type: 'playerInfo',
                player: {
                    id: peerManager.localPlayerId,
                    x: gameEngine.localPlayer.x,
                    y: gameEngine.localPlayer.y,
                    health: gameEngine.localPlayer.health,
                    name: gameEngine.localPlayer.name
                }
            });
        }
    };

    peerManager.onPeerDisconnected = (peerId) => {
        console.log('👋 Peer disconnected:', peerId);
        peerManager.updatePeerConnectionsDisplay();
        
        // Remove player from game
        if (gameEngine) {
            gameEngine.removePlayer(peerId);
        }
    };
}

function setupMessageRouting() {
    // Game messages
    peerManager.onGameMessage = (message) => {
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

            default:
                console.log('🤷 Unknown game message:', message.type);
        }
    };

    // Chat messages
    peerManager.onChatMessage = (message) => {
        if (gameEngine && message.type === 'chat') {
            gameEngine.handleRemoteChat(message);
        }
    };
}

function setupUIHandlers() {
    // Menu navigation
    window.showMainMenu = () => {
        document.getElementById('mainMenu').style.display = 'flex';
        document.getElementById('peerMenu').style.display = 'none';
    };

    window.showPeerList = () => {
        const playerName = document.getElementById('playerName').value.trim();
        if (!playerName) {
            alert('Please enter your warrior name!');
            return;
        }

        document.getElementById('mainMenu').style.display = 'none';
        document.getElementById('peerMenu').style.display = 'flex';
        
        // Set player name
        if (gameEngine) {
            gameEngine.setLocalPlayerName(playerName);
        }

        // Update peer ID display
        if (peerManager.localPlayerId) {
            document.getElementById('myPeerId').textContent = peerManager.localPlayerId;
        }
    };

    window.startGame = () => {
        const playerName = document.getElementById('playerName').value.trim();
        if (!playerName) {
            alert('Please enter your warrior name!');
            return;
        }

        let roomCode = document.getElementById('roomCode').value.trim().toUpperCase();
        if (!roomCode) {
            // Generate random room code
            roomCode = peerManager.generateRoomCode();
            document.getElementById('roomCode').value = roomCode;
        }

        if (gameEngine) {
            gameEngine.setLocalPlayerName(playerName);
        }

        // Join room and start game
        peerManager.joinRoom(roomCode, playerName);
        hideAllMenus();
        startGameLoop();
        
        // Update UI
        document.getElementById('currentRoom').textContent = roomCode;
        
        console.log(`🏟️ Started game in room: ${roomCode}`);
    };

    window.connectToPeer = () => {
        const targetPeerId = document.getElementById('targetPeerId').value.trim();
        if (!targetPeerId) {
            alert('Please enter a peer ID!');
            return;
        }

        const playerName = document.getElementById('playerName').value.trim();
        if (!playerName) {
            alert('Please enter your warrior name!');
            return;
        }

        if (gameEngine) {
            gameEngine.setLocalPlayerName(playerName);
        }

        peerManager.connectToPeer(targetPeerId);
        hideAllMenus();
        startGameLoop();
    };

    // Chat input
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && chatInput.value.trim()) {
                sendChatMessage(chatInput.value.trim());
                chatInput.value = '';
            }
        });
    }

    // Start latency measurements
    setInterval(() => {
        if (peerManager && isGameRunning) {
            peerManager.measureLatency();
        }
    }, 5000);

    // Update UI stats
    setInterval(updateUIStats, 1000);
}

function hideAllMenus() {
    document.getElementById('mainMenu').style.display = 'none';
    document.getElementById('peerMenu').style.display = 'none';
}

function startGameLoop() {
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

function sendChatMessage(message) {
    if (peerManager) {
        peerManager.broadcastToAllPeers({
            type: 'chat',
            message: message,
            playerName: gameEngine.localPlayer?.name || 'Unknown'
        });
        
        // Add to local chat
        if (gameEngine) {
            gameEngine.addChatMessage(gameEngine.localPlayer?.name || 'You', message, true);
        }
    }
}

function updateUIStats() {
    if (!peerManager || !isGameRunning) return;

    // Update latency display
    const latencyEl = document.getElementById('latencyDisplay');
    if (latencyEl) {
        const stats = peerManager.getStats();
        latencyEl.textContent = `${Math.round(stats.averageLatency)}ms`;
    }

    // Update peer count
    const peerCountEl = document.getElementById('peerCount');
    if (peerCountEl) {
        peerCountEl.textContent = peerManager.getConnectedPeers().length;
    }

    // Reset packet counters for rate calculation
    if (peerManager.stats) {
        peerManager.stats.packetsSent = 0;
        peerManager.stats.packetsReceived = 0;
    }
}

// Modified game engine message handlers to work with PeerJS
function setupGameEngineForPeerJS() {
    if (!gameEngine) return;

    // Override the WebRTC manager methods to use PeerJS
    gameEngine.sendNetworkUpdates = function() {
        if (!this.localPlayer || !peerManager) return;

        // Only send if position changed significantly
        const dx = this.localPlayer.x - this.lastPositionSent.x;
        const dy = this.localPlayer.y - this.lastPositionSent.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > this.positionThreshold) {
            peerManager.broadcastToAllPeers({
                type: 'position',
                x: this.localPlayer.x,
                y: this.localPlayer.y,
                velocity: this.localPlayer.velocity,
                health: this.localPlayer.health
            });

            this.lastPositionSent = { x: this.localPlayer.x, y: this.localPlayer.y };
        }
    };

    // Override combat system for PeerJS
    if (combatSystem) {
        const originalFireSingle = combatSystem.fireSingleShot;
        combatSystem.fireSingleShot = function(player, angle, weapon) {
            originalFireSingle.call(this, player, angle, weapon);
            
            // Find the projectile we just created
            const projectiles = Array.from(gameEngine.projectiles.values());
            const latestProjectile = projectiles[projectiles.length - 1];
            
            if (latestProjectile && peerManager) {
                peerManager.broadcastToAllPeers({
                    type: 'shot',
                    projectile: latestProjectile,
                    playerPosition: { x: player.x, y: player.y },
                    timestamp: performance.now(),
                    weapon: this.selectedWeapon
                });
            }
        };
    }
}

// Initialize game engine overrides when everything is ready
setTimeout(setupGameEngineForPeerJS, 1000);

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (peerManager) {
        peerManager.disconnect();
    }
    
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
});

// Development helpers
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    window.addEventListener('keydown', (e) => {
        if (e.code === 'F1') {
            e.preventDefault();
            console.log('🐛 Debug info:');
            console.log('PeerJS Stats:', peerManager?.getStats());
            console.log('Connected Peers:', peerManager?.getConnectedPeers());
            console.log('Available Rooms:', peerManager?.listAvailableRooms());
        }
        
        if (e.code === 'F3') {
            e.preventDefault();
            // Quick connect for testing - connects to any available peer
            const peers = peerManager?.getConnectedPeers() || [];
            console.log('🔧 Quick connect test - current peers:', peers.length);
            
            // Generate a test room
            const testRoom = 'TEST123';
            document.getElementById('roomCode').value = testRoom;
            startGame();
        }
    });
    
    console.log('🐛 Development mode: Press F1 for debug, F3 for quick test');
}

console.log('🎮 PeerJS main controller loaded successfully');