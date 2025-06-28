/**
 * @file game-engine.js
 * @brief Real-time game engine optimized for P2P WebRTC gaming
 * 
 * Features:
 * - 60 FPS rendering and physics
 * - Smooth interpolation for network lag
 * - Client-side prediction
 * - Delta compression for position updates
 */

class GameEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        // Game state
        this.players = new Map();
        this.projectiles = new Map();
        this.particles = [];
        this.localPlayer = null;
        
        // Timing
        this.lastTime = 0;
        this.deltaTime = 0;
        this.frameCount = 0;
        this.fps = 60;
        
        // Input state
        this.keys = {};
        this.mouse = { x: 0, y: 0, clicked: false };
        
        // Camera
        this.camera = { x: 0, y: 0 };
        
        // Game settings
        this.worldSize = { width: 2000, height: 1500 };
        this.playerSpeed = 200; // pixels per second
        this.projectileSpeed = 500;
        
        // Network optimization
        this.lastPositionSent = { x: 0, y: 0 };
        this.positionThreshold = 5; // Minimum movement to send update
        this.lastNetworkUpdate = 0;
        this.networkUpdateRate = 1000 / 20; // 20 updates per second
        
        this.setupEventListeners();
        this.initializeLocalPlayer();
    }
    
    setupEventListeners() {
        // Keyboard input
        window.addEventListener('keydown', (e) => {
            this.keys[e.code] = true;
            e.preventDefault();
        });
        
        window.addEventListener('keyup', (e) => {
            this.keys[e.code] = false;
            e.preventDefault();
        });
        
        // Mouse input
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouse.x = e.clientX - rect.left;
            this.mouse.y = e.clientY - rect.top;
        });
        
        this.canvas.addEventListener('mousedown', (e) => {
            this.mouse.clicked = true;
            this.handleMouseClick(e);
        });
        
        this.canvas.addEventListener('mouseup', (e) => {
            this.mouse.clicked = false;
        });
        
        // Prevent context menu
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Chat input
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && chatInput.value.trim()) {
                    this.sendChatMessage(chatInput.value.trim());
                    chatInput.value = '';
                }
            });
        }
    }
    
    initializeLocalPlayer() {
        this.localPlayer = {
            id: 'local',
            x: this.worldSize.width / 2,
            y: this.worldSize.height / 2,
            health: 100,
            maxHealth: 100,
            color: '#00ff41',
            name: 'Player',
            lastUpdate: performance.now(),
            velocity: { x: 0, y: 0 },
            targetPosition: null
        };
        
        this.players.set('local', this.localPlayer);
    }
    
    update(currentTime) {
        this.deltaTime = (currentTime - this.lastTime) / 1000;
        this.lastTime = currentTime;
        
        // Update FPS counter
        this.frameCount++;
        if (this.frameCount % 60 === 0) {
            this.fps = Math.round(1 / this.deltaTime);
        }
        
        this.updateLocalPlayer();
        this.updateRemotePlayers();
        this.updateProjectiles();
        this.updateParticles();
        this.updateCamera();
        
        // Send network updates
        if (currentTime - this.lastNetworkUpdate > this.networkUpdateRate) {
            this.sendNetworkUpdates();
            this.lastNetworkUpdate = currentTime;
        }
    }
    
    updateLocalPlayer() {
        if (!this.localPlayer) return;
        
        const player = this.localPlayer;
        let moved = false;
        
        // Handle movement input
        const moveSpeed = this.playerSpeed * this.deltaTime;
        
        if (this.keys['KeyW'] || this.keys['ArrowUp']) {
            player.y -= moveSpeed;
            moved = true;
        }
        if (this.keys['KeyS'] || this.keys['ArrowDown']) {
            player.y += moveSpeed;
            moved = true;
        }
        if (this.keys['KeyA'] || this.keys['ArrowLeft']) {
            player.x -= moveSpeed;
            moved = true;
        }
        if (this.keys['KeyD'] || this.keys['ArrowRight']) {
            player.x += moveSpeed;
            moved = true;
        }
        
        // Keep player in bounds
        player.x = Math.max(20, Math.min(this.worldSize.width - 20, player.x));
        player.y = Math.max(20, Math.min(this.worldSize.height - 20, player.y));
        
        // Update velocity for network prediction
        if (moved) {
            const dx = player.x - this.lastPositionSent.x;
            const dy = player.y - this.lastPositionSent.y;
            player.velocity.x = dx / this.deltaTime;
            player.velocity.y = dy / this.deltaTime;
        } else {
            player.velocity.x *= 0.9; // Friction
            player.velocity.y *= 0.9;
        }
    }
    
    updateRemotePlayers() {
        const currentTime = performance.now();
        
        for (const [playerId, player] of this.players) {
            if (playerId === 'local') continue;
            
            // Interpolate to target position for smooth movement
            if (player.targetPosition) {
                const timeSinceUpdate = currentTime - player.lastUpdate;
                const lerpFactor = Math.min(timeSinceUpdate / 100, 1); // 100ms interpolation
                
                player.x += (player.targetPosition.x - player.x) * lerpFactor;
                player.y += (player.targetPosition.y - player.y) * lerpFactor;
                
                if (lerpFactor >= 1) {
                    player.targetPosition = null;
                }
            }
            
            // Client-side prediction for remote players
            if (player.velocity) {
                const predictedX = player.x + player.velocity.x * this.deltaTime;
                const predictedY = player.y + player.velocity.y * this.deltaTime;
                
                // Only apply prediction if it keeps player in bounds
                if (predictedX > 20 && predictedX < this.worldSize.width - 20) {
                    player.x = predictedX;
                }
                if (predictedY > 20 && predictedY < this.worldSize.height - 20) {
                    player.y = predictedY;
                }
            }
        }
    }
    
    updateProjectiles() {
        const currentTime = performance.now();
        
        for (const [projectileId, projectile] of this.projectiles) {
            // Update position
            projectile.x += projectile.velocity.x * this.deltaTime;
            projectile.y += projectile.velocity.y * this.deltaTime;
            
            // Check bounds
            if (projectile.x < 0 || projectile.x > this.worldSize.width ||
                projectile.y < 0 || projectile.y > this.worldSize.height) {
                this.projectiles.delete(projectileId);
                continue;
            }
            
            // Check lifetime
            if (currentTime - projectile.createdAt > projectile.lifetime) {
                this.projectiles.delete(projectileId);
                continue;
            }
            
            // Check collision with players (only for remote projectiles)
            if (projectile.ownerId !== this.localPlayer?.id) {
                this.checkProjectileCollision(projectileId, projectile);
            }
        }
    }
    
    updateParticles() {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];
            
            particle.x += particle.velocity.x * this.deltaTime;
            particle.y += particle.velocity.y * this.deltaTime;
            particle.life -= this.deltaTime;
            particle.alpha = particle.life / particle.maxLife;
            
            if (particle.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }
    
    updateCamera() {
        if (this.localPlayer) {
            // Center camera on local player
            this.camera.x = this.localPlayer.x - this.canvas.width / 2;
            this.camera.y = this.localPlayer.y - this.canvas.height / 2;
            
            // Keep camera in world bounds
            this.camera.x = Math.max(0, Math.min(this.worldSize.width - this.canvas.width, this.camera.x));
            this.camera.y = Math.max(0, Math.min(this.worldSize.height - this.canvas.height, this.camera.y));
        }
    }
    
    render() {
        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Save context for camera transform
        this.ctx.save();
        this.ctx.translate(-this.camera.x, -this.camera.y);
        
        // Render world background
        this.renderWorldBackground();
        
        // Render players
        for (const player of this.players.values()) {
            this.renderPlayer(player);
        }
        
        // Render projectiles
        for (const projectile of this.projectiles.values()) {
            this.renderProjectile(projectile);
        }
        
        // Render particles
        for (const particle of this.particles) {
            this.renderParticle(particle);
        }
        
        // Restore context
        this.ctx.restore();
        
        // Render UI elements (not affected by camera)
        this.renderUI();
    }
    
    renderWorldBackground() {
        // Grid pattern
        this.ctx.strokeStyle = '#111';
        this.ctx.lineWidth = 1;
        
        const gridSize = 50;
        const startX = Math.floor(this.camera.x / gridSize) * gridSize;
        const startY = Math.floor(this.camera.y / gridSize) * gridSize;
        
        for (let x = startX; x < this.camera.x + this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, this.camera.y);
            this.ctx.lineTo(x, this.camera.y + this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let y = startY; y < this.camera.y + this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(this.camera.x, y);
            this.ctx.lineTo(this.camera.x + this.canvas.width, y);
            this.ctx.stroke();
        }
        
        // World border
        this.ctx.strokeStyle = '#00ff41';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(0, 0, this.worldSize.width, this.worldSize.height);
    }
    
    renderPlayer(player) {
        const isLocal = player.id === 'local';
        
        // Player body
        this.ctx.fillStyle = player.color || (isLocal ? '#00ff41' : '#ff4141');
        this.ctx.beginPath();
        this.ctx.arc(player.x, player.y, 15, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Player border
        this.ctx.strokeStyle = isLocal ? '#ffffff' : '#888888';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Health bar
        const barWidth = 30;
        const barHeight = 4;
        const healthRatio = player.health / (player.maxHealth || 100);
        
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(player.x - barWidth/2, player.y - 25, barWidth, barHeight);
        
        this.ctx.fillStyle = healthRatio > 0.5 ? '#00ff00' : healthRatio > 0.25 ? '#ffff00' : '#ff0000';
        this.ctx.fillRect(player.x - barWidth/2, player.y - 25, barWidth * healthRatio, barHeight);
        
        // Player name
        if (player.name) {
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '12px Courier New';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(player.name, player.x, player.y - 35);
        }
        
        // Weapon direction indicator for local player
        if (isLocal) {
            const mouseWorldX = this.mouse.x + this.camera.x;
            const mouseWorldY = this.mouse.y + this.camera.y;
            const angle = Math.atan2(mouseWorldY - player.y, mouseWorldX - player.x);
            
            this.ctx.strokeStyle = '#ff0000';
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.moveTo(player.x, player.y);
            this.ctx.lineTo(
                player.x + Math.cos(angle) * 25,
                player.y + Math.sin(angle) * 25
            );
            this.ctx.stroke();
        }
    }
    
    renderProjectile(projectile) {
        this.ctx.fillStyle = projectile.color || '#ffff00';
        this.ctx.beginPath();
        this.ctx.arc(projectile.x, projectile.y, projectile.radius || 3, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Trail effect
        this.ctx.strokeStyle = projectile.color || '#ffff00';
        this.ctx.lineWidth = 1;
        this.ctx.globalAlpha = 0.5;
        this.ctx.beginPath();
        this.ctx.moveTo(projectile.x, projectile.y);
        this.ctx.lineTo(
            projectile.x - projectile.velocity.x * 0.1,
            projectile.y - projectile.velocity.y * 0.1
        );
        this.ctx.stroke();
        this.ctx.globalAlpha = 1;
    }
    
    renderParticle(particle) {
        this.ctx.fillStyle = particle.color;
        this.ctx.globalAlpha = particle.alpha;
        this.ctx.beginPath();
        this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.globalAlpha = 1;
    }
    
    renderUI() {
        // Crosshair
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - 10, centerY);
        this.ctx.lineTo(centerX + 10, centerY);
        this.ctx.moveTo(centerX, centerY - 10);
        this.ctx.lineTo(centerX, centerY + 10);
        this.ctx.stroke();
        
        // Update UI elements
        this.updateUIStats();
    }
    
    updateUIStats() {
        // FPS
        const fpsEl = document.getElementById('fpsCounter');
        if (fpsEl) fpsEl.textContent = this.fps;
        
        // Peer count
        const peerCountEl = document.getElementById('peerCount');
        if (peerCountEl && window.webrtcManager) {
            peerCountEl.textContent = window.webrtcManager.getConnectedPeers().length;
        }
        
        // Health bar
        if (this.localPlayer) {
            const healthFill = document.getElementById('healthFill');
            if (healthFill) {
                const healthPercent = (this.localPlayer.health / this.localPlayer.maxHealth) * 100;
                healthFill.style.width = `${healthPercent}%`;
            }
        }
    }
    
    handleMouseClick(event) {
        if (!this.localPlayer) return;
        
        // Calculate world coordinates
        const worldX = this.mouse.x + this.camera.x;
        const worldY = this.mouse.y + this.camera.y;
        
        // Fire projectile
        this.fireProjectile(this.localPlayer.x, this.localPlayer.y, worldX, worldY);
    }
    
    fireProjectile(fromX, fromY, targetX, targetY) {
        const angle = Math.atan2(targetY - fromY, targetX - fromX);
        const projectileId = `${this.localPlayer.id}-${Date.now()}-${Math.random()}`;
        
        const projectile = {
            id: projectileId,
            x: fromX + Math.cos(angle) * 20, // Start slightly in front of player
            y: fromY + Math.sin(angle) * 20,
            velocity: {
                x: Math.cos(angle) * this.projectileSpeed,
                y: Math.sin(angle) * this.projectileSpeed
            },
            ownerId: this.localPlayer.id,
            damage: 20,
            color: '#ffff00',
            radius: 3,
            createdAt: performance.now(),
            lifetime: 3000 // 3 seconds
        };
        
        this.projectiles.set(projectileId, projectile);
        
        // Send to other players
        if (window.webrtcManager) {
            window.webrtcManager.broadcastToAllPeers('combat', {
                type: 'projectile',
                projectile: projectile
            });
        }
    }
    
    checkProjectileCollision(projectileId, projectile) {
        if (!this.localPlayer) return;
        
        const dx = projectile.x - this.localPlayer.x;
        const dy = projectile.y - this.localPlayer.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 15 + (projectile.radius || 3)) {
            // Hit!
            this.localPlayer.health = Math.max(0, this.localPlayer.health - (projectile.damage || 20));
            
            // Remove projectile
            this.projectiles.delete(projectileId);
            
            // Create hit effect
            this.createHitEffect(this.localPlayer.x, this.localPlayer.y);
            
            // Notify attacker of hit
            if (window.webrtcManager) {
                window.webrtcManager.sendToPeer(projectile.ownerId, 'combat', {
                    type: 'hit',
                    targetId: this.localPlayer.id,
                    damage: projectile.damage || 20
                });
            }
        }
    }
    
    createHitEffect(x, y) {
        for (let i = 0; i < 10; i++) {
            this.particles.push({
                x: x + (Math.random() - 0.5) * 20,
                y: y + (Math.random() - 0.5) * 20,
                velocity: {
                    x: (Math.random() - 0.5) * 200,
                    y: (Math.random() - 0.5) * 200
                },
                color: '#ff0000',
                size: Math.random() * 3 + 1,
                life: 0.5,
                maxLife: 0.5,
                alpha: 1
            });
        }
    }
    
    sendNetworkUpdates() {
        if (!this.localPlayer || !window.webrtcManager) return;
        
        // Only send if position changed significantly
        const dx = this.localPlayer.x - this.lastPositionSent.x;
        const dy = this.localPlayer.y - this.lastPositionSent.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > this.positionThreshold) {
            window.webrtcManager.broadcastToAllPeers('game', {
                type: 'position',
                x: this.localPlayer.x,
                y: this.localPlayer.y,
                velocity: this.localPlayer.velocity,
                health: this.localPlayer.health
            });
            
            this.lastPositionSent = { x: this.localPlayer.x, y: this.localPlayer.y };
        }
    }
    
    sendChatMessage(message) {
        if (window.webrtcManager) {
            window.webrtcManager.broadcastToAllPeers('chat', {
                type: 'chat',
                message: message,
                playerName: this.localPlayer.name
            });
            
            this.addChatMessage(this.localPlayer.name, message, true);
        }
    }
    
    addChatMessage(playerName, message, isLocal = false) {
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            const messageEl = document.createElement('div');
            messageEl.style.color = isLocal ? '#00ff41' : '#ffffff';
            messageEl.innerHTML = `<strong>${playerName}:</strong> ${message}`;
            chatMessages.appendChild(messageEl);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    // Network message handlers
    handleRemotePlayerUpdate(message) {
        const playerId = message.fromPeer;
        let player = this.players.get(playerId);
        
        if (!player) {
            // Create new remote player
            player = {
                id: playerId,
                x: message.x || 0,
                y: message.y || 0,
                health: message.health || 100,
                maxHealth: 100,
                color: '#ff4141',
                name: `Player ${playerId.slice(0, 8)}`,
                lastUpdate: performance.now(),
                velocity: message.velocity || { x: 0, y: 0 }
            };
            this.players.set(playerId, player);
        }
        
        // Update position with smooth interpolation
        player.targetPosition = { x: message.x, y: message.y };
        player.velocity = message.velocity || { x: 0, y: 0 };
        player.health = message.health || player.health;
        player.lastUpdate = performance.now();
    }
    
    handleRemoteProjectile(message) {
        const projectile = message.projectile;
        if (projectile && projectile.ownerId !== this.localPlayer?.id) {
            this.projectiles.set(projectile.id, projectile);
        }
    }
    
    handleRemoteChat(message) {
        this.addChatMessage(message.playerName || 'Unknown', message.message);
    }
    
    removePlayer(playerId) {
        this.players.delete(playerId);
    }
    
    setLocalPlayerName(name) {
        if (this.localPlayer) {
            this.localPlayer.name = name;
        }
    }
}