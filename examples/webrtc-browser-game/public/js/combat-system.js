/**
 * @file combat-system.js
 * @brief Advanced combat mechanics for P2P real-time gaming
 * 
 * Features:
 * - Hit validation and anti-cheat
 * - Damage calculation with weapon types
 * - Lag compensation for hit detection
 * - Combat statistics tracking
 */

class CombatSystem {
    constructor(gameEngine) {
        this.gameEngine = gameEngine;
        
        // Weapon definitions
        this.weapons = {
            pistol: {
                damage: 20,
                fireRate: 300, // ms between shots
                range: 400,
                projectileSpeed: 600,
                accuracy: 0.95,
                color: '#ffff00'
            },
            rifle: {
                damage: 35,
                fireRate: 150,
                range: 600,
                projectileSpeed: 800,
                accuracy: 0.85,
                color: '#ff8800'
            },
            shotgun: {
                damage: 60,
                fireRate: 800,
                range: 200,
                projectileSpeed: 400,
                accuracy: 0.7,
                color: '#ff0000',
                pellets: 5
            }
        };
        
        // Combat state
        this.selectedWeapon = 'pistol';
        this.lastFireTime = 0;
        this.combatStats = {
            shotsFired: 0,
            shotsHit: 0,
            damageDealt: 0,
            damageReceived: 0,
            kills: 0,
            deaths: 0
        };
        
        // Hit validation for anti-cheat
        this.recentShots = new Map(); // Track shots for validation
        this.maxShotHistory = 100;
        this.maxLatencyTolerance = 200; // ms
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Weapon switching
        window.addEventListener('keydown', (e) => {
            switch (e.code) {
                case 'Digit1':
                    this.switchWeapon('pistol');
                    break;
                case 'Digit2':
                    this.switchWeapon('rifle');
                    break;
                case 'Digit3':
                    this.switchWeapon('shotgun');
                    break;
            }
        });
        
        // Mouse input for firing
        this.gameEngine.canvas.addEventListener('mousedown', (e) => {
            if (e.button === 0) { // Left click
                this.handleFireAttempt();
            }
        });
    }
    
    switchWeapon(weaponType) {
        if (this.weapons[weaponType]) {
            this.selectedWeapon = weaponType;
            this.showWeaponSwitchIndicator(weaponType);
        }
    }
    
    showWeaponSwitchIndicator(weaponType) {
        // Visual feedback for weapon switch
        const indicator = document.createElement('div');
        indicator.textContent = `Switched to ${weaponType.toUpperCase()}`;
        indicator.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #00ff41;
            font-weight: bold;
            font-size: 18px;
            pointer-events: none;
            z-index: 1000;
            animation: fadeOut 1s ease-out forwards;
        `;
        
        document.body.appendChild(indicator);
        setTimeout(() => indicator.remove(), 1000);
    }
    
    handleFireAttempt() {
        const currentTime = performance.now();
        const weapon = this.weapons[this.selectedWeapon];
        
        // Check fire rate
        if (currentTime - this.lastFireTime < weapon.fireRate) {
            return false;
        }
        
        const player = this.gameEngine.localPlayer;
        if (!player) return false;
        
        // Calculate firing direction
        const mouseWorldX = this.gameEngine.mouse.x + this.gameEngine.camera.x;
        const mouseWorldY = this.gameEngine.mouse.y + this.gameEngine.camera.y;
        const angle = Math.atan2(mouseWorldY - player.y, mouseWorldX - player.x);
        
        // Check range
        const distance = Math.sqrt(
            Math.pow(mouseWorldX - player.x, 2) + 
            Math.pow(mouseWorldY - player.y, 2)
        );
        
        if (distance > weapon.range) {
            this.showRangeIndicator();
            return false;
        }
        
        // Fire weapon
        if (weapon.pellets) {
            // Shotgun - multiple projectiles
            this.fireShotgun(player, angle, weapon);
        } else {
            // Single projectile
            this.fireSingleShot(player, angle, weapon);
        }
        
        this.lastFireTime = currentTime;
        this.combatStats.shotsFired++;
        
        return true;
    }
    
    fireSingleShot(player, baseAngle, weapon) {
        // Apply accuracy (random spread)
        const spread = (1 - weapon.accuracy) * 0.2; // Max 0.2 radians spread
        const angle = baseAngle + (Math.random() - 0.5) * spread;
        
        const projectileId = `${player.id}-${Date.now()}-${Math.random()}`;
        
        const projectile = {
            id: projectileId,
            x: player.x + Math.cos(angle) * 20,
            y: player.y + Math.sin(angle) * 20,
            velocity: {
                x: Math.cos(angle) * weapon.projectileSpeed,
                y: Math.sin(angle) * weapon.projectileSpeed
            },
            ownerId: player.id,
            damage: weapon.damage,
            color: weapon.color,
            radius: 3,
            createdAt: performance.now(),
            lifetime: (weapon.range / weapon.projectileSpeed) * 1000,
            weapon: this.selectedWeapon
        };
        
        this.gameEngine.projectiles.set(projectileId, projectile);
        this.recordShot(projectileId, projectile);
        
        // Send to other players with shot validation data
        if (window.webrtcManager) {
            window.webrtcManager.broadcastToAllPeers('combat', {
                type: 'shot',
                projectile: projectile,
                playerPosition: { x: player.x, y: player.y },
                timestamp: performance.now(),
                weapon: this.selectedWeapon
            });
        }
        
        // Visual effects
        this.createMuzzleFlash(player.x, player.y, angle);
        this.playFireSound(this.selectedWeapon);
    }
    
    fireShotgun(player, baseAngle, weapon) {
        const pelletSpread = 0.3; // Radians
        
        for (let i = 0; i < weapon.pellets; i++) {
            const angle = baseAngle + (Math.random() - 0.5) * pelletSpread;
            const projectileId = `${player.id}-${Date.now()}-${i}-${Math.random()}`;
            
            const projectile = {
                id: projectileId,
                x: player.x + Math.cos(angle) * 20,
                y: player.y + Math.sin(angle) * 20,
                velocity: {
                    x: Math.cos(angle) * weapon.projectileSpeed,
                    y: Math.sin(angle) * weapon.projectileSpeed
                },
                ownerId: player.id,
                damage: weapon.damage / weapon.pellets, // Damage split among pellets
                color: weapon.color,
                radius: 2,
                createdAt: performance.now(),
                lifetime: (weapon.range / weapon.projectileSpeed) * 1000,
                weapon: this.selectedWeapon,
                isPellet: true
            };
            
            this.gameEngine.projectiles.set(projectileId, projectile);
            this.recordShot(projectileId, projectile);
        }
        
        // Send shotgun blast to other players
        if (window.webrtcManager) {
            window.webrtcManager.broadcastToAllPeers('combat', {
                type: 'shotgunBlast',
                baseAngle: baseAngle,
                playerPosition: { x: player.x, y: player.y },
                timestamp: performance.now(),
                weapon: this.selectedWeapon,
                pelletCount: weapon.pellets
            });
        }
        
        this.createMuzzleFlash(player.x, player.y, baseAngle);
        this.playFireSound(this.selectedWeapon);
    }
    
    recordShot(projectileId, projectile) {
        // Store shot for validation
        this.recentShots.set(projectileId, {
            ...projectile,
            firedAt: performance.now()
        });
        
        // Clean up old shots
        if (this.recentShots.size > this.maxShotHistory) {
            const oldestKey = this.recentShots.keys().next().value;
            this.recentShots.delete(oldestKey);
        }
    }
    
    createMuzzleFlash(x, y, angle) {
        // Create muzzle flash particles
        for (let i = 0; i < 8; i++) {
            this.gameEngine.particles.push({
                x: x + Math.cos(angle) * 25,
                y: y + Math.sin(angle) * 25,
                velocity: {
                    x: Math.cos(angle + (Math.random() - 0.5) * 0.5) * 100,
                    y: Math.sin(angle + (Math.random() - 0.5) * 0.5) * 100
                },
                color: '#ffaa00',
                size: Math.random() * 4 + 2,
                life: 0.1,
                maxLife: 0.1,
                alpha: 1
            });
        }
    }
    
    playFireSound(weaponType) {
        // Simple audio feedback (you could replace with actual sounds)
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        // Different sounds for different weapons
        switch (weaponType) {
            case 'pistol':
                oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
                break;
            case 'rifle':
                oscillator.frequency.setValueAtTime(600, audioContext.currentTime);
                break;
            case 'shotgun':
                oscillator.frequency.setValueAtTime(400, audioContext.currentTime);
                break;
        }
        
        oscillator.type = 'square';
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.1);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.1);
    }
    
    showRangeIndicator() {
        const indicator = document.createElement('div');
        indicator.textContent = 'OUT OF RANGE';
        indicator.style.cssText = `
            position: absolute;
            top: 60%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #ff0000;
            font-weight: bold;
            font-size: 16px;
            pointer-events: none;
            z-index: 1000;
            animation: fadeOut 1s ease-out forwards;
        `;
        
        document.body.appendChild(indicator);
        setTimeout(() => indicator.remove(), 1000);
    }
    
    // Handle incoming shots from other players
    handleRemoteShot(message) {
        const { projectile, playerPosition, timestamp, weapon } = message;
        
        // Basic validation
        if (!this.validateShot(projectile, playerPosition, timestamp, weapon)) {
            console.warn('Invalid shot received from', message.fromPeer);
            return;
        }
        
        // Add projectile to game
        this.gameEngine.projectiles.set(projectile.id, projectile);
    }
    
    handleRemoteShotgunBlast(message) {
        const { baseAngle, playerPosition, timestamp, weapon, pelletCount } = message;
        const weaponDef = this.weapons[weapon];
        
        if (!weaponDef) return;
        
        const pelletSpread = 0.3;
        
        for (let i = 0; i < pelletCount; i++) {
            const angle = baseAngle + (Math.random() - 0.5) * pelletSpread;
            const projectileId = `${message.fromPeer}-${timestamp}-${i}`;
            
            const projectile = {
                id: projectileId,
                x: playerPosition.x + Math.cos(angle) * 20,
                y: playerPosition.y + Math.sin(angle) * 20,
                velocity: {
                    x: Math.cos(angle) * weaponDef.projectileSpeed,
                    y: Math.sin(angle) * weaponDef.projectileSpeed
                },
                ownerId: message.fromPeer,
                damage: weaponDef.damage / weaponDef.pellets,
                color: weaponDef.color,
                radius: 2,
                createdAt: performance.now(),
                lifetime: (weaponDef.range / weaponDef.projectileSpeed) * 1000,
                weapon: weapon,
                isPellet: true
            };
            
            this.gameEngine.projectiles.set(projectileId, projectile);
        }
    }
    
    validateShot(projectile, playerPosition, timestamp, weapon) {
        const weaponDef = this.weapons[weapon];
        if (!weaponDef) return false;
        
        // Check if damage is reasonable
        if (projectile.damage > weaponDef.damage * 1.1) { // 10% tolerance
            return false;
        }
        
        // Check if speed is reasonable
        const speed = Math.sqrt(
            projectile.velocity.x * projectile.velocity.x + 
            projectile.velocity.y * projectile.velocity.y
        );
        if (speed > weaponDef.projectileSpeed * 1.1) {
            return false;
        }
        
        // Check timestamp (basic lag compensation)
        const latency = performance.now() - timestamp;
        if (latency > this.maxLatencyTolerance) {
            return false;
        }
        
        return true;
    }
    
    // Handle hit confirmation from other players
    handleHitConfirmation(message) {
        const { targetId, damage, projectileId } = message;
        
        // Verify we fired this shot
        if (this.recentShots.has(projectileId)) {
            this.combatStats.shotsHit++;
            this.combatStats.damageDealt += damage;
            
            // Visual feedback
            this.showHitIndicator(damage);
            
            // Clean up shot record
            this.recentShots.delete(projectileId);
            
            console.log(`✅ Hit confirmed: ${damage} damage to ${targetId}`);
        }
    }
    
    handleDamageReceived(message) {
        const { damage, attackerId, projectileId } = message;
        
        this.combatStats.damageReceived += damage;
        
        // Update health
        if (this.gameEngine.localPlayer) {
            this.gameEngine.localPlayer.health = Math.max(0, 
                this.gameEngine.localPlayer.health - damage);
            
            // Check for death
            if (this.gameEngine.localPlayer.health <= 0) {
                this.handleDeath(attackerId);
            }
        }
        
        // Send hit confirmation back to attacker
        if (window.webrtcManager) {
            window.webrtcManager.sendToPeer(attackerId, 'combat', {
                type: 'hitConfirmed',
                targetId: this.gameEngine.localPlayer?.id,
                damage: damage,
                projectileId: projectileId
            });
        }
        
        // Visual effect
        this.createDamageIndicator(damage);
    }
    
    handleDeath(killerId) {
        this.combatStats.deaths++;
        
        // Respawn after delay
        setTimeout(() => {
            this.respawnPlayer();
        }, 3000);
        
        // Notify killer
        if (window.webrtcManager && killerId) {
            window.webrtcManager.sendToPeer(killerId, 'combat', {
                type: 'killConfirmed',
                killedPlayerId: this.gameEngine.localPlayer?.id
            });
        }
        
        console.log('💀 Player died');
    }
    
    respawnPlayer() {
        if (this.gameEngine.localPlayer) {
            // Reset health
            this.gameEngine.localPlayer.health = this.gameEngine.localPlayer.maxHealth;
            
            // Random spawn position
            this.gameEngine.localPlayer.x = Math.random() * (this.gameEngine.worldSize.width - 100) + 50;
            this.gameEngine.localPlayer.y = Math.random() * (this.gameEngine.worldSize.height - 100) + 50;
            
            console.log('🔄 Player respawned');
        }
    }
    
    showHitIndicator(damage) {
        const indicator = document.createElement('div');
        indicator.textContent = `+${damage}`;
        indicator.className = 'hit-indicator';
        indicator.style.cssText = `
            position: absolute;
            top: 30%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #00ff00;
            font-weight: bold;
            font-size: 20px;
            pointer-events: none;
            z-index: 1000;
        `;
        
        document.body.appendChild(indicator);
        setTimeout(() => indicator.remove(), 1000);
    }
    
    createDamageIndicator(damage) {
        const indicator = document.createElement('div');
        indicator.textContent = `-${damage}`;
        indicator.className = 'hit-indicator';
        indicator.style.cssText = `
            position: absolute;
            top: 70%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #ff0000;
            font-weight: bold;
            font-size: 18px;
            pointer-events: none;
            z-index: 1000;
        `;
        
        document.body.appendChild(indicator);
        setTimeout(() => indicator.remove(), 1000);
    }
    
    getStats() {
        return {
            ...this.combatStats,
            accuracy: this.combatStats.shotsFired > 0 ? 
                     (this.combatStats.shotsHit / this.combatStats.shotsFired * 100).toFixed(1) : 0,
            kdr: this.combatStats.deaths > 0 ? 
                 (this.combatStats.kills / this.combatStats.deaths).toFixed(2) : this.combatStats.kills
        };
    }
    
    getCurrentWeapon() {
        return {
            type: this.selectedWeapon,
            ...this.weapons[this.selectedWeapon]
        };
    }
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        0% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        100% { opacity: 0; transform: translate(-50%, -50%) scale(1.2); }
    }
`;
document.head.appendChild(style);