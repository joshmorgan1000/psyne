/**
 * @file game-engine-fixed.js
 * @brief Fixed version of game engine with proper input handling
 * 
 * This is a minimal fix for the input blocking issue.
 * Include this after the original game-engine.js to override the problematic method.
 */

// Override the setupEventListeners method to fix input blocking
if (window.GameEngine) {
    GameEngine.prototype.setupEventListeners = function() {
        // Keyboard input
        window.addEventListener('keydown', (e) => {
            // Don't capture keys when typing in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            this.keys[e.code] = true;
            
            // Only prevent default for game control keys
            if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 
                 'Digit1', 'Digit2', 'Digit3', 'Space'].includes(e.code)) {
                e.preventDefault();
            }
        });
        
        window.addEventListener('keyup', (e) => {
            // Don't capture keys when typing in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            this.keys[e.code] = false;
            
            // Only prevent default for game control keys
            if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 
                 'Digit1', 'Digit2', 'Digit3', 'Space'].includes(e.code)) {
                e.preventDefault();
            }
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
    };
}