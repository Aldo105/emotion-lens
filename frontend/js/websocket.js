class WebSocketManager {
    constructor(onMessageCallback, onStatusChangeCallback) {
        this.ws = null;
        this.onMessage = onMessageCallback;
        this.onStatusChange = onStatusChangeCallback;
        this.isConnected = false;
        
        this.frameInterval = null;
        this.webcam = null;
        this.recalibrateRequested = false;
    }

    connect(webcamManager) {
        this.webcam = webcamManager;
        this.ws = new WebSocket(CONFIG.WS_URL);

        this.ws.onopen = () => {
            console.log("WebSocket Connected");
            this.isConnected = true;
            this.onStatusChange('connected');
            this.startFrameLoop();
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.onMessage(data);
            } catch (err) {
                console.error("Error parsing WS message:", err);
            }
        };

        this.ws.onclose = () => {
            console.log("WebSocket Disconnected");
            this.isConnected = false;
            this.onStatusChange('disconnected');
            this.stopFrameLoop();
        };

        this.ws.onerror = (error) => {
            console.error("WebSocket Error:", error);
            this.onStatusChange('error');
        };
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
        this.stopFrameLoop();
    }

    requestRecalibrate() {
        if (this.isConnected) {
            this.recalibrateRequested = true;
        }
    }

    startFrameLoop() {
        if (this.frameInterval) return;
        
        const delayMs = 1000 / CONFIG.CAPTURE_FPS;
        this.frameInterval = setInterval(() => {
            if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                const b64Frame = this.webcam.captureFrameBase64();
                if (b64Frame) {
                    const toggleEvm = document.getElementById('toggle-evm');
                    const evmChecked = toggleEvm ? toggleEvm.checked : false;
                    
                    const payload = {
                        frame: b64Frame,
                        evm: evmChecked
                    };
                    
                    if (this.recalibrateRequested) {
                        payload.recalibrate = true;
                        this.recalibrateRequested = false;
                    }
                    
                    this.ws.send(JSON.stringify(payload));
                }
            }
        }, delayMs);
    }

    stopFrameLoop() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
    }
}
