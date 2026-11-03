class WebSocketManager {
    constructor(onMessageCallback, onStatusChangeCallback) {
        this.ws = null;
        this.onMessage = onMessageCallback;
        this.onStatusChange = onStatusChangeCallback;
        this.isConnected = false;
        
        this.frameInterval = null;
        this.webcam = null;
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
                if ((data.type === 'frame_result' || data.type === 'quality_warning') && this.inFlight > 0) {
                    this.inFlight--;
                }
                if (data.type === 'status' && data.data && data.data.session_id) {
                    this.lastSessionId = data.data.session_id;
                    console.log("[WS] Session ID initialized:", this.lastSessionId);
                }
                this.onMessage(data);
            } catch (err) {
                console.error("Error parsing WS message:", err);
            }
        };

        this.ws.onclose = (event) => {
            // 4401: the server refused the socket because the session expired
            // (backend/app/auth_middleware.py). goToLogin lives in app.js.
            if (event.code === 4401 && typeof goToLogin === 'function') {
                goToLogin();
                return;
            }
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

    startFrameLoop() {
        if (this.frameInterval) return;

        // Backpressure. The server analyses ~17 frames/s on its 2 vCPUs while
        // the camera loop asks for 20, and it never drops stale frames, so a
        // queue grew for as long as the session ran (2 s late after 15 s, and
        // 10 s late with two people connected). Never keep more than
        // MAX_IN_FLIGHT frames unanswered: the rate settles at what the
        // server can do and the emotion shown stays current. Every frame is
        // answered by one frame_result or quality_warning, except an
        // undecodable one, so a silence of STALL_MS resets the count rather
        // than freezing the stream.
        const MAX_IN_FLIGHT = 2;
        const STALL_MS = 1500;
        this.inFlight = 0;
        this.lastSentAt = 0;

        const delayMs = 1000 / CONFIG.CAPTURE_FPS;
        this.frameInterval = setInterval(() => {
            if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                const now = performance.now();
                if (this.inFlight >= MAX_IN_FLIGHT) {
                    if (now - this.lastSentAt < STALL_MS) return;
                    this.inFlight = 0;
                }
                const b64Frame = this.webcam.captureFrameBase64();
                if (b64Frame) {
                    const toggleEvm = document.getElementById('toggle-evm');
                    const evmChecked = toggleEvm ? toggleEvm.checked : false;
                    
                    const payload = {
                        frame: b64Frame,
                        evm: evmChecked
                    };
                    
                    this.ws.send(JSON.stringify(payload));
                    this.inFlight++;
                    this.lastSentAt = now;
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
