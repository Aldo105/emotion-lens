class WebcamManager {
    constructor() {
        this.videoElement = document.getElementById('webcam-video');
        this.canvasElement = document.getElementById('overlay-canvas');
        this.ctx = this.canvasElement.getContext('2d');
        
        this.stream = null;
        
        // Hidden canvas for downscaling frames to send to backend
        this.captureCanvas = document.createElement('canvas');
        this.captureCanvas.width = CONFIG.CAPTURE_WIDTH;
        this.captureCanvas.height = CONFIG.CAPTURE_HEIGHT;
        this.captureCtx = this.captureCanvas.getContext('2d');
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: "user"
                },
                audio: false
            });
            this.videoElement.srcObject = this.stream;
            
            // Wait for video to load metadata to get dimensions
            return new Promise((resolve) => {
                this.videoElement.onloadedmetadata = () => {
                    this.videoElement.play();
                    this.resizeCanvas();
                    resolve(true);
                };
            });
        } catch (err) {
            console.error("Error accessing webcam: ", err);
            alert("Could not access webcam. Please ensure permissions are granted.");
            return false;
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.videoElement.srcObject = null;
        }
    }

    resizeCanvas() {
        const rect = this.videoElement.getBoundingClientRect();
        this.canvasElement.width = rect.width;
        this.canvasElement.height = rect.height;
    }

    /**
     * Grabs current video frame, scales it down, and returns as Base64 JPEG.
     */
    captureFrameBase64() {
        if (!this.stream || this.videoElement.readyState < 2) return null;
        
        // Draw video frame to hidden canvas
        this.captureCtx.drawImage(
            this.videoElement, 
            0, 0, 
            CONFIG.CAPTURE_WIDTH, 
            CONFIG.CAPTURE_HEIGHT
        );
        
        // Convert to base64 jpeg string (quality 0.7 for bandwidth)
        return this.captureCanvas.toDataURL('image/jpeg', 0.7);
    }
}
