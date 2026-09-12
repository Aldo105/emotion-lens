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
        
        // Convert to base64 jpeg string (quality 0.85 for AU precision)
        return this.captureCanvas.toDataURL('image/jpeg', 0.85);
    }

    /**
     * Draw a semi-transparent overlay with a centered face-shape guide.
     */
    drawFaceGuide(color = 'rgba(79, 70, 229, 0.6)') {
        if (!this.canvasElement) return;
        const w = this.canvasElement.width;
        const h = this.canvasElement.height;
        this.ctx.clearRect(0, 0, w, h);
        
        // Dark overlay outside the guide
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        this.ctx.fillRect(0, 0, w, h);
        
        // Clip oval guide
        this.ctx.save();
        this.ctx.beginPath();
        const cx = w / 2;
        const cy = h / 2;
        const rx = w * 0.25;
        const ry = h * 0.35;
        this.ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        
        this.ctx.clip();
        this.ctx.clearRect(0, 0, w, h);
        this.ctx.restore();
        
        // Draw dashed border
        this.ctx.beginPath();
        this.ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([8, 4]);
        this.ctx.stroke();
        
        // Draw instructions label
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 12px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('CENTRA TU ROSTRO AQUI', cx, cy - ry - 15);
    }

    /**
     * Clear all drawings/overlays from the canvas.
     */
    clearOverlay() {
        if (!this.canvasElement) return;
        this.ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    }
}
