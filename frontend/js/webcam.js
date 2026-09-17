const CAMERA_PREF_KEY = 'emotionlens-camera-id';

class WebcamManager {
    constructor() {
        this.videoElement = document.getElementById('webcam-video');
        this.canvasElement = document.getElementById('overlay-canvas');
        this.ctx = this.canvasElement.getContext('2d');

        this.stream = null;
        this.deviceId = this.loadPreferredDevice();

        // Hidden canvas for downscaling frames to send to backend
        this.captureCanvas = document.createElement('canvas');
        this.captureCanvas.width = CONFIG.CAPTURE_WIDTH;
        this.captureCanvas.height = CONFIG.CAPTURE_HEIGHT;
        this.captureCtx = this.captureCanvas.getContext('2d');
    }

    loadPreferredDevice() {
        // Reading storage throws in some privacy modes, and the session must
        // still start, so a failure just falls back to the system default.
        try {
            return localStorage.getItem(CAMERA_PREF_KEY) || null;
        } catch (err) {
            return null;
        }
    }

    setPreferredDevice(deviceId) {
        this.deviceId = deviceId || null;
        try {
            if (this.deviceId) {
                localStorage.setItem(CAMERA_PREF_KEY, this.deviceId);
            } else {
                localStorage.removeItem(CAMERA_PREF_KEY);
            }
        } catch (err) {
            /* preference is not persisted, but the choice still applies now */
        }
    }

    async listCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(d => d.kind === 'videoinput');
        } catch (err) {
            console.error("Error listing cameras: ", err);
            return [];
        }
    }

    buildConstraints() {
        const video = {
            width: { ideal: 1280 },
            height: { ideal: 720 }
        };
        if (this.deviceId) {
            video.deviceId = { exact: this.deviceId };
        } else {
            video.facingMode = "user";
        }
        return { video, audio: false };
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia(this.buildConstraints());
        } catch (err) {
            // A remembered camera that is no longer plugged in must not leave
            // the user stuck: drop the preference and retry with the default.
            if (this.deviceId) {
                console.warn(`Remembered camera unavailable (${err.name}), falling back to default.`);
                this.setPreferredDevice(null);
                return this.start();
            }
            console.error("Error accessing webcam: ", err);
            alert("No se pudo acceder a la cámara. Verifica que los permisos estén concedidos.");
            return false;
        }

        this.videoElement.srcObject = this.stream;

        // Wait for video to load metadata to get dimensions
        return new Promise((resolve) => {
            this.videoElement.onloadedmetadata = () => {
                this.videoElement.play();
                this.resizeCanvas();
                resolve(true);
            };
        });
    }

    /**
     * Switch to another camera, restarting the stream if one is already live.
     */
    async switchCamera(deviceId) {
        this.setPreferredDevice(deviceId);
        if (!this.stream) return true;
        this.stop();
        return this.start();
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
