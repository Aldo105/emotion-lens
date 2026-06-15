class DashboardUI {
    constructor(chartsManager) {
        this.charts = chartsManager;
        
        // DOM Elements
        this.elStatusBadge = document.getElementById('connection-status');
        this.elStatusDot = this.elStatusBadge.querySelector('.dot');
        this.elStatusText = this.elStatusBadge.querySelector('.status-text');
        
        this.elCalibration = document.getElementById('calibration-bar');
        this.elCalibrationFill = document.getElementById('calibration-fill');
        
        this.elEmotionIcon = document.getElementById('current-emotion-icon');
        this.elEmotionName = document.getElementById('current-emotion-name');
        this.elConfidenceFill = document.getElementById('emotion-confidence-fill');
        this.elConfidenceText = document.getElementById('emotion-confidence-text');
        
        this.elHrBpm = document.getElementById('hr-bpm');
        this.elHrHeart = document.getElementById('hr-heart-icon');
        this.elHrStressText = document.getElementById('hr-stress-text');
        
        this.elCongruenceValue = document.getElementById('congruence-value');
        this.elCongruenceLabel = document.getElementById('congruence-label');
        
        this.elBreakdowns = {
            stability: document.getElementById('breakdown-stability'),
            alignment: document.getElementById('breakdown-alignment'),
            baseline: document.getElementById('breakdown-baseline'),
            physio: document.getElementById('breakdown-physio')
        };
        
        this.elMicroLog = document.getElementById('micro-log-list');
        this.elMicroCount = document.getElementById('micro-count');
        this.microCount = 0;
        
        // Camera quality elements
        this.elCameraQualityValue = document.getElementById('camera-quality-value');
        this.elCameraWarnings = document.getElementById('camera-warnings');
        
        // Animation state
        this.heartbeatTimeout = null;
    }

    updateStatus(status) {
        this.elStatusDot.className = 'dot'; // reset
        if (status === 'connected') {
            this.elStatusDot.classList.add('active');
            this.elStatusText.textContent = 'Live Analysis';
        } else if (status === 'disconnected') {
            this.elStatusDot.classList.add('disconnected');
            this.elStatusText.textContent = 'Disconnected';
        } else {
            this.elStatusDot.classList.add('disconnected');
            this.elStatusText.textContent = 'Error';
        }
    }

    handleMessage(data) {
        if (data.type === 'status' || data.type === 'calibration_complete') {
            console.log("Status:", data.message);
            return;
        }

        if (data.type === 'frame_result') {
            this.updateCalibration(data.is_calibrating, data.calibration_progress);
            this.updateEmotion(data.emotion, data.confidence);
            this.updateHeartRate(data.heart_rate);
            this.updateCongruence(data.congruence_score, data.congruence_breakdown);
            this.updateEVMFrame(data.evm_frame);
            this.updateCameraQuality(data.camera_quality);
            
            if (data.micro_expression) {
                this.addMicroExpression(data.micro_expression);
            }
            
            // Update charts
            this.charts.updateTimeline(data.timestamp, data.emotion_probabilities);
        }
    }

    updateCameraQuality(quality) {
        if (!quality) return;

        const score = quality.score;
        const warnings = quality.warnings || [];

        if (this.elCameraQualityValue) {
            this.elCameraQualityValue.className = 'quality-indicator';
            if (score > 0.8) {
                this.elCameraQualityValue.textContent = 'Good';
                this.elCameraQualityValue.classList.add('good');
            } else if (score > 0.5) {
                this.elCameraQualityValue.textContent = 'Fair';
                this.elCameraQualityValue.classList.add('fair');
            } else {
                this.elCameraQualityValue.textContent = 'Poor';
                this.elCameraQualityValue.classList.add('poor');
            }
        }

        if (this.elCameraWarnings) {
            if (warnings.length > 0) {
                this.elCameraWarnings.innerHTML = warnings.map(w => `<div>⚠️ ${w}</div>`).join('');
                this.elCameraWarnings.classList.remove('hidden');
            } else {
                this.elCameraWarnings.classList.add('hidden');
                this.elCameraWarnings.innerHTML = '';
            }
        }
    }

    updateCalibration(isCalibrating, progress) {
        if (isCalibrating) {
            this.elCalibration.classList.remove('hidden');
            this.elCalibrationFill.style.width = `${progress * 100}%`;
        } else {
            this.elCalibration.classList.add('hidden');
        }
    }

    updateEmotion(emotionKey, confidence) {
        const config = CONFIG.EMOTIONS[emotionKey] || CONFIG.EMOTIONS['neutral'];
        const confPercent = Math.round(confidence * 100);
        
        this.elEmotionIcon.textContent = config.icon;
        this.elEmotionName.textContent = config.label;
        this.elEmotionName.style.color = config.color;
        
        this.elConfidenceFill.style.width = `${confPercent}%`;
        this.elConfidenceFill.style.backgroundColor = config.color;
        this.elConfidenceText.textContent = `${confPercent}%`;
    }

    updateHeartRate(hrData) {
        if (!hrData || !hrData.signal_ready) {
            this.elHrBpm.textContent = '--';
            this.elHrStressText.textContent = 'Calibrating...';
            return;
        }

        this.elHrBpm.textContent = Math.round(hrData.bpm);
        
        // Stress mapping
        const stressVal = hrData.stress_indicator;
        let stressText = "Low";
        let stressColor = "var(--text-muted)";
        if (stressVal > 0.6) {
            stressText = "High";
            stressColor = "var(--danger)";
        } else if (stressVal > 0.3) {
            stressText = "Moderate";
            stressColor = "var(--warning)";
        }
        
        this.elHrStressText.textContent = stressText;
        this.elHrStressText.style.color = stressColor;

        // Visual heartbeat pulse
        this.elHrHeart.classList.add('beat');
        if (this.heartbeatTimeout) clearTimeout(this.heartbeatTimeout);
        this.heartbeatTimeout = setTimeout(() => {
            this.elHrHeart.classList.remove('beat');
        }, 150);
    }

    updateEVMFrame(evmFrameB64) {
        const toggleEvm = document.getElementById('toggle-evm');
        const processedVideo = document.getElementById('processed-video');
        const webcamVideo = document.getElementById('webcam-video');
        
        if (toggleEvm && toggleEvm.checked && evmFrameB64) {
            if (processedVideo) {
                processedVideo.src = evmFrameB64;
                processedVideo.style.opacity = '1';
                processedVideo.style.pointerEvents = 'auto';
            }
            if (webcamVideo) {
                webcamVideo.style.opacity = '0';
                webcamVideo.style.pointerEvents = 'none';
            }
        } else {
            if (processedVideo) {
                processedVideo.style.opacity = '0';
                processedVideo.style.pointerEvents = 'none';
            }
            if (webcamVideo) {
                webcamVideo.style.opacity = '1';
                webcamVideo.style.pointerEvents = 'auto';
            }
        }
    }

    updateCongruence(score, breakdown) {
        if (score === 0 && (!breakdown || Object.keys(breakdown).length === 0)) {
            // Not calibrated yet
            this.elCongruenceValue.textContent = '--';
            this.elCongruenceLabel.textContent = 'Calibrating...';
            this.charts.updateGauge(0, 'grey');
            return;
        }

        const s = Math.round(score);
        this.elCongruenceValue.textContent = s;
        
        let label = "High Match";
        let color = "green"; // Passed to chart logic
        this.elCongruenceValue.style.color = "var(--success)";
        
        if (s < 50) {
            label = "Low Match";
            color = "red";
            this.elCongruenceValue.style.color = "var(--danger)";
        } else if (s < 80) {
            label = "Moderate";
            color = "yellow";
            this.elCongruenceValue.style.color = "var(--warning)";
        }

        this.elCongruenceLabel.textContent = label;
        this.charts.updateGauge(s, color);

        // Update breakdown
        if (breakdown) {
            this.elBreakdowns.stability.textContent = Math.round(breakdown.stability || 0);
            this.elBreakdowns.alignment.textContent = Math.round(breakdown.micro_alignment || 0);
            this.elBreakdowns.baseline.textContent = Math.round(breakdown.baseline || 0);
            this.elBreakdowns.physio.textContent = Math.round(breakdown.physiological || 0);
        }
    }

    addMicroExpression(micro) {
        this.microCount++;
        this.elMicroCount.textContent = `${this.microCount} Detected`;
        
        // Remove empty state if present
        const emptyState = this.elMicroLog.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        const mins = Math.floor(micro.timestamp / 60).toString().padStart(2, '0');
        const secs = Math.floor(micro.timestamp % 60).toString().padStart(2, '0');
        
        const div = document.createElement('div');
        div.className = `micro-item ${micro.is_contradictory ? 'contradictory' : ''}`;
        
        const em = CONFIG.EMOTIONS[micro.detected_emotion] || CONFIG.EMOTIONS['neutral'];
        
        div.innerHTML = `
            <span class="micro-time">${mins}:${secs}</span>
            <span class="micro-desc">${em.icon} <b>${em.label}</b> micro-expression</span>
            <span class="micro-score">Score: ${micro.relevance_score}</span>
        `;
        
        this.elMicroLog.prepend(div);
        
        // Keep only last 50 items
        if (this.elMicroLog.children.length > 50) {
            this.elMicroLog.removeChild(this.elMicroLog.lastChild);
        }
    }
}
