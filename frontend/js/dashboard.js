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
        this.collectedMicroExpressions = [];
        
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

        // Handle quality gate warnings (frame discarded)
        if (data.type === 'quality_warning') {
            this.updateCameraQuality(data.camera_quality);
            return;
        }

        if (data.type === 'frame_result') {
            this.updateCalibration(data.is_calibrating, data.calibration_progress);
            this.updateEmotion(data.emotion, data.confidence);
            this.updateHeartRate(data.heart_rate);
            this.updateCongruence(data.congruence_score, data.congruence_breakdown);
            this.updateEVMFrame(data.evm_frame, data.preprocessed_frame);
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
        const suggestions = quality.suggestions || [];
        const gate = quality.quality_gate || 'pass';

        // ── Overall score label ──
        if (this.elCameraQualityValue) {
            this.elCameraQualityValue.className = 'quality-indicator';
            if (gate === 'pass') {
                this.elCameraQualityValue.textContent = `${Math.round(score * 100)}% ✓`;
                this.elCameraQualityValue.classList.add('good');
            } else if (gate === 'degraded') {
                this.elCameraQualityValue.textContent = `${Math.round(score * 100)}% ⚠`;
                this.elCameraQualityValue.classList.add('fair');
            } else {
                this.elCameraQualityValue.textContent = `${Math.round(score * 100)}% ✗`;
                this.elCameraQualityValue.classList.add('poor');
            }
        }

        // ── Quality bars ──
        this._updateQualityBar('qbar-brightness', quality.brightness, 0.2, 0.5);
        this._updateQualityBar('qbar-contrast', (quality.contrast || 40) / 80, 0.3, 0.6);
        this._updateQualityBar('qbar-sharpness', Math.min((quality.sharpness || 100) / 200, 1.0), 0.25, 0.5);
        
        // Position bar: based on head pose
        const pose = quality.head_pose || {yaw: 0, pitch: 0};
        const poseScore = Math.max(0, 1.0 - (Math.abs(pose.yaw) + Math.abs(pose.pitch)) / 40);
        this._updateQualityBar('qbar-position', poseScore, 0.4, 0.7);
        
        this._updateQualityBar('qbar-balance', quality.lighting_balance || 1.0, 0.6, 0.8);

        // ── Warnings area ──
        if (this.elCameraWarnings) {
            if (warnings.length > 0) {
                this.elCameraWarnings.innerHTML = warnings.map(w => `<div>⚠️ ${w}</div>`).join('');
                this.elCameraWarnings.classList.remove('hidden');
            } else {
                this.elCameraWarnings.classList.add('hidden');
                this.elCameraWarnings.innerHTML = '';
            }
        }

        // ── Suggestions (rotate every 5s) ──
        const sugBox = document.getElementById('quality-suggestions');
        if (sugBox && suggestions.length > 0) {
            if (!this._suggestionIdx) this._suggestionIdx = 0;
            const now = Date.now();
            if (!this._lastSuggestionTime || now - this._lastSuggestionTime > 5000) {
                this._suggestionIdx = (this._suggestionIdx + 1) % suggestions.length;
                this._lastSuggestionTime = now;
            }
            sugBox.textContent = '💡 ' + suggestions[this._suggestionIdx];
            sugBox.classList.remove('hidden');
        } else if (sugBox) {
            sugBox.classList.add('hidden');
        }

        // ── Quality gate banner ──
        const banner = document.getElementById('quality-gate-banner');
        if (banner) {
            if (gate === 'fail') {
                banner.textContent = '⛔ Calidad insuficiente — sigue las sugerencias para continuar';
                banner.classList.remove('hidden');
            } else {
                banner.classList.add('hidden');
            }
        }
    }

    _updateQualityBar(id, value, redThreshold, greenThreshold) {
        const bar = document.getElementById(id);
        if (!bar) return;
        const pct = Math.max(0, Math.min(100, value * 100));
        bar.style.width = `${pct}%`;
        // Color: red below redThreshold, yellow in between, green above greenThreshold
        if (value < redThreshold) {
            bar.className = 'quality-bar-fill bar-red';
        } else if (value < greenThreshold) {
            bar.className = 'quality-bar-fill bar-yellow';
        } else {
            bar.className = 'quality-bar-fill bar-green';
        }
    }

    updateCalibration(isCalibrating, progress) {
        if (isCalibrating) {
            this.elCalibration.classList.remove('hidden');
            this.elCalibrationFill.style.width = `${progress * 100}%`;
            
            // Draw face guide overlay (Fase 5.2)
            if (this.webcam) {
                this.webcam.drawFaceGuide();
            }

            // Show calibration instructions
            const instrEl = document.getElementById('calibration-instructions');
            if (instrEl) {
                const remaining = Math.max(0, Math.ceil(30 * (1 - progress)));
                instrEl.innerHTML = `
                    <div class="calib-title">🎯 Calibrando — ${remaining}s restantes</div>
                    <div class="calib-tips">
                        <span>😐 Expresion neutral</span>
                        <span>👀 Mira a la camara</span>
                        <span>🤫 No hables</span>
                        <span>👁️ Parpadea normal</span>
                    </div>
                `;
                instrEl.classList.remove('hidden');
            }
        } else {
            this.elCalibration.classList.add('hidden');
            const instrEl = document.getElementById('calibration-instructions');
            if (instrEl) instrEl.classList.add('hidden');

            // Clear face guide overlay (Fase 5.2)
            if (this.webcam) {
                this.webcam.clearOverlay();
            }
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

    updateEVMFrame(evmFrameB64, preprocessedFrameB64) {
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
        } else if (preprocessedFrameB64) {
            if (processedVideo) {
                processedVideo.src = preprocessedFrameB64;
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
        this.collectedMicroExpressions.push(micro);
        
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
