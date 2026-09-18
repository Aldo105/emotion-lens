class DashboardUI {
    constructor(chartsManager) {
        this.charts = chartsManager;

        // ── DOM refs cached once at construction (never getElementById on hot path) ──
        this.elStatusBadge    = document.getElementById('connection-status');
        this.elStatusDot      = this.elStatusBadge.querySelector('.dot');
        this.elStatusText     = this.elStatusBadge.querySelector('.status-text');

        this.elCalibration     = document.getElementById('calibration-bar');
        this.elCalibrationFill = document.getElementById('calibration-fill');
        this.elCalibInstr      = document.getElementById('calibration-instructions');

        this.elEmotionIcon      = document.getElementById('current-emotion-icon');
        this.elEmotionName      = document.getElementById('current-emotion-name');
        this.elConfidenceFill   = document.getElementById('emotion-confidence-fill');
        this.elConfidenceText   = document.getElementById('emotion-confidence-text');

        this.elHrBpm        = document.getElementById('hr-bpm');
        this.elHrHeart      = document.getElementById('hr-heart-icon');
        this.elHrStressText = document.getElementById('hr-stress-text');

        this.elCongruenceValue = document.getElementById('congruence-value');
        this.elCongruenceLabel = document.getElementById('congruence-label');

        this.elLiveScoreValue = document.getElementById('live-score-value');
        this.elLiveScoreFill  = document.getElementById('live-score-fill');
        this.elLiveScoreLabel = document.getElementById('live-score-label');

        this.elBreakdowns = {
            stability: document.getElementById('breakdown-stability'),
            alignment: document.getElementById('breakdown-alignment'),
            baseline:  document.getElementById('breakdown-baseline'),
            physio:    document.getElementById('breakdown-physio'),
        };

        this.elMicroLog   = document.getElementById('micro-log-list');
        this.elMicroCount = document.getElementById('micro-count');
        this.microCount   = 0;
        this.collectedMicroExpressions = [];

        // Camera quality elements
        this.elCameraQualityValue = document.getElementById('camera-quality-value');
        this.elCameraWarnings     = document.getElementById('camera-warnings');
        this.elQualityBars = {
            brightness: document.getElementById('qbar-brightness'),
            contrast:   document.getElementById('qbar-contrast'),
            sharpness:  document.getElementById('qbar-sharpness'),
            position:   document.getElementById('qbar-position'),
            balance:    document.getElementById('qbar-balance'),
        };
        this.elQualitySuggestions = document.getElementById('quality-suggestions');
        this.elQualityGateBanner  = document.getElementById('quality-gate-banner');

        // EVM / processed video elements
        this.elToggleEvm      = document.getElementById('toggle-evm');
        this.elProcessedVideo = document.getElementById('processed-video');
        this.elWebcamVideo    = document.getElementById('webcam-video');

        // Noise indicator elements
        this.elNoiseIndicator = document.getElementById('noise-indicator');
        this.elNoiseIcon      = document.getElementById('noise-icon');
        this.elNoiseLabel     = document.getElementById('noise-label');

        // Animation / suggestion state
        this.heartbeatTimeout  = null;
        this._suggestionIdx    = 0;
        this._lastSuggestionTime = 0;
    }

    updateStatus(status) {
        this.elStatusDot.className = 'dot'; // reset classes
        if (status === 'connected') {
            this.elStatusDot.classList.add('active');
            this.elStatusText.textContent = 'Análisis en vivo';
        } else if (status === 'disconnected') {
            this.elStatusDot.classList.add('disconnected');
            this.elStatusText.textContent = 'Desconectado';
        } else {
            this.elStatusDot.classList.add('disconnected');
            this.elStatusText.textContent = 'Error';
        }
    }

    handleMessage(data) {
        if (data.type === 'status' || data.type === 'calibration_complete') {
            return; // no-op for these status messages
        }

        if (data.type === 'quality_warning') {
            this.updateCameraQuality(data.camera_quality);
            return;
        }

        if (data.type === 'frame_result') {
            this.updateCalibration(
                data.is_calibrating, data.calibration_progress, data.calibration_pose
            );
            this.updateEmotion(data.emotion, data.confidence, data.is_calibrating);
            if (!data.is_calibrating) {
                this.updatePeakEmotion(data.peak_emotion);
            }
            this.updateHeartRate(data.heart_rate);
            this.updateCongruence(data.congruence_score, data.congruence_breakdown);
            this.updateLiveScore(data.live_score);
            this.updateEVMFrame(data.evm_frame);
            this.updateCameraQuality(data.camera_quality);

            if (data.micro_expression) {
                this.addMicroExpression(data.micro_expression);
            }

            this.updateNoiseState(data.noise_state);

            // Chart update throttled inside DashboardCharts. Skipped while
            // calibrating for the same reason the reading above is hidden --
            // otherwise the timeline opens with 30s of pre-baseline values.
            if (!data.is_calibrating) {
                this.charts.updateTimeline(data.timestamp, data.emotion_probabilities);
            }
        }
    }

    updateNoiseState(noiseState) {
        if (!this.elNoiseIndicator) return;

        if (!noiseState || (!noiseState.is_speaking && !noiseState.is_yawning && !noiseState.is_tic)) {
            this.elNoiseIndicator.style.display = 'none';
            return;
        }

        this.elNoiseIndicator.style.display = 'flex';

        if (noiseState.is_speaking) {
            this.elNoiseLabel.textContent = 'Hablando';
            this.elNoiseIcon.innerHTML = Icons.render('activity');
            this.elNoiseIndicator.className = 'noise-indicator speaking';
        } else if (noiseState.is_yawning) {
            this.elNoiseLabel.textContent = 'Bostezando';
            this.elNoiseIcon.innerHTML = Icons.render('clock');
            this.elNoiseIndicator.className = 'noise-indicator yawning';
        } else if (noiseState.is_tic) {
            this.elNoiseLabel.textContent = 'Tic detectado';
            this.elNoiseIcon.innerHTML = Icons.render('spark');
            this.elNoiseIndicator.className = 'noise-indicator tic';
        }
    }

    updateCameraQuality(quality) {
        if (!quality) return;

        const score    = quality.score;
        const warnings = quality.warnings   || [];
        const suggestions = quality.suggestions || [];
        const gate     = quality.quality_gate || 'pass';

        // Overall score label
        if (this.elCameraQualityValue) {
            this.elCameraQualityValue.className = 'quality-indicator';
            if (gate === 'pass') {
                this.elCameraQualityValue.textContent = `${Math.round(score * 100)}%`;
                this.elCameraQualityValue.classList.add('good');
            } else if (gate === 'degraded') {
                this.elCameraQualityValue.textContent = `${Math.round(score * 100)}%`;
                this.elCameraQualityValue.classList.add('fair');
            } else {
                this.elCameraQualityValue.textContent = `${Math.round(score * 100)}%`;
                this.elCameraQualityValue.classList.add('poor');
            }
        }

        // Quality bars (using cached refs)
        this._setBar(this.elQualityBars.brightness, quality.brightness,                                   0.2, 0.5);
        this._setBar(this.elQualityBars.contrast,   (quality.contrast   || 40) / 80,                     0.3, 0.6);
        this._setBar(this.elQualityBars.sharpness,  Math.min((quality.sharpness || 100) / 200, 1.0),     0.25, 0.5);
        const pose = quality.head_pose || { yaw: 0, pitch: 0 };
        const poseScore = Math.max(0, 1.0 - (Math.abs(pose.yaw) + Math.abs(pose.pitch)) / 40);
        this._setBar(this.elQualityBars.position,   poseScore,                                            0.4, 0.7);
        this._setBar(this.elQualityBars.balance,    quality.lighting_balance || 1.0,                     0.6, 0.8);

        // Warnings
        if (this.elCameraWarnings) {
            if (warnings.length > 0) {
                this.elCameraWarnings.innerHTML = warnings.map(w => `<div>${Icons.render('alert')} ${w}</div>`).join('');
                this.elCameraWarnings.classList.remove('hidden');
            } else {
                this.elCameraWarnings.classList.add('hidden');
                this.elCameraWarnings.innerHTML = '';
            }
        }

        // Suggestions — rotate every 5 s
        if (this.elQualitySuggestions && suggestions.length > 0) {
            const now = Date.now();
            if (now - this._lastSuggestionTime > 5000) {
                this._suggestionIdx = (this._suggestionIdx + 1) % suggestions.length;
                this._lastSuggestionTime = now;
            }
            this.elQualitySuggestions.innerHTML = Icons.render('spark') + ' ' + suggestions[this._suggestionIdx];
            this.elQualitySuggestions.classList.remove('hidden');
        } else if (this.elQualitySuggestions) {
            this.elQualitySuggestions.classList.add('hidden');
        }

        // Quality gate banner
        if (this.elQualityGateBanner) {
            if (gate === 'fail') {
                this.elQualityGateBanner.textContent = 'Calidad insuficiente — sigue las sugerencias para continuar';
                this.elQualityGateBanner.classList.remove('hidden');
            } else {
                this.elQualityGateBanner.classList.add('hidden');
            }
        }
    }

    /** Set a quality bar element's width and color class in one call. */
    _setBar(bar, value, redThreshold, greenThreshold) {
        if (!bar) return;
        bar.style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
        bar.className = 'quality-bar-fill ' + (
            value < redThreshold   ? 'bar-red' :
            value < greenThreshold ? 'bar-yellow' :
                                     'bar-green'
        );
    }

    updateCalibration(isCalibrating, progress, pose) {
        if (isCalibrating) {
            this.elCalibration.classList.remove('hidden');
            this.elCalibrationFill.style.width = `${progress * 100}%`;

            if (this.webcam) this.webcam.drawFaceGuide(
                pose && pose.aligned ? 'rgba(127, 168, 140, 0.9)' : 'rgba(79, 70, 229, 0.6)'
            );

            if (this.elCalibInstr) {
                this.elCalibInstr.innerHTML = pose
                    ? this.renderPoseInstructions(pose)
                    : `<div class="calib-title">${Icons.render('target')} Calibrando…</div>`;
                this.elCalibInstr.classList.remove('hidden');
            }
        } else {
            this.elCalibration.classList.add('hidden');
            if (this.elCalibInstr) this.elCalibInstr.classList.add('hidden');
            if (this.webcam) this.webcam.clearOverlay();
        }
    }

    renderPoseInstructions(pose) {
        const held = Math.round(pose.pose_progress * 100);
        const steps = ['center', 'right', 'left', 'down', 'up'].map((name, i) => {
            const state = i < pose.pose_index ? 'done'
                        : i === pose.pose_index ? 'active' : 'pending';
            return `<span class="calib-step calib-step-${state}">${i + 1}</span>`;
        }).join('');

        // The subject cannot see the yaw/pitch numbers, so the only feedback
        // that matters is whether the pose currently counts. Without it they
        // hold a position that is silently being rejected.
        const status = pose.aligned
            ? `<span class="calib-ok">Posición correcta — mantén así (${held}%)</span>`
            : `<span class="calib-wait">Ajusta la posición para empezar a medir</span>`;

        return `
            <div class="calib-title">
                ${Icons.render('target')} Calibración guiada — paso ${pose.pose_index + 1} de ${pose.pose_total}
            </div>
            <div class="calib-steps">${steps}</div>
            <div class="calib-pose-label">${this.escapeHtml(pose.label)}</div>
            <div class="calib-pose-status">${status}</div>
            <div class="calib-note">
                Se mide tu rostro en reposo desde varios ángulos. Girar la cabeza
                acorta las distancias entre los puntos de la cara, así que sin
                estas medidas cualquier giro se confundiría con un cambio de
                expresión. Mantén el rostro <strong>neutral</strong> en cada paso.
            </div>
        `;
    }

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str == null ? '' : String(str);
        return div.innerHTML;
    }

    updatePeakEmotion(peak) {
        if (!this.elPeakEmotion) {
            const host = this.elEmotionName && this.elEmotionName.parentElement;
            if (!host) return;
            const el = document.createElement('div');
            el.className = 'peak-emotion hidden';
            el.id = 'peak-emotion';
            host.appendChild(el);
            this.elPeakEmotion = el;
        }

        if (!peak) return;

        // Held briefly on screen: the flash itself lasts a few frames, which
        // is too short to read.
        const config = CONFIG.EMOTIONS[peak.emotion] || CONFIG.EMOTIONS['neutral'];
        // Los picos marcados para revisión (sorpresa) se enuncian como
        // observación: dura demasiado poco para sostenerlo como estado, así
        // que lo único que se afirma es que hubo un cambio que mirar.
        const nota = peak.review && peak.note
            ? `<div class="peak-note">${peak.note}</div>`
            : '';
        this.elPeakEmotion.innerHTML =
            `<span class="peak-flash">${Icons.render('spark', {size: 14})}</span> destello: ` +
            `${config.icon} ${config.label}` +
            ` <span class="peak-conf">${Math.round(peak.confidence * 100)}%</span>` +
            nota;
        this.elPeakEmotion.style.borderColor = config.color;
        this.elPeakEmotion.classList.remove('hidden');

        if (this.peakTimeout) clearTimeout(this.peakTimeout);
        this.peakTimeout = setTimeout(() => {
            if (this.elPeakEmotion) this.elPeakEmotion.classList.add('hidden');
        }, peak.review ? 5000 : 2500);
    }

    updateEmotion(emotionKey, confidence, isCalibrating) {
        // Predictions made before the baseline exists have not had the subject's
        // resting facial morphology subtracted, so they carry whatever bias that
        // face has at rest. Showing them invites reading a real emotion into an
        // artifact of the calibration period.
        if (isCalibrating) {
            this.elEmotionIcon.innerHTML = Icons.render('clock');
            this.elEmotionName.textContent  = 'Calibrando...';
            this.elEmotionName.style.color  = 'var(--text-muted)';
            this.elConfidenceFill.style.width = '0%';
            this.elConfidenceText.textContent = '--';
            return;
        }

        const config      = CONFIG.EMOTIONS[emotionKey] || CONFIG.EMOTIONS['neutral'];
        const confPercent = Math.round(confidence * 100);

        this.elEmotionIcon.textContent = config.icon;
        this.elEmotionName.textContent = config.label;
        this.elEmotionName.style.color = config.color;

        this.elConfidenceFill.style.width           = `${confPercent}%`;
        this.elConfidenceFill.style.backgroundColor = config.color;
        this.elConfidenceText.textContent           = `${confPercent}%`;
    }

    updateHeartRate(hrData) {
        if (!hrData || !hrData.signal_ready) {
            this.elHrBpm.textContent = '--';
            // Once the estimator is producing spectra, a blank reading means the
            // pulse signal is too weak to trust rather than still warming up —
            // saying "Calibrando" forever reads as a malfunction.
            const estimating = hrData && hrData.bpm_confidence > 0;
            this.elHrStressText.textContent = estimating ? 'Señal débil' : 'Calibrando...';
            this.elHrStressText.style.color = 'var(--text-muted)';
            return;
        }

        this.elHrBpm.textContent = Math.round(hrData.bpm);

        const stressVal = hrData.stress_indicator;
        let stressText  = 'Bajo';
        let stressColor = 'var(--text-muted)';
        if (stressVal > 0.6) {
            stressText  = 'Alto';
            stressColor = 'var(--danger)';
        } else if (stressVal > 0.3) {
            stressText  = 'Moderado';
            stressColor = 'var(--warning)';
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
        // Only swap away from the live <video> feed when the user actually
        // asked for the pulse overlay. A previous version also swapped to a
        // server-round-tripped "preprocessed" frame whenever any arrived,
        // regardless of this toggle, replacing the smooth native camera
        // with a slow, choppy slideshow even with EVM off.
        const showEvm = this.elToggleEvm && this.elToggleEvm.checked && evmFrameB64;

        if (showEvm) {
            const src = evmFrameB64;
            if (this.elProcessedVideo) {
                this.elProcessedVideo.src             = src;
                this.elProcessedVideo.style.opacity       = '1';
                this.elProcessedVideo.style.pointerEvents = 'auto';
            }
            if (this.elWebcamVideo) {
                this.elWebcamVideo.style.opacity       = '0';
                this.elWebcamVideo.style.pointerEvents = 'none';
            }
        } else {
            if (this.elProcessedVideo) {
                this.elProcessedVideo.style.opacity       = '0';
                this.elProcessedVideo.style.pointerEvents = 'none';
            }
            if (this.elWebcamVideo) {
                this.elWebcamVideo.style.opacity       = '1';
                this.elWebcamVideo.style.pointerEvents = 'auto';
            }
        }
    }

    updateLiveScore(liveScore) {
        if (!this.elLiveScoreValue) return;

        // Absent while calibrating, and null until the window holds enough
        // samples — showing a number built on two frames would swing on noise.
        if (!liveScore || liveScore.score === null || liveScore.score === undefined) {
            this.elLiveScoreValue.textContent = '--';
            this.elLiveScoreValue.style.color = 'var(--text-main)';
            this.elLiveScoreFill.style.width = '0%';
            this.elLiveScoreLabel.textContent = i18n.t('live.liveScoreWaiting', 'Midiendo...');
            return;
        }

        const s = Math.round(liveScore.score);
        let color = 'var(--success)';
        let label = i18n.t('live.liveScoreGood', 'Va bien');

        if (s < 40) {
            color = 'var(--danger)';
            label = i18n.t('live.liveScoreLow', 'Señales negativas');
        } else if (s < 65) {
            color = 'var(--warning)';
            label = i18n.t('live.liveScoreMid', 'Mixto');
        }

        this.elLiveScoreValue.textContent = s;
        this.elLiveScoreValue.style.color = color;
        this.elLiveScoreFill.style.width = `${s}%`;
        this.elLiveScoreFill.style.backgroundColor = color;
        this.elLiveScoreLabel.textContent = label;
    }

    updateCongruence(score, breakdown) {
        if (score === 0 && (!breakdown || Object.keys(breakdown).length === 0)) {
            this.elCongruenceValue.textContent = '--';
            this.elCongruenceLabel.textContent = 'Calibrando...';
            this.charts.updateGauge(0, 'grey');
            return;
        }

        const s = Math.round(score);
        this.elCongruenceValue.textContent = s;

        let label = 'Alta coincidencia';
        let color = 'green';
        this.elCongruenceValue.style.color = 'var(--success)';

        if (s < 50) {
            label = 'Baja coincidencia';
            color = 'red';
            this.elCongruenceValue.style.color = 'var(--danger)';
        } else if (s < 80) {
            label = 'Moderada';
            color = 'yellow';
            this.elCongruenceValue.style.color = 'var(--warning)';
        }

        this.elCongruenceLabel.textContent = label;
        this.charts.updateGauge(s, color);

        if (breakdown) {
            this.elBreakdowns.stability.textContent = Math.round(breakdown.stability        || 0);
            this.elBreakdowns.alignment.textContent = Math.round(breakdown.micro_alignment  || 0);
            this.elBreakdowns.baseline.textContent  = Math.round(breakdown.baseline         || 0);
            this.elBreakdowns.physio.textContent    = Math.round(breakdown.physiological    || 0);
        }
    }

    addMicroExpression(micro) {
        this.microCount++;
        this.elMicroCount.textContent = `${this.microCount} Detectadas`;
        this.collectedMicroExpressions.push(micro);

        // Remove empty-state placeholder
        const emptyState = this.elMicroLog.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        const mins = Math.floor(micro.timestamp / 60).toString().padStart(2, '0');
        const secs = Math.floor(micro.timestamp % 60).toString().padStart(2, '0');

        const em = CONFIG.EMOTIONS[micro.detected_emotion] || CONFIG.EMOTIONS['neutral'];

        // Build item with DOM API (faster, XSS-safe, no innerHTML)
        const div = document.createElement('div');
        div.className = `micro-item${micro.is_contradictory ? ' contradictory' : ''}`;

        const timeSpan = document.createElement('span');
        timeSpan.className   = 'micro-time';
        timeSpan.textContent = `${mins}:${secs}`;

        const descSpan = document.createElement('span');
        descSpan.className = 'micro-desc';
        const b = document.createElement('b');
        b.textContent = em.label;
        descSpan.append(b, ' microexpresión');

        const scoreSpan = document.createElement('span');
        scoreSpan.className   = 'micro-score';
        scoreSpan.textContent = `Puntaje: ${micro.relevance_score}`;

        div.append(timeSpan, descSpan, scoreSpan);
        this.elMicroLog.prepend(div);

        // Keep only last 50 items to avoid unbounded DOM growth
        if (this.elMicroLog.children.length > 50) {
            this.elMicroLog.removeChild(this.elMicroLog.lastChild);
        }
    }
}
