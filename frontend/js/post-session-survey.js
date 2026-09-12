/**
 * EmotionLens — Post-Session Survey
 * 
 * Optional survey shown to the SUBJECT after session ends.
 * Collects self-reported accuracy, moment validations, and emotional state.
 * Data is used to measure and calibrate system precision.
 */

class PostSessionSurvey {
    constructor() {
        this.overlay = document.getElementById('survey-modal');
        this.sessionId = null;
        this.results = {
            accuracyRating: 0.5,
            selfReportedEmotion: null,
            attemptedSuppression: false,
            momentValidations: [],
        };
    }

    /**
     * Show the survey modal with session data.
     * @param {Object} sessionData - Contains session_id, micro_expressions, dominant_emotion, etc.
     */
    show(sessionData) {
        if (!this.overlay) return;
        this.sessionId = sessionData.session_id || null;

        const card = this.overlay.querySelector('.survey-card');
        if (!card) return;

        // Build the 3 sections
        card.innerHTML = `
            <h2>📋 Encuesta Post-Sesion</h2>
            <p class="survey-subtitle">Tu feedback ayuda a mejorar la precision del sistema. Es opcional y anonimo.</p>

            <!-- Section 1: Accuracy Slider -->
            <div class="survey-section">
                <h3>1. Precision General</h3>
                <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">
                    ¿Que tan preciso fue el software en detectar tus emociones?
                </p>
                <div class="survey-slider-container">
                    <span style="font-size: 0.75rem; color: var(--text-muted);">😞 Nada</span>
                    <input type="range" class="survey-slider" id="survey-accuracy" min="0" max="100" value="50">
                    <span style="font-size: 0.75rem; color: var(--text-muted);">Muy 😊</span>
                    <span class="survey-slider-value" id="survey-accuracy-val">50%</span>
                </div>
            </div>

            <!-- Section 2: Moment Validations -->
            <div class="survey-section" id="survey-moments-section">
                <h3>2. Momentos Clave (opcional)</h3>
                <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">
                    El software detecto estos momentos. ¿Son correctos?
                </p>
                <div id="survey-moments-container"></div>
            </div>

            <!-- Section 3: Self-Report -->
            <div class="survey-section">
                <h3>3. Tu Estado Emocional Real</h3>
                <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">
                    ¿Cual fue tu estado emocional predominante durante la sesion?
                </p>
                <div class="survey-emotion-selector" id="survey-emotion-selector">
                    <span class="survey-emotion-pill" data-emotion="neutral">😐 Tranquilo</span>
                    <span class="survey-emotion-pill" data-emotion="nervousness">😰 Nervioso</span>
                    <span class="survey-emotion-pill" data-emotion="happy">😊 Contento</span>
                    <span class="survey-emotion-pill" data-emotion="angry">😠 Molesto</span>
                    <span class="survey-emotion-pill" data-emotion="fear">😨 Ansioso</span>
                    <span class="survey-emotion-pill" data-emotion="sad">😢 Triste</span>
                    <span class="survey-emotion-pill" data-emotion="confidence">😎 Seguro</span>
                </div>

                <div class="survey-toggle-row" style="margin-top: 16px;">
                    <label>¿Intentaste ocultar alguna emocion?</label>
                    <label class="toggle-switch" style="margin-left: 12px;">
                        <input type="checkbox" id="survey-suppression">
                        <span class="slider"></span>
                    </label>
                </div>
            </div>

            <!-- Actions -->
            <div class="survey-actions">
                <button class="btn-survey-skip" id="survey-skip-btn">Saltar encuesta</button>
                <button class="btn-survey-submit" id="survey-submit-btn">Enviar y ver reporte</button>
            </div>
        `;

        // Populate moment validation cards
        this._buildMomentCards(sessionData.micro_expressions || []);

        // Wire up events
        this._wireEvents();

        // Show
        this.overlay.classList.remove('hidden');
    }

    _wireEvents() {
        // Slider
        const slider = document.getElementById('survey-accuracy');
        const valDisplay = document.getElementById('survey-accuracy-val');
        if (slider) {
            slider.addEventListener('input', () => {
                valDisplay.textContent = slider.value + '%';
                this.results.accuracyRating = parseInt(slider.value) / 100;
            });
        }

        // Emotion pills
        const pills = document.querySelectorAll('.survey-emotion-pill');
        pills.forEach(pill => {
            pill.addEventListener('click', () => {
                pills.forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                this.results.selfReportedEmotion = pill.dataset.emotion;
            });
        });

        // Suppression toggle
        const suppression = document.getElementById('survey-suppression');
        if (suppression) {
            suppression.addEventListener('change', () => {
                this.results.attemptedSuppression = suppression.checked;
            });
        }

        // Submit
        const submitBtn = document.getElementById('survey-submit-btn');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this.submit());
        }

        // Skip
        const skipBtn = document.getElementById('survey-skip-btn');
        if (skipBtn) {
            skipBtn.addEventListener('click', () => this.skip());
        }
    }

    _buildMomentCards(microExpressions) {
        const container = document.getElementById('survey-moments-container');
        const section = document.getElementById('survey-moments-section');
        if (!container || !section) return;

        if (microExpressions.length === 0) {
            section.style.display = 'none';
            return;
        }

        // Show max 8 most relevant moments
        const moments = microExpressions
            .sort((a, b) => (b.relevance_score || 0) - (a.relevance_score || 0))
            .slice(0, 8);

        container.innerHTML = '';
        moments.forEach((m, i) => {
            const timeStr = this._formatTime(m.timestamp || 0);
            const card = document.createElement('div');
            card.className = 'survey-moment-card';
            card.innerHTML = `
                <div class="moment-info">
                    <span class="moment-time">${timeStr}</span>
                    ${m.emotion || 'Microexpresion'} ${m.contradictory ? '(contradictoria)' : ''}
                </div>
                <div class="survey-moment-btns">
                    <button data-idx="${i}" data-verdict="correct">✅</button>
                    <button data-idx="${i}" data-verdict="incorrect">❌</button>
                    <button data-idx="${i}" data-verdict="unsure">🤷</button>
                </div>
            `;
            container.appendChild(card);

            // Init validation result
            this.results.momentValidations.push({
                timestamp: m.timestamp,
                emotion: m.emotion,
                verdict: null,
            });
        });

        // Wire moment buttons
        container.querySelectorAll('.survey-moment-btns button').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = parseInt(btn.dataset.idx);
                const verdict = btn.dataset.verdict;
                this.results.momentValidations[idx].verdict = verdict;

                // Visual feedback
                const row = btn.parentElement;
                row.querySelectorAll('button').forEach(b => {
                    b.className = '';
                });
                btn.classList.add(`selected-${verdict}`);
            });
        });
    }

    async submit() {
        if (this.sessionId) {
            try {
                const resp = await fetch(`${CONFIG.API_URL}/feedback/${this.sessionId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        overall_accuracy_rating: this.results.accuracyRating,
                        self_reported_emotion: this.results.selfReportedEmotion,
                        attempted_suppression: this.results.attemptedSuppression,
                        moment_validations: this.results.momentValidations,
                    }),
                });
                if (resp.ok) {
                    console.log('[Survey] Feedback submitted successfully');
                }
            } catch (err) {
                console.warn('[Survey] Failed to submit feedback:', err);
            }
        }
        this.hide();
    }

    skip() {
        this.hide();
    }

    hide() {
        if (this.overlay) {
            this.overlay.classList.add('hidden');
        }
    }

    _formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
}
