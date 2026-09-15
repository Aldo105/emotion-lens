class SessionManager {
    constructor() {
        this.container = document.getElementById('sessions-grid');
        this.sessions = [];
        this.loaded = false;
    }

    async loadSessions() {
        this.container.innerHTML = `
            <div class="sessions-loading">
                <div class="spinner"></div>
                <span>Cargando sesiones...</span>
            </div>
        `;

        try {
            const res = await fetch(`${CONFIG.API_URL}/sessions/`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            this.sessions = data.sessions || data || [];
            this.renderSessions();
            this.loaded = true;
        } catch (err) {
            console.error('Failed to load sessions:', err);
            this.container.innerHTML = `
                <div class="sessions-empty-state">
                    <span class="empty-icon">📂</span>
                    <h3>No se pudieron cargar las sesiones</h3>
                    <p>${err.message}</p>
                    <button class="btn btn-secondary" onclick="sessionManager.loadSessions()">Reintentar</button>
                </div>
            `;
        }
    }

    renderSessions() {
        if (!this.sessions.length) {
            this.container.innerHTML = `
                <div class="sessions-empty-state">
                    <span class="empty-icon">📂</span>
                    <h3>Aún no hay sesiones</h3>
                    <p>Inicia un análisis en vivo o sube un video para crear tu primera sesión.</p>
                </div>
            `;
            return;
        }

        this.container.innerHTML = this.sessions.map(session => this.renderCard(session)).join('');
    }

    renderCard(session) {
        const date = new Date(session.created_at || session.start_time || Date.now());
        const dateStr = date.toLocaleDateString('es-ES', {
            month: 'short', day: 'numeric', year: 'numeric'
        });
        const timeStr = date.toLocaleTimeString('es-ES', {
            hour: '2-digit', minute: '2-digit'
        });

        const statusMap = {
            'active': { class: 'status-active', label: 'Activa', icon: '🔴' },
            'completed': { class: 'status-completed', label: 'Completada', icon: '✅' },
            'cancelled': { class: 'status-cancelled', label: 'Cancelada', icon: '❌' },
            'processing': { class: 'status-active', label: 'Procesando', icon: '⏳' }
        };
        const status = statusMap[session.status] || statusMap['completed'];

        const dominantEmotion = session.dominant_emotion || session.summary?.dominant_emotion || 'neutral';
        const emotionConfig = CONFIG.EMOTIONS[dominantEmotion] || CONFIG.EMOTIONS['neutral'];

        const congruence = session.congruence_score ?? session.summary?.congruence_score ?? '--';
        const congruenceDisplay = typeof congruence === 'number' ? Math.round(congruence) : congruence;

        const candidateName = session.candidate_name || session.name || 'Candidato desconocido';
        const duration = session.duration || session.summary?.duration || null;
        const durationStr = duration ? this.formatDuration(duration) : '--:--';

        return `
            <div class="session-card glass-panel" data-session-id="${session.id}">
                <div class="session-card-header">
                    <div class="session-card-meta">
                        <h3 class="session-card-name">${this.escapeHtml(candidateName)}</h3>
                        <span class="session-card-date">${dateStr} · ${timeStr}</span>
                    </div>
                    <span class="session-status-badge ${status.class}">${status.icon} ${status.label}</span>
                </div>

                <div class="session-card-stats">
                    <div class="session-stat">
                        <span class="session-stat-label">Emoción dominante</span>
                        <span class="session-stat-value">${emotionConfig.icon} ${emotionConfig.label}</span>
                    </div>
                    <div class="session-stat">
                        <span class="session-stat-label">Congruencia</span>
                        <span class="session-stat-value session-congruence">${congruenceDisplay}%</span>
                    </div>
                    <div class="session-stat">
                        <span class="session-stat-label">Duración</span>
                        <span class="session-stat-value">${durationStr}</span>
                    </div>
                </div>

                <div class="session-card-actions">
                    <button class="btn btn-primary btn-sm" onclick="sessionManager.viewSession('${session.id}')">
                        Ver reporte
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="sessionManager.downloadPDF('${session.id}')">
                        📄 PDF
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="sessionManager.downloadCSV('${session.id}')">
                        📊 CSV
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="sessionManager.downloadEVM('${session.id}')" title="Descargar Video EVM">
                        🎥 EVM
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick="sessionManager.downloadMicroHighlights('${session.id}')" title="Descargar video de validación de microexpresiones">
                        🔬 Microexpresiones
                    </button>
                    <button class="btn btn-danger btn-sm" onclick="sessionManager.deleteSession('${session.id}')">
                        🗑️
                    </button>
                </div>
            </div>
        `;
    }

    async viewSession(id) {
        try {
            const res = await fetch(`${CONFIG.API_URL}/reports/${id}/data`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const report = await res.json();
            this.showDetailModal(report, id);
            this.renderAnalysis(report.interview_analysis);
            this.renderTasksAndEvents(report.interview_analysis);
            this.renderValidation(report.feedback);
        } catch (err) {
            console.error('Failed to load session report:', err);
            this.showToast('Error al cargar el reporte: ' + err.message, 'error');
        }
    }

    showDetailModal(report, sessionId) {
        // Remove any existing modal
        const existing = document.getElementById('session-detail-modal');
        if (existing) existing.remove();

        const emotionBars = Object.entries(report.emotion_distribution || {})
            .sort(([, a], [, b]) => b - a)
            .map(([emotion, value]) => {
                const cfg = CONFIG.EMOTIONS[emotion] || CONFIG.EMOTIONS['neutral'];
                const pct = Math.round(value * 100);
                return `
                    <div class="detail-emotion-row">
                        <span class="detail-emotion-label">${cfg.icon} ${cfg.label}</span>
                        <div class="detail-emotion-bar-track">
                            <div class="detail-emotion-bar-fill" style="width: ${pct}%; background: ${cfg.color}"></div>
                        </div>
                        <span class="detail-emotion-pct">${pct}%</span>
                    </div>
                `;
            }).join('');

        const congruence = report.congruence_score != null ? Math.round(report.congruence_score) : '--';
        const microCount = report.micro_expression_count ?? report.micro_expressions?.length ?? 0;

        const modal = document.createElement('div');
        modal.id = 'session-detail-modal';
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content glass-panel">
                <div class="modal-header">
                    <h2>📋 Reporte de Sesión</h2>
                    <button class="icon-btn modal-close" onclick="sessionManager.closeModal()">✕</button>
                </div>

                <div class="modal-body">
                    <div class="modal-section">
                        <h3>Candidato</h3>
                        <p class="modal-candidate-name">${this.escapeHtml(report.candidate_name || 'Desconocido')}</p>
                    </div>

                    <div class="modal-stats-row">
                        <div class="modal-stat-card">
                            <span class="modal-stat-value">${congruence}%</span>
                            <span class="modal-stat-label">Congruencia</span>
                        </div>
                        <div class="modal-stat-card">
                            <span class="modal-stat-value">${microCount}</span>
                            <span class="modal-stat-label">Microexpresiones</span>
                        </div>
                        <div class="modal-stat-card">
                            <span class="modal-stat-value">${report.duration ? this.formatDuration(report.duration) : '--'}</span>
                            <span class="modal-stat-label">Duración</span>
                        </div>
                    </div>

                    <div class="modal-section">
                        <h3>Distribución de Emociones</h3>
                        <div class="detail-emotion-bars">${emotionBars || '<p class="empty-state">Sin datos</p>'}</div>
                    </div>

                    <!-- Interview Analysis Section -->
                    <div id="detail-analysis-section" style="display: none;">
                        <h3 class="detail-section-title">🧠 Análisis de Entrevista</h3>
                        
                        <!-- Dimension Scores Radar -->
                        <div class="analysis-dimensions" id="detail-dimensions">
                            <!-- Filled dynamically -->
                        </div>
                        
                        <!-- Overall Score -->
                        <div class="analysis-overall" id="detail-overall-score"></div>
                        
                        <!-- Behavioral Patterns -->
                        <div class="analysis-patterns" id="detail-patterns"></div>
                        
                        <!-- Red Flags -->
                        <div class="analysis-flags" id="detail-red-flags"></div>
                        
                        <!-- Recommendations -->
                        <div class="analysis-recommendations" id="detail-recommendations"></div>
                    </div>

                    <!-- Task Friction / Event Timeline -->
                    <div id="detail-tasks-section" style="display: none;">
                        <h3 class="detail-section-title">🎯 Tareas y Eventos</h3>
                        <div id="detail-tasks"></div>
                        <div id="detail-events"></div>
                    </div>

                    <!-- Human Validation Section -->
                    <div id="detail-validation-section" style="display: none;">
                        <h3 class="detail-section-title">🔍 Validación Humana</h3>
                        <div id="detail-validation"></div>
                    </div>
                </div>

                <div class="modal-footer">
                    <button class="btn btn-primary" onclick="sessionManager.downloadPDF('${sessionId}')">📄 Descargar PDF</button>
                    <button class="btn btn-secondary" onclick="sessionManager.downloadCSV('${sessionId}')">📊 Descargar CSV</button>
                    <button class="btn btn-secondary" onclick="sessionManager.downloadEVM('${sessionId}')">🎥 Video EVM</button>
                    <button class="btn btn-secondary" onclick="sessionManager.downloadMicroHighlights('${sessionId}')">🔬 Video Microexpresiones</button>
                    <button class="btn btn-secondary" onclick="sessionManager.closeModal()">Cerrar</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        // Trigger entrance animation
        requestAnimationFrame(() => modal.classList.add('active'));

        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) this.closeModal();
        });
    }

    renderTasksAndEvents(analysis) {
        const section = document.getElementById('detail-tasks-section');
        const tasksEl = document.getElementById('detail-tasks');
        const eventsEl = document.getElementById('detail-events');
        if (!section || !tasksEl || !eventsEl) return;

        const tasks = (analysis && analysis.task_analysis) || [];
        const events = (analysis && analysis.event_timeline) || [];

        if (tasks.length === 0 && events.length === 0) {
            section.style.display = 'none';
            return;
        }
        section.style.display = 'block';

        // Tasks ranked by friction — the highest one is what to redesign first.
        if (tasks.length > 0) {
            const ranked = [...tasks].sort((a, b) => b.friction_score - a.friction_score);
            tasksEl.innerHTML = '<h4 style="margin:12px 0 8px;">Fricción por tarea</h4>' +
                ranked.map(t => {
                    const color = t.friction_score >= 60 ? '#ef4444'
                                : t.friction_score >= 35 ? '#f59e0b' : '#10b981';
                    return `
                    <div class="pattern-item" style="border-left:3px solid ${color};">
                        <div style="display:flex; justify-content:space-between; align-items:baseline;">
                            <strong>${this._escape(t.task_name)}</strong>
                            <span style="color:${color}"><b>${Math.round(t.friction_score)}</b>/100</span>
                        </div>
                        <div style="font-size:0.82rem; color:var(--text-secondary); margin-top:4px;">
                            ${t.start_video_time}–${t.end_video_time} · ${Math.round(t.duration_seconds)}s
                            ${t.completed ? '' : ' · <b>sin completar</b>'}
                            · ${t.error_count} error(es) · ${t.confusion_count} confusión(es)
                        </div>
                        <div style="font-size:0.82rem; color:var(--text-secondary);">
                            Pico de nerviosismo ${t.peak_nervousness} en ${t.peak_nervousness_time}
                        </div>
                        <div style="font-size:0.85rem; margin-top:4px;">${this._escape(t.interpretation)}</div>
                    </div>`;
                }).join('');
        } else {
            tasksEl.innerHTML = '';
        }

        // Only friction points are worth a reviewer's time here.
        const friction = events.filter(e => e.is_friction_point);
        if (friction.length > 0) {
            eventsEl.innerHTML = '<h4 style="margin:16px 0 8px;">Puntos de fricción marcados</h4>' +
                friction.map(e => `
                    <div class="pattern-item medium">
                        <div style="display:flex; justify-content:space-between;">
                            <strong>${this._escape(e.label)}</strong>
                            <span style="opacity:0.75;">⏱ ${e.video_time}</span>
                        </div>
                        <div style="font-size:0.85rem; margin-top:4px;">${this._escape(e.content)}</div>
                        <div style="font-size:0.82rem; color:var(--text-secondary); margin-top:4px;">
                            ${e.emotion_before} → ${e.emotion_after}
                            · nerviosismo ${e.nervousness_change >= 0 ? '+' : ''}${e.nervousness_change}
                            · congruencia ${e.congruence_change >= 0 ? '+' : ''}${e.congruence_change}
                        </div>
                        <div style="font-size:0.82rem; margin-top:4px;">${this._escape(e.interpretation)}</div>
                    </div>`).join('');
        } else {
            eventsEl.innerHTML = '';
        }
    }

    _escape(text) {
        const div = document.createElement('div');
        div.textContent = text == null ? '' : String(text);
        return div.innerHTML;
    }

    renderValidation(feedback) {
        const section = document.getElementById('detail-validation-section');
        const container = document.getElementById('detail-validation');
        if (!section || !container) return;

        const metrics = feedback && feedback.validation_metrics;
        if (!metrics || metrics.agreement_rate === null || metrics.agreement_rate === undefined) {
            section.style.display = 'none';
            return;
        }

        section.style.display = 'block';

        const pct = Math.round(metrics.agreement_rate * 100);
        const color = pct >= 75 ? '#10b981' : pct >= 50 ? '#f59e0b' : '#ef4444';
        const decided = metrics.confirmed + metrics.rejected;

        const bandLabels = {
            high: 'Alta (80-100)',
            medium: 'Media (60-79)',
            low: 'Baja (<60)',
            unknown: 'Sin registrar',
        };
        const bandRows = Object.entries(metrics.by_relevance_band || {})
            .filter(([, b]) => b.agreement_rate !== null)
            .map(([key, b]) => `
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; padding:4px 0;">
                    <span>${bandLabels[key] || key}</span>
                    <span><b>${Math.round(b.agreement_rate * 100)}%</b>
                        <span style="color:var(--text-secondary)">(${b.confirmed}/${b.confirmed + b.rejected})</span>
                    </span>
                </div>`)
            .join('');

        container.innerHTML = `
            <div style="text-align:center; margin-bottom:12px;">
                <div class="score-big" style="color:${color}">${pct}%</div>
                <div style="color:var(--text-secondary); font-size:0.85rem;">
                    Concordancia con revisión humana (${metrics.confirmed} de ${decided} confirmadas)
                </div>
            </div>
            ${bandRows ? `<h4 style="margin:12px 0 4px; font-size:0.9rem;">Por relevancia de la detección</h4>${bandRows}` : ''}
            <p style="font-size:0.78rem; color:var(--text-muted); margin-top:10px;">
                Momentos marcados como "no estoy seguro" (${metrics.unsure}) o sin revisar
                (${metrics.unanswered}) se excluyen del porcentaje.
            </p>
        `;
    }

    /** Backend pattern/flag "type" and recommendation "category" slugs -> Spanish label. */
    static TYPE_LABELS = {
        rapid_confusion: 'Confusión Rápida',
        nervous_spiral: 'Espiral de Nerviosismo',
        social_masking: 'Enmascaramiento Social',
        stress_recovery: 'Recuperación de Estrés',
        emotional_flatline: 'Aplanamiento Emocional',
        confidence_decline: 'Baja de Confianza',
        authentic_engagement: 'Compromiso Auténtico',
        emotion_transition: 'Transición Emocional',
        technical: 'Técnico',
        behavioral: 'Conductual',
        communication: 'Comunicación',
        resilience: 'Resiliencia',
        confidence: 'Confianza',
        usability: 'Usabilidad',
    };

    _typeLabel(slug) {
        if (!slug) return '';
        return SessionManager.TYPE_LABELS[slug] || slug.replace(/_/g, ' ');
    }

    renderAnalysis(analysis) {
        const section = document.getElementById('detail-analysis-section');
        if (!analysis || !analysis.dimension_scores) {
            if (section) section.style.display = 'none';
            return;
        }
        section.style.display = 'block';
        
        // Render dimension scores
        const dims = document.getElementById('detail-dimensions');
        const scores = analysis.dimension_scores;
        const dimConfig = CONFIG.ANALYSIS_DIMENSIONS || {};
        
        dims.innerHTML = Object.entries(scores)
            .filter(([key]) => key !== 'overall')
            .map(([key, score]) => {
                const cfg = dimConfig[key] || { label: key, color: '#888', icon: '📊' };
                const color = score >= 70 ? '#10b981' : score >= 40 ? '#f59e0b' : '#ef4444';
                return `
                    <div class="dimension-card">
                        <div style="font-size: 1.2rem">${cfg.icon}</div>
                        <div class="dimension-score" style="color: ${color}">${Math.round(score)}</div>
                        <div class="dimension-label">${cfg.label}</div>
                        <div class="dimension-bar">
                            <div class="dimension-bar-fill" style="width: ${score}%; background: ${cfg.color}"></div>
                        </div>
                    </div>
                `;
            }).join('');
        
        // Overall score
        const overallEl = document.getElementById('detail-overall-score');
        const overall = scores.overall || 0;
        const overallColor = overall >= 70 ? '#10b981' : overall >= 40 ? '#f59e0b' : '#ef4444';
        overallEl.innerHTML = `
            <div class="score-big" style="color: ${overallColor}">${Math.round(overall)}/100</div>
            <div style="color: var(--text-secondary); margin-top: 4px;">Puntaje General de la Entrevista</div>
        `;

        // Behavioral patterns
        const patternsEl = document.getElementById('detail-patterns');
        const patterns = analysis.behavioral_patterns || [];
        if (patterns.length > 0) {
            patternsEl.innerHTML = '<h4 style="margin: 12px 0 8px;">Patrones de Comportamiento</h4>' +
                patterns.map(p => {
                    const isPositive = p.is_positive || ['stress_recovery', 'authentic_engagement'].includes(p.type);
                    const cls = isPositive ? 'positive' : (p.severity === 'medium' ? 'medium' : 'negative');
                    const timeBadge = (p.timestamp !== null && p.timestamp !== undefined)
                        ? `<span style="float:right; opacity:0.75;">⏱ ${this.formatDuration(p.timestamp)}</span>` : '';
                    return `<div class="pattern-item ${cls}">
                        <strong>${isPositive ? '✅' : '⚠️'} ${this._typeLabel(p.type).toUpperCase()}</strong>${timeBadge}
                        <div style="font-size: 0.85rem; margin-top: 4px;">${p.description}</div>
                        ${p.context_question ? `<div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">Contexto: ${p.context_question}</div>` : ''}
                    </div>`;
                }).join('');
        } else {
            patternsEl.innerHTML = '';
        }

        // Red flags
        const flagsEl = document.getElementById('detail-red-flags');
        const flags = analysis.red_flags || [];
        if (flags.length > 0) {
            flagsEl.innerHTML = '<h4 style="margin: 12px 0 8px;">🔴 Señales de Alerta</h4>' +
                flags.map(f => `<div class="flag-item ${f.severity}">
                    <strong>${this._typeLabel(f.type)}</strong>
                    <div style="font-size: 0.85rem; margin-top: 4px;">${f.evidence}</div>
                </div>`).join('');
        } else {
            flagsEl.innerHTML = '';
        }

        // Recommendations
        const recsEl = document.getElementById('detail-recommendations');
        const recs = analysis.recommendations || [];
        if (recs.length > 0) {
            recsEl.innerHTML = '<h4 style="margin: 12px 0 8px;">📋 Recomendaciones</h4>' +
                recs.map(r => `<div class="recommendation-item ${r.priority}">
                    <span style="text-transform: uppercase; font-size: 0.7rem; font-weight: 700; color: var(--text-secondary);">${this._typeLabel(r.category)}</span>
                    <div style="margin-top: 4px;">${r.text}</div>
                </div>`).join('');
        } else {
            recsEl.innerHTML = '';
        }
    }

    closeModal() {
        const modal = document.getElementById('session-detail-modal');
        if (modal) {
            modal.classList.remove('active');
            setTimeout(() => modal.remove(), 300);
        }
    }

    async deleteSession(id) {
        if (!confirm('¿Seguro que quieres eliminar esta sesión? Esta acción no se puede deshacer.')) return;

        try {
            const res = await fetch(`${CONFIG.API_URL}/sessions/${id}`, { method: 'DELETE' });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            // Remove card from DOM with animation
            const card = this.container.querySelector(`[data-session-id="${id}"]`);
            if (card) {
                card.style.transition = 'all 0.3s ease';
                card.style.opacity = '0';
                card.style.transform = 'scale(0.9)';
                setTimeout(() => {
                    card.remove();
                    this.sessions = this.sessions.filter(s => s.id !== id);
                    if (!this.sessions.length) this.renderSessions();
                }, 300);
            }

            this.showToast('Sesión eliminada correctamente', 'success');
        } catch (err) {
            console.error('Failed to delete session:', err);
            this.showToast('Error al eliminar la sesión: ' + err.message, 'error');
        }
    }

    downloadPDF(id) {
        window.open(`${CONFIG.API_URL}/reports/${id}/pdf`, '_blank');
    }

    downloadCSV(id) {
        window.open(`${CONFIG.API_URL}/reports/${id}/csv`, '_blank');
    }

    async downloadEVM(id) {
        try {
            this.showToast('Verificando disponibilidad del video EVM...', 'info');
            const res = await fetch(`${CONFIG.API_URL}/reports/${id}/evm-status`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            if (data.status === 'ready') {
                const sizeInfo = data.size_mb ? ` (${data.size_mb} MB)` : '';
                this.showToast(`Iniciando descarga de video EVM${sizeInfo}...`, 'success');
                window.open(`${CONFIG.API_URL}/reports/${id}/evm-video`, '_blank');
            } else if (data.status === 'rendering') {
                const pct = Math.round((data.progress || 0) * 100);
                const phase = data.phase ? ` [${data.phase}]` : '';
                this.showToast(`El video EVM se está procesando: ${pct}%${phase}. Intenta en unos momentos.`, 'info');
            } else if (data.status === 'failed') {
                this.showToast('El procesamiento del video EVM falló para esta sesión.', 'error');
            } else {
                this.showToast('No hay grabación de video disponible para esta sesión.', 'warning');
            }
        } catch (err) {
            console.error('Error al solicitar video EVM:', err);
            this.showToast('Error al verificar video EVM: ' + err.message, 'error');
        }
    }

    async downloadMicroHighlights(id) {
        try {
            this.showToast('Verificando disponibilidad del video de microexpresiones...', 'info');
            const res = await fetch(`${CONFIG.API_URL}/reports/${id}/micro-highlights-status`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            if (data.status === 'ready') {
                const sizeInfo = data.size_mb ? ` (${data.size_mb} MB)` : '';
                this.showToast(`Iniciando descarga de video de microexpresiones${sizeInfo}...`, 'success');
                window.open(`${CONFIG.API_URL}/reports/${id}/micro-highlights-video`, '_blank');
            } else if (data.status === 'rendering') {
                const pct = Math.round((data.progress || 0) * 100);
                const phase = data.phase ? ` [${data.phase}]` : '';
                this.showToast(`El video de microexpresiones se está procesando: ${pct}%${phase}. Intenta en unos momentos.`, 'info');
            } else if (data.status === 'failed') {
                this.showToast('El procesamiento del video de microexpresiones falló para esta sesión.', 'error');
            } else {
                this.showToast('No se detectaron microexpresiones en esta sesión, o el video aún no está disponible.', 'warning');
            }
        } catch (err) {
            console.error('Error al solicitar video de microexpresiones:', err);
            this.showToast('Error al verificar video de microexpresiones: ' + err.message, 'error');
        }
    }

    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);
        requestAnimationFrame(() => toast.classList.add('active'));
        setTimeout(() => {
            toast.classList.remove('active');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}
