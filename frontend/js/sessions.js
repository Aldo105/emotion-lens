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
                <span>Loading sessions...</span>
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
                    <h3>Could not load sessions</h3>
                    <p>${err.message}</p>
                    <button class="btn btn-secondary" onclick="sessionManager.loadSessions()">Retry</button>
                </div>
            `;
        }
    }

    renderSessions() {
        if (!this.sessions.length) {
            this.container.innerHTML = `
                <div class="sessions-empty-state">
                    <span class="empty-icon">📂</span>
                    <h3>No sessions yet</h3>
                    <p>Start a live analysis or upload a video to create your first session.</p>
                </div>
            `;
            return;
        }

        this.container.innerHTML = this.sessions.map(session => this.renderCard(session)).join('');
    }

    renderCard(session) {
        const date = new Date(session.created_at || session.start_time || Date.now());
        const dateStr = date.toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', year: 'numeric'
        });
        const timeStr = date.toLocaleTimeString('en-US', {
            hour: '2-digit', minute: '2-digit'
        });

        const statusMap = {
            'active': { class: 'status-active', label: 'Active', icon: '🔴' },
            'completed': { class: 'status-completed', label: 'Completed', icon: '✅' },
            'cancelled': { class: 'status-cancelled', label: 'Cancelled', icon: '❌' },
            'processing': { class: 'status-active', label: 'Processing', icon: '⏳' }
        };
        const status = statusMap[session.status] || statusMap['completed'];

        const dominantEmotion = session.dominant_emotion || session.summary?.dominant_emotion || 'neutral';
        const emotionConfig = CONFIG.EMOTIONS[dominantEmotion] || CONFIG.EMOTIONS['neutral'];

        const congruence = session.congruence_score ?? session.summary?.congruence_score ?? '--';
        const congruenceDisplay = typeof congruence === 'number' ? Math.round(congruence) : congruence;

        const candidateName = session.candidate_name || session.name || 'Unknown Candidate';
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
                        <span class="session-stat-label">Dominant Emotion</span>
                        <span class="session-stat-value">${emotionConfig.icon} ${emotionConfig.label}</span>
                    </div>
                    <div class="session-stat">
                        <span class="session-stat-label">Congruence</span>
                        <span class="session-stat-value session-congruence">${congruenceDisplay}%</span>
                    </div>
                    <div class="session-stat">
                        <span class="session-stat-label">Duration</span>
                        <span class="session-stat-value">${durationStr}</span>
                    </div>
                </div>

                <div class="session-card-actions">
                    <button class="btn btn-primary btn-sm" onclick="sessionManager.viewSession('${session.id}')">
                        View Report
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
        } catch (err) {
            console.error('Failed to load session report:', err);
            this.showToast('Failed to load report: ' + err.message, 'error');
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
                    <h2>📋 Session Report</h2>
                    <button class="icon-btn modal-close" onclick="sessionManager.closeModal()">✕</button>
                </div>

                <div class="modal-body">
                    <div class="modal-section">
                        <h3>Candidate</h3>
                        <p class="modal-candidate-name">${this.escapeHtml(report.candidate_name || 'Unknown')}</p>
                    </div>

                    <div class="modal-stats-row">
                        <div class="modal-stat-card">
                            <span class="modal-stat-value">${congruence}%</span>
                            <span class="modal-stat-label">Congruence</span>
                        </div>
                        <div class="modal-stat-card">
                            <span class="modal-stat-value">${microCount}</span>
                            <span class="modal-stat-label">Micro-Expressions</span>
                        </div>
                        <div class="modal-stat-card">
                            <span class="modal-stat-value">${report.duration ? this.formatDuration(report.duration) : '--'}</span>
                            <span class="modal-stat-label">Duration</span>
                        </div>
                    </div>

                    <div class="modal-section">
                        <h3>Emotion Distribution</h3>
                        <div class="detail-emotion-bars">${emotionBars || '<p class="empty-state">No data</p>'}</div>
                    </div>
                    
                    <!-- Interview Analysis Section -->
                    <div id="detail-analysis-section" style="display: none;">
                        <h3 class="detail-section-title">🧠 Interview Analysis</h3>
                        
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
                </div>

                <div class="modal-footer">
                    <button class="btn btn-primary" onclick="sessionManager.downloadPDF('${sessionId}')">📄 Download PDF</button>
                    <button class="btn btn-secondary" onclick="sessionManager.downloadCSV('${sessionId}')">📊 Download CSV</button>
                    <button class="btn btn-secondary" onclick="sessionManager.downloadEVM('${sessionId}')">🎥 Video EVM</button>
                    <button class="btn btn-secondary" onclick="sessionManager.closeModal()">Close</button>
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
            <div style="color: var(--text-secondary); margin-top: 4px;">Overall Interview Score</div>
        `;
        
        // Behavioral patterns
        const patternsEl = document.getElementById('detail-patterns');
        const patterns = analysis.behavioral_patterns || [];
        if (patterns.length > 0) {
            patternsEl.innerHTML = '<h4 style="margin: 12px 0 8px;">Behavioral Patterns</h4>' +
                patterns.map(p => {
                    const isPositive = ['stress_recovery', 'authentic_engagement'].includes(p.type);
                    const cls = isPositive ? 'positive' : (p.severity === 'medium' ? 'medium' : 'negative');
                    return `<div class="pattern-item ${cls}">
                        <strong>${isPositive ? '✅' : '⚠️'} ${p.type.replace(/_/g, ' ').toUpperCase()}</strong>
                        <div style="font-size: 0.85rem; margin-top: 4px;">${p.description}</div>
                        ${p.context_question ? `<div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">Context: ${p.context_question}</div>` : ''}
                    </div>`;
                }).join('');
        } else {
            patternsEl.innerHTML = '';
        }
        
        // Red flags
        const flagsEl = document.getElementById('detail-red-flags');
        const flags = analysis.red_flags || [];
        if (flags.length > 0) {
            flagsEl.innerHTML = '<h4 style="margin: 12px 0 8px;">🔴 Red Flags</h4>' +
                flags.map(f => `<div class="flag-item ${f.severity}">
                    <strong>${f.type.replace(/_/g, ' ')}</strong>
                    <div style="font-size: 0.85rem; margin-top: 4px;">${f.evidence}</div>
                </div>`).join('');
        } else {
            flagsEl.innerHTML = '';
        }
        
        // Recommendations
        const recsEl = document.getElementById('detail-recommendations');
        const recs = analysis.recommendations || [];
        if (recs.length > 0) {
            recsEl.innerHTML = '<h4 style="margin: 12px 0 8px;">📋 Recommendations</h4>' +
                recs.map(r => `<div class="recommendation-item ${r.priority}">
                    <span style="text-transform: uppercase; font-size: 0.7rem; font-weight: 700; color: var(--text-secondary);">${r.category}</span>
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
        if (!confirm('Are you sure you want to delete this session? This action cannot be undone.')) return;

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

            this.showToast('Session deleted successfully', 'success');
        } catch (err) {
            console.error('Failed to delete session:', err);
            this.showToast('Failed to delete session: ' + err.message, 'error');
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
