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
                </div>

                <div class="modal-footer">
                    <button class="btn btn-primary" onclick="sessionManager.downloadPDF('${sessionId}')">📄 Download PDF</button>
                    <button class="btn btn-secondary" onclick="sessionManager.downloadCSV('${sessionId}')">📊 Download CSV</button>
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
