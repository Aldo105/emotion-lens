class ComparisonManager {
    constructor() {
        this.selectA = document.getElementById('compare-select-a');
        this.selectB = document.getElementById('compare-select-b');
        this.btnCompare = document.getElementById('btn-compare');
        this.resultsContainer = document.getElementById('compare-results');
        this.radarCanvas = document.getElementById('compare-radar-chart');
        this.radarChart = null;

        this.btnCompare.addEventListener('click', () => this.compare());
    }

    async loadSessionOptions() {
        try {
            const res = await fetch(`${CONFIG.API_URL}/sessions/`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            const sessions = (data.sessions || data || [])
                .filter(s => s.status === 'completed');

            this.populateDropdown(this.selectA, sessions);
            this.populateDropdown(this.selectB, sessions);
        } catch (err) {
            console.error('Failed to load sessions for comparison:', err);
        }
    }

    populateDropdown(selectEl, sessions) {
        const currentVal = selectEl.value;
        selectEl.innerHTML = '<option value="" disabled selected>Select a session...</option>';

        sessions.forEach(session => {
            const date = new Date(session.created_at || session.start_time || Date.now());
            const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            const name = session.candidate_name || session.name || 'Unknown';
            const opt = document.createElement('option');
            opt.value = session.id;
            opt.textContent = `${name} — ${dateStr}`;
            selectEl.appendChild(opt);
        });

        if (currentVal) selectEl.value = currentVal;
    }

    async compare() {
        const idA = this.selectA.value;
        const idB = this.selectB.value;

        if (!idA || !idB) {
            this.showMessage('Please select two sessions to compare.');
            return;
        }

        if (idA === idB) {
            this.showMessage('Please select two different sessions.');
            return;
        }

        this.btnCompare.disabled = true;
        this.btnCompare.textContent = 'Comparing...';

        try {
            const res = await fetch(`${CONFIG.API_URL}/sessions/compare`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_ids: [idA, idB] })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const result = await res.json();
            this.renderResults(result);
        } catch (err) {
            console.error('Comparison failed:', err);
            this.showMessage('Comparison failed: ' + err.message);
        } finally {
            this.btnCompare.disabled = false;
            this.btnCompare.textContent = '🔄 Compare Sessions';
        }
    }

    renderResults(result) {
        const a = result.sessions?.[0] || result.candidate_a || result;
        const b = result.sessions?.[1] || result.candidate_b || {};

        this.resultsContainer.innerHTML = `
            <div class="compare-columns">
                <div class="compare-column glass-panel">
                    <h3 class="compare-candidate-name">${this.escapeHtml(a.candidate_name || 'Candidate A')}</h3>
                    ${this.renderMetricCard(a)}
                </div>
                <div class="compare-divider">
                    <span class="compare-vs">VS</span>
                </div>
                <div class="compare-column glass-panel">
                    <h3 class="compare-candidate-name">${this.escapeHtml(b.candidate_name || 'Candidate B')}</h3>
                    ${this.renderMetricCard(b)}
                </div>
            </div>

            <div class="compare-radar-container glass-panel">
                <h3>Key Metrics Comparison</h3>
                <div class="radar-chart-wrapper">
                    <canvas id="compare-radar-chart"></canvas>
                </div>
            </div>
        `;

        // Re-acquire canvas ref after innerHTML replacement
        this.radarCanvas = document.getElementById('compare-radar-chart');
        this.renderRadarChart(a, b);
    }

    renderMetricCard(data) {
        const congruence = data.congruence_score != null ? Math.round(data.congruence_score) : '--';
        const microCount = data.micro_expression_count ?? 0;

        const emotionBars = Object.entries(data.emotion_distribution || {})
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5)
            .map(([emotion, value]) => {
                const cfg = CONFIG.EMOTIONS[emotion] || CONFIG.EMOTIONS['neutral'];
                const pct = Math.round(value * 100);
                return `
                    <div class="compare-emotion-row">
                        <span class="compare-emotion-label">${cfg.icon} ${cfg.label}</span>
                        <div class="compare-bar-track">
                            <div class="compare-bar-fill" style="width: ${pct}%; background: ${cfg.color}"></div>
                        </div>
                        <span class="compare-emotion-pct">${pct}%</span>
                    </div>
                `;
            }).join('');

        let congruenceColor = 'var(--success)';
        if (congruence !== '--') {
            if (congruence < 50) congruenceColor = 'var(--danger)';
            else if (congruence < 80) congruenceColor = 'var(--warning)';
        }

        return `
            <div class="compare-stat-row">
                <div class="compare-stat">
                    <span class="compare-stat-value" style="color: ${congruenceColor}">${congruence}%</span>
                    <span class="compare-stat-label">Congruence</span>
                </div>
                <div class="compare-stat">
                    <span class="compare-stat-value">${microCount}</span>
                    <span class="compare-stat-label">Micro-Expr.</span>
                </div>
            </div>
            <div class="compare-emotion-section">
                <h4>Top Emotions</h4>
                ${emotionBars || '<p class="empty-state">No data</p>'}
            </div>
        `;
    }

    renderRadarChart(a, b) {
        if (this.radarChart) {
            this.radarChart.destroy();
            this.radarChart = null;
        }

        const ctx = this.radarCanvas.getContext('2d');
        const labels = ['Congruence', 'Confidence', 'Nervousness', 'Stability', 'Micro-Expr Rate'];

        const getMetric = (data, key, fallback = 0) => {
            return data[key] ?? data.summary?.[key] ?? data.breakdown?.[key] ?? fallback;
        };

        const dataA = [
            getMetric(a, 'congruence_score'),
            (a.emotion_distribution?.confidence || 0) * 100,
            (a.emotion_distribution?.nervousness || 0) * 100,
            getMetric(a, 'stability_score', getMetric(a, 'stability', 50)),
            Math.min((getMetric(a, 'micro_expression_count') / 20) * 100, 100)
        ];
        const dataB = [
            getMetric(b, 'congruence_score'),
            (b.emotion_distribution?.confidence || 0) * 100,
            (b.emotion_distribution?.nervousness || 0) * 100,
            getMetric(b, 'stability_score', getMetric(b, 'stability', 50)),
            Math.min((getMetric(b, 'micro_expression_count') / 20) * 100, 100)
        ];

        this.radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: a.candidate_name || 'Candidate A',
                        data: dataA,
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.15)',
                        borderWidth: 2,
                        pointBackgroundColor: '#6366f1',
                        pointBorderColor: '#6366f1',
                        pointRadius: 4
                    },
                    {
                        label: b.candidate_name || 'Candidate B',
                        data: dataB,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.15)',
                        borderWidth: 2,
                        pointBackgroundColor: '#10b981',
                        pointBorderColor: '#10b981',
                        pointRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#9ba1b0',
                            usePointStyle: true,
                            boxWidth: 8
                        }
                    }
                },
                scales: {
                    r: {
                        min: 0,
                        max: 100,
                        ticks: {
                            stepSize: 25,
                            color: '#9ba1b0',
                            backdropColor: 'transparent'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.08)'
                        },
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.08)'
                        },
                        pointLabels: {
                            color: '#9ba1b0',
                            font: { size: 12, family: 'Inter' }
                        }
                    }
                }
            }
        });
    }

    showMessage(text) {
        this.resultsContainer.innerHTML = `
            <div class="compare-message glass-panel">
                <p>${text}</p>
            </div>
        `;
    }

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}
