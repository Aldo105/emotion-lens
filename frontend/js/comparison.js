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
        selectEl.innerHTML = '<option value="" disabled selected>Selecciona una sesión...</option>';

        sessions.forEach(session => {
            const date = new Date(session.created_at || session.start_time || Date.now());
            const dateStr = date.toLocaleDateString('es-ES', { month: 'short', day: 'numeric' });
            const name = session.candidate_name || session.name || 'Desconocido';
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
            this.showMessage('Selecciona dos sesiones para comparar.');
            return;
        }

        if (idA === idB) {
            this.showMessage('Selecciona dos sesiones diferentes.');
            return;
        }

        this.btnCompare.disabled = true;
        this.btnCompare.textContent = 'Comparando...';

        try {
            // Métricas de usabilidad de cada sesión, más los datos de sesión
            // para saber si se trata de la misma persona.
            const [uxA, uxB, infoA, infoB] = await Promise.all([
                fetch(`${CONFIG.API_URL}/reports/${idA}/ux-metrics`).then(r => r.json()),
                fetch(`${CONFIG.API_URL}/reports/${idB}/ux-metrics`).then(r => r.json()),
                fetch(`${CONFIG.API_URL}/sessions/${idA}`).then(r => r.json()),
                fetch(`${CONFIG.API_URL}/sessions/${idB}`).then(r => r.json()),
            ]);
            this.renderResults({ uxA, uxB, infoA, infoB });
        } catch (err) {
            console.error('Comparison failed:', err);
            this.showMessage('Error al comparar: ' + err.message);
        } finally {
            this.btnCompare.disabled = false;
            this.btnCompare.innerHTML = Icons.render('refresh') + ' Comparar Sesiones';
        }
    }

    renderResults({ uxA, uxB, infoA, infoB }) {
        const nombreA = infoA.name || `Sesión ${infoA.id}`;
        const nombreB = infoB.name || `Sesión ${infoB.id}`;

        // Comparar dos personas distintas no es válido con este sistema: sobre
        // 20 actores el acierto va del 9% al 73% según el individuo, así que la
        // diferencia entre dos personas puede venir del sistema y no de ellas.
        // Entre dos sesiones de la MISMA persona ese sesgo se cancela.
        const mismaPersona = infoA.candidate_name && infoB.candidate_name
            && infoA.candidate_name.trim().toLowerCase() === infoB.candidate_name.trim().toLowerCase();

        const aviso = mismaPersona ? '' : `
            <div class="compare-warning">
                ${Icons.render('alert')}
                <div>
                    <strong>Comparación entre personas distintas</strong>
                    <p>La precisión del sistema varía entre el 9% y el 73% según la persona,
                    así que las diferencias de abajo pueden deberse al sistema y no a los
                    participantes. Para comparar diseños, usa sesiones de la misma persona
                    en cada variante.</p>
                </div>
            </div>`;

        this.resultsContainer.innerHTML = `
            ${aviso}
            <div class="compare-columns">
                <div class="compare-column glass-panel">
                    <h3 class="compare-candidate-name">${this.escapeHtml(nombreA)}</h3>
                    ${this.renderUxCard(uxA)}
                </div>
                <div class="compare-divider"><span class="compare-vs">vs</span></div>
                <div class="compare-column glass-panel">
                    <h3 class="compare-candidate-name">${this.escapeHtml(nombreB)}</h3>
                    ${this.renderUxCard(uxB)}
                </div>
            </div>
            ${this.renderDiff(uxA, uxB, nombreA, nombreB)}
        `;
    }

    renderUxCard(ux) {
        if (!ux || !ux.disponible) {
            return '<p class="compare-empty">Sin datos suficientes en esta sesión.</p>';
        }
        const g = ux.global;
        const filas = [
            ['Duración', this.formatoTiempo(g.duracion_s)],
            ['Tiempo sonriendo', `${g.tiempo_sonriendo_pct}%`],
            ['Primera sonrisa', g.primera_sonrisa_s != null ? `${g.primera_sonrisa_s}s` : '—'],
            ['Eventos expresivos', g.eventos_expresivos],
            ['Eventos por minuto', g.eventos_por_minuto],
            ['Variabilidad expresiva', g.variabilidad_expresiva],
        ];
        return `<div class="ux-metric-list">` + filas.map(([k, v]) => `
            <div class="ux-metric-row">
                <span class="ux-metric-label">${k}</span>
                <span class="ux-metric-value">${v}</span>
            </div>`).join('') + `</div>`;
    }

    renderDiff(uxA, uxB, nombreA, nombreB) {
        if (!uxA?.disponible || !uxB?.disponible) return '';
        const a = uxA.global, b = uxB.global;

        const comparaciones = [
            { etiqueta: 'Tiempo sonriendo', a: a.tiempo_sonriendo_pct, b: b.tiempo_sonriendo_pct,
              unidad: '%', masEsMejor: true },
            { etiqueta: 'Eventos por minuto', a: a.eventos_por_minuto, b: b.eventos_por_minuto,
              unidad: '', masEsMejor: null },
            { etiqueta: 'Variabilidad expresiva', a: a.variabilidad_expresiva, b: b.variabilidad_expresiva,
              unidad: '', masEsMejor: null },
        ];

        return `
            <div class="compare-radar-container glass-panel">
                <h3>Diferencias</h3>
                <table class="compare-diff-table">
                    <thead>
                        <tr>
                            <th>Métrica</th>
                            <th>${this.escapeHtml(nombreA)}</th>
                            <th>${this.escapeHtml(nombreB)}</th>
                            <th>Diferencia</th>
                        </tr>
                    </thead>
                    <tbody>
                    ${comparaciones.map(c => {
                        const d = (c.b - c.a);
                        const signo = d > 0 ? '+' : '';
                        const clase = c.masEsMejor === null ? '' : (d > 0 ? 'diff-up' : (d < 0 ? 'diff-down' : ''));
                        return `<tr>
                            <td>${c.etiqueta}</td>
                            <td>${c.a}${c.unidad}</td>
                            <td>${c.b}${c.unidad}</td>
                            <td class="${clase}">${signo}${Math.round(d * 100) / 100}${c.unidad}</td>
                        </tr>`;
                    }).join('')}
                    </tbody>
                </table>
                <p class="compare-note">
                    Solo se comparan señales que resisten un cambio de cara: la sonrisa
                    (95% de acierto sobre 20 personas) y el recuento de cambios expresivos,
                    que no depende de acertar qué emoción hubo.
                </p>
            </div>
        `;
    }

    formatoTiempo(seg) {
        if (seg == null) return '—';
        const m = Math.floor(seg / 60), s = Math.round(seg % 60);
        return m ? `${m}m ${s}s` : `${s}s`;
    }

    renderRadarChart(a, b) {
        if (this.radarChart) {
            this.radarChart.destroy();
            this.radarChart = null;
        }

        const ctx = this.radarCanvas.getContext('2d');
        const labels = ['Congruencia', 'Confianza', 'Nerviosismo', 'Estabilidad', 'Tasa de Microexpr.'];

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
                        label: a.candidate_name || 'Candidato A',
                        data: dataA,
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.15)',
                        borderWidth: 2,
                        pointBackgroundColor: '#6366f1',
                        pointBorderColor: '#6366f1',
                        pointRadius: 4
                    },
                    {
                        label: b.candidate_name || 'Candidato B',
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
