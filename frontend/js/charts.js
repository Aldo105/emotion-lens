class DashboardCharts {
    constructor() {
        this.timelineChart = null;
        this.gaugeChart    = null;

        // Data arrays for timeline
        this.timeLabels = [];
        this.emotionData = {
            'happy': [], 'sad': [], 'angry': [], 'surprise': [],
            'disgust': [], 'fear': [], 'neutral': [], 'nervousness': [], 'confidence': [],
        };

        // Show only last N data points to keep the chart readable
        this.maxDataPoints = 60;

        // Throttle: only push a chart render at most once per second
        // (WebSocket sends frames at 20 fps but the chart doesn't need to redraw that fast)
        this._lastChartRender = 0;
        this._pendingUpdate   = false;

        this.initTimelineChart();
        this.initGaugeChart();
    }

    initTimelineChart() {
        const ctx = document.getElementById('timeline-chart').getContext('2d');

        const datasets = Object.keys(CONFIG.EMOTIONS).map(emotion => ({
            label:           CONFIG.EMOTIONS[emotion].label,
            data:            this.emotionData[emotion],
            borderColor:     getComputedStyle(document.documentElement)
                                 .getPropertyValue(
                                     CONFIG.EMOTIONS[emotion].color.replace('var(', '').replace(')', '')
                                 ).trim(),
            backgroundColor: 'transparent',
            borderWidth:     2,
            tension:         0.4,
            pointRadius:     0,
            hidden:          emotion === 'neutral', // hide neutral by default
        }));

        this.timelineChart = new Chart(ctx, {
            type: 'line',
            data: { labels: this.timeLabels, datasets },
            options: {
                responsive:          true,
                maintainAspectRatio: false,
                animation:           false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        position: 'top',
                        labels:   { color: '#9ba1b0', usePointStyle: true, boxWidth: 8 },
                    },
                    tooltip: { mode: 'index', intersect: false },
                },
                scales: {
                    x: { display: false },
                    y: {
                        min:  0,
                        max:  100,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#9ba1b0' },
                    },
                },
            },
        });
    }

    initGaugeChart() {
        const ctx = document.getElementById('congruence-gauge').getContext('2d');

        this.gaugeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Congruence', 'Gap'],
                datasets: [{
                    data:            [0, 100],
                    backgroundColor: ['#94a3b8', 'rgba(255,255,255,0.05)'],
                    borderWidth:     0,
                    circumference:   180,
                    rotation:        270,
                }],
            },
            options: {
                responsive:          true,
                maintainAspectRatio: false,
                cutout:              '80%',
                plugins: {
                    legend:  { display: false },
                    tooltip: { enabled: false },
                },
                animation: { animateRotate: false, animateScale: false },
            },
        });
    }

    /**
     * Push new data into the timeline buffers and schedule a throttled redraw.
     * Called on every WebSocket frame (~20 fps); the chart only repaints once/second.
     */
    updateTimeline(timestamp, probabilities) {
        const mins = Math.floor(timestamp / 60).toString().padStart(2, '0');
        const secs = Math.floor(timestamp % 60).toString().padStart(2, '0');
        this.timeLabels.push(`${mins}:${secs}`);

        Object.keys(this.emotionData).forEach(emotion => {
            this.emotionData[emotion].push((probabilities[emotion] || 0) * 100);
        });

        // Trim to maxDataPoints
        if (this.timeLabels.length > this.maxDataPoints) {
            this.timeLabels.shift();
            Object.keys(this.emotionData).forEach(e => this.emotionData[e].shift());
        }

        // Throttled render: at most once per second
        const now = Date.now();
        if (!this._pendingUpdate) {
            this._pendingUpdate = true;
            const delay = Math.max(0, 1000 - (now - this._lastChartRender));
            setTimeout(() => {
                // 'none' skips the transition animation pass — still redraws data
                this.timelineChart.update('none');
                this._lastChartRender = Date.now();
                this._pendingUpdate   = false;
            }, delay);
        }
    }

    updateGauge(score, colorName) {
        const ds = this.gaugeChart.data.datasets[0];
        ds.data[0] = score;
        ds.data[1] = 100 - score;

        ds.backgroundColor[0] =
            colorName === 'yellow' ? '#f59e0b' :
            colorName === 'red'    ? '#ef4444' :
            score === 0            ? '#94a3b8' :
                                     '#10b981'; // green

        // 'none' avoids pointless animation recalculation
        this.gaugeChart.update('none');
    }

    reset() {
        this.timeLabels.length = 0;
        Object.keys(this.emotionData).forEach(e => { this.emotionData[e].length = 0; });
        this.timelineChart.update('none');
        this.updateGauge(0, 'grey');
    }
}
