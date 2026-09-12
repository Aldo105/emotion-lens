class DashboardCharts {
    constructor() {
        this.timelineChart = null;
        this.gaugeChart = null;
        
        // Data arrays for timeline
        this.timeLabels = [];
        this.emotionData = {
            'happy': [], 'sad': [], 'angry': [], 'surprise': [],
            'disgust': [], 'fear': [], 'neutral': [], 'nervousness': [], 'confidence': []
        };
        
        this.maxDataPoints = 60; // Keep last N points on screen
        
        this.initTimelineChart();
        this.initGaugeChart();
    }

    initTimelineChart() {
        const ctx = document.getElementById('timeline-chart').getContext('2d');
        
        // Create datasets for each emotion
        const datasets = Object.keys(CONFIG.EMOTIONS).map(emotion => {
            return {
                label: CONFIG.EMOTIONS[emotion].label,
                data: this.emotionData[emotion],
                borderColor: getComputedStyle(document.documentElement).getPropertyValue(CONFIG.EMOTIONS[emotion].color.replace('var(', '').replace(')', '')).trim(),
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0,
                hidden: emotion === 'neutral' // Hide neutral by default to reduce clutter
            };
        });

        this.timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.timeLabels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#9ba1b0',
                            usePointStyle: true,
                            boxWidth: 8
                        }
                    },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    x: {
                        display: false // Hide X axis labels for cleaner look
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#9ba1b0' }
                    }
                }
            }
        });
    }

    initGaugeChart() {
        const ctx = document.getElementById('congruence-gauge').getContext('2d');
        
        this.gaugeChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Congruence', 'Gap'],
                datasets: [{
                    data: [0, 100],
                    backgroundColor: [
                        '#94a3b8', // Default color, will change based on score
                        'rgba(255, 255, 255, 0.05)'
                    ],
                    borderWidth: 0,
                    circumference: 180,
                    rotation: 270
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '80%',
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                animation: { animateRotate: false, animateScale: false }
            }
        });
    }

    updateTimeline(timestamp, probabilities) {
        // Format timestamp as mm:ss
        const mins = Math.floor(timestamp / 60).toString().padStart(2, '0');
        const secs = Math.floor(timestamp % 60).toString().padStart(2, '0');
        this.timeLabels.push(`${mins}:${secs}`);

        // Update data arrays
        Object.keys(this.emotionData).forEach(emotion => {
            const prob = (probabilities[emotion] || 0) * 100;
            this.emotionData[emotion].push(prob);
        });

        // Keep only recent data points to avoid crowding
        if (this.timeLabels.length > this.maxDataPoints) {
            this.timeLabels.shift();
            Object.keys(this.emotionData).forEach(emotion => {
                this.emotionData[emotion].shift();
            });
        }

        this.timelineChart.update();
    }

    updateGauge(score, colorName) {
        const gaugeData = this.gaugeChart.data.datasets[0];
        gaugeData.data[0] = score;
        gaugeData.data[1] = 100 - score;
        
        // Map color name to CSS variable
        let cssColor = '#10b981'; // Green default
        if (colorName === 'yellow') cssColor = '#f59e0b';
        if (colorName === 'red') cssColor = '#ef4444';
        if (score === 0) cssColor = '#94a3b8'; // Grey if not ready
        
        gaugeData.backgroundColor[0] = cssColor;
        this.gaugeChart.update();
    }
    
    reset() {
        this.timeLabels.length = 0;
        Object.keys(this.emotionData).forEach(emotion => {
            this.emotionData[emotion].length = 0;
        });
        this.timelineChart.update();
        this.updateGauge(0, 'grey');
    }
}
