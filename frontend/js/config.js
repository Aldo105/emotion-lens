const CONFIG = {
    // Backend API URLs
    WS_URL: `ws://${window.location.host}/ws/emotion`,
    API_URL: `http://${window.location.host}/api`,
    
    // Video capture settings
    CAPTURE_FPS: 20,        // Frames per second — 20 FPS needed for micro-expression detection
    CAPTURE_WIDTH: 640,     // 640×480 is sufficient for MediaPipe 468-landmark precision
    CAPTURE_HEIGHT: 480,
    
    // Emotion mapping (colors & emojis)
    EMOTIONS: {
        'happy': { color: 'var(--color-happy)', icon: '😊', label: 'Happy' },
        'sad': { color: 'var(--color-sad)', icon: '😢', label: 'Sad' },
        'angry': { color: 'var(--color-angry)', icon: '😠', label: 'Angry' },
        'surprise': { color: 'var(--color-surprise)', icon: '😲', label: 'Surprise' },
        'disgust': { color: 'var(--color-disgust)', icon: '🤢', label: 'Disgust' },
        'fear': { color: 'var(--color-fear)', icon: '😨', label: 'Fear' },
        'neutral': { color: 'var(--color-neutral)', icon: '😐', label: 'Neutral' },
        'nervousness': { color: 'var(--color-nervousness)', icon: '😰', label: 'Nervous' },
        'confidence': { color: 'var(--color-confidence)', icon: '😎', label: 'Confident' }
    },
    
    ANALYSIS_DIMENSIONS: {
        technical_mastery: { label: 'Technical Mastery', color: '#3b82f6', icon: '🔧' },
        emotional_stability: { label: 'Emotional Stability', color: '#10b981', icon: '⚖️' },
        authenticity: { label: 'Authenticity', color: '#8b5cf6', icon: '🎭' },
        self_confidence: { label: 'Self Confidence', color: '#f59e0b', icon: '💪' },
        communication: { label: 'Communication', color: '#ec4899', icon: '💬' },
    },
    SEVERITY_COLORS: {
        high: '#ef4444',
        medium: '#f59e0b',
        low: '#6b7280',
    }
};
