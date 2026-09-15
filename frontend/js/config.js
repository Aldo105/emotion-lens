const CONFIG = {
    // Backend API URLs
    WS_URL: `ws://${window.location.host}/ws/emotion`,
    API_URL: `http://${window.location.host}/api`,
    
    // Video capture settings
    CAPTURE_FPS: 20,        // Frames per second — 20 FPS needed for micro-expression detection
    CAPTURE_WIDTH: 640,     // 640×480 is sufficient for MediaPipe 468-landmark precision
    CAPTURE_HEIGHT: 480,
    
    // Emotion mapping (colors & emojis). Labels match frontend/i18n/es.json's
    // "emotions" section, which is what overwrites these on a language toggle
    // (see I18nManager.updateEmotionLabels) — keeping them in sync here avoids
    // the label changing on first toggle away and back to Spanish.
    EMOTIONS: {
        'happy': { color: 'var(--color-happy)', icon: '😊', label: 'Feliz' },
        'sad': { color: 'var(--color-sad)', icon: '😢', label: 'Triste' },
        'angry': { color: 'var(--color-angry)', icon: '😠', label: 'Enojado' },
        'surprise': { color: 'var(--color-surprise)', icon: '😲', label: 'Sorpresa' },
        'disgust': { color: 'var(--color-disgust)', icon: '🤢', label: 'Disgusto' },
        'fear': { color: 'var(--color-fear)', icon: '😨', label: 'Miedo' },
        'neutral': { color: 'var(--color-neutral)', icon: '😐', label: 'Neutral' },
        'nervousness': { color: 'var(--color-nervousness)', icon: '😰', label: 'Nervioso' },
        'confidence': { color: 'var(--color-confidence)', icon: '😎', label: 'Confiado' }
    },

    ANALYSIS_DIMENSIONS: {
        technical_mastery: { label: 'Dominio Técnico', color: '#3b82f6', icon: '🔧' },
        emotional_stability: { label: 'Estabilidad Emocional', color: '#10b981', icon: '⚖️' },
        authenticity: { label: 'Autenticidad', color: '#8b5cf6', icon: '🎭' },
        self_confidence: { label: 'Autoconfianza', color: '#f59e0b', icon: '💪' },
        communication: { label: 'Comunicación', color: '#ec4899', icon: '💬' },
    },
    SEVERITY_COLORS: {
        high: '#ef4444',
        medium: '#f59e0b',
        low: '#6b7280',
    }
};
