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
        'happy': { color: 'var(--color-happy)', label: 'Feliz' },
        'sad': { color: 'var(--color-sad)', label: 'Triste' },
        'angry': { color: 'var(--color-angry)', label: 'Enojado' },
        'surprise': { color: 'var(--color-surprise)', label: 'Sorpresa' },
        'disgust': { color: 'var(--color-disgust)', label: 'Disgusto' },
        'fear': { color: 'var(--color-fear)', label: 'Miedo' },
        'neutral': { color: 'var(--color-neutral)', label: 'Neutral' },
        'nervousness': { color: 'var(--color-nervousness)', label: 'Nervioso' },
        'confidence': { color: 'var(--color-confidence)', label: 'Confiado' }
    },

    // Literal values rather than CSS variables: these feed canvas charts,
    // which cannot resolve var(). Muted to match the pastel palette and chosen
    // to stay legible on both the cream and the charcoal background.
    ANALYSIS_DIMENSIONS: {
        technical_mastery: { label: 'Dominio Técnico', color: '#7F9BC4' },
        emotional_stability: { label: 'Estabilidad Emocional', color: '#7FA88C' },
        authenticity: { label: 'Autenticidad', color: '#A38BC4' },
        self_confidence: { label: 'Autoconfianza', color: '#D9A566' },
        communication: { label: 'Comunicación', color: '#CE87A6' },
    },
    SEVERITY_COLORS: {
        high: '#C97D7D',
        medium: '#D9A566',
        low: '#9B9892',
    }
};
