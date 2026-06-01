const CONFIG = {
    // Backend API URLs
    WS_URL: `ws://${window.location.host}/ws/emotion`,
    API_URL: `http://${window.location.host}/api`,
    
    // Video capture settings
    CAPTURE_FPS: 10,        // Frames per second to send to backend
    CAPTURE_WIDTH: 640,     // Downscale width for faster transmission
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
    }
};
