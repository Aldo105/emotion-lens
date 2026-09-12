/**
 * EmotionLens — Internationalization (i18n) Module
 *
 * Provides bilingual support (English / Spanish) for the entire dashboard.
 * Loads language packs from /i18n/{lang}.json and applies translations
 * to all elements with `data-i18n` attributes.
 *
 * Usage:
 *   const i18n = new I18nManager('en');
 *   await i18n.init();
 *   i18n.t('live.startSession');  // → "Start Session"
 *   i18n.setLanguage('es');       // switches to Spanish
 */

class I18nManager {
    constructor(defaultLang = 'en') {
        this.currentLang = localStorage.getItem('emotionlens-lang') || defaultLang;
        this.translations = {};
        this.supportedLangs = ['en', 'es'];
    }

    /**
     * Initialize — load the current language pack.
     */
    async init() {
        await this.loadLanguage(this.currentLang);
        this.applyTranslations();
        this.updateLangButton();
    }

    /**
     * Load a language pack from /i18n/{lang}.json.
     */
    async loadLanguage(lang) {
        try {
            const res = await fetch(`/i18n/${lang}.json`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            this.translations = await res.json();
            this.currentLang = lang;
            localStorage.setItem('emotionlens-lang', lang);
        } catch (err) {
            console.error(`[i18n] Failed to load language '${lang}':`, err);
            // Fallback to English if Spanish fails
            if (lang !== 'en') {
                console.warn('[i18n] Falling back to English');
                await this.loadLanguage('en');
            }
        }
    }

    /**
     * Switch language and re-apply all translations.
     */
    async setLanguage(lang) {
        if (!this.supportedLangs.includes(lang)) {
            console.warn(`[i18n] Unsupported language: ${lang}`);
            return;
        }
        await this.loadLanguage(lang);
        this.applyTranslations();
        this.updateLangButton();
        this.updateEmotionLabels();
    }

    /**
     * Toggle between EN ↔ ES.
     */
    async toggle() {
        const next = this.currentLang === 'en' ? 'es' : 'en';
        await this.setLanguage(next);
    }

    /**
     * Get a translated string by dot-notation key.
     * Example: t('live.startSession') → "Start Session"
     */
    t(key, fallback = '') {
        const parts = key.split('.');
        let value = this.translations;
        for (const part of parts) {
            if (value && typeof value === 'object' && part in value) {
                value = value[part];
            } else {
                return fallback || key;
            }
        }
        return typeof value === 'string' ? value : (fallback || key);
    }

    /**
     * Scan the DOM for [data-i18n] attributes and set textContent.
     * Supports:
     *   - data-i18n="key"             → sets textContent
     *   - data-i18n-placeholder="key" → sets placeholder attribute
     *   - data-i18n-title="key"       → sets title attribute
     */
    applyTranslations() {
        // Text content
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            const text = this.t(key);
            if (text && text !== key) {
                el.textContent = text;
            }
        });

        // Placeholders
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            const key = el.getAttribute('data-i18n-placeholder');
            const text = this.t(key);
            if (text && text !== key) {
                el.placeholder = text;
            }
        });

        // Titles
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            const key = el.getAttribute('data-i18n-title');
            const text = this.t(key);
            if (text && text !== key) {
                el.title = text;
            }
        });
    }

    /**
     * Update the language toggle button text.
     */
    updateLangButton() {
        const btn = document.getElementById('btn-lang');
        if (btn) {
            btn.textContent = this.currentLang.toUpperCase();
            btn.title = this.currentLang === 'en'
                ? 'Switch to Español'
                : 'Cambiar a English';
        }
    }

    /**
     * Update CONFIG.EMOTIONS labels based on the current language.
     */
    updateEmotionLabels() {
        if (typeof CONFIG !== 'undefined' && CONFIG.EMOTIONS) {
            for (const [key, cfg] of Object.entries(CONFIG.EMOTIONS)) {
                const label = this.t(`emotions.${key}`);
                if (label && label !== `emotions.${key}`) {
                    cfg.label = label;
                }
            }
        }
    }
}
