/**
 * EmotionLens — login page.
 *
 * Loaded without app.js: before a session exists only i18n.js, icons.js and
 * this file are public (see backend/app/auth_middleware.py PUBLIC_PATHS).
 */
(function () {
    const THEME_KEY = 'emotionlens-theme';

    // Same rule as backend/app/auth_middleware.py safe_next: local paths only,
    // and no control characters (browsers strip tab/CR/LF, so "/\t/x" → "//x").
    function safeNext(raw) {
        if (!raw || raw[0] !== '/' || raw.startsWith('//') || raw.startsWith('/\\')
            || /[\u0000-\u001f\u007f]/.test(raw)) return '/';
        return raw;
    }

    function applyTheme(theme, button) {
        if (theme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
        button.innerHTML = Icons.render(theme === 'dark' ? 'sun' : 'moon');
    }

    document.addEventListener('DOMContentLoaded', async () => {
        // Look up all elements and attach listeners synchronously, before any awaits.
        const form = document.getElementById('login-form');
        const user = document.getElementById('login-username');
        const pass = document.getElementById('login-password');
        const submit = document.getElementById('login-submit');
        const error = document.getElementById('login-error');
        const reveal = document.getElementById('login-reveal');
        const btnTheme = document.getElementById('btn-theme');
        const btnLang = document.getElementById('btn-lang');

        const next = safeNext(new URLSearchParams(window.location.search).get('next'));

        const updateReveal = () => {
            const shown = pass.type === 'text';
            const label = i18n.t(shown ? 'login.hidePassword' : 'login.showPassword');
            reveal.title = label;
            reveal.setAttribute('aria-label', label);
            reveal.setAttribute('aria-pressed', String(shown));
        };

        // Load and apply theme immediately, before i18n init.
        let theme = 'light';
        try { theme = localStorage.getItem(THEME_KEY) || 'light'; } catch (err) { /* default */ }
        applyTheme(theme, btnTheme);

        // Attach theme listener synchronously.
        btnTheme.addEventListener('click', () => {
            theme = theme === 'dark' ? 'light' : 'dark';
            applyTheme(theme, btnTheme);
            try { localStorage.setItem(THEME_KEY, theme); } catch (err) { /* not persisted */ }
        });

        // Initialize i18n with a ready flag; lang listener can run before init completes.
        const i18n = new I18nManager('es');
        let i18nReady = false;

        const setTitle = () => { document.title = `${i18n.t('login.title')} · EmotionLens`; };

        // Attach lang listener synchronously; guard with ready flag.
        btnLang.addEventListener('click', async () => {
            if (!i18nReady) return;
            await i18n.toggle();
            setTitle();
            updateReveal();
        });

        // Attach reveal listener synchronously.
        reveal.addEventListener('click', () => {
            pass.type = pass.type === 'password' ? 'text' : 'password';
            updateReveal();
            pass.focus();
        });

        // Attach form submit listener synchronously; preventDefault must be first.
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            error.textContent = '';
            if (!user.value.trim() || !pass.value) {
                error.textContent = i18n.t('login.invalid');
                return;
            }
            submit.disabled = true;
            submit.textContent = i18n.t('login.submitting');
            try {
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: user.value.trim(), password: pass.value }),
                });
                if (response.status === 204) {
                    window.location.replace(next);
                    return;
                }
                error.textContent = i18n.t(
                    response.status === 429 ? 'login.locked'
                        : response.status === 401 ? 'login.invalid'
                            : 'login.unavailable');
            } catch (err) {
                error.textContent = i18n.t('login.unavailable');
            }
            pass.value = '';
            pass.focus();
            submit.disabled = false;
            submit.textContent = i18n.t('login.submit');
        });

        // Now do the awaits. Set focus and init i18n.
        updateReveal();
        user.focus();

        await i18n.init();
        i18n.updateLangButton();
        i18nReady = true;
        setTitle();

        // With the login switched off (local development) there is nothing to do here.
        try {
            const status = await fetch('/api/auth/status').then((r) => r.json());
            if (!status.enabled) {
                window.location.replace(next);
                return;
            }
        } catch (err) {
            /* keep the form; submitting will report the problem */
        }

        // Only after both i18n and status check complete, enable the submit button.
        submit.disabled = false;
    });
})();
