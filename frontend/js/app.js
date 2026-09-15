// Global manager references (needed by onclick handlers in HTML)
let sessionManager;
let comparisonManager;
let videoUploadManager;
let postSessionSurvey;
let i18n;

document.addEventListener('DOMContentLoaded', async () => {

    // ══════════════════════════════════════════
    //  i18n (Internationalization)
    // ══════════════════════════════════════════
    i18n = new I18nManager('es');
    await i18n.init();

    // Language toggle button
    const btnLang = document.getElementById('btn-lang');
    if (btnLang) {
        btnLang.addEventListener('click', () => i18n.toggle());
    }

    // ══════════════════════════════════════════
    //  ETHICAL DISCLAIMER
    // ══════════════════════════════════════════
    const disclaimerModal = document.getElementById('disclaimer-modal');
    const btnAccept = document.getElementById('btn-disclaimer-accept');
    const disclaimerKey = 'emotionlens-disclaimer-accepted';

    if (disclaimerModal && !localStorage.getItem(disclaimerKey)) {
        disclaimerModal.classList.add('active');
    } else if (disclaimerModal) {
        disclaimerModal.remove();
    }

    if (btnAccept) {
        btnAccept.addEventListener('click', () => {
            localStorage.setItem(disclaimerKey, 'true');
            disclaimerModal.classList.remove('active');
            setTimeout(() => disclaimerModal.remove(), 400);
        });
    }

    // ══════════════════════════════════════════
    //  CORE COMPONENTS
    // ══════════════════════════════════════════
    const charts = new DashboardCharts();
    const ui = new DashboardUI(charts);
    const webcam = new WebcamManager();
    ui.webcam = webcam; // Pass webcam reference for drawing overlays
    const ws = new WebSocketManager(
        (data) => ui.handleMessage(data),
        (status) => ui.updateStatus(status)
    );

    // Initialize Phase 7 managers
    sessionManager = new SessionManager();
    comparisonManager = new ComparisonManager();
    videoUploadManager = new VideoUploadManager();
    postSessionSurvey = new PostSessionSurvey();

    // ── DOM Elements ──
    const btnStart = document.getElementById('btn-start-session');
    const btnEnd = document.getElementById('btn-end-session');
    const btnRecalibrate = document.getElementById('btn-recalibrate');
    const timerDisplay = document.getElementById('session-timer');
    const noteInput = document.getElementById('note-input');
    const btnAddNote = document.getElementById('btn-add-note');
    const notesList = document.getElementById('notes-list');

    let sessionActive = false;
    let sessionTimerInterval = null;
    let sessionStartTime = null;

    // ── Timer ──
    function updateTimer() {
        if (!sessionStartTime) return;
        const now = new Date();
        const diff = Math.floor((now - sessionStartTime) / 1000);
        
        const hrs = Math.floor(diff / 3600).toString().padStart(2, '0');
        const mins = Math.floor((diff % 3600) / 60).toString().padStart(2, '0');
        const secs = Math.floor(diff % 60).toString().padStart(2, '0');
        
        timerDisplay.textContent = `${hrs}:${mins}:${secs}`;
    }

    btnStart.addEventListener('click', async () => {
        const streamReady = await webcam.start();
        if (streamReady) {
            ws.connect(webcam);
            
            sessionActive = true;
            sessionStartTime = new Date();
            sessionTimerInterval = setInterval(updateTimer, 1000);
            
            btnStart.classList.add('hidden');
            btnEnd.classList.remove('hidden');
            charts.reset();
            
            // Reset session/UI micro-expression counters
            ui.collectedMicroExpressions = [];
            ui.microCount = 0;
            if (ui.elMicroCount) ui.elMicroCount.textContent = '0 Detected';
            if (ui.elMicroLog) ui.elMicroLog.innerHTML = '<div class="empty-state">No micro-expressions detected yet.</div>';
        }
    });

    // ── End Session ──
    btnEnd.addEventListener('click', () => {
        ws.disconnect();
        webcam.stop();
        
        sessionActive = false;
        clearInterval(sessionTimerInterval);
        
        btnStart.classList.remove('hidden');
        btnEnd.classList.add('hidden');

        // Show optional post-session survey for the subject
        postSessionSurvey.show({
            session_id: ws.lastSessionId || null,
            micro_expressions: ui.collectedMicroExpressions || [],
            dominant_emotion: null,
        });
    });

    // ── Recalibrate ──
    if (btnRecalibrate) {
        btnRecalibrate.addEventListener('click', () => {
            if (ws.isConnected) {
                ws.requestRecalibrate();
            }
        });
    }

    // ── Window resize ──
    window.addEventListener('resize', () => {
        if (sessionActive) webcam.resizeCanvas();
    });

    // ── Add Note / Event Marker Logic ──
    const MARKER_LABELS = {
        task_start:   '▶️ Inicio tarea',
        task_end:     '⏹️ Fin tarea',
        error:        '⚠️ Error',
        confusion:    '❓ Confusión',
        key_question: '💬 Pregunta clave',
    };

    /** Seconds elapsed since the session started — what the backend correlates on. */
    function elapsedSeconds() {
        if (!sessionStartTime) return 0;
        return (Date.now() - sessionStartTime.getTime()) / 1000;
    }

    function renderNote({ text, tag, time }) {
        const div = document.createElement('div');
        div.className = 'note-item' + (tag ? ' tagged' : '');

        const header = document.createElement('div');
        header.className = 'note-header';
        const timeSpan = document.createElement('span');
        timeSpan.textContent = `🕒 ${time}`;
        const tagSpan = document.createElement('span');
        tagSpan.textContent = tag ? MARKER_LABELS[tag] || tag : 'Usuario';
        header.appendChild(timeSpan);
        header.appendChild(tagSpan);

        const noteText = document.createElement('div');
        noteText.className = 'note-text';
        noteText.textContent = text;  // Safe — no HTML injection

        div.appendChild(header);
        div.appendChild(noteText);
        notesList.prepend(div);
    }

    /**
     * Persist the note. Without this the interviewer_notes table stays empty
     * and every note-driven analysis (task friction, question correlation)
     * silently has nothing to work with.
     */
    async function persistNote(content, tag, timestamp) {
        const sessionId = ws.lastSessionId;
        if (!sessionId) {
            console.warn('[Notes] No active session id — note kept locally only');
            return;
        }
        try {
            const resp = await fetch(`${CONFIG.API_URL}/notes/${sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content, tag, timestamp }),
            });
            if (!resp.ok) console.warn('[Notes] Save failed:', resp.status);
        } catch (err) {
            console.warn('[Notes] Save failed:', err);
        }
    }

    function addNote(tag = null) {
        const typed = noteInput.value.trim();
        // A marker doesn't need typed text; a plain note does.
        const text = typed || (tag ? MARKER_LABELS[tag] : '');
        if (!text) return;

        renderNote({ text, tag, time: timerDisplay.textContent });
        persistNote(text, tag, elapsedSeconds());
        noteInput.value = '';
    }

    btnAddNote.addEventListener('click', () => addNote());
    noteInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') addNote();
    });

    document.querySelectorAll('#event-markers .btn-marker').forEach(btn => {
        btn.addEventListener('click', () => {
            if (!sessionActive) return;
            addNote(btn.dataset.tag);
        });
    });

    // ══════════════════════════════════════════
    //  TAB NAVIGATION
    // ══════════════════════════════════════════
    const navTabs = document.querySelectorAll('.nav-tab');
    const viewPanels = document.querySelectorAll('.view-panel');
    const navIndicator = document.getElementById('nav-indicator');

    function positionIndicator(tab) {
        if (!tab || !navIndicator) return;
        const tabRect = tab.getBoundingClientRect();
        const navRect = tab.parentElement.getBoundingClientRect();
        navIndicator.style.width = `${tabRect.width}px`;
        navIndicator.style.left = `${tabRect.left - navRect.left}px`;
    }

    function switchView(targetId) {
        // Hide all panels
        viewPanels.forEach(panel => panel.classList.remove('active'));

        // Show target panel
        const target = document.getElementById(targetId);
        if (target) target.classList.add('active');

        // Update tab states
        navTabs.forEach(tab => tab.classList.remove('active'));
        const activeTab = document.querySelector(`[data-view="${targetId}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
            positionIndicator(activeTab);
        }

        // Trigger data loads on tab activation
        if (targetId === 'view-sessions') {
            sessionManager.loadSessions();
        } else if (targetId === 'view-compare') {
            comparisonManager.loadSessionOptions();
        }
    }

    // Attach click handlers to nav tabs
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const viewId = tab.getAttribute('data-view');
            switchView(viewId);
        });
    });

    // Position indicator on the initially active tab
    const initialTab = document.querySelector('.nav-tab.active');
    if (initialTab) {
        // Wait for fonts/layout to settle
        requestAnimationFrame(() => positionIndicator(initialTab));
    }

    // Reposition indicator on window resize
    window.addEventListener('resize', () => {
        const active = document.querySelector('.nav-tab.active');
        if (active) positionIndicator(active);
    });

});

