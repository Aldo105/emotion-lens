// Global manager references (needed by onclick handlers in HTML)
let sessionManager;
let comparisonManager;
let videoUploadManager;
let i18n;

document.addEventListener('DOMContentLoaded', async () => {

    // ══════════════════════════════════════════
    //  i18n (Internationalization)
    // ══════════════════════════════════════════
    i18n = new I18nManager('en');
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
    const ws = new WebSocketManager(
        (data) => ui.handleMessage(data),
        (status) => ui.updateStatus(status)
    );

    // Initialize Phase 7 managers
    sessionManager = new SessionManager();
    comparisonManager = new ComparisonManager();
    videoUploadManager = new VideoUploadManager();

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

    // ── Start Session ──
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

    // ── Add Note Logic ──
    function addNote() {
        const text = noteInput.value.trim();
        if (!text) return;
        
        const time = timerDisplay.textContent;
        const div = document.createElement('div');
        div.className = 'note-item';
        div.innerHTML = `
            <div class="note-header">
                <span>🕒 ${time}</span>
                <span>User</span>
            </div>
            <div class="note-text">${text}</div>
        `;
        
        notesList.prepend(div);
        noteInput.value = '';
    }

    btnAddNote.addEventListener('click', addNote);
    noteInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') addNote();
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

