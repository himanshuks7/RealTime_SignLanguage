/**
 * Sign Language Translator — Frontend Logic
 * Polls the Flask API for detection state and updates the UI.
 */

(function () {
    'use strict';

    // ── DOM Elements ────────────────────────────────────────────────────
    const statusPill      = document.getElementById('status-pill');
    const statusText      = statusPill.querySelector('.status-text');
    const fpsBadge        = document.getElementById('fps-badge');
    const detectionBadge  = document.getElementById('detection-badge');
    const detectedSign    = document.getElementById('detected-sign');
    const confidenceBar   = document.getElementById('confidence-bar');
    const confidenceText  = document.getElementById('confidence-text');
    const currentSentence = document.getElementById('current-sentence');
    const signsGrid       = document.getElementById('signs-grid');
    const historyList     = document.getElementById('history-list');
    const btnClear        = document.getElementById('btn-clear');
    const btnSave         = document.getElementById('btn-save');
    const btnToggle       = document.getElementById('btn-toggle');
    const toast           = document.getElementById('toast');
    const toastMessage    = document.getElementById('toast-message');

    // ── State ───────────────────────────────────────────────────────────
    let lastSign = '';
    let isDetectionActive = true;
    let signsInitialized = false;
    let pollInterval = null;

    // ── Word categories for chip coloring ───────────────────────────────
    const categories = {
        'i': 'subject', 'you': 'subject', 'we': 'subject',
        'want': 'verb', 'need': 'verb',
        'help': 'object', 'food': 'object', 'water': 'object',
        'cat': 'object', 'dog': 'object',
        'hello': 'greeting', 'thanks': 'greeting',
        'please': 'modifier', 'sorry': 'modifier',
        'yes': 'affirmation', 'no': 'affirmation'
    };

    // ── Toast ───────────────────────────────────────────────────────────
    function showToast(message, duration = 2500) {
        toastMessage.textContent = message;
        toast.classList.add('visible');
        setTimeout(() => toast.classList.remove('visible'), duration);
    }

    // ── Initialize signs grid ───────────────────────────────────────────
    function initSignsGrid(signs) {
        if (signsInitialized || !signs.length) return;
        signsGrid.innerHTML = '';

        signs.forEach(sign => {
            const chip = document.createElement('span');
            chip.className = 'sign-chip';
            chip.textContent = sign;
            chip.id = `chip-${sign}`;
            chip.dataset.category = categories[sign] || 'unknown';
            signsGrid.appendChild(chip);
        });

        signsInitialized = true;
    }

    // ── Update UI from state ────────────────────────────────────────────
    function updateUI(data) {
        // Status
        statusPill.className = 'status-pill';
        if (!data.is_running) {
            statusText.textContent = 'Connecting...';
        } else if (!data.detection_active) {
            statusPill.classList.add('paused');
            statusText.textContent = 'Paused';
        } else {
            statusPill.classList.add('active');
            statusText.textContent = 'Detecting';
        }

        // FPS
        fpsBadge.textContent = `${Math.round(data.fps || 0)} FPS`;

        // Detection badge
        if (data.sign && data.confidence > 0) {
            detectionBadge.classList.add('visible');
            detectedSign.textContent = data.sign;
            
            const pct = Math.round(data.confidence * 100);
            confidenceBar.style.width = pct + '%';
            confidenceText.textContent = pct + '%';

            // Color by confidence level
            confidenceBar.className = 'confidence-bar';
            if (data.confidence > 0.85) {
                confidenceBar.classList.add('high');
            } else if (data.confidence > 0.7) {
                confidenceBar.classList.add('medium');
            } else {
                confidenceBar.classList.add('low');
            }

            // Highlight active chip
            if (data.sign !== lastSign) {
                document.querySelectorAll('.sign-chip.active').forEach(el =>
                    el.classList.remove('active')
                );
                const activeChip = document.getElementById(`chip-${data.sign}`);
                if (activeChip) activeChip.classList.add('active');
                lastSign = data.sign;
            }
        } else {
            detectionBadge.classList.remove('visible');
            if (!data.sign) {
                // Clear active chips after a delay
                setTimeout(() => {
                    if (!lastSign) {
                        document.querySelectorAll('.sign-chip.active').forEach(el =>
                            el.classList.remove('active')
                        );
                    }
                }, 2000);
                lastSign = '';
            }
        }

        // Sentence
        if (data.sentence) {
            currentSentence.textContent = data.sentence;
            currentSentence.classList.remove('placeholder');
        } else {
            currentSentence.textContent = 'Waiting for signs...';
            currentSentence.classList.add('placeholder');
        }

        // History
        if (data.history && data.history.length > 0) {
            const currentHTML = historyList.innerHTML;
            const newHTML = data.history
                .map((s, i) =>
                    `<li><span class="history-index">${i + 1}.</span> ${escapeHTML(s)}</li>`
                )
                .reverse()
                .join('');

            if (currentHTML !== newHTML) {
                historyList.innerHTML = newHTML;
            }
        }

        // Signs grid
        if (data.signs) {
            initSignsGrid(data.signs);
        }

        // Toggle button state
        isDetectionActive = data.detection_active;
        btnToggle.innerHTML = isDetectionActive
            ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Pause`
            : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Resume`;
    }

    // ── API Calls ───────────────────────────────────────────────────────
    async function fetchState() {
        try {
            const res = await fetch('/api/state');
            if (res.ok) {
                const data = await res.json();
                updateUI(data);
            }
        } catch (err) {
            statusPill.className = 'status-pill';
            statusText.textContent = 'Disconnected';
        }
    }

    async function clearSentence() {
        try {
            const res = await fetch('/api/clear', { method: 'POST' });
            if (res.ok) {
                showToast('🗑️ Sentence cleared');
            }
        } catch (err) {
            showToast('❌ Failed to clear');
        }
    }

    async function saveSentences() {
        try {
            const res = await fetch('/api/save', { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                showToast(`💾 Saved ${data.count} sentence(s)`);
            }
        } catch (err) {
            showToast('❌ Failed to save');
        }
    }

    async function toggleDetection() {
        try {
            const res = await fetch('/api/toggle_detection', { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                showToast(data.detection_active ? '▶️ Detection resumed' : '⏸️ Detection paused');
            }
        } catch (err) {
            showToast('❌ Failed to toggle');
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────
    function escapeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Event Listeners ─────────────────────────────────────────────────
    btnClear.addEventListener('click', clearSentence);
    btnSave.addEventListener('click', saveSentences);
    btnToggle.addEventListener('click', toggleDetection);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        switch (e.key.toLowerCase()) {
            case 'c': clearSentence(); break;
            case 's': saveSentences(); break;
            case 'p': toggleDetection(); break;
        }
    });

    // ── Start Polling ───────────────────────────────────────────────────
    pollInterval = setInterval(fetchState, 250);  // 4 times per second
    fetchState();  // Initial fetch

})();
