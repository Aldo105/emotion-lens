class VideoUploadManager {
    constructor() {
        this.dropZone = document.getElementById('upload-drop-zone');
        this.fileInput = document.getElementById('upload-file-input');
        this.fileInfo = document.getElementById('upload-file-info');
        this.btnUpload = document.getElementById('btn-upload-video');
        this.uploadProgressContainer = document.getElementById('upload-progress-container');
        this.uploadProgressFill = document.getElementById('upload-progress-fill');
        this.uploadProgressText = document.getElementById('upload-progress-text');
        this.processingContainer = document.getElementById('processing-progress-container');
        this.processingFill = document.getElementById('processing-progress-fill');
        this.processingText = document.getElementById('processing-progress-text');
        this.statusMessage = document.getElementById('upload-status-message');

        this.selectedFile = null;
        this.pollInterval = null;
        this.maxFileSize = 500 * 1024 * 1024; // 500MB

        this.initEvents();
    }

    initEvents() {
        // Drag and drop events
        ['dragenter', 'dragover'].forEach(evt => {
            this.dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.dropZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(evt => {
            this.dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.dropZone.classList.remove('drag-over');
            });
        });

        this.dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length) this.handleFile(files[0]);
        });

        // Click to browse
        this.dropZone.addEventListener('click', () => this.fileInput.click());

        this.fileInput.addEventListener('change', () => {
            if (this.fileInput.files.length) {
                this.handleFile(this.fileInput.files[0]);
            }
        });

        // Upload button
        this.btnUpload.addEventListener('click', () => this.upload());
    }

    handleFile(file) {
        // Validate type
        if (!file.type.startsWith('video/')) {
            this.showStatus('Please select a video file (mp4, webm, mov, etc.).', 'error');
            return;
        }

        // Validate size
        if (file.size > this.maxFileSize) {
            this.showStatus(`File too large. Maximum size is ${this.formatSize(this.maxFileSize)}.`, 'error');
            return;
        }

        this.selectedFile = file;
        this.fileInfo.innerHTML = `
            <div class="file-info-card">
                <span class="file-icon">🎬</span>
                <div class="file-details">
                    <span class="file-name">${this.escapeHtml(file.name)}</span>
                    <span class="file-size">${this.formatSize(file.size)}</span>
                </div>
                <button class="icon-btn file-remove" onclick="videoUploadManager.clearFile(event)">✕</button>
            </div>
        `;
        this.fileInfo.classList.remove('hidden');
        this.btnUpload.classList.remove('hidden');
        this.showStatus('', '');
        this.resetProgress();
    }

    clearFile(e) {
        if (e) e.stopPropagation();
        this.selectedFile = null;
        this.fileInput.value = '';
        this.fileInfo.classList.add('hidden');
        this.btnUpload.classList.add('hidden');
        this.resetProgress();
        this.showStatus('', '');
    }

    resetProgress() {
        this.uploadProgressContainer.classList.add('hidden');
        this.uploadProgressFill.style.width = '0%';
        this.uploadProgressText.textContent = '0%';
        this.processingContainer.classList.add('hidden');
        this.processingFill.style.width = '0%';
        this.processingText.textContent = '0%';
    }

    async upload() {
        if (!this.selectedFile) return;

        this.btnUpload.disabled = true;
        this.btnUpload.textContent = 'Uploading...';
        this.uploadProgressContainer.classList.remove('hidden');
        this.showStatus('Uploading video...', 'info');

        const formData = new FormData();
        formData.append('file', this.selectedFile);

        try {
            // Use XMLHttpRequest for upload progress
            const result = await this.uploadWithProgress(formData);

            this.uploadProgressFill.style.width = '100%';
            this.uploadProgressText.textContent = '100%';
            this.showStatus('Upload complete! Processing video...', 'success');

            // Start polling for processing progress
            const sessionId = result.session_id || result.id;
            if (sessionId) {
                this.startProcessingPoll(sessionId);
            } else {
                this.showStatus('Upload complete!', 'success');
                this.btnUpload.disabled = false;
                this.btnUpload.textContent = '⬆️ Upload Video';
            }
        } catch (err) {
            console.error('Upload failed:', err);
            this.showStatus('Upload failed: ' + err.message, 'error');
            this.btnUpload.disabled = false;
            this.btnUpload.textContent = '⬆️ Upload Video';
        }
    }

    uploadWithProgress(formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const pct = Math.round((e.loaded / e.total) * 100);
                    this.uploadProgressFill.style.width = `${pct}%`;
                    this.uploadProgressText.textContent = `${pct}%`;
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        resolve(JSON.parse(xhr.responseText));
                    } catch {
                        resolve({});
                    }
                } else {
                    reject(new Error(`Server responded with ${xhr.status}`));
                }
            });

            xhr.addEventListener('error', () => reject(new Error('Network error')));
            xhr.addEventListener('abort', () => reject(new Error('Upload aborted')));

            xhr.open('POST', `${CONFIG.API_URL}/videos/upload`);
            xhr.send(formData);
        });
    }

    startProcessingPoll(sessionId) {
        this.processingContainer.classList.remove('hidden');
        this.processingFill.classList.add('processing-pulse');

        this.pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`${CONFIG.API_URL}/videos/${sessionId}/progress`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();

                const progress = data.progress ?? data.percentage ?? 0;
                const pct = Math.round(progress);

                this.processingFill.style.width = `${pct}%`;
                this.processingText.textContent = `${pct}%`;

                if (pct >= 100 || data.status === 'completed') {
                    this.stopProcessingPoll();
                    this.processingFill.classList.remove('processing-pulse');
                    this.showStatus('', '');

                    this.statusMessage.innerHTML = `
                        <div class="upload-success">
                            <span class="success-icon">✅</span>
                            <h3>Video processed successfully!</h3>
                            <p>Your analysis results are ready to view.</p>
                            <button class="btn btn-primary" onclick="videoUploadManager.goToSessions()">
                                View Results →
                            </button>
                        </div>
                    `;

                    this.btnUpload.disabled = false;
                    this.btnUpload.textContent = '⬆️ Upload Video';
                }
            } catch (err) {
                console.error('Progress poll error:', err);
            }
        }, 2000);
    }

    stopProcessingPoll() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    goToSessions() {
        // Switch to sessions tab
        const sessionsTab = document.querySelector('[data-view="view-sessions"]');
        if (sessionsTab) sessionsTab.click();
    }

    showStatus(message, type) {
        if (!message) {
            this.statusMessage.innerHTML = '';
            return;
        }

        const iconMap = {
            'info': 'ℹ️',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️'
        };

        this.statusMessage.innerHTML = `
            <div class="upload-status-msg upload-status-${type}">
                <span>${iconMap[type] || ''} ${message}</span>
            </div>
        `;
    }

    formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    }

    escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}
