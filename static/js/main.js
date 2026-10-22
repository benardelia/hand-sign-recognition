// Global State
let activeTab = 'overview';
let statusInterval = null;
let chartInstance = null;
let signsData = [];
let wasRecording = false; // Track recording state transitions

// Landmark Player state
let playerData = null;
let playerFrameIdx = 0;
let isPlaying = false;
let playInterval = null;
let isLooping = true;

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initClock();
    fetchSigns();
    initSpeechSynthesis();
    initClipboard();
    initDatasetForm();
    initModelTrainer();
    initLandmarkPlayer();
    initMode();
});

// --- CLOCK COMPONENT ---
function initClock() {
    const timeEl = document.getElementById('live-time');
    const update = () => {
        const d = new Date();
        timeEl.textContent = d.toLocaleTimeString();
    };
    update();
    setInterval(update, 1000);
}

// --- TAB SYSTEM ---
function initTabs() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });
}

function switchTab(tabId) {
    if (activeTab === tabId) return;
    
    // Manage active navigation item
    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('data-tab') === tabId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Manage active panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        if (panel.id === `tab-${tabId}`) {
            panel.classList.add('active');
        } else {
            panel.classList.remove('active');
        }
    });
    
    // Update headers
    const titleEl = document.getElementById('current-tab-title');
    const subtitleEl = document.getElementById('current-tab-subtitle');
    
    if (tabId === 'overview') {
        titleEl.textContent = 'Dashboard Overview';
        subtitleEl.textContent = 'Real-time status, labels inventory, and models overview.';
        fetchSigns();
    } else if (tabId === 'translator') {
        titleEl.textContent = 'Live Sign Translator';
        subtitleEl.textContent = 'Live camera translation powered by MediaPipe Holistic and on-device sized models.';
    } else if (tabId === 'builder') {
        titleEl.textContent = 'Dataset Builder';
        subtitleEl.textContent = 'Collect new sign landmark sequences to train the model.';
        fetchSigns();
    } else if (tabId === 'trainer') {
        titleEl.textContent = 'Model Trainer';
        subtitleEl.textContent = 'Train a quality profile (low / high / experimental) with live analytics outputs.';
    } else if (tabId === 'player') {
        titleEl.textContent = 'Visual Landmark Player';
        subtitleEl.textContent = 'Verify and replay collected skeletal landmark sequence files (.npy).';
        fetchNpyFilesList();
    }
    
    activeTab = tabId;
    if (tabId === 'translator' || tabId === 'builder') {
        activateFeed(tabId);
    } else {
        deactivateFeeds();
    }
}

// --- REAL-TIME POLLING ---
function initRealtimePoll() {
    // Local camera mode only: the server owns the camera and the translation state.
    statusInterval = setInterval(() => {
        // Poll status only if we are in translator or builder tabs
        if (activeTab !== 'translator' && activeTab !== 'builder') return;
        
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                updateUIState(data);
            })
            .catch(err => console.error('Error polling status:', err));
    }, 150);
}

function updateUIState(data) {
    // 1. Update active model badge
    const modelBadge = document.getElementById('active-model-name');
    const statusText = document.getElementById('sys-status-text');
    const statusDot = document.getElementById('sys-status-dot');
    
    const profileName = (data.profile || activeProfile || '').toUpperCase();
    modelBadge.textContent = data.model_loaded ? `${profileName} model` : `${profileName}: no model`;
    modelBadge.style.color = data.model_loaded ? '' : 'var(--yellow)';
    
    // 2. Manage states of translation screen
    if (activeTab === 'translator') {
        // Sentence overlay
        const sentenceEl = document.getElementById('active-sentence');
        sentenceEl.textContent = data.current_sentence;
        
        // Active predicted gloss word
        const wordEl = document.getElementById('pred-word');
        wordEl.textContent = data.last_prediction ? data.last_prediction : '-';
        
        // Confidence bar & text
        const confidenceBar = document.getElementById('pred-confidence-bar');
        const confidenceText = document.getElementById('pred-confidence-text');
        const confPercent = Math.round(data.confidence * 100);
        confidenceBar.style.width = `${confPercent}%`;
        confidenceText.textContent = `${confPercent}%`;
        
        // Gloss bubbles queue
        const glossContainer = document.getElementById('active-gloss-bubbles');
        if (data.gloss_buffer && data.gloss_buffer.length > 0) {
            glossContainer.innerHTML = data.gloss_buffer.map(word => 
                `<span class="gloss-bubble">${word}</span>`
            ).join('');
        } else {
            glossContainer.innerHTML = `<span class="text-muted small">Empty buffer</span>`;
        }
    }
    
    renderRecordingState(data);
}

function renderRecordingState(data) {
    const statusText = document.getElementById('sys-status-text');
    const statusDot = document.getElementById('sys-status-dot');
    const startRecordBtn = document.getElementById('start-record-btn');
    if (data.is_recording) {
        statusDot.className = 'dot pulse yellow';
        statusText.textContent = `Recording '${data.recording_label}'`;
        
        if (startRecordBtn) {
            startRecordBtn.disabled = true;
            startRecordBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Recording '${data.recording_label}'...`;
        }
        
        if (activeTab === 'builder') {
            const rBox = document.getElementById('record-status-box');
            const rTitle = document.getElementById('record-status-title');
            const rFill = document.getElementById('record-status-fill');
            const rNums = document.getElementById('record-status-numbers');
            
            rBox.classList.remove('hidden');
            
            if (data.recording_status === 'countdown') {
                rTitle.textContent = 'GET READY (Countdown)';
                rFill.style.width = '100%';
                rFill.style.background = 'var(--yellow)';
                rNums.textContent = 'Preparing...';
            } else if (data.recording_status === 'recording') {
                rTitle.textContent = 'CAPTURING LANDMARKS';
                const percent = Math.round((data.recording_progress || 0) * 100);
                rFill.style.width = `${percent}%`;
                rFill.style.background = 'linear-gradient(90deg, var(--pink), var(--primary))';
                rNums.textContent = data.recording_frames_collected
                    ? `${data.recording_frames_collected} frames captured`
                    : `${percent}%`;
            } else if (data.recording_status === 'saving') {
                rTitle.textContent = 'UPLOADING SEQUENCE...';
                rFill.style.width = '100%';
                rFill.style.background = 'var(--primary)';
                rNums.textContent = 'Saving dataset...';
            } else if (data.recording_status === 'saved') {
                rTitle.textContent = 'SEQUENCE SAVED!';
                rFill.style.width = '100%';
                rFill.style.background = 'var(--green)';
                rNums.textContent = data.recording_frames_collected ? `${data.recording_frames_collected} frames` : 'Saved';
            } else if (data.recording_status === 'error') {
                rTitle.textContent = 'SAMPLE REJECTED';
                rFill.style.width = '100%';
                rFill.style.background = 'var(--red, #ef4444)';
                rNums.textContent = data.recording_error || 'Upload failed';
            }
        }
    } else {
        if (wasRecording) setSystemStatus('green', cameraStatusText());
        
        if (startRecordBtn) {
            startRecordBtn.disabled = false;
            startRecordBtn.innerHTML = `<i class="fa-solid fa-record-vinyl"></i> Initiate Sequence Capture`;
        }
        
        if (activeTab === 'builder') {
            document.getElementById('record-status-box').classList.add('hidden');
        }
    }

    // Auto-refresh dataset signs list when recording transitions from true to false
    if (wasRecording && !data.is_recording) {
        console.log("Recording completed. Auto-refreshing signs list...");
        fetchSigns();
    }
    wasRecording = data.is_recording;
}

// --- TEXT-TO-SPEECH (TTS) & ACTIONS ---
function initSpeechSynthesis() {
    const speakBtn = document.getElementById('speak-sentence-btn');
    speakBtn.addEventListener('click', () => {
        const sentence = document.getElementById('active-sentence').textContent;
        if (sentence && sentence !== 'Start signing to reconstruct sentences...' && sentence !== 'Sentence Cleared.') {
            const utterance = new SpeechSynthesisUtterance(sentence);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
            logReconstruction(`Spoke sentence: "${sentence}"`);
        }
    });
}

function initClipboard() {
    const copyBtn = document.getElementById('copy-sentence-btn');
    copyBtn.addEventListener('click', () => {
        const sentence = document.getElementById('active-sentence').textContent;
        if (sentence && sentence !== 'Start signing to reconstruct sentences...' && sentence !== 'Sentence Cleared.') {
            navigator.clipboard.writeText(sentence)
                .then(() => {
                    const originalHTML = copyBtn.innerHTML;
                    copyBtn.innerHTML = `<i class="fa-solid fa-check green"></i> <span>Copied!</span>`;
                    logReconstruction(`Copied sentence to clipboard.`);
                    setTimeout(() => {
                        copyBtn.innerHTML = originalHTML;
                    }, 2000);
                })
                .catch(err => console.error('Copy failed:', err));
        }
    });
    
    const clearBtn = document.getElementById('clear-sentence-btn');
    clearBtn.addEventListener('click', () => {
        const req = appMode === 'local'
            ? fetch('/api/clear_sentence', { method: 'POST' })
            : fetch('/api/session/clear', { method: 'POST', headers: { 'X-Session-Id': sessionId, 'X-Profile': activeProfile } });
        req.then(res => res.json())
            .then(() => {
                pendingFrames = [];
                document.getElementById('active-sentence').textContent = 'Sentence Cleared.';
                logReconstruction('Cleared sentence buffer.');
            });
    });
}

function logReconstruction(msg) {
    const logsContainer = document.getElementById('reconstruction-logs');
    const timeStr = new Date().toLocaleTimeString();
    const item = document.createElement('div');
    item.className = 'log-item translation';
    item.innerHTML = `<span class="text-muted">[${timeStr}]</span> ${msg}`;
    logsContainer.appendChild(item);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// --- DATASET MANAGEMENT ---
function fetchSigns() {
    fetch('/api/signs')
        .then(res => res.json())
        .then(data => {
            signsData = data.signs;
            updateOverviewStats();
            renderSignsGrid();
            renderOverviewTable();
        })
        .catch(err => console.error('Error fetching signs:', err));
}

function updateOverviewStats() {
    const totalSigns = signsData.length;
    const totalSamples = signsData.reduce((acc, sign) => acc + sign.samples, 0);
    
    document.getElementById('stat-total-signs').textContent = totalSigns;
    document.getElementById('stat-total-samples').textContent = totalSamples;
    
    // Inventory table counters
    document.getElementById('signs-inventory-count').textContent = `${totalSigns} Signs`;
}

function renderOverviewTable() {
    const tbody = document.getElementById('signs-inventory-tbody');
    if (signsData.length === 0) {
        tbody.innerHTML = `<tr><td colspan="3" class="text-center text-muted">No signs collected yet. Go to 'Dataset Builder'.</td></tr>`;
        return;
    }
    
    tbody.innerHTML = signsData.map(sign => `
        <tr>
            <td><strong class="highlighted-text">${sign.name}</strong></td>
            <td>${sign.samples} sequence files<br><span class="text-muted small">${formatSources(sign.sources)}</span></td>
            <td>
                <div class="table-actions">
                    <button class="btn btn-secondary btn-icon" onclick="switchTab('player')" title="Inspect Landmarks">
                        <i class="fa-solid fa-play"></i>
                    </button>
                    <button class="btn btn-danger btn-icon" onclick="confirmDeleteSign('${sign.name}', event)" title="Delete entire sign dataset">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </div>
            </td>
        </tr>
    `).join('');
}

function renderSignsGrid() {
    const grid = document.getElementById('signs-grid-container');
    if (signsData.length === 0) {
        grid.innerHTML = `<div class="text-center text-muted col-span-all">No collected signs found. Record some sequences above!</div>`;
        return;
    }
    
    grid.innerHTML = signsData.map(sign => `
        <div class="sign-grid-item">
            <button class="btn-delete-sign" onclick="confirmDeleteSign('${sign.name}', event)" title="Delete entire sign dataset">
                <i class="fa-solid fa-trash-can"></i>
            </button>
            <div class="sign-name">${sign.name}</div>
            <div class="sign-count"><i class="fa-solid fa-folder-open"></i> ${sign.samples} samples</div>
            <div class="sign-sources text-muted small">${formatSources(sign.sources)}</div>
        </div>
    `).join('');
}

// "web 12 · phone 3" - where the samples of a sign were recorded (dataset balance check).
function formatSources(sources) {
    if (!sources) return '';
    return Object.entries(sources).map(([src, n]) => `${src} ${n}`).join(' · ');
}

function initDatasetForm() {
    const form = document.getElementById('record-sign-form');
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const labelInput = document.getElementById('label-input');
        const label = labelInput.value.trim().toUpperCase();
        if (!label) return;
        
        // Sound prompt: countdown starts
        speakText("Preparing recording.");
        
        if (appMode === 'browser') {
            startBrowserRecording(label);
            return;
        }
        
        fetch('/api/start_recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: label })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                console.log(data.message);
            } else {
                alert(data.message);
            }
        })
        .catch(err => console.error('Error starting recording:', err));
        
        // Removed: labelInput.value = ''; to allow consecutive recordings for the same label
    });
}

function speakText(text) {
    if ('speechSynthesis' in window) {
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1.2;
        window.speechSynthesis.speak(u);
    }
}

// --- MODEL TRAINING LOGIC & PLOTTING ---
function initModelTrainer() {
    const trainBtn = document.getElementById('start-train-btn');
    const epochInput = document.getElementById('train-epochs');
    const batchInput = document.getElementById('train-batch');
    
    // Initialize Chart.js
    initTrainingChart();
    
    trainBtn.addEventListener('click', () => {
        const epochs = parseInt(epochInput.value) || null;
        const batch = parseInt(batchInput.value) || null;
        const profile = document.getElementById('train-profile').value || activeProfile;
        
        // Show training info box
        document.getElementById('training-progress-info').classList.remove('hidden');
        document.getElementById('total-epochs').textContent = epochs || 'profile default';
        
        // Clear terminal
        const term = document.getElementById('training-terminal-console');
        term.innerHTML = '<div class="term-line system">[System] Contacting backend to initiate model training...</div>';
        
        // Reset chart data
        resetChartData();
        
        apiFetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile, epochs, batch_size: batch })
        })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    // Connect SSE log stream
                    connectTrainingLogsStream();
                } else {
                    term.innerHTML += `<div class="term-line text-danger">[Error] ${data.message}</div>`;
                }
            })
            .catch(err => {
                term.innerHTML += `<div class="term-line text-danger">[Error] ${err}</div>`;
            });
    });
}

function connectTrainingLogsStream() {
    const term = document.getElementById('training-terminal-console');
    const sse = new EventSource('/stream_train_logs');
    
    sse.onmessage = (e) => {
        const data = JSON.parse(e.data);
        
        if (data.type === 'log') {
            const line = document.createElement('div');
            line.className = 'term-line';
            line.textContent = data.message;
            term.appendChild(line);
            term.scrollTop = term.scrollHeight;
        } 
        else if (data.type === 'epoch_metrics') {
            // A new profile started ('all' trains several in a row)
            if (data.epoch === 1) resetChartData();
            // Update stats indicators
            document.getElementById('current-epoch').textContent = data.epoch;
            document.getElementById('mini-loss').textContent = data.loss.toFixed(4);
            document.getElementById('mini-acc').textContent = `${(data.accuracy * 100).toFixed(1)}%`;
            
            // Push to Chart.js
            updateChartData(data.epoch, data.accuracy, data.loss, data.val_accuracy, data.val_loss);
            
            const line = document.createElement('div');
            line.className = 'term-line highlight';
            line.textContent = `[Epoch ${data.epoch}] Loss: ${data.loss.toFixed(4)} | Acc: ${data.accuracy.toFixed(4)} | Val Loss: ${data.val_loss.toFixed(4)} | Val Acc: ${data.val_accuracy.toFixed(4)}`;
            term.appendChild(line);
            term.scrollTop = term.scrollHeight;
            
            // Text to speech when completed or at intervals
            if (data.epoch % 25 === 0) {
                speakText(`Epoch ${data.epoch} complete.`);
            }
        } 
        else if (data.type === 'status') {
            sse.close();
            const line = document.createElement('div');
            line.className = data.code === 0 ? 'term-line highlight' : 'term-line text-danger';
            line.style.fontWeight = 'bold';
            line.textContent = `[Process Exit] ${data.message}`;
            term.appendChild(line);
            term.scrollTop = term.scrollHeight;
            
            speakText(data.code === 0 ? "Model training complete!" : "Model training failed.");
            
            // Reload overall metrics and the trained-profile list
            fetchSigns();
            refreshConfig();
        }
    };
    
    sse.onerror = (err) => {
        console.error('SSE Error:', err);
        sse.close();
        term.innerHTML += `<div class="term-line text-danger">[System] SSE stream connection lost.</div>`;
    };
}

function initTrainingChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Acc',
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    data: [],
                    yAxisID: 'y'
                },
                {
                    label: 'Val Acc',
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    borderWidth: 2,
                    data: [],
                    yAxisID: 'y'
                },
                {
                    label: 'Train Loss',
                    borderColor: '#ec4899',
                    backgroundColor: 'rgba(236, 72, 153, 0.1)',
                    borderWidth: 2,
                    data: [],
                    yAxisID: 'y1'
                },
                {
                    label: 'Val Loss',
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    data: [],
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    title: { display: true, text: 'Accuracy', color: '#9ca3af' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: { display: true, text: 'Loss', color: '#9ca3af' },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#9ca3af' }
                },
                x: {
                    ticks: { color: '#9ca3af' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f3f4f6' }
                }
            }
        }
    });
}

function updateChartData(epoch, acc, loss, val_acc, val_loss) {
    if (!chartInstance) return;
    chartInstance.data.labels.push(epoch);
    chartInstance.data.datasets[0].data.push(acc);
    chartInstance.data.datasets[1].data.push(val_acc);
    chartInstance.data.datasets[2].data.push(loss);
    chartInstance.data.datasets[3].data.push(val_loss);
    chartInstance.update();
}

function resetChartData() {
    if (!chartInstance) return;
    chartInstance.data.labels = [];
    chartInstance.data.datasets.forEach(dataset => dataset.data = []);
    chartInstance.update();
}

// --- LANDMARK SKELETON PLAYER ---
function initLandmarkPlayer() {
    const playBtn = document.getElementById('player-play-btn');
    const prevBtn = document.getElementById('player-prev-btn');
    const nextBtn = document.getElementById('player-next-btn');
    const loopBtn = document.getElementById('player-loop-btn');
    const scrubber = document.getElementById('player-scrubber');
    
    // Playback events
    playBtn.addEventListener('click', togglePlayback);
    prevBtn.addEventListener('click', prevFrame);
    nextBtn.addEventListener('click', nextFrame);
    loopBtn.addEventListener('click', toggleLoop);
    
    scrubber.addEventListener('input', (e) => {
        playerFrameIdx = parseInt(e.target.value);
        drawActiveFrame();
    });
}

function togglePlayback() {
    const playBtn = document.getElementById('player-play-btn');
    if (isPlaying) {
        clearInterval(playInterval);
        isPlaying = false;
        playBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
    } else {
        if (!playerData) return;
        isPlaying = true;
        playBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
        playInterval = setInterval(() => {
            playerFrameIdx++;
            if (playerFrameIdx >= playerData.frames.length) {
                if (isLooping) {
                    playerFrameIdx = 0;
                } else {
                    togglePlayback(); // Pause
                    playerFrameIdx = playerData.frames.length - 1;
                }
            }
            document.getElementById('player-scrubber').value = playerFrameIdx;
            drawActiveFrame();
        }, 65); // ~15 FPS playback speed for detailed review
    }
}

function toggleLoop() {
    const loopBtn = document.getElementById('player-loop-btn');
    isLooping = !isLooping;
    if (isLooping) {
        loopBtn.classList.add('toggle-active');
    } else {
        loopBtn.classList.remove('toggle-active');
    }
}

function prevFrame() {
    if (!playerData) return;
    if (isPlaying) togglePlayback();
    playerFrameIdx = (playerFrameIdx - 1 + playerData.frames.length) % playerData.frames.length;
    document.getElementById('player-scrubber').value = playerFrameIdx;
    drawActiveFrame();
}

function nextFrame() {
    if (!playerData) return;
    if (isPlaying) togglePlayback();
    playerFrameIdx = (playerFrameIdx + 1) % playerData.frames.length;
    document.getElementById('player-scrubber').value = playerFrameIdx;
    drawActiveFrame();
}

function fetchNpyFilesList() {
    const container = document.getElementById('player-files-accordion');
    container.innerHTML = '<div class="text-center text-muted">Loading skeletal files database...</div>';
    
    fetch('/api/npy_files')
        .then(res => res.json())
        .then(data => {
            renderFilesAccordion(data);
        })
        .catch(err => {
            console.error('Error fetching file list:', err);
            container.innerHTML = '<div class="text-center text-danger">Failed to load files data.</div>';
        });
}

function renderFilesAccordion(data) {
    const container = document.getElementById('player-files-accordion');
    const keys = Object.keys(data);
    
    if (keys.length === 0) {
        container.innerHTML = '<div class="text-center text-muted">No landmarks data folders found. Record signs first!</div>';
        return;
    }
    
    container.innerHTML = keys.map((label, idx) => `
        <div class="accordion-group">
            <div class="accordion-header" onclick="toggleAccordion(this, 'acc-content-${idx}')">
                <span>${label} (${data[label].length} samples)</span>
                <i class="fa-solid fa-chevron-right"></i>
            </div>
            <div class="accordion-content" id="acc-content-${idx}">
                ${data[label].map(file => `
                    <div class="file-item-container">
                        <div class="file-item" data-path="${file.path}" onclick="loadNpySequence('${file.path}', this)">
                            ${file.filename}
                        </div>
                        <button class="btn-delete-file" onclick="confirmDeleteFile('${file.path}', '${file.filename}', event)" title="Delete sample">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

function toggleAccordion(header, contentId) {
    header.classList.toggle('active');
    const content = document.getElementById(contentId);
    content.classList.toggle('active');
}

function loadNpySequence(filepath, element) {
    // Manage active visual state in accordion list
    document.querySelectorAll('.file-item-container').forEach(el => el.classList.remove('active'));
    
    const container = element.closest('.file-item-container');
    if (container) {
        container.classList.add('active');
    }
    
    if (isPlaying) togglePlayback();
    
    fetch(`/api/npy_data?path=${encodeURIComponent(filepath)}`)
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            
            playerData = data;
            playerData.filepath = filepath;
            playerFrameIdx = 0;
            
            document.getElementById('active-file-title').textContent = data.filename;
            const source = data.meta && data.meta.source ? ` · ${data.meta.source}` : '';
            document.getElementById('active-file-badge').textContent = `Holistic (${data.frames.length} frames${source})`;
            document.getElementById('player-scrubber').max = data.frames.length - 1;
            document.getElementById('player-scrubber').value = 0;
            
            drawActiveFrame();
        })
        .catch(err => console.error('Error loading landmark details:', err));
}

// Custom 2D Skeleton Canvas Painter
function drawActiveFrame() {
    if (!playerData) return;
    
    const canvas = document.getElementById('skeleton-canvas');
    const ctx = canvas.getContext('2d');
    const frame = playerData.frames[playerFrameIdx];
    
    // Clear canvas with dark futuristic style background
    ctx.fillStyle = '#080610';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Scale and translate coordinates to fill the screen
    // Since coordinates are normal relative values (0.0 to 1.0)
    const scaleX = canvas.width;
    const scaleY = canvas.height;
    
    // Frame counter overlay updates
    document.getElementById('player-frame-overlay').textContent = `Frame: ${playerFrameIdx + 1} / ${playerData.frames.length}`;
    
    // 1. Draw FACE landmarks
    if (frame.face && frame.face.length > 0) {
        ctx.fillStyle = 'rgba(167, 139, 250, 0.25)'; // Purple-mesh points
        
        // Draw selected facial outline connections or cloud of dots to prevent lag
        // Draw dots at a smaller density for visual efficiency
        for (let i = 0; i < frame.face.length; i += 4) { // Sample every 4th point
            const pt = frame.face[i];
            const px = pt.x * scaleX;
            const py = pt.y * scaleY;
            
            ctx.beginPath();
            ctx.arc(px, py, 1, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // Highlight specific facial outline curves for realism
        const lipsIndices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];
        ctx.strokeStyle = 'rgba(236, 72, 153, 0.45)'; // Pink contour lips
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < lipsIndices.length; i++) {
            const pt = frame.face[lipsIndices[i]];
            if (pt) {
                const px = pt.x * scaleX;
                const py = pt.y * scaleY;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
        }
        ctx.closePath();
        ctx.stroke();
    }
    
    // 2. Draw POSE skeleton
    if (frame.pose && frame.pose.length > 0) {
        // Line styling
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.7)'; // Indigo joints
        ctx.lineWidth = 2.5;
        
        // Standard pose connections
        const connections = [
            [11, 12], [11, 13], [13, 15], // Left arm
            [12, 14], [14, 16], // Right arm
            [11, 23], [12, 24], [23, 24]  // Torso hips
        ];
        
        connections.forEach(([sIdx, eIdx]) => {
            const pt1 = frame.pose[sIdx];
            const pt2 = frame.pose[eIdx];
            
            if (pt1 && pt2 && pt1.visibility > 0.4 && pt2.visibility > 0.4) {
                ctx.beginPath();
                ctx.moveTo(pt1.x * scaleX, pt1.y * scaleY);
                ctx.lineTo(pt2.x * scaleX, pt2.y * scaleY);
                ctx.stroke();
            }
        });
        
        // Draw joints as glowing dots
        ctx.fillStyle = 'rgba(99, 102, 241, 0.9)';
        [11, 12, 13, 14, 15, 16, 23, 24].forEach(idx => {
            const pt = frame.pose[idx];
            if (pt && pt.visibility > 0.4) {
                ctx.beginPath();
                ctx.arc(pt.x * scaleX, pt.y * scaleY, 4, 0, 2 * Math.PI);
                ctx.fill();
            }
        });
    }
    
    // 3. Draw HANDS skeletons
    const handConnections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index
        [5, 9], [9, 10], [10, 11], [11, 12], // Middle
        [9, 13], [13, 14], [14, 15], [15, 16], // Ring
        [13, 17], [17, 18], [18, 19], [19, 20], [0, 17] // Pinky & Palm
    ];
    
    // Draw Left Hand (cyan glow)
    if (frame.left_hand && frame.left_hand.length > 0 && frame.left_hand[0].x !== 0.0) {
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
        ctx.lineWidth = 2;
        handConnections.forEach(([s, e]) => {
            const p1 = frame.left_hand[s];
            const p2 = frame.left_hand[e];
            if (p1 && p2) {
                ctx.beginPath();
                ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
                ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
                ctx.stroke();
            }
        });
        
        ctx.fillStyle = '#06b6d4';
        frame.left_hand.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * scaleX, pt.y * scaleY, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }
    
    // Draw Right Hand (green glow)
    if (frame.right_hand && frame.right_hand.length > 0 && frame.right_hand[0].x !== 0.0) {
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
        ctx.lineWidth = 2;
        handConnections.forEach(([s, e]) => {
            const p1 = frame.right_hand[s];
            const p2 = frame.right_hand[e];
            if (p1 && p2) {
                ctx.beginPath();
                ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
                ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
                ctx.stroke();
            }
        });
        
        ctx.fillStyle = '#10b981';
        frame.right_hand.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x * scaleX, pt.y * scaleY, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }
}

// --- DATASET & FILE DELETION HELPERS ---
function confirmDeleteSign(label, event) {
    if (event) event.stopPropagation();
    
    const uppercaseConfirmation = prompt(
        `WARNING: This will permanently delete the entire sign dataset for "${label}" (including all recorded sequence files).\n\nTo confirm deletion, please type the sign name in UPPERCASE letters below:`
    );
    
    if (uppercaseConfirmation === null) {
        return; // User cancelled
    }
    
    if (uppercaseConfirmation.trim().toUpperCase() !== label.toUpperCase()) {
        alert("Verification failed. The sign dataset was not deleted.");
        return;
    }
    
    // Call delete sign endpoint
    apiFetch('/api/delete_sign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: label })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            speakText(`Deleted ${label} dataset.`);
            
            // Check if active player sequence is part of this deleted dataset
            if (playerData && playerData.filepath) {
                const normalizedFilepath = playerData.filepath.replace(/\\/g, '/');
                const searchPattern = `/Holistic_Landmarks/${label}/`;
                const altSearchPattern = `/${label}/`;
                if (normalizedFilepath.includes(searchPattern) || normalizedFilepath.includes(altSearchPattern)) {
                    // Reset player
                    playerData = null;
                    if (isPlaying) togglePlayback();
                    document.getElementById('active-file-title').textContent = "No Sequence Selected";
                    document.getElementById('active-file-badge').textContent = "-";
                    // Clear skeleton canvas
                    const canvas = document.getElementById('skeleton-canvas');
                    if (canvas) {
                        const ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                    }
                    document.getElementById('player-frame-overlay').textContent = "Frame: 0/30";
                }
            }
            
            // Refresh signs and file list
            fetchSigns();
            fetchNpyFilesList();
        } else {
            alert(data.message || "Failed to delete the sign dataset.");
        }
    })
    .catch(err => {
        console.error('Error deleting sign:', err);
        alert('An error occurred while deleting the sign dataset.');
    });
}

function confirmDeleteFile(filepath, filename, event) {
    if (event) event.stopPropagation();
    
    if (!confirm(`Are you sure you want to permanently delete the recording "${filename}"?`)) {
        return; // User cancelled
    }
    
    // Call delete file endpoint
    apiFetch('/api/delete_file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: filepath })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            // Check if this was the active playing file
            if (playerData && playerData.filepath === filepath) {
                // Reset player
                playerData = null;
                if (isPlaying) togglePlayback();
                document.getElementById('active-file-title').textContent = "No Sequence Selected";
                document.getElementById('active-file-badge').textContent = "-";
                // Clear skeleton canvas
                const canvas = document.getElementById('skeleton-canvas');
                if (canvas) {
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
                document.getElementById('player-frame-overlay').textContent = "Frame: 0/30";
            }
            
            // Refresh data list
            fetchNpyFilesList();
            // Also refresh stats (since count changed)
            fetchSigns();
        } else {
            alert(data.message || "Failed to delete the file.");
        }
    })
    .catch(err => {
        console.error('Error deleting file:', err);
        alert('An error occurred while deleting the file.');
    });
}

// --- MODE, QUALITY PROFILE & ACCESS TOKEN ---
// 'browser': landmarks are extracted in this page (static/js/pipeline.js) and only numbers are
//            sent to the server. Works on any HTTPS deployment and for many users at once.
// 'local'  : the Flask host's own webcam (LOCAL_CAMERA=1), streamed back as MJPEG.
let appConfig = null;
let appMode = 'browser';
let activeProfile = null;
let pipeline = null;
let pipelineLoading = null;
let selectedDeviceId = '';
let releaseTimer = null;
// MediaPipe Holistic holds a few hundred MB (GPU textures + graph). Free it when no camera
// tab has been open for this long; coming back reloads it from the browser cache.
const RELEASE_AFTER_MS = 30000;
let pendingFrames = [];
let isSendingFrames = false;
const sessionId = getSessionId();
const recorder = { status: 'idle', label: '', frames: [], times: [], start: 0, countdownEnd: 0, error: '' };
const MIN_RECORD_FRAMES = 8;

function getSessionId() {
    let id = null;
    try { id = sessionStorage.getItem('handsign-session'); } catch (e) { /* storage unavailable */ }
    if (!id) {
        id = window.crypto && crypto.randomUUID
            ? crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
        try { sessionStorage.setItem('handsign-session', id); } catch (e) { /* storage unavailable */ }
    }
    return id;
}

function storedValue(key, fallback) {
    try { return localStorage.getItem(key) || fallback; } catch (e) { return fallback; }
}

function storeValue(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* storage unavailable */ }
}

// fetch() that sends the optional access token and asks for it once when the server answers 401.
async function apiFetch(url, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    const token = storedValue('handsign-token', '');
    if (token) headers['X-Handsign-Token'] = token;
    let res = await fetch(url, { ...options, headers });
    if (res.status === 401) {
        const entered = prompt('This server requires an access token for this action:');
        if (entered) {
            storeValue('handsign-token', entered.trim());
            headers['X-Handsign-Token'] = entered.trim();
            res = await fetch(url, { ...options, headers });
        }
    }
    return res;
}

async function initMode() {
    try {
        appConfig = await (await fetch('/api/config')).json();
    } catch (e) {
        console.error('Failed to load /api/config', e);
        setSystemStatus('red', 'Server unreachable');
        return;
    }
    appMode = appConfig.local_camera ? 'local' : 'browser';
    activeProfile = storedValue('handsign-profile', appConfig.default_profile);
    if (!appConfig.profiles[activeProfile]) activeProfile = appConfig.default_profile;
    renderProfileControls();

    if (appMode === 'local') {
        initRealtimePoll();
        fetchAvailableCameras();
    } else {
        await listBrowserCameras();
    }
    setSystemStatus('yellow', 'Camera idle');
    setInterval(updateFpsStat, 1000);
    if (activeTab === 'translator' || activeTab === 'builder') activateFeed(activeTab);
}

async function refreshConfig() {
    try {
        appConfig = await (await fetch('/api/config')).json();
        renderProfileControls();
    } catch (e) {
        console.error('Failed to refresh config', e);
    }
}

function setSystemStatus(color, text) {
    const dot = document.getElementById('sys-status-dot');
    const label = document.getElementById('sys-status-text');
    if (dot) dot.className = `dot pulse ${color}`;
    if (label) label.textContent = text;
}

function cameraStatusText() {
    if (appMode === 'local') return 'Local camera';
    return pipeline && pipeline.running ? `Browser camera (${pipeline.delegate})` : 'Camera idle';
}

function profileLabel(name, p) {
    return `${name.toUpperCase()} · ${p.camera.width}x${p.camera.height} @ ${p.camera.fps}fps${p.trained ? '' : ' (untrained)'}`;
}

function renderProfileControls() {
    const select = document.getElementById('profile-select');
    if (select) {
        select.innerHTML = Object.entries(appConfig.profiles).map(([name, p]) =>
            `<option value="${name}" ${name === activeProfile ? 'selected' : ''}>${profileLabel(name, p)}</option>`
        ).join('');
        select.onchange = () => changeProfile(select.value);
    }

    const trainSelect = document.getElementById('train-profile');
    if (trainSelect) {
        const current = trainSelect.value || activeProfile;
        trainSelect.innerHTML = Object.keys(appConfig.profiles).map(name =>
            `<option value="${name}" ${name === current ? 'selected' : ''}>${name.toUpperCase()}</option>`
        ).join('') + `<option value="all" ${current === 'all' ? 'selected' : ''}>ALL PROFILES</option>`;
    }

    const trained = Object.values(appConfig.profiles).filter(p => p.trained).length;
    const stat = document.getElementById('stat-model-loaded');
    if (stat) {
        stat.textContent = `${trained} / ${Object.keys(appConfig.profiles).length} trained`;
        stat.className = trained ? 'highlighted-text' : 'text-danger';
    }

    const tbody = document.getElementById('models-tbody');
    if (tbody) {
        tbody.innerHTML = Object.entries(appConfig.profiles).map(([name, p]) => {
            const m = p.metrics || {};
            const acc = m.tflite_val_accuracy ?? m.val_accuracy;
            const size = m.tflite_bytes ? `${(m.tflite_bytes / 1024).toFixed(0)} KB tflite` : (p.target === 'desktop' ? 'Keras (PC)' : '-');
            return `<tr>
                <td><strong class="highlighted-text">${name.toUpperCase()}</strong><br><span class="text-muted small">${p.description}</span></td>
                <td>${p.trained ? `v${p.version}` : '<span class="text-danger">not trained</span>'}</td>
                <td>${acc !== undefined ? `${(acc * 100).toFixed(0)}%` : '-'}</td>
                <td>${size}</td>
            </tr>`;
        }).join('');
    }
    updateModelBadge();
}

function updateModelBadge() {
    const badge = document.getElementById('active-model-name');
    const p = appConfig && appConfig.profiles[activeProfile];
    if (!badge || !p) return;
    badge.textContent = p.trained ? `${activeProfile.toUpperCase()} model` : `${activeProfile.toUpperCase()}: not trained`;
    badge.style.color = p.trained ? '' : 'var(--yellow)';
}

function updateFpsStat() {
    const el = document.getElementById('stat-fps');
    if (!el) return;
    if (appMode === 'browser' && pipeline && pipeline.running) {
        el.textContent = `${Math.round(pipeline.fpsEstimate)} FPS`;
    } else if (appConfig && appConfig.profiles[activeProfile]) {
        el.textContent = `${appConfig.profiles[activeProfile].camera.fps} FPS target`;
    }
}

function changeProfile(name) {
    if (appMode === 'local') {
        alert('Local camera mode uses the server PROFILE setting. Restart the server with PROFILE=' + name + ' to switch.');
        document.getElementById('profile-select').value = activeProfile;
        return;
    }
    activeProfile = name;
    storeValue('handsign-profile', name);
    pendingFrames = [];
    updateModelBadge();
    if (pipeline) {
        pipeline.setCamera(appConfig.profiles[name].camera);
        if (pipeline.stream) restartBrowserCamera();
    }
    logReconstruction(`Quality profile: ${profileLabel(name, appConfig.profiles[name])}`);
}

// --- CAMERA FEEDS ---
function feedElements(tabId) {
    return tabId === 'translator'
        ? { img: document.getElementById('webcam-feed-translator'), container: document.querySelector('#tab-translator .camera-container') }
        : { img: document.getElementById('webcam-feed-builder'), container: document.querySelector('#tab-builder .camera-container-small') };
}

function activateFeed(tabId) {
    if (!appConfig) return; // initMode() activates the feed once the config is loaded
    const { img, container } = feedElements(tabId);
    const other = feedElements(tabId === 'translator' ? 'builder' : 'translator');
    if (appMode === 'local') {
        other.img.src = '';
        img.style.display = '';
        img.src = '/video_feed';
        setSystemStatus('green', 'Local camera');
        return;
    }
    img.style.display = 'none';
    other.img.style.display = 'none';
    clearTimeout(releaseTimer);
    startBrowserPipeline(container);
}

function deactivateFeeds() {
    if (appMode === 'local') {
        document.getElementById('webcam-feed-translator').src = '';
        document.getElementById('webcam-feed-builder').src = '';
        return;
    }
    if (pipeline) {
        pipeline.stop();
        pipeline.stopCamera();
        clearTimeout(releaseTimer);
        releaseTimer = setTimeout(() => {
            if (activeTab !== 'translator' && activeTab !== 'builder') pipeline.release();
        }, RELEASE_AFTER_MS);
    }
    pendingFrames = [];
    setSystemStatus('yellow', 'Camera idle');
}

// Hidden browser tab: turn the camera off (privacy, CPU/GPU); resume when shown again.
document.addEventListener('visibilitychange', () => {
    if (!appConfig || appMode !== 'browser') return;
    if (document.hidden) {
        deactivateFeeds();
    } else if (activeTab === 'translator' || activeTab === 'builder') {
        activateFeed(activeTab);
    }
});

function showFeedMessage(container, message) {
    let msg = container.querySelector('.feed-message');
    if (!msg) {
        msg = document.createElement('div');
        msg.className = 'feed-message';
        container.prepend(msg);
    }
    msg.textContent = message;
    msg.style.display = message ? '' : 'none';
}

async function loadPipeline() {
    setSystemStatus('yellow', 'Loading MediaPipe...');
    if (!window.HandsignPipeline) {
        await new Promise(resolve => window.addEventListener('handsign-pipeline-ready', resolve, { once: true }));
    }
    const p = new window.HandsignPipeline(appConfig.mediapipe);
    await p.load();
    p.onFrame = handleBrowserFrame;
    p.onOverlay = drawRecordingOverlay;
    p.onError = () => {
        setSystemStatus('red', 'Tracking failed (WebGL required)');
        const container = p.canvas.parentElement;
        if (container) showFeedMessage(container, 'Landmark tracking failed. Enable hardware acceleration / WebGL in your browser and reload.');
    };
    pipeline = p;
}

async function startBrowserPipeline(container) {
    if (!window.isSecureContext || !navigator.mediaDevices) {
        setSystemStatus('red', 'Camera Blocked (HTTPS Required)');
        showFeedMessage(container, 'Camera access needs HTTPS (or http://localhost). Open this site over HTTPS.');
        return;
    }
    showFeedMessage(container, 'Loading hand/body tracking model...');
    try {
        if (!pipeline) {
            pipelineLoading = pipelineLoading || loadPipeline().finally(() => { pipelineLoading = null; });
            await pipelineLoading;
        } else if (!pipeline.loaded) {
            setSystemStatus('yellow', 'Loading MediaPipe...');
            pipelineLoading = pipelineLoading || pipeline.load().finally(() => { pipelineLoading = null; });
            await pipelineLoading;
        }
    } catch (e) {
        console.error('MediaPipe failed to load', e);
        pipelineLoading = null;
        setSystemStatus('red', 'Tracking model failed to load');
        showFeedMessage(container, 'Could not load MediaPipe (check your connection) - reload to retry.');
        return;
    }
    // The user may have left the camera tabs while the model was loading.
    if (activeTab !== 'translator' && activeTab !== 'builder') return;

    container.prepend(pipeline.canvas);
    pipeline.setCamera(appConfig.profiles[activeProfile].camera);
    try {
        if (!pipeline.stream) {
            showFeedMessage(container, 'Waiting for camera permission...');
            await pipeline.startCamera(selectedDeviceId);
            await listBrowserCameras();
        }
    } catch (err) {
        console.warn('Could not access browser camera:', err);
        setSystemStatus('red', 'Camera Blocked / Unavailable');
        showFeedMessage(container, 'Camera blocked or unavailable. Allow camera access and reload.');
        return;
    }
    showFeedMessage(container, '');
    pipeline.start();
    setSystemStatus('green', cameraStatusText());
}

async function restartBrowserCamera() {
    if (!pipeline) return;
    try {
        await pipeline.startCamera(selectedDeviceId);
    } catch (err) {
        console.warn('Camera restart failed:', err);
        setSystemStatus('red', 'Camera Blocked / Unavailable');
    }
}

async function listBrowserCameras() {
    const select = document.getElementById('camera-source-select');
    if (!select || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    if (!devices.length) {
        select.innerHTML = '<option value="">Browser camera</option>';
        return;
    }
    const activeId = pipeline && pipeline.stream
        ? (pipeline.stream.getVideoTracks()[0].getSettings().deviceId || selectedDeviceId)
        : selectedDeviceId;
    select.innerHTML = devices.map((d, i) =>
        `<option value="${d.deviceId}" ${d.deviceId === activeId ? 'selected' : ''}>${d.label || `Camera ${i + 1}`}</option>`
    ).join('');
}

function fetchAvailableCameras() {
    const select = document.getElementById('camera-source-select');
    if (!select) return;
    fetch('/api/cameras')
        .then(res => res.json())
        .then(data => {
            select.innerHTML = data.cameras.map(cam =>
                `<option value="${cam.id}" ${cam.active ? 'selected' : ''}>${cam.name}</option>`
            ).join('');
        })
        .catch(err => console.error('Error fetching available cameras:', err));
}

function changeCameraSource(cameraId) {
    if (appMode === 'browser') {
        selectedDeviceId = cameraId;
        if (pipeline && pipeline.stream) restartBrowserCamera();
        return;
    }
    const select = document.getElementById('camera-source-select');
    if (select) select.disabled = true;
    speakText("Switching camera.");
    fetch('/api/select_camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: parseInt(cameraId) })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            for (const id of ['webcam-feed-translator', 'webcam-feed-builder']) {
                const feed = document.getElementById(id);
                if (feed && feed.src.includes('/video_feed')) feed.src = '/video_feed?' + Date.now();
            }
        } else {
            alert(data.message || "Failed to switch camera source.");
        }
    })
    .catch(err => {
        console.error('Error switching camera source:', err);
        alert("An error occurred while switching the camera source.");
    })
    .finally(() => {
        if (select) select.disabled = false;
        fetchAvailableCameras();
    });
}

// --- BROWSER PIPELINE: LIVE TRANSLATION & RECORDING ---
function handleBrowserFrame(frame, ts) {
    if (recorder.status === 'countdown' && ts >= recorder.countdownEnd) {
        recorder.status = 'recording';
        recorder.start = ts;
        recorder.frames = [];
        recorder.times = [];
    }
    if (recorder.status === 'recording') {
        recorder.frames.push(frame);
        recorder.times.push(ts);
        // Slow devices: keep recording until there are enough frames (max 3x the duration).
        const elapsed = ts - recorder.start;
        const target = appConfig.dataset.record_seconds * 1000;
        if ((elapsed >= target && recorder.frames.length >= MIN_RECORD_FRAMES) || elapsed >= 3 * target) {
            finishBrowserRecording();
        }
        renderRecordingState(recorderState());
        return;
    }
    if (recorder.status === 'countdown') {
        renderRecordingState(recorderState());
        return;
    }
    if (activeTab === 'translator') {
        pendingFrames.push(frame);
        if (pendingFrames.length > 60) pendingFrames.splice(0, pendingFrames.length - 60);
        flushFrames();
    }
}

// Frames that arrive while a request is in flight are sent together in the next one.
async function flushFrames() {
    if (isSendingFrames || pendingFrames.length === 0) return;
    isSendingFrames = true;
    const batch = pendingFrames;
    pendingFrames = [];
    const body = new Float32Array(batch.length * 1662);
    batch.forEach((f, i) => body.set(f, i * 1662));
    try {
        const res = await fetch('/api/stream/frames', {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream', 'X-Session-Id': sessionId, 'X-Profile': activeProfile },
            body: body.buffer
        });
        const data = await res.json();
        if (data.status === 'success') updateUIState({ ...data, ...recorderState() });
    } catch (e) {
        console.error('Frame streaming error:', e);
    } finally {
        isSendingFrames = false;
    }
}

function recorderState() {
    return {
        is_recording: recorder.status !== 'idle',
        recording_status: recorder.status,
        recording_label: recorder.label,
        recording_frames_collected: recorder.frames.length,
        recording_progress: recorder.status === 'recording' && recorder.times.length
            ? Math.min(1, (recorder.times[recorder.times.length - 1] - recorder.start) / (appConfig.dataset.record_seconds * 1000))
            : 0,
        recording_error: recorder.error,
    };
}

function startBrowserRecording(label) {
    if (!pipeline || !pipeline.running) {
        alert('The camera is not running yet. Allow camera access and wait for the preview to appear.');
        return;
    }
    if (recorder.status !== 'idle') return;
    Object.assign(recorder, {
        status: 'countdown', label, frames: [], times: [], error: '',
        countdownEnd: performance.now() + appConfig.dataset.countdown_seconds * 1000
    });
    renderRecordingState(recorderState());
}

function float32ToBase64(arr) {
    const bytes = new Uint8Array(arr.buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(binary);
}

async function finishBrowserRecording() {
    const frames = recorder.frames;
    const times = recorder.times;
    if (frames.length < MIN_RECORD_FRAMES) {
        recorder.status = 'error';
        recorder.error = `Only ${frames.length} frames captured - this device is too slow ` +
            `(${Math.round(pipeline.fpsEstimate)} fps). Try the LOW profile.`;
        renderRecordingState(recorderState());
        setTimeout(() => { recorder.status = 'idle'; recorder.frames = []; renderRecordingState(recorderState()); }, 4000);
        return;
    }
    recorder.status = 'saving';
    renderRecordingState(recorderState());

    const buf = new Float32Array(frames.length * 1662);
    frames.forEach((f, i) => buf.set(f, i * 1662));
    const t0 = times[0];
    const duration = (times[times.length - 1] - t0) / 1000;
    const meta = {
        source: 'web',
        profile: activeProfile,
        mirrored: true,
        fps: duration > 0 ? Math.round((frames.length - 1) / duration * 10) / 10 : null,
        timestamps_ms: times.map(t => Math.round((t - t0) * 10) / 10),
        resolution: [pipeline.work.width, pipeline.work.height],
        delegate: pipeline.delegate,
        user_agent: navigator.userAgent.slice(0, 200),
    };
    try {
        const res = await apiFetch('/api/samples', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: recorder.label, frames_b64: float32ToBase64(buf), meta })
        });
        const data = await res.json();
        if (data.status === 'success') {
            recorder.status = 'saved';
            fetchSigns();
        } else {
            recorder.status = 'error';
            recorder.error = data.message;
        }
    } catch (e) {
        recorder.status = 'error';
        recorder.error = String(e);
    }
    renderRecordingState(recorderState());
    setTimeout(() => {
        recorder.status = 'idle';
        recorder.frames = [];
        renderRecordingState(recorderState());
    }, recorder.status === 'error' ? 3000 : 1500);
}

function drawRecordingOverlay(ctx, w, h) {
    if (recorder.status === 'idle') return;
    ctx.save();
    if (recorder.status === 'countdown') {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.fillRect(0, 0, w, h);
        const remaining = (recorder.countdownEnd - performance.now()) / 1000;
        ctx.fillStyle = '#facc15';
        ctx.font = `bold ${Math.round(h / 4)}px Outfit, sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText(remaining > 0 ? String(Math.ceil(remaining)) : 'GO!', w / 2, h / 2 + h / 12);
        ctx.font = `${Math.round(h / 20)}px Outfit, sans-serif`;
        ctx.fillStyle = '#fff';
        ctx.fillText(`Prepare sign: '${recorder.label}'`, w / 2, h - 20);
    } else if (recorder.status === 'recording') {
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 6;
        ctx.strokeRect(0, 0, w, h);
        ctx.fillStyle = '#ef4444';
        ctx.font = `bold ${Math.round(h / 18)}px Outfit, sans-serif`;
        ctx.fillText('RECORDING...', 14, 36);
    } else {
        const ok = recorder.status !== 'error';
        ctx.strokeStyle = ok ? '#10b981' : '#ef4444';
        ctx.lineWidth = 6;
        ctx.strokeRect(0, 0, w, h);
        ctx.fillStyle = ok ? '#10b981' : '#ef4444';
        ctx.font = `bold ${Math.round(h / 14)}px Outfit, sans-serif`;
        ctx.textAlign = 'center';
        const text = { saving: 'UPLOADING...', saved: 'SEQUENCE SAVED!', error: 'REJECTED' }[recorder.status] || '';
        ctx.fillText(text, w / 2, h / 2);
    }
    ctx.restore();
}
