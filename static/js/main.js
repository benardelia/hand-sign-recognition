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
    initRealtimePoll();
    initSpeechSynthesis();
    initClipboard();
    initDatasetForm();
    initModelTrainer();
    initLandmarkPlayer();
    initCameraSelector();
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
        subtitleEl.textContent = 'Live camera translation powered by MediaPipe Holistic and LSTM Neural Networks.';
        // Load active feed and turn off hidden feed
        document.getElementById('webcam-feed-translator').src = '/video_feed';
        document.getElementById('webcam-feed-builder').src = '';
    } else if (tabId === 'builder') {
        titleEl.textContent = 'Dataset Builder';
        subtitleEl.textContent = 'Collect new sign landmark sequences to train the model.';
        // Load active feed and turn off hidden feed
        document.getElementById('webcam-feed-builder').src = '/video_feed';
        document.getElementById('webcam-feed-translator').src = '';
        fetchSigns();
    } else if (tabId === 'trainer') {
        titleEl.textContent = 'LSTM Model Trainer';
        subtitleEl.textContent = 'Train your custom neural network with live analytics outputs.';
        // Stop camera feed to save CPU/GPU during training
        document.getElementById('webcam-feed-translator').src = '';
        document.getElementById('webcam-feed-builder').src = '';
    } else if (tabId === 'player') {
        titleEl.textContent = 'Visual Landmark Player';
        subtitleEl.textContent = 'Verify and replay collected skeletal landmark sequence files (.npy).';
        document.getElementById('webcam-feed-translator').src = '';
        document.getElementById('webcam-feed-builder').src = '';
        fetchNpyFilesList();
    }
    
    activeTab = tabId;
}

// --- REAL-TIME POLLING ---
function initRealtimePoll() {
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
    
    if (data.model_loaded) {
        modelBadge.innerHTML = `<i class="fa-solid fa-circle-nodes"></i> Holistic Active`;
        document.getElementById('stat-model-loaded').textContent = 'LOADED (Active)';
        document.getElementById('stat-model-loaded').className = 'highlighted-text';
    } else {
        modelBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation" style="color: var(--yellow)"></i> No Model`;
        document.getElementById('stat-model-loaded').textContent = 'NOT TRAINED';
        document.getElementById('stat-model-loaded').className = 'text-danger';
    }
    
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
    
    // 3. Manage states of recording
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
                const percent = (data.recording_frames_collected / 30) * 100;
                rFill.style.width = `${percent}%`;
                rFill.style.background = 'linear-gradient(90deg, var(--pink), var(--primary))';
                rNums.textContent = `Frame ${data.recording_frames_collected} / 30`;
            } else if (data.recording_status === 'saving' || data.recording_status === 'saved') {
                rTitle.textContent = 'SEQUENCE SAVED!';
                rFill.style.width = '100%';
                rFill.style.background = 'var(--green)';
                rNums.textContent = 'Saving dataset...';
            }
        }
    } else {
        statusDot.className = 'dot pulse green';
        statusText.textContent = 'Webcam Active';
        
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
        fetch('/api/clear_sentence', { method: 'POST' })
            .then(res => res.json())
            .then(() => {
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
            <td>${sign.samples} sequence files</td>
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
        </div>
    `).join('');
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
        const epochs = parseInt(epochInput.value) || 150;
        const batch = parseInt(batchInput.value) || 32;
        
        // Show training info box
        document.getElementById('training-progress-info').classList.remove('hidden');
        document.getElementById('total-epochs').textContent = epochs;
        
        // Clear terminal
        const term = document.getElementById('training-terminal-console');
        term.innerHTML = '<div class="term-line system">[System] Contacting backend to initiate model training...</div>';
        
        // Reset chart data
        resetChartData();
        
        fetch('/api/train', { method: 'POST' })
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
            
            // Reload overall metrics
            fetchSigns();
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
            document.getElementById('active-file-badge').textContent = 'Holistic (30 Frames)';
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
    fetch('/api/delete_sign', {
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
    fetch('/api/delete_file', {
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

// --- CAMERA SOURCE SELECTOR ---
function initCameraSelector() {
    fetchAvailableCameras();
}

function fetchAvailableCameras() {
    const select = document.getElementById('camera-source-select');
    if (!select) return;
    
    fetch('/api/cameras')
        .then(res => res.json())
        .then(data => {
            select.innerHTML = data.cameras.map(cam => `
                <option value="${cam.id}" ${cam.active ? 'selected' : ''}>
                    ${cam.name}
                </option>
            `).join('');
        })
        .catch(err => console.error('Error fetching available cameras:', err));
}

function changeCameraSource(cameraId) {
    const select = document.getElementById('camera-source-select');
    if (select) select.disabled = true; // Temporary disable to prevent spam
    
    speakText("Switching camera.");
    
    fetch('/api/select_camera', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: parseInt(cameraId) })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            console.log(data.message);
            // Refresh feeds by reloading sources if they are currently streaming
            const transFeed = document.getElementById('webcam-feed-translator');
            const buildFeed = document.getElementById('webcam-feed-builder');
            
            if (transFeed && transFeed.src.includes('/video_feed')) {
                transFeed.src = '/video_feed?' + new Date().getTime();
            }
            if (buildFeed && buildFeed.src.includes('/video_feed')) {
                buildFeed.src = '/video_feed?' + new Date().getTime();
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
        fetchAvailableCameras(); // Re-sync state
    });
}
