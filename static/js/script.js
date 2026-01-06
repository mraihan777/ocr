// Canvas Drawing & Prediction Logic

// Canvas Setup
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize canvas
function initCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

initCanvas();

// Drawing Functions
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.type === 'touchstart') {
        const touch = e.touches[0];
        lastX = (touch.clientX - rect.left) * scaleX;
        lastY = (touch.clientY - rect.top) * scaleY;
    } else {
        lastX = (e.clientX - rect.left) * scaleX;
        lastY = (e.clientY - rect.top) * scaleY;
    }
}

function draw(e) {
    if (!isDrawing) return;

    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let currentX, currentY;

    if (e.type === 'touchmove') {
        const touch = e.touches[0];
        currentX = (touch.clientX - rect.left) * scaleX;
        currentY = (touch.clientY - rect.top) * scaleY;
    } else {
        currentX = (e.clientX - rect.left) * scaleX;
        currentY = (e.clientY - rect.top) * scaleY;
    }

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    lastX = currentX;
    lastY = currentY;
}

function stopDrawing() {
    isDrawing = false;
}

// Event Listeners for Drawing
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// Upload Functionality
const uploadBtn = document.getElementById('uploadBtn');
const imageUpload = document.getElementById('imageUpload');

uploadBtn.addEventListener('click', () => {
    imageUpload.click();
});

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                // Clear canvas
                initCanvas();

                // Calculate scaling to fit image in canvas while maintaining aspect ratio
                const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
                const x = (canvas.width / 2) - (img.width / 2) * scale;
                const y = (canvas.height / 2) - (img.height / 2) * scale;

                // Draw image on canvas
                ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Clear Button
document.getElementById('clearBtn').addEventListener('click', () => {
    initCanvas();
    hideResults();
    hideError();
});

// Predict Button
document.getElementById('predictBtn').addEventListener('click', async () => {
    // Hide previous results
    hideError();

    // Show loading state
    showLoading();

    try {
        // Get canvas image data
        const imageData = canvas.toDataURL('image/png');

        // Send to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Terjadi kesalahan saat prediksi');
        }

    } catch (error) {
        console.error('Error:', error);
        showError('Tidak dapat terhubung ke server. Pastikan Flask server berjalan.');
    }
});

// Display Functions
function showLoading() {
    const resultsContent = document.getElementById('resultsContent');
    resultsContent.innerHTML = `
        <div class="empty-state">
            <svg width="80" height="80" viewBox="0 0 80 80" fill="none" style="animation: spin 1s linear infinite;">
                <circle cx="40" cy="40" r="35" stroke="currentColor" stroke-width="4" opacity="0.2"/>
                <path d="M 40 5 A 35 35 0 0 1 75 40" stroke="url(#gradient2)" stroke-width="4" stroke-linecap="round"/>
                <defs>
                    <linearGradient id="gradient2" x1="40" y1="5" x2="75" y2="40">
                        <stop stop-color="#667eea"/>
                        <stop offset="1" stop-color="#764ba2"/>
                    </linearGradient>
                </defs>
            </svg>
            <p>Memproses...</p>
        </div>
    `;

    // Add spin animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
}

function displayResults(data) {
    // Hide empty state
    document.getElementById('resultsContent').style.display = 'none';

    // Show prediction result
    const predictionResult = document.getElementById('predictionResult');
    predictionResult.style.display = 'block';

    // Update predicted letter
    document.getElementById('predictedLetter').textContent = data.prediction;

    // Update confidence
    const confidencePercent = (data.confidence * 100).toFixed(1);
    document.getElementById('confidence').textContent = `${confidencePercent}% Confidence`;

    // Update top predictions
    const topPredictions = document.getElementById('topPredictions');
    topPredictions.innerHTML = '';

    const predictions = data.top_predictions;
    for (const [letter, probability] of Object.entries(predictions)) {
        const percent = (probability * 100).toFixed(1);
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <div class="prediction-letter">${letter}</div>
            <div class="prediction-bar">
                <div class="bar">
                    <div class="bar-fill" style="width: ${percent}%"></div>
                </div>
            </div>
            <div class="prediction-confidence">${percent}%</div>
        `;
        topPredictions.appendChild(item);
    }
}

function hideResults() {
    document.getElementById('resultsContent').style.display = 'flex';
    document.getElementById('resultsContent').innerHTML = `
        <div class="empty-state">
            <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
                <circle cx="40" cy="40" r="35" stroke="currentColor" stroke-width="2" opacity="0.2"/>
                <path d="M30 45C30 45 35 50 40 50C45 50 50 45 50 45" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                <circle cx="32" cy="32" r="2" fill="currentColor"/>
                <circle cx="48" cy="32" r="2" fill="currentColor"/>
            </svg>
            <p>Belum ada prediksi</p>
            <span>Gambar huruf terlebih dahulu</span>
        </div>
    `;
    document.getElementById('predictionResult').style.display = 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    document.getElementById('errorText').textContent = message;
    errorDiv.style.display = 'flex';

    // Hide results
    hideResults();
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}
