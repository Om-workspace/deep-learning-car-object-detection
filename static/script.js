const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const loader = document.getElementById('loader');
const resultCard = document.getElementById('resultCard');
const resultLabel = document.getElementById('resultLabel');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');

let selectedFile = null;

// Event Listeners for UI & File input
browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});
dropzone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
});

dropzone.addEventListener('drop', e => {
    let dt = e.dataTransfer;
    let files = dt.files;
    handleFiles(files);
});

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        selectedFile = files[0];
        
        if (!selectedFile.type.startsWith('image/')) {
            alert('Please select a valid image file. (JPG, PNG, JPEG)');
            return;
        }

        // Preview the image for the user
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            predictBtn.classList.remove('hidden');
            
            // Reset previous prediction results
            resultCard.classList.add('hidden');
            confidenceFill.style.width = '0%';
            predictBtn.textContent = 'Analyze Image';
        }
        reader.readAsDataURL(selectedFile);
    }
}

predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Build form data representation
    const formData = new FormData();
    formData.append('file', selectedFile);

    // Switch UI to loading state
    predictBtn.classList.add('hidden');
    loader.classList.remove('hidden');
    resultCard.classList.add('hidden');

    try {
        // Send to our Flask Backend Endpoint
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            resetUI();
            return;
        }

        displayResult(data);
    } catch (error) {
        alert('An server error occurred during prediction. Check your network or Flask backend.');
        console.error(error);
        resetUI();
    }
});

function displayResult(data) {
    // Hide loader, show results
    loader.classList.add('hidden');
    resultCard.classList.remove('hidden');
    
    // Switch button to allow trying again quickly
    predictBtn.classList.remove('hidden');
    predictBtn.textContent = 'Analyze Another Image';

    resultLabel.textContent = data.label;
    
    // Dynamically style colors based on AI outcome
    if (data.is_car) {
        resultLabel.className = 'success-text';
        confidenceFill.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
    } else {
        resultLabel.className = 'danger-text';
        confidenceFill.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
    }

    // Convert decimal probability into UI percentage
    const percent = Math.floor(data.confidence * 100);
    
    // Smooth progress bar fill animation
    setTimeout(() => {
        confidenceFill.style.width = `${percent}%`;
    }, 100);

    // Smooth number counting animation
    animateCounter(confidenceText, 0, percent, 1500);
}

function animateCounter(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        // ease out effect
        const currentCount = Math.floor(progress * (end - start) + start);
        obj.innerHTML = currentCount + '%';
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function resetUI() {
    loader.classList.add('hidden');
    predictBtn.classList.remove('hidden');
}

// ----------------------------------------
// Load Interactive Accuracy Chart
// ----------------------------------------
async function loadMetricsChart() {
    try {
        const response = await fetch('/static/metrics.json');
        if (!response.ok) return;
        const data = await response.json();
        
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        
        // Define Custom Gradient for Area Fills
        const adamGradient = ctx.createLinearGradient(0, 0, 0, 400);
        adamGradient.addColorStop(0, 'rgba(0, 242, 254, 0.4)');
        adamGradient.addColorStop(1, 'rgba(0, 242, 254, 0.0)');

        const sgdGradient = ctx.createLinearGradient(0, 0, 0, 400);
        sgdGradient.addColorStop(0, 'rgba(16, 185, 129, 0.4)');
        sgdGradient.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

        const rmsGradient = ctx.createLinearGradient(0, 0, 0, 400);
        rmsGradient.addColorStop(0, 'rgba(245, 158, 11, 0.4)');
        rmsGradient.addColorStop(1, 'rgba(245, 158, 11, 0.0)');

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.epochs.map(e => 'Epoch ' + e),
                datasets: [
                    {
                        label: 'Adam',
                        data: data.adam,
                        borderColor: '#00f2fe',
                        backgroundColor: adamGradient,
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: '#00f2fe'
                    },
                    {
                        label: 'SGD',
                        data: data.sgd,
                        borderColor: '#10b981',
                        backgroundColor: sgdGradient,
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: '#10b981'
                    },
                    {
                        label: 'RMSprop',
                        data: data.rmsprop,
                        borderColor: '#f59e0b',
                        backgroundColor: rmsGradient,
                        borderWidth: 3,
                        tension: 0.4,
                        fill: true,
                        pointBackgroundColor: '#f59e0b'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#ffffff', font: { family: 'Outfit', size: 14 } }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0,0,0,0.8)'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Validation Accuracy', color: '#fff' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    } catch (e) {
        console.log("No metrics data available yet.", e);
    }
}

document.addEventListener('DOMContentLoaded', loadMetricsChart);
