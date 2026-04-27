/**
 * Displays the classification result in the UI.
 * @param {object} data - The prediction data from the server.
 */
function displayPrediction(data) {
    document.getElementById('resultIcon').textContent = data.icon || '🔍';
    document.getElementById('resultTitle').textContent = data.full_name;
    document.getElementById('resultType').textContent = data.class === 'B' ? 'Biodegradable' : 'Non-Biodegradable';
    
    const confidence = Math.round((data.confidence || 0) * 100);
    document.getElementById('confidenceValue').textContent = confidence + '%';
    document.getElementById('confidenceFill').style.width = confidence + '%';
    document.getElementById('resultCard').style.borderTopColor = data.color;

    const fillElement = document.getElementById('confidenceFill');
    if (data.class === 'B') {
        fillElement.style.background = 'linear-gradient(90deg, #4ade80, #22c55e)';
    } else if (data.class === 'N') {
        fillElement.style.background = 'linear-gradient(90deg, #f87171, #dc2626)';
    }
    
    const statusElement = document.getElementById('confidenceStatus');
    if (data.high_confidence) {
        statusElement.className = 'confidence-status high';
        statusElement.innerHTML = '<i class="fas fa-check-circle"></i> High Confidence';
    } else {
        statusElement.className = 'confidence-status low';
        statusElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Low Confidence';
    }
}

/**
 * Shows a preview of the selected image file.
 */
function previewImage() {
    const input = document.getElementById('imageUpload');
    const preview = document.getElementById('imagePreview');
    const testBtn = document.getElementById('testImageBtn');

    if (input.files && input.files[0]) {
        preview.src = URL.createObjectURL(input.files[0]);
        preview.classList.remove('hidden');
        testBtn.classList.remove('hidden');
    }
}

/**
 * Sends the uploaded image to the server for classification.
 * @param {Event} event - The button click event.
 */
function testUploadedImage(event) {
    const btn = event.currentTarget;
    const fileInput = document.getElementById('imageUpload');
    if (!fileInput.files || fileInput.files.length === 0) {
        showToast('Please select an image first.', 'error');
        return;
    }
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Classifying...';

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    fetch('/classify', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayPrediction(data.prediction);
                showToast('Classification complete!');
            } else {
                showToast(`Error: ${data.error}`, 'error');
            }
        })
        .catch(err => {
            console.error('Upload and classify failed:', err);
            showToast('Upload failed. See console for details.', 'error');
        })
        .finally(() => {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-cogs"></i> Classify This Image';
        });
}

/**
 * Shows a toast notification.
 * @param {string} message - The message to display.
 * @param {string} type - 'success', 'error', or 'info'.
 */
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.querySelector('#toastMessage').textContent = message;
    toast.className = `show ${type}`;
    setTimeout(() => { toast.className = toast.className.replace('show', ''); }, 3000);
}

/**
 * Sends a request to shut down the server.
 */
function quitApplication() {
    if (confirm('Are you sure you want to shut down the server? This will close the application.')) {
        fetch('/shutdown', { method: 'POST' })
            .then(() => {
                showToast('Server is shutting down...', 'info');
                document.body.innerHTML = "<h1>Server has been shut down. You can close this window.</h1>";
            })
            .catch(error => {
                showToast('Could not shut down server. Please close the terminal window manually.', 'error');
                console.error('Shutdown failed:', error);
            });
    }
}
