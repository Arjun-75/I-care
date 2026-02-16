 const form = document.getElementById('uploadForm');
        const octPreview = document.getElementById('octPreview');
        const fundusPreview = document.getElementById('fundusPreview');
        const resultContainer = document.getElementById('resultContainer');
        const predictionResult = document.getElementById('predictionResult');
        const confidenceBar = document.getElementById('confidenceBar');
        const modelInfo = document.getElementById('modelInfo');
        const diseaseExplanation = document.getElementById('diseaseExplanation');
        const errorMessage = document.getElementById('errorMessage');

        // Handle image previews
        document.getElementById('octFile').addEventListener('change', function(e) {
            handleImagePreview(e.target.files[0], octPreview);
        });

        document.getElementById('fundusFile').addEventListener('change', function(e) {
            handleImagePreview(e.target.files[0], fundusPreview);
        });

        function handleImagePreview(file, previewContainer) {
            previewContainer.innerHTML = '';
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    const previewDiv = document.createElement('div');
                    previewDiv.className = 'image-preview';
                    previewDiv.appendChild(img);
                    previewContainer.appendChild(previewDiv);
                }
                reader.readAsDataURL(file);
            }
        }

        // Handle form submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            errorMessage.textContent = '';
            resultContainer.style.display = 'none';

            const formData = new FormData();
            const octFile = document.getElementById('octFile').files[0];
            const fundusFile = document.getElementById('fundusFile').files[0];

            if (!octFile && !fundusFile) {
                showError('Please upload at least one image');
                return;
            }

            if (octFile) formData.append('oct_file', octFile);
            if (fundusFile) formData.append('fundus_file', fundusFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                    return;
                }

                displayResults(data);
            } catch (err) {
                showError('An error occurred during analysis. Please try again.');
            }
        });

        function displayResults(data) {
            resultContainer.style.display = 'block';
            const isNormal = data.prediction === 'normal';
            const confidence = parseFloat(data.confidence);

            predictionResult.innerHTML = `
                <div class="prediction-badge ${isNormal ? 'badge-normal' : 'badge-disease'}">
                    ${data.prediction.toUpperCase()} ${!isNormal ? 'Detected' : ''}
                </div>
                <h2>${isNormal ? 'No Issues Found' : data.prediction}</h2>
            `;

            confidenceBar.style.width = `${confidence}%`;
            modelInfo.textContent = `Analysis based on: ${data.used_models.join(' + ')} model(s)`;
            diseaseExplanation.textContent = data.explanation;

            // Display image previews if available
            if (data.image_urls.oct) {
                octPreview.innerHTML = `<div class="image-preview">
                    <img src="${data.image_urls.oct}" alt="OCT Scan">
                </div>`;
            }
            if (data.image_urls.fundus) {
                fundusPreview.innerHTML = `<div class="image-preview">
                    <img src="${data.image_urls.fundus}" alt="Fundus Image">
                </div>`;
            }
        }

        function showError(message) {
            errorMessage.textContent = message;
            setTimeout(() => errorMessage.textContent = '', 5000);
        }