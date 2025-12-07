import axios from 'axios';

// API base URL
const API_BASE_URL = 'http://localhost:8000';

// Create axios instance
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'multipart/form-data',
    },
});

/**
 * Predict the class of an X-ray image
 * @param {File} imageFile - The image file to classify
 * @returns {Promise<{prediction: string, confidence: number}>}
 */
export const predictImage = async (imageFile) => {
    const formData = new FormData();
    formData.append('file', imageFile);

    try {
        const response = await api.post('/predict', formData);
        return response.data;
    } catch (error) {
        if (error.response) {
            // Server responded with error
            throw new Error(error.response.data.detail || 'Prediction failed');
        } else if (error.request) {
            // Request made but no response
            throw new Error('No response from server. Make sure the backend is running.');
        } else {
            // Something else happened
            throw new Error('Error making request: ' + error.message);
        }
    }
};

/**
 * Check if the API server is healthy
 * @returns {Promise<object>}
 */
export const checkHealth = async () => {
    try {
        const response = await api.get('/health');
        return response.data;
    } catch (error) {
        throw new Error('Backend server is not responding');
    }
};
