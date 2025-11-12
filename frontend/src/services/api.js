import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const holmesAPI = {
  // Health check
  getHealth: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Categorize single transaction
  categorizeTransaction: async (transaction) => {
    const response = await apiClient.post('/categorize', transaction);
    return response.data;
  },

  // Categorize batch
  categorizeBatch: async (transactions) => {
    const response = await apiClient.post('/categorize/batch', { transactions });
    return response.data;
  },

  // Submit feedback
  submitFeedback: async (feedback) => {
    const response = await apiClient.post('/feedback', feedback);
    return response.data;
  },

  // Get feedback stats
  getFeedbackStats: async () => {
    const response = await apiClient.get('/feedback/stats');
    return response.data;
  },

  // Get monitoring stats
  getMonitoringStats: async () => {
    const response = await apiClient.get('/monitoring/stats');
    return response.data;
  },

  // Get model info
  getModelInfo: async () => {
    const response = await apiClient.get('/model/info');
    return response.data;
  },

  // Get taxonomy
  getTaxonomy: async () => {
    const response = await apiClient.get('/taxonomy');
    return response.data;
  },
};

export default holmesAPI;
