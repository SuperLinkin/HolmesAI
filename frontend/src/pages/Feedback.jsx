import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { MessageSquare, Send, CheckCircle } from 'lucide-react'
import holmesAPI from '../services/api'

const Feedback = () => {
  const queryClient = useQueryClient()
  const [formData, setFormData] = useState({
    transaction_id: '',
    merchant: '',
    predicted_category: '',
    corrected_category: '',
    confidence: '',
    notes: ''
  })
  const [submitted, setSubmitted] = useState(false)

  const { data: feedbackStats, isLoading } = useQuery(
    'feedbackStats',
    holmesAPI.getFeedbackStats
  )

  const { data: taxonomy } = useQuery('taxonomy', holmesAPI.getTaxonomy)

  const submitMutation = useMutation(
    holmesAPI.submitFeedback,
    {
      onSuccess: () => {
        queryClient.invalidateQueries('feedbackStats')
        setSubmitted(true)
        setTimeout(() => {
          setFormData({
            transaction_id: '',
            merchant: '',
            predicted_category: '',
            corrected_category: '',
            confidence: '',
            notes: ''
          })
          setSubmitted(false)
        }, 3000)
      }
    }
  )

  const handleSubmit = (e) => {
    e.preventDefault()
    submitMutation.mutate({
      ...formData,
      confidence: parseFloat(formData.confidence)
    })
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  const categories = taxonomy?.categories?.map(c => c.name) || []

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Feedback Management</h1>
        <p className="text-gray-600 mt-2">
          Submit corrections to improve model accuracy through continuous learning
        </p>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card bg-blue-50 border-blue-200">
          <p className="text-sm text-gray-600 mb-1">Total Corrections</p>
          <p className="text-3xl font-bold text-blue-900">
            {feedbackStats?.total_corrections?.toLocaleString() || '0'}
          </p>
        </div>
        <div className="card bg-green-50 border-green-200">
          <p className="text-sm text-gray-600 mb-1">Avg Original Confidence</p>
          <p className="text-3xl font-bold text-green-900">
            {((feedbackStats?.avg_original_confidence || 0) * 100).toFixed(1)}%
          </p>
        </div>
        <div className="card bg-purple-50 border-purple-200">
          <p className="text-sm text-gray-600 mb-1">Next Retraining</p>
          <p className="text-lg font-bold text-purple-900">
            {feedbackStats?.total_corrections >= 100 ? 'Ready' : `${100 - (feedbackStats?.total_corrections || 0)} more needed`}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Feedback Form */}
        <div className="card">
          <div className="flex items-center gap-2 mb-6">
            <MessageSquare className="text-primary-600" size={24} />
            <h2 className="text-xl font-semibold">Submit Feedback</h2>
          </div>

          {submitted && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
              <div className="flex items-center gap-2">
                <CheckCircle className="text-green-600" size={20} />
                <p className="text-green-800 font-medium">Feedback submitted successfully!</p>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Transaction ID <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={formData.transaction_id}
                onChange={(e) => setFormData({ ...formData, transaction_id: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Merchant <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={formData.merchant}
                onChange={(e) => setFormData({ ...formData, merchant: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Predicted Category <span className="text-red-500">*</span>
              </label>
              <select
                value={formData.predicted_category}
                onChange={(e) => setFormData({ ...formData, predicted_category: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                required
              >
                <option value="">Select category...</option>
                {categories.map(cat => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Corrected Category <span className="text-red-500">*</span>
              </label>
              <select
                value={formData.corrected_category}
                onChange={(e) => setFormData({ ...formData, corrected_category: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                required
              >
                <option value="">Select category...</option>
                {categories.map(cat => (
                  <option key={cat} value={cat}>{cat}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Original Confidence <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={formData.confidence}
                onChange={(e) => setFormData({ ...formData, confidence: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="0.85"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Notes (Optional)
              </label>
              <textarea
                value={formData.notes}
                onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                rows="3"
                placeholder="Additional context or explanation..."
              />
            </div>

            <button
              type="submit"
              disabled={submitMutation.isLoading}
              className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <Send size={18} />
              {submitMutation.isLoading ? 'Submitting...' : 'Submit Feedback'}
            </button>
          </form>
        </div>

        {/* Most Corrected Categories */}
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">Most Corrected Categories</h3>
            {feedbackStats?.most_corrected_categories ? (
              <div className="space-y-3">
                {Object.entries(feedbackStats.most_corrected_categories)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 10)
                  .map(([category, count], index) => (
                    <div key={category} className="flex items-center gap-3">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center font-semibold text-sm">
                        {index + 1}
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-medium">{category}</span>
                          <span className="text-sm text-gray-500">{count} corrections</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-yellow-500 h-2 rounded-full"
                            style={{
                              width: `${(count / Math.max(...Object.values(feedbackStats.most_corrected_categories))) * 100}%`
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">No correction data available</p>
            )}
          </div>

          {/* Retraining Info */}
          <div className="card bg-gradient-to-br from-purple-50 to-blue-50 border-2 border-purple-200">
            <h3 className="text-lg font-semibold mb-3">Continuous Learning</h3>
            <div className="space-y-2 text-sm text-gray-700">
              <p>✓ Feedback improves accuracy by 3-5% per quarter</p>
              <p>✓ Automatic retraining with 100+ corrections</p>
              <p>✓ Minimum 7 days between training cycles</p>
            </div>
            <div className="mt-4 pt-4 border-t border-purple-200">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Progress to retraining</span>
                <span className="text-sm font-semibold">
                  {Math.min((feedbackStats?.total_corrections || 0), 100)}/100
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
                <div
                  className="bg-purple-600 h-3 rounded-full transition-all"
                  style={{ width: `${Math.min(((feedbackStats?.total_corrections || 0) / 100) * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Feedback
