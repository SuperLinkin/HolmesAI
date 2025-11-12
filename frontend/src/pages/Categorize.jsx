import React, { useState } from 'react'
import { useMutation } from 'react-query'
import { Tag, Upload, AlertCircle } from 'lucide-react'
import holmesAPI from '../services/api'

const Categorize = () => {
  const [transaction, setTransaction] = useState({
    transaction_id: '',
    merchant: '',
    amount: '',
    date: new Date().toISOString().slice(0, 16),
    channel: 'in-store',
    location: ''
  })

  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const categorizeMutation = useMutation(
    holmesAPI.categorizeTransaction,
    {
      onSuccess: (data) => {
        setResult(data)
        setError(null)
      },
      onError: (err) => {
        setError(err.response?.data?.detail || 'Failed to categorize transaction')
        setResult(null)
      }
    }
  )

  const handleSubmit = (e) => {
    e.preventDefault()

    // Validate
    if (!transaction.merchant || !transaction.amount) {
      setError('Merchant and amount are required')
      return
    }

    categorizeMutation.mutate({
      ...transaction,
      amount: parseFloat(transaction.amount),
      transaction_id: transaction.transaction_id || `TXN${Date.now()}`
    })
  }

  const handleReset = () => {
    setTransaction({
      transaction_id: '',
      merchant: '',
      amount: '',
      date: new Date().toISOString().slice(0, 16),
      channel: 'in-store',
      location: ''
    })
    setResult(null)
    setError(null)
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-50 border-green-200'
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    return 'text-red-600 bg-red-50 border-red-200'
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Categorize Transaction</h1>
        <p className="text-gray-600 mt-2">
          Enter transaction details to get AI-powered category predictions
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="card">
          <div className="flex items-center gap-2 mb-6">
            <Tag className="text-primary-600" size={24} />
            <h2 className="text-xl font-semibold">Transaction Details</h2>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Transaction ID (Optional)
              </label>
              <input
                type="text"
                value={transaction.transaction_id}
                onChange={(e) => setTransaction({ ...transaction, transaction_id: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="Auto-generated if empty"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Merchant <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={transaction.merchant}
                onChange={(e) => setTransaction({ ...transaction, merchant: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="e.g., STARBUCKS STORE #12345"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Amount <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                step="0.01"
                value={transaction.amount}
                onChange={(e) => setTransaction({ ...transaction, amount: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="0.00"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Date & Time
              </label>
              <input
                type="datetime-local"
                value={transaction.date}
                onChange={(e) => setTransaction({ ...transaction, date: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Channel
              </label>
              <select
                value={transaction.channel}
                onChange={(e) => setTransaction({ ...transaction, channel: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="in-store">In-Store</option>
                <option value="online">Online</option>
                <option value="mobile">Mobile</option>
                <option value="phone">Phone</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Location
              </label>
              <input
                type="text"
                value={transaction.location}
                onChange={(e) => setTransaction({ ...transaction, location: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                placeholder="e.g., Seattle, WA"
              />
            </div>

            <div className="flex gap-3 pt-4">
              <button
                type="submit"
                disabled={categorizeMutation.isLoading}
                className="flex-1 btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {categorizeMutation.isLoading ? 'Categorizing...' : 'Categorize'}
              </button>
              <button
                type="button"
                onClick={handleReset}
                className="px-6 btn-secondary"
              >
                Reset
              </button>
            </div>
          </form>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <AlertCircle className="text-red-600 mt-0.5" size={20} />
                <div>
                  <h3 className="font-semibold text-red-900">Error</h3>
                  <p className="text-sm text-red-700 mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}

          {result && (
            <>
              {/* Main Result */}
              <div className="card bg-gradient-to-br from-primary-50 to-blue-50 border-2 border-primary-200">
                <div className="text-center">
                  <p className="text-sm text-gray-600 mb-2">Predicted Category</p>
                  <h2 className="text-3xl font-bold text-gray-900 mb-4">
                    {result.category}
                  </h2>
                  <div className={`inline-flex items-center px-4 py-2 rounded-full border ${getConfidenceColor(result.confidence)}`}>
                    <span className="font-semibold">
                      {(result.confidence * 100).toFixed(1)}% Confidence
                    </span>
                  </div>
                  <div className="mt-4 pt-4 border-t border-primary-200">
                    <p className="text-xs text-gray-600">
                      Processing Time: <span className="font-semibold">{result.processing_time_ms?.toFixed(1)}ms</span>
                    </p>
                  </div>
                </div>
              </div>

              {/* Probability Breakdown */}
              <div className="card">
                <h3 className="text-lg font-semibold mb-4">Probability Breakdown</h3>
                <div className="space-y-3">
                  {Object.entries(result.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
                    .map(([category, probability]) => (
                      <div key={category}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-medium text-gray-700">
                            {category}
                          </span>
                          <span className="text-sm text-gray-600">
                            {(probability * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full transition-all"
                            style={{ width: `${probability * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>

              {/* Low Confidence Warning */}
              {result.confidence < 0.7 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-start gap-2">
                    <AlertCircle className="text-yellow-600 mt-0.5" size={20} />
                    <div>
                      <h3 className="font-semibold text-yellow-900">Low Confidence</h3>
                      <p className="text-sm text-yellow-700 mt-1">
                        This prediction has low confidence. Consider reviewing manually or providing feedback.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {!result && !error && (
            <div className="card bg-gray-50 text-center py-12">
              <Upload className="mx-auto text-gray-400 mb-4" size={48} />
              <h3 className="text-lg font-medium text-gray-600 mb-2">
                No Results Yet
              </h3>
              <p className="text-sm text-gray-500">
                Enter transaction details and click "Categorize" to see predictions
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Categorize
