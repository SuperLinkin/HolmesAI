import React from 'react'
import { useQuery } from 'react-query'
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { TrendingUp, DollarSign, Package } from 'lucide-react'
import holmesAPI from '../services/api'

const COLORS = ['#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']

const Analytics = () => {
  const { data: modelInfo, isLoading: modelLoading } = useQuery('modelInfo', holmesAPI.getModelInfo)
  const { data: monitoringStats } = useQuery('monitoringStats', holmesAPI.getMonitoringStats)

  if (modelLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  // Transform category distribution for charts
  const categoryData = monitoringStats?.category_distribution
    ? Object.entries(monitoringStats.category_distribution)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value)
    : []

  // Feature importance data
  const featureData = modelInfo?.feature_importance?.slice(0, 10) || []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-600 mt-2">
          Comprehensive insights into model performance and predictions
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Categories</p>
              <p className="text-3xl font-bold mt-1">{modelInfo?.n_classes || 15}</p>
            </div>
            <div className="p-3 bg-blue-100 rounded-full">
              <Package className="text-blue-600" size={24} />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Model Type</p>
              <p className="text-2xl font-bold mt-1">{modelInfo?.model_type || 'LightGBM'}</p>
            </div>
            <div className="p-3 bg-green-100 rounded-full">
              <TrendingUp className="text-green-600" size={24} />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Target Accuracy</p>
              <p className="text-3xl font-bold mt-1">93%</p>
            </div>
            <div className="p-3 bg-purple-100 rounded-full">
              <DollarSign className="text-purple-600" size={24} />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Category Distribution Bar Chart */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-6">Category Distribution</h2>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={categoryData.slice(0, 8)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} fontSize={12} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Category Distribution Pie Chart */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-6">Category Proportion</h2>
          <ResponsiveContainer width="100%" height={350}>
            <PieChart>
              <Pie
                data={categoryData.slice(0, 8)}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {categoryData.slice(0, 8).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature Importance */}
      {featureData.length > 0 && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-6">Top Feature Importance</h2>
          <div className="space-y-3">
            {featureData.map((item, index) => {
              const maxImportance = Math.max(...featureData.map(f => f.importance))
              const percentage = (item.importance / maxImportance) * 100

              return (
                <div key={item.feature}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center font-semibold text-xs">
                        {index + 1}
                      </span>
                      <span className="text-sm font-medium">{item.feature}</span>
                    </div>
                    <span className="text-sm text-gray-600">{item.importance.toFixed(4)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5 ml-9">
                    <div
                      className="bg-gradient-to-r from-primary-500 to-primary-700 h-2.5 rounded-full transition-all"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Model Classes */}
      {modelInfo?.classes && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-6">Available Categories ({modelInfo.classes.length})</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {modelInfo.classes.map((category) => (
              <div
                key={category}
                className="px-4 py-3 bg-gray-50 border border-gray-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-colors"
              >
                <span className="text-sm font-medium text-gray-700">{category}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* System Performance */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-6">System Performance</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-2">Embedding Dimension</p>
            <p className="text-2xl font-bold text-blue-600">384</p>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-2">Target Latency</p>
            <p className="text-2xl font-bold text-green-600">&lt;200ms</p>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-2">Monthly Capacity</p>
            <p className="text-2xl font-bold text-purple-600">20M</p>
          </div>
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-2">Similarity Threshold</p>
            <p className="text-2xl font-bold text-orange-600">0.82</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analytics
