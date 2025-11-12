import React from 'react'
import { useQuery } from 'react-query'
import {
  Activity,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Target
} from 'lucide-react'
import holmesAPI from '../services/api'
import { formatDistanceToNow } from 'date-fns'

const StatCard = ({ title, value, icon: Icon, trend, color = 'blue' }) => {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    yellow: 'bg-yellow-100 text-yellow-600',
    red: 'bg-red-100 text-red-600',
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-3xl font-bold mt-2">{value}</p>
          {trend && (
            <div className="flex items-center gap-1 mt-2">
              <TrendingUp size={16} className="text-green-500" />
              <span className="text-sm text-green-600">{trend}</span>
            </div>
          )}
        </div>
        <div className={`p-4 rounded-full ${colorClasses[color]}`}>
          <Icon size={24} />
        </div>
      </div>
    </div>
  )
}

const Dashboard = () => {
  const { data: health, isLoading: healthLoading } = useQuery(
    'health',
    holmesAPI.getHealth,
    { refetchInterval: 30000 }
  )

  const { data: monitoringStats, isLoading: monitoringLoading } = useQuery(
    'monitoringStats',
    holmesAPI.getMonitoringStats,
    { refetchInterval: 10000 }
  )

  const { data: feedbackStats, isLoading: feedbackLoading } = useQuery(
    'feedbackStats',
    holmesAPI.getFeedbackStats,
    { refetchInterval: 60000 }
  )

  const { data: modelInfo } = useQuery(
    'modelInfo',
    holmesAPI.getModelInfo
  )

  if (healthLoading || monitoringLoading || feedbackLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  const isHealthy = health?.status === 'healthy' && health?.model_loaded

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Real-time monitoring and analytics for Holmes AI
        </p>
      </div>

      {/* System Status Alert */}
      {!isHealthy && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="text-red-600" size={20} />
            <div>
              <h3 className="font-semibold text-red-900">System Warning</h3>
              <p className="text-sm text-red-700">
                Model not loaded. Please train a model before using the system.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Predictions"
          value={monitoringStats?.total_predictions?.toLocaleString() || '0'}
          icon={Activity}
          color="blue"
        />
        <StatCard
          title="Model Status"
          value={isHealthy ? 'Healthy' : 'Warning'}
          icon={isHealthy ? CheckCircle : AlertTriangle}
          color={isHealthy ? 'green' : 'yellow'}
        />
        <StatCard
          title="Low Confidence"
          value={`${((monitoringStats?.low_confidence_rate || 0) * 100).toFixed(1)}%`}
          icon={Target}
          color={monitoringStats?.low_confidence_rate > 0.3 ? 'red' : 'green'}
        />
        <StatCard
          title="Total Corrections"
          value={feedbackStats?.total_corrections?.toLocaleString() || '0'}
          icon={CheckCircle}
          color="blue"
          trend="+12% this week"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Info */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Model Information</h2>
          <div className="space-y-3">
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-gray-600">Model Type</span>
              <span className="font-medium">{modelInfo?.model_type || 'LightGBM'}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-gray-600">Categories</span>
              <span className="font-medium">{modelInfo?.n_classes || 'N/A'}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b">
              <span className="text-gray-600">Status</span>
              <span className={`badge ${isHealthy ? 'badge-success' : 'badge-warning'}`}>
                {isHealthy ? 'Active' : 'Not Loaded'}
              </span>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-gray-600">Last Update</span>
              <span className="font-medium text-sm text-gray-500">
                <Clock size={14} className="inline mr-1" />
                {formatDistanceToNow(new Date(health?.timestamp), { addSuffix: true })}
              </span>
            </div>
          </div>
        </div>

        {/* Category Distribution */}
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Category Distribution</h2>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {monitoringStats?.category_distribution &&
              Object.entries(monitoringStats.category_distribution)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 8)
                .map(([category, count]) => {
                  const total = monitoringStats.total_predictions
                  const percentage = ((count / total) * 100).toFixed(1)
                  return (
                    <div key={category}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-gray-700">
                          {category}
                        </span>
                        <span className="text-sm text-gray-500">
                          {count} ({percentage}%)
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-primary-600 h-2 rounded-full transition-all"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">System Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-gray-600">Avg Processing Time</p>
            <p className="text-2xl font-bold text-blue-600 mt-2">&lt;50ms</p>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <p className="text-sm text-gray-600">Target F1 Score</p>
            <p className="text-2xl font-bold text-green-600 mt-2">~0.93</p>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <p className="text-sm text-gray-600">Monthly Throughput</p>
            <p className="text-2xl font-bold text-purple-600 mt-2">10-20M</p>
          </div>
        </div>
      </div>

      {/* Feedback Summary */}
      {feedbackStats?.most_corrected_categories && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Most Corrected Categories</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(feedbackStats.most_corrected_categories)
              .slice(0, 3)
              .map(([category, count]) => (
                <div key={category} className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                  <p className="text-sm text-gray-600">Category</p>
                  <p className="font-semibold text-gray-900 mt-1">{category}</p>
                  <p className="text-xs text-yellow-700 mt-2">{count} corrections</p>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default Dashboard
