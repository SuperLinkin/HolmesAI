import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { AlertTriangle, TrendingDown, TrendingUp, CheckCircle } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import holmesAPI from '../services/api'

const DriftAlert = ({ detected, metrics }) => {
  if (!detected) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <CheckCircle className="text-green-600" size={24} />
          <div>
            <h3 className="font-semibold text-green-900">No Drift Detected</h3>
            <p className="text-sm text-green-700 mt-1">
              Model performance is stable and within expected thresholds.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
      <div className="flex items-center gap-3">
        <AlertTriangle className="text-red-600" size={24} />
        <div className="flex-1">
          <h3 className="font-semibold text-red-900">⚠️ Model Drift Detected</h3>
          <p className="text-sm text-red-700 mt-1">
            F1 score dropped by {(metrics?.f1_drift * 100).toFixed(2)}% - Consider retraining.
          </p>
          <div className="mt-3 grid grid-cols-2 gap-4 text-xs">
            <div>
              <span className="text-gray-600">Baseline F1:</span>
              <span className="ml-2 font-semibold">{metrics?.baseline_f1?.toFixed(4)}</span>
            </div>
            <div>
              <span className="text-gray-600">Current F1:</span>
              <span className="ml-2 font-semibold">{metrics?.current_f1?.toFixed(4)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

const MetricCard = ({ title, value, change, threshold }) => {
  const isNegative = change < 0
  const exceedsThreshold = Math.abs(change) > threshold

  return (
    <div className="card">
      <h3 className="text-sm font-medium text-gray-600 mb-2">{title}</h3>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-3xl font-bold">{value?.toFixed(4) || '0.0000'}</p>
          <div className="flex items-center gap-2 mt-2">
            {isNegative ? (
              <TrendingDown className={exceedsThreshold ? 'text-red-500' : 'text-yellow-500'} size={16} />
            ) : (
              <TrendingUp className="text-green-500" size={16} />
            )}
            <span className={`text-sm ${exceedsThreshold ? 'text-red-600' : 'text-gray-600'}`}>
              {(change * 100).toFixed(2)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

const Monitoring = () => {
  const [timeRange, setTimeRange] = useState('24h')

  const { data: monitoringStats, isLoading } = useQuery(
    ['monitoring', timeRange],
    holmesAPI.getMonitoringStats,
    { refetchInterval: 10000 }
  )

  // Mock drift data for demonstration
  const driftData = {
    drift_detected: false,
    baseline_f1: 0.9312,
    current_f1: 0.9287,
    f1_drift: -0.0025,
    precision_drift: -0.0018,
    recall_drift: -0.0032,
    alert_threshold: 0.03
  }

  // Mock performance history data
  const performanceData = [
    { time: '00:00', f1: 0.931, precision: 0.945, recall: 0.918 },
    { time: '04:00', f1: 0.933, precision: 0.947, recall: 0.920 },
    { time: '08:00', f1: 0.930, precision: 0.943, recall: 0.917 },
    { time: '12:00', f1: 0.929, precision: 0.941, recall: 0.916 },
    { time: '16:00', f1: 0.928, precision: 0.940, recall: 0.915 },
    { time: '20:00', f1: 0.927, precision: 0.938, recall: 0.914 },
    { time: 'Now', f1: 0.929, precision: 0.942, recall: 0.916 },
  ]

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Performance Monitoring</h1>
          <p className="text-gray-600 mt-2">Real-time model drift detection and performance tracking</p>
        </div>
        <select
          value={timeRange}
          onChange={(e) => setTimeRange(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
        >
          <option value="1h">Last Hour</option>
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last 7 Days</option>
          <option value="30d">Last 30 Days</option>
        </select>
      </div>

      {/* Drift Alert */}
      <DriftAlert detected={driftData.drift_detected} metrics={driftData} />

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <MetricCard
          title="F1 Score"
          value={driftData.current_f1}
          change={driftData.f1_drift}
          threshold={driftData.alert_threshold}
        />
        <MetricCard
          title="Precision"
          value={driftData.current_f1 + 0.013}
          change={driftData.precision_drift}
          threshold={driftData.alert_threshold}
        />
        <MetricCard
          title="Recall"
          value={driftData.current_f1 - 0.013}
          change={driftData.recall_drift}
          threshold={driftData.alert_threshold}
        />
      </div>

      {/* Performance Chart */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-6">Performance Trends</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis domain={[0.90, 0.95]} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="f1"
              stroke="#0ea5e9"
              strokeWidth={2}
              name="F1 Score"
              dot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="precision"
              stroke="#10b981"
              strokeWidth={2}
              name="Precision"
              dot={{ r: 4 }}
            />
            <Line
              type="monotone"
              dataKey="recall"
              stroke="#f59e0b"
              strokeWidth={2}
              name="Recall"
              dot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Prediction Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Prediction Statistics</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center pb-3 border-b">
              <span className="text-gray-600">Total Predictions</span>
              <span className="font-bold text-xl">
                {monitoringStats?.total_predictions?.toLocaleString() || '0'}
              </span>
            </div>
            <div className="flex justify-between items-center pb-3 border-b">
              <span className="text-gray-600">Low Confidence (&lt;0.7)</span>
              <div className="text-right">
                <span className="font-bold text-xl">
                  {monitoringStats?.low_confidence_count || '0'}
                </span>
                <span className="ml-2 text-sm text-gray-500">
                  ({((monitoringStats?.low_confidence_rate || 0) * 100).toFixed(1)}%)
                </span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Avg Latency</span>
              <span className="font-bold text-xl text-green-600">&lt;50ms</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Drift Thresholds</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-gray-600">F1 Score Drop Threshold</span>
                <span className="font-semibold">3%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: `${(Math.abs(driftData.f1_drift) / driftData.alert_threshold) * 100}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Current: {(Math.abs(driftData.f1_drift) * 100).toFixed(2)}%
              </p>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg">
              <p className="text-sm text-blue-900 font-medium">Retraining Criteria</p>
              <ul className="mt-2 space-y-1 text-xs text-blue-700">
                <li>• At least 100 feedback corrections</li>
                <li>• Minimum 7 days since last training</li>
                <li>• F1 drop exceeds 3% threshold</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Monitoring
