import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Monitoring from './pages/Monitoring'
import Feedback from './pages/Feedback'
import Analytics from './pages/Analytics'
import Categorize from './pages/Categorize'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/monitoring" element={<Monitoring />} />
        <Route path="/feedback" element={<Feedback />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/categorize" element={<Categorize />} />
      </Routes>
    </Layout>
  )
}

export default App
