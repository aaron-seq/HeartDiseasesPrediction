import React, { useState } from 'react'
import axios from 'axios'

export default function App() {
  const [status, setStatus] = useState<string>('idle')
  const [message, setMessage] = useState<string>('')

  const ping = async () => {
    setStatus('loading')
    try {
      const { data } = await axios.get('/api/v1/health')
      setMessage(JSON.stringify(data, null, 2))
      setStatus('ok')
    } catch (e: any) {
      setMessage(e?.message || 'error')
      setStatus('error')
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif' }}>
      <h1>Heart Disease Prediction</h1>
      <p>Backend health: <strong>{status}</strong></p>
      <button onClick={ping}>Ping API</button>
      <pre>{message}</pre>
    </div>
  )
}
