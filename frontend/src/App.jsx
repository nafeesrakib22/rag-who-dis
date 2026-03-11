import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import RetrievalTrace from './components/RetrievalTrace'

export default function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [stages, setStages] = useState(null)
  const [error, setError] = useState(null)
  const [chunkCount, setChunkCount] = useState(null)
  const [hybridAlpha, setHybridAlpha] = useState(0.7)
  const [toast, setToast] = useState(null)

  useEffect(() => { fetchStatus() }, [])

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3500)
  }

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status')
      const data = await res.json()
      setChunkCount(data.chunk_count)
      setHybridAlpha(data.hybrid_alpha)
    } catch { /* ignore */ }
  }

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim() || loading) return
    setLoading(true)
    setError(null)
    setStages(null)
    try {
      const res = await fetch('/api/retrieve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Server error')
      }
      const data = await res.json()
      setStages(data.stages)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleIngest = async (file) => {
    const fd = new FormData()
    fd.append('file', file)
    try {
      const res = await fetch('/api/ingest', { method: 'POST', body: fd })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail) }
      const data = await res.json()
      showToast(`✅ ${data.message}`)
      setChunkCount(data.chunk_count)
    } catch (e) {
      showToast(`❌ ${e.message}`, 'error')
    }
  }

  const updateHybridAlpha = async (val) => {
    setHybridAlpha(val)
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hybrid_alpha: val })
      })
      if (!res.ok) throw new Error('Failed to update alpha')
    } catch (e) {
      showToast(`❌ Error: ${e.message}`, 'error')
    }
  }

  const handleClear = async () => {
    try {
      await fetch('/api/clear', { method: 'POST' })
      setChunkCount(0)
      setStages(null)
      showToast('Knowledge base cleared.')
    } catch (e) {
      showToast(`❌ ${e.message}`, 'error')
    }
  }

  return (
    <div className="app">
      <Sidebar
        chunkCount={chunkCount}
        hybridAlpha={hybridAlpha}
        onAlphaChange={updateHybridAlpha}
        onIngest={handleIngest}
        onClear={handleClear}
      />

      <main className="retrieval-main">
        <div className="retrieval-header">
          <h2>Retrieve Who dis</h2>
          <p>Enter a query to see how the pipeline retrieves and ranks chunks from your knowledge base.</p>
        </div>

        <form className="search-form" onSubmit={handleSearch}>
          <input
            className="search-input"
            type="text"
            placeholder="Enter your query…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            disabled={loading}
          />
          <button className="btn btn-primary search-btn" type="submit" disabled={loading || !query.trim()}>
            {loading ? <span className="spinner" /> : 'Search'}
          </button>
        </form>

        {error && <div className="error-banner">⚠️ {error}</div>}

        {stages && (
          <RetrievalTrace stages={stages} query={query} />
        )}

        {!stages && !loading && !error && (
          <div className="empty-state">
            <div className="empty-icon">🔍</div>
            <p>Enter a query above to inspect the retrieval pipeline.</p>
          </div>
        )}
      </main>

      {toast && (
        <div className={`toast ${toast.type}`}>{toast.msg}</div>
      )}
    </div>
  )
}
