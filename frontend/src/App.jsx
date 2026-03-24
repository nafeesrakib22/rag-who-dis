import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import RetrievalTrace from './components/RetrievalTrace'

export default function App() {
  // ── View mode ──────────────────────────────────────────────
  const [view, setView] = useState('chat') // 'chat' | 'retrieve'

  // ── Shared state ───────────────────────────────────────────
  const [chunkCount, setChunkCount] = useState(null)
  const [hybridAlpha, setHybridAlpha] = useState(0.7)
  const [toast, setToast] = useState(null)

  // ── Chat state ──────────────────────────────────────────────
  const [messages, setMessages] = useState([])
  const [chatLoading, setChatLoading] = useState(false)
  const [traceMessageId, setTraceMessageId] = useState(null) // which message's trace to show

  // ── Retrieve state ──────────────────────────────────────────
  const [query, setQuery] = useState('')
  const [retrieveLoading, setRetrieveLoading] = useState(false)
  const [stages, setStages] = useState(null)
  const [retrieveError, setRetrieveError] = useState(null)

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

  // ── Chat handlers ───────────────────────────────────────────
  const handleSend = async (question) => {
    const userMsg = { id: Date.now(), role: 'user', content: question }
    setMessages(prev => [...prev, userMsg])
    setChatLoading(true)
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Server error')
      const assistantMsg = {
        id: data.message_id,
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        stages: data.stages,
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (e) {
      setMessages(prev => [...prev, {
        id: Date.now(),
        role: 'assistant',
        content: `⚠️ ${e.message}`,
        error: true,
      }])
    } finally {
      setChatLoading(false)
    }
  }

  const handleViewTrace = (msgId) => setTraceMessageId(msgId)
  const handleCloseTrace = () => setTraceMessageId(null)

  const traceMessage = messages.find(m => m.id === traceMessageId)

  // ── Retrieve handlers ───────────────────────────────────────
  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim() || retrieveLoading) return
    setRetrieveLoading(true)
    setRetrieveError(null)
    setStages(null)
    try {
      const res = await fetch('/api/retrieve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
      })
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Server error') }
      const data = await res.json()
      setStages(data.stages)
    } catch (e) {
      setRetrieveError(e.message)
    } finally {
      setRetrieveLoading(false)
    }
  }

  // ── Sidebar handlers ────────────────────────────────────────
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
      setMessages([])
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

      <div className="main-area">
        {/* ── Tab bar ── */}
        <div className="tab-bar">
          <button
            className={`tab-btn ${view === 'chat' ? 'active' : ''}`}
            onClick={() => setView('chat')}
          >
            💬 Chat
          </button>
          <button
            className={`tab-btn ${view === 'retrieve' ? 'active' : ''}`}
            onClick={() => { setView('retrieve'); setTraceMessageId(null) }}
          >
            🔍 Retrieval Trace
          </button>
        </div>

        {/* ── Chat view ── */}
        {view === 'chat' && (
          <div className="chat-view">
            <ChatPanel
              messages={messages}
              loading={chatLoading}
              onSend={handleSend}
              onViewTrace={handleViewTrace}
            />
            {traceMessage && (
              <RetrievalTrace
                stages={traceMessage.stages}
                query={traceMessage.content}
                onClose={handleCloseTrace}
              />
            )}
          </div>
        )}

        {/* ── Retrieve view ── */}
        {view === 'retrieve' && (
          <div className="retrieval-main">
            <div className="retrieval-header">
              <h2>Retrieval Trace</h2>
              <p>Inspect how the pipeline retrieves and ranks chunks — no LLM involved.</p>
            </div>

            <form className="search-form" onSubmit={handleSearch}>
              <input
                className="search-input"
                type="text"
                placeholder="Enter your query…"
                value={query}
                onChange={e => setQuery(e.target.value)}
                disabled={retrieveLoading}
              />
              <button className="btn btn-primary search-btn" type="submit" disabled={retrieveLoading || !query.trim()}>
                {retrieveLoading ? <span className="spinner" /> : 'Search'}
              </button>
            </form>

            {retrieveError && <div className="error-banner">⚠️ {retrieveError}</div>}

            {stages && (
              <RetrievalTrace stages={stages} query={query} />
            )}

            {!stages && !retrieveLoading && !retrieveError && (
              <div className="empty-state">
                <div className="empty-icon">🔍</div>
                <p>Enter a query above to inspect the retrieval pipeline.</p>
              </div>
            )}
          </div>
        )}
      </div>

      {toast && (
        <div className={`toast ${toast.type}`}>{toast.msg}</div>
      )}
    </div>
  )
}
