import { useState, useEffect, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import RetrievalTrace from './components/RetrievalTrace'

// Generate a UUID v4 for session identification
function generateSessionId() {
  return crypto.randomUUID()
}

export default function App() {
  // ── View mode ──────────────────────────────────────────────
  const [view, setView] = useState('chat') // 'chat' | 'retrieve'

  // ── Shared state ───────────────────────────────────────────
  const [chunkCount, setChunkCount] = useState(null)
  const [hybridAlpha, setHybridAlpha] = useState(0.7)
  const [useReranker, setUseReranker] = useState(true)
  const [llmProvider, setLlmProvider] = useState('gemini')
  const [toast, setToast] = useState(null)

  // ── Session ID — resets on page refresh or New Chat ────────
  const [sessionId, setSessionId] = useState(() => generateSessionId())
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
      if (data.use_reranker !== undefined) setUseReranker(data.use_reranker)
      if (data.llm_provider) setLlmProvider(data.llm_provider)
    } catch { /* ignore */ }
  }


  // ── Chat handlers ───────────────────────────────────────────
  const handleSend = async (question) => {
    const userMsg = { id: Date.now(), role: 'user', content: question }
    const assistantId = crypto.randomUUID()
    // Add user message + empty assistant placeholder for streaming
    setMessages(prev => [...prev, userMsg, {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
      stages: null,
      streaming: true,
    }])
    setChatLoading(true)

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          history: messages.map(m => ({ role: m.role, content: m.content })),
          session_id: sessionId,
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Server error')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // Parse complete SSE messages from the buffer
        const parts = buffer.split('\n\n')
        buffer = parts.pop() // keep incomplete tail

        for (const part of parts) {
          const eventMatch = part.match(/^event: (.+)$/m)
          const dataMatch = part.match(/^data: (.*)$/m)
          if (!eventMatch || !dataMatch) continue
          const event = eventMatch[1]
          const data = dataMatch[1].replace(/\\n/g, '\n')

          if (event === 'token') {
            setMessages(prev => prev.map(m =>
              m.id === assistantId
                ? { ...m, content: m.content + data }
                : m
            ))
          } else if (event === 'sources') {
            const sources = JSON.parse(data)
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, sources } : m
            ))
          } else if (event === 'stages') {
            const stages = JSON.parse(data)
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, stages } : m
            ))
          } else if (event === 'error') {
            setMessages(prev => prev.map(m =>
              m.id === assistantId
                ? { ...m, content: `⚠️ ${data}`, error: true }
                : m
            ))
          }
          // 'done' — just stop processing
        }
      }

      // Mark streaming complete
      setMessages(prev => prev.map(m =>
        m.id === assistantId ? { ...m, streaming: false } : m
      ))
    } catch (e) {
      setMessages(prev => prev.map(m =>
        m.id === assistantId
          ? { ...m, content: `⚠️ ${e.message}`, error: true, streaming: false }
          : m
      ))
    } finally {
      setChatLoading(false)
    }
  }

  const handleViewTrace = (msgId) => setTraceMessageId(msgId)
  const handleCloseTrace = () => setTraceMessageId(null)
  const traceMessage = messages.find(m => m.id === traceMessageId)

  // ── New Chat — resets session and clears messages ───────────
  const handleNewChat = async () => {
    setMessages([])
    setTraceMessageId(null)
    const newId = generateSessionId()
    setSessionId(newId)
    // Explicitly notify backend to free the old local LLM session
    try { await fetch('/api/reset-session', { method: 'POST' }) } catch { /* ignore */ }
  }

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

  const updateReranker = async (val) => {
    setUseReranker(val)
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_reranker: val })
      })
      if (!res.ok) throw new Error('Failed to update reranker setting')
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
        useReranker={useReranker}
        llmProvider={llmProvider}
        onAlphaChange={updateHybridAlpha}
        onRerankerChange={updateReranker}
        onIngest={handleIngest}
        onClear={handleClear}
        onNewChat={handleNewChat}
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
