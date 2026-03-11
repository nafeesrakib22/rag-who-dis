import { useState, useRef, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatPanel from './components/ChatPanel'
import RetrievalTrace from './components/RetrievalTrace'

export default function App() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [chunkCount, setChunkCount] = useState(null)
  const [hybridAlpha, setHybridAlpha] = useState(0.7)
  const [activeTrace, setActiveTrace] = useState(null)
  const [toast, setToast] = useState(null)

  // Fetch status on mount
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

  const sendMessage = async (question) => {
    if (!question.trim() || loading) return

    const userMsg = { role: 'user', content: question, id: Date.now() }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Error from server')
      }
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        stages: data.stages,
        id: data.message_id,
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ ${e.message}`,
        sources: [],
        id: Date.now(),
        error: true,
      }])
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
      setMessages([])
      showToast('Knowledge base cleared.')
    } catch (e) {
      showToast(`❌ ${e.message}`, 'error')
    }
  }

  return (
    <div className={`app ${activeTrace ? 'with-trace' : ''}`}>
      <Sidebar
        chunkCount={chunkCount}
        hybridAlpha={hybridAlpha}
        onAlphaChange={updateHybridAlpha}
        onIngest={handleIngest}
        onClear={handleClear}
      />
      <ChatPanel
        messages={messages}
        loading={loading}
        onSend={sendMessage}
        onViewTrace={(msgId) => {
          const idx = messages.findIndex(m => m.id === msgId)
          if (idx === -1) return
          const msg = messages[idx]
          // If it's an assistant message, the query is usually the previous user message
          const question = (msg.role === 'assistant' && idx > 0) ? messages[idx - 1].content : msg.content
          setActiveTrace({ stages: msg.stages, query: question })
        }}
      />
      {activeTrace && (
        <RetrievalTrace
          stages={activeTrace.stages}
          query={activeTrace.query}
          onClose={() => setActiveTrace(null)}
        />
      )}
      {toast && (
        <div className={`toast ${toast.type}`}>{toast.msg}</div>
      )}
    </div>
  )
}
