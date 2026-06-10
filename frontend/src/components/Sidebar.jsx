import { useRef, useState } from 'react'

export default function Sidebar({ chunkCount, hybridAlpha, onAlphaChange, onIngest, onClear, useReranker, onRerankerChange, llmProvider, onNewChat, authRequired, adminToken, onAdminTokenChange }) {

    const fileRef = useRef(null)
    const [busy, setBusy] = useState(false)
    const [drag, setDrag] = useState(false)

    const handleFile = async (file) => {
        if (!file || busy) return
        setBusy(true)
        await onIngest(file)
        setBusy(false)
        if (fileRef.current) fileRef.current.value = ''
    }

    const onDrop = (e) => {
        e.preventDefault(); setDrag(false)
        const file = e.dataTransfer.files[0]
        if (file) handleFile(file)
    }

    return (
        <aside className="sidebar">
            {/* Logo */}
            <div className="sidebar-logo">
                <div className="logo-icon">🪓</div>
                <h1>Bring me Thanos!</h1>
            </div>

            {/* Knowledge base status */}
            <div className="sidebar-section">
                <label>Knowledge Base</label>
                <div className="status-badge">
                    <div className="dot" />
                    {chunkCount === null
                        ? 'Connecting…'
                        : `${chunkCount.toLocaleString()} chunk${chunkCount !== 1 ? 's' : ''} indexed`}
                </div>
            </div>

            {/* File upload */}
            <div className="sidebar-section">
                <label>Add Documents</label>
                {busy ? (
                    <div className="upload-progress">
                        <div className="spinner" />
                        Ingesting file…
                    </div>
                ) : (
                    <div
                        className={`upload-area ${drag ? 'drag-over' : ''}`}
                        onClick={() => fileRef.current?.click()}
                        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
                        onDragLeave={() => setDrag(false)}
                        onDrop={onDrop}
                    >
                        <div className="upload-icon">📄</div>
                        <div>Drop a PDF or Markdown file here</div>
                        <div style={{ fontSize: '11px', marginTop: '4px', opacity: 0.6 }}>or click to browse</div>
                        <input
                            ref={fileRef}
                            type="file"
                            accept=".pdf,.md,.txt"
                            onChange={(e) => handleFile(e.target.files[0])}
                        />
                    </div>
                )}
            </div>

            {/* Search Balance Slider */}
            <div className="sidebar-section">
                <label>Search Balance</label>
                <div className="alpha-control">
                    <div className="alpha-labels">
                        <span>Keyword</span>
                        <span>Vector</span>
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={hybridAlpha ?? 0.7}
                        onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
                        className="alpha-slider"
                    />
                    <div className="alpha-value">
                        Alpha: <strong>{(hybridAlpha ?? 0.7).toFixed(2)}</strong>
                    </div>
                </div>
            </div>

            {/* Reranker Toggle */}
            <div className="sidebar-section">
                <label>Re-ranking</label>
                <div className="alpha-control">
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', textTransform: 'none', fontSize: '13px', color: 'var(--text)' }}>
                        <input
                            type="checkbox"
                            checked={useReranker ?? true}
                            onChange={(e) => onRerankerChange(e.target.checked)}
                            style={{ width: '16px', height: '16px', cursor: 'pointer' }}
                        />
                        Enable Stage 2 Cross-Encoder
                    </label>
                </div>
            </div>


            {/* Admin Token (only shown when server requires auth) */}
            {authRequired && (
                <div className="sidebar-section">
                    <label>Admin Token</label>
                    <input
                        type="password"
                        className="admin-token-input"
                        placeholder="Enter admin token…"
                        value={adminToken}
                        onChange={(e) => onAdminTokenChange(e.target.value)}
                        style={{
                            width: '100%',
                            padding: '6px 8px',
                            fontSize: '12px',
                            borderRadius: '6px',
                            border: '1px solid var(--border)',
                            background: 'var(--bg)',
                            color: 'var(--text)',
                        }}
                    />
                    <div style={{ fontSize: '10px', opacity: 0.5, marginTop: '4px' }}>
                        Set ADMIN_TOKEN in .env to protect settings.
                    </div>
                </div>
            )}

            {/* Actions */}
            <div className="sidebar-section" style={{ marginTop: 'auto' }}>
                <label>Actions</label>
                <button className="btn btn-primary" onClick={onNewChat} style={{ marginBottom: '8px' }}>
                    ✨ New Chat
                </button>
                <button className="btn btn-ghost" onClick={onClear}>
                    🗑 Clear knowledge base
                </button>
            </div>

            {/* Stack info */}
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', padding: '0 8px', lineHeight: 1.8 }}>
                <div>📦 Weaviate (hybrid search)</div>
                <div>🧠 embeddinggemma-300m (HF)</div>
                <div>🔀 bge-reranker-v2-m3</div>
                <div>🤖 LLM: <strong>{llmProvider === 'local' ? 'gemma-4-E2B (local)' : 'Gemini API'}</strong></div>
            </div>
        </aside>
    )
}
