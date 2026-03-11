import { useRef, useState } from 'react'

export default function Sidebar({ chunkCount, hybridAlpha, onAlphaChange, onIngest, onClear }) {
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
                <div className="logo-icon">🔮</div>
                <h1>RAG Chat</h1>
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
                        value={hybridAlpha || 0.7}
                        onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
                        className="alpha-slider"
                    />
                    <div className="alpha-value">
                        Alpha: <strong>{(hybridAlpha || 0.7).toFixed(2)}</strong>
                    </div>
                </div>
            </div>

            {/* Actions */}
            <div className="sidebar-section" style={{ marginTop: 'auto' }}>
                <label>Actions</label>
                <button className="btn btn-ghost" onClick={onClear}>
                    🗑 Clear knowledge base
                </button>
            </div>

            {/* Stack info */}
            <div style={{ fontSize: '11px', color: 'var(--text-dim)', padding: '0 8px', lineHeight: 1.8 }}>
                <div>📦 Weaviate (hybrid search)</div>
                <div>🧠 embeddinggemma (Ollama)</div>
                <div>🔀 ms-marco cross-encoder</div>
                <div>🤖 Gemini 2.5 flash</div>
            </div>
        </aside>
    )
}
