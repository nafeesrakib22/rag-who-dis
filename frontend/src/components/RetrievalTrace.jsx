import SourceCard from './SourceCard'

export default function RetrievalTrace({ stages, query, onClose }) {
    if (!stages) return null

    const initial = stages.initial || []
    const reranked = stages.reranked || []

    const isWinner = (chunk) =>
        reranked.some(r => r.chunk_index === chunk.chunk_index && r.source === chunk.source)

    const inner = (
        <>
            <section className="trace-section">
                <div className="section-header">
                    <h4>Stage 1: Hybrid Search</h4>
                    <span className="count-pill">{initial.length} candidates</span>
                </div>
                <div className="trace-list">
                    {initial.map(src => (
                        <div key={`${src.source}-${src.chunk_index}`} className={`trace-item ${isWinner(src) ? 'winner' : ''}`}>
                            {isWinner(src) && <div className="winner-badge">Winner</div>}
                            <SourceCard source={src} query={query} />
                        </div>
                    ))}
                </div>
            </section>

            <div className="trace-divider">
                <div className="arrow">↓</div>
                <div className="label">Re-ranking</div>
            </div>

            <section className="trace-section">
                <div className="section-header">
                    <h4>Stage 2: Cross-Encoder Reranked</h4>
                    {reranked.length > 0 ? (
                        <span className="count-pill">{reranked.length} winners</span>
                    ) : (
                        <span className="count-pill disabled" style={{ background: 'var(--border)', color: 'var(--text-dim)' }}>Disabled</span>
                    )}
                </div>
                <div className="trace-list">
                    {reranked.length > 0 ? (
                        reranked.map(src => (
                            <SourceCard key={src.n} source={src} query={query} />
                        ))
                    ) : (
                        <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-dim)', fontSize: '13px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px' }}>
                            🚫 Stage 2 re-ranking is bypassed
                        </div>
                    )}
                </div>
            </section>

        </>
    )

    // Side-panel mode (used from chat view)
    if (onClose) {
        return (
            <aside className="retrieval-trace">
                <div className="trace-header">
                    <div>
                        <h3>Retrieval Trace</h3>
                        <p className="trace-query">"{query.length > 60 ? query.slice(0, 57) + '…' : query}"</p>
                    </div>
                    <button className="close-btn" onClick={onClose} title="Close">×</button>
                </div>
                <div className="trace-content">{inner}</div>
            </aside>
        )
    }

    // Standalone mode (used from retrieval tab)
    return (
        <div className="retrieval-trace standalone">{inner}</div>
    )
}
