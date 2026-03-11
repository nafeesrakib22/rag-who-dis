import SourceCard from './SourceCard'

export default function RetrievalTrace({ stages, query }) {
    if (!stages) return null

    const initial = stages.initial || []
    const reranked = stages.reranked || []

    // Helper to check if a chunk from initial stage made it to the top reranked
    const isWinner = (chunk) =>
        reranked.some(r => r.chunk_index === chunk.chunk_index && r.source === chunk.source)

    return (
        <div className="retrieval-trace standalone">
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
                    <span className="count-pill">{reranked.length} winners</span>
                </div>
                <div className="trace-list">
                    {reranked.map(src => (
                        <SourceCard key={src.n} source={src} query={query} />
                    ))}
                </div>
            </section>
        </div>
    )
}
