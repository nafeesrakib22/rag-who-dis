import SourceCard from './SourceCard'

export default function RetrievalTrace({ stages, query, onClose }) {
    if (!stages) return null

    const initial = stages.initial || []
    const reranked = stages.reranked || []

    // Helper to check if a chunk from initial stage made it to the top 5
    const isWinner = (chunk) => {
        return reranked.some(r => r.chunk_index === chunk.chunk_index && r.source === chunk.source)
    }

    return (
        <div className="retrieval-trace">
            <div className="trace-header">
                <div>
                    <h3>Retrieval Trace</h3>
                    <p>Inspect how the pipeline found your answer</p>
                </div>
                <button className="close-btn" onClick={onClose}>✕</button>
            </div>

            <div className="trace-content">
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
                        <h4>Stage 2: Cross-Encoder</h4>
                        <span className="count-pill">{reranked.length} winners</span>
                    </div>
                    <div className="trace-list">
                        {reranked.map(src => (
                            <SourceCard key={src.n} source={src} query={query} />
                        ))}
                    </div>
                </section>
            </div>
        </div>
    )
}
