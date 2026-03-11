import { useState } from 'react'

export default function SourceCard({ source, query }) {
    const [open, setOpen] = useState(false)

    // Helper to highlight words from query in text
    const HighlightText = ({ text, query }) => {
        if (!query || !text) return text

        // Escape regex special chars and filter significant words
        const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        const words = query.toLowerCase()
            .split(/\s+/)
            .map(w => w.replace(/[?.,!()[\]{}]/g, ''))
            .filter(w => w.length > 3)

        if (words.length === 0) return text

        try {
            const regex = new RegExp(`(${words.map(escape).join('|')})`, 'gi')
            const parts = text.split(regex)

            return (
                <span>
                    {parts.map((part, i) =>
                        regex.test(part) ? (
                            <mark key={i} className="match-highlight">{part}</mark>
                        ) : part
                    )}
                </span>
            )
        } catch (e) {
            console.error("Regex error:", e)
            return text
        }
    }

    return (
        <div className={`source-card ${open ? 'open' : ''}`}>
            <div className="source-header" onClick={() => setOpen(o => !o)}>
                <div className="source-num">{source.n}</div>
                <div className="source-info">
                    <div className="source-name">{source.source}</div>
                    <div className="source-meta">
                        Page {source.page} · Chunk {source.chunk_index}
                    </div>
                </div>
                <div className="source-scores">
                    {source.hybrid_score != null && (
                        <span className="score-pill hybrid">
                            hybrid {source.hybrid_score.toFixed(3)}
                        </span>
                    )}
                    {source.rerank_score != null && (
                        <span className="score-pill rerank">
                            rerank {source.rerank_score > 0 ? '+' : ''}{source.rerank_score.toFixed(2)}
                        </span>
                    )}
                </div>
                <span className="source-chevron">▾</span>
            </div>

            {open && (
                <div className="source-preview">
                    <HighlightText text={source.text || source.preview} query={query} />
                </div>
            )}
        </div>
    )
}
