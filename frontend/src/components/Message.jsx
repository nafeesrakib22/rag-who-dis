import { useState } from 'react'
import SourceCard from './SourceCard'

export default function Message({ message, onViewTrace }) {
    const isUser = message.role === 'user'
    const time = new Date(message.id).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const hasStages = message.stages && Object.keys(message.stages).length > 0

    return (
        <div className={`message ${message.role}`}>
            <div className="message-meta">
                <span>{isUser ? 'You' : 'RAG Assistant'} · {time}</span>
                {!isUser && hasStages && (
                    <button className="trace-btn" onClick={() => onViewTrace(message.id)}>
                        🔍 Trace
                    </button>
                )}
            </div>

            <div className={`bubble ${message.error ? 'error' : ''}`}>
                {message.content}
            </div>

            {!isUser && message.sources?.length > 0 && (
                <div className="sources">
                    <div className="sources-header">Sources ({message.sources.length})</div>
                    {message.sources.map(src => (
                        <SourceCard key={src.n} source={src} />
                    ))}
                </div>
            )}
        </div>
    )
}
