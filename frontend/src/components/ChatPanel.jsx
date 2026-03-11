import { useRef, useEffect, useState } from 'react'
import Message from './Message'

const SUGGESTIONS = [
    'What is the difference between supervised and unsupervised learning?',
    'What is the output vector shape of EdgeFace?',
    'What is the input image size expected by EdgeFace?',
    'How does backpropagation work?',
]

export default function ChatPanel({ messages, loading, onSend, onViewTrace }) {
    const [draft, setDraft] = useState('')
    const bottomRef = useRef(null)
    const textareaRef = useRef(null)

    // Auto-scroll on new messages
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages, loading])

    const submit = () => {
        const q = draft.trim()
        if (!q || loading) return
        setDraft('')
        onSend(q)
        // Reset textarea height
        if (textareaRef.current) textareaRef.current.style.height = 'auto'
    }

    const onKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            submit()
        }
    }

    const onInput = (e) => {
        setDraft(e.target.value)
        // Auto-grow textarea
        e.target.style.height = 'auto'
        e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px'
    }

    return (
        <div className="chat-panel">
            <div className="chat-header">
                <div>
                    <h2>RAG Chat</h2>
                    <p>Ask questions grounded in your documents</p>
                </div>
            </div>

            <div className="message-list">
                {messages.length === 0 && !loading ? (
                    <div className="empty-state">
                        <div className="empty-icon">🔍</div>
                        <h3>Ask anything about your documents</h3>
                        <p>Ingest a PDF or Markdown file using the sidebar, then ask questions here.</p>
                        <div className="chips">
                            {SUGGESTIONS.map(s => (
                                <button key={s} className="chip" onClick={() => onSend(s)}>
                                    {s.length > 60 ? s.slice(0, 57) + '…' : s}
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    <>
                        {messages.map(msg => (
                            <Message key={msg.id} message={msg} onViewTrace={onViewTrace} />
                        ))}
                        {loading && (
                            <div className="message assistant">
                                <div className="message-meta">RAG Assistant</div>
                                <div className="typing-bubble">
                                    <div className="typing-dot" />
                                    <div className="typing-dot" />
                                    <div className="typing-dot" />
                                </div>
                            </div>
                        )}
                        <div ref={bottomRef} />
                    </>
                )}
            </div>

            <div className="chat-input-area">
                <div className="input-row">
                    <textarea
                        ref={textareaRef}
                        rows={1}
                        placeholder="Ask a question about your documents… (Enter to send, Shift+Enter for newline)"
                        value={draft}
                        onChange={onInput}
                        onKeyDown={onKeyDown}
                        disabled={loading}
                    />
                    <button
                        className="send-btn"
                        onClick={submit}
                        disabled={!draft.trim() || loading}
                        title="Send"
                    >
                        ↑
                    </button>
                </div>
                <p className="input-hint">Answers are grounded in your uploaded documents with citations.</p>
            </div>
        </div>
    )
}
