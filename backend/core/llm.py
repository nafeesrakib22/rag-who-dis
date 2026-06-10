"""
llm.py — LLM Service Layer

Two providers, one interface:

  GeminiLLMService  — calls Google Gemini API (stateless, history injected as text)
  LocalLLMService   — runs gemma-4-E2B-it via litert-lm (stateful KV cache sessions)

Use get_llm_service() to get the right one based on config.LLM_PROVIDER.

LOCAL LLM SESSION MODEL:
  A single active conversation session is kept alive for the lifetime of the
  backend process. Each session ID maps to a litert-lm conversation object
  whose KV cache grows incrementally — subsequent turns are faster because
  the model doesn't re-process prior tokens.

  Only one session is active at a time. When a new session_id is seen,
  the old session is closed and a new one is created. Sessions are never
  persisted to disk — a backend restart resets all conversations.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from google import genai
from . import config


# ── System prompt used by the local LLM session ──────────────────────────────

_LOCAL_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based ONLY on the "
    "context provided in each user message.\n\n"
    "Rules:\n"
    "- Base your answer SOLELY on the provided context. Do not use outside knowledge.\n"
    "- Answer in the SAME LANGUAGE as the question.\n"
    "- Cite your sources inline using [Source N] notation.\n"
    '- If the answer cannot be found in the context, say: '
    '"I don\'t have enough information in the provided documents to answer this."'
)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseLLMService(ABC):

    @abstractmethod
    def generate_content(self, prompt: str, temperature: float = None) -> str:
        """Stateless single-turn generation (used for query condensation, summarisation, etc.)"""
        ...

    def stream_content(self, prompt: str, temperature: float = None) -> Iterator[str]:
        """Stream generation token-by-token. Default falls back to non-streaming."""
        yield self.generate_content(prompt, temperature)


# ── Gemini (cloud, stateless) ─────────────────────────────────────────────────

class GeminiLLMService(BaseLLMService):
    """Calls the Google Gemini API. All history is injected as text in the prompt."""

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in .env or environment.")
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.GEMINI_MODEL

    def generate_content(self, prompt: str, temperature: float = None) -> str:
        temp = temperature if temperature is not None else config.GEMINI_TEMPERATURE
        print(f"[llm] Calling Gemini ({self.model})...")
        completion = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": temp},
        )
        return completion.text.strip()

    def stream_content(self, prompt: str, temperature: float = None) -> Iterator[str]:
        """Yield text chunks as Gemini streams them."""
        temp = temperature if temperature is not None else config.GEMINI_TEMPERATURE
        print(f"[llm] Streaming from Gemini ({self.model})...")
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config={"temperature": temp},
        ):
            if chunk.text:
                yield chunk.text


# ── Local LLM (litert-lm, stateful KV cache) ──────────────────────────────────

class LocalLLMService(BaseLLMService):
    """
    Runs gemma-4-E2B-it locally via litert-lm.

    Maintains a single active conversation session whose KV cache grows
    incrementally. Each call to chat() only processes the new user turn —
    the model already has all prior turns cached.
    """

    def __init__(self, model_path: str):
        import litert_lm  # lazy import — only loaded when local mode is active

        print(f"[llm-local] Loading litert-lm engine from '{model_path}'...")
        self._litert_lm = litert_lm
        self._engine = litert_lm.Engine(model_path, backend=litert_lm.Backend.CPU)

        # Active session state (one session at a time)
        self._active_session_id: str | None = None
        self._active_session_ctx = None   # context manager object
        self._active_session = None       # the conversation object

        print("[llm-local] Engine ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_content(self, prompt: str, temperature: float = None) -> str:
        """
        Stateless single-turn generation (for query condensation / summarisation).
        Creates a temporary session that is closed immediately after.
        """
        print("[llm-local] Stateless generation call...")
        with self._engine.create_conversation() as conv:
            response = conv.send_message(prompt)
            return response["content"][0]["text"].strip()

    def chat(self, question: str, context_block: str, session_id: str) -> str:
        """
        Stateful multi-turn chat using the KV cache session.

        The retrieved context is injected alongside the question for this turn.
        Prior turns live in the KV cache — they are NOT re-sent.
        """
        session = self._get_or_create_session(session_id)

        # Inject retrieval context for this turn only
        message = (
            f"CONTEXT FROM DOCUMENTS:\n{context_block}\n\n"
            f"QUESTION:\n{question}"
        )

        print(f"[llm-local] Sending turn to session '{session_id[:8]}...'")
        response = session.send_message(message)
        return response["content"][0]["text"].strip()

    def reset_session(self) -> None:
        """Close and discard the active session. Next chat() call creates a fresh one."""
        self._close_active_session()
        print("[llm-local] Session reset.")

    # ── Session management ────────────────────────────────────────────────────

    def _get_or_create_session(self, session_id: str):
        if self._active_session_id == session_id and self._active_session is not None:
            return self._active_session

        # New session_id → close old session, open a fresh one
        self._close_active_session()

        print(f"[llm-local] Creating new session '{session_id[:8]}...'")
        system_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": _LOCAL_SYSTEM_PROMPT}],
            }
        ]
        self._active_session_ctx = self._engine.create_conversation(
            messages=system_messages
        )
        self._active_session = self._active_session_ctx.__enter__()
        self._active_session_id = session_id
        return self._active_session

    def _close_active_session(self) -> None:
        if self._active_session_ctx is not None:
            try:
                self._active_session_ctx.__exit__(None, None, None)
            except Exception as e:
                print(f"[llm-local] Warning: error closing session — {e}")
        self._active_session = None
        self._active_session_ctx = None
        self._active_session_id = None


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_service() -> BaseLLMService:
    """Return the correct LLM service based on LLM_PROVIDER in config."""
    if config.LLM_PROVIDER == "local":
        return LocalLLMService(config.LOCAL_MODEL_PATH)
    return GeminiLLMService()
