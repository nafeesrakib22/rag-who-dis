import logging
import os


def configure_logging() -> None:
    """
    Configure root logger for the whole application.

    Level is controlled by the LOG_LEVEL environment variable (default INFO).
    Set LOG_LEVEL=DEBUG to see verbose pipeline steps (embeddings, reranker
    scores, query condensation, etc.).
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
