"""
main.py — CLI Entry Point

This is the command-line interface for your RAG system.

Usage:
    # Ingest a document (add it to the knowledge base):
    python main.py add data/myfile.pdf
    python main.py add data/notes.md

    # Ask a question:
    python main.py ask "What is the main topic of the document?"

    # Check how many chunks are in the DB:
    python main.py status

    # Clear the entire knowledge base:
    python main.py clear
"""

import argparse
import sys
from backend.core.logging_config import configure_logging
from backend.core.rag import RAGPipeline

configure_logging()


def print_banner():
    print("""
╔══════════════════════════════════════════╗
║         RAG System — from scratch        ║
║   Weaviate + sentence-transformers       ║
║              + Gemini API                ║
╚══════════════════════════════════════════╝
""")


def cmd_add(args, pipeline: RAGPipeline):
    """Ingest one or more documents."""
    for file_path in args.files:
        try:
            pipeline.ingest(file_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ Error: {e}")
            sys.exit(1)


def cmd_ask(args, pipeline: RAGPipeline):
    """Ask a question and print the answer with citations."""
    question = " ".join(args.question)  # allow multi-word without quotes

    result = pipeline.ask(question)

    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    print(result["answer"])

    if result["sources"]:
        print("\n" + "-"*60)
        print("SOURCES")
        print("-"*60)
        for src in result["sources"]:
            rerank = f"  |  rerank {src['rerank_score']:+.2f}" if src.get("rerank_score") is not None else ""
            hybrid = f"  |  hybrid {src['hybrid_score']:.4f}" if src.get("hybrid_score") is not None else ""
            print(f"  [Source {src['n']}] {src['source']}  |  page {src['page']}  "
                  f"|  chunk {src['chunk_index']}{hybrid}{rerank}")
            print(f"            \"{src['preview']}\"")
            print()


def cmd_status(args, pipeline: RAGPipeline):
    """Show how many chunks are stored in Weaviate."""
    count = pipeline.store.count()
    print(f"\n📦 Weaviate contains {count} chunk(s).")


def cmd_clear(args, pipeline: RAGPipeline):
    """Delete all chunks from the collection."""
    count = pipeline.store.count()
    if count == 0:
        print("Collection is already empty.")
        return

    confirm = input(f"⚠️  This will delete all {count} chunks. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        pipeline.store.clear()
        print("✅ Collection cleared.")
    else:
        print("Aborted.")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="RAG system — ingest documents and ask questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- add ---
    add_parser = subparsers.add_parser("add", help="Ingest a document into the knowledge base.")
    add_parser.add_argument("files", nargs="+", help="Path(s) to .pdf, .md, or .txt files.")

    # --- ask ---
    ask_parser = subparsers.add_parser("ask", help="Ask a question.")
    ask_parser.add_argument("question", nargs="+", help="Your question (no quotes needed).")

    # --- status ---
    subparsers.add_parser("status", help="Show how many chunks are in the knowledge base.")

    # --- clear ---
    subparsers.add_parser("clear", help="Delete all stored chunks.")

    args = parser.parse_args()

    # Initialize the pipeline (loads embedder + connects to Weaviate)
    try:
        pipeline = RAGPipeline()
    except EnvironmentError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Dispatch to the right command
    commands = {
        "add": cmd_add,
        "ask": cmd_ask,
        "status": cmd_status,
        "clear": cmd_clear,
    }
    commands[args.command](args, pipeline)

    # Close Weaviate connection cleanly to avoid ResourceWarning
    if hasattr(pipeline.store, "close"):
        pipeline.store.close()


if __name__ == "__main__":
    main()
