import sys
import os
from src.loader import load_document, is_text_corrupted

def peek_extraction(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"\n--- Processing: {file_path} ---\n")
    
    # We call load_document which will use our new load_pdf (with OCR fallback)
    pages = load_document(file_path)
    
    if not pages:
        print("No text extracted.")
        return

    print(f"\n--- Extracted {len(pages)} pages ---")
    for i, page in enumerate(pages[:2], 1): # Peek first 2 pages
        text_preview = page['text'][:500] + "..." if len(page['text']) > 500 else page['text']
        print(f"\n[Page {page['page']}] Preview:\n{text_preview}")
        print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python peek_text.py <path_to_pdf>")
    else:
        peek_extraction(sys.argv[1])
