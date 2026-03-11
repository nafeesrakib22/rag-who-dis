
from backend.core.loader import load_document, is_text_corrupted

def test_loader(file_path):
    print(f"Testing loader for: {file_path}")
    pages = load_document(file_path)
    
    if not pages:
        print("No pages loaded.")
        return

    # Print a sample of the first page
    sample_text = pages[0]['text'][:300]
    print("\n--- EXTRACTED TEXT SAMPLE (Page 1) ---")
    print(sample_text)
    print("--------------------------------------\n")
    
    corrupted = is_text_corrupted(sample_text)
    print(f"Is text detected as corrupted? {corrupted}")
    
    if corrupted:
        print("ALERT: The detection logic says it's corrupted.")
    else:
        print("SUCCESS: The detection logic thinks the text is clean.")

if __name__ == "__main__":
    pdf_path = "data/KhaledaZia.pdf"
    if os.path.exists(pdf_path):
        test_loader(pdf_path)
    else:
        print(f"File not found: {pdf_path}")
