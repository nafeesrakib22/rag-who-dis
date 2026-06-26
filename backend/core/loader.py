

import json
import logging
import os
from pathlib import Path
from . import config  # noqa: F401 — side-effect import: ensures .env is loaded

logger = logging.getLogger(__name__)


def is_text_corrupted(text: str) -> bool:
    """
    Detect if the extracted text looks like corrupted Bangla.
    """
    if not text or len(text.strip()) < 10:
        return False
        
    # Ignore text that is primarily ASCII (English/Numbers/Symbols).
    # This prevents English documents from being flagged as "bad Bangla".
    ascii_chars = [char for char in text if ord(char) < 128 and char.isalpha()]
    total_alpha = sum(1 for char in text if char.isalpha())
    
    if total_alpha > 0 and (len(ascii_chars) / total_alpha) > 0.5:
        return False # Primarily English, not corrupted Bangla
        
    # 1. Basic Unicode Range Check
    bangla_unicode_chars = [char for char in text if '\u0980' <= char <= '\u09ff']
    
    if total_alpha == 0:
        return False
        
    unicode_ratio = len(bangla_unicode_chars) / total_alpha
    if unicode_ratio < 0.7:
        return True # Definitely corrupted/non-Unicode
        
    # 2. Linguistic "Anchor Word" Check
    common_words = ["এবং", "করে", "জন্য", "ছিল", "থেকে", "সাথে", "একটি", "করতে", "হবে", "এই"]
    text_lower = text.lower()
    
    matches = sum(1 for word in common_words if word in text_lower)
    
    if len(bangla_unicode_chars) > 100 and matches == 0:
        return True
        
    return False


# Module-level cache for EasyOCR reader
_OCR_READER_CACHE = None

def get_ocr_reader():
    """
    Lazy-initialize and cache the EasyOCR reader.
    Handles hardware compatibility checks beyond basic CUDA availability.
    """
    global _OCR_READER_CACHE
    if _OCR_READER_CACHE is None:
        import easyocr
        import torch
        
        gpu_available = False
        if torch.cuda.is_available():
            # Check if current PyTorch kernels support this GPU's architecture.
            # MX150 is 6.1, but many modern Torch binaries require >= 7.0.
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 7:
                gpu_available = True
                logger.info("GPU detected (sm_%d%d). Enabling GPU for OCR.", major, minor)
            else:
                logger.warning(
                    "GPU detected (sm_%d%d) but PyTorch kernels lack support. Falling back to CPU.",
                    major, minor,
                )
        else:
            logger.info("No GPU detected. OCR will run on CPU.")
            
        # Initialize reader for Bangla and English
        # We also pass 'quantize=True' for CPU mode to reduce memory footprint
        # and prevent RAM exhaustion at higher DPIs.
        _OCR_READER_CACHE = easyocr.Reader(['bn', 'en'], gpu=gpu_available, quantize=True)
    return _OCR_READER_CACHE


def google_ocr_with_gemini(images) -> list[str]:

    from google import genai
    import time
    
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in .env. Falling back to local OCR.")
        return None

    client = genai.Client(api_key=api_key)
    
    model_id = 'gemini-2.5-flash'
    
    texts = []
    logger.info("Using Google Vision OCR (%s) for %d pages...", model_id, len(images))

    for i, img in enumerate(images, start=1):
        max_retries = 3
        retry_delay = 5  # seconds
        success = False
        
        for attempt in range(max_retries):
            try:
                # google-genai accepts PIL images directly
                response = client.models.generate_content(
                    model=model_id,
                    contents=[
                        "Extract all text from this image exactly as it appears. Maintain the layout where possible. Output ONLY the extracted text in the original language (Bangla/English).",
                        img
                    ]
                )
                extracted_text = response.text.strip()
                texts.append(extracted_text)
                logger.debug("Google OCR done for page %d.", i)
                success = True
                # Small polite delay between pages for free tier (RPM limits)
                time.sleep(2)
                break
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(
                        "Rate limit on page %d (attempt %d/%d). Retrying in %ds...",
                        i, attempt + 1, max_retries, retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("OCR error on page %d: %s", i, e)
                    break
        
        if not success:
            texts.append("") # fail gracefully if all retries fail
            
    return texts


def load_pdf_ocr(file_path: str, source_name: str = None) -> list[dict]:

    import numpy as np
    from pdf2image import convert_from_path
    import gc

    strategy = os.environ.get("OCR_STRATEGY", "local").lower()
    filename = source_name or os.path.basename(file_path)
    logger.info("OCR fallback starting for '%s' (strategy: %s)...", filename, strategy)

    # DPI selection: Vision models handle lower DPI better than local OCR.
    dpi = 200 if strategy == "local" else 150
    images = convert_from_path(file_path, dpi=dpi)
    pages = []

    if strategy == "llm":
        results = google_ocr_with_gemini(images)
        if results:
            for i, text in enumerate(results, start=1):
                if text.strip():
                    pages.append({
                        "text": text.strip(),
                        "source": filename,
                        "page": i,
                    })
            return pages
        else:
            logger.warning("Google Vision OCR failed. Falling back to local OCR...")

    # Default to Local OCR
    reader = get_ocr_reader()
    for page_num, img in enumerate(images, start=1):
        img_arr = np.array(img)
        results = reader.readtext(img_arr, detail=0)
        text = "\n".join(results)
        
        if text.strip():
            pages.append({
                "text": text.strip(),
                "source": filename,
                "page": page_num,
            })
            logger.debug("Local OCR done for page %d.", page_num)
        
        del img_arr
        img.close()
        del img
        gc.collect()

    return pages


def load_pdf(file_path: str, source_name: str = None) -> list[dict]:
    """
    Load a PDF file and extract text page by page.

    Attempts standard extraction with PyMuPDF first. If the resulting text
    appears corrupted (broken font mapping), it falls back to OCR.
    """
    import fitz

    doc = fitz.open(file_path)
    filename = source_name or Path(file_path).name
    use_ocr = False
    temp_pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        
        # Trigger OCR if the page is empty (scanned) or text is corrupted (bad encoding)
        if not text:
            logger.warning("Page %d is empty. Triggering OCR fallback...", page_num)
            use_ocr = True
            break

        if is_text_corrupted(text):
            logger.warning("Corruption detected on page %d. Triggering OCR fallback...", page_num)
            use_ocr = True
            break
        
        temp_pages.append({
            "text": text,
            "source": filename,
            "page": page_num,
        })

    if use_ocr:
        # If even one page is corrupted, we re-extract the whole doc with OCR
        # to ensure consistency in quality across pages.
        return load_pdf_ocr(file_path, source_name=source_name)
        
    logger.info("Loaded PDF '%s': %d pages via PyMuPDF.", filename, len(temp_pages))
    return temp_pages


def load_text(file_path: str, source_name: str = None) -> list[dict]:
    """
    Load a Markdown or plain text (.txt) file.

    Returns:
        A list with a single dict (the entire file is treated as "page 1"):
        [{"text": "...", "source": "filename.txt", "page": 1}]
    """
    filename = source_name or Path(file_path).name
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    logger.info("Loaded text/markdown '%s': %d characters.", filename, len(text))
    return [{"text": text, "source": filename, "page": 1}]


def load_json(file_path: str, source_name: str = None) -> list[dict]:
    """
    Load a JSON file containing structured records and convert each record
    into a natural-language text string suitable for embedding.

    Supports two layouts:

    1. NESTED (Banglalink-style API response):
       {
         "data": [
           { "type": "all", "title_en": "All", "packs": [ {...}, {...} ] },
           ...
         ]
       }
       Each pack under data[].packs[] becomes one "page".

    2. FLAT array:
       [ {"name": "...", "price": ...}, ... ]
       Each object in the top-level list becomes one "page".

    The natural-language rendering is what the embedding model sees, so it
    should be human-readable rather than raw JSON — this dramatically improves
    retrieval quality for questions like "cheapest 1-day pack" or
    "how do I activate 598 taka bundle".
    """
    filename = source_name or Path(file_path).name

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = []

    # ── Helper: convert one pack dict → natural language string ─────────────
    def pack_to_text(pack: dict, category: str = "") -> str:
        parts = []

        if category:
            parts.append(f"Category: {category}")

        name = pack.get("name_en") or pack.get("name") or pack.get("title_en", "")
        if name:
            parts.append(f"Pack: {name}")

        price = pack.get("price_tk") or pack.get("price")
        if price is not None:
            parts.append(f"Price: {price} Tk")

        validity = pack.get("validity_days")
        unit = pack.get("validity_unit", "days")
        if validity:
            parts.append(f"Validity: {validity} {unit}")

        mb = pack.get("internet_volume_mb")
        if mb:
            gb = round(mb / 1024, 1)
            parts.append(f"Data: {gb} GB ({mb} MB)")

        sms = pack.get("sms_volume")
        if sms:
            parts.append(f"SMS: {sms}")

        mins = pack.get("minute_volume")
        if mins:
            parts.append(f"Minutes: {mins}")

        call_rate = pack.get("callrate_offer") or pack.get("call_rate_unit")
        if call_rate:
            parts.append(f"Call rate: {call_rate}")

        ussd = pack.get("ussd_en") or pack.get("ussd")
        if ussd:
            parts.append(f"Activate: {ussd}")

        check_ussd = pack.get("balance_check_ussd_bn") or pack.get("balance_check_ussd")
        if check_ussd and not check_ussd.startswith("<"):  # skip HTML strings
            parts.append(f"Balance check: {check_ussd}")

        product_code = pack.get("product_code")
        if product_code:
            parts.append(f"Product code: {product_code}")

        return " | ".join(parts) if parts else json.dumps(pack, ensure_ascii=False)

    # ── Layout 1: nested API response ─────────────────────────────────────
    if isinstance(data, dict) and "data" in data:
        categories = data["data"]
        pack_num = 0
        for cat in categories:
            category_name = cat.get("title_en", "")
            for pack in cat.get("packs", []):
                text = pack_to_text(pack, category=category_name)
                if text:
                    pack_num += 1
                    pages.append({
                        "text":         text,
                        "source":       filename,
                        "page":         pack_num,
                        "price_tk":     float(pack.get("price_tk") or pack.get("price") or 0),
                        "validity_days": int(pack.get("validity_days") or 0),
                    })
        logger.info("Loaded JSON '%s': %d packs from nested structure.", filename, pack_num)

    # ── Layout 2: flat list ────────────────────────────────────────────────
    elif isinstance(data, list):
        for i, item in enumerate(data, start=1):
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = pack_to_text(item)
            else:
                continue
            if text:
                pages.append({"text": text, "source": filename, "page": i})
        logger.info("Loaded JSON '%s': %d records from flat list.", filename, len(pages))

    else:
        raise ValueError(
            f"Unsupported JSON shape in '{filename}'. "
            "Expected either a dict with a 'data' key, or a top-level list."
        )

    return pages


def load_document(file_path: str, source_name: str = None) -> list[dict]:
    """
    Dispatcher: choose loader based on file extension.

    source_name overrides the filename stored in chunk metadata. Pass it when
    file_path is a temporary path and the original filename should be preserved.
    """
    ext = Path(file_path).suffix.lower()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if ext == ".pdf":
        return load_pdf(file_path, source_name=source_name)
    elif ext in (".md", ".txt"):
        return load_text(file_path, source_name=source_name)
    elif ext == ".json":
        return load_json(file_path, source_name=source_name)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use .pdf, .md, .txt, or .json")
