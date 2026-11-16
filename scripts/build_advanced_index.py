"""Multimodal Ingestion & Indexing Pipeline for Advanced 100-page Challenge.
Intelligent Hybrid Engine:
- Auto-detects input PDF in data/ directory or via --pdf argument.
- Pages with embedded digital text are parsed instantly via PyMuPDF.
- Visual graphic/image pages are rendered to optimized JPEGs and parsed via Gemini 3.1 Flash-Lite (2.8s/page).
Outputs unified knowledge chunks to data/advanced/multimodal_chunks.json.
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import pymupdf

load_dotenv()

ADVANCED_DIR = Path("data/advanced")
IMAGES_DIR = ADVANCED_DIR / "images"
CACHE_DIR = ADVANCED_DIR / "cache"
OUTPUT_CHUNKS_FILE = ADVANCED_DIR / "multimodal_chunks.json"

ADVANCED_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VISION_PROMPT = """Bạn là chuyên gia phân tích báo cáo tài chính và thị giác máy tính.
Hãy phân tích hình ảnh trang báo cáo thường niên này và bóc tách ĐẦY ĐỦ, CHÍNH XÁC:
1. TIÊU ĐỀ & VĂN BẢN (Text): Giữ nguyên văn các đề mục, đoạn văn bản, tên người, chức danh, chiến lược.
2. BẢNG BIỂU (Tables): Chuyển đổi mọi bảng số liệu sang định dạng Markdown table kèm tên cột, hàng và đơn vị đo (đồng, tỷ đồng, %, tấn, triệu lít,...).
3. BIỂU ĐỒ (Charts): Nếu có biểu đồ (cột, đường, tròn,...), hãy bóc tách:
   - Loại biểu đồ và tiêu đề
   - Trục hoành (năm/kỳ), trục tung (đơn vị tính)
   - Tất cả giá trị số liệu cụ thể hiển thị trên biểu đồ
   - Chú giải (legend) và xu hướng.
4. SƠ ĐỒ / HÌNH ẢNH (Diagrams): Mô tả cơ cấu tổ chức, sơ đồ chuỗi giá trị, quy trình.

Yêu cầu: Trung thực tuyệt đối, không suy diễn ngoài ảnh, giữ nguyên tiếng Việt có dấu."""


def find_target_pdf(custom_path: str = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    data_dir = Path("data")
    pdf_files = list(data_dir.glob("*.pdf")) + list(Path(".").glob("*.pdf")) + list(Path("starter/data").glob("*.pdf"))
    # Filter out repaired or temporary
    valid_pdfs = [p for p in pdf_files if "repaired" not in p.name.lower()]

    if not valid_pdfs:
        raise FileNotFoundError("Không tìm thấy file PDF nào trong thư mục data/")

    # Pick the most recently modified PDF
    valid_pdfs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return valid_pdfs[0]


def init_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY không tìm thấy trong .env")
    from google import genai
    return genai.Client(api_key=api_key)


def main():
    parser = argparse.ArgumentParser(description="Multimodal PDF Ingestion")
    parser.add_argument("--pdf", type=str, default=None, help="Đường dẫn file PDF")
    args = parser.parse_args()

    pdf_path = find_target_pdf(args.pdf)
    print(f"\n==========================================")
    print(f"   MULTIMODAL INGESTION PIPELINE")
    print(f"==========================================")
    print(f"Tài liệu mục tiêu: {pdf_path.resolve()} ({pdf_path.stat().st_size / 1024 / 1024:.2f} MB)")

    doc = pymupdf.open(str(pdf_path))
    total_pages = len(doc)
    print(f"Tổng số trang: {total_pages} trang (Yêu cầu >= 100 trang: {'ĐẠT' if total_pages >= 100 else 'KHÔNG ĐẠT'})")

    client = init_gemini_client()
    candidate_models = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]

    all_chunks = []
    text_page_count = 0
    vision_page_count = 0

    for idx, page in enumerate(doc):
        page_num = idx + 1
        cache_file = CACHE_DIR / f"page_{page_num:03d}.json"

        # Check existing cache
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_chunk = json.load(f)
                if cached_chunk.get("text"):
                    all_chunks.append(cached_chunk)
                    print(f"[{page_num:03d}/{total_pages}] Nạp từ cache ({cached_chunk.get('type')}).")
                    continue
            except Exception:
                pass

        # Try digital text extraction first
        raw_text = page.get_text() or ""
        cleaned_text = "\n".join(l.strip() for l in raw_text.split("\n") if l.strip())

        # If digital text is rich (> 150 chars), keep directly (instant 0.001s)
        if len(cleaned_text) > 150:
            text_page_count += 1
            chunk_data = {
                "id": f"doc_p{page_num:03d}_text",
                "source_id": pdf_path.stem,
                "source": str(pdf_path.as_posix()),
                "page": page_num,
                "type": "digital_text_table",
                "text": cleaned_text,
                "image_ref": None,
                "metadata": {
                    "page": page_num,
                    "char_count": len(cleaned_text),
                    "is_visual": False
                }
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            all_chunks.append(chunk_data)
            print(f"[{page_num:03d}/{total_pages}] Trích xuất text kỹ thuật số ({len(cleaned_text)} ký tự).")
            continue

        # Otherwise, page is visual graphic / chart / scanned image
        # Render optimized lightweight image (100 DPI -> ~20 KB)
        vision_page_count += 1
        pix = page.get_pixmap(dpi=100)
        img_path = IMAGES_DIR / f"page_{page_num:03d}.jpg"
        pix.save(str(img_path))
        img_size_kb = img_path.stat().st_size / 1024

        print(f"[{page_num:03d}/{total_pages}] Gọi Gemini Vision bóc tách ảnh đồ họa ({img_size_kb:.1f} KB)...")

        transcription = ""
        max_retries = 3
        for attempt in range(max_retries):
            model_to_use = candidate_models[min(attempt, len(candidate_models) - 1)]
            try:
                with open(img_path, "rb") as f_img:
                    img_bytes = f_img.read()

                from google.genai import types
                t0 = time.time()
                res = client.models.generate_content(
                    model=model_to_use,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        VISION_PROMPT
                    ],
                    config={"temperature": 0.1}
                )
                elapsed = time.time() - t0
                transcription = res.text.strip()
                print(f"    --> Phân tích thành công bằng {model_to_use} ({elapsed:.1f}s, {len(transcription)} ký tự)")
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_sec = 15
                    print(f"    --> Quota rate limit. Tạm dừng {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    print(f"    --> Thử lại lần {attempt+1}/{max_retries} với {model_to_use}: {e}")
                    time.sleep(2)

        if not transcription:
            print(f"    --> Cảnh báo: Trang {page_num} không có văn bản, tạo chunk ghi chú ảnh.")
            transcription = f"[Trang {page_num} là hình ảnh đồ họa / ảnh bìa minh họa]"

        chunk_data = {
            "id": f"doc_p{page_num:03d}_visual",
            "source_id": pdf_path.stem,
            "source": str(pdf_path.as_posix()),
            "page": page_num,
            "type": "visual_multimodal",
            "text": transcription,
            "image_ref": str(img_path.as_posix()),
            "metadata": {
                "page": page_num,
                "is_visual": True,
                "image_path": str(img_path.as_posix()),
                "char_count": len(transcription)
            }
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        all_chunks.append(chunk_data)
        time.sleep(1)

    # Save final unified chunks
    with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n==========================================")
    print(f"   HOÀN TẤT NẠP DỮ LIỆU MULTIMODAL")
    print(f"==========================================")
    print(f"Tổng số chunk: {len(all_chunks)} chunks từ {total_pages} trang.")
    print(f" - Trang text kỹ thuật số: {text_page_count}")
    print(f" - Trang đồ họa / ảnh phân tích qua Vision: {vision_page_count}")
    print(f"Dữ liệu đã lưu tại: {OUTPUT_CHUNKS_FILE}")


if __name__ == "__main__":
    main()
