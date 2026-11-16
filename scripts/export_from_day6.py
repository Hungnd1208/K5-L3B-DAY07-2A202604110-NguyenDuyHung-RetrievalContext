"""Export chunks and questions from Day 6 to Day 7 format using Day 6 ingestion & chunking."""
import json
import re
import sys
from pathlib import Path

# Paths
DAY6_DIR = Path("../K5-L3B-DAY06-NguyenDuyHung-2A202604110-DataIngestion").resolve()
if not DAY6_DIR.exists():
    DAY6_DIR = Path("d:/VinUniLab/K5-L3B-DAY06-NguyenDuyHung-2A202604110-DataIngestion").resolve()

sys.path.insert(0, str(DAY6_DIR / "starter"))

from src.ingestion import load_pdf
from src.chunking import by_paragraph

PDF_PATH = DAY6_DIR / "starter/data/BÁO CÁO LÝ THUYẾT CNN.pdf"
BENCHMARK_PATH = DAY6_DIR / "benchmark/benchmark.json"

OUTPUT_DIR = Path("starter/data/converted")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"Loading PDF from {PDF_PATH}...")
    pages = load_pdf(PDF_PATH, document_id="bao-cao-cnn", source="starter/data/BÁO CÁO LÝ THUYẾT CNN.pdf")
    print(f"Loaded {len(pages)} content pages.")

    raw_chunks = []
    for p in pages:
        raw_chunks.extend(by_paragraph(p, max_size=700))
    print(f"Created {len(raw_chunks)} chunks (Matches Day 6 Strategy B/C exactly).")

    # Format chunks for Day 7
    all_chunks = []
    for c in raw_chunks:
        all_chunks.append({
            "id": c.chunk_id,
            "source_id": c.document_id,
            "source": c.source,
            "page": c.page,
            "text": c.text,
            "metadata": c.metadata
        })

    # Read Day 6 Benchmark questions
    print(f"Reading benchmark questions from {BENCHMARK_PATH}...")
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_raw = json.load(f)

    # Convert answerable questions to Day 7 format
    questions = []
    for q in bench_raw:
        evidence_str = q.get("evidence_quote", "")
        item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "gold": [{
                "source_id": q.get("gold_source_id", "bao-cao-cnn"),
                "page": q.get("gold_page"),
                "evidence": evidence_str
            }],
            "expected_answer": q["expected_answer"]
        }
        if q.get("filter"):
            item["filter"] = q["filter"]
        questions.append(item)

    # Ensure evidence string matches chunk text cleanly
    for q in questions:
        # Special case: q19 original quote was split across 2 chunks by 700-char paragraph chunking
        if q["id"] == "q19":
            q["gold"] = [
                {
                    "source_id": "bao-cao-cnn",
                    "page": 29,
                    "evidence": "CNN có khả năng tự động học và trích xuất các đặc trưng quan trọng từ dữ liệu thông qua các lớp tích chập."
                },
                {
                    "source_id": "bao-cao-cnn",
                    "page": 29,
                    "evidence": "Khác với các phương pháp truyền thống yêu cầu người dùng phải thiết kế đặc trưng thủ công,"
                }
            ]
            continue

        for g in q["gold"]:
            ev = g["evidence"]
            found = False
            for c in all_chunks:
                if c["source_id"] == g["source_id"] and c["page"] == g["page"] and ev in c["text"]:
                    found = True
                    break
            if not found:
                norm_ev = " ".join(ev.split())
                for c in all_chunks:
                    if c["source_id"] == g["source_id"] and c["page"] == g["page"]:
                        norm_c = " ".join(c["text"].split())
                        if norm_ev in norm_c:
                            words = ev.split()
                            pattern = r"\s+".join(re.escape(w) for w in words)
                            m = re.search(pattern, c["text"])
                            if m:
                                g["evidence"] = m.group(0)
                                found = True
                                break
            if not found:
                print(f"WARNING: Evidence for {q['id']} not found in target page {g['page']} chunk!")

    # 4 Unanswerable questions: 2 for dev, 2 for test
    unanswerable_questions = [
        {
            "id": "u01",
            "split": "dev",
            "question": "Cơ chế Multi-Head Self-Attention trong kiến trúc Transformer hoạt động như thế nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu chỉ tập trung vào CNN và Deep Learning truyền thống (ANN, RNN), không có chương mục nào về Transformer hay Self-Attention."
        },
        {
            "id": "u02",
            "split": "dev",
            "question": "Thuật toán phát hiện đối tượng YOLOv8 cải tiến kiến trúc Backbone và Head như thế nào so với YOLOv5?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu chỉ nêu khái niệm chung về bài toán Object Detection, không đề cập đến họ mô hình YOLO hay phiên bản YOLOv8."
        },
        {
            "id": "u03",
            "split": "test",
            "question": "Phòng thí nghiệm sử dụng bao nhiêu GPU NVIDIA A100 để huấn luyện mô hình ResNet-50 trong báo cáo này?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu là báo cáo lý thuyết thuần túy, không có thông số cấu hình cụ thể về số lượng GPU hay phòng lab thực nghiệm."
        },
        {
            "id": "u04",
            "split": "test",
            "question": "Kỹ thuật FlashAttention-2 tối ưu hóa bộ nhớ SRAM và HBM trên chip GPU như thế nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu không chứa thông tin về kỹ thuật tối ưu phần cứng FlashAttention."
        }
    ]

    # Combine into cases for Agent benchmark
    cases = []
    for q in questions:
        cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })
    cases.extend(unanswerable_questions)

    # Save chunks.json
    chunks_file = OUTPUT_DIR / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks to {chunks_file} ({len(all_chunks)} chunks)")

    # Save questions.json
    questions_file = OUTPUT_DIR / "questions.json"
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Saved retrieval questions to {questions_file} ({len(questions)} questions)")

    # Save cases.json
    cases_file = OUTPUT_DIR / "cases.json"
    with open(cases_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"Saved agent cases to {cases_file} ({len(cases)} cases: {sum(1 for c in cases if c['split']=='dev')} dev, {sum(1 for c in cases if c['split']=='test')} test)")


if __name__ == "__main__":
    main()
