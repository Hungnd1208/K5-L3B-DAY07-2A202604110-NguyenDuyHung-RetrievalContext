"""Run benchmark across multiple retrieval strategies (Lexical, Dense, Hybrid) on Dev and Test splits."""
import argparse
import json
from pathlib import Path
import sys

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import LexicalRetriever, DenseRetriever, HybridRetriever, evaluate_retrieval

CHUNKS_PATH = Path("starter/data/converted/chunks.json")
QUESTIONS_PATH = Path("starter/data/converted/questions.json")
RESULTS_DIR = Path("starter/results/retrieval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval strategies.")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    args = parser.parse_args()

    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loading questions from {QUESTIONS_PATH}...")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    splits = ["dev", "test"] if args.split == "all" else [args.split]

    print("\n--- Initializing Retrievers ---")
    lexical = LexicalRetriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical)

    configs = [
        {"name": "1_Baseline_Lexical", "retriever": lexical, "top_k": 3, "budget": 800, "desc": "Lexical BM25, top-3, budget 800"},
        {"name": "2_Dense_Retrieval", "retriever": dense, "top_k": 3, "budget": 800, "desc": "Dense Cosine, top-3, budget 800"},
        {"name": "3_Hybrid_RRF", "retriever": hybrid, "top_k": 3, "budget": 800, "desc": "Hybrid Dense+Lexical (RRF k=60), top-3, budget 800"},
        {"name": "4_Improve_Budget_1500", "retriever": hybrid, "top_k": 3, "budget": 1500, "desc": "Hybrid RRF, top-3, expanded budget 1500"},
        {"name": "5_Improve_TopK_5", "retriever": hybrid, "top_k": 5, "budget": 1500, "desc": "Hybrid RRF, top-5, expanded budget 1500"},
    ]

    all_summaries = {}

    for split in splits:
        q_split = [q for q in all_questions if q["split"] == split]
        print(f"\n==========================================")
        print(f"   EVALUATING SPLIT: {split.upper()} (N = {len(q_split)})")
        print(f"==========================================")

        for cfg in configs:
            # If test split, only run Baseline and Selected Config (e.g. 5_Improve_TopK_5)
            if split == "test" and cfg["name"] not in ["1_Baseline_Lexical", "5_Improve_TopK_5"]:
                continue

            summary, traces = evaluate_retrieval(
                retriever=cfg["retriever"],
                questions=q_split,
                top_k=cfg["top_k"],
                budget=cfg["budget"]
            )

            cfg_key = f"{cfg['name']}_{split}"
            all_summaries[cfg_key] = {
                "split": split,
                "config": cfg["desc"],
                **summary
            }

            # Save trace
            trace_file = RESULTS_DIR / f"traces_{cfg_key}.json"
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "traces": traces}, f, ensure_ascii=False, indent=2)

            print(f"[{split.upper()}] {cfg['name']}: Hit@1={summary['hit1']}/{summary['n']} ({summary['hit1_rate']}%) | "
                  f"Hit@3={summary['hit3']}/{summary['n']} ({summary['hit3_rate']}%) | "
                  f"ContextHit={summary['context_hit']}/{summary['n']} ({summary['context_evidence_rate']}%) | "
                  f"Latency={summary['median_ms']}ms")

    # Save summary table
    summary_file = RESULTS_DIR / "summary_table.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\nSaved all results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
