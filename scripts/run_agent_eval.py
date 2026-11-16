"""Run and evaluate Agent on Benchmark cases (Dev or Test).
Generates traces.json, reviews.json, and aggregated agent summary metrics.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root and starter to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "starter"))

from starter.agentic import run_agent
from starter.agent_metrics import aggregate
from src.retrieval import LexicalRetriever, DenseRetriever, HybridRetriever
from src.agent import GeminiController

CASES_PATH = Path("starter/data/converted/cases.json")
QUESTIONS_PATH = Path("starter/data/converted/questions.json")
CHUNKS_PATH = Path("starter/data/converted/chunks.json")
RESULTS_DIR = Path("starter/results/agent")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class RetrieverAdapter:
    """Adapts Day 7 retrievers to the Protocol required by run_agent."""
    def __init__(self, retriever):
        self.retriever = retriever

    def search(self, query: str, *, filters: dict = None, top_k: int = 3) -> list[dict]:
        return self.retriever.search(query=query, top_k=top_k, filters=filters)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent on Benchmark cases.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--strategy", choices=["lexical", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_searches", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading benchmark cases from {CASES_PATH}...")
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    print(f"Loading retrieval questions from {QUESTIONS_PATH}...")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        retrieval_questions = {q["id"]: q for q in json.load(f)}

    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        chunks_map = {c["id"]: c for c in chunks}

    # Filter cases for requested split
    cases = [c for c in all_cases if c["split"] == args.split]
    print(f"Running Agent on {len(cases)} cases for split: {args.split.upper()} (Strategy: {args.strategy}, Budget: {args.budget})")

    # Initialize retriever
    lexical = LexicalRetriever(chunks)
    if args.strategy == "lexical":
        base_retriever = lexical
    elif args.strategy == "dense":
        base_retriever = DenseRetriever(chunks)
    else:
        dense = DenseRetriever(chunks)
        base_retriever = HybridRetriever(dense, lexical)

    retriever = RetrieverAdapter(base_retriever)
    controller = GeminiController()

    traces = []
    reviews = []
    t_start = time.time()

    for idx, case in enumerate(cases):
        cid = case["id"]
        q_text = case["question"]
        is_answerable = case["answerable"]
        filters = retrieval_questions.get(cid, {}).get("filter")

        print(f"\n[{idx+1}/{len(cases)}] Processing {cid} (answerable={is_answerable}): {q_text[:70]}...")
        t0 = time.time()
        try:
            res = run_agent(
                question=q_text,
                controller=controller,
                retriever=retriever,
                filters=filters,
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget
            )
            elapsed = time.time() - t0
            citations = res.get("citations", [])
            trace_entry = {
                "query_id": cid,
                "question": q_text,
                "status": res["status"],
                "answer": res.get("answer", ""),
                "citations": citations,
                "events": res.get("events", []),
                "latency_sec": round(elapsed, 2)
            }
            traces.append(trace_entry)

            # Review evaluation matching Day 7 Rubric
            if is_answerable:
                q_meta = retrieval_questions.get(cid, {})
                gold_list = q_meta.get("gold", [])

                # Check citation support
                citation_support = []
                for cite_id in citations:
                    c_chunk = chunks_map.get(cite_id)
                    supported = False
                    if c_chunk:
                        for g in gold_list:
                            if c_chunk["page"] == g.get("page") and g.get("evidence", "") in c_chunk["text"]:
                                supported = True
                                break
                    citation_support.append(supported)

                # Answer grade: 2 (supported & answered), 1 (partial), 0 (abstain / wrong)
                if res["status"] == "answered" and any(citation_support):
                    answer_grade = 2
                elif res["status"] == "answered":
                    answer_grade = 1
                else:
                    # Abstaining on answerable question is scored 0 per benchmark spec
                    answer_grade = 0

                reviews.append({
                    "query_id": cid,
                    "answer_grade": answer_grade,
                    "citation_support": citation_support
                })
            else:
                # Unanswerable question: check abstention
                abstention_correct = (res["status"] == "abstain")
                # Every citation (if any mistakenly produced) must have a boolean in citation_support
                citation_support = [False] * len(citations)
                reviews.append({
                    "query_id": cid,
                    "citation_support": citation_support,
                    "abstention_correct": abstention_correct
                })

            print(f"    --> Status: {res['status']} | Citations: {res.get('citations', [])} | Answer: {res.get('answer', '')[:90]}...")

        except Exception as e:
            print(f"    --> Error executing {cid}: {e}")
            traces.append({
                "query_id": cid,
                "question": q_text,
                "status": "error",
                "error": str(e),
                "events": [],
                "citations": []
            })

    total_time = time.time() - t_start

    # Save traces and reviews
    prefix = f"{args.strategy}_{args.split}"
    trace_path = RESULTS_DIR / f"traces_{prefix}.json"
    reviews_path = RESULTS_DIR / f"reviews_{prefix}.json"
    summary_path = RESULTS_DIR / f"summary_{prefix}.json"

    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    with open(reviews_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    # Compute aggregate metrics via starter/agent_metrics.py
    metrics = aggregate(cases, traces, reviews)
    metrics["total_time_sec"] = round(total_time, 2)
    metrics["strategy"] = args.strategy
    metrics["split"] = args.split

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n==========================================")
    print(f"   AGENT BENCHMARK RESULTS ({args.split.upper()} - {args.strategy.upper()})")
    print("==========================================")
    print(f"Total Cases: {len(cases)} ({sum(1 for c in cases if c['answerable'])} answerable, {sum(1 for c in cases if not c['answerable'])} unanswerable)")
    print(f"Tool Use Rate: {metrics['tool_use']['n']}/{metrics['tool_use']['N']} ({round(metrics['tool_use']['n']/max(metrics['tool_use']['N'],1)*100, 1)}%)")
    print(f"Grounded Answer: mean = {metrics['grounded_answer']['mean']:.2f}/2.0 | [0: {metrics['grounded_answer']['0']}, 1: {metrics['grounded_answer']['1']}, 2: {metrics['grounded_answer']['2']}]")
    print(f"Citation Support: {metrics['citation_validity']['n']}/{metrics['citation_validity']['N']} valid")
    print(f"Abstention Accuracy: {metrics['abstention']['n']}/{metrics['abstention']['N']}")
    print(f"Total Time: {metrics['total_time_sec']}s")
    print(f"Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
