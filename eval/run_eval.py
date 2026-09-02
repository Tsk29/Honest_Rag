"""Evaluation harness for the CRAG relevance grader.

The relevance grader (see DEFAULT_RELEVANCY_PROMPT_TEMPLATE in workflow.py) is
the single most trust-critical judgment in this pipeline: it decides whether
the app trusts local document retrieval or falls back to a Firecrawl web
search. If it's wrong, either good local context gets thrown away for a web
search, or bad/irrelevant context gets treated as relevant and fed straight
into the final answer. There was previously zero measurement of how good
this judgment actually is - this script is that measurement.

It imports DEFAULT_RELEVANCY_PROMPT_TEMPLATE directly from workflow.py (never
reimplements it) and parses the LLM's yes/no output the same way
eval_relevance does: strip any <think>...</think> block, then treat the
presence of the substring "yes" in the remaining text as a positive call.
This deliberately does NOT call CorrectiveRAGWorkflow's internals - just the
prompt template and the grading contract ("chunk + query in, yes/no out") -
so the harness keeps working if eval_relevance itself gets refactored.

Usage:
    OPENAI_API_KEY=sk-...  python eval/run_eval.py           # real evaluation
    python eval/run_eval.py --mock                            # harness smoke test only
    python eval/run_eval.py                                   # auto-falls back to --mock if no key

--mock uses a crude deterministic keyword-overlap heuristic instead of a real
LLM. It exists ONLY so the harness's data loading and metrics computation can
be verified to work without an API key. It says nothing about how good the
real grader is - only a real OPENAI_API_KEY run does that.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow import DEFAULT_RELEVANCY_PROMPT_TEMPLATE

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.jsonl")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to",
    "for", "what", "how", "does", "do", "did", "and", "or", "with", "at",
    "by", "from", "this", "that", "it", "its",
}


def load_dataset(path=DATASET_PATH):
    """Load the labeled eval set from JSONL: {query, document_chunk, label}."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def parse_relevance(raw_text):
    """Same parsing eval_relevance uses: strip <think> blocks, then check
    whether 'yes' appears anywhere in the lowercased remaining text."""
    stripped = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    return "yes" if "yes" in stripped.lower() else "no"


def mock_complete(query_str, document_chunk):
    """Deterministic keyword-overlap heuristic used ONLY in --mock mode to
    smoke-test the harness itself (dataset loading, prompt formatting,
    parsing, metrics). It is not a stand-in for the real grader's quality -
    do not read anything into mock-mode metrics beyond "the plumbing works".
    """
    query_words = set(re.findall(r"[a-z]+", query_str.lower())) - STOPWORDS
    chunk_words = set(re.findall(r"[a-z]+", document_chunk.lower())) - STOPWORDS
    overlap = query_words & chunk_words
    return "yes" if len(overlap) >= 2 else "no"


def real_complete(llm, prompt):
    return llm.complete(prompt).text


def grade_example(example, llm, use_mock):
    query_str = example["query"]
    document_chunk = example["document_chunk"]
    prompt = DEFAULT_RELEVANCY_PROMPT_TEMPLATE.format(
        context_str=document_chunk, query_str=query_str
    )
    if use_mock:
        raw = mock_complete(query_str, document_chunk)
    else:
        raw = real_complete(llm, prompt)
    return parse_relevance(raw), raw


def compute_metrics(results):
    tp = fp = tn = fn = 0
    for r in results:
        if r["label"] == "yes" and r["prediction"] == "yes":
            tp += 1
        elif r["label"] == "no" and r["prediction"] == "yes":
            fp += 1
        elif r["label"] == "no" and r["prediction"] == "no":
            tn += 1
        elif r["label"] == "yes" and r["prediction"] == "no":
            fn += 1

    total = len(results)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the CRAG relevance grader (DEFAULT_RELEVANCY_PROMPT_TEMPLATE) against a labeled dataset."
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use a deterministic mock LLM instead of a real API call. For testing the harness itself only.",
    )
    parser.add_argument("--dataset", default=DATASET_PATH, help="Path to the eval dataset (JSONL).")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use for real evaluation.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    use_mock = args.mock or not api_key

    if use_mock and not args.mock:
        print("WARNING: OPENAI_API_KEY is not set - auto-falling back to --mock mode.")
        print("This only proves the harness scaffolding works. It does NOT evaluate the real grader.")
        print()

    llm = None
    if not use_mock:
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model=args.model, api_key=api_key)

    examples = load_dataset(args.dataset)

    if use_mock:
        print("=" * 72)
        print("MOCK MODE - using a keyword-overlap heuristic, NOT a real LLM.")
        print("This only proves data loading + metrics computation work.")
        print("For a real evaluation: OPENAI_API_KEY=sk-... python eval/run_eval.py")
        print("=" * 72)
        print()

    results = []
    for example in examples:
        prediction, raw = grade_example(example, llm, use_mock)
        results.append({
            "id": example.get("id"),
            "query": example["query"],
            "document_chunk": example["document_chunk"],
            "label": example["label"],
            "prediction": prediction,
            "raw_response": raw,
        })

    metrics = compute_metrics(results)

    print(f"Examples evaluated: {len(results)}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")
    print()

    wrong = [r for r in results if r["prediction"] != r["label"]]
    if wrong:
        print(f"--- {len(wrong)} misclassified example(s) ---")
        for r in wrong:
            print(f"[id={r['id']}] label={r['label']}  predicted={r['prediction']}")
            print(f"  query: {r['query']}")
            print(f"  chunk: {r['document_chunk'][:150]}")
            print(f"  raw response: {r['raw_response']!r}")
            print()
    else:
        print("No misclassifications.")


if __name__ == "__main__":
    main()
