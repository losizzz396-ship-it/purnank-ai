"""
Retrieval evaluation harness for Purnank's vector stores.

WHY THIS EXISTS
----------------
"Train the AI to know the textbooks better" is really a retrieval-quality
problem, and retrieval quality is invisible unless you measure it. This
script lets you build a small labeled test set once, then re-run it any
time you change chunking, hybrid weighting, or reranking, to see whether
retrieval actually got better or just *felt* different.

HOW TO USE
----------
1. Fill in EVAL_SET below with real questions from your subject area and,
   for each, a substring you know should appear in a genuinely correct
   retrieved chunk (e.g. a key term, formula name, or definition phrase).
   Start with 15-30 — you don't need hundreds to catch regressions.
2. Run:  python eval_retrieval.py
3. It reports hit-rate (was a correct-looking chunk retrieved at all) and
   mean reciprocal rank (how high up it was) for each configuration, so you
   can compare semantic-only vs hybrid vs hybrid+rerank on YOUR content.

This does not require your Groq key or network access — it only exercises
the vector stores, so you can run it locally as often as you like.
"""
import pickle
import numpy as np

# ---------------------------------------------------------------------------
# EDIT THIS: real questions from your subject areas, each paired with a
# substring that must appear in a chunk for it to count as a correct hit.
# Keep the substring short and distinctive (a term/phrase, not a full sentence).
# ---------------------------------------------------------------------------
EVAL_SET = [
    # {"query": "what is Ohm's law", "expect_substring": "Ohm"},
    # {"query": "difference between mitosis and meiosis", "expect_substring": "meiosis"},
    # {"query": "define photosynthesis", "expect_substring": "photosynthesis"},
]

CONFIGS = [
    {"name": "semantic-only", "use_hybrid": False, "rerank": False},
    {"name": "hybrid", "use_hybrid": True, "rerank": False},
    {"name": "hybrid+rerank", "use_hybrid": True, "rerank": True},
]

N_RESULTS = 5


def evaluate(store, eval_set, config):
    hits = 0
    reciprocal_ranks = []
    for item in eval_set:
        result = store.query(
            item["query"],
            n_results=N_RESULTS,
            use_hybrid=config["use_hybrid"],
            rerank=config["rerank"],
        )
        docs = result["documents"][0]
        rank = None
        for i, doc in enumerate(docs, start=1):
            if item["expect_substring"].lower() in doc.lower():
                rank = i
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(eval_set)
    hit_rate = hits / n if n else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return hit_rate, mrr


def main():
    if not EVAL_SET:
        print("⚠️ EVAL_SET is empty. Add 15-30 labeled (query, expect_substring) "
              "pairs at the top of this file, then rerun.")
        return

    with open("textbook_store.pkl", "rb") as f:
        textbook_store = pickle.load(f)

    print(f"Evaluating on {len(EVAL_SET)} queries, top-{N_RESULTS} retrieval\n")
    print(f"{'Config':<18} {'Hit Rate':<12} {'MRR':<8}")
    print("-" * 40)
    for config in CONFIGS:
        hit_rate, mrr = evaluate(textbook_store, EVAL_SET, config)
        print(f"{config['name']:<18} {hit_rate*100:>6.1f}%      {mrr:.3f}")

    print("\nHit Rate = % of queries where a correct chunk appeared in top-"
          f"{N_RESULTS}. MRR = how high up it ranked on average (1.0 = always #1).")
    print("If hybrid+rerank isn't clearly best on your content, the extra "
          "latency probably isn't worth it — try adjusting hybrid_alpha "
          "in vector_store.py instead.")


if __name__ == "__main__":
    main()
