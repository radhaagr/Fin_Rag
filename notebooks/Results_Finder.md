**What the numbers show, plainly:**

Recall climbs steadily through k=30, then flattens: +0.130 from k=5→10, +0.063 from k=10→20, +0.090 from k=20→30, then only +0.008 (k=30→40) and +0.016 (k=40→50). 
The recall_captured_frac=0.92 at k=30 confirms this numerically — 92% of the total recall gain across your tested range has already happened by k=30.

NDCG rises with recall up to k=30 (0.269→0.341), then dips at k=40 (0.337) before a small partial recovery at k=50 (0.339). 
This is why ndcg_captured_frac = 1.02 — over 100%, which only happens when the metric isn't monotonic. 
NDCG peaking mid-range rather than at your largest k is itself a finding, not an artifact of the elbow function.
Both elbows land at the same k=30. That's a more useful result than the "recall plateaus first, NDCG keeps climbing" pattern we discussed hypothetically earlier — here, 
this specific config converges on both fronts at the same point.

**What I'd conclude for this (splitter, retriever) config specifically:**

**k=30 is a genuine, joint sweet spot** — retrieving more than the top 30 chunks buys almost nothing on recall and appears to slightly hurt ranking quality rather than help it. 
If this config is what you deploy, there's little reason to pass more than ~30 chunks downstream.
The NDCG dip past k=30 suggests relevant docs found at deeper ranks (31–40) aren't well-ordered relative to what's already found — consistent with the earlier observation that 
recall and NDCG can diverge (finding more ≠ ranking well). 
This is a specific, concrete argument for adding the cross-encoder reranker on top of retrieval at k=30: rerank the top-30 candidates rather than expanding the pool further, 
since expanding the pool isn't the lever that's working here.
Absolute NDCG (~0.34 at its best) is well below recall (~0.60) at the same k — the gap we flagged earlier as a ranking problem, not a coverage problem, still holds at every k in this table.

One caveat before treating the k=40 dip as a real effect rather than noise: n_queries_scored=61 is a fairly small sample. 
A 0.004–0.008 NDCG swing between adjacent k values could plausibly be within normal query-to-query variance at that sample size rather than a genuine "ranking gets worse past k=30" phenomenon. 
Before concluding this dip is meaningful (versus recall, which moved by 0.130+ early on and is unambiguous), 
I'd want to see whether it holds up: either bootstrap a confidence interval over the 61 queries, or check whether the same dip-then-recover pattern shows up for other configs in your 
sweep at the same k values — if it's config-specific and small, it's likely noise; if it appears consistently across configs, it's a real property of the corpus/qrels at that depth.
