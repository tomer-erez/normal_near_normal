# Retrieval Metric Bounds — MIMIC-CXR Official Test Set

Gallery size: **5,159 images**. Metrics are macro-averaged over queries with at least one relevant image.

---

## Query Types

| Type | Total queries | Queries with n_rel > 0 | Median n_relevant | Mean n_relevant |
|---|---|---|---|---|
| Single (`"atelectasis"`) | 13 | 13 | 539 | 695 |
| Pair (`"atelectasis and edema"`) | 78 | 66 | 34 | 102 |
| Negative (`"atelectasis and no cardiomegaly"`) | 156 | 118 | 21 | 48 |

---

## P@k

**Upper bound is always 1.0** — a perfect model always retrieves only relevant items in the top-k.

**Random baseline = n_relevant / N** (prevalence of the label pattern), independent of k:

| Query type | Random P@k (any k) | Interpretation |
|---|---|---|
| Single | 0.1347 | Single labels are common — 13.5% of the gallery matches on average |
| Pair | 0.0197 | Co-occurrence is rarer — 2% match both labels |
| Negative | 0.0093 | Specific combination — 0.9% match "yes A and no B" |

Your model needs to be well above these baselines to claim meaningful retrieval.

---

## R@k

**Random baseline = k / N** — identical across all query types (does not depend on label prevalence):

| k | Random R@k |
|---|---|
| 1 | 0.0002 |
| 5 | 0.0010 |
| 10 | 0.0019 |

**Upper bound** depends on how many relevant images exist per query:

| Query type | R@1 upper | R@5 upper | R@10 upper | Note |
|---|---|---|---|---|
| Single | 0.0032 | 0.0159 | 0.0318 | Hundreds of relevant images — even a perfect model retrieves <4% of them in top-10 |
| Pair | 0.0603 | 0.2651 | 0.4109 | 21% of pair queries have <10 relevant images (R@10=1.0 achievable for those) |
| Negative | 0.1597 | 0.4214 | 0.5508 | 32% of queries have <10 relevant images — most dynamic range |

---

## What This Means in Practice

**Single queries**: R@k is nearly meaningless — the range from random (0.0002) to perfect (0.032) is tiny because there are hundreds of relevant images and you can only retrieve 10. **Focus on P@k here.** A good model should push P@10 from the 0.13 random baseline toward 1.0.

**Pair queries**: Both P@k and R@k are informative. The P@k random baseline (0.02) is low, so even moderate scores represent strong retrieval. R@10 upper bound is 0.41 — achievable.

**Negative queries**: The most diagnostically useful query type for the negative-aware loss. Random P@k is only 0.009, upper bound is 1.0, and R@10 can reach 0.55. This is where the difference between vanilla CLIP and negative-aware training should be most visible.

---

## Summary Table (all metrics, random vs. upper bound)

| Query | k | P@k random | P@k upper | R@k random | R@k upper |
|---|---|---|---|---|---|
| Single | 1 | 0.1347 | 1.0 | 0.0002 | 0.0032 |
| Single | 5 | 0.1347 | 1.0 | 0.0010 | 0.0159 |
| Single | 10 | 0.1347 | 1.0 | 0.0019 | 0.0318 |
| Pair | 1 | 0.0197 | 1.0 | 0.0002 | 0.0603 |
| Pair | 5 | 0.0197 | 1.0 | 0.0010 | 0.2651 |
| Pair | 10 | 0.0197 | 1.0 | 0.0019 | 0.4109 |
| Negative | 1 | 0.0093 | 1.0 | 0.0002 | 0.1597 |
| Negative | 5 | 0.0093 | 1.0 | 0.0010 | 0.4214 |
| Negative | 10 | 0.0093 | 1.0 | 0.0019 | 0.5508 |
