# SpamShield — One-Page Technical Summary
**Team 1 · MSDS 630 Residency NLP Hackathon**

**Problem.** A mobile carrier wants to block scam texts without ever blocking a real message from family or a doctor, so the priority metric is precision / false-positive rate, not accuracy.

**Data.** UCI SMS Spam Collection: 5,574 English messages, 13.4% spam. After removing 415 exact duplicates: 5,159 messages (12.4% spam), split stratified 3,611 train / 774 dev / 774 test (seed 42).

**Method.** (1) Baseline: Naive Bayes from scratch, add-1 smoothing, log-space scoring, URL token + lowercase tokenizer. (2) scikit-learn MultinomialNB and logistic regression on word+bigram TF-IDF (vectorizer fit on train only). (3) Decision threshold tuned on dev to target FPR ≤ 0.5%, applied once to test. (4) Stretch: leetspeak robustness with/without a de-leet normalizer.

**Results (test set, spam = positive class).**
| model | precision | recall | F1 | FPR |
|---|---|---|---|---|
| Always predict ham | – | 0.000 | – | 0.000 (accuracy 0.876) |
| NB from scratch, thr 0.5 | 0.944 | 0.885 | 0.914 | 0.0074 |
| NB from scratch, tuned thr | 0.924 | 0.885 | 0.904 | 0.0103 |
| sklearn MultinomialNB (TF-IDF) | 1.000 | 0.698 | 0.822 | 0.0000 |
| Logistic regression, thr 0.5 | 1.000 | 0.875 | 0.933 | 0.0000 |
| **Logistic regression, tuned thr** | **0.989** | **0.938** | **0.963** | **0.0015** |

Leetspeak ("FR33 PR1ZE"): NB recall fell 0.885 → 0.833; de-leet normalization restored it to 0.885 without raising false positives.

**Findings to discuss.** Logistic regression beat Naive Bayes on every metric that matters to the client. Our NB threshold tuned on dev did *not* transfer: test FPR was 1.03%, above the 0.5% target. NB probabilities are extremely overconfident (mostly near 0 or 1), so threshold tuning is unstable. With ~678 legitimate test messages, one false positive moves FPR by 0.15 points, so these estimates are noisy.

**Limitations.** Small, old (2000s UK/Singapore) dataset; modern scams (toll notices, package-hold links) are poorly covered — on our 7 hand-written modern messages NB missed a toll-payment scam and flagged a legitimate appointment confirmation; that probe is anecdotal. Exact-duplicate removal does not catch near-duplicates. De-leet heuristics can mangle legitimate tokens.

**Next steps.** Collect a larger modern labeled set; calibrate probabilities; cross-validate the threshold; try DistilBERT; add sender metadata.
