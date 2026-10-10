# SpamShield — One-Page Technical Summary
**Team 1 · [team member names] · MSDS 630 Residency NLP Hackathon**
(The formatted one-page version is `SpamShield_Technical_Summary.docx` / `.pdf`; this is the same content as plain text.)

**Problem.** A regional carrier wants to block scam texts without ever blocking a real message from family or a doctor. False positives cost far more than missed spam, so we lead with spam precision and the false-positive rate, not accuracy (predicting "not spam" for everything scores 87.6%).

**Data.** UCI SMS Spam Collection: 5,574 English messages (13.4% spam). 415 exact duplicates removed before splitting, leaving 5,159 (12.4% spam), split stratified 3,611 train / 774 dev / 774 test (seed 42). Vectorizer fit on train only; threshold tuned on dev; test scored once.

**Method.** (1) Naive Bayes from scratch (add-1 smoothing, log-space). (2) scikit-learn Naive Bayes and logistic regression on word+bigram TF-IDF. (3) Threshold tuned on dev for FPR <= 0.5%, applied once to test. (4) Leetspeak robustness with/without a de-leet normalizer; 7 hand-written modern scam-style messages.

**Results (test set: 96 spam, 678 real; spam = positive class).**
| model | precision | recall | F1 | FPR |
|---|---|---|---|---|
| Naive Bayes (scratch), threshold 0.5 | 0.944 | 0.885 | 0.914 | 0.74% |
| Naive Bayes (scratch), dev-tuned threshold | 0.924 | 0.885 | 0.904 | 1.03% |
| scikit-learn Naive Bayes, TF-IDF | 1.000 | 0.698 | 0.822 | 0.00% |
| Logistic regression, threshold 0.5 | 1.000 | 0.875 | 0.933 | 0.00% |
| **Logistic regression, dev-tuned threshold** | **0.989** | **0.938** | **0.963** | **0.15%** |

Logistic regression has the best point estimates, but with 5 vs 1 false positives out of 678 the 95% intervals overlap (0.32–1.71% vs 0.03–0.83%), so the ranking is suggestive, not proven. Leetspeak cut Naive Bayes recall from 0.885 to 0.833; the normalizer restored 0.885 with no extra false positives.

**Limitations.** Older, largely UK and Singapore data; on 7 modern scam-style messages we wrote, the model missed a toll-payment scam and blocked an appointment confirmation (anecdotal). The dev-tuned Naive Bayes threshold did not transfer (0.44% FPR on dev, 1.03% on test); 92.8% of its probabilities are above 0.99 or below 0.01. Near-duplicates are not removed, and one extra false positive moves the rate by 0.15 points.

**Next steps.** Collect 30–50 modern labeled scam messages; calibrate probabilities and choose the threshold by cross-validation; add features for phone numbers and per-minute prices; fine-tune DistilBERT.
