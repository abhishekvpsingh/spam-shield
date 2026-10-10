# Slide outline (≤10 slides, 10 min + 5 Q&A) — each member speaks
1. **Client problem** — scam texts; false positives costlier than missed spam (Lead)
2. **Data** — 5,574 msgs, 13% spam; 415 duplicates removed; why accuracy is a trap (87.6%) (Data Eng.)
3. **Approach & splits** — train/dev/test, vectorizer on train only, threshold on dev (Data Eng.)
4. **Baseline: Naive Bayes from scratch** — formula, add-1, matches our Assignment 1 hand calc (Modeling)
5. **LIVE DEMO** — paste message → verdict + driving words (Demo Lead)
6. **Results table** — baseline vs. improved; confusion matrices (`results/confusion_matrices.png`) (Evaluation)
7. **Error analysis** — five misclassified messages, explained in our own words (Evaluation)
8. **Robustness** — leetspeak test; modern-scam probe (Modeling)
9. **Limitations & ethics** — dataset age, overconfident NB, threshold instability, cost of false positives (Lead)
10. **Next week** — modern data, calibration, DistilBERT
