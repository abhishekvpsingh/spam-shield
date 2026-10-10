# SpamShield — MSDS 630 Residency NLP Hackathon (Project 1)

SMS spam filter for a carrier where **false positives cost far more than missed spam**.
Naive Bayes is implemented from scratch (counts + add-1 smoothing, Jurafsky & Martin Ch. 4) and compared
against scikit-learn Naive Bayes and TF-IDF logistic regression.

## Setup and run
```bash
pip install -r requirements.txt          # Python 3.10+
# data is already in data/sms.tsv (UCI SMS Spam Collection, 5,574 messages; mirror of the UCI/Kaggle file)
python -m spamshield.nb                  # sanity check: reproduces our Assignment 1 hand calculations
python run_experiments.py                # all experiments -> results/results.md, metrics.csv, confusion_matrices.png
streamlit run app.py                     # live demo (runs fully offline)
```

## Layout
| file | purpose |
|---|---|
| `spamshield/nb.py` | tokenizer + Naive Bayes from scratch + per-word explanation |
| `run_experiments.py` | dedupe, split, train, evaluate, threshold tuning, error analysis, adversarial test |
| `app.py` | Streamlit demo: verdict + probability + words that drove it |
| `results/` | generated tables and figures (re-created by `run_experiments.py`) |
| `summary.md`, `slides_outline.md` | one-page technical summary; 10-slide outline |

## Method notes (what protects the numbers)
- Exact duplicates removed **before** splitting (415 dropped), so test messages are not copies of training messages.
- Stratified 70/15/15 train/dev/test split, fixed seed. Threshold tuned on **dev only**, then applied once to test.
- TF-IDF vectorizer is fit on the training split only.
- Spam is the positive class; we report precision, recall, F1 and false-positive rate, never accuracy alone
  (always predicting "ham" scores 87.6% accuracy on our test set).

## AI-assistance disclosure (required by the hackathon rules)
> EDIT THIS to be accurate for your team. Draft: "Claude (Anthropic) was used to generate the initial code for the
> Naive Bayes implementation, experiment script, and Streamlit demo. The team reviewed, ran, and modified the code,
> and each member can explain it. The error-analysis explanations, interpretation of results, and slides were written by the team."
> Only keep sentences that are true.

## Who did what
| role | name | contribution |
|---|---|---|
| Team Lead / PM | | |
| Data Engineer | | |
| Modeling Lead | | |
| Evaluation Lead | | |
| Product & Demo Lead | | |
