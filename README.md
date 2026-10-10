# SpamShield — MSDS 630 Residency NLP Hackathon (Project 1)

SMS spam filter for a carrier where **false positives cost far more than missed spam**.
Naive Bayes is implemented from scratch (counts + add-1 smoothing, Jurafsky & Martin Ch. 4) and compared
against scikit-learn Naive Bayes and TF-IDF logistic regression.

## Setup and run
```bash
pip install -r requirements.txt          # Python 3.10+
# data is already in data/sms.tsv (UCI SMS Spam Collection, 5,574 messages; downloaded from github.com/justmarkham/pycon-2016-tutorial, a copy of the UCI file by Almeida & Hidalgo)
python -m spamshield.nb                  # sanity check: reproduces our Assignment 1 hand calculations
python run_experiments.py                # all experiments -> results/results.md, metrics.csv, confusion_matrices.png
python analysis_extras.py                # confidence intervals, threshold-transfer check, word evidence -> results/extras.md
streamlit run app.py                     # live demo (runs fully offline)
python app_gradio.py                     # same demo with Gradio (this is the one used in Google Colab)
```

**Google Colab (no installs):** open `SpamShield_Colab.ipynb` in Colab, run the cells top to bottom, and upload this zip when asked.

## Layout
| file | purpose |
|---|---|
| `spamshield/nb.py` | tokenizer + Naive Bayes from scratch + per-word explanation |
| `run_experiments.py` | dedupe, split, train, evaluate, threshold tuning, error analysis, adversarial test |
| `analysis_extras.py` | reproducible extra numbers cited in the slides/summary (confidence intervals, overconfidence, word counts) |
| `app.py`, `app_gradio.py` | Demo (Streamlit / Gradio): verdict + probability + words that drove it |
| `SpamShield_Colab.ipynb` | Colab notebook that runs the whole project step by step |
| `results/` | generated tables and figures: `results.md`, `metrics.csv`, `confusion_matrices.png`, `extras.md` |
| `results/error_analysis.md` | five misclassified messages with evidence-based explanations (**draft: team must verify**) |
| `SpamShield_Technical_Summary.docx` / `.pdf` | the one-page technical summary deliverable |
| `summary.md`, `slides_outline.md` | plain-text copies of the summary and the slide plan |

## Method notes (what protects the numbers)
- Exact duplicates removed **before** splitting (415 dropped), so test messages are not copies of training messages.
- Stratified 70/15/15 train/dev/test split, fixed seed. Threshold tuned on **dev only**, then applied once to test.
- TF-IDF vectorizer is fit on the training split only.
- Spam is the positive class; we report precision, recall, F1 and false-positive rate, never accuracy alone
  (always predicting "ham" scores 87.6% accuracy on our test set).
- Test set is small (96 spam, 678 real): 95% intervals for NB vs logistic regression overlap, so the ranking is
  suggestive, not proven (`results/extras.md`).


## AI-assistance disclosure (required by the hackathon rules)
Claude (Anthropic) was used to generate the initial code for the from-scratch Naive Bayes
implementation, the experiment scripts, the Streamlit and Gradio demos, the slide deck, the
one-page technical summary, and a first draft of the error analysis. The team ran the code,
checked the results against the hand-calculated Assignment 1 example, and tested the demo.
Every team member is expected to be able to explain any line of code and any number in the deck.

## Who did what
| role | name | contribution |
|---|---|---|
| Team Lead / PM | Pratap Gurung | Framing, problem statement, limits and ethics |
| Data Engineer | Krushitha Janagama | Data loading, deduplication, train/dev/test split, approach |
| Modeling Lead | Ganesh Babu Palacharla | Naive Bayes baseline, robustness tests |
| Evaluation Lead | Abhishek Virendra Pratap Singh | Results table, error analysis |
| Product & Demo Lead | Varsha Reddy Polla | Live demo |
