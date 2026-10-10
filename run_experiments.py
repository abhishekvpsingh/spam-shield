"""Runs every experiment and writes results/results.md + results/*.csv|png.

Usage:  python run_experiments.py
"""
import csv
import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from spamshield.nb import NaiveBayes

SEED = 42
TARGET_FPR = 0.005  # client: false positives are far costlier than missed spam
os.makedirs("results", exist_ok=True)  # Create the output folder if it is missing.


# ----------------------------------------------------------------- data
def load_data(path="data/sms.tsv"):
    # QUOTE_NONE: some messages contain stray quote characters
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE)  # Table with label and text columns.
    n_raw = len(df)  # Message count before removing duplicates.
    df["key"] = df.text.str.lower().str.strip()  # Match duplicates despite case or outer spaces.
    df = df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)  # Remove duplicates before splitting; renumber rows.
    return df, n_raw  # Return the cleaned table and original count.


def split(df):
    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)  # Keep 70% for training; preserve label shares.
    dev, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)  # Split the remaining 30% into two 15% sets.
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)  # Three tables with fresh row numbers.


# ----------------------------------------------------------------- metrics
def report(y_true, y_pred, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["ham", "spam"]).ravel()  # Correct ham, false alarms, missed spam, correct spam.
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=["spam"], zero_division=0)  # Each spam metric is a one-value array.
    # Return a dictionary of counts and scores; rates range from 0 to 1.
    return {"model": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": p[0], "recall": r[0], "F1": f1[0], "FPR": fp / (fp + tn),
            "accuracy": (tp + tn) / (tp + tn + fp + fn)}


def pick_threshold(probs, y_true, target_fpr):
    """Lowest threshold (=highest recall) whose false-positive rate on `dev` stays <= target."""
    y = np.array(y_true)  # Label array, e.g. ['ham', 'spam'].
    ham_probs = probs[y == "ham"]  # Spam probabilities for actual ham messages.
    best = 1.0  # Starting cutoff if no candidate meets the target.
    for t in np.unique(np.concatenate([probs, [0.5, 1.0]])):  # Try each distinct score plus two fixed cutoffs.
        if (ham_probs >= t).mean() <= target_fpr:  # Check the fraction of ham incorrectly flagged as spam.
            best = min(best, t)  # Keep the lowest qualifying cutoff.
    return float(best)  # Return one number, e.g. 0.98.


def md_table(rows):
    cols = ["model", "precision", "recall", "F1", "FPR", "accuracy", "TP", "FP", "FN", "TN"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]  # Markdown header and separator rows.
    for r in rows:
        cells = [r["model"]] + [f"{r[c]:.4f}" for c in cols[1:6]] + [str(int(r[c])) for c in cols[6:]]  # Scores like 0.9500; counts like 42.
        out.append("| " + " | ".join(cells) + " |")  # Add one table row per model.
    return "\n".join(out)  # Return the whole Markdown table as text.


def save_confusion_figure(rows):
    import matplotlib
    matplotlib.use("Agg")  # Save plots without opening a window.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(rows), figsize=(3.3 * len(rows), 3.4))  # One plot per model, arranged side by side.
    for ax, r in zip(axes, rows):
        m = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])  # 2-by-2 table: true labels in rows, predictions in columns.
        ax.imshow(m, cmap="Blues")  # Larger counts appear in darker blue.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=14,  # Print the count in each cell.
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred ham", "pred spam"])
        ax.set_yticks([0, 1], ["true ham", "true spam"])
        ax.set_title(r["model"].replace(" (", "\n("), fontsize=8)
    fig.tight_layout()  # Adjust spacing around labels and titles.
    fig.savefig("results/confusion_matrices.png", dpi=150)  # Save all model plots in one PNG image.


# ----------------------------------------------------------------- main
def main():
    df, n_raw = load_data()  # Get cleaned messages and the original message count.
    train, dev, test = split(df)  # Training, threshold-tuning, and final evaluation tables.
    lines = ["# SpamShield results\n"]  # Collect text for the Markdown report.
    lines.append(f"Raw messages: {n_raw}; after exact-duplicate removal: {len(df)} "
                 f"({n_raw - len(df)} duplicates dropped BEFORE splitting).\n")
    lines.append(f"Split (stratified, seed {SEED}): train {len(train)} / dev {len(dev)} / test {len(test)}. "
                 f"Spam share: {(df.label == 'spam').mean():.1%}.\n")
    majority = (test.label == "ham").mean()  # Accuracy when every prediction is 'ham'.
    lines.append(f"Accuracy of the 'predict ham for everything' baseline on test: {majority:.4f} "
                 f"(why accuracy alone is misleading).\n")

    rows = []  # Store one metric dictionary per model setup.

    # 1) MVP: Naive Bayes from scratch, default threshold 0.5
    nb = NaiveBayes().fit(train.text, train.label)  # Train the custom classifier on labeled messages.
    test_pred_nb = [nb.predict(t) for t in test.text]  # Labels like ['ham', 'spam', 'ham'].
    rows.append(report(test.label, test_pred_nb, "NB from scratch (thr 0.5)"))  # Add test scores for the default cutoff.

    # 2) Should-have: tune threshold on DEV to hit the target FPR, apply once to TEST
    dev_probs = np.array([nb.spam_probability(t) for t in dev.text])  # One spam probability per dev message.
    thr = pick_threshold(dev_probs, dev.label, TARGET_FPR)  # Choose a cutoff using dev labels only.
    test_probs_nb = np.array([nb.spam_probability(t) for t in test.text])  # Test scores like [0.02, 0.95].
    pred_tuned = np.where(test_probs_nb >= thr, "spam", "ham")  # Turn test probabilities into label strings.
    rows.append(report(test.label, pred_tuned, f"NB from scratch (tuned thr {thr:.4g}, dev FPR<={TARGET_FPR:.1%})"))

    # 3) scikit-learn comparisons. Vectorizer is fit on TRAIN ONLY.
    cv_tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)  # Weight words and two-word phrases seen in 2+ messages.
    Xtr = cv_tfidf.fit_transform(train.text)  # Learn features; output a sparse messages-by-features matrix.
    Xdev, Xte = cv_tfidf.transform(dev.text), cv_tfidf.transform(test.text)  # Reuse training features for dev and test matrices.

    sk_nb = MultinomialNB(alpha=1.0).fit(Xtr, train.label)  # Train sklearn Naive Bayes with add-1 smoothing.
    rows.append(report(test.label, sk_nb.predict(Xte), "sklearn MultinomialNB (TF-IDF)"))

    lr = LogisticRegression(max_iter=2000, C=10, class_weight=None).fit(Xtr, train.label)  # Train logistic regression on the same features.
    rows.append(report(test.label, lr.predict(Xte), "Logistic Regression (TF-IDF, thr 0.5)"))

    spam_idx = list(lr.classes_).index("spam")  # Find the spam column in the probability matrix.
    thr_lr = pick_threshold(lr.predict_proba(Xdev)[:, spam_idx], dev.label, TARGET_FPR)  # Tune using one spam score per dev message.
    pred_lr_t = np.where(lr.predict_proba(Xte)[:, spam_idx] >= thr_lr, "spam", "ham")  # Test labels using the tuned cutoff.
    rows.append(report(test.label, pred_lr_t, f"Logistic Regression (tuned thr {thr_lr:.3f})"))

    pd.DataFrame(rows).to_csv("results/metrics.csv", index=False)  # Save a table with one row per model setup.
    save_confusion_figure(rows)  # Save the comparison image.
    lines.append("## Test-set results (spam = positive class)\n")
    lines.append(md_table(rows) + "\n")

    # 4) error analysis: 5 misclassified test messages from the tuned from-scratch NB
    wrong = [(t, y, p, pr) for t, y, p, pr in zip(test.text, test.label, pred_tuned, test_probs_nb) if y != p]  # Errors as (text, truth, prediction, probability).
    fps = [w for w in wrong if w[1] == "ham"]  # Ham messages incorrectly marked as spam.
    fns = [w for w in wrong if w[1] == "spam"]  # Spam messages incorrectly marked as ham.
    random.Random(SEED).shuffle(fps)  # Shuffle errors in a repeatable order.
    random.Random(SEED).shuffle(fns)  # Repeatably shuffle missed spam too.
    chosen = (fps[:2] + fns[:3])[:5] if len(fps) >= 2 else (fps + fns)[:5]  # Pick up to 5 errors, favoring 2 false alarms and 3 misses.
    lines.append("## Error analysis (tuned NB from scratch) — fill in the 'why' column from the evidence below\n")
    for i, (t, y, p, pr) in enumerate(chosen, 1):
        spammy, hammy = nb.explain(t, 4)  # Up to 4 (word, score) clues for each label.
        lines.append(f"**{i}. true={y}, predicted={p}, P(spam)={pr:.3f}**  \n> {t}\n")
        lines.append("  - words pushing toward spam: " + ", ".join(f"`{w}` ({s:+.1f})" for w, s in spammy))
        lines.append("  - words pushing toward ham: " + ", ".join(f"`{w}` ({s:+.1f})" for w, s in hammy))
        lines.append("  - WHY (write your own explanation here): \n")
    lines.append(f"Total errors at the tuned threshold: {len(fps)} false positives, {len(fns)} missed spam.\n")

    # 5) stretch-lite: adversarial spellings, with and without the normalization defence
    rng = random.Random(SEED)  # Make spelling changes repeatable across runs.
    swap = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}  # Letter-to-digit replacements for the spelling test.

    def perturb(text):
        out = []  # Collect changed and unchanged words.
        for w in text.split():  # Split the message at whitespace.
            if len(w) >= 4 and rng.random() < 0.6:  # Select longer words with a 60% chance.
                w = "".join(swap.get(c, c) if c in swap and rng.random() < 0.7 else c for c in w.lower())  # Replace eligible letters with a 70% chance each.
            out.append(w)  # Keep the word in its original position.
        return " ".join(out)  # Return a message with spellings such as 'fr33'.

    spam_test = test[test.label == "spam"].text.tolist()  # List of actual spam messages.
    ham_test = test[test.label == "ham"].text.tolist()  # List of actual ham messages.
    adv = [perturb(t) for t in spam_test]  # One altered message per original spam message.
    nb_plain = nb  # Reuse the model without spelling normalization.
    nb_norm = NaiveBayes(deleet=True).fit(train.text, train.label)  # Train another model that undoes leetspeak.
    adv_rows = []
    for name, model in [("no normalization", nb_plain), ("with de-leet normalization", nb_norm)]:
        rec_clean = np.mean([model.predict(t) == "spam" for t in spam_test])  # Fraction of original spam correctly detected.
        rec_adv = np.mean([model.predict(t) == "spam" for t in adv])  # Fraction of altered spam correctly detected.
        fpr = np.mean([model.predict(t) == "spam" for t in ham_test])  # Fraction of ham incorrectly flagged.
        adv_rows.append((name, rec_clean, rec_adv, fpr))  # Store the model name and three rates from 0 to 1.
    lines.append("## Adversarial-spelling test (NB from scratch, threshold 0.5)\n")
    lines.append("| model | spam recall (original) | spam recall (leetspeak) | ham false-positive rate |\n|---|---|---|---|")
    for n, a, b, c in adv_rows:
        lines.append(f"| {n} | {a:.3f} | {b:.3f} | {c:.3f} |")
    lines.append(f"\nExample perturbed message: `{adv[0]}`\n")

    # 6) modern scam-style messages to probe generalization (written by us; edit/extend these)
    modern = [
        "USPS: Your package is on hold due to an incomplete address. Update now: usps-track.example.com/confirm",
        "Your bank account has been locked. Verify your identity immediately at secure-bank.example.net",
        "Congratulations! You've been selected to receive a $500 Amazon gift card. Claim now before it expires",
        "Toll road notice: unpaid balance of $6.99. Pay today to avoid a late fee. tollpay.example.org",
        "Hey are we still on for dinner at 7? I can pick up mom on the way",
        "Your appointment with Dr. Lee is confirmed for Tuesday 3pm. Reply C to confirm or R to reschedule",
        "running 10 min late, start without me",
    ]
    truth = ["spam"] * 4 + ["ham"] * 3  # Expected labels for the 7 hand-written messages.
    lines.append("## Generalization probe: modern scam-style messages (tiny hand-written set — treat as anecdotal)\n")
    lines.append("| message | truth | NB (thr 0.5) | NB tuned | P(spam) |\n|---|---|---|---|---|")
    for t, y in zip(modern, truth):
        pr = nb.spam_probability(t)  # Spam score for this new message, e.g. 0.87.
        lines.append(f"| {t[:70]}… | {y} | {nb.predict(t)} | {'spam' if pr >= thr else 'ham'} | {pr:.3f} |")

    with open("results/results.md", "w") as f:
        f.write("\n".join(lines) + "\n")  # Save headings, tables, and examples as a Markdown report.
    print("\n".join(lines))  # Show the same report text in the terminal.


if __name__ == "__main__":
    main()  # Run all experiments when this file is executed directly.
