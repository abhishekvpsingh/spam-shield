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
os.makedirs("results", exist_ok=True)


# ----------------------------------------------------------------- data
def load_data(path="data/sms.tsv"):
    # QUOTE_NONE: some messages contain stray quote characters
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE)
    n_raw = len(df)
    df["key"] = df.text.str.lower().str.strip()
    df = df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)  # dedupe BEFORE splitting
    return df, n_raw


def split(df):
    train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
    dev, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)


# ----------------------------------------------------------------- metrics
def report(y_true, y_pred, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["ham", "spam"]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=["spam"], zero_division=0)
    return {"model": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": p[0], "recall": r[0], "F1": f1[0], "FPR": fp / (fp + tn),
            "accuracy": (tp + tn) / (tp + tn + fp + fn)}


def pick_threshold(probs, y_true, target_fpr):
    """Lowest threshold (=highest recall) whose false-positive rate on `dev` stays <= target."""
    y = np.array(y_true)
    ham_probs = probs[y == "ham"]
    best = 1.0
    for t in np.unique(np.concatenate([probs, [0.5, 1.0]])):
        if (ham_probs >= t).mean() <= target_fpr:
            best = min(best, t)
    return float(best)


def md_table(rows):
    cols = ["model", "precision", "recall", "F1", "FPR", "accuracy", "TP", "FP", "FN", "TN"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [r["model"]] + [f"{r[c]:.4f}" for c in cols[1:6]] + [str(int(r[c])) for c in cols[6:]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def save_confusion_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(rows), figsize=(3.3 * len(rows), 3.4))
    for ax, r in zip(axes, rows):
        m = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=14,
                        color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred ham", "pred spam"])
        ax.set_yticks([0, 1], ["true ham", "true spam"])
        ax.set_title(r["model"].replace(" (", "\n("), fontsize=8)
    fig.tight_layout()
    fig.savefig("results/confusion_matrices.png", dpi=150)


# ----------------------------------------------------------------- main
def main():
    df, n_raw = load_data()
    train, dev, test = split(df)
    lines = ["# SpamShield results\n"]
    lines.append(f"Raw messages: {n_raw}; after exact-duplicate removal: {len(df)} "
                 f"({n_raw - len(df)} duplicates dropped BEFORE splitting).\n")
    lines.append(f"Split (stratified, seed {SEED}): train {len(train)} / dev {len(dev)} / test {len(test)}. "
                 f"Spam share: {(df.label == 'spam').mean():.1%}.\n")
    majority = (test.label == "ham").mean()
    lines.append(f"Accuracy of the 'predict ham for everything' baseline on test: {majority:.4f} "
                 f"(why accuracy alone is misleading).\n")

    rows = []

    # 1) MVP: Naive Bayes from scratch, default threshold 0.5
    nb = NaiveBayes().fit(train.text, train.label)
    test_pred_nb = [nb.predict(t) for t in test.text]
    rows.append(report(test.label, test_pred_nb, "NB from scratch (thr 0.5)"))

    # 2) Should-have: tune threshold on DEV to hit the target FPR, apply once to TEST
    dev_probs = np.array([nb.spam_probability(t) for t in dev.text])
    thr = pick_threshold(dev_probs, dev.label, TARGET_FPR)
    test_probs_nb = np.array([nb.spam_probability(t) for t in test.text])
    pred_tuned = np.where(test_probs_nb >= thr, "spam", "ham")
    rows.append(report(test.label, pred_tuned, f"NB from scratch (tuned thr {thr:.4g}, dev FPR<={TARGET_FPR:.1%})"))

    # 3) scikit-learn comparisons. Vectorizer is fit on TRAIN ONLY.
    cv_tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = cv_tfidf.fit_transform(train.text)
    Xdev, Xte = cv_tfidf.transform(dev.text), cv_tfidf.transform(test.text)

    sk_nb = MultinomialNB(alpha=1.0).fit(Xtr, train.label)
    rows.append(report(test.label, sk_nb.predict(Xte), "sklearn MultinomialNB (TF-IDF)"))

    lr = LogisticRegression(max_iter=2000, C=10, class_weight=None).fit(Xtr, train.label)
    rows.append(report(test.label, lr.predict(Xte), "Logistic Regression (TF-IDF, thr 0.5)"))

    spam_idx = list(lr.classes_).index("spam")
    thr_lr = pick_threshold(lr.predict_proba(Xdev)[:, spam_idx], dev.label, TARGET_FPR)
    pred_lr_t = np.where(lr.predict_proba(Xte)[:, spam_idx] >= thr_lr, "spam", "ham")
    rows.append(report(test.label, pred_lr_t, f"Logistic Regression (tuned thr {thr_lr:.3f})"))

    pd.DataFrame(rows).to_csv("results/metrics.csv", index=False)
    save_confusion_figure(rows)
    lines.append("## Test-set results (spam = positive class)\n")
    lines.append(md_table(rows) + "\n")

    # 4) error analysis: 5 misclassified test messages from the tuned from-scratch NB
    wrong = [(t, y, p, pr) for t, y, p, pr in zip(test.text, test.label, pred_tuned, test_probs_nb) if y != p]
    fps = [w for w in wrong if w[1] == "ham"]
    fns = [w for w in wrong if w[1] == "spam"]
    random.Random(SEED).shuffle(fps)
    random.Random(SEED).shuffle(fns)
    chosen = (fps[:2] + fns[:3])[:5] if len(fps) >= 2 else (fps + fns)[:5]
    lines.append("## Error analysis (tuned NB from scratch) — fill in the 'why' column from the evidence below\n")
    for i, (t, y, p, pr) in enumerate(chosen, 1):
        spammy, hammy = nb.explain(t, 4)
        lines.append(f"**{i}. true={y}, predicted={p}, P(spam)={pr:.3f}**  \n> {t}\n")
        lines.append("  - words pushing toward spam: " + ", ".join(f"`{w}` ({s:+.1f})" for w, s in spammy))
        lines.append("  - words pushing toward ham: " + ", ".join(f"`{w}` ({s:+.1f})" for w, s in hammy))
        lines.append("  - WHY (write your own explanation here): \n")
    lines.append(f"Total errors at the tuned threshold: {len(fps)} false positives, {len(fns)} missed spam.\n")

    # 5) stretch-lite: adversarial spellings, with and without the normalization defence
    rng = random.Random(SEED)
    swap = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}

    def perturb(text):
        out = []
        for w in text.split():
            if len(w) >= 4 and rng.random() < 0.6:
                w = "".join(swap.get(c, c) if c in swap and rng.random() < 0.7 else c for c in w.lower())
            out.append(w)
        return " ".join(out)

    spam_test = test[test.label == "spam"].text.tolist()
    ham_test = test[test.label == "ham"].text.tolist()
    adv = [perturb(t) for t in spam_test]
    nb_plain = nb
    nb_norm = NaiveBayes(deleet=True).fit(train.text, train.label)
    adv_rows = []
    for name, model in [("no normalization", nb_plain), ("with de-leet normalization", nb_norm)]:
        rec_clean = np.mean([model.predict(t) == "spam" for t in spam_test])
        rec_adv = np.mean([model.predict(t) == "spam" for t in adv])
        fpr = np.mean([model.predict(t) == "spam" for t in ham_test])
        adv_rows.append((name, rec_clean, rec_adv, fpr))
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
    truth = ["spam"] * 4 + ["ham"] * 3
    lines.append("## Generalization probe: modern scam-style messages (tiny hand-written set — treat as anecdotal)\n")
    lines.append("| message | truth | NB (thr 0.5) | NB tuned | P(spam) |\n|---|---|---|---|---|")
    for t, y in zip(modern, truth):
        pr = nb.spam_probability(t)
        lines.append(f"| {t[:70]}… | {y} | {nb.predict(t)} | {'spam' if pr >= thr else 'ham'} | {pr:.3f} |")

    with open("results/results.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
