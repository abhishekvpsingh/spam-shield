"""Naive Bayes spam classifier built from scratch (counts + add-1 / Laplace smoothing).

Everything here maps directly to Jurafsky & Martin Ch. 4:
  P(c)        = (# training docs in class c) / (# training docs)
  P(w | c)    = (count(w, c) + 1) / (total tokens in c + |V|)
  score(c)    = log P(c) + sum over words in the message of log P(w | c)
We work in log space so tiny probabilities do not underflow to zero.
"""
import math
import re
from collections import Counter

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
TOKEN_RE = re.compile(r"[a-z0-9$@£]+")

# characters scammers swap in to dodge filters ("FR33 PR1ZE")
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "$": "s", "@": "a", "7": "t"})


def tokenize(text, deleet=False):
    """Lowercase, replace URLs, split into tokens. Optionally undo leetspeak.

    deleet=True is the 'normalization defence' from the stretch goal: a token that
    mixes letters and digits ("fr33", "pr1z3") gets its digits mapped back to letters.
    Pure numbers ("2005", "86688") and tokens with <2 letters are left alone.
    """
    text = URL_RE.sub(" urltoken ", text.lower())
    tokens = TOKEN_RE.findall(text)
    if not deleet:
        return tokens
    out = []
    for t in tokens:
        letters = sum(c.isalpha() for c in t)
        if letters >= 2 and any(c.isdigit() or c in "$@" for c in t):
            t = t.translate(LEET)
        out.append(t)
    return out


class NaiveBayes:
    def __init__(self, deleet=False):
        self.deleet = deleet

    # ---------- training ----------
    def fit(self, texts, labels):
        self.classes = sorted(set(labels))
        n_docs = len(labels)
        self.word_counts = {c: Counter() for c in self.classes}
        doc_counts = Counter(labels)
        for text, c in zip(texts, labels):
            self.word_counts[c].update(tokenize(text, self.deleet))
        self.vocab = set().union(*[set(wc) for wc in self.word_counts.values()])
        self.V = len(self.vocab)
        self.total_tokens = {c: sum(self.word_counts[c].values()) for c in self.classes}
        self.log_prior = {c: math.log(doc_counts[c] / n_docs) for c in self.classes}
        return self

    def log_likelihood(self, word, c):
        """log P(word | c) with add-1 smoothing."""
        return math.log((self.word_counts[c][word] + 1) / (self.total_tokens[c] + self.V))

    # ---------- prediction ----------
    def _tokens_in_vocab(self, text):
        # words never seen in training are ignored (standard treatment of unknowns)
        return [w for w in tokenize(text, self.deleet) if w in self.vocab]

    def log_scores(self, text):
        toks = self._tokens_in_vocab(text)
        return {c: self.log_prior[c] + sum(self.log_likelihood(w, c) for w in toks) for c in self.classes}

    def spam_probability(self, text):
        """P(spam | text) from the two log scores (softmax / log-sum-exp)."""
        s = self.log_scores(text)
        d = s["spam"] - s["ham"]
        if d > 500:
            return 1.0
        if d < -500:
            return 0.0
        return 1.0 / (1.0 + math.exp(-d))

    def predict(self, text, threshold=0.5):
        return "spam" if self.spam_probability(text) >= threshold else "ham"

    def explain(self, text, top_k=5):
        """Words that pushed the verdict toward spam (+) or ham (-): log P(w|spam) - log P(w|ham)."""
        contrib = {}
        for w in set(self._tokens_in_vocab(text)):
            contrib[w] = self.log_likelihood(w, "spam") - self.log_likelihood(w, "ham")
        ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        spammy = [(w, s) for w, s in ranked if s > 0][:top_k]
        hammy = [(w, s) for w, s in ranked[::-1] if s < 0][:top_k]
        return spammy, hammy


if __name__ == "__main__":  # tiny sanity check on the toy set from Assignment 1
    train = ["the movie has no plot", "honestly pretty boring", "pretty interesting movie"]
    y = ["neg", "neg", "pos"]
    m = NaiveBayes().fit(train, y)
    print("V =", m.V, "(expected 9)")
    print("P(pretty|neg) = 2/17 ->", round(math.exp(m.log_likelihood("pretty", "neg")), 4), "vs", round(2 / 17, 4))
    print("P(plot|pos)   = 1/12 ->", round(math.exp(m.log_likelihood("plot", "pos")), 4), "vs", round(1 / 12, 4))
    scores = m.log_scores("pretty enjoyable plot")
    print("prediction for 'pretty enjoyable plot':", max(scores, key=scores.get), "(expected neg)")
