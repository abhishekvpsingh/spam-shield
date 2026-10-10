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
    text = URL_RE.sub(" urltoken ", text.lower())  # Lowercase text; links become 'urltoken'.
    tokens = TOKEN_RE.findall(text)  # List of words, e.g. ['free', 'prize'].
    if not deleet:
        return tokens  # Return the list without spelling changes.
    out = []
    for t in tokens:
        letters = sum(c.isalpha() for c in t)  # Count letters; 'fr33' has 2.
        if letters >= 2 and any(c.isdigit() or c in "$@" for c in t):
            t = t.translate(LEET)  # Undo substitutions, e.g. 'fr33' becomes 'free'.
        out.append(t)  # Keep each processed word in order.
    return out  # Return normalized words as a list.


class NaiveBayes:
    def __init__(self, deleet=False):
        self.deleet = deleet  # Store whether to undo leetspeak spelling.

    # ---------- training ----------
    def fit(self, texts, labels):
        self.classes = sorted(set(labels))  # Unique labels, e.g. ['ham', 'spam'].
        n_docs = len(labels)  # Total number of training messages.
        self.word_counts = {c: Counter() for c in self.classes}  # Word counts for each label.
        doc_counts = Counter(labels)  # Message counts, e.g. {'ham': 80, 'spam': 20}.
        for text, c in zip(texts, labels):
            self.word_counts[c].update(tokenize(text, self.deleet))  # Count every word occurrence.
        self.vocab = set().union(*[set(wc) for wc in self.word_counts.values()])  # Set of all known words.
        self.V = len(self.vocab)  # Number of distinct training words.
        self.total_tokens = {c: sum(self.word_counts[c].values()) for c in self.classes}  # Total words per label.
        self.log_prior = {c: math.log(doc_counts[c] / n_docs) for c in self.classes}  # Log of each label's message share.
        return self  # Return the trained model for calls like NaiveBayes().fit(...).

    def log_likelihood(self, word, c):
        """log P(word | c) with add-1 smoothing."""
        return math.log((self.word_counts[c][word] + 1) / (self.total_tokens[c] + self.V))  # A log probability; +1 avoids zero counts.

    # ---------- prediction ----------
    def _tokens_in_vocab(self, text):
        # words never seen in training are ignored (standard treatment of unknowns)
        return [w for w in tokenize(text, self.deleet) if w in self.vocab]  # List containing only known words.

    def log_scores(self, text):
        toks = self._tokens_in_vocab(text)  # Get words the model can score.
        return {c: self.log_prior[c] + sum(self.log_likelihood(w, c) for w in toks) for c in self.classes}  # Scores like {'ham': -12.0, 'spam': -8.0}.

    def spam_probability(self, text):
        """P(spam | text) from the two log scores (softmax / log-sum-exp)."""
        s = self.log_scores(text)  # Get the ham and spam log scores.
        d = s["spam"] - s["ham"]  # Positive values favor spam.
        if d > 500:
            return 1.0  # Very strong spam score; avoid overflow below.
        if d < -500:
            return 0.0  # Very strong ham score; avoid overflow below.
        return 1.0 / (1.0 + math.exp(-d))  # Probability from 0 to 1, e.g. 0.92.

    def predict(self, text, threshold=0.5):
        return "spam" if self.spam_probability(text) >= threshold else "ham"  # Return one label: 'spam' or 'ham'.

    def explain(self, text, top_k=5):
        """Words that pushed the verdict toward spam (+) or ham (-): log P(w|spam) - log P(w|ham)."""
        contrib = {}  # Store one score per distinct known word.
        for w in set(self._tokens_in_vocab(text)):
            contrib[w] = self.log_likelihood(w, "spam") - self.log_likelihood(w, "ham")  # Positive favors spam; negative favors ham.
        # sort by score, then alphabetically, so ties always come out in the same order
        ranked = sorted(contrib.items(), key=lambda kv: (-kv[1], kv[0]))  # Highest score first; tied words sort alphabetically.
        spammy = [(w, s) for w, s in ranked if s > 0][:top_k]  # Strongest spam clues, e.g. [('prize', 2.1)].
        hammy = [(w, s) for w, s in ranked[::-1] if s < 0][:top_k]  # Strongest ham clues, e.g. [('dinner', -1.5)].
        return spammy, hammy  # Return two lists of (word, score) pairs.


if __name__ == "__main__":  # tiny sanity check on the toy set from Assignment 1
    train = ["the movie has no plot", "honestly pretty boring", "pretty interesting movie"]
    y = ["neg", "neg", "pos"]
    m = NaiveBayes().fit(train, y)
    print("V =", m.V, "(expected 9)")
    print("P(pretty|neg) = 2/17 ->", round(math.exp(m.log_likelihood("pretty", "neg")), 4), "vs", round(2 / 17, 4))
    print("P(plot|pos)   = 1/12 ->", round(math.exp(m.log_likelihood("plot", "pos")), 4), "vs", round(1 / 12, 4))
    scores = m.log_scores("pretty enjoyable plot")
    print("prediction for 'pretty enjoyable plot':", max(scores, key=scores.get), "(expected neg)")
