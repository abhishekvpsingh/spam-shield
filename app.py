"""SpamShield demo. Run:  streamlit run app.py
Paste a message -> verdict, spam probability, and the words that drove the decision."""
import csv

import pandas as pd
import streamlit as st

from run_experiments import load_data, split
from spamshield.nb import NaiveBayes

st.set_page_config(page_title="SpamShield", page_icon="🛡️")


@st.cache_resource  # Reuse the trained model for each normalization setting.
def get_model(deleet):
    df, _ = load_data()  # Read messages and remove duplicates.
    train, _, _ = split(df)  # same training split as the reported results
    return NaiveBayes(deleet=deleet).fit(train.text, train.label)  # Return a trained classifier.


st.title("🛡️ SpamShield")
st.caption("Naive Bayes built from scratch (add-1 smoothing). Trained on the UCI SMS Spam Collection.")

deleet = st.sidebar.checkbox("Undo leetspeak (FR33 → free)", value=True)  # True when spelling normalization is enabled.
threshold = st.sidebar.slider("Spam threshold", 0.00, 1.00, 0.50, 0.01,  # Zero blocks all messages; 1 blocks only scores of 1.
                              help="Raise it to block less: fewer false positives, more missed spam.")
model = get_model(deleet)  # Get the model matching the checkbox setting.

examples = {
    "Prize scam": "WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461 now",
    "Leetspeak scam": "FR33 PR1ZE! Cl41m your 5 gift card now",
    "Family message": "Hey, running 10 min late. Can you tell mom I'll call her after dinner?",
    "Doctor's office": "Your appointment with Dr. Lee is confirmed for Tuesday at 3pm. Reply C to confirm.",
}
choice = st.selectbox("Try an example (or type your own below)", [""] + list(examples))  # Return the selected example name.
text = st.text_area("Message", value=examples.get(choice, ""), height=120)  # Message text entered or selected by the user.

if text.strip():  # Classify only messages with non-whitespace text.
    p = model.spam_probability(text)  # Spam probability from 0 to 1, e.g. 0.92.
    spammy, hammy = model.explain(text, 5)  # Two lists of up to 5 (word, score) pairs.
    if p >= threshold:  # Block messages at or above the chosen cutoff.
        st.error(f"🚫 BLOCKED as spam  (P(spam) = {p:.3g}; threshold = {threshold:.2f})")
    else:
        st.success(f"✅ Delivered  (P(spam) = {p:.3g}; threshold = {threshold:.2f})")
    st.progress(min(max(p, 0.0), 1.0))  # Show spam probability as a filled progress bar.
    c1, c2 = st.columns(2)  # Display spam and ham clues side by side.
    c1.subheader("Pushed toward spam")
    c1.table(pd.DataFrame(spammy, columns=["word", "log-odds"]) if spammy else pd.DataFrame({"word": []}))  # Table of words with positive spam scores.
    c2.subheader("Pushed toward ham")
    c2.table(pd.DataFrame(hammy, columns=["word", "log-odds"]) if hammy else pd.DataFrame({"word": []}))  # Table of words with negative spam scores.
    st.caption("Words never seen in training are ignored. Log-odds = log P(word|spam) − log P(word|ham).")
