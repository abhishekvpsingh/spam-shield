"""SpamShield demo. Run:  streamlit run app.py
Paste a message -> verdict, spam probability, and the words that drove the decision."""
import csv

import pandas as pd
import streamlit as st

from run_experiments import load_data, split
from spamshield.nb import NaiveBayes

st.set_page_config(page_title="SpamShield", page_icon="🛡️")


@st.cache_resource
def get_model(deleet):
    df, _ = load_data()
    train, _, _ = split(df)  # same training split as the reported results
    return NaiveBayes(deleet=deleet).fit(train.text, train.label)


st.title("🛡️ SpamShield")
st.caption("Naive Bayes built from scratch (add-1 smoothing). Trained on the UCI SMS Spam Collection.")

deleet = st.sidebar.checkbox("Undo leetspeak (FR33 → free)", value=True)
threshold = st.sidebar.slider("Spam threshold", 0.05, 0.99, 0.50, 0.01,
                              help="Raise it to block less: fewer false positives, more missed spam.")
model = get_model(deleet)

examples = {
    "Prize scam": "WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461 now",
    "Leetspeak scam": "FR33 PR1ZE! Cl41m your 5 gift card now",
    "Family message": "Hey, running 10 min late. Can you tell mom I'll call her after dinner?",
    "Doctor's office": "Your appointment with Dr. Lee is confirmed for Tuesday at 3pm. Reply C to confirm.",
}
choice = st.selectbox("Try an example (or type your own below)", [""] + list(examples))
text = st.text_area("Message", value=examples.get(choice, ""), height=120)

if text.strip():
    p = model.spam_probability(text)
    spammy, hammy = model.explain(text, 5)
    if p >= threshold:
        st.error(f"🚫 BLOCKED as spam  (P(spam) = {p:.3f})")
    else:
        st.success(f"✅ Delivered  (P(spam) = {p:.3f})")
    st.progress(min(max(p, 0.0), 1.0))
    c1, c2 = st.columns(2)
    c1.subheader("Pushed toward spam")
    c1.table(pd.DataFrame(spammy, columns=["word", "log-odds"]) if spammy else pd.DataFrame({"word": []}))
    c2.subheader("Pushed toward ham")
    c2.table(pd.DataFrame(hammy, columns=["word", "log-odds"]) if hammy else pd.DataFrame({"word": []}))
    st.caption("Words never seen in training are ignored. Log-odds = log P(word|spam) − log P(word|ham).")
