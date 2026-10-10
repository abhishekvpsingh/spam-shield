"""SpamShield demo (Gradio version, works inside Google Colab).
Local:  python app_gradio.py      Colab:  from app_gradio import demo; demo.launch()"""
import gradio as gr

from run_experiments import load_data, split
from spamshield.nb import NaiveBayes

_models = {}  # Cache models by normalization setting: True or False.


def get_model(deleet):
    if deleet not in _models:  # Train only when this setting has no cached model.
        df, _ = load_data()  # Load the cleaned message table.
        train, _, _ = split(df)  # same training split as the reported results
        _models[deleet] = NaiveBayes(deleet=deleet).fit(train.text, train.label)  # Store the trained classifier.
    return _models[deleet]  # Return the cached model for this setting.


def check(text, threshold, deleet):
    if not text or not text.strip():  # Check for an empty or whitespace-only message.
        return "Type or paste a message.", ""  # Prompt text and an empty explanation.
    model = get_model(deleet)  # Load or train the selected model.
    p = model.spam_probability(text)  # One spam probability, e.g. 0.923.
    spammy, hammy = model.explain(text, 5)  # Up to 5 (word, score) clues for each label.
    verdict = f"{'BLOCKED as spam' if p >= threshold else 'Delivered (not spam)'}   |   P(spam) = {p:.3f}"  # Verdict with a probability rounded to 3 decimals.
    why = ("Pushed toward spam: " + (", ".join(f"{w} ({s:+.1f})" for w, s in spammy) or "none") + "\n"  # Clues like 'prize (+2.1)', or 'none'.
           "Pushed toward ham:  " + (", ".join(f"{w} ({s:+.1f})" for w, s in hammy) or "none") + "\n"
           "(Words never seen in training are ignored.)")
    return verdict, why  # Two strings shown in the output text boxes.


demo = gr.Interface(
    fn=check,  # Run this function when the user submits a message.
    inputs=[gr.Textbox(lines=4, label="Message"),
            gr.Slider(0.05, 0.99, value=0.5, step=0.01, label="Spam threshold (raise = block less)"),
            gr.Checkbox(value=True, label="Undo leetspeak (FR33 -> free)")],
    outputs=[gr.Textbox(label="Verdict"), gr.Textbox(label="Words that drove the decision", lines=4)],  # Show the verdict and word explanation.
    examples=[["WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461 now", 0.5, True],
              ["FR33 PR1ZE! Cl41m your 5 gift card now", 0.5, True],
              ["Hey, running 10 min late. Can you tell mom I'll call her after dinner?", 0.5, True],
              ["Your appointment with Dr. Lee is confirmed for Tuesday at 3pm. Reply C to confirm.", 0.5, True]],
    title="SpamShield", description="Naive Bayes from scratch (add-1 smoothing), trained on the UCI SMS Spam Collection.",
    flagging_mode="never")

if __name__ == "__main__":
    demo.launch()  # Start the local Gradio web interface.
