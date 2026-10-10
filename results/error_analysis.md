# Error analysis: five misclassified test messages

*First draft written with AI assistance (see the disclosure in the README). Owner: Abhishek Virendra Pratap Singh (Evaluation Lead). Every team member should read it and be able to defend each explanation in Q&A.*

**Review status:** each explanation below was checked against `results/results.md` and `results/extras.md` by: ______ (initials) — edit anything you disagree with, then remove this line.

Model: Naive Bayes from scratch, dev-tuned threshold 0.3132 (7 false positives, 11 missed spam on the test set).
Word counts below are "number of training messages containing the word" (spam / ham), from `analysis_extras.py`.

## How to read the evidence
Naive Bayes adds up one log-likelihood-ratio per known word, plus the class prior. Words never seen in training are
**ignored**. So a message is decided by (a) which of its words were seen before and (b) how lopsided those words'
spam/ham counts are. Every error below comes from one of those two things.

## False positives (real messages blocked, the costly kind for this client)

**1. "Nokia phone is lovly.."** — true ham, P(spam) = 0.931
- Only two informative words survive: "lovly" is a typo never seen in training (ignored), and "is" is nearly neutral.
- "nokia" appears in 28 spam training messages and **0** ham messages (handset-offer spam), worth +4.7 toward spam.
  Add-1 smoothing turns that 28-vs-0 gap into a large ratio, and with so few other words nothing offsets it.
- Lesson: on very short messages a single lopsided word decides the verdict.

**2. "Unlimited texts. Limited minutes."** — true ham, P(spam) = 0.887
- Its known words with an effect all lean spam and none lean ham: "unlimited" 8 spam / 1 ham, "texts" 11 / 8.
- In this dataset, talk about plans, texts and minutes mostly comes from carrier promotions, so the vocabulary alone
  looks like a promotion, even though the message reads as a casual personal remark.
- Lesson: topic words are not the same as intent; the model has no way to see tone or who is speaking.

## Missed spam (spam delivered)

**3. "ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS ... I O £250 3 WANT £800"** — true spam, P(spam) = 0.018
- Real spam cues are present: "£250" (10 spam / 0 ham), "£800" (6 / 0), "0870" (3 / 0), "free" (102 / 36).
- They are outweighed by informal first-person words that are strongly ham ("i" appears in 25 spam but 1,369 ham
  messages), and six words are unseen and ignored (3mobile, chatlines, inclu, servs, l8er, bailiff).
- This was a close call: the log-score gap between classes was only about 4 units.

**4. "Can U get 2 phone NOW? I wanna chat 2 set up meet Call me NOW on 09096102316 ... Calls£1/min..."** — true spam, P(spam) ≈ 0
- The phone number (09096102316) and the long run-together suffix ("minmoremobsemspobox45po139wa") are one-off tokens
  never seen in training, so the model throws them away. "calls£1" was known (+2.5) but is too weak on its own.
- What remains is conversational wording ("i", "me", "here", "can") that is overwhelmingly ham ("me" 13 spam / 445 ham).
  The class log-scores differed by about 13 units.
- Lesson: exact-match tokens cannot generalise to new numbers or new spellings; patterns (a regex for "long digit
  string" or "£ amount per minute") would catch what a vocabulary cannot.

**5. "Hi I'm sue. I am 20 years old and work as a lapdancer ... text SUE to 89555 ... 150ppmsg 18+"** — true spam, P(spam) ≈ 0
- Spam cues exist ("18" 23 spam / 0 ham, "89555", "textoperator", "text"), but the message is long (36 tokens vs about
  25 on average for spam) and full of first-person words that point to ham ("i" 25 / 1,369, "my" 9 / 395, "me" 13 / 445).
- Naive Bayes treats every word as independent evidence, so many weak ham-leaning words add up and swamp a few strong
  spam words. The class log-scores differed by about 22 units. This is the independence assumption failing.
- Some cues are also unseen and ignored (the postcode "g2 1da" and "150ppmsg").

## What the five errors have in common
1. Unknown words are discarded, so new numbers, brand-new spellings and unusual strings give no evidence.
2. Independence assumption: many mild ham words can outvote a few strong spam words in long messages.
3. Dataset age and domain: carrier-promotion vocabulary and 2000s texting style shape what "looks like spam".
4. Possible fixes to test (not yet done): regex features for digit strings and £/min patterns, character n-grams, a
   calibrated model such as logistic regression, and more recent training data.