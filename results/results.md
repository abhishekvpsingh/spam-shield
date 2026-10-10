# SpamShield results

Raw messages: 5574; after exact-duplicate removal: 5159 (415 duplicates dropped BEFORE splitting).

Split (stratified, seed 42): train 3611 / dev 774 / test 774. Spam share: 12.4%.

Accuracy of the 'predict ham for everything' baseline on test: 0.8760 (why accuracy alone is misleading).

## Test-set results (spam = positive class)

| model | precision | recall | F1 | FPR | accuracy | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| NB from scratch (thr 0.5) | 0.9444 | 0.8854 | 0.9140 | 0.0074 | 0.9793 | 85 | 5 | 11 | 673 |
| NB from scratch (tuned thr 0.3132, dev FPR<=0.5%) | 0.9239 | 0.8854 | 0.9043 | 0.0103 | 0.9767 | 85 | 7 | 11 | 671 |
| sklearn MultinomialNB (TF-IDF) | 1.0000 | 0.6979 | 0.8221 | 0.0000 | 0.9625 | 67 | 0 | 29 | 678 |
| Logistic Regression (TF-IDF, thr 0.5) | 1.0000 | 0.8750 | 0.9333 | 0.0000 | 0.9845 | 84 | 0 | 12 | 678 |
| Logistic Regression (tuned thr 0.374) | 0.9890 | 0.9375 | 0.9626 | 0.0015 | 0.9910 | 90 | 1 | 6 | 677 |

## Error analysis (tuned NB from scratch) — fill in the 'why' column from the evidence below

**1. true=ham, predicted=spam, P(spam)=0.931**  
> Nokia phone is lovly..

  - words pushing toward spam: `nokia` (+4.7), `phone` (+0.3)
  - words pushing toward ham: `is` (-0.5)
  - WHY (write your own explanation here): 

**2. true=ham, predicted=spam, P(spam)=0.887**  
> Unlimited texts. Limited minutes.

  - words pushing toward spam: `unlimited` (+2.6), `texts` (+1.4), `minutes` (+0.1)
  - words pushing toward ham: 
  - WHY (write your own explanation here): 

**3. true=spam, predicted=ham, P(spam)=0.018**  
> ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE MINS. INDIA CUST SERVs SED YES. L8ER GOT MEGA BILL. 3 DONT GIV A SHIT. BAILIFF DUE IN DAYS. I O £250 3 WANT £800

  - words pushing toward spam: `£250` (+3.5), `£800` (+3.0), `0870` (+2.5), `free` (+2.4)
  - words pushing toward ham: `i` (-3.2), `got` (-2.6), `shit` (-2.2), `dont` (-1.7)
  - WHY (write your own explanation here): 

**4. true=spam, predicted=ham, P(spam)=0.000**  
> Can U get 2 phone NOW? I wanna chat 2 set up meet Call me NOW on 09096102316 U can cum here 2moro Luv JANE xx Calls£1/minmoremobsEMSPOBox45PO139WA

  - words pushing toward spam: `calls£1` (+2.5), `chat` (+1.9), `call` (+1.5), `2` (+0.5)
  - words pushing toward ham: `i` (-3.2), `me` (-2.5), `here` (-1.9), `can` (-1.8)
  - WHY (write your own explanation here): 

**5. true=spam, predicted=ham, P(spam)=0.000**  
> Hi I'm sue. I am 20 years old and work as a lapdancer. I love sex. Text me live - I'm i my bedroom now. text SUE to 89555. By TextOperator G2 1DA 150ppmsg 18+

  - words pushing toward spam: `18` (+4.3), `89555` (+1.8), `textoperator` (+1.8), `text` (+1.6)
  - words pushing toward ham: `i` (-3.2), `my` (-2.8), `me` (-2.5), `work` (-2.4)
  - WHY (write your own explanation here): 

Total errors at the tuned threshold: 7 false positives, 11 missed spam.

## Adversarial-spelling test (NB from scratch, threshold 0.5)

| model | spam recall (original) | spam recall (leetspeak) | ham false-positive rate |
|---|---|---|---|
| no normalization | 0.885 | 0.833 | 0.007 |
| with de-leet normalization | 0.885 | 0.885 | 0.007 |

Example perturbed message: `WINNER!! As a v4lu3d network customer you have b33n 53l3ct3d to r3c3iv3a £900 pr1ze r3w4rd! To claim call 09061701461. Claim cod3 kl341. Valid 12 hours only.`

## Generalization probe: modern scam-style messages (tiny hand-written set — treat as anecdotal)

| message | truth | NB (thr 0.5) | NB tuned | P(spam) |
|---|---|---|---|---|
| USPS: Your package is on hold due to an incomplete address. Update now… | spam | spam | spam | 0.594 |
| Your bank account has been locked. Verify your identity immediately at… | spam | spam | spam | 0.939 |
| Congratulations! You've been selected to receive a $500 Amazon gift ca… | spam | spam | spam | 1.000 |
| Toll road notice: unpaid balance of $6.99. Pay today to avoid a late f… | spam | ham | ham | 0.034 |
| Hey are we still on for dinner at 7? I can pick up mom on the way… | ham | ham | ham | 0.000 |
| Your appointment with Dr. Lee is confirmed for Tuesday 3pm. Reply C to… | ham | spam | spam | 0.824 |
| running 10 min late, start without me… | ham | ham | ham | 0.001 |
