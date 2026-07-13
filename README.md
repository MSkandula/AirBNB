# Sydney Airbnb — Predicting Guest Satisfaction

This is a machine learning project I built end to end in Python, using real Airbnb data for Sydney. The goal is simple to state: can I predict whether a listing will be one of the top-rated ones, using only the things a host actually controls — their property, their amenities, how they run the place — without cheating by looking at the review scores themselves?

I built it twice. The first version (v1) is a complete, working classifier. Then I went back, found the things I'd done in a way that wouldn't survive scrutiny from a real ML engineer, and rebuilt it properly as v2. I've kept both in this repo on purpose, because the honest story here isn't "I trained a model" — it's "I trained a model, then figured out what was wrong with it and fixed it." That second part is the one I actually care about.

**Best model: a tuned XGBoost classifier, ROC-AUC around 0.77.** But the headline number isn't really the point of this project, and I'll explain why below.

<img width="1200" height="1406" alt="sydney airbnb project poster" src="https://github.com/user-attachments/assets/5afc0c66-b033-4be4-ac95-3648963159e4" />

---

## The one thing I want you to take away

If you read nothing else, read this part.

The strongest driver of guest satisfaction in this data isn't price, isn't location, isn't how many amenities a place has. **It's how many listings the host manages.**

| Host scale | Top-tier rate | Listings |
|---|---|---|
| Single listing | **46.6%** | 4,505 |
| 2–5 listings | 31.8% | 2,576 |
| 6–20 listings | 20.8% | 1,955 |
| 20+ listings | **16.2%** | 2,544 |

A host with one listing has almost a 1-in-2 chance of being top-tier. A host running 20 or more drops to about 1-in-6. That's a huge, clean gradient, and it makes intuitive sense: someone renting out their own place personally cares more than a company managing hundreds of them.

What makes this finding worth talking about is how it cleared up something confusing. Early in the analysis, listings that let guests book instantly looked *worse* than ones that made guests ask first — which is backwards, since convenience should help. But when I dug in, instant-booking usage climbs steadily with host size (about 20% for single hosts, up to nearly 50% for the big operators). So instant-booking was never the actual cause of lower ratings — it was just a symptom of being a large commercial operation. The real driver underneath both was host scale.

In stats terms that's a confound, and finding one is the kind of thing that separates real analysis from just running a model. What I like most is that I found it two completely separate ways: first by hand, just grouping the data and looking; and later the SHAP explainability step landed on the exact same conclusion on its own. Two different methods, same answer — that's about as much confidence as you can get.

<img width="1106" height="651" alt="v2_Beeswarm" src="https://github.com/user-attachments/assets/0c649548-f21a-4704-a9e7-713f1b2b93be" />

---

## Where the data came from, and the problem I had to deal with

The data is from [Inside Airbnb](http://insideairbnb.com/), a public non-commercial project that scrapes Airbnb listings for cities around the world. I used the September 2025 Sydney scrape — about 17,700 listings with 79 columns each.

Here's the thing I didn't expect: **the price column was completely empty.** Not messy, not partial — 100% missing, every single row. Price was going to be one of my main features, so this actually mattered.

I had a choice. I could go find a cleaner dataset where price was there and pretend the problem never happened. Or I could treat it the way you'd have to treat it in an actual job — figure out what's going on, document it, and work around it. I went with the second one. I even checked a second file from the same scrape (the calendar data, 5+ million rows) to confirm price really was gone everywhere and it wasn't just a glitch in one file. It was gone. So I built the whole project without it, using only the attributes a host controls, and I say so plainly rather than hiding it.

I'd rather show that I can handle imperfect data than show a suspiciously clean project.

---

## How I set up the prediction

I turned this into a yes/no question: is a listing "top-tier" or not?

Two decisions I want to explain, because they weren't arbitrary:

**Where I drew the "top-tier" line.** Airbnb ratings are heavily bunched up near 5 stars — most listings sit above 4.8. If I'd split on an obvious-looking number, I'd have ended up calling 90% of listings "top-tier," which is a useless label. So I tested a few cutoffs and picked 4.9, because it splits the data roughly 1/3 top-tier and 2/3 not — a real, meaningful minority group the model can actually learn to spot.

**Filtering out unreliable ratings.** A listing with 2 reviews and a perfect 5.0 isn't a great listing — it's two data points. I dropped anything with fewer than 5 reviews so the ratings I'm learning from are at least somewhat trustworthy. Same idea as not trusting a survey with 2 respondents.

---

## v1 to v2 — what I changed and why it matters

v1 works. But when I looked at it critically, the biggest problem was **data leakage**: I was filling in missing values and encoding categories using the *whole* dataset before splitting into train and test. That means information from the test set quietly bled into how the training data was prepared — so my test score was measuring something slightly dishonest. The test set is supposed to be data the model has genuinely never seen.

v2 fixes this by putting every preprocessing step inside a proper scikit-learn `Pipeline`, so all of it is learned from the training data alone and applied to the test data separately. Here's what changed:

| | v1 | v2 |
|---|---|---|
| Preprocessing | Done before the split (leakage) | All inside a leakage-proof pipeline |
| How I judged the model | ROC-AUC only | ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix |
| Decision threshold | Just used the default 0.5 | Tested a range and tied it to business cost |
| Where it fails | Didn't check | Broke performance down by segment |
| Trustworthy probabilities | Didn't check | Calibration curve + Brier score |

And here's the honest part I'm proud of: **fixing the leakage barely changed the score** — it went from 0.771 to 0.770. That could look like a letdown, but it's actually the useful finding. It tells me the leakage was mild in this case. Doing it right is still the right thing to do, and being able to explain *why* it didn't move the number much is worth more than the number itself.

<img width="865" height="431" alt="v1_Confusion matrix" src="https://github.com/user-attachments/assets/6920e85f-6151-4139-8468-fe12d54455fa" />

<img width="651" height="455" alt="v2_ConfusionMatrix" src="https://github.com/user-attachments/assets/5b469f77-ef09-415b-99d4-70e19251201d" />


---

## The parts of v2 I think are strongest

**Choosing a threshold like it's a real decision.** A model doesn't actually output "yes/no" — it outputs a probability, and someone has to decide where to draw the line. 0.5 is just a default nobody thought about. I swept across thresholds and showed the trade-off: push it up to 0.7 and the model gets more precise (fewer false alarms) but misses more real top-tier listings. Which one you want depends entirely on what the model's being used for — flagging listings to promote (you want precision) versus flagging hosts who might need support (you want to catch as many as possible). I'd rather show I understand that choice than just accept a default.

**Looking at where the model fails, not just its average score.** One overall number hides a lot. When I broke performance down by host size, something interesting showed up: the model actually ranks *better* for big commercial hosts, but at the fixed 0.5 threshold it *catches* far fewer of their top-tier listings (recall drops from about 0.82 for single hosts to 0.37 for the big ones). Same model, opposite-looking trends, for a reason I can explain. Most junior projects never look below the headline metric — this is where I think this one stands out.

**Checking whether the probabilities are honest.** I tested calibration — basically, when the model says "70% likely," does that actually happen 70% of the time? It doesn't. The model is over-confident, which is a known side effect of the way I handled the class imbalance. So the probabilities are fine for *ranking* listings but shouldn't be read as literal confidence. I put this in the project even though it's unflattering, because knowing the limits of your own model is the whole point.


---

## What's in this repo

```
sydney_airbnb_guest_satisfaction.ipynb            — v1: the full project, start to finish
sydney_airbnb_guest_satisfaction-version_2.ipynb  — v2: rebuilt with a proper pipeline and deeper evaluation
top_tier_xgb_model.pkl                            — the trained model, saved
model_feature_columns.pkl                         — the exact feature columns it expects
*.png                                             — plots and the project poster
```

## Tools I used

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib.

## If you want to run it yourself

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
```

Grab `listings.csv` from the [Inside Airbnb Sydney page](http://insideairbnb.com/get-the-data/), drop it next to the notebooks, and run top to bottom.

## Things I know could be better

I try to be upfront about the gaps rather than pretend they aren't there:

- **Price is genuinely missing** from this scrape. If an earlier scrape had it, that'd be worth pulling in.
- **The probabilities aren't calibrated.** If a real use case needed honest probabilities, I'd wrap the model in scikit-learn's calibration and retest.
- **A real deployment might need different thresholds for different host types**, given what the segment analysis showed.
- **I dropped `property_type`** (60 different values) to keep things clean — grouping it sensibly could add some signal back.
- **This is one snapshot in time** (September 2025), so there's no seasonal pattern in it.

---

**Mahesh Sai Kandula** — Master of Data Science, Macquarie University
[LinkedIn](https://www.linkedin.com/in/mahesh-kandula-b6393622a/) · [GitHub](https://github.com/MSkandula)
