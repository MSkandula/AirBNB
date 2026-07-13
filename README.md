# Sydney Airbnb — Predicting Guest Satisfaction Tier

An end-to-end Python machine learning project: real, imperfect public data taken from inspection through cleaning, exploratory analysis, feature engineering, model comparison, hyperparameter tuning, and SHAP explainability.

**Best model: tuned XGBoost, ROC-AUC 0.771** — predicting whether an Airbnb listing lands in the top ~32% by guest satisfaction, using only attributes a host can actually see and control.

---

## The headline finding

**Host portfolio size is the single strongest driver of guest satisfaction in this data — and it explains away a result that initially looked backwards.**

| Host scale | Top-tier rate | Listings |
|---|---|---|
| Single listing | **46.6%** | 4,505 |
| 2–5 listings | 31.8% | 2,576 |
| 6–20 listings | 20.8% | 1,955 |
| 20+ listings | **16.2%** | 2,544 |

Listings that *aren't* instant-bookable initially looked like they perform better than instant-bookable ones — the opposite of what convenience should predict. The reason: instant-booking usage rises in lockstep with host portfolio size (19.6% → 49.8%), because large-scale commercial hosts can't personally vet every request. `instant_bookable` wasn't a cause of lower satisfaction — it was a marker for it, standing in for the real driver, host scale.

This was found first through manual exploratory analysis, then **independently confirmed by SHAP on the final trained model** — two unrelated methods, same conclusion.

---

## Data source

[Inside Airbnb](http://insideairbnb.com/) — Sydney, NSW, Australia, September 2025 scrape (`listings.csv`, detailed version: 17,730 listings × 79 columns). Independent, non-commercial, publicly available.

**A real data-sourcing problem, handled rather than avoided:** `price` arrived 100% null in this scrape — verified at both the listing level and, separately, in the companion `calendar.csv` (5.2M rows, also 100% null). Rather than switch to a cleaner dataset, this was treated as a genuine data limitation to document and design around — the model is built entirely on operational and property attributes a host controls, not price.

## Objective

Binary classification: is a listing **"top-tier"** (`review_scores_rating` ≥ 4.9, chosen by testing class balance across several cutoffs, not by picking a percentile)? Restricted to listings with ≥ 5 reviews, to avoid rewarding small-sample noise (a 2-review listing with a perfect 5.0 rating is not evidence of quality).
<img width="1200" height="1406" alt="sydney airbnb project poster" src="https://github.com/user-attachments/assets/db6e440a-7035-4748-bb9c-34f1a16eedbc" />

## Methodology

1. **Inspection** — shape, dtypes, missingness audit across all 79 columns
2. **Cleaning** — dropped 6 fully-null columns; recovered `bathrooms` via regex on `bathrooms_text`; applied a minimum-reviews reliability filter (11,580 rows remain)
3. **Target & leakage** — defined `top_tier`; excluded review sub-scores and `host_is_superhost` (both leak the target — see notebook for the reasoning)
4. **EDA** — found the host-scale confound described above
5. **Feature engineering** — host tenure, response/acceptance rates, amenities parsed from a JSON-like field into count + binary flags, a volume-based missing-data strategy (flag + impute for large gaps, quiet impute for negligible ones)
6. **Modelling** — Logistic Regression baseline → Random Forest → XGBoost, compared honestly
7. **Tuning** — `RandomizedSearchCV` with 3-fold cross-validation, 25-combination search
8. **Explainability** — SHAP (TreeExplainer) on the final model

## Model comparison

| Model | ROC-AUC (test) |
|---|---|
| Logistic Regression (baseline) | 0.741 |
| Random Forest (untuned) | 0.765 |
| XGBoost (untuned) | 0.762 |
| **XGBoost (tuned)** | **0.771** |

Each step bought a small, real improvement — not a dramatic one. Most of the predictive power was already present in the simplest model; added complexity earned incremental gains on top of it.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · matplotlib

## Repository structure

```
sydney_airbnb_guest_satisfaction.ipynb   — full annotated pipeline, cleaning through SHAP
top_tier_xgb_model.pkl                   — trained, tuned XGBoost model (joblib)
model_feature_columns.pkl                — exact feature column list/order the model expects
confusion_matrix.png                     — test-set confusion matrix, tuned model
shap_summary.png                         — SHAP beeswarm plot, feature impact by listing
```

## Limitations & future work

- `price` is genuinely unavailable in this scrape; a future version could attempt an earlier scrape period if price data was populated then
- `property_type` (60 unique values) excluded from this baseline due to cardinality — a grouped version could recover some signal
- `neighbourhood_cleansed` is one-hot encoded as-is; target or frequency encoding may generalise better
- Single point-in-time snapshot (September 2025) — no seasonality captured

## Reproducing this

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
```
Download `listings.csv` from [Inside Airbnb's Sydney page](http://insideairbnb.com/get-the-data/), place it alongside the notebook, and run top to bottom.

---

**Mahesh Sai Kandula** — Master of Data Science, Macquarie University · [LinkedIn](https://www.linkedin.com/in/mahesh-kandula-b6393622a/) · [GitHub](https://github.com/MSkandula)
