# Airline-Loyalty-Project

# Unlocking Behavioral Intelligence in Airline Loyalty Programs

Consulting & Analytics Club — IIT Guwahati | Summer Projects 2026

An end-to-end analytics pipeline that predicts customer churn and segments an airline's loyalty program members, turning two years of raw flight-activity logs into a decision-ready retention strategy.

---

## Problem Statement

Build a solution that helps the marketing team:
1. Identify customers who are likely to disengage (churn)
2. Understand which members are most valuable
3. Recommend targeted actions to improve retention

## Dataset

| File | Rows | Description |
|---|---|---|
| `Customer Loyalty History.csv` | 16,737 | One row per customer — demographics, tier, CLV, enrollment |
| `Customer Flight Activity.csv` | 392,936 | One row per customer per month (Jan 2017 – Dec 2018) |
| `Calendar.csv` | 2,557 | Date reference table |

Scope: 16,737 Canadian loyalty members, 2 years of monthly flight activity.

---

## Pipeline Overview

The notebook is organized into three sequential phases, each with a clear input/output contract.

### Phase 1 — Data Understanding & Exploration
Pure observation, no data is modified.
- Schema and null audit across both files
- Distinguished structural nulls (Salary NaN = student with no income, Cancellation NaN = still active) from true missingness
- Found and logged 20 negative `Salary` values (sign-error typos) and confirmed zero orphaned records on the join key — merge is clean
- Established that single-month inactivity is normal (54.5% of monthly rows are zero-flight), but zero flights across the full 24-month period is statistically anomalous — this became the basis for the churn label

### Phase 2 — Feature Engineering & Churn Label
Transforms 392,936 monthly rows into one model-ready row per customer.

Churn label — two definitions compared and combined:
- *Formal churn:* recorded cancellation (only 12.3% of members — too narrow on its own)
- *Behavioral churn:* zero flights across the entire 2017–2018 period
- Final churn rate: 16.0% (2,686 / 16,737) — realistic and defensible for a loyalty program; class imbalance handled via `scale_pos_weight` in XGBoost

Feature groups engineered (33 features total):
- Demographic — Salary, Education, Marital Status, Loyalty Card tier, `Tenure_Months`
- Geographic — Province consolidated into 3 Regions (East / West / North) to fix category sparsity
- Behavioral — total/avg/max flights, distance, points accumulated & redeemed, redemption rate
- Recency — `Months_Since_Last_Flight` (later confirmed by SHAP as the single strongest predictor)
- Seasonal — per-season flight counts, season-concentration score (captures seasonal vs. year-round flyers)
- Derived — log-transformed CLV, flight consistency score

Categorical encoding: One-Hot for unordered categories (Gender, Marital Status, Region), Label Encoding for ordered categories (Loyalty Card, Education) — chosen deliberately since K-Means' Euclidean distance would otherwise impose false ordinal structure on one-hot-appropriate columns.

Output: `modelling2.csv` — 16,737 rows × 33 features

### Phase 3 — Machine Learning Modelling

Task A — Churn Prediction
Three candidates trained and tuned via `GridSearchCV(cv=5, scoring='recall')` on the same split:

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 97% | 88% | 0.92 | 97.0% |
| XGBoost | 95% | 91% | 0.93 | 97.8% |
| Random Forest (selected) | 96% | 91% | 0.94 | 98.1% |

Recall was prioritized as the primary metric — a missed churner is costlier than a false alarm. Every scored customer also gets a churn probability and a risk tier (High ≥ 0.5, Medium 0.2–0.5, Low < 0.2) for prioritizing retention spend.

Task B — Customer Segmentation (K-Means, validated against Agglomerative/Ward clustering)
Optimal k=3 confirmed by elbow method, silhouette score, and Davies-Bouldin index, cross-checked against a dendrogram from Ward-linkage hierarchical clustering.

| Segment | Size | Churn Rate | Avg Tenure | Profile |
|---|---|---|---|---|
| High-Frequency New Flyers | 5% | 11.3% | 9 months | Fly most, never redeem points, newest members |
| Disengaged At-Risk Members | 26.4% | 53.4% | 24.6 months | Inactive 21/24 months, draining points |
| Loyal Core Members | 68.6% | 2.0% | 45.7 months | Longest tenured, stable, program backbone |

Key business insight: 19% of High-Frequency New Flyers are already high-risk despite flying the most — frequent flying without program engagement (0% redemption rate) is an early warning sign, not a loyalty signal.

---

## Tech Stack

- Data processing: pandas, NumPy
- Modelling: scikit-learn (Logistic Regression, Random Forest, K-Means, Agglomerative Clustering), XGBoost
- Visualization: matplotlib, seaborn
- Dashboard: Streamlit + Plotly

## Repository Structure

```
├── Airline_Loyalty_Complete_Report_github.ipynb   # Full analysis: Phase 1 → 2 → 3
├── Customer Loyalty History.csv                   # Raw input (not committed — see Data note)
├── Customer Flight Activity.csv                    # Raw input (not committed — see Data note)
├── Calendar.csv                                     # Raw input (not committed — see Data note)
├── modelling2.csv                                   # Phase 2 output — model-ready feature matrix
├── final_customer_segments.csv                       # Phase 3 output — churn scores + segments
└── README.md
```

> Data note: Raw CSVs are from the Consulting & Analytics Club, IIT Guwahati Summer Projects 2026 dataset and are not redistributed in this repo. Place the three source files in the project root before running the notebook top to bottom.

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit plotly
jupyter notebook Airline_Loyalty_Complete_Report_github.ipynb
```

Run all cells sequentially — Phase 2 depends on Phase 1's cleaned frames, and Phase 3 depends on `modelling2.csv` from Phase 2.

## Author

Shashank Shekhar — M.Sc. Statistics (Applied Statistics and Informatics), IIT Bombay
