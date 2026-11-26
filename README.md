# Customer churn, CLV and retention targeting for a Singapore loyalty app

**Question:** Which at-risk customers should a Singapore retailer send a S$5 retention voucher to, and what is
the expected profit, measured honestly?

> **Data note:** There is no public customer-level transaction data for Singapore businesses, so this
> project uses a **synthetic** loyalty-app dataset (40,000 customers, ~2.75 years). Its demographics are
> calibrated to **SingStat**: planning-area weights from June 2026 resident counts (28 residential areas,
> ~98% of residents), dwelling mix from the General Household Survey 2025 (HDB 1-3rm 23.9%, 4rm 31.2%,
> 5rm/EA 22.1%, condo 17.9%, landed 4.7%), and age bands from Table M810011 for residents aged 18-64
> (10.9 / 21.4 / 23.7 / 22.5 / 21.6%). It also uses 9% GST and CNY, 11.11/12.12, Great Singapore Sale and
> public-holiday seasonality. **Basket sizes and all behavioural parameters are assumptions**, and 65+ are
> assumed not to use the app (see `reference/SOURCES.md`). The retention campaign is a **randomized
> experiment** with a known mechanism, so targeting rules can be evaluated without guessing.
> `src/appendix_real_data.py` checks the same methods on two real datasets (CDNOW, Online Retail II).

## Pipeline

```
generate_data.py -> sql/features.sql (DuckDB) -> train_churn.py -> clv.py -> decision.py -> uplift.py -> sensitivity.py + playbook.py
                                      \-> survival.csv -> r/survival.R
```

| Step | What it does |
|---|---|
| `make data` | Simulates customers, orders and a randomized campaign (`data/raw/`) |
| `make features` | SQL window-function features at 7 quarterly cutoffs; 90-day churn labels |
| `make train` | Logistic baseline vs LightGBM, time-based split, isotonic calibration, SHAP |
| `make clv` | BG/NBD + Gamma-Gamma 12-month CLV, validated on the next 90 days |
| `make decide` | Profit curves for 4 targeting rules, scored on the randomized test |
| `make uplift` | Cross-fitted T-learner uplift model, bootstrap confidence intervals |
| `make sensitivity` | Grid over voucher cost, margin and save rate; which rule wins where (`reports/sensitivity_grid.csv`) |
| `make playbook` | Target / Skip / Unclear verdict per segment, from measured randomized lift with 95% intervals |
| `make test` | Leakage and sanity tests (no future columns, no label overlap, randomization balance) |
| `make tableau` | Tableau-ready extracts (planning-area summary with coordinates, risk x value matrix) and a matplotlib dashboard *preview* |
| `make appendix` | Same methods on **real** CDNOW transactions: BG/NBD holdout validation and churn model comparison |
| `make appendix-retail` | Same, on real **UCI Online Retail II** (first download it and run `python -m src.prepare_online_retail`, see below) |
| `make survival` | Kaplan-Meier + Cox models in R |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all            # about a few minutes
make survival       # included in `make all`; needs R with the `survival` package (tested with survival 3.5)
```

## Design choices
- **Snapshot design:** features use orders before the cutoff; labels use the following 90 days. Train on
  the first five cutoffs, validate on the sixth, test on the last (2025-01-01), whose label windows
  never overlap the next cutoff's features.
- **"In play" population:** only customers who ordered in the previous 180 days are scored.
- **Calibration:** isotonic regression on the validation window, because probabilities feed dollar decisions.
- **Net of GST:** all revenue is divided by 1.09 before margin is applied.
- **Evaluation:** assumption-based profit chooses the cutoff (no outcome peeking); the randomized
  treated-vs-control difference measures realised incremental profit.

## Results (seed 42)

**Synthetic data, randomized test (19,118 in-play customers at the 2025-01-01 cutoff)**
- LightGBM ROC-AUC 0.863 and PR-AUC 0.817 vs logistic 0.850; calibrated probabilities track observed churn by decile.
- Ranking by P(churn) alone **loses money** (S$-8.5k when targeting the top 40%) because it picks customers who have already gone. Risk and persuadability are negatively correlated (-0.18).
- **Smart targeting pays only when contact is costly relative to margin** (25% margin, measured on the randomized test):

| Cost per voucher | Target everyone | Uplift rule | Churn x CLV rule | RFM top 40% |
|---|---|---|---|---|
| S$1 | **S$31.4k** | S$26.3k | S$26.9k | S$20.7k |
| S$2 | S$12.2k | **S$16.1k** | S$13.8k | S$13.0k |
| S$3 | S$-6.9k | **S$13.0k** | S$4.0k | S$5.4k |
| S$5 | S$-45.1k | **S$8.1k** | S$-2.1k | S$-9.9k |

- At S$2 the uplift rule's 95% interval is [0.7k, 29.8k] vs [-5.0k, 31.6k] for targeting everyone: the intervals overlap, so only the S$3-5 results are clear-cut.
- **Segment playbook:** none of the 20 segments (age, channel, housing, region) clears 95% confidence as Target or Skip. At about 19k customers, fine segments are too noisy to resolve; the uplift model, which uses all features jointly, does better (correlation 0.23 with the true uplift driver).
- **Survival (R):** median lifetime 219 days for Partner-bank-promo customers vs 269 for Referral (log-rank p < 1e-10); hazard ratios 1.09 (Partner bank promo), 1.10 (age 18-24), 0.95 (Referral). Housing type and region are not significant. Concordance is only 0.52, and the proportional-hazards assumption is mildly violated for channel (p = 0.025).

**Real-data appendix**

| Dataset | Customers / orders | BG/NBD holdout (180d), predicted vs actual | Churn ROC-AUC: logistic / LightGBM / BG/NBD |
|---|---|---|---|
| CDNOW (1997-98) | 23,502 / 69,579 | 12,349 vs 12,202 (corr 0.70) | 0.710 / 0.707 / **0.721** |
| UCI Online Retail II (2009-11) | 5,878 / 36,969 | 7,620 vs 7,743 (corr 0.85) | 0.762 / 0.768 / **0.772** |

- **A four-parameter structural model (BG/NBD) matched or beat gradient boosting on both real datasets.** Differences of about 0.01 AUC on 2,700-6,000 test customers are within noise, but the pattern is consistent, and real-world accuracy (0.71-0.77) is far below the synthetic 0.86.
- Gamma-Gamma overstates average order value: by 8% on CDNOW (40.24 vs 37.13) and 12% on Online Retail II (GBP 470 vs 418; wholesale customers make spend heavy-tailed).
- Neither real dataset has a campaign, so uplift and profit evaluation cannot be repeated on real data.
- Online Retail II: CC BY 4.0, Chen (2012), UCI ML Repository, DOI 10.24432/C5CG6D. Cleaning: drop cancellations ("C" invoices), non-positive quantity or price, and guest checkouts (no customer ID); one order = one invoice.

## Getting Online Retail II
Download `online_retail_II.xlsx` from the UCI repository (dataset 502), then:
```bash
pip install -r requirements.txt
python -m src.prepare_online_retail --src online_retail_II.xlsx --out data/external/online_retail_ii_orders.csv
make appendix-retail
```

## Reading the results honestly
- The segment verdicts use 95% intervals across about 20 comparisons; with that many tests you expect roughly one false positive by chance, so treat any future "Target" label as a hypothesis to test.
- In the sensitivity grid the uplift rule does not depend on the assumed save rate, so each (cost, margin) scenario appears three times; count winners over the 12 distinct scenarios (uplift 9, target everyone 3 at save rate 0.15).
- Point estimates only in the grid. The robust conclusion is the cost pattern in the table, not a precise profit figure.

## Limitations
Synthetic data (demographics calibrated to SingStat; behaviour, basket sizes and the 18-64 scope are assumptions); one campaign, one 90-day outcome window; CLV x P(churn) partly double counts dropout;
margin and cost are assumptions; the uplift signal is modest and noisy.

## Replacing the synthetic data
Keep the same schemas (`customers`, `orders`, `campaign`) and everything downstream works. Mind PDPA:
anonymise IDs and avoid precise addresses; planning area is the finest geography you need.
