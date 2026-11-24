"""
Appendix: does the same method work on REAL transaction data?

Default dataset: CDNOW (real CD-store purchases, 23,570 customers, Jan 1997 - Jun 1998), bundled with the
`lifetimes` package. To use another dataset (e.g. UCI Online Retail II, which this sandbox could not
download), pass a CSV:

    python -m src.prepare_online_retail --src online_retail_II.xlsx --out data/external/online_retail_ii_orders.csv
    python -m src.appendix_real_data --csv data/external/online_retail_ii_orders.csv --name online_retail_ii

What it checks
  1. BG/NBD + Gamma-Gamma: fit on a calibration period, predict a true holdout period, compare by decile.
  2. Churn model (same snapshot/time-split logic as the main project): logistic vs LightGBM vs the
     BG/NBD's own "expected orders" score, all evaluated on the latest snapshot.
There is no campaign in this data, so no uplift or profit evaluation is possible; that is a real limitation.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import lifetimes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H, IN_PLAY = 90, 180


def load(args) -> pd.DataFrame:
    if args.csv:
        d = pd.read_csv(args.csv)
        d = d.rename(columns={args.cust: "cust", args.date: "date", args.amount: "amount"})
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    else:
        f = os.path.join(os.path.dirname(lifetimes.__file__), "datasets", "CDNOW_master.txt")
        d = pd.read_csv(f, sep=r"\s+", dtype={"customer_id": str})
        d = d.rename(columns={"customer_id": "cust", "dollar_value": "amount"})
        d["date"] = pd.to_datetime(d["date"].astype(str), format="%Y%m%d")
    d = d[["cust", "date", "amount"]].dropna()
    return d[d.amount > 0].reset_index(drop=True)


def snapshot(tx: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    c = pd.Timestamp(cutoff)
    past = tx[tx.date < c]
    g = past.groupby("cust")
    f = pd.DataFrame({"last": g.date.max(), "first": g.date.min(), "n_orders": g.size(), "rev": g.amount.sum()})
    for w in (90, 180):
        f[f"n_{w}d"] = past[past.date >= c - pd.Timedelta(days=w)].groupby("cust").size()
    f = f.fillna(0)
    f["recency"] = (c - f["last"]).dt.days
    f["tenure"] = (c - f["first"]).dt.days
    f["aov"] = f.rev / f.n_orders
    f["avg_gap"] = (f["last"] - f["first"]).dt.days / (f.n_orders - 1).replace(0, np.nan)
    f["overdue"] = (f.recency / f.avg_gap.clip(lower=1)).fillna(0)     # same-day repeat orders give gap 0
    f = f[f.recency <= IN_PLAY]                                    # "in play" customers only
    fut = tx[(tx.date >= c) & (tx.date < c + pd.Timedelta(days=H))].groupby("cust").size()
    f["churn"] = (~f.index.isin(fut.index)).astype(int)
    f["cutoff"] = c
    return f.drop(columns=["last", "first"]).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--cust", default="cust")
    ap.add_argument("--date", default="date"); ap.add_argument("--amount", default="amount")
    ap.add_argument("--name", help="tag for output files (default: cdnow, or the CSV file name)")
    args = ap.parse_args()
    tx = load(args)
    name = args.name or (Path(args.csv).stem if args.csv else "cdnow")
    end = tx.date.max()
    print(f"{tx.cust.nunique():,} customers | {len(tx):,} orders | {tx.date.min().date()} to {end.date()}")

    # ---- 1. BG/NBD holdout validation ----
    cal_end = end - pd.Timedelta(days=180)
    cal = calibration_and_holdout_data(tx, "cust", "date", calibration_period_end=cal_end,
                                       observation_period_end=end, monetary_value_col="amount")
    bgf = BetaGeoFitter(penalizer_coef=0.001).fit(cal.frequency_cal, cal.recency_cal, cal["T_cal"])
    cal["pred"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        cal.duration_holdout.iloc[0], cal.frequency_cal, cal.recency_cal, cal["T_cal"])
    cal["decile"] = pd.qcut(cal.pred.rank(method="first"), 10, labels=False) + 1
    dec = cal.groupby("decile")[["pred", "frequency_holdout"]].mean()
    print(f"\n[1] BG/NBD holdout ({int(cal.duration_holdout.iloc[0])} days) by predicted-purchase decile")
    print(dec.round(3).T.to_string())
    print(f"    total predicted {cal.pred.sum():,.0f} vs actual {cal.frequency_holdout.sum():,.0f} "
          f"| corr {cal.pred.corr(cal.frequency_holdout):.3f}")
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(dec.index, dec.pred, "o-", label="BG/NBD predicted")
    plt.plot(dec.index, dec.frequency_holdout, "s--", label="Actual")
    plt.xlabel("Predicted-purchase decile"); plt.ylabel("Avg purchases in holdout"); plt.legend()
    plt.title(f"Real data ({name}): BG/NBD holdout"); plt.tight_layout()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"reports/figures/appendix_{name}_bgnbd_holdout.png", dpi=150); plt.close()

    # Gamma-Gamma: spend per order, holdout check on repeat buyers
    rep = cal[(cal.frequency_cal > 0) & (cal.monetary_value_cal > 0) & (cal.frequency_holdout > 0)]
    ggf = GammaGammaFitter(penalizer_coef=0.01).fit(cal[(cal.frequency_cal > 0) & (cal.monetary_value_cal > 0)].frequency_cal,
                                                   cal[(cal.frequency_cal > 0) & (cal.monetary_value_cal > 0)].monetary_value_cal)
    pred_aov = ggf.conditional_expected_average_profit(rep.frequency_cal, rep.monetary_value_cal)
    print(f"    Gamma-Gamma avg order value: predicted {pred_aov.mean():.2f} vs actual {rep.monetary_value_holdout.mean():.2f}")

    # ---- 2. churn models on time-split snapshots ----
    cuts = pd.date_range(tx.date.min() + pd.Timedelta(days=90), end - pd.Timedelta(days=H), freq="QS")
    cuts = [c for c in cuts if c + pd.Timedelta(days=H) <= end]
    snaps = pd.concat([snapshot(tx, c.strftime("%Y-%m-%d")) for c in cuts], ignore_index=True)
    test_c = snaps.cutoff.max()
    valid_c = sorted(snaps.cutoff.unique())[-2]
    tr = snaps[snaps.cutoff < valid_c]; te = snaps[snaps.cutoff == test_c]
    assert tr.cutoff.max() + pd.Timedelta(days=H) <= valid_c <= test_c      # no label overlap
    X = ["recency", "n_orders", "n_90d", "n_180d", "rev", "aov", "tenure", "avg_gap", "overdue"]
    print(f"\n[2] churn snapshots: {[c.date().isoformat() for c in cuts]} | train rows {len(tr):,}, test rows {len(te):,}, "
          f"test churn rate {te.churn.mean():.1%}")
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[X].fillna(0), tr.churn)
    gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, verbose=-1).fit(tr[X], tr.churn)
    # BG/NBD score fitted only on orders before the test cutoff
    s = summary_data_from_transaction_data(tx[tx.date < test_c], "cust", "date",
                                           observation_period_end=test_c - pd.Timedelta(days=1))
    b2 = BetaGeoFitter(penalizer_coef=0.001).fit(s.frequency, s.recency, s["T"])
    s["exp90"] = b2.conditional_expected_number_of_purchases_up_to_time(H, s.frequency, s.recency, s["T"])
    te = te.merge(s[["exp90"]], left_on="cust", right_index=True, how="left")
    scores = {"Logistic": logit.predict_proba(te[X].fillna(0))[:, 1],
              "LightGBM": gbm.predict_proba(te[X])[:, 1], "BG/NBD (1 - exp. orders)": -te.exp90.values}
    res = pd.DataFrame([{"model": k, "roc_auc": roc_auc_score(te.churn, v), "pr_auc": average_precision_score(te.churn, v)}
                        for k, v in scores.items()]).round(4)
    res.loc[len(res)] = ["(base rate)", 0.5, round(te.churn.mean(), 4)]
    print(res.to_string(index=False))
    res.to_csv(f"reports/appendix_{name}_metrics.csv", index=False)


if __name__ == "__main__":
    main()
