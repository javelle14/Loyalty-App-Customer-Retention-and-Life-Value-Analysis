"""
Sensitivity analysis: how do the winning targeting rule and its profit change when the business
assumptions (voucher cost, gross margin, assumed save rate) change?

All profits are measured on the randomized test (treated vs control), not assumed.
The cutoff for the churn x CLV rule is chosen from ASSUMED profit (no outcome peeking).
"""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from .config import GROSS_MARGIN

OUT, REP = Path("data/processed"), Path("reports")
COSTS, MARGINS, SAVE_RATES = [1, 2, 3, 5], [0.15, 0.25, 0.35], [0.05, 0.15, 0.25]


def rct_profit(d, mask, cost, margin):
    s = d[mask]
    t, c = s[s.campaign_treated == 1], s[s.campaign_treated == 0]
    if len(t) < 30 or len(c) < 30:
        return np.nan
    return ((t.fut_net_rev_90d.mean() - c.fut_net_rev_90d.mean()) * margin - cost) * len(s)


def main() -> None:
    te = pd.read_parquet(OUT / "test_scored.parquet")
    clv = pd.read_parquet(OUT / "clv_test.parquet")[["customer_id", "clv_12m_margin"]]
    up = pd.read_parquet(OUT / "uplift_scores.parquet")
    d = te.merge(clv, on="customer_id").merge(up, on="customer_id").dropna(subset=["campaign_treated"]).reset_index(drop=True)
    rfm = d.n_orders_all.rank(pct=True) + d.net_rev_all.rank(pct=True) + d.recency_days.rank(pct=True)

    rows = []
    for cost, m, sr in product(COSTS, MARGINS, SAVE_RATES):
        f = m / GROSS_MARGIN                           # CLV and tau_hat were built at GROSS_MARGIN
        var = d.p_churn * d.clv_12m_margin * f
        order = var.rank(ascending=False, method="first")
        ks = np.arange(0.05, 1.0001, 0.05)
        assumed = [(sr * var[order <= k * len(d)] - cost).sum() for k in ks]
        k_star = ks[int(np.argmax(assumed))]
        all_ = np.ones(len(d), bool)
        rules = {
            "Target everyone": all_,
            "Churn x CLV rule": (order <= k_star * len(d)).values,
            "RFM top 40%": (rfm.rank(ascending=False, method="first") <= 0.4 * len(d)).values,
            "P(churn) top 40%": (d.p_churn.rank(ascending=False, method="first") <= 0.4 * len(d)).values,
            "Uplift rule": (d.tau_hat * f > cost).values,
        }
        res = {k: rct_profit(d, v, cost, m) for k, v in rules.items()}
        best = max(res, key=lambda k: -np.inf if np.isnan(res[k]) else res[k])
        rows.append({"cost_sgd": cost, "gross_margin": m, "assumed_save_rate": sr, "k_star_churn_clv": k_star,
                     **{f"profit_{k}": round(v) for k, v in res.items()}, "winner": best,
                     "winner_profit": round(res[best])})
    g = pd.DataFrame(rows)
    g.to_csv(REP / "sensitivity_grid.csv", index=False)
    print("Winner counts across", len(g), "scenarios:\n", g.winner.value_counts().to_string())
    # save_rate only changes the churn x CLV cutoff, so compare rules by (cost, margin)
    piv = g.groupby(["cost_sgd", "gross_margin"]).first().reset_index()
    print("\nProfit by rule (S$), save_rate=0.05 cutoff for churn x CLV:")
    print(piv[["cost_sgd", "gross_margin"] + [c for c in g.columns if c.startswith("profit_")] + ["winner"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
