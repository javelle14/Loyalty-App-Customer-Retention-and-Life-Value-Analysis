"""
Segment playbook: which customer groups should marketing target, skip, or test further?

For each segment we report the MEASURED incremental margin per customer from the randomized
campaign (treated minus control) net of cost, with a 95% interval, plus the uplift model's average
prediction as a cross-check. Verdicts use the interval, so noisy segments are labelled Unclear.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import GROSS_MARGIN, COST_PER_TARGET

OUT, REP = Path("data/processed"), Path("reports")
SEGMENTS = ["age_band", "acquisition_channel", "housing_type", "region"]


def main() -> None:
    te = pd.read_parquet(OUT / "test_scored.parquet")
    up = pd.read_parquet(OUT / "uplift_scores.parquet")
    d = te.merge(up, on="customer_id").dropna(subset=["campaign_treated"]).copy()
    d["margin_90d"] = d.fut_net_rev_90d * GROSS_MARGIN

    rows = []
    for seg in SEGMENTS:
        for val, g in d.groupby(seg):
            t, c = g[g.campaign_treated == 1].margin_90d, g[g.campaign_treated == 0].margin_90d
            lift = t.mean() - c.mean() - COST_PER_TARGET
            se = np.sqrt(t.var() / len(t) + c.var() / len(c))
            lo, hi = lift - 1.96 * se, lift + 1.96 * se
            verdict = "Target" if lo > 0 else "Skip" if hi < 0 else "Unclear"
            rows.append({"dimension": seg, "segment": val, "n": len(g), "avg_p_churn": round(g.p_churn.mean(), 3),
                         "measured_net_lift_sgd": round(lift, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2),
                         "model_avg_tau_minus_cost": round(g.tau_hat.mean() - COST_PER_TARGET, 2),
                         "verdict": verdict})
    pb = pd.DataFrame(rows).sort_values(["dimension", "measured_net_lift_sgd"], ascending=[True, False])
    pb.to_csv(REP / "segment_playbook.csv", index=False)
    pb.to_csv(REP / "tableau" / "segment_playbook.csv", index=False)
    print(pb.to_string(index=False))
    print("\nVerdict counts:", pb.verdict.value_counts().to_dict())


if __name__ == "__main__":
    main()
