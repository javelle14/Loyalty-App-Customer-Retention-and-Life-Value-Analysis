"""
Build the Tableau-ready extracts and a matplotlib PREVIEW of the intended dashboard.

NOTE: the preview is a reference picture drawn with matplotlib, not a Tableau workbook. Build the real
dashboard in Tableau Desktop/Public by following docs/TABLEAU_GUIDE.md.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import GROSS_MARGIN, COST_PER_TARGET

OUT, REP, TAB = Path("data/processed"), Path("reports"), Path("reports/tableau")


def main() -> None:
    cs = pd.read_csv(TAB / "customer_scores.csv")
    ref = pd.read_csv("reference/sg_planning_areas.csv")
    pb = pd.read_csv(REP / "segment_playbook.csv")
    curves = pd.read_csv(REP / "profit_curves.csv")
    upl = pd.read_csv(REP / "uplift_profit_curve.csv")

    # ---- extract 1: planning-area summary with coordinates (for a symbol map) ----
    def lift(g):
        t, c = g[g.campaign_treated == 1].margin_90d, g[g.campaign_treated == 0].margin_90d
        return t.mean() - c.mean() - COST_PER_TARGET
    pa = cs.groupby("planning_area").apply(lambda g: pd.Series({
        "customers": len(g), "avg_p_churn": g.p_churn.mean(), "value_at_risk_sgd": g.value_at_risk.sum(),
        "targeted_customers": g.target_flag.sum(), "measured_net_lift_per_customer": lift(g)}), include_groups=False)
    pa = ref[["planning_area", "region", "lat", "lon", "resident_pop_2026"]].merge(pa.reset_index(), on="planning_area")
    pa.to_csv(TAB / "planning_area_summary.csv", index=False)

    # ---- extract 2: risk x value matrix ----
    mx = cs.groupby(["risk_tier", "value_tier"]).agg(customers=("customer_id", "size"),
                                                     value_at_risk_sgd=("value_at_risk", "sum")).reset_index()
    mx.to_csv(TAB / "risk_value_matrix.csv", index=False)

    # ---- preview figure ----
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.25, 1], hspace=0.38, wspace=0.5)
    kp = fig.add_subplot(gs[0, :]); kp.axis("off")
    k = [("Customers scored", f"{len(cs):,}"), ("Avg P(churn)", f"{cs.p_churn.mean():.1%}"),
         ("Value at risk", f"S${cs.value_at_risk.sum():,.0f}"), ("Targeted (rule)", f"{int(cs.target_flag.sum()):,}")]
    for i, (a, b) in enumerate(k):
        kp.text(0.06 + i * 0.24, 0.62, b, fontsize=24, weight="bold"); kp.text(0.06 + i * 0.24, 0.2, a, fontsize=11, color="grey")
    kp.set_title("Retention targeting dashboard: REFERENCE PREVIEW (build in Tableau; see docs/TABLEAU_GUIDE.md)", loc="left", fontsize=12)

    a1 = fig.add_subplot(gs[1, 0])
    m = mx.pivot(index="risk_tier", columns="value_tier", values="customers").reindex(["High", "Medium", "Low"])[["Low", "Medium", "High"]]
    a1.imshow(m.values, cmap="Blues")
    for i in range(3):
        for j in range(3):
            a1.text(j, i, f"{m.values[i, j]:,}", ha="center", va="center")
    a1.set_xticks(range(3), m.columns); a1.set_yticks(range(3), m.index)
    a1.set_xlabel("Value tier (12m margin CLV)"); a1.set_ylabel("Churn-risk tier"); a1.set_title("Risk x value matrix")

    a2 = fig.add_subplot(gs[1, 1])
    sc = a2.scatter(pa.lon, pa.lat, s=pa.value_at_risk_sgd / pa.value_at_risk_sgd.max() * 900, c=pa.avg_p_churn, cmap="OrRd", alpha=0.8, edgecolor="k")
    for r in pa.nlargest(6, "value_at_risk_sgd").itertuples():
        a2.annotate(r.planning_area, (r.lon, r.lat), fontsize=7, xytext=(4, 4), textcoords="offset points")
    a2.set_aspect(1); a2.set_xticks([]); a2.set_yticks([]); plt.colorbar(sc, ax=a2, shrink=0.6, pad=0.03).set_label("avg P(churn)", fontsize=8)
    a2.set_title("Planning areas: size = value at risk")

    a3 = fig.add_subplot(gs[1, 2])
    for name in ["Model: P(churn) x CLV", "RFM heuristic", "P(churn) only"]:
        s = curves[curves.strategy == name]; a3.plot(s.pct_targeted * 100, s.profit_rct, label=name)
    a3.plot(upl.pct_targeted * 100, upl.profit_rct, "k--", label="Uplift ranking")
    a3.axhline(0, c="grey", lw=0.8); a3.legend(fontsize=7); a3.set_xlabel("% of at-risk base targeted")
    a3.set_title("Measured incremental profit (S$)")
    fig.savefig(REP / "figures" / "dashboard_preview.png", dpi=140, bbox_inches="tight"); plt.close()
    print("wrote planning_area_summary.csv, risk_value_matrix.csv, dashboard_preview.png")


if __name__ == "__main__":
    main()
