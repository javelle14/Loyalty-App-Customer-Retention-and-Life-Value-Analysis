"""Sanity and leakage tests. Run after `make all`:  pytest -q"""
from pathlib import Path
import pandas as pd
import pytest
from src.config import CUTOFFS, TRAIN_CUTOFFS, VALID_CUTOFF, TEST_CUTOFF, HORIZON_DAYS
from src.train_churn import NUM, CAT, TARGET

P = Path("data/processed")


@pytest.fixture(scope="module")
def snap():
    return pd.read_parquet(P / "snapshots.parquet")


def test_no_future_columns_used_as_features():
    banned = ("fut_", "campaign_", "churn_")
    assert not [c for c in NUM + CAT if c.startswith(banned)]


def test_target_matches_future_orders(snap):
    assert ((snap.fut_orders_90d == 0).astype(int) == snap[TARGET]).all()


def test_features_only_use_past(snap):
    # latest order strictly before cutoff, and within the in-play window
    assert (snap.recency_days >= 1).all() and (snap.recency_days <= 180).all()
    assert (snap.tenure_days >= 0).all()


def test_time_split_has_no_label_overlap():
    cut = pd.to_datetime(CUTOFFS)
    for t in TRAIN_CUTOFFS:
        assert pd.Timestamp(t) + pd.Timedelta(days=HORIZON_DAYS) <= pd.Timestamp(VALID_CUTOFF)
    assert pd.Timestamp(VALID_CUTOFF) + pd.Timedelta(days=HORIZON_DAYS) <= pd.Timestamp(TEST_CUTOFF)
    assert list(cut) == sorted(cut)


def test_one_row_per_customer_per_cutoff(snap):
    assert not snap.duplicated(["customer_id", "cutoff"]).any()


def test_campaign_is_randomized_and_balanced(snap):
    t = snap[pd.to_datetime(snap.cutoff) == pd.Timestamp(TEST_CUTOFF)].dropna(subset=["campaign_treated"])
    assert 0.47 < t.campaign_treated.mean() < 0.53
    for col in ["recency_days", "n_orders_all", "net_rev_all"]:   # pre-treatment covariates balanced
        a, b = t[t.campaign_treated == 1][col], t[t.campaign_treated == 0][col]
        assert abs(a.mean() - b.mean()) < 0.1 * t[col].std()


def test_scored_test_set_is_only_the_test_cutoff():
    te = pd.read_parquet(P / "test_scored.parquet")
    assert (pd.to_datetime(te.cutoff) == pd.Timestamp(TEST_CUTOFF)).all()
    assert te.p_churn.between(0, 1).all()


# ---------- calibration to SingStat reference tables ----------
def test_generated_customers_match_singstat_calibration():
    cust = pd.read_parquet(Path("data/raw/customers.parquet"))
    ref_h = pd.read_csv("reference/sg_dwelling_2025.csv").set_index("housing_type").share_pct
    ref_h = ref_h / ref_h.sum()
    got_h = cust.housing_type.value_counts(normalize=True)
    assert (got_h.reindex(ref_h.index) - ref_h).abs().max() < 0.015       # within 1.5 pp

    ref_a = pd.read_csv("reference/sg_planning_areas.csv").set_index("planning_area").resident_pop_2026
    ref_a = ref_a / ref_a.sum()
    got_a = cust.planning_area.value_counts(normalize=True)
    assert (got_a.reindex(ref_a.index) - ref_a).abs().max() < 0.01        # within 1 pp
    assert set(cust.planning_area) == set(ref_a.index)


def test_age_bands_match_singstat_m810011():
    cust = pd.read_csv("reference/sg_age_2026.csv").set_index("age_band")
    ref = cust.population_2026 / cust.population_2026.sum()
    got = pd.read_parquet(Path("data/raw/customers.parquet")).age_band.value_counts(normalize=True)
    assert set(got.index) == set(ref.index)
    assert (got.reindex(ref.index) - ref).abs().max() < 0.01              # within 1 pp
    # the 5-year file must reproduce SingStat's published 2026 resident total
    assert pd.read_csv("reference/sg_residents_age_2026_5yr.csv").residents_2026.sum() == 4231524
