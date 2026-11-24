"""
Turn UCI Online Retail II line items into an order-level CSV (cust, date, amount) for the appendix.

Accepts the UCI .xlsx (both sheets) or a CSV with the same columns. Column names are matched flexibly
(Invoice/InvoiceNo, Price/UnitPrice, CustomerID/"Customer ID").

Cleaning rules (state these in your write-up):
  * drop invoices starting with 'C' (cancellations) and lines with Quantity <= 0 or Price <= 0
  * drop rows with no customer ID (guest checkouts cannot be tracked over time)
  * one order = one invoice; amount = sum(Quantity * Price) in GBP; date = invoice date
Usage:
  python -m src.prepare_online_retail --src online_retail_II.xlsx --out data/external/online_retail_ii_orders.csv
"""
import argparse
from pathlib import Path
import pandas as pd

ALIASES = {"invoice": ["Invoice", "InvoiceNo"], "qty": ["Quantity"], "date": ["InvoiceDate"],
           "price": ["Price", "UnitPrice"], "cust": ["Customer ID", "CustomerID"]}


def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in {list(df.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.src.lower().endswith((".xlsx", ".xls")):
        raw = pd.concat(pd.read_excel(a.src, sheet_name=None).values(), ignore_index=True)   # needs openpyxl
    else:
        raw = pd.read_csv(a.src, dtype={"Invoice": str, "InvoiceNo": str})
    c = {k: pick(raw, v) for k, v in ALIASES.items()}
    n0 = len(raw)
    d = raw.rename(columns={v: k for k, v in c.items()})[list(c)]
    d["invoice"] = d.invoice.astype(str)
    d = d[~d.invoice.str.upper().str.startswith("C") & (d.qty > 0) & (d.price > 0)].dropna(subset=["cust"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["amount"] = d.qty * d.price
    o = d.groupby(["cust", "invoice"]).agg(date=("date", "min"), amount=("amount", "sum")).reset_index()
    o["cust"] = o.cust.astype(int).astype(str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    o[["cust", "date", "amount"]].to_csv(a.out, index=False)
    print(f"{n0:,} line items -> {len(o):,} orders | {o.cust.nunique():,} customers | "
          f"{o.date.min().date()} to {o.date.max().date()} | median order GBP {o.amount.median():.0f}")


if __name__ == "__main__":
    main()
