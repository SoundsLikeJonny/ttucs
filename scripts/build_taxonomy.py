"""Convert the official UCS "Full List" spreadsheet into the CSV shipped as package data.

Usage:
    python scripts/build_taxonomy.py "UCS v8.2.1 Full List.xlsx" \
        "UCS v8.2.1 Top Level Categories.xlsx" --version 8.2.1

The spreadsheets are distributed by the UCS initiative (https://universalcategorysystem.com/).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "src" / "ttucs" / "resources"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("full_list", type=Path)
    ap.add_argument("top_level", type=Path, nargs="?")
    ap.add_argument("--version", default="8.2.1")
    args = ap.parse_args()

    df = pd.read_excel(args.full_list, header=2).iloc[:, :6]
    df.columns = ["Category", "SubCategory", "CatID", "CatShort", "Explanations", "Synonyms"]
    df = df.dropna(subset=["CatID"])
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    assert not df.CatID.duplicated().any(), "duplicate CatIDs in source list"
    out = OUT_DIR / f"ucs_v{args.version}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} CatIDs / {df.Category.nunique()} categories -> {out}")

    if args.top_level:
        top = pd.read_excel(args.top_level, header=1).iloc[:, :2]
        top.columns = ["Category", "Explanation"]
        top = top.dropna(subset=["Category"])
        top_out = OUT_DIR / f"ucs_v{args.version}_top_level.csv"
        top.to_csv(top_out, index=False)
        print(f"wrote {len(top)} top-level categories -> {top_out}")


if __name__ == "__main__":
    main()
