"""Read the Grid-India monthly Excel files. Owner: M1.
Never invents data: if columns cannot be identified it stops with a clear message."""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import config

# Prefer exact SCADA display names (with '|P' stripped); fallback to substring matching
PREFERRED_TAGS = {
    "demand_mw": ["NLDC_DEMAND"],
    "solar_mw": ["ALL_IND_SOLAR"],
    "wind_mw": ["ALL_INDIA_WIND"],
}
KEYS = {"demand_mw": ["demand"], "solar_mw": ["solar"], "wind_mw": ["wind"]}
TS_KEYS = ["time", "timestamp", "datetime", "date_time", "date time", "date"]


def list_excel_files(raw_dir: Path = config.RAW_DIR) -> List[Path]:
    files = [
        p for p in Path(raw_dir).rglob("*")
        if p.suffix.lower() in (".xlsx", ".xls") and not p.name.startswith("~$")
    ]
    return sorted(files)


def inspect_excel_files(raw_dir: Path = config.RAW_DIR, max_files: int = 3) -> None:
    """Print sheets, columns and first rows so you can fill config.COLUMN_OVERRIDES."""
    files = list_excel_files(raw_dir)
    if not files:
        print(f"No Excel files found in {raw_dir}. Put the downloaded files there.")
        return
    print(f"{len(files)} Excel files found. Showing the first {min(max_files, len(files))}:")
    for f in files[:max_files]:
        xl = pd.ExcelFile(f)
        print(f"\n=== {f.name}  sheets={xl.sheet_names}")
        for s in xl.sheet_names:
            df = xl.parse(s, nrows=5)
            print(f"--- sheet '{s}' columns: {list(df.columns)}")
            print(df.head(3).to_string())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Promote row 0 to header if it holds SCADA display names (e.g. NLDC_DEMAND|P)."""
    if len(df) > 0 and pd.isna(df.iloc[0, 0]):
        first_row = [str(v) for v in df.iloc[0].values]
        if any("|P" in v or "DEMAND" in v.upper() for v in first_row):
            new_cols = [
                str(df.columns[i]) if pd.isna(val) else str(val).replace("|P", "").strip()
                for i, val in enumerate(df.iloc[0].values)
            ]
            df = df.iloc[1:].copy()
            df.columns = new_cols
            return df.reset_index(drop=True)

    df = df.copy()
    df.columns = [str(c).replace("|P", "").strip() for c in df.columns]
    return df


def _candidates(columns, keys) -> List[str]:
    out = []
    for c in columns:
        lc = str(c).lower()
        if any(k in lc for k in keys):
            if getattr(config, "REGION_KEYWORD", None) and config.REGION_KEYWORD.lower() not in lc:
                continue
            out.append(c)
    return out


def _pick(columns, name: str, keys, required: bool) -> Optional[str]:
    override = getattr(config, "COLUMN_OVERRIDES", {}).get(name)
    if override:
        clean_override = override.replace("|P", "").strip()
        if clean_override in columns:
            return clean_override
        if override in columns:
            return override
        raise ValueError(f"COLUMN_OVERRIDES['{name}']='{override}' not found. Columns: {list(columns)}")

    # 1. Check exact match against known Grid-India SCADA display names first
    for tag in PREFERRED_TAGS.get(name, []):
        for c in columns:
            if str(c).strip().upper() == tag:
                return c

    # 2. Fallback to substring matching
    cands = _candidates(columns, keys)
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        if required:
            raise ValueError(
                f"Could not find a '{name}' column. Columns: {list(columns)}. "
                f"Set config.COLUMN_OVERRIDES['{name}']."
            )
        return None
    raise ValueError(
        f"Ambiguous '{name}' columns {cands}. Set config.REGION_KEYWORD or "
        f"config.COLUMN_OVERRIDES['{name}']."
    )


def _timestamp(df: pd.DataFrame) -> pd.Series:
    override = getattr(config, "COLUMN_OVERRIDES", {}).get("timestamp")
    if override:
        return pd.to_datetime(df[override], errors="coerce")
    lower = {str(c).lower().strip(): c for c in df.columns}
    for k in TS_KEYS:
        for lc, c in lower.items():
            if lc == k or (k in lc and "update" not in lc):
                if k == "date" and any("time" in x and x != lc for x in lower):
                    time_col = [lower[x] for x in lower if "time" in x and x != lc][0]
                    return pd.to_datetime(df[c].astype(str) + " " + df[time_col].astype(str), errors="coerce")
                return pd.to_datetime(df[c], errors="coerce")
    raise ValueError(
        f"No timestamp column found. Columns: {list(df.columns)}. "
        f"Set config.COLUMN_OVERRIDES['timestamp']."
    )


def _extract(df: pd.DataFrame, resample_freq: Optional[str] = "15min") -> pd.DataFrame:
    df = _normalize_columns(df)

    if getattr(config, "REGION_COLUMN", None) and getattr(config, "REGION_VALUE", None):
        df = df[df[config.REGION_COLUMN].astype(str).str.lower() == str(config.REGION_VALUE).lower()]

    cols = list(df.columns)
    out = pd.DataFrame({"timestamp": _timestamp(df)})

    for name, keys in KEYS.items():
        c = _pick(cols, name, keys, required=name in ("demand_mw", "solar_mw"))
        if c is not None:
            out[name] = pd.to_numeric(df[c], errors="coerce").values

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Clip nighttime SCADA sensor drift on solar (< 0 MW -> 0 MW)
    if "solar_mw" in out.columns:
        out["solar_mw"] = out["solar_mw"].clip(lower=0.0)

    # Standardize mixed sampling frequencies (10s in 2023 vs 5min in 2022)
    if resample_freq:
        out = (
            out.drop_duplicates(subset=["timestamp"])
            .set_index("timestamp")
            .resample(resample_freq)
            .mean()
            .reset_index()
        )

    return out


def load_grid_data(raw_dir: Path = config.RAW_DIR, resample_freq: Optional[str] = "15min") -> pd.DataFrame:
    """Return merged data with columns timestamp, demand_mw, solar_mw, (wind_mw), source."""
    files = list_excel_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No Excel files in {raw_dir}. Add the dataset first (see data/README.md)."
        )
    frames, skipped = [], []
    for f in files:
        xl = pd.ExcelFile(f)
        if getattr(config, "SHEET_NAME", None):
            sheets = [config.SHEET_NAME]
        else:
            # Skip OSIsoft metadata and scratchpad sheets when Sheet1 is present
            sheets = ["Sheet1"] if "Sheet1" in xl.sheet_names else [
                s for s in xl.sheet_names if not s.startswith("_")
            ]

        for s in sheets:
            try:
                frames.append(_extract(xl.parse(s), resample_freq=resample_freq).assign(source=f.name))
            except ValueError as e:
                skipped.append((f.name, s, str(e)[:120]))

    if not frames:
        raise ValueError(
            "No sheet could be parsed. Run --inspect and set config.COLUMN_OVERRIDES. "
            f"First errors: {skipped[:3]}"
        )
    print(f"Parsed {len(frames)} sheets from {len(files)} files; skipped {len(skipped)}.")
    for item in skipped[:10]:
        print("  skipped:", item)
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)