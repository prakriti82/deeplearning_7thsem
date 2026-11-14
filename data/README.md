# Data (not committed to Git)

Place the downloaded **Grid-India / NERLDC hourly demand, solar and wind** Excel files
(Mendeley Data, DOI 10.17632/y58jknpgs8 - use the latest version, check the licence)
inside `data/raw/` (sub-folders are fine).

Then:
1. `python scripts/prepare_dataset.py --inspect`  -> shows sheets and columns.
2. Fill `COLUMN_OVERRIDES` / `REGION_*` in `src/config.py` if auto-detection says it is ambiguous.
3. `python scripts/prepare_dataset.py` -> writes `data/processed/dataset.csv`, `splits.json`, `data_report.json`.

M1 documents here: file list, version, licence, sheet layout, units, conversions, known problems.
M2 documents here: weather locations, weather mode (actual vs forecast), limitations.
