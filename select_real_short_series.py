from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "public_benchmarks"
RAW_DIR = DATA_DIR / "raw"
SELECTED_DIR = DATA_DIR / "selected"
OUT_DIR = ROOT / "outputs" / "real_short_series"

SEED = 2026
MIN_LEN = 40
MAX_LEN = 180

URLS = {
    "M4_Yearly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Yearly-train.csv",
    "M4_Quarterly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Quarterly-train.csv",
    "M4_Monthly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv",
    "M3_Yearly": "https://raw.githubusercontent.com/forvis/forvis.github.io/master/data/M3_yearly_TSTS.csv",
    "M3_Quarterly": "https://raw.githubusercontent.com/forvis/forvis.github.io/master/data/M3_quarterly_TSTS.csv",
    "M3_Monthly": "https://raw.githubusercontent.com/forvis/forvis.github.io/master/data/M3_monthly_TSTS.csv",
}


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally committing downloaded benchmark data.
    (DATA_DIR / ".gitignore").write_text(
        "raw/\nselected/\n*.csv\n",
        encoding="utf-8"
    )


def download(name):
    path = RAW_DIR / f"{name}.csv"

    if path.exists() and path.stat().st_size > 1024:
        print(f"[INFO] Using existing file: {path}")
        return path

    if path.exists():
        path.unlink()

    url = URLS[name]
    print(f"[INFO] Downloading {name} from {url}")

    try:
        urllib.request.urlretrieve(url, path)
    except Exception:
        if path.exists():
            path.unlink()
        raise

    if not path.exists() or path.stat().st_size <= 1024:
        raise RuntimeError(f"Downloaded file is missing or too small: {path}")

    print(f"[INFO] Downloaded {name}: {path.stat().st_size} bytes")
    return path


def valid_values(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < MIN_LEN or len(values) > MAX_LEN:
        return None
    if not np.all(np.isfinite(values)):
        return None
    return values


def select_from_candidates(candidates, label):
    if not candidates:
        raise RuntimeError(f"No valid candidates found for {label}")

    candidates = sorted(candidates, key=lambda x: x[0])
    rng = np.random.default_rng(SEED)
    idx = int(rng.integers(0, len(candidates)))
    selected_id, selected_values = candidates[idx]

    print(f"[SELECTED] {label}: {selected_id} length={len(selected_values)}")
    return selected_id, selected_values, len(candidates)


def select_m4(path, frequency):
    df = pd.read_csv(path)
    candidates = []

    for _, row in df.iterrows():
        series_id = str(row.iloc[0]).strip()
        values = valid_values(row.iloc[1:].values)
        if values is not None:
            candidates.append((series_id, values))

    print(f"[INFO] M4 {frequency} candidates: {len(candidates)}")
    return select_from_candidates(candidates, f"M4 {frequency}")


def select_m3(path, frequency):
    # M3 TSTS files used here have no header:
    # series_id, category, value, timestamp
    df = pd.read_csv(
        path,
        header=None,
        names=["series_id", "category", "value", "timestamp"]
    )

    required = {"series_id", "value", "timestamp"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"M3 {frequency} columns not recognized: {list(df.columns)}")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    candidates = []

    for series_id, group in df.groupby("series_id"):
        group = group.sort_values("timestamp")
        values = valid_values(group["value"].values)
        if values is not None:
            candidates.append((str(series_id).strip(), values))

    print(f"[INFO] M3 {frequency} candidates: {len(candidates)}")
    return select_from_candidates(candidates, f"M3 {frequency}")


def save_selected(source, frequency, selected_id, values):
    safe_id = str(selected_id).replace("/", "_").replace("\\", "_")
    file_name = f"{source}_{frequency}_{safe_id}.csv"
    out_path = SELECTED_DIR / file_name

    pd.DataFrame({
        "index": range(1, len(values) + 1),
        "value": values
    }).to_csv(out_path, index=False)

    return out_path.relative_to(ROOT).as_posix()


def main():
    ensure_dirs()

    metadata = []

    yemen_path = ROOT / "data" / "tax_revenues.csv"
    if not yemen_path.exists():
        raise FileNotFoundError(f"Missing private Yemen dataset: {yemen_path}")

    yemen_df = pd.read_csv(yemen_path)
    possible_value_cols = ["revenue_million_yer", "value", "revenue", "tax_revenue"]

    value_col = None
    for col in possible_value_cols:
        if col in yemen_df.columns:
            value_col = col
            break

    if value_col is None:
        raise RuntimeError(f"Could not detect value column in tax_revenues.csv. Columns: {list(yemen_df.columns)}")

    yemen_len = int(yemen_df[value_col].dropna().shape[0])
    print(f"[INFO] Yemen tax revenue length: {yemen_len}")

    metadata.append({
        "series_id": "yemen_tax_revenue",
        "source": "local_private",
        "frequency": "Monthly",
        "length": yemen_len,
        "selection_seed": "NA",
        "candidate_count": "NA",
        "file_path": "data/tax_revenues.csv"
    })

    selections = [
        ("M4_Yearly", "M4", "Yearly", select_m4),
        ("M4_Quarterly", "M4", "Quarterly", select_m4),
        ("M4_Monthly", "M4", "Monthly", select_m4),
        ("M3_Yearly", "M3", "Yearly", select_m3),
        ("M3_Quarterly", "M3", "Quarterly", select_m3),
    ]

    for key, source, frequency, selector in selections:
        raw_path = download(key)
        selected_id, values, candidate_count = selector(raw_path, frequency)
        selected_path = save_selected(source, frequency, selected_id, values)

        metadata.append({
            "series_id": f"{source}_{frequency}_{selected_id}",
            "source": source,
            "frequency": frequency,
            "length": len(values),
            "selection_seed": SEED,
            "candidate_count": candidate_count,
            "file_path": selected_path
        })

    if len(metadata) != 6:
        raise RuntimeError(f"Expected 6 selected series including Yemen, got {len(metadata)}")

    out_csv = OUT_DIR / "selected_real_series.csv"
    pd.DataFrame(metadata).to_csv(out_csv, index=False)

    print(f"[DONE] Wrote {out_csv}")
    print(pd.DataFrame(metadata))


if __name__ == "__main__":
    main()