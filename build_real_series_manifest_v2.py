#!/usr/bin/env python3
"""
Build the V2 real-series development/final manifest WITHOUT fitting or forecasting any model.

Sources are loaded through datasetsforecast==1.0.1:
- M3 Yearly / Quarterly / Monthly
- M4 Yearly / Quarterly / Monthly

The script performs only:
1) source download/load,
2) model-independent eligibility screening,
3) deterministic SHA-256 ranking,
4) legacy-series exclusion,
5) source and series fingerprinting,
6) writing the frozen ID manifest.

No PC-NFPSO or baseline forecast is computed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from datasetsforecast.m3 import M3
from datasetsforecast.m4 import M4


PROTOCOL_VERSION = "2.0"
TRAIN_FRACTION = 0.80
MIN_LENGTH = 40
MAX_LENGTH = 180
MIN_TEST = 8
MIN_SUPERVISED_TRAIN_ROWS = 20
MIN_MASE_SCALE = 1e-12
DEV_PER_STRATUM = 10
FINAL_PER_STRATUM = 20

ROOT = Path("data") / "v2_real_manifest"
SOURCE_ROOT = ROOT / "sources"
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)

OUT_JSON = Path("configs") / "v2" / "real_series_manifest_v2.json"
OUT_CSV = ROOT / "real_series_manifest_v2.csv"
OUT_COUNTS = ROOT / "eligible_counts_v2.json"
OUT_SOURCE_HASHES = ROOT / "source_sha256_v2.json"

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

STRATA = [
    ("M3", "Yearly", 1),
    ("M3", "Quarterly", 4),
    ("M3", "Monthly", 12),
    ("M4", "Yearly", 1),
    ("M4", "Quarterly", 4),
    ("M4", "Monthly", 12),
]

LEGACY_EXCLUSIONS = {
    ("M4", "Yearly", "Y6688"),
    ("M4", "Quarterly", "Q6355"),
    ("M4", "Monthly", "M6356"),
    ("M3", "Yearly", "Y404"),
    ("M3", "Quarterly", "Q687"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def series_sha256(y: np.ndarray) -> str:
    # Explicit little-endian float64 representation for stable byte-level fingerprinting.
    arr = np.asarray(y, dtype="<f8")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def lag_for_period(m: int) -> int:
    return min(12, max(5, int(m)))


def mase_scale(train: np.ndarray, m: int) -> float:
    m = int(m)
    if len(train) <= m:
        return float("nan")
    return float(np.mean(np.abs(train[m:] - train[:-m])))


def eligibility(y: np.ndarray, m: int) -> tuple[bool, dict]:
    y = np.asarray(y, dtype=float)
    n = int(len(y))
    L = lag_for_period(m)
    split = int(math.floor(TRAIN_FRACTION * n))
    n_test = n - split
    n_supervised_train = split - L

    meta = {
        "n": n,
        "split_index": split,
        "n_train": split,
        "n_test": n_test,
        "seasonal_period": int(m),
        "lag_L": L,
        "n_supervised_train": n_supervised_train,
    }

    if n < MIN_LENGTH or n > MAX_LENGTH:
        meta["eligible_reason"] = "length"
        return False, meta
    if not np.all(np.isfinite(y)):
        meta["eligible_reason"] = "nonfinite"
        return False, meta
    if n_test < MIN_TEST:
        meta["eligible_reason"] = "test_too_short"
        return False, meta
    if n_supervised_train < MIN_SUPERVISED_TRAIN_ROWS:
        meta["eligible_reason"] = "supervised_train_too_short"
        return False, meta

    scale = mase_scale(y[:split], m)
    meta["mase_scale_train_only"] = scale
    if not np.isfinite(scale) or scale <= MIN_MASE_SCALE:
        meta["eligible_reason"] = "mase_scale"
        return False, meta

    meta["eligible_reason"] = "ok"
    return True, meta


def ranking_digest(source: str, frequency: str, series_id: str) -> str:
    key = f"pc-nfpso-v2|{source}|{frequency}|{series_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_group(source: str, group: str) -> pd.DataFrame:
    if source == "M3":
        df, *_ = M3.load(directory=str(SOURCE_ROOT), group=group)
    elif source == "M4":
        df, *_ = M4.load(directory=str(SOURCE_ROOT), group=group, cache=False)
    else:
        raise ValueError(source)

    required = {"unique_id", "y"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{source}/{group}: missing columns {sorted(missing)}")

    out = df[["unique_id", "y"]].copy()
    out["unique_id"] = out["unique_id"].astype(str)
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out


rows = []
counts = {}

for source, frequency, m in STRATA:
    print(f"[LOAD] {source}/{frequency}")
    df = load_group(source, frequency)
    eligible_rows = []

    for series_id, g in df.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(dtype=float)
        ok, meta = eligibility(y, m)

        if (source, frequency, series_id) in LEGACY_EXCLUSIONS:
            continue
        if not ok:
            continue

        eligible_rows.append({
            "source": source,
            "frequency": frequency,
            "series_id": series_id,
            "rank_sha256": ranking_digest(source, frequency, series_id),
            "series_sha256_float64_le": series_sha256(y),
            **meta,
        })

    eligible_rows.sort(key=lambda r: (r["rank_sha256"], r["series_id"]))
    needed = DEV_PER_STRATUM + FINAL_PER_STRATUM
    if len(eligible_rows) < needed:
        raise RuntimeError(
            f"{source}/{frequency}: only {len(eligible_rows)} eligible series; "
            f"{needed} required."
        )

    selected = eligible_rows[:needed]
    for i, r in enumerate(selected):
        r["partition"] = "development" if i < DEV_PER_STRATUM else "final_locked"
        r["selection_index_within_stratum"] = i + 1
        rows.append(r)

    counts[f"{source}_{frequency}"] = {
        "eligible_after_legacy_exclusion": len(eligible_rows),
        "development_selected": DEV_PER_STRATUM,
        "final_selected": FINAL_PER_STRATUM,
    }
    print(
        f"[OK] {source}/{frequency}: eligible={len(eligible_rows)}, "
        f"dev={DEV_PER_STRATUM}, final={FINAL_PER_STRATUM}"
    )

manifest_df = pd.DataFrame(rows)
if len(manifest_df) != 180:
    raise RuntimeError(f"Expected 180 selected series, got {len(manifest_df)}")

if manifest_df.duplicated(["source", "frequency", "series_id"]).any():
    raise RuntimeError("Duplicate selected series IDs detected.")

# Verify exact partition totals and stratum balance.
part_counts = manifest_df["partition"].value_counts().to_dict()
if part_counts.get("development") != 60 or part_counts.get("final_locked") != 120:
    raise RuntimeError(f"Unexpected partition totals: {part_counts}")

for source, frequency, _ in STRATA:
    q = manifest_df[
        (manifest_df["source"] == source)
        & (manifest_df["frequency"] == frequency)
    ]
    pc = q["partition"].value_counts().to_dict()
    if pc.get("development") != 10 or pc.get("final_locked") != 20:
        raise RuntimeError(f"Unexpected stratum partition: {source}/{frequency}: {pc}")

# Hash downloaded raw/source artifacts, excluding derived pickle caches.
source_files = []
for p in sorted(SOURCE_ROOT.rglob("*")):
    if not p.is_file():
        continue
    if p.suffix.lower() in {".p", ".pickle"}:
        continue
    source_files.append({
        "path": p.relative_to(SOURCE_ROOT).as_posix(),
        "size_bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    })

pkg_version = importlib.metadata.version("datasetsforecast")

source_hash_doc = {
    "datasetsforecast_version": pkg_version,
    "source_root": SOURCE_ROOT.as_posix(),
    "files": source_files,
}
OUT_SOURCE_HASHES.write_text(
    json.dumps(source_hash_doc, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

manifest_records = manifest_df.to_dict(orient="records")
manifest = {
    "protocol_version": PROTOCOL_VERSION,
    "status": "LOCKED",
    "selection_method": "deterministic SHA-256 ranking after model-independent eligibility screening",
    "ranking_key_template": "pc-nfpso-v2|source|frequency|series_id",
    "loader": {
        "package": "datasetsforecast",
        "version": pkg_version,
        "note": (
            "M3 and M4 are loaded as the complete series representation returned "
            "by datasetsforecast 1.0.1; V2 then applies its own frozen 80/20 split."
        ),
    },
    "eligibility": {
        "minimum_length": MIN_LENGTH,
        "maximum_length": MAX_LENGTH,
        "train_fraction": TRAIN_FRACTION,
        "minimum_test_observations": MIN_TEST,
        "minimum_supervised_training_rows_after_lagging": MIN_SUPERVISED_TRAIN_ROWS,
        "minimum_mase_scale": MIN_MASE_SCALE,
        "lag_rule": "L=min(12,max(5,m))",
        "mase_scale_source": "pre-test training segment only",
    },
    "legacy_exclusions": [
        {"source": s, "frequency": f, "series_id": i}
        for s, f, i in sorted(LEGACY_EXCLUSIONS)
    ],
    "counts": {
        "development": int((manifest_df["partition"] == "development").sum()),
        "final_locked": int((manifest_df["partition"] == "final_locked").sum()),
        "total": int(len(manifest_df)),
        "by_stratum": counts,
    },
    "source_hash_manifest": OUT_SOURCE_HASHES.as_posix(),
    "series_fingerprint_encoding": "SHA-256 of contiguous little-endian float64 series bytes",
    "selected_series": manifest_records,
    "prohibition": (
        "No forecasting model may be fitted or evaluated on final_locked series "
        "before the complete Gate-2 protocol is committed and tagged."
    ),
}

OUT_JSON.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
manifest_df.to_csv(OUT_CSV, index=False)

OUT_COUNTS.write_text(
    json.dumps(
        {
            "counts": counts,
            "partition_totals": part_counts,
            "total": len(manifest_df),
        },
        indent=2,
    ),
    encoding="utf-8",
)

# Hash the manifest artifacts themselves.
print()
print("MANIFEST JSON =", OUT_JSON)
print("MANIFEST CSV  =", OUT_CSV)
print("SOURCE HASHES =", OUT_SOURCE_HASHES)
print("DATASETSFORECAST VERSION =", pkg_version)
print("TOTAL =", len(manifest_df))
print("DEVELOPMENT =", int((manifest_df["partition"] == "development").sum()))
print("FINAL_LOCKED =", int((manifest_df["partition"] == "final_locked").sum()))
print("SOURCE FILES HASHED =", len(source_files))
print("MANIFEST SHA256 =", sha256_file(OUT_JSON))
print("CSV SHA256 =", sha256_file(OUT_CSV))
print("VERDICT = REAL MANIFEST BUILT")
