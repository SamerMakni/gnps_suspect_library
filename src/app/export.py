from __future__ import annotations
import os, io, re, json, math, time, hashlib, urllib.parse, datetime as dt
from pathlib import Path
from typing import Optional, Dict, Iterable, Tuple

import streamlit as st
import pandas as pd
from pyteomics import mgf
import requests
import spectrum_utils.spectrum as sus

import suspects.config as cfg 

from app.export_utils import (
    timestamp as _timestamp,
    normalize_index_cols as _normalize_index_cols,
    resolve_index_path as _resolve_index_path,
    load_cached_spec as _load_cached_spec,
    to_mgf_dict as _to_mgf_dict,
    save_spec_json as _save_spec_json,
    query_msv_spectrum as _query_msv_spectrum,
)


DATA_DIR       = Path(getattr(cfg, "data_dir", "data"))
INTERIM_DIR    = DATA_DIR / "interim"
PROCESSED_DIR  = DATA_DIR / "processed"
CACHE_DIR      = Path(getattr(cfg, "spectra_cache_dir", DATA_DIR / "spectra_cache"))
INDEX_PARQUET  = CACHE_DIR / "usi_index.parquet"
UNIQUE_PARQUET = INTERIM_DIR / "suspects_CACHE_unique.parquet"

TEST_LIMIT = 20 

def render() -> None:
    st.header("Export MGF")

    if not UNIQUE_PARQUET.exists():
        st.error("Unique parquet not found. Compose suspects first.")
        return
    if not INDEX_PARQUET.exists():
        st.warning("Index parquet not found. You can still export (cache-only paths will be resolved), but rebuilding the index is recommended.")


    filters_cols = st.columns(2)
    with filters_cols[0]:
        tol_da  = st.number_input("Abs tolerance (Da)", value=0.02, step=0.01, format="%.2f")
    with filters_cols[1]:
        tol_ppm = st.number_input("Rel tolerance (ppm)", value=20.0, step=1.0, format="%.0f")
    input_cols = st.columns([0.42, 0.42, 0.16])
    with input_cols[0]:
        fetch_online = st.checkbox("Fetch online if spectrum is not available", value=False)
    with input_cols[1]:
        enforce = st.checkbox("Enforce precursor m/z match", value=True)
    with input_cols[2]:
        run = st.button(f"Export mgf")
    if not run:
        return

    with st.spinner("Loading suspects & cache index…"):
        suspects = pd.read_parquet(UNIQUE_PARQUET)
        if "SuspectUsi" not in suspects.columns:
            st.error("Unique parquet must contain 'SuspectUsi'.")
            return
        suspects = suspects.dropna(subset=["SuspectUsi"])

        if INDEX_PARQUET.exists():
            idx_raw = pd.read_parquet(INDEX_PARQUET)
            idx = _normalize_index_cols(idx_raw)

            sus = suspects.merge(idx[["usi", "path"]], left_on="SuspectUsi", right_on="usi", how="left")
        else:
            sus = suspects.copy()
            sus["path"] = pd.NA

        total = len(sus)
        if total == 0:
            st.warning("No suspects found in unique parquet.")
            return

        # Testing limit
        if TEST_LIMIT is not None:
            sus = sus.head(TEST_LIMIT).copy()
            total = len(sus)

    out_path = PROCESSED_DIR / f"export_{_timestamp()}.mgf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # just some progress indication
    prog = st.progress(0.0)
    status = st.empty()
    start = time.time()

    written = 0
    cache_hits = 0
    fetched = 0
    misses = 0
    dm_fail = 0
    empty_peaks = 0

    def _within_tol(spec_mz: float | None, pepmass: float, tol_da: float, tol_ppm: float) -> bool:
        # accept when either value is missing/zero (can’t validate)
        if not spec_mz or spec_mz <= 0 or not pepmass or pepmass <= 0:
            return True
        dm = abs(spec_mz - pepmass)
        return (dm <= tol_da) or (dm <= pepmass * tol_ppm * 1e-6)

    # Main loop, might add threading later if needed
    with open(out_path, "w") as f_out:
        for i, (_, row) in enumerate(sus.iterrows(), 1):
            usi = str(row["SuspectUsi"])
            adduct = row.get("Adduct", "")
            pepmass = float(row.get("SuspectPrecursorMass", 0.0))

            spec = None

            # Try cache JSON if present in index
            if pd.notna(row.get("path", pd.NA)):
                p = _resolve_index_path(str(row["path"]))
                if p.exists():
                    spec = _load_cached_spec(p, usi)
                    if spec is not None:
                        cache_hits += 1

            # If still none, optionally fetch online and save immediately to cache sharded path
            if spec is None and fetch_online:
                spec = _query_msv_spectrum(usi)
                if spec is not None and spec.mz is not None and len(spec.mz) > 0:
                    fetched += 1
                    try:
                        _save_spec_json(usi, spec)  # persist for future runs
                    except Exception:
                        pass

            if spec is None:
                misses += 1
            else:
                if spec.mz is None or len(spec.mz) == 0:
                    empty_peaks += 1
                elif enforce and not _within_tol(spec.precursor_mz, pepmass, tol_da, tol_ppm):
                    dm_fail += 1
                else:
                    mgf.write([_to_mgf_dict(spec, pepmass, adduct=adduct)], f_out)
                    written += 1

            frac = i / total
            prog.progress(frac)
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = (total - i) / rate if rate > 0 else 0.0
            status.caption(
                f"{i}/{total} processed — written {written} | cache {cache_hits} | online {fetched} Δm fail {dm_fail} | empty {empty_peaks} — ETA ~ {remaining:,.1f}s"
            )

    if written == 0:
        st.error("No spectra were written. Check that cache paths resolve, or enable online fetching.")
    else:
        st.success(f"Export complete: wrote {written:,} spectra to {out_path.name}")

    try:
        with open(out_path, "rb") as f:
            data = f.read()
        st.download_button(
            label=f"Download {out_path.name}",
            data=data,
            file_name=out_path.name,
            mime="application/octet-stream",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(f"File saved at: {out_path} (couldn’t stage for download: {e})")