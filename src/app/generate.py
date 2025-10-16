from __future__ import annotations
import os
import datetime
from pathlib import Path
import time
import traceback
import logging

import streamlit as st
import pandas as pd

from suspects import pipeline
import suspects.config as cfg

from app.generate_utils import apply_to_config as _apply_to_config
from app.generate_utils import format_size as _format_size



def render():
    st.sidebar.header("Logs")
    log_placeholder = st.sidebar.empty()

    if "suspect_logs" not in st.session_state:
        st.session_state.suspect_logs = []

    def _push_log(message: str | None, level: str = "INFO"):
        if not message:
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.suspect_logs.append(f"{ts} [{level.upper()}] {message.rstrip()}")
        log_placeholder.code("\n".join(st.session_state.suspect_logs), language="text")

    cache_dir   = getattr(cfg, "cache_dir", os.path.join(cfg.data_dir, "cache"))
    p_ids       = getattr(cfg, "cache_ids",      os.path.join(cache_dir, "ids.parquet"))
    p_pairs     = getattr(cfg, "cache_pairs",    os.path.join(cache_dir, "pairs.parquet"))
    p_clusters  = getattr(cfg, "cache_clusters", os.path.join(cache_dir, "clusters.parquet"))

    with st.container():
        st.subheader("Using Living Data from 2022-03-30")

        IDS_ROWS = 237_994
        PAIRS_ROWS = 17_613_669
        CLUSTERS_ROWS = 10_113_895

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("IDs (library annotations)")
            st.write(f"rows: {IDS_ROWS:,}")
            st.write(f"size: {_format_size(p_ids)}")

        with c2:
            st.caption("Pairs (cosine edges)")
            st.write(f"rows: {PAIRS_ROWS:,}")
            st.write(f"size: {_format_size(p_pairs)}")

        with c3:
            st.caption("Clusters (spectral nodes)")
            st.write(f"rows: {CLUSTERS_ROWS:,}")
            st.write(f"size: {_format_size(p_clusters)}")

    st.markdown("---")

    with st.form("compose_form", clear_on_submit=False):
        st.subheader("Upload Extra IDs (optional)")
        uploaded_annotations_file = st.file_uploader(
            "Additional spectrum annotations",
            type=["tsv"],
            help="If provided, these IDs are merged and the outputs can be restricted to their LibraryUsi.",
            key="annotations_file",
        )


        st.subheader("Parameters")
        cache_min_cosine = getattr(cfg, "cache_min_cosine", float(getattr(cfg, "min_cosine", 0.8)))
        baked_ppm = getattr(cfg, "cache_max_ppm", getattr(cfg, "max_ppm", 20.0))
        baked_shared = getattr(cfg, "cache_min_shared_peaks", getattr(cfg, "min_shared_peaks", 6))
        st.caption(
            f"Preapplied filters: ppm ≤ {baked_ppm}, shared peaks ≥ {baked_shared}, "
            f"min cosine (cache) = {cache_min_cosine}"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            min_cosine = st.number_input(
                "min_cosine",
                value=max(float(cfg.min_cosine), cache_min_cosine),
                min_value=cache_min_cosine,
                max_value=1.0,
                step=0.01,
                help="You can only raise the threshold relative to the cache.",
                key="min_cosine",
            )
            min_delta_mz = st.number_input("min_delta_mz", value=float(cfg.min_delta_mz), step=0.05, key="min_delta_mz")
        with col2:
            interval_width = st.number_input("interval_width", value=float(cfg.interval_width), step=0.1, key="interval_width")
            bin_width = st.number_input("bin_width", value=float(cfg.bin_width), step=0.001, format="%.3f", key="bin_width")
        with col3:
            peak_height = st.number_input("peak_height", value=float(cfg.peak_height), step=1.0, key="peak_height")
            max_distance = st.number_input("max_dist", value=float(cfg.max_dist), step=0.001, format="%.3f", key="max_dist")

        st.markdown("---")
        compose_pressed = st.form_submit_button("Compose suspects")

    if compose_pressed:
        st.session_state.suspect_logs = []
        _push_log("Starting suspects composition…")

        missing = [p for p in (p_ids, p_pairs, p_clusters) if not Path(p).exists()]
        if missing:
            st.error("Backbone cache missing:\n" + "\n".join(f"• {m}" for m in missing))
            st.stop()

        # Apply params to cfg only now
        _apply_to_config(
            min_cosine=min_cosine,
            min_delta_mz=min_delta_mz,
            interval_width=interval_width,
            bin_width=bin_width,
            peak_height=peak_height,
            max_distance=max_distance,
            uploaded_annotations_file=uploaded_annotations_file,
        )

        def report_callback(message: str | None = None, frac: float | None = None):
            if message:
                _push_log(message, level="info")

        annotations_dataframe = None
        if uploaded_annotations_file is not None:
            try:
                with st.spinner("Reading annotations TSV…"):
                    annotations_dataframe = pd.read_csv(uploaded_annotations_file, sep="\t")
                    _push_log(f"Loaded annotations: {len(annotations_dataframe):,} rows")
            except Exception as e:
                _push_log(f"Failed to read uploaded annotations: {e}", level="error")
                st.error("Could not read the uploaded TSV. Check the file and try again.")
                st.stop()

        try:
            with st.spinner("Composing from cache…"):
                time.sleep(0.3)
                final_message = pipeline.generate_suspects(
                    annotations_dataframe, report_cb=report_callback
                )
            st.success(final_message)
            _push_log("Done.")
        except Exception as e:
            _push_log(f"ERROR: {e}", level="error")
            st.error("Suspects composition failed — check logs in the sidebar.")
            st.exception(e)
