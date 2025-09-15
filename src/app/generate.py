from __future__ import annotations
import io
import logging
import os
from pathlib import Path
from io import StringIO
import time
import traceback

import streamlit as st
import pandas as pd

from suspects import pipeline
import suspects.config as cfg
import datetime

def render() -> None:
    """Render the Streamlit page for configuring paths, tuning filters, and launching suspect generation."""
    st.subheader("Data Paths")

    st.sidebar.header("Logs")
    log_placeholder = st.sidebar.empty()

    column_left, column_right = st.columns(2)
    with column_left:
        data_dir = cfg.data_dir
        living_data_dir = st.text_input("GNPS living data", value=str(cfg.living_data_dir))
        global_network_dir = st.text_input("Global Molecular Network", value=str(cfg.global_network_dir))
    with column_right:
        unimod_file = cfg.unimod_file
        global_network_task_id = cfg.global_network_task_id
        uploaded_annotations_file = st.file_uploader("Additional Spectrum Annotations (optional)", type=["tsv"])
        annotations_dataframe = pd.read_csv(uploaded_annotations_file, sep="\t") if uploaded_annotations_file else pd.DataFrame()

    mass_shift_annotation_url = cfg.mass_shift_annotation_url if cfg.mass_shift_annotation_url else ""

    st.subheader("Filters")
    column1, column2, column3 = st.columns(3)
    with column1:
        max_ppm = st.number_input("max_ppm", value=float(cfg.max_ppm), step=1.0)
        min_shared_peaks = st.number_input("min_shared_peaks", value=int(cfg.min_shared_peaks), step=1)
        min_cosine = st.number_input("min_cosine", value=float(cfg.min_cosine), step=0.05, min_value=0.0, max_value=1.0)
    with column2:
        min_delta_mz = st.number_input("min_delta_mz", value=float(cfg.min_delta_mz), step=0.05)
        interval_width = st.number_input("interval_width", value=float(cfg.interval_width), step=0.1)
    with column3:
        bin_width = st.number_input("bin_width", value=float(cfg.bin_width), step=0.001, format="%.3f")
        peak_height = st.number_input("peak_height", value=float(cfg.peak_height), step=1.0)
        max_distance = st.number_input("max_dist", value=float(cfg.max_dist), step=0.001, format="%.3f")

    st.markdown("---")

    def _apply_to_config():
        """Persist current UI selections back into the suspects configuration module."""
        cfg.data_dir = os.path.realpath(data_dir)
        cfg.living_data_dir = living_data_dir
        cfg.global_network_dir = global_network_dir
        cfg.global_network_task_id = global_network_task_id
        cfg.unimod_file = unimod_file
        cfg.filename_ids = uploaded_annotations_file if uploaded_annotations_file else None
        cfg.mass_shift_annotation_url = mass_shift_annotation_url if mass_shift_annotation_url.strip() else None

        cfg.max_ppm = float(max_ppm)
        cfg.min_shared_peaks = int(min_shared_peaks)
        cfg.min_cosine = float(min_cosine)

        cfg.min_delta_mz = float(min_delta_mz)
        cfg.interval_width = float(interval_width)
        cfg.bin_width = float(bin_width)
        cfg.peak_height = float(peak_height)
        cfg.max_dist = float(max_distance)

    generate_button_pressed = st.button("Generate suspects")

    if generate_button_pressed:
        _apply_to_config()
        problems = []

        if not Path(cfg.global_network_dir).exists():
            problems.append(f"global_network_dir not found: {cfg.global_network_dir}")

        if problems:
            st.error(problems)
            st.stop()

        logs = []

        def _push_log(message: str | None, level: str = "INFO"):
            """Append a log line to the sidebar log display."""
            if not message:
                return
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            formatted = f"{timestamp} [{level.upper()}] {message.rstrip()}"
            logs.append(formatted)
            log_placeholder.code("\n".join(logs), language="text")

        try:
            with st.spinner("Generating suspects…"):
                time.sleep(2)
                def report_callback(message: str | None = None, frac: float | None = None):
                    """Pipeline progress callback: streams messages into the sidebar logs."""
                    _push_log(message, level="info")

                final_message = pipeline.generate_suspects(annotations_dataframe, report_cb=report_callback)

            st.success(final_message)

        except Exception as e:
            _push_log(f"ERROR: {e}", level="error")
            st.error("Suspects generation failed — check logs in the sidebar.")
            st.exception(e)
