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


def format_size(path: str | os.PathLike) -> str:
    try:
        sz = Path(path).stat().st_size
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if sz < 1024:
                return f"{sz:.0f} {unit}"
            sz /= 1024
    except Exception:
        pass
    return "?"

def apply_to_config(
    min_cosine: float,
    min_delta_mz: float,
    interval_width: float,
    bin_width: float,
    peak_height: float,
    max_distance: float,
    uploaded_annotations_file,
):
    """
    Persist only the parameters that the cached path honors.
    Avoid touching global paths; this tab composes strictly 'from cache'.
    """
    cfg.filename_ids = uploaded_annotations_file if uploaded_annotations_file else None
    cfg.min_cosine = float(min_cosine)
    cfg.min_delta_mz = float(min_delta_mz)
    cfg.interval_width = float(interval_width)
    cfg.bin_width = float(bin_width)
    cfg.peak_height = float(peak_height)
    cfg.max_dist = float(max_distance)