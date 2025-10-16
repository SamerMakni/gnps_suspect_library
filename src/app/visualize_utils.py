from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def default_filters(dataframe: pd.DataFrame) -> Dict[str, Any]:
    search = ""

    min_cos = 0.8 if "Cosine" in dataframe.columns else 0.0

    dm_range = (None, None)
    if "DeltaMass" in dataframe.columns:
        dm_range = numeric_range(dataframe["DeltaMass"])

    gdm_range = (None, None)
    if "GroupDeltaMass" in dataframe.columns and dataframe["GroupDeltaMass"].notna().any():
        gdm_range = numeric_range(dataframe["GroupDeltaMass"].dropna())

    ionmodes = (
        sorted(dataframe["IonMode"].dropna().astype(str).unique())
        if "IonMode" in dataframe.columns
        else []
    )
    instruments = (
        sorted(dataframe["Instrument"].dropna().astype(str).unique())
        if "Instrument" in dataframe.columns
        else []
    )

    return dict(
        search=search,
        min_cos=min_cos,
        dm_range=dm_range,
        gdm_range=gdm_range,
        ionmodes=ionmodes,
        instruments=instruments,
    )


def load_all(interim_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load only grouped & unique parquet files if present. Keys: 'grouped', 'unique'"""
    candidate_files = {
        "grouped": list(interim_dir.glob("suspects_CACHE_grouped.parquet")),
        "unique": list(interim_dir.glob("suspects_CACHE_unique.parquet")),
    }
    dataframes_by_kind: Dict[str, pd.DataFrame] = {}
    for kind_key, path_list in candidate_files.items():
        if path_list:
            latest_path = max(path_list, key=lambda x: x.stat().st_mtime)
            dataframes_by_kind[kind_key] = load_parquet(latest_path)
    return dataframes_by_kind


# ---------- Filter UI & logic ----------

def numeric_range(series: pd.Series) -> Tuple[float, float]:
    if series.dropna().empty:
        return (0.0, 0.0)
    min_value, max_value = float(series.min()), float(series.max())
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return (0.0, 0.0)
    if min_value == max_value:
        pad = max(1e-6, abs(min_value) * 0.05)
        return (min_value - pad, max_value + pad)
    return (min_value, max_value)


def build_filters(dataframe: pd.DataFrame, kind: str, initial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    st.subheader("Filters")
    initial = initial or default_filters(dataframe)

    def _clip_pair(lohi, fallback):
        if lohi is None or None in lohi:
            return fallback
        lo, hi = lohi
        flo, fhi = fallback
        return (max(flo, float(lo)), min(fhi, float(hi)))

    def _sanitize_selected(selected_list, options):
        return [x for x in (selected_list or []) if x in options] or options

    column_search = st.text_input(
        "Search CompoundName (contains)",
        value=str(initial.get("search", "")),
    ).strip()

    min_cosine = 0.0
    if "Cosine" in dataframe.columns:
        default_cosine = 0.8
        min_cosine = st.slider(
            "Minimum cosine",
            0.0,
            1.0,
            float(initial.get("min_cos", default_cosine)),
            step=0.01,
        )

    delta_mass_low, delta_mass_high = None, None
    if "DeltaMass" in dataframe.columns:
        fallback = numeric_range(dataframe["DeltaMass"])
        lo, hi = _clip_pair(initial.get("dm_range"), fallback)
        min_value, max_value = fallback
        delta_mass_low, delta_mass_high = st.slider(
            "Δm (DeltaMass) range",
            min_value=float(min_value),
            max_value=float(max_value),
            value=(float(lo), float(hi)),
            step=(max_value - min_value) / 200 if max_value > min_value else 0.001,
        )

    group_dm_low, group_dm_high = None, None
    if "GroupDeltaMass" in dataframe.columns and dataframe["GroupDeltaMass"].notna().any():
        fallback = numeric_range(dataframe["GroupDeltaMass"].dropna())
        g_lo, g_hi = _clip_pair(initial.get("gdm_range"), fallback)
        min_group, max_group = fallback
        group_dm_low, group_dm_high = st.slider(
            "Grouped Δm (GroupDeltaMass) range",
            min_value=float(min_group),
            max_value=float(max_group),
            value=(float(g_lo), float(g_hi)),
            step=(max_group - min_group) / 200 if max_group > min_group else 0.001,
        )

    ionmode_options = (
        sorted(dataframe["IonMode"].dropna().astype(str).unique())
        if "IonMode" in dataframe.columns
        else []
    )
    selected_ion_modes = st.multiselect(
        "IonMode",
        ionmode_options,
        default=_sanitize_selected(initial.get("ionmodes"), ionmode_options),
    )

    instrument_options = (
        sorted(dataframe["Instrument"].dropna().astype(str).unique())
        if "Instrument" in dataframe.columns
        else []
    )
    selected_instruments = st.multiselect(
        "Instrument",
        instrument_options,
        default=_sanitize_selected(initial.get("instruments"), instrument_options),
    )

    return dict(
        search=column_search,
        min_cos=min_cosine,
        dm_range=(delta_mass_low, delta_mass_high),
        gdm_range=(group_dm_low, group_dm_high),
        ionmodes=selected_ion_modes,
        instruments=selected_instruments,
    )


def apply_filters(dataframe: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    df = dataframe

    s = filters["search"]
    if s and "CompoundName" in df.columns:
        df = df[df["CompoundName"].astype(str).str.contains(s, case=False, na=False)]

    min_cos = filters["min_cos"]
    if "Cosine" in df.columns and min_cos is not None:
        df = df[df["Cosine"] >= float(min_cos)]

    lo, hi = filters["dm_range"]
    if lo is not None and hi is not None and "DeltaMass" in df.columns:
        df = df[df["DeltaMass"].between(lo, hi)]

    g_lo, g_hi = filters["gdm_range"]
    if g_lo is not None and g_hi is not None and "GroupDeltaMass" in df.columns:
        mask = df["GroupDeltaMass"].notna()
        df = df[~mask | df["GroupDeltaMass"].between(g_lo, g_hi)]

    if "IonMode" in df.columns and filters["ionmodes"]:
        df = df[df["IonMode"].astype(str).isin(filters["ionmodes"])]
    if "Instrument" in df.columns and filters["instruments"]:
        df = df[df["Instrument"].astype(str).isin(filters["instruments"])]

    return df


def plot_summary(dataframe: pd.DataFrame):
    st.subheader("Summary")
    num_rows = len(dataframe)
    num_unique_compounds = (
        dataframe["CompoundName"].nunique() if "CompoundName" in dataframe.columns else 0
    )
    mean_cosine = (
        dataframe["Cosine"].mean()
        if "Cosine" in dataframe.columns and not dataframe["Cosine"].empty
        else np.nan
    )
    fraction_grouped = np.nan
    if "GroupDeltaMass" in dataframe.columns:
        non_null = dataframe["GroupDeltaMass"].notna().sum()
        fraction_grouped = (non_null / len(dataframe)) if len(dataframe) else np.nan

    c = st.columns(4)
    c[0].metric("Rows (filtered)", f"{num_rows:,}")
    c[1].metric("Unique compounds", f"{num_unique_compounds:,}")
    c[2].metric("Mean cosine", f"{mean_cosine:.3f}" if np.isfinite(mean_cosine) else "—")
    c[3].metric(
        "Grouped Δm (fraction)",
        f"{fraction_grouped:.2%}" if np.isfinite(fraction_grouped) else "—",
    )


def plot_delta_mass_hist(dataframe: pd.DataFrame, column: str, title: str):
    if column not in dataframe.columns or dataframe[column].dropna().empty:
        st.info(f"No data for {title}.")
        return
    source_df = pd.DataFrame({column: dataframe[column].dropna().astype(float)})
    chart = (
        alt.Chart(source_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=60), title=column),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip(f"{column}:Q", bin=True), alt.Tooltip("count():Q")],
        )
        .properties(height=220, title=title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def plot_cosine_vs_dm(dataframe: pd.DataFrame):
    if (
        "Cosine" not in dataframe.columns
        or "DeltaMass" not in dataframe.columns
        or dataframe.empty
    ):
        st.info("Cosine vs Δm not available.")
        return
    source_df = dataframe[["Cosine", "DeltaMass"]].dropna()
    if source_df.empty:
        st.info("Cosine vs Δm not available.")
        return
    chart = (
        alt.Chart(source_df)
        .mark_point(opacity=0.6)
        .encode(
            x=alt.X("DeltaMass:Q", title="Δm (Da)"),
            y=alt.Y("Cosine:Q", title="Cosine"),
            tooltip=["DeltaMass:Q", "Cosine:Q"],
        )
        .properties(height=260, title="Cosine vs Δm")
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def top_rationales(dataframe: pd.DataFrame, top_n: int = 15):
    if "Rationale" not in dataframe.columns or dataframe["Rationale"].dropna().empty:
        st.info("No rationale column found.")
        return
    exploded = (
        dataframe.assign(Rationale=dataframe["Rationale"].fillna(""))
        .assign(Rationale=lambda d: d["Rationale"].str.split(r"\|"))
        .explode("Rationale")
    )
    exploded = exploded[exploded["Rationale"].str.strip() != ""]
    counts_df = exploded["Rationale"].value_counts().reset_index()
    counts_df.columns = ["Rationale", "Count"]
    counts_df = counts_df.head(top_n)

    chart = (
        alt.Chart(counts_df)
        .mark_bar()
        .encode(
            y=alt.Y("Rationale:N", sort="-x", title="Rationale"),
            x=alt.X("Count:Q"),
            tooltip=["Rationale:N", "Count:Q"],
        )
        .properties(height=22 * len(counts_df) + 20, title="Top rationales")
    )
    st.altair_chart(chart, use_container_width=True)


def download_buttons(dataframe: pd.DataFrame, label_prefix: str = "Download filtered"):
    st.subheader("Export filtered view")
    col_csv, col_parquet = st.columns(2)

    csv_buf = io.StringIO()
    dataframe.to_csv(csv_buf, index=False)
    col_csv.download_button(
        label=f"{label_prefix} (CSV)",
        data=csv_buf.getvalue(),
        file_name="suspects_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    pq_buf = io.BytesIO()
    dataframe.to_parquet(pq_buf, index=False)
    col_parquet.download_button(
        label=f"{label_prefix} (Parquet)",
        data=pq_buf.getvalue(),
        file_name="suspects_filtered.parquet",
        mime="application/octet-stream",
        use_container_width=True,
    )