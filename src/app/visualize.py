from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

import altair as alt


# helpers for config and loading data from paths, i'm thinking of moving these to a newly created app/io.py or app/utils.py
@st.cache_data(show_spinner=False)
def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file from the given path into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        Path to the Parquet file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_parquet(path)


def _load_all(interim_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load up to three parquet files if present, return a dict keyed by type.
    Keys: 'unfiltered', 'grouped', 'unique'
    """
    candidate_files = {
        "unfiltered": list(interim_dir.glob("suspects_*_unfiltered.parquet")),
        "grouped":    list(interim_dir.glob("suspects_*_grouped.parquet")),
        "unique":     list(interim_dir.glob("suspects_*_unique.parquet")),
    }
    dataframes_by_kind: Dict[str, pd.DataFrame] = {}
    for kind_key, path_list in candidate_files.items():
        if path_list:
            latest_path = max(path_list, key=lambda x: x.stat().st_mtime)
            dataframes_by_kind[kind_key] = _load_parquet(latest_path)
    return dataframes_by_kind


# streamlit UI components for filtering, plotting, downloading; same as the data loaders i'm thinking of moving these to a newly created app/io.py or app/utils.py
def _numeric_range(series: pd.Series) -> Tuple[float, float]:
    """Compute a safe numeric range (min, max) for a series, padded if constant or invalid.

    Parameters
    ----------
    series : pd.Series
        Input numeric series.

    Returns
    -------
    Tuple[float, float]
        Tuple of (min_value, max_value).
    """
    if series.dropna().empty:
        return (0.0, 0.0)
    min_value, max_value = float(series.min()), float(series.max())
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return (0.0, 0.0)
    if min_value == max_value:
        pad = max(1e-6, abs(min_value) * 0.05)
        return (min_value - pad, max_value + pad)
    return (min_value, max_value)


def _build_filters(dataframe: pd.DataFrame, kind: str) -> Dict[str, Any]:
    """Render filtering widgets and return the chosen filter values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to derive filter options from.
    kind : str
        Dataset type key ('unfiltered', 'grouped', or 'unique').

    Returns
    -------
    Dict[str, Any]
        Dictionary of selected filter values.
    """
    st.subheader("Filters")

    column_search = st.text_input("Search CompoundName (contains)", value="").strip()

    min_cosine = 0.0
    if "Cosine" in dataframe.columns:
        default_cosine = 0.8 if kind in ("grouped", "unique") else 0.0
        min_cosine = st.slider("Minimum cosine", 0.0, 1.0, float(default_cosine), step=0.01)

    delta_mass_low, delta_mass_high = None, None
    if "DeltaMass" in dataframe.columns:
        min_value, max_value = _numeric_range(dataframe["DeltaMass"])
        delta_mass_low, delta_mass_high = st.slider(
            "Δm (DeltaMass) range",
            min_value=float(min_value),
            max_value=float(max_value),
            value=(float(min_value), float(max_value)),
            step=(max_value - min_value) / 200 if max_value > min_value else 0.001,
        )

    group_dm_low, group_dm_high = None, None
    if "GroupDeltaMass" in dataframe.columns and dataframe["GroupDeltaMass"].notna().any():
        min_group, max_group = _numeric_range(dataframe["GroupDeltaMass"].dropna())
        group_dm_low, group_dm_high = st.slider(
            "Grouped Δm (GroupDeltaMass) range",
            min_value=float(min_group),
            max_value=float(max_group),
            value=(float(min_group), float(max_group)),
            step=(max_group - min_group) / 200 if max_group > min_group else 0.001,
        )

    ionmode_options = sorted(dataframe["IonMode"].dropna().astype(str).unique()) if "IonMode" in dataframe.columns else []
    selected_ion_modes = st.multiselect("IonMode", ionmode_options, default=ionmode_options)

    instrument_options = sorted(dataframe["Instrument"].dropna().astype(str).unique()) if "Instrument" in dataframe.columns else []
    selected_instruments = st.multiselect("Instrument", instrument_options, default=instrument_options)

    return dict(
        search=column_search,
        min_cos=min_cosine,
        dm_range=(delta_mass_low, delta_mass_high),
        gdm_range=(group_dm_low, group_dm_high),
        ionmodes=selected_ion_modes,
        instruments=selected_instruments,
    )


def _apply_filters(dataframe: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply the selected filters to the provided DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame to filter.
    filters : Dict[str, Any]
        Dictionary of filter values produced by _build_filters.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    filtered_df = dataframe.copy()

    search_term = filters["search"]
    if search_term and "CompoundName" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["CompoundName"].astype(str).str.contains(search_term, case=False, na=False)]

    min_cosine = filters["min_cos"]
    if "Cosine" in filtered_df.columns and min_cosine is not None:
        filtered_df = filtered_df[filtered_df["Cosine"] >= float(min_cosine)]

    delta_mass_low, delta_mass_high = filters["dm_range"]
    if delta_mass_low is not None and delta_mass_high is not None and "DeltaMass" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["DeltaMass"].between(delta_mass_low, delta_mass_high)]

    group_dm_low, group_dm_high = filters["gdm_range"]
    if group_dm_low is not None and group_dm_high is not None and "GroupDeltaMass" in filtered_df.columns:
        group_mask = filtered_df["GroupDeltaMass"].notna()
        filtered_df = filtered_df[~group_mask | filtered_df["GroupDeltaMass"].between(group_dm_low, group_dm_high)]

    if "IonMode" in filtered_df.columns and filters["ionmodes"]:
        filtered_df = filtered_df[filtered_df["IonMode"].astype(str).isin(filters["ionmodes"])]
    if "Instrument" in filtered_df.columns and filters["instruments"]:
        filtered_df = filtered_df[filtered_df["Instrument"].astype(str).isin(filters["instruments"])]

    return filtered_df


# plots and some stats
def _plot(dataframe: pd.DataFrame, kind: str):
    """Display high-level metrics for the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to summarize.
    kind : str
        Dataset type key.
    """
    st.subheader("Summary")

    num_rows = len(dataframe)
    num_unique_compounds = dataframe["CompoundName"].nunique() if "CompoundName" in dataframe.columns else 0
    mean_cosine = dataframe["Cosine"].mean() if "Cosine" in dataframe.columns and not dataframe["Cosine"].empty else np.nan
    fraction_grouped = np.nan
    if "GroupDeltaMass" in dataframe.columns:
        non_null = dataframe["GroupDeltaMass"].notna().sum()
        fraction_grouped = (non_null / len(dataframe)) if len(dataframe) else np.nan

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{num_rows:,}")
    metric_columns[1].metric("Unique compounds", f"{num_unique_compounds:,}")
    metric_columns[2].metric("Mean cosine", f"{mean_cosine:.3f}" if np.isfinite(mean_cosine) else "—")
    metric_columns[3].metric("Grouped Δm (fraction)", f"{fraction_grouped:.2%}" if np.isfinite(fraction_grouped) else "—")


def _plot_delta_mass_hist(dataframe: pd.DataFrame, column: str, title: str):
    """Render a histogram for a Δm-like numeric column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing the column.
    column : str
        Column name to plot.
    title : str
        Title for the chart.
    """
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


def _plot_cosine_vs_dm(dataframe: pd.DataFrame):
    """Scatter plot of Cosine vs Δm (DeltaMass).

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame expected to contain 'Cosine' and 'DeltaMass' columns.
    """
    if "Cosine" not in dataframe.columns or "DeltaMass" not in dataframe.columns or dataframe.empty:
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


def _top_rationales(dataframe: pd.DataFrame, top_n: int = 15):
    """Display a bar chart of the most frequent rationale labels.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing a 'Rationale' column (pipe-separated labels).
    top_n : int, optional
        Number of top rationales to display, by default 15.
    """
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


# downloaders
def _download_buttons(dataframe: pd.DataFrame, label_prefix: str = "Download filtered"):
    """Render CSV and Parquet download buttons for the provided DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to export.
    label_prefix : str, optional
        Label prefix for the buttons, by default "Download filtered".
    """
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


# render main tab, same as in generate.py
def render():
    """Render the Streamlit UI to visualize and explore suspects datasets."""
    st.caption("Visualize suspects produced by the pipeline (unfiltered / grouped / unique).")

    interim_dir = Path("../data/interim")
    st.write(f"**Data folder:** `{interim_dir}`")
    if not Path(interim_dir).exists():
        st.error(f"Data folder not found please generate suspects: `{interim_dir}`")
        return
        

    datasets_by_kind = _load_all(interim_dir)

    if not datasets_by_kind:
        st.warning("No parquet files found in `data/interim/` (expected *_unfiltered.parquet, *_grouped.parquet, *_unique.parquet), please generate suspects first.")
        return

    kind_labels = {
        "unfiltered": "Unfiltered",
        "grouped": "Grouped",
        "unique": "Unique",
    }
    available_kinds = [k for k in ("unfiltered", "grouped", "unique") if k in datasets_by_kind]
    selected_kind = st.segmented_control(
        "Dataset",
        options=available_kinds,
        format_func=lambda k: kind_labels.get(k, k).title(),
        default=available_kinds[0],
    )

    selected_df = datasets_by_kind[selected_kind]

    # some info about the dataframe
    with st.expander("Columns & types", expanded=False):
        st.write(f"Shape: {selected_df.shape[0]:,} rows × {selected_df.shape[1]:,} columns")
        st.dataframe(
            pd.DataFrame({"column": selected_df.columns, "dtype": [str(t) for t in selected_df.dtypes]}),
            use_container_width=True,
            hide_index=True,
        )

    _plot(selected_df, selected_kind)

    filters = _build_filters(selected_df, selected_kind)
    filtered_df = _apply_filters(selected_df, filters)

    st.subheader("Filtered table")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=360)

    st.subheader("Charts")
    histogram_columns = st.columns(2)
    with histogram_columns[0]:
        _plot_delta_mass_hist(filtered_df, "DeltaMass", "Δm (DeltaMass) histogram")
    with histogram_columns[1]:
        if "GroupDeltaMass" in filtered_df.columns:
            _plot_delta_mass_hist(filtered_df, "GroupDeltaMass", "Grouped Δm histogram")
        else:
            st.info("No GroupDeltaMass in this dataset.")

    _plot_cosine_vs_dm(filtered_df)

    if selected_kind in ("grouped", "unique"):
        _top_rationales(filtered_df)

    _download_buttons(filtered_df, label_prefix=f"Download {kind_labels.get(selected_kind, selected_kind)} filtered")

    with st.expander("USI examples", expanded=False):
        usi_columns = [c for c in ("LibraryUsi", "SuspectUsi") if c in filtered_df.columns]
        if usi_columns:
            head_df = filtered_df[usi_columns].drop_duplicates().head(10)
            st.dataframe(head_df, use_container_width=True, hide_index=True)
        else:
            st.info("No USI columns found.")
