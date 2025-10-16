from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from app.visualize_utils import (
    default_filters,
    load_all,
    build_filters,
    apply_filters,
    plot_summary,
    plot_delta_mass_hist,
    top_rationales,
    download_buttons,
)
 

def render():
    """Visualize grouped/unique suspects with charts first, then search form, then results."""
    st.caption("Visualize composed suspects.")

    interim_dir = Path("../data/interim")
    if not Path(interim_dir).exists():
        st.error(f"Data folder not found. Please generate suspects: `{interim_dir}`")
        return

    datasets_by_kind = load_all(interim_dir)
    if not datasets_by_kind:
        st.warning(
            "No parquet files found in `data/interim/` (expected *_grouped.parquet and/or *_unique.parquet)."
        )
        return

    kind_labels = {"grouped": "Grouped", "unique": "Unique"}
    available_kinds = [k for k in ("unique", "grouped") if k in datasets_by_kind]
    selected_kind = st.segmented_control(
        "Dataset",
        options=available_kinds,
        format_func=lambda k: kind_labels.get(k, k).title(),
        default=available_kinds[0],
    )
    selected_df = datasets_by_kind[selected_kind]

    with st.expander("Columns & types", expanded=False):
        st.write(f"Shape: {selected_df.shape[0]:,} rows × {selected_df.shape[1]:,} columns")
        st.dataframe(
            pd.DataFrame(
                {"column": selected_df.columns, "dtype": [str(t) for t in selected_df.dtypes]}
            ),
            use_container_width=True,
            hide_index=True,
        )

    default_filter_values = default_filters(selected_df)
    current_filters = st.session_state.get("filters", default_filter_values)

    if selected_kind not in st.session_state.get("filters_kind", ""):
        current_filters = default_filter_values
        st.session_state["filters"] = current_filters
        st.session_state["filters_kind"] = selected_kind

    filtered_df = apply_filters(selected_df, current_filters)

    plot_summary(filtered_df)

    st.subheader("Charts")
    st.caption(
        f"Charts reflect **{len(filtered_df):,} rows** (full filtered set), not just the first 100 shown below."
    )
    histogram_columns = st.columns(2)
    with histogram_columns[0]:
        plot_delta_mass_hist(filtered_df, "DeltaMass", "Δm (DeltaMass) histogram")
    with histogram_columns[1]:
        if "GroupDeltaMass" in filtered_df.columns:
            plot_delta_mass_hist(filtered_df, "GroupDeltaMass", "Grouped Δm histogram")
        else:
            st.info("No GroupDeltaMass in this dataset.")

    if selected_kind in ("grouped", "unique"):
        top_rationales(filtered_df)

    with st.form("search_form", clear_on_submit=False):
        new_filters = build_filters(selected_df, selected_kind, initial=current_filters)
        submitted = st.form_submit_button("Search")

    if submitted:
        st.session_state["filters"] = new_filters
        st.session_state["filters_kind"] = selected_kind
        st.rerun()

    st.subheader("Search results (first 100 rows)")
    preview = filtered_df.head(100)
    if preview.empty:
        st.warning("No rows match your filters.")
    else:
        st.dataframe(preview, use_container_width=True, hide_index=True, height=360)

    download_buttons(
        filtered_df, label_prefix=f"Download {kind_labels.get(selected_kind, selected_kind)} filtered"
    )

    with st.expander("USI examples", expanded=False):
        usi_columns = [c for c in ("LibraryUsi", "SuspectUsi") if c in filtered_df.columns]
        if usi_columns:
            head_df = filtered_df[usi_columns].drop_duplicates().head(10)
            st.dataframe(head_df, use_container_width=True, hide_index=True)
        else:
            st.info("No USI columns found.")