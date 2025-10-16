from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="GNPS Suspects Prototype", layout="centered")

st.title("GNPS Suspects")

from app import generate, visualize, export


tab_generate, tab_visualize, tab_export = st.tabs(["Generate", "Visualize", "Export"])

with tab_generate:
    generate.render()

with tab_visualize:
    visualize.render()

with tab_export:
    export.render()
    
