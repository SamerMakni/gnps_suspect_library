from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="GNPS Suspects Prototype", layout="centered")

st.title("GNPS Suspects")

from app import generate, visualize


tab_generate, tab_visualize = st.tabs(["Generate", "Visualize"])

with tab_generate:
    generate.render()

with tab_visualize:
    visualize.render()