import streamlit as st


from ui_utils import render_header_svg

st.set_page_config(page_icon="images/logo-mark-fullcolor-RGB-transBG.svg", layout="wide")

render_header_svg("images/graphrag.svg", 200)

render_header_svg("images/bottom-header.svg", 200)
