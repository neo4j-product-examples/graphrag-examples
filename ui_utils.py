import base64
import streamlit as st


def render_centered_svg_from_str(svg: str, px):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = f'''<img src="data:image/svg+xml;base64,{b64}" style=" max-width: {px}px;height: auto;display: block;margin-left: auto;margin-right: auto;"/>'''
    st.write(html, unsafe_allow_html=True)


def render_header_svg(file_path: str, px: int = 200):
    with open(file_path) as f:
        svg = f.read()
    render_centered_svg_from_str(svg, px)
