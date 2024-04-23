import pandas
import streamlit as st
from retreivers import retrieval_with_tools

st.set_page_config(page_icon="images/logo-mark-fullcolor-RGB-transBG.svg", layout="wide")

st.header('Resume Search:')
text_search = st.text_input("Search for talent:", value="")

n_cards_per_row = 3
if text_search:
    search_results = retrieval_with_tools(prompt=text_search)
    for k in range(len(search_results)):
        row = search_results[k]
        i = k % n_cards_per_row
        if i == 0:
            st.write("---")
            cols = st.columns(n_cards_per_row, gap="large")
        # draw the card
        with cols[k % n_cards_per_row]:
            with st.container(height=500):
                inner_col1, inner_col2 = st.columns(2)
                with inner_col1:
                    st.image(f'img/{row["entityId"][:-4]}.jpg')
                with inner_col2:
                    st.markdown(f" ## {row['name'].strip()}")
                    st.markdown(f" ### {row['role'].strip()}")
                st.markdown(f"**{row['description']}**")
                st.markdown(row['hitData'])
