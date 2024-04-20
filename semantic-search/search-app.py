import pandas
import streamlit as st
from langchain_community.graphs.neo4j_graph import Neo4jGraph

NEO4J_URI = st.secrets['RESUME_NEO4J_URI']
NEO4J_USERNAME = st.secrets['RESUME_NEO4J_USERNAME']
NEO4J_PASSWORD = st.secrets['RESUME_NEO4J_PASSWORD']

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

search_results = graph.query('''
MATCH(p:Person)
WITH p LIMIT 9
MATCH (p)-[r:HAS_SKILL]->(s)
WITH p, collect({skillName:r.name , skillLevel:r.level}) AS skills
MATCH(p)-[:HAS_POSITION]->(s)
WITH p, skills, collect(s {.*}) AS positions
MATCH(p)-[:HAS_EDUCATION]->(e)
RETURN p.entityId AS entityId, p.name AS name, p.base64Image AS base64Image, p.role AS role, 
p.description AS description, skills, positions, collect(e {.*}) AS education
''')
st.set_page_config(page_icon="images/logo-mark-fullcolor-RGB-transBG.svg", layout="wide")

st.header('Resume Search:')
text_search = st.text_input("Search for talent:", value="")

n_cards_per_row = 3
if text_search:
    for k in range(len(search_results)):
        row = search_results[k]
        i = k % n_cards_per_row
        if i == 0:
            st.write("---")
            cols = st.columns(n_cards_per_row, gap="large")
        # draw the card
        with cols[k % n_cards_per_row]:
            st.caption(f"{row['entityId'].strip()} - {row['role'].strip()} ")
            st.markdown(f"**{row['description']}**")
            st.markdown(f"*{row['skills']}*")
            st.markdown(f"**{row['positions']}**")
