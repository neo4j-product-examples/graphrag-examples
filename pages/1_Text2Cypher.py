import streamlit as st

from graphrag import GraphRAGChain, GraphRAGText2CypherChain
from ui_utils import render_header_svg, get_neo4j_url_from_uri

NORTHWIND_NEO4J_URI = st.secrets['NORTHWIND_NEO4J_URI']
NORTHWIND_NEO4J_USERNAME = st.secrets['NORTHWIND_NEO4J_USERNAME']
NORTHWIND_NEO4J_PASSWORD = st.secrets['NORTHWIND_NEO4J_PASSWORD']

st.set_page_config(page_icon="images/logo-mark-fullcolor-RGB-transBG.svg", layout="wide")
render_header_svg("images/graphrag.svg", 200)
render_header_svg("images/bottom-header.svg", 200)
st.markdown(' ')
with st.expander('Dataset Info:'):
    st.markdown('''#### [H&M Fashion Dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data), a sample of a real retail dataset including customer purchase data and rich information around products such as names, types, descriptions, department sections, etc.
    ''')
    st.image('images/northwind-data-model.png', width=800)
    st.markdown(
        f'''use the following queries in [Neo4j Browser]({get_neo4j_url_from_uri(NORTHWIND_NEO4J_URI)}) to explore the data:''')
    st.code('CALL db.schema.visualization()', language='cypher')
    st.code('''MATCH p=()-[]->()-[]->() RETURN p LIMIT 300''', language='cypher')

prompt_instructions_with_schema = '''#Context 

You have expertise in neo4j cypher query language and based on below graph data model schema, you are going to help me write cypher queries. 

Node Labels and Properties

["Customer"], ["country:String", "address:String", "contactTitle:String", "phone:String", "city:String", "Bloom_Link:String", "contactName:String", "postalCode:String", "companyName:String", "customerID:String", "region:String", "fax:String"]
["Supplier"], ["country:String", "address:String", "contactTitle:String", "supplierID:String", "phone:String", "city:String", "contactName:String", "postalCode:String", "companyName:String", "fax:String", "region:String", "homePage:String"]
["Order"], ["shipCity:String", "orderID:String", "freight:String", "requiredDate:String", "employeeID:String", "shipPostalCode:String", "shipName:String", "shipCountry:String", "shipAddress:String", "shipVia:String", "customerID:String", "shipRegion:String", "shippedDate:String", "orderDate:String"]
["Category"], ["description:String", "categoryName:String", "picture:String", "categoryID:String"]
["Product"], ["reorderLevel:String", "unitsInStock:String", "unitPrice:String", "supplierID:String", "productID:String", "discontinued:String", "quantityPerUnit:String", "unitsOnOrder:String", "productName:String", "categoryID:String"]
["Order_Detail"], ["unitPrice:String", "discount:String", "quantity:String", "productID:String", "orderID:String"]
["Address"], ["addressID", "name", "address", "city", "region", "postalCode", "country"]

Accepted graph traversal paths

(:Customer)-[:ORDERED]->(:Order), 
(:Product)-[:BELONGS_TO]->(:Category),
(:Product)-[:SUPPLIED_BY]->(:Supplier), 
(:Order)-[:ORDER_CONTAINS]->(:Product), 
(:Order)-[:SHIPPED_TO]->(:Address)
'''

prompt_instructions_vector_only = """You are a product and retail expert who can answer questions based only on the context below.
* Answer the question STRICTLY based on the context provided in JSON below.
* Do not assume or retrieve any information outside of the context 
* Think step by step before answering.
* Do not return helpful or extra text or apologies
* List the results in rich text format if there are more than one results"""

top_k_vector_only = 5
vector_index_name = 'product_text_embeddings'

vector_only_rag_chain = GraphRAGChain(
    neo4j_uri=NORTHWIND_NEO4J_URI,
    neo4j_auth=(NORTHWIND_NEO4J_USERNAME, NORTHWIND_NEO4J_PASSWORD),
    vector_index_name=vector_index_name,
    prompt_instructions=prompt_instructions_vector_only,
    k=top_k_vector_only)

graphrag_t2c_chain = GraphRAGText2CypherChain(
    neo4j_uri=NORTHWIND_NEO4J_URI,
    neo4j_auth=(NORTHWIND_NEO4J_USERNAME, NORTHWIND_NEO4J_PASSWORD),
    prompt_instructions=prompt_instructions_with_schema,
    properties_to_remove_from_cypher_res=['textEmbedding'])

prompt = st.text_input("submit a prompt:", value="")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vector Only")
    if prompt:
        with st.spinner('Running Vector Only RAG...'):
            with st.expander('__Response:__', True):
                st.markdown(vector_only_rag_chain.invoke(prompt))
            with st.expander("__Context used to answer this prompt:__"):
                st.json(vector_only_rag_chain.last_used_context)
            with st.expander("__Query used to retrieve context:__"):
                vector_rag_query = vector_only_rag_chain.get_full_retrieval_query(prompt)
                st.markdown(f"""
                This query only uses vector search.  The vector search will return the highest ranking `nodes` based on the vector similarity `score`(for this example we chose `{top_k_vector_only}` nodes)
                """)
                st.code(vector_rag_query, language='cypher')
                st.markdown('### Visualize Retrieval in Neo4j')
                st.markdown('To explore the results in Neo4j do the following:\n' +
                            '* Go to [Neo4j Workspace](https://workspace.neo4j.io/connection/connect) and enter your credentials\n' +
                            '* In the Query panel run the above query')
                st.link_button("Try in Neo4j Workspace!", "https://workspace.neo4j.io/connection/connect")

            st.success('Done!')

with col2:
    st.subheader("Text2Cypher")

    if prompt:
        with st.spinner('Running GraphRAG...'):
            with st.expander('__Response:__', True):
                st.markdown(graphrag_t2c_chain.invoke(prompt))

            with st.expander("__Context used to answer this prompt:__"):
                st.json(graphrag_t2c_chain.last_used_context)

            with st.expander("__Query used to retrieve context:__"):
                graph_rag_query = graphrag_t2c_chain.last_retrieval_query
                st.markdown(f"""
                """)
                st.code(graph_rag_query, language='cypher')
                st.markdown('### Visualize Retrieval in Neo4j')
                st.markdown('To explore the results in Neo4j do the following:\n' +
                            '* Go to [Neo4j Workspace](https://workspace.neo4j.io/connection/connect) and enter your credentials\n' +
                            '* In the Query panel run the above query')
                st.link_button("Try in Neo4j Workspace!", "https://workspace.neo4j.io/connection/connect")

            st.success('Done!')

st.markdown("---")

st.markdown("""
<style>
  table {
    width: 100%;
    border-collapse: collapse;
    border: none !important;
    font-family: "Source Sans Pro", sans-serif;
    color: rgba(49, 51, 63, 0.6);
    font-size: 0.9rem;
  }

  tr {
    border: none !important;
  }

  th {
    text-align: center;
    colspan: 3;
    border: none !important;
    color: #0F9D58;
  }

  th, td {
    padding: 2px;
    border: none !important;
  }
</style>

<table>
  <tr>
    <th colspan="3">Sample questions to try</th>
  </tr>
  <tr>
    <td>Find the top 10 customers by orders and get the 5 most common products among those orders</td>
    <td>What are the products purchased often together? Provide only the top 10</td>
    <td></td>
  </tr>
  <tr>
    <td>Who is the top customer in USA with highest number of orders with Beverages product category? Return the Bloom link for that customer</td>
    <td>A customer is purchasing "Sir Rodney's Marmalade". What top 4 products can we recommend based on past customer transactions?</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>
""", unsafe_allow_html=True)
