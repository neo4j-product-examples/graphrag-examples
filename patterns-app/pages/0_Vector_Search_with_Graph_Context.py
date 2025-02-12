import streamlit as st

from graphrag import GraphRAGChain
from ui_utils import render_header_svg, get_neo4j_url_from_uri

NORTHWIND_NEO4J_URI = st.secrets['NORTHWIND_NEO4J_URI']
NORTHWIND_NEO4J_USERNAME = st.secrets['NORTHWIND_NEO4J_USERNAME']
NORTHWIND_NEO4J_PASSWORD = st.secrets['NORTHWIND_NEO4J_PASSWORD']
NORTHWIND_NEO4J_DATABASE = st.secrets.get('NORTHWIND_NEO4J_DATABASE', 'neo4j')


st.set_page_config(page_icon="images/logo-mark-fullcolor-RGB-transBG.svg", layout="wide")
render_header_svg("images/graphrag.svg", 200)
render_header_svg("images/bottom-header.svg", 200)
st.markdown(' ')
with st.expander('Dataset Info:'):
    st.markdown('''####  Northwind: Sales data for Northwind Traders, a fictitious specialty foods export/import company.
    ''')
    st.image('images/northwind-data-model.png', width=800)
    st.markdown(
        f'''use the following queries in [Neo4j Browser]({get_neo4j_url_from_uri(NORTHWIND_NEO4J_URI)}) to explore the data:''')
    st.code('CALL db.schema.visualization()', language='cypher')
    st.code('''MATCH p=()-[]->()-[]->() RETURN p LIMIT 300''', language='cypher')


graph_retrieval_query = """WITH node AS product, score 
MATCH (product)<-[:ORDER_CONTAINS]-(o:Order)<-[:ORDERED]-(c:Customer)
MATCH (product)-[:SUPPLIED_BY]->(s:Supplier)
WITH product, s, c, count(*) AS orderCount, score
WITH product, 
    score,
    s.companyName AS productSupplierName, 
    orderCount,
    {customerName:c.companyName, orderCount:orderCount} AS customerData
WITH product, 
    score,
    productSupplierName,
    collect(customerData) AS customerData,
    sum(orderCount) as totalOrders
MATCH (product)<-[:ORDER_CONTAINS]-(o:Order)-[:ORDER_CONTAINS]->(recommendedProduct:Product)
WITH product, 
    score, 
    productSupplierName, 
    totalOrders, 
    customerData, 
    recommendedProduct, count(*) AS copurchaseCount
WHERE copurchaseCount > 2
WITH product, 
    score, 
    productSupplierName, 
    totalOrders, 
    customerData, 
    collect({recommendedProduct:recommendedProduct.productName, copurchaseCount:copurchaseCount}) AS recommendedProducts
RETURN  product.text AS text, 
    score,
    {
        productSupplierName: productSupplierName, 
        totalOrders: totalOrders, 
        customerData: customerData, 
        recommendedProducts: recommendedProducts
    } AS metadata
"""

prompt_instructions = """You are a product and retail expert who can answer questions based only on the context below.
* Answer the question STRICTLY based on the context provided in JSON below.
* Do not assume or retrieve any information outside of the context 
* Think step by step before answering.
* Do not return helpful or extra text or apologies
* List the results in rich text format if there are more than one results"""

top_k = 5
vector_index_name = 'product_text_embeddings'

vector_only_rag_chain = GraphRAGChain(
    neo4j_uri=NORTHWIND_NEO4J_URI,
    neo4j_username=NORTHWIND_NEO4J_USERNAME,
    neo4j_password=NORTHWIND_NEO4J_PASSWORD,
    neo4j_database=NORTHWIND_NEO4J_DATABASE,
    vector_index_name=vector_index_name,
    prompt_instructions=prompt_instructions,
    k=top_k)

graphrag_chain = GraphRAGChain(
    neo4j_uri=NORTHWIND_NEO4J_URI,
    neo4j_username=NORTHWIND_NEO4J_USERNAME,
    neo4j_password=NORTHWIND_NEO4J_PASSWORD,
    neo4j_database=NORTHWIND_NEO4J_DATABASE,
    vector_index_name=vector_index_name,
    prompt_instructions=prompt_instructions,
    graph_retrieval_query=graph_retrieval_query,
    k=top_k)

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
                This query only uses vector search.  The vector search will return the highest ranking `nodes` based on the vector similarity `score`(for this example we chose `{top_k}` nodes)
                """)
                st.code(vector_rag_query, language='cypher')
                st.markdown('### Visualize Retrieval in Neo4j')
                st.markdown('To explore the results in Neo4j do the following:\n' +
                            f'* Go to [Neo4j Browser]({get_neo4j_url_from_uri(NORTHWIND_NEO4J_URI)}) and enter your credentials\n' +
                            '* Run the above queries')
                st.link_button("Try in Neo4j Browser!", get_neo4j_url_from_uri(NORTHWIND_NEO4J_URI))

            st.success('Done!')

with col2:
    st.subheader("Vector Search & Graph Context")

    if prompt:
        with st.spinner('Running GraphRAG...'):
            with st.expander('__Response:__', True):
                st.markdown(graphrag_chain.invoke(prompt))

            with st.expander("__Context used to answer this prompt:__"):
                st.json(graphrag_chain.last_used_context)

            with st.expander("__Query used to retrieve context:__"):
                graph_rag_query = graphrag_chain.get_full_retrieval_query(prompt)
                st.markdown(f"""The following Cypher query was used to obtain vector results enriched with additional context from the graph. The query initially performs a vector search, returning the highest ranking `nodes` based on their vector similarity `score`. In this example, we selected `{top_k}` nodes. Subsequently, the query performs further graph traversals and aggregation to gather context. You can think of this context as 'metadata,' but with the advantages of real-time collection and the flexibility to use robust patterns.
                """)
                st.code(graph_rag_query, language='cypher')
                st.markdown('### Visualize Retrieval in Neo4j')
                st.markdown('To explore the results in Neo4j do the following:\n' +
                            f'* Go to [Neo4j Browser]({get_neo4j_url_from_uri(NORTHWIND_NEO4J_URI)}) and enter your credentials\n' +
                            '* Run the above queries')
                st.link_button("Try in Neo4j Browser!", get_neo4j_url_from_uri(NORTHWIND_NEO4J_URI))

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
    <td>What are 3 popular cheeses?</td>
    <td>What are 3 popular cheeses? - How are you measuring popularity?</td>
    <td>What are 3 popular cheeses? What else can you recommend to go with them? - How are you measuring popularity?</td>
  </tr>
  <tr>
    <td>What are 3 popular cheeses? What customers buy the most of these?</td>
    <td>What customers are buying the most caffeinated beverages? Can you also list the beverages and the amount they are buying?</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
  </tr>
</table>
""", unsafe_allow_html=True)
