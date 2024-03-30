import streamlit as st

from rag_vector_graph import GraphRAGChain
from ui_utils import render_header_svg

st.set_page_config(page_icon="images/logo-mark-fullcolor-RGB-transBG.svg", layout="wide")

render_header_svg("images/graphrag.svg", 200)

render_header_svg("images/bottom-header.svg", 200)

graph_retrieval_query = """
WITH node AS product, score 
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
    vector_index_name,
    prompt_instructions,
    k=top_k)

graphrag_chain = GraphRAGChain(
    vector_index_name,
    prompt_instructions,
    graph_retrieval_query=graph_retrieval_query,
    k=top_k)

prompt = st.text_input("Submit a prompt:", value="")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vector Only")

    with st.expander("Just Vector Search Retrieval:"):
        st.markdown("#### No Additional Context")
        st.markdown(f"""
        This will just do a vector search for retrieval.  The vector search will return the highest ranking `nodes` based on the vector similarity `score`(for this example we chose `{top_k}` nodes)
        """)
    if prompt:
        with st.spinner('Running GraphRAG...'):
            st.markdown(vector_only_rag_chain.invoke(prompt))
            with st.expander("Context used to answer this prompt:"):
                st.markdown("#### Context Pulled from Database")
                st.code(vector_only_rag_chain.last_used_context, language='json')
            st.success('Done!')

with col2:
    st.subheader("Vector Search & Graph Context")

    with st.expander("Incorporate Graph Context to Vector Search Results:"):
        st.markdown("#### Cypher Query For Additional Graph Context")
        st.markdown(f"""
        The following Cypher was used to achieve additional context from the graph. This query is run after the vector search.  The vector search will return the highest ranking `nodes` based on the vector similarity `score`(for this example we chose `{top_k}` nodes) . The below query then takes those nodes and scores and performs additional graph traversals and aggregation logic to obtain context.  In some ways you can think of this additional context as 'metadata' with the added benfitis of it being collected in real-time with the ability tp use very flecxible and robust patterns to do so. 
        """)
        st.code(graph_retrieval_query, language='cypher')

    if prompt:
        with st.spinner('Running GraphRAG...'):
            st.markdown(graphrag_chain.invoke(prompt))
            with st.expander("Context used to answer this prompt:"):
                st.markdown("#### Context Pulled from Database")
                st.code(graphrag_chain.last_used_context, language='json')
            st.success('Done!')

css='''
<style>
    [data-testid="stExpander"] div:has(>.streamlit-expanderContent) {
        overflow: scroll;
        height: 400px;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)