import streamlit as st

from graph_rag import GraphRAGChain, GraphRAGText2CypherChain
from ui_utils import render_header_svg

st.set_page_config(page_icon="images/logo-mark-fullcolor-RGB-transBG.svg", layout="wide")

render_header_svg("images/graphrag.svg", 200)

render_header_svg("images/bottom-header.svg", 200)

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
    vector_index_name,
    prompt_instructions_vector_only,
    k=top_k_vector_only)

graphrag_t2c_chain = GraphRAGText2CypherChain(prompt_instructions_with_schema,
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
                vector_rag_query = vector_only_rag_chain.get_browser_queries(prompt)
                st.markdown(f"""
                This query only uses vector search.  The vector search will return the highest ranking `nodes` based on the vector similarity `score`(for this example we chose `{top_k_vector_only}` nodes)
                """)
                st.code(vector_rag_query['query_body'], language='cypher')

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

            st.success('Done!')