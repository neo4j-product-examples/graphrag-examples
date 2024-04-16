import streamlit as st

from graphrag import GraphRAGPreFilterChain, DynamicGraphRAGChain
from ui_utils import render_header_svg, get_neo4j_url_from_uri

HM_NEO4J_URI = st.secrets['HM_NEO4J_URI']
HM_NEO4J_USERNAME = st.secrets['HM_NEO4J_USERNAME']
HM_NEO4J_PASSWORD = st.secrets['HM_NEO4J_PASSWORD']

st.set_page_config(page_icon="images/logo-mark-fullcolor-RGB-transBG.svg", layout="wide")
st.markdown(' ')
render_header_svg("images/graphrag.svg", 200)
render_header_svg("images/bottom-header.svg", 200)
st.markdown(' ')
with st.expander('Dataset Info:'):
    st.markdown('''#### [H&M Fashion Dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data), a sample of a real retail dataset including customer purchase data and rich information around products such as names, types, descriptions, department sections, etc.
    ''')
    st.image('images/hm-data-model.png', width=800)
    st.markdown(f'''use the following queries in [Neo4j Browser]({get_neo4j_url_from_uri(HM_NEO4J_URI)}) to explore the data:''')
    st.code('CALL db.schema.visualization()', language='cypher')
    st.code('''MATCH (p:Product)<-[v:VARIANT_OF]-(a:Article)<-[t:PURCHASED]-(c:Customer)
    RETURN * LIMIT 150''', language='cypher')
st.markdown('''### Task: Generate email content for personalized product recommendations based of customer interests, purchase history, and time of year.''')


top_k = 20
vector_index_name = 'product_text_embeddings'

prefilter_graph_retrieval_query = """
MATCH (:Customer {customerId:$customerId})-[:PURCHASED]->(:Article)
<-[:PURCHASED]-(:Customer)-[:PURCHASED]->(recArticle:Article)-[:VARIANT_OF]->(product:Product)
WITH count(recArticle) AS recommendationScore, product
ORDER BY recommendationScore DESC LIMIT 100
WITH product AS node, {recommendationScore:recommendationScore} AS prefilterMetadata"""

postfilter_graph_retrieval_query = """WITH node AS product, score AS searchScore
OPTIONAL MATCH(product)<-[:VARIANT_OF]-(:Article)<-[:PURCHASED]-(:Customer)
-[:PURCHASED]->(a:Article)<-[:PURCHASED]-(:Customer {customerId: $customerId})

WITH count(a) AS purchaseScore, product.text AS text, searchScore, product.productCode AS productCode
RETURN text,
    (1+purchaseScore)*searchScore AS score,
    {productCode: productCode, purchaseScore:purchaseScore, searchScore:searchScore} AS metadata
ORDER BY purchaseScore DESC, searchScore DESC LIMIT 20"""

graphrag_prefilter_chain = GraphRAGPreFilterChain(neo4j_uri=HM_NEO4J_URI,
                                                  neo4j_auth=(HM_NEO4J_USERNAME, HM_NEO4J_PASSWORD),
                                                  vector_index_name='product_text_embeddings',
                                                  graph_prefilter_query=prefilter_graph_retrieval_query,
                                                  k=top_k)

graphrag_postfilter_chain = DynamicGraphRAGChain(neo4j_uri=HM_NEO4J_URI,
                                                 neo4j_auth=(HM_NEO4J_USERNAME, HM_NEO4J_PASSWORD),
                                                 vector_index_name='product_text_embeddings',
                                                 graph_retrieval_query=postfilter_graph_retrieval_query,
                                                 k=100)


def generate_prompt(cstmr_name_input, time_of_year_input):
    return f'''
    You are a personal assistant named Sally for a fashion, home, and beauty company called HRM.
    write an email to {cstmr_name_input}, one of your customers, to promote and summarize products relevant for them 
    given the current season / time of year: {time_of_year_input}. 
    Please only mention the products listed in the context below. Do not come up with or add any new products to the list.
    Select the best 4 to 5 product subset from the context that best match the time of year: {time_of_year_input}.
    Each product comes with an https `url` field. Make sure to provide that https url with descriptive name text in markdown for each product.
    '''


examples = [
    [
        'Alex Smith',
        'Oversized Sweaters',
        'daae10780ecd14990ea190a1e9917da33fe96cd8cfa5e80b67b4600171aa77e0',
        'Feb, 2024',
    ],
    [
        'Robin Fischer',
        'Oversized Sweaters',
        '819f4eab1fd76b932fd403ae9f427de8eb9c5b64411d763bb26b5c8c3c30f16f',
        'Feb, 2024'
    ],
    [
        'Chris Johnson',
        'Oversized Sweaters',
        '44b0898ecce6cc1268dfdb0f91e053db014b973f67e34ed8ae28211410910693',
        'Feb, 2024'
    ],
    [
        'Robin Fischer',
        'denim jeans',
        '819f4eab1fd76b932fd403ae9f427de8eb9c5b64411d763bb26b5c8c3c30f16f',
        'Feb, 2024'
    ]
]

preset_example = st.selectbox("select an example case:", examples)

with st.form('input_form'):
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        customer_name = st.text_input("customer name:", value=preset_example[0], key='customer_name_input')
        customer_id = st.text_input("customerId:", value=preset_example[2], key='customer_id_input')

    with input_col2:
        customer_interests = st.text_input("customer interest(s):", value=preset_example[1],
                                           key='customer_interests_input')
        time_of_year = st.text_input("time of year:", value=preset_example[3], key='time_of_year_input')

    gen_content = st.form_submit_button('Generate Content')

if gen_content:
    st.markdown('### Initial Prompt: ')
    st.markdown(generate_prompt(customer_name, time_of_year))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Graph Post-Filtering")
    if gen_content:
        with st.spinner('Running GraphRAG...'):
            with st.expander('__Response:__', True):
                st.markdown(graphrag_postfilter_chain.invoke(
                    generate_prompt(customer_name, time_of_year),
                    retrieval_search_text=customer_interests,
                    query_params={"customerId": customer_id})
                )
            with st.expander("__Context used to answer this prompt:__"):
                st.json(graphrag_postfilter_chain.last_used_context)

            with st.expander("__Query used to retrieve context:__"):
                graphrag_post_filter_queries = graphrag_postfilter_chain.get_last_browser_queries()
                st.code(graphrag_post_filter_queries['params_query'], language='cypher')
                st.code(graphrag_post_filter_queries['query_body'], language='cypher')
                st.markdown('### Visualize Retrieval in Neo4j')
                st.markdown('To explore the results in Neo4j do the following:\n' +
                            f'* Go to [Neo4j Browser]({get_neo4j_url_from_uri(HM_NEO4J_URI)}) and enter your credentials\n' +
                            '* In the Query panel run the above query')
                st.link_button("Try in Neo4j Browser!", get_neo4j_url_from_uri(HM_NEO4J_URI))

            st.success('Done!')

with col2:
    st.subheader("Graph Pre-Filtering")
    if gen_content:
        with st.spinner('Running GraphRAG...'):
            with st.expander('__Response:__', True):
                st.markdown(graphrag_prefilter_chain.invoke(
                    generate_prompt(customer_name, time_of_year),
                    retrieval_search_text=customer_interests,
                    query_params={"customerId": customer_id})
                )
            with st.expander("__Context used to answer this prompt:__"):
                st.json(graphrag_prefilter_chain.last_used_context)

            with st.expander("__Query used to retrieve context:__"):
                graphrag_prefilter_queries = graphrag_prefilter_chain.get_last_browser_queries()
                st.code(graphrag_prefilter_queries['params_query'], language='cypher')
                st.code(graphrag_prefilter_queries['query_body'], language='cypher')
                st.markdown('### Visualize Retrieval in Neo4j')
                st.markdown('To explore the results in Neo4j do the following:\n' +
                            f'* Go to [Neo4j Browser]({get_neo4j_url_from_uri(HM_NEO4J_URI)}) and enter your credentials\n' +
                            '* In the Query panel run the above query')
                st.link_button("Try in Neo4j Browser!", get_neo4j_url_from_uri(HM_NEO4J_URI))

            st.success('Done!')
