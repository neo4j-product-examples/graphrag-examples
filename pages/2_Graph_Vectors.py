import streamlit as st

from graphrag import DynamicGraphRAGChain
from ui_utils import render_header_svg, get_neo4j_url_from_uri

HM_NEO4J_URI = st.secrets['HM_NEO4J_URI']
HM_NEO4J_USERNAME = st.secrets['HM_NEO4J_USERNAME']
HM_NEO4J_PASSWORD = st.secrets['HM_NEO4J_PASSWORD']

st.set_page_config(page_icon="images/logo-mark-fullcolor-RGB-transBG.svg", layout="wide")
render_header_svg("images/graphrag.svg", 200)
render_header_svg("images/bottom-header.svg", 200)
st.markdown(' ')
with st.expander('Dataset Info:'):
    st.markdown('''#### [H&M Fashion Dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data): A sample of real-world retail data, including customer purchase data and rich information around products such as names, types, descriptions, department sections, etc.
    ''')
    st.image('images/hm-data-model.png', width=800)
    st.markdown(f'''use the following queries in [Neo4j Browser]({get_neo4j_url_from_uri(HM_NEO4J_URI)}) to explore the data:''')
    st.code('CALL db.schema.visualization()', language='cypher')
    st.code('''MATCH (p:Product)<-[v:VARIANT_OF]-(a:Article)<-[t:PURCHASED]-(c:Customer) RETURN * LIMIT 150''', language='cypher')
st.markdown('''### Task: Generate fashion recommendations to pair with customer's recent purchases and interests given time of year.''')


vector_index_name = 'product_text_embeddings'

graph_retrieval_query = """WITH node AS searchProduct, score AS searchScore
MATCH(searchProduct)<-[:VARIANT_OF]-(searchArticle:Article)
WHERE  searchArticle.graphEmbedding IS NOT NULL
CALL db.index.vector.queryNodes('article_graph_embeddings', 10, searchArticle.graphEmbedding) YIELD node, score
WHERE score < 1.0
MATCH (node)-[:VARIANT_OF]->(product)
RETURN product.`text` AS text, 
    max(score) AS score, 
    product {.*, `text`: Null, `textEmbedding`: Null, id: Null} AS metadata
ORDER by score DESC LIMIT 20"""

graph_vector_chain = DynamicGraphRAGChain(neo4j_uri=HM_NEO4J_URI,
                                          neo4j_auth=(HM_NEO4J_USERNAME, HM_NEO4J_PASSWORD),
                                          vector_index_name='product_text_embeddings',
                                          graph_retrieval_query=graph_retrieval_query,
                                          k=10)

vector_only_chain = DynamicGraphRAGChain(neo4j_uri=HM_NEO4J_URI,
                                         neo4j_auth=(HM_NEO4J_USERNAME, HM_NEO4J_PASSWORD),
                                         vector_index_name='product_text_embeddings',
                                         k=10)


def generate_prompt(cstmr_name_input, time_of_year_input, cstmr_interests_input):
    return f'''
    You are a personal assistant named Sally for a fashion, home, and beauty company called HRM.
    write an email to {cstmr_name_input}, one of your customers, to recommend and summarize products that pair well with their 
    recent purchases and searches given: 
    - the current season / time of year: {time_of_year_input} 
    - Recent purchases / searches: {cstmr_interests_input}
    
    Please only mention the products listed in the context below. Do not come up with or add any new products to the list.
    The below candidates are recommended based on the purchase patterns of other customers in the HRM database.
    Select the best 4 to 5 product subset from the context that best match the time of year: {time_of_year_input} and to pair with recent purchases.
    Each product comes with an https `url` field. Make sure to provide that https url with descriptive name text in markdown for each product.
    '''


examples = [
    [
        'Alex Smith',
        'Oversized Sweaters',
        'Feb, 2024',
    ],
    [
        'Robin Fischer',
        'Oversized Sweaters',
        'Feb, 2024'
    ],
    [
        'Chris Johnson',
        'Oversized Sweaters',
        'Feb, 2024'
    ],
    [
        'Robin Fischer',
        'denim jeans',
        'Feb, 2024'
    ]
]

preset_example = st.selectbox("select an example case:", examples)

with st.form('input_form'):
    customer_name = st.text_input("customer name:", value=preset_example[0], key='customer_name_input')
    customer_interests = st.text_input("recent purchase(s) and interest(s):", value=preset_example[1], key='customer_interests_input')
    time_of_year = st.text_input("time of year:", value=preset_example[2], key='time_of_year_input')
    gen_content = st.form_submit_button('Generate Content')

if gen_content:
    st.markdown('### Initial Prompt: ')
    st.markdown(generate_prompt(customer_name, time_of_year, customer_interests))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Vector Only")
    if gen_content:
        with st.spinner('Running Vector Only RAG...'):
            with st.expander('__Response:__', True):
                st.markdown(vector_only_chain.invoke(generate_prompt(customer_name, time_of_year, customer_interests),
                                                     retrieval_search_text=customer_interests)
                            )
            with st.expander("__Context used to answer this prompt:__"):
                st.json(vector_only_chain.last_used_context)

            with st.expander("__Query used to retrieve context:__"):
                vector_only_queries = vector_only_chain.get_last_browser_queries()
                st.code(vector_only_queries['params_query'], language='cypher')
                st.code(vector_only_queries['query_body'], language='cypher')
                st.markdown('### Visualize Retrieval in Neo4j')
                st.markdown('To explore the results in Neo4j do the following:\n' +
                            f'* Go to [Neo4j Browser]({get_neo4j_url_from_uri(HM_NEO4J_URI)}) and enter your credentials\n' +
                            '* Run the above queries')
                st.link_button("Try in Neo4j Browser!", get_neo4j_url_from_uri(HM_NEO4J_URI))

            st.success('Done!')

with col2:
    st.subheader("GraphRAG With Graph Vectors")
    if gen_content:
        with st.spinner('Running GraphRAG...'):
            with st.expander('__Response:__', True):
                st.markdown(graph_vector_chain.invoke(generate_prompt(customer_name, time_of_year, customer_interests),
                                                      retrieval_search_text=customer_interests)
                            )
            with st.expander("__Context used to answer this prompt:__"):
                st.json(graph_vector_chain.last_used_context)

            with st.expander("__Query used to retrieve context:__"):
                graph_vector_queries = graph_vector_chain.get_last_browser_queries()
                st.code(graph_vector_queries['params_query'], language='cypher')
                st.code(graph_vector_queries['query_body'], language='cypher')
                st.markdown('### Visualize Retrieval in Neo4j')
                st.markdown('To explore the results in Neo4j do the following:\n' +
                            f'* Go to [Neo4j Browser]({get_neo4j_url_from_uri(HM_NEO4J_URI)}) and enter your credentials\n' +
                            '* Run the above queries')
                st.link_button("Try in Neo4j Browser!", get_neo4j_url_from_uri(HM_NEO4J_URI))

            st.success('Done!')
