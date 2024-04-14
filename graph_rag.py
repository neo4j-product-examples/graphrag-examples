import json
from collections import OrderedDict
from typing import Dict, List, Tuple

from langchain.prompts.prompt import PromptTemplate
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

NORTHWIND_NEO4J_URI = st.secrets['NORTHWIND_NEO4J_URI']
NORTHWIND_NEO4J_USERNAME = st.secrets['NORTHWIND_NEO4J_USERNAME']
NORTHWIND_NEO4J_PASSWORD = st.secrets['NORTHWIND_NEO4J_PASSWORD']

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(temperature=0, model_name='gpt-4', streaming=True)

PROMPT_CONTEXT_TEMPLATE = """

# Question
{input}

# Here is the context:
{context}
"""

T2C_PROMPT_TEMPLATE = '''
# Ask:
{input}

Remove english explanation, provide just the Cypher code. 
'''
T2C_RESPONSE_PROMPT_TEMPLATE = """
Transform below data to human readable format with bullets if needed, And summarize it in a sentence or two if possible
# Sample Ask and Response :
## Ask:
Get distinct watch terms ?

## Response:
[\"alert\",\"attorney\",\"bad\",\"canceled\",\"charge\"]

## Output:
Here are the distinct watch terms
- "alert"
- "attorney"
- "bad"
- "canceled"
- "charge"

# Generate similar output for below Ask and Response

## Ask
${input}

## Response:
${context}

## Output: 
"""


def format_doc(doc: Document) -> Dict:
    res = OrderedDict()
    res['text'] = doc.page_content
    res.update(doc.metadata)
    return res


def remove_key_from_dict(x, keys_to_remove):
    if isinstance(x, dict):
        x_clean = dict()
        for k, v in x.items():
            if k not in keys_to_remove:
                x_clean[k] = remove_key_from_dict(v, keys_to_remove)
    elif isinstance(x, list):
        x_clean = [remove_key_from_dict(i, keys_to_remove) for i in x]
    else:
        x_clean = x
    return x_clean


class GraphRAGChain:
    def __init__(self, neo4j_uri: str,
                 neo4j_auth: Tuple[str, str],
                 vector_index_name: str,
                 prompt_instructions: str,
                 graph_retrieval_query: str = None,
                 k: int = 5):
        self.store = Neo4jVector.from_existing_index(
            embedding=embedding_model,
            url=neo4j_uri,
            username=neo4j_auth[0],
            password=neo4j_auth[1],
            index_name=vector_index_name,
            retrieval_query=graph_retrieval_query)

        self.retriever = self.store.as_retriever(search_kwargs={"k": k})

        self.prompt = PromptTemplate.from_template(prompt_instructions + PROMPT_CONTEXT_TEMPLATE)

        self.chain = ({'context': self.retriever | self._format_and_save_context, 'input': RunnablePassthrough()}
                      | self.prompt
                      | llm
                      | StrOutputParser())

        self.last_used_context = None

        self.k = k

        default_retrieval = (
            f"RETURN node.`{self.store.text_node_property}` AS text, score, "
            f"node {{.*, `{self.store.text_node_property}`: Null, "
            f"`{self.store.embedding_node_property}`: Null, id: Null }} AS metadata"
        )
        self.retrieval_query = (
            self.store.retrieval_query if self.store.retrieval_query else default_retrieval
        )

    def _format_and_save_context(self, docs) -> str:
        res = json.dumps([format_doc(d) for d in docs], indent=1)
        self.last_used_context = res
        return res

    def invoke(self, prompt: str):
        return self.chain.invoke(prompt)

    def get_full_retrieval_query_template(self):
        query_head = """CALL db.index.vector.queryNodes($index, $k, $embedding)
YIELD node, score
"""
        return query_head + self.retrieval_query

    def get_full_retrieval_query(self, prompt: str):
        query_head = f"""WITH {self.store.embedding.embed_query(prompt)}
    AS queryVector
CALL db.index.vector.queryNodes('{self.store.index_name}', {self.k}, queryVector)
YIELD node, score
        """
        return query_head + self.retrieval_query

    def get_browser_queries(self, prompt: str):
        params_query = f":params{{index:'{self.store.index_name}', k:{self.k}, embedding:{self.store.embedding.embed_query(prompt)}}}"
        query_head = """CALL db.index.vector.queryNodes($index, $k, $embedding)
YIELD node, score
"""
        return {'params_query': params_query, 'query_body': query_head + self.retrieval_query}


class GraphRAGText2CypherChain:
    def __init__(self, neo4j_uri: str,
                 neo4j_auth: Tuple[str, str],
                 prompt_instructions: str,
                 properties_to_remove_from_cypher_res: List = None):
        self.store = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_auth[0],
            password=neo4j_auth[1],
        )
        self.t2c_prompt = PromptTemplate.from_template(prompt_instructions + T2C_PROMPT_TEMPLATE)
        self.prompt = PromptTemplate.from_template(T2C_RESPONSE_PROMPT_TEMPLATE)
        self.chain = ({
                          'context': self.t2c_prompt | llm | StrOutputParser() | self._format_and_save_query | self.store.query | self._format_and_save_context,
                          'input': RunnablePassthrough()
                      }
                      | self.prompt
                      | llm
                      | StrOutputParser())
        self.last_used_context = None
        self.last_retrieval_query = None
        self.properties_to_remove_from_cypher_res = properties_to_remove_from_cypher_res

    def _format_and_save_context(self, docs) -> str:
        if self.properties_to_remove_from_cypher_res is not None:
            docs = remove_key_from_dict(docs, self.properties_to_remove_from_cypher_res)
        res = json.dumps(docs, indent=1)
        self.last_used_context = res
        return res

    def _format_and_save_query(self, s) -> str:
        self.last_retrieval_query = s
        print(s)
        return s

    def invoke(self, prompt: str):
        return self.chain.invoke(prompt)
