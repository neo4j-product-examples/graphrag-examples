import json
from collections import OrderedDict
from typing import Dict

from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

NEO4J_URI = st.secrets['NEO4J_URI']
NEO4J_USERNAME = st.secrets['NEO4J_USERNAME']
NEO4J_PASSWORD = st.secrets['NEO4J_PASSWORD']

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(temperature=0, model_name='gpt-4', streaming=True)

PROMPT_CONTEXT_TEMPLATE = """

# Question
{input}

# Here is the context:

{context}
"""


def format_doc(doc: Document) -> Dict:
    res = OrderedDict()
    res['text'] = doc.page_content
    res.update(doc.metadata)
    return res


class GraphRAGChain:
    def __init__(self, vector_index_name: str, prompt_instructions: str, graph_retrieval_query: str = None, k: int = 5):
        self.store = Neo4jVector.from_existing_index(
            embedding=embedding_model,
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
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
