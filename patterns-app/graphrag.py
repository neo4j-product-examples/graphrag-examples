import json
from collections import OrderedDict
from dataclasses import dataclass
from operator import itemgetter
from typing import Dict, List, Tuple, Optional

from langchain.prompts.prompt import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(temperature=0, model_name='gpt-4o', streaming=True)
t2c_llm = ChatOpenAI(temperature=0, model_name='gpt-4', streaming=True)

VECTOR_QUERY_HEAD = """CALL db.index.vector.queryNodes($index, $k, $embedding)
YIELD node, score
"""

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


def format_res_dicts(d: Dict) -> Dict:
    res = OrderedDict()
    for k, v in d.items():
        if k != "metadata":
            res[k] = v
    for k, v in d['metadata'].items():
        if v is not None:
            res[k] = v
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


@dataclass(frozen=True)
class Neo4jCredentials:
    uri: str
    password: str
    username: str = "neo4j"
    database: str = "neo4j"


class GraphRAGChain:
    def __init__(self,
                 vector_index_name: str,
                 prompt_instructions: str,
                 graph_retrieval_query: str = None,
                 k: int = 5,
                 neo4j_uri: Optional[str] = None,
                 neo4j_username: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: Optional[str] = None
                 ):
        self.store = Neo4jVector.from_existing_index(
            embedding=embedding_model,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
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
    def __init__(self,
                 prompt_instructions: str,
                 properties_to_remove_from_cypher_res: List = None,
                 neo4j_uri: Optional[str] = None,
                 neo4j_username: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: Optional[str] = None
                 ):
        self.store = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )
        self.t2c_prompt = PromptTemplate.from_template(prompt_instructions + T2C_PROMPT_TEMPLATE)
        self.prompt = PromptTemplate.from_template(T2C_RESPONSE_PROMPT_TEMPLATE)
        self.chain = ({
                          'context': self.t2c_prompt | t2c_llm | StrOutputParser() | self._format_and_save_query | self.store.query | self._format_and_save_context,
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
        return s

    def invoke(self, prompt: str):
        return self.chain.invoke(prompt)


class GraphRAGPreFilterChain:
    def __init__(self,
                 vector_index_name: str,
                 prompt_instructions: str = '',
                 graph_prefilter_query: str = 'MATCH(node) WITH node, {} AS prefilterMetadata',
                 k: int = 5,
                 neo4j_uri: Optional[str] = None,
                 neo4j_username: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: Optional[str] = None
                 ):
        self.vectorStore = Neo4jVector.from_existing_index(
            embedding=embedding_model,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            index_name=vector_index_name)

        self.store = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )

        self.embedding_model = embedding_model

        self.vector_search_template = f"""
WITH node, prefilterMetadata, vector.similarity.cosine($embedding, node.`{self.vectorStore.embedding_node_property}`) AS score
WHERE score IS NOT NULL
WITH node.`{self.vectorStore.text_node_property}` AS text, 
    score, 
    node {{.*, `{self.vectorStore.text_node_property}`: Null, `{self.vectorStore.embedding_node_property}`: Null, id: Null}} AS searchMetadata,
    prefilterMetadata
RETURN text, score, apoc.map.merge(searchMetadata, prefilterMetadata) AS metadata
ORDER by score DESC LIMIT toInteger($k)
            """

        self.retrieval_query_template = graph_prefilter_query + '\n' + self.vector_search_template

        self.prompt = PromptTemplate.from_template(prompt_instructions + PROMPT_CONTEXT_TEMPLATE)

        self.chain = ({
                          'context': (lambda x: x['retrieverInput']) | RunnableLambda(
                              self.retriever) | self._format_and_save_context,
                          'input': (lambda x: x['prompt'])
                      }
                      | self.prompt
                      | llm
                      | StrOutputParser())

        self.last_used_context = None
        self.last_retrieval_query = None
        self.last_retrieval_query_params = None
        self.k = k

    def _format_and_save_context(self, docs) -> str:
        res = json.dumps([format_res_dicts(doc) for doc in docs], indent=1)
        self.last_used_context = res
        return res

    def _format_and_save_query(self, template: str, params: Dict):
        self.last_retrieval_query = template
        self.last_retrieval_query_params = params

    def get_last_browser_queries(self):
        params_string = json.dumps(self.last_retrieval_query_params)
        params_query = f":params {params_string}"
        return {'params_query': params_query,
                'params_url_query': f'/browser?cmd=params&arg={params_string}',
                'query_body': self.last_retrieval_query}

    def retriever(self, x):
        query_vector = self.embedding_model.embed_query(x['searchPrompt'])
        params = {**x['queryParams'], **{'index': self.vectorStore.index_name, 'k': self.k, 'embedding': query_vector}}
        res = self.store.query(self.retrieval_query_template, params=params)
        self._format_and_save_query(self.retrieval_query_template, params)
        return res

    def invoke(self, prompt: str, retrieval_search_text: str = None, query_params: Dict = None):
        if retrieval_search_text is None:
            retrieval_search_text = prompt
        if query_params is None:
            query_params = dict()
        return self.chain.invoke(
            {'retrieverInput': {'searchPrompt': retrieval_search_text, 'queryParams': query_params},
             'prompt': prompt})


class DynamicGraphRAGChain:
    def __init__(self,
                 vector_index_name: str,
                 prompt_instructions: str = '',
                 graph_retrieval_query: str = None,
                 k: int = 5,
                 neo4j_uri: Optional[str] = None,
                 neo4j_username: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: Optional[str] = None
                 ):
        self.vectorStore = Neo4jVector.from_existing_index(
            embedding=embedding_model,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            index_name=vector_index_name,
            retrieval_query=graph_retrieval_query)

        self.store = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
        )

        self.embedding_model = embedding_model

        self.prompt = PromptTemplate.from_template(prompt_instructions + PROMPT_CONTEXT_TEMPLATE)

        self.chain = ({
                          'context': (lambda x: x['retrieverInput']) | RunnableLambda(
                              self.retriever) | self._format_and_save_context,
                          'input': (lambda x: x['prompt'])
                      }
                      | self.prompt
                      | llm
                      | StrOutputParser())

        self.k = k

        default_retrieval = (
            f"RETURN node.`{self.vectorStore.text_node_property}` AS text, score, "
            f"node {{.*, `{self.vectorStore.text_node_property}`: Null, "
            f"`{self.vectorStore.embedding_node_property}`: Null, id: Null }} AS metadata"
        )
        self.retrieval_query = (
            self.vectorStore.retrieval_query if self.vectorStore.retrieval_query else default_retrieval
        )

        self.full_retrieval_query_template = VECTOR_QUERY_HEAD + self.retrieval_query
        self.last_used_context = None
        self.last_retrieval_query = None
        self.last_retrieval_query_params = None

    def _format_and_save_context(self, docs) -> str:
        res = json.dumps([format_res_dicts(doc) for doc in docs], indent=1)
        self.last_used_context = res
        return res

    def _format_and_save_query(self, template: str, params: Dict):
        self.last_retrieval_query = template
        self.last_retrieval_query_params = params

    def get_last_browser_queries(self):
        params_string = json.dumps(self.last_retrieval_query_params)
        params_query = f":params {params_string}"
        return {'params_query': params_query,
                'params_url_query': f'/browser?cmd=params&arg={params_string}',
                'query_body': self.last_retrieval_query}

    def retriever(self, x):
        query_vector = self.embedding_model.embed_query(x['searchPrompt'])
        params = {**x['queryParams'], **{'index': self.vectorStore.index_name, 'k': self.k, 'embedding': query_vector}}
        res = self.store.query(self.full_retrieval_query_template, params=params)
        self._format_and_save_query(self.full_retrieval_query_template, params)
        return res

    def invoke(self, prompt: str, retrieval_search_text: str = None, query_params: Dict = None):
        if retrieval_search_text is None:
            retrieval_search_text = prompt
        if query_params is None:
            query_params = dict()
        return self.chain.invoke({
            'retrieverInput': {'searchPrompt': retrieval_search_text, 'queryParams': query_params},
            'prompt': prompt
        })
