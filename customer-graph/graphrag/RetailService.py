from neo4j import GraphDatabase
from typing import List
from CustomerSchema import Product
from neo4j_graphrag.retrievers import VectorCypherRetriever, Text2CypherRetriever, VectorRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from formatters import node_record_formatter
from neo4j_graphrag.llm import OpenAILLM


class RetailService:
    def __init__(self, uri, user, pwd):
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        self._driver = driver
        self._openai_embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        # Create LLM object. Used to generate the CYPHER queries
        self._llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0.5})

    async def get_products_similar_text(self, prompt_text: str) -> List[Product]:
        #Set up vector retriever
        retriever = VectorRetriever(
            driver=self._driver,
            index_name="product_text_embeddings",
            embedder=self._openai_embedder,
            result_formatter=node_record_formatter
        )

        # run vector search query on excerpts and get results containing the relevant agreement and clause
        retriever_result = retriever.search(query_text=prompt_text, top_k=20)

        #set up List to be returned
        products = []
        for item in retriever_result.items:
            p: Product = item.content
            products.append(p)

        return products
