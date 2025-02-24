import asyncio
import logging

from neo4j import GraphDatabase
from typing import List
from customer_schema import Product, CustomerSegment, Supplier, ProductInfo, SupplierInfo
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
        self._llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0.5})

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

    async def get_product_recommendations(self, segment_item_ids_or_codes: List[int]) -> List[Product]:
        res = self._driver.execute_query("""
        //recommend from product codes
        MATCH (customer:Customer)-[:ORDERED]->()-[:CONTAINS]->()-[:VARIANT_OF]->
        (interestedInProducts:Product)<-[:VARIANT_OF]-(interestedInArticles:Article)<-[:CONTAINS]-()<-[:ORDERED]
        -(:Customer)-[:ORDERED]->()-[:CONTAINS]->(recArticle:Article)-[:VARIANT_OF]->(product:Product)
        WHERE (interestedInArticles.articleId IN $itemIds) 
            OR (interestedInProducts.productCode IN $itemIds)
            OR (customer.segmentId IN $itemIds)
        WITH count(recArticle) AS recommendationScore, product
        RETURN product ORDER BY recommendationScore DESC LIMIT 20
        """, itemIds=segment_item_ids_or_codes)

        products = []
        for item in res.records:
            src = item.data()['product']
            s: Product = {k: src[k] for k in Product.__annotations__ if k in src}
            products.append(s)
        return products

    async def run_customer_segmentation(self) -> List[CustomerSegment]:
        # drop gds graph and segmentIds if they exists
        self._driver.execute_query("CALL gds.graph.drop('co-purchase-123', false) YIELD graphName")
        self._driver.execute_query("MATCH(n:Customer) REMOVE n.segmentId")
        # perform projection
        self._driver.execute_query("""
        MATCH(c1:Customer)-[:ORDERED]->()-[:CONTAINS]->(a:Article)<-[:CONTAINS]-()<-[:ORDERED]-(c2:Customer)
        WHERE elementId(c1) < elementId(c2)
        WITH c1, c2, count(a) AS coPurchaseCount
        WITH gds.graph.project('co-purchase-123', c1, c2, { 
            relationshipProperties: { coPurchaseCount: coPurchaseCount }}, 
            {undirectedRelationshipTypes: ['*']}) AS g
        RETURN g.graphName AS graph, g.nodeCount AS nodes, g.relationshipCount AS rels       
        """)
        # run community detection
        self._driver.execute_query("""
        CALL gds.leiden.write('co-purchase-123', { relationshipWeightProperty: 'coPurchaseCount', randomSeed: 7474, writeProperty: 'segmentId', concurrency:1})
        YIELD communityCount, nodePropertiesWritten
        RETURN communityCount, nodePropertiesWritten   
        """)
        # pull customer segments
        res = self._driver.execute_query("""
        MATCH(c:Customer) WHERE c.segmentId IS NOT NULL
        RETURN c.segmentId AS segmentId, count(c) AS numberOfCustomers ORDER BY numberOfCustomers DESC
        """)

        segments = []
        for item in res.records:
            s: CustomerSegment = item.data()
            segments.append(s)
        return segments

    async def get_product_order_supplier_info(self, product_codes: List[int]) -> list[ProductInfo]:
        res = self._driver.execute_query("""
        MATCH(p:Product)<-[:VARIANT_OF]-(a:Article)-[:SUPPLIED_BY]->(s)
        WHERE p.productCode IN $productCodes
        WITH *,
          COUNT {MATCH (:Order)-[:CONTAINS]->(a)} AS numberOfOrders, 
          COUNT {MATCH (:CreditNote)-[:REFUND_OF_ARTICLE]-(a)} AS numberOfRefunds
        RETURN p.productCode AS productCode,  
          sum(numberOfOrders) AS totalOrders, 
          sum(numberOfRefunds) AS totalReturns,
          collect({supplierId:s.supplierId, name:s.name, numberOfOrders:numberOfOrders, numberOfRefunds:numberOfRefunds}) AS supplierInfos
        """, productCodes=product_codes)

        product_infos = []
        for item in res.records:
            product_info: ProductInfo = item.data()
            product_infos.append(product_info)
        return product_infos

    async def get_supplier_order_product_info(self, supplier_ids: List[int]) -> list[SupplierInfo]:
        res = self._driver.execute_query("""
        MATCH(p:Product)<-[:VARIANT_OF]-(:Article)-[:SUPPLIED_BY]->(s)
        WHERE s.supplierId IN $supplierIds
        WITH DISTINCT p, s,
          COUNT {MATCH (:Order)-[:CONTAINS]->()-[:VARIANT_OF]->(p)} AS numberOfOrders, 
          COUNT {MATCH (:CreditNote)-[:REFUND_OF_ARTICLE]-()-[:VARIANT_OF]->(p)} AS numberOfRefunds
        RETURN s.supplierId AS supplierId,  
          sum(numberOfOrders) AS totalOrders, 
          sum(numberOfRefunds) AS totalReturns,
          collect({productCode:p.productCode, name:s.name, numberOfOrders:numberOfOrders, numberOfRefunds:numberOfRefunds}) AS supplierInfos
        """, supplierIds=supplier_ids)

        supplier_infos = []
        for item in res.records:
            supplier_info: SupplierInfo = item.data()
            supplier_infos.append(supplier_info)
        return supplier_infos

    async def text_to_cypher_query(self, user_question: str) -> str:
        with open("../ontos/text-to-cypher.json", "r", encoding="utf-8") as file:
            query_schema = file.read()
            #print(query_schema)

        # Initialize the retriever
        retriever = Text2CypherRetriever(
            driver=self._driver,
            llm=self._llm,
            neo4j_schema=query_schema,
            custom_prompt="""
Task: Generate a Cypher statement for querying a Neo4j graph database from a user input. 
- Do not include triple backticks ``` or ```cypher or any additional text except the generated Cypher statement in your response.
- Do not use any properties or relationships not included in the schema.

Schema:
{schema}

Examples (optional):
{examples}

Input:
{query_text}

Cypher query:
"""
        )

        # Generate a Cypher query using the LLM, send it to the Neo4j database, and return the results
        retriever_result = retriever.search(query_text=user_question)

        max_retries = 3
        attempt = 0
        errors = []

        while attempt < max_retries:
            try:
                # Attempt retrieval
                retriever_result = retriever.search(query_text=user_question)

                answer = ""
                logging.info(f"Text2Cypher Query:\n{retriever_result.metadata['cypher']}")
                for item in retriever_result.items:
                    content = str(item.content)
                    if content:
                        answer += content + '\n\n'

                return answer  # Return the result if successful

            except Exception as e:
                # Capture the error and append it to user_question
                error_message = f"\nError on last attempt number {attempt + 1}: {str(e)}"
                errors.append(error_message)
                user_question += error_message  # Append errors to user question for LLM awareness
                attempt += 1

                if attempt < max_retries:
                    await asyncio.sleep(1)  # Small delay before retrying

        # If all retries fail, return an error message
        return f"Failed after {max_retries} attempts. Errors: {errors}"
