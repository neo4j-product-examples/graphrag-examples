

from typing import List, Optional, Annotated
from CustomerSchema import Product, CustomerSegment
from semantic_kernel.functions import kernel_function
from RetailService import RetailService


class RetailPlugin:

    def __init__(self, retail_service: RetailService ):
        self.retail_service = retail_service

    @kernel_function
    async def search_products(self, prompt_text: str) -> Annotated[List[Product], "A list of products with potentially relevant text descriptions"]:
        """search product text based on user prompt and return most semantically similar ones. Please re-order or filter further based on additional context from user. """
        return await self.retail_service.get_products_similar_text(prompt_text)


    @kernel_function
    async def recommend_products(self, item_ids_or_codes: List[int]) -> Annotated[List[Product], "A list of products ordered by recommendation score"]:
        """retrieve product recommendations given a list of product codes or articles ids. Please re-order or filter further based on additional context from user."""
        return await self.retail_service.get_product_recommendations(item_ids_or_codes=item_ids_or_codes)
    @kernel_function
    async def create_customer_segments(self) -> Annotated[List[CustomerSegment], "A list of customer segments"]:
        """Creates Customer segments based on user purchase behavior.  Generally needs to be done just once per session"""
        return await self.retail_service.run_customer_segmentation()

    @kernel_function
    async def answer_general_question(self, user_question: str) -> Annotated[str, "An answer to user_question"]:
        """Answer obtained by turning user_question into a CYPHER query"""
        return await self.retail_service.text_to_cypher_query(user_question=user_question)
