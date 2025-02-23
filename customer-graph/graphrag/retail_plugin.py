

from typing import List, Optional, Annotated
from customer_schema import Product, CustomerSegment, ProductInfo, SupplierInfo
from semantic_kernel.functions import kernel_function
from retial_service import RetailService


class RetailPlugin:

    def __init__(self, retail_service: RetailService ):
        self.retail_service = retail_service

    @kernel_function
    async def search_products(self, prompt_text: str) -> Annotated[List[Product], "A list of products with potentially relevant text descriptions"]:
        """search product text based on user prompt and return most semantically similar ones. Please re-order or filter further based on additional context from user. """
        return await self.retail_service.get_products_similar_text(prompt_text)


    @kernel_function
    async def recommend_products(self, segment_item_ids_or_codes: List[int]) -> Annotated[List[Product], "A list of products ordered by recommendation score"]:
        """retrieve product recommendations given a list of product codes, articles ids, or segment ids. Please re-order or filter further based on additional context from user."""
        return await self.retail_service.get_product_recommendations(segment_item_ids_or_codes=segment_item_ids_or_codes)
    @kernel_function
    async def create_customer_segments(self) -> Annotated[List[CustomerSegment], "A list of customer segments"]:
        """Creates Customer segments based on user purchase behavior.  Generally needs to be done just once per session"""
        return await self.retail_service.run_customer_segmentation()

    @kernel_function
    async def get_product_order_supplier_info(self, product_codes: List[int]) -> Annotated[List[ProductInfo], "A list of product order, refund and supplier info"]:
        """DO not use if you don't have explicit product codes. Given a list of product codes, gets statistics for total orders and refunds as well by supplier for each product. Do not use for customer segment ids."""
        return await self.retail_service.get_product_order_supplier_info(product_codes=product_codes)

    @kernel_function
    async def get_supplier_order_product_info(self, supplier_ids: List[int]) -> Annotated[List[SupplierInfo], "A list of supplier order, refund and product info"]:
        """DO not use if you don't have explicit supplier ids. Given a list of supplier ids, gets statistics for the total orders and refunds  as well by product delivered for each supplier. Do not use for customer segment ids."""
        return await self.retail_service.get_supplier_order_product_info(supplier_ids=supplier_ids)


    @kernel_function
    async def answer_general_question(self, user_question: str) -> Annotated[str, "An answer to user_question"]:
        """Answer obtained by turning user_question into a CYPHER query."""
        return await self.retail_service.text_to_cypher_query(user_question=user_question)
