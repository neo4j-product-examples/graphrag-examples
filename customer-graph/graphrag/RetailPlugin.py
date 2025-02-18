

from typing import List, Optional, Annotated
from CustomerSchema import Product
from semantic_kernel.functions import kernel_function
from RetailService import RetailService


class RetailPlugin:

    def __init__(self, retail_service: RetailService ):
        self.retail_service = retail_service

    @kernel_function
    async def search_products(self, prompt_text: str) -> Annotated[List[Product], "A list of products with potentially relevant text descriptions"]:
        """search product text based on user prompt and return most semantically similar ones. Please re-order or filter further based on additional context from user. """
        return await self.retail_service.get_products_similar_text(prompt_text)
