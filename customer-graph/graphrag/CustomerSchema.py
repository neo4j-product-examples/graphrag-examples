from typing import TypedDict, List

# Categories /////////////////////////////////////////////
class Category(TypedDict):
    name: str


class ProductCategory(Category):
    pass


class ProductType(Category):
    pass


# Entities /////////////////////////////////////////////
class Entity(TypedDict):
    address: str
    name: str


class Supplier(Entity):
    supplierId: int


# Items /////////////////////////////////////////////
class Item(TypedDict):
    name: str


class Product(Item):
    description: str
    productCode: int
    PART_OF: List[Category]


class Article(Item):
    articleId: int
    colourGroupCode: int
    colourGroupName: str
    graphicalAppearanceName: str
    graphicalAppearanceNo: int
    SUPPLIED_BY: List[Supplier]
    VARIANT_OF: List[Product]


# Events /////////////////////////////////////////////
class Event(TypedDict):
    date: str


class Order(Event):
    orderId: int
    CONTAINS: List[Article]


class CreditNote(Event):
    amount: float
    creditNoteId: int
    reason: str
    REFUND_FOR_ORDER: List[Order]
    REFUND_OF_ARTICLE: List[Article]


# Entities /////////////////////////////////////////////
class Customer(Entity):
    age: str
    clubMemberStatus: str
    customerId: int
    fashionNewsFrequency: str
    postalCode: int
    ORDERED: List[Order]






