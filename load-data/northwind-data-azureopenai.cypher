//THIS FILE IS FOR WHEN YOU ARE USING AZURE OPENAI ENDPOINT:

// Provide the endpoint details
:param provider => 'AzureOpenAI';
:param openAIKey => "<your Azure OpenAI API Key>";
:param resource => "<your Azure OpenAI resource name">;
:param deployment => "<your Azure OpenAI embedding model deployment name>"

/////////////////////////////////////////////////////////
// Load Northwind Data
/////////////////////////////////////////////////////////
CREATE CONSTRAINT Product_productID IF NOT EXISTS FOR (p:Product) REQUIRE (p.productID) IS UNIQUE;
CREATE CONSTRAINT Category_categoryID IF NOT EXISTS FOR (c:Category) REQUIRE (c.categoryID) IS UNIQUE;
CREATE CONSTRAINT Supplier_supplierID IF NOT EXISTS FOR (s:Supplier) REQUIRE (s.supplierID) IS UNIQUE;
CREATE CONSTRAINT Customer_customerID IF NOT EXISTS FOR (c:Customer) REQUIRE (c.customerID) IS UNIQUE;
CREATE CONSTRAINT Order_orderID IF NOT EXISTS FOR (o:Order) REQUIRE (o.orderID) IS UNIQUE;
CREATE CONSTRAINT Address_addressID IF NOT EXISTS FOR (a:Address) REQUIRE (a.addressID) IS UNIQUE;

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/products.csv" AS row
MERGE (n:Product {productID:row.productID})
SET n += row,
n.unitPrice = toFloat(row.unitPrice),
n.unitsInStock = toInteger(row.unitsInStock), n.unitsOnOrder = toInteger(row.unitsOnOrder),
n.reorderLevel = toInteger(row.reorderLevel), n.discontinued = (row.discontinued <> "0");

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/categories.csv" AS row
MERGE (n:Category {categoryID:row.categoryID})
SET n += row;

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/suppliers.csv" AS row
MERGE (n:Supplier {supplierID:row.supplierID})
SET n += row;

MATCH (p:Product),(c:Category)
WHERE p.categoryID = c.categoryID
MERGE (p)-[:BELONGS_TO]->(c);

MATCH (p:Product),(s:Supplier)
WHERE p.supplierID = s.supplierID
MERGE (s)<-[:SUPPLIED_BY]-(p);

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/customers.csv" AS row
MERGE (n:Customer {customerID:row.customerID})
SET n += row;

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/orders.csv" AS row
MERGE (o:Order {orderID:row.orderID})
SET o.customerID = row.customerID,
    o.employeeID = row.employeeID,
    o.orderDate = row.orderDate,
    o.requiredDate = row.requiredDate,
    o.shippedDate = row.shippedDate,
    o.shipVia = row.shipVia,
    o.freight = row.freight
MERGE (a:Address {addressID: apoc.text.join([coalesce(row.shipName, ''), coalesce(row.shipAddress, ''),
    coalesce(row.shipCity, ''), coalesce(row.shipRegion, ''), coalesce(row.shipPostalCode, ''),
    coalesce(row.shipCountry, '')], ', ')})
SET a.name = row.shipName,
    a.address = row.shipAddress,
    a.city = row.shipCity,
    a.region = row.shipRegion,
    a.postalCode = row.shipPostalCode,
    a.country = row.shipCountry
MERGE (o)-[:SHIPPED_TO]->(a)

WITH o
MATCH (c:Customer)
WHERE c.customerID = o.customerID
MERGE (c)-[:ORDERED]->(o);

LOAD CSV WITH HEADERS FROM "https://data.neo4j.com/northwind/order-details.csv" AS row
MATCH (p:Product), (o:Order)
WHERE p.productID = row.productID AND o.orderID = row.orderID
MERGE (o)-[details:ORDER_CONTAINS]->(p)
SET details = row,
details.quantity = toInteger(row.quantity);

/////////////////////////////////////////////////////////
// Set Text Property and Vector Index
/////////////////////////////////////////////////////////
//create text and embedding vector properties

MATCH(p:Product)-[:BELONGS_TO]-(c:Category)
SET p.text = "Product Category: " + c.categoryName + ' - ' + c.description + "\nProduct Name: " + p.productName
WITH p, genai.vector.encode(p.text, token: $provider, {$openAIKey, resource: $resource, deployment: $deployment}) AS textEmbedding
CALL db.create.setNodeVectorProperty(p,'textEmbedding', textEmbedding)
RETURN p.productID, p.text, p.textEmbedding;

//create vector index
CREATE VECTOR INDEX product_text_embeddings
FOR (n:Product) ON (n.textEmbedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 1536,
 `vector.similarity_function`: 'cosine'
}};

//await index coming online
CALL db.awaitIndex("product_text_embeddings", 300);
