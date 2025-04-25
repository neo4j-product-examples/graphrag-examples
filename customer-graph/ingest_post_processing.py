import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
NEO4J_URI=os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE=os.getenv("NEO4J_DATABASE")



# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# create text properties for product
print("Formatting Product Text")
driver.execute_query('''
MATCH(p:Product)
OPTIONAL MATCH(p)-[:PART_OF]->(c:ProductCategory)
OPTIONAL MATCH(p)-[:PART_OF]->(t:ProductType)
SET p.text = '##Product\n' +
    'Name: ' + coalesce(p.name,'') + '\n' +
    'Type: ' + coalesce(t.name, '') + '\n' +
    'Category: ' + coalesce(c.name, '') + '\n' +
    'Description: ' + coalesce(p.description, ''),
    p.url = 'https://representative-domain/product/' + p.productCode
RETURN count(p) AS propertySetCount
''')

# create text embeddings for products
print("Creating Product Text Embeddings")
with driver.session(database=NEO4J_DATABASE) as session:
    session.run('''
    MATCH (n:Product) WHERE size(n.description) <> 0
    WITH collect(n) AS nodes, toInteger(rand()*$numberOfBatches) AS partition
    CALL(nodes) {
        CALL genai.vector.encodeBatch([node IN nodes| node.text], "OpenAI", { token: $token})
        YIELD index, vector
        CALL db.create.setNodeVectorProperty(nodes[index], "textEmbedding", vector)
    } IN TRANSACTIONS OF 1 ROW
    ''', token=os.getenv("OPENAI_API_KEY"), numberOfBatches=200)

# create vector index on text embeddings
print("Creating Product Vector Index")
driver.execute_query('''
CREATE VECTOR INDEX product_text_embeddings IF NOT EXISTS FOR (n:Product) ON (n.textEmbedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: toInteger($dimension),
 `vector.similarity_function`: 'cosine'
}}
''', dimension=1536)

# wait for index to come online
driver.execute_query('CALL db.awaitIndex("product_text_embeddings", 300)')


driver.close()

