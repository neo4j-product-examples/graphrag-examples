import asyncio, os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from neo4j_graphrag.experimental.components.text_splitters.langchain import LangChainTextSplitterAdapter
from neo4j_graphrag.experimental.components.types import PdfDocument, DocumentInfo
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver
from rdflib import Graph
from RAGSchemaFromOnto import getSchemaFromOnto

load_dotenv()
NEO4J_URI=os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")



# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

neo4j_schema = getSchemaFromOnto("ontos/customer.ttl")
print(neo4j_schema)

# Create DocumentLoader
class PdfLoaderWithPageBreaks(DataLoader):
    async def run(self, filepath: Path) -> PdfDocument:
        loader = PyPDFLoader(filepath)
        text = ''
        async for page in loader.alazy_load():
            text = text + " __PAGE__BREAK__ " + page.page_content
        return PdfDocument(
            text=text,
            document_info=DocumentInfo(path=filepath),)

# Create a Splitter object
splitter = LangChainTextSplitterAdapter(
    CharacterTextSplitter(chunk_size=15_000, chunk_overlap=0, separator=" __PAGE__BREAK__ ")
)

# Create an Embedder object
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# Instantiate the LLM
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        #"max_tokens": 3000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)

#Instantiate the SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    pdf_loader=PdfLoaderWithPageBreaks(),
    text_splitter=splitter,
    embedder=embedder,
    entities=list(neo4j_schema.entities.values()),
    relations=list(neo4j_schema.relations.values()),
    potential_schema=neo4j_schema.potential_schema,
    on_error="IGNORE",
    from_pdf=True,
)

# LOAD PRODUCT DESCRIPTIONS
#asyncio.run(kg_builder.run_async(file_path='data/fashion-catalog.pdf'))


# LOAD CREDIT NOTES
asyncio.run(kg_builder.run_async(file_path='data/credit-notes.pdf'))

# perform entity resolution
print("Performing Entity Resolution")
driver.execute_query('''
MATCH (n:Article)
WITH n.articleId AS id, collect(n) as nodes
CALL apoc.refactor.mergeNodes(nodes, {
  properties: {
      `.*`: 'combine'
  },
  mergeRels: true
})
YIELD node
RETURN node;
''')

driver.execute_query('''
MATCH (n:Order)
WITH n.orderId AS id, collect(n) as nodes
CALL apoc.refactor.mergeNodes(nodes, {
  properties: {
      `.*`: 'combine'
  },
  mergeRels: true
})
YIELD node
RETURN node
''')


print("Removing Unneeded Nodes")
driver.execute_query('MATCH (n:Product) WHERE n:__Entity__ DETACH DELETE n')


driver.close()