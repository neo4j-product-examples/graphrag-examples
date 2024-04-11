# GraphRAG Examples
This repo contains a streamlit app for introducing and teaching example GraphRAG patterns.

## Loading Data
This sample app uses the classic Northwind database. We will provide a script for loading the data from raw source and setting up the text embeddings and vector index. Until then you can use `neo4j.dump` file to automatically setup the database with vectors and all. 

## Running the Streamlit App
1. Create a `secrets.toml` file using `secrets.toml.example` as a template
    ```bash
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    vi .streamlit/secrets.toml
    ```
    fill in the below credentials.  You can use a blank sandbox or any other Neo4j instance.
    ```env
    # OpenAI
    OPENAI_API_KEY = "sk-..."
    
    # NEO4J
    NEO4J_URI = "neo4j+s://<xxxxx>.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "<password>"
    ```

2. Install requirements (recommended in an isolated python virtual environment): `pip install -r requirements.txt`
3. Run app: `streamlit run Home.py --server.port=80`

