# GraphRAG Examples
This repo contains a streamlit app for introducing and teaching example GraphRAG patterns.

## Running the Streamlit App
Follow the below steps to run the sample app:
1. Load the data: This app uses the classic Northwind database - sales data for Northwind Traders, a fictitious specialty foods export/import company. There are two options for loading the data, both of which include generating text embeddings and a vector index for Product nodes.
   - Option 1 - Load from Source: Run the Cypher from `northwind-data.cypher` in an empty database. At the top of that script, you will need to replace `<your OpenAI API Key>` with your own OpenAI api key. 
   - Option 2 - Use the Database Dump: The `neo4j.dump` file has a copy of the database with everything already setup. If you are using Aura, follow the [Import Database](https://neo4j.com/docs/aura/auradb/importing/import-database/#_import_database]directions) directions in the docs. Otherwise, if you are using Neo4j Desktop or another self-managed instance see [neo4j-admin database load](https://neo4j.com/docs/operations-manual/current/backup-restore/restore-dump/).
2. Create a `secrets.toml` file using `secrets.toml.example` as a template
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

3. Install requirements (recommended in an isolated python virtual environment): `pip install -r requirements.txt`
4. Run app: `streamlit run Home.py --server.port=80`

