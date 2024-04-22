# GraphRAG Examples
This repo contains a streamlit app for introducing and teaching example GraphRAG patterns.

## Running the Streamlit App
Follow the below steps to run the sample app:

### 1. Get an OpenAI API Key
The sample app uses OpenAI to demonstrate embedding and LLM capabilities.  To get an OpenAI API key:
1. Create an [OpenAI account](https://platform.openai.com/signup) if you don't have one already. Otherwise, [sign in](https://platform.openai.com/login). 
2. Navigate to the [API key page](https://platform.openai.com/account/api-keys) and "Create new secret key". Optionally naming the key. Save this somewhere safe, and do not share it with anyone.

### 2. Load the Data
This app uses two datasets:
1. The classic __Northwind Database__: Sales data for Northwind Traders, a fictitious specialty foods export/import company.
2. A sample of the __[H&M Fashion Dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)__: Real-world retail data, including customer purchases and rich information around products such as names, types, descriptions, department sections, etc.

The app has four pages in total, each of which relay on one of these datasets:

| App Page                         | Dataset Used        |
|----------------------------------|---------------------|
| Vector Search With Graph Context | Northwind           |
| Text2Cypher                      | Northwind           |
| Graph Vectors                    | H&M Fashion Dataset |
| Graph Filtering                  | H&M Fashion Dataset |


For the entire app to work, each dataset must be loaded into its own Neo4j database. If you choose not to load one of the datasets, the associated pages will not function which may be acceptable if those pages are not of interest to you.

__To Load Northwind__:
1. create an empty database on a Neo4j deployment type of your choosing.  Good options include a [blank Neo4j Sandbox](https://neo4j.com/sandbox/) or an [Aura Free](https://neo4j.com/cloud/aura-free/) instance
2. Run the Cypher from [`load-data/northwind-data.cypher`](load-data/northwind-data.cypher) on that database through Neo4j Browser. At the top of that script, you will need to replace `<your OpenAI API Key>` with your own OpenAI api key.

__To Load the H&M Fashion Dataset__:
1. This dataset involves some graph machine learning stuff. As such, you will need to create an empty Neo4j database with [Graph Data Science](https://neo4j.com/docs/graph-data-science/current/introduction/) enabled.  There is no Aura Free option for this. A couple good options include:
   - (free) Starting a blank graph data science [Neo4j Sandbox](https://sandbox.neo4j.com/) which should be sufficient for learning and exploration. 
   - (paid) use an [AuraDS instance](https://console.neo4j.io/?product=aura-ds). This is a paid option ($1.00 USD per hour) but should run significantly faster for loading, indexing, querying, and running GDS algorithms
2. Run the Notebook [`load-data/hm-data.ipynb`](load-data/hm-data.ipynb). It will attempt to read Neo4j and Open AI credentials from a secrets.toml file. You can create that file per directions below or replace with hard-coded credentials in the notebook.

### 3. Configure App and Environment
1. Create a `secrets.toml` file using `secrets.toml.example` as a template:
    ```bash
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    vi .streamlit/secrets.toml
    ```
2. Fill in the below credentials in the `secrets.toml` file.

    ```env
   # OpenAI
   OPENAI_API_KEY = "sk-..."
   
   # NEO4J
   NORTHWIND_NEO4J_URI = "neo4j+s://<xxxxx>.databases.neo4j.io"
   NORTHWIND_NEO4J_USERNAME = "neo4j"
   NORTHWIND_NEO4J_PASSWORD = "<password>"
   
   HM_NEO4J_URI = "neo4j+s://<xxxxx>.databases.neo4j.io"
   HM_NEO4J_USERNAME = "neo4j"
   HM_NEO4J_PASSWORD = "<password>"
   HM_AURA_DS = false
    ```
3. Install requirements (recommended in an isolated python virtual environment): 
   ```bash 
   pip install -r requirements.txt
   ```

### 3. Run the App
Run the app with the command: `streamlit run Home.py --server.port=80`

